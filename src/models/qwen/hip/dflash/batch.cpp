#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace gufo::hip {
namespace {

struct BatchCompletion {
  hipStream_t stream;
  bool completed{false};

  ~BatchCompletion() {
    if (!completed)
      (void)hipStreamSynchronize(stream);
  }
};

// One reusable allocation belongs to the coordinating executor. Nothing in
// this workspace is committed request state.
struct BlockWorkspace {
  BlockWorkspace(float* storage, std::size_t rows,
                 const core::ModelConfig& config,
                 const speculative::QwenDFlashConfig& draft) {
    const auto take = [&](std::size_t elements) {
      elements_ = (elements_ + 63U) & ~std::size_t{63U};
      float* pointer = storage != nullptr ? storage + elements_ : nullptr;
      elements_ += elements;
      return pointer;
    };
    const auto hidden_size = config.hidden_size;
    const auto kv_size =
        static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
    const auto dynamic_size =
        2U * draft.conv_kernel_size * (hidden_size / draft.conv_group_size);
    hidden = take(rows * hidden_size);
    normed = take(rows * hidden_size);
    conv = take(rows * hidden_size);
    dynamic = take(rows * dynamic_size);
    query = take(rows * config.AttentionSize());
    key = take(rows * kv_size);
    value = take(rows * kv_size);
    attention = take(rows * config.AttentionSize());
    projected = take(rows * hidden_size);
    gate = take(rows * config.intermediate_size);
    up = take(rows * config.intermediate_size);
    down = take(rows * hidden_size);
    logits = take(rows * config.vocab_size);
    selector = take(rows * draft.selector_rank);
    tokens = reinterpret_cast<std::uint32_t*>(take(rows));
    uniforms = take(rows);
    candidates =
        reinterpret_cast<std::uint32_t*>(take(rows * draft.selector_top_k));
    probabilities = take(rows * draft.selector_top_k);
    confidences = take(rows);
    const auto partials =
        kernels::DFlashSelectorScratchElements(config.vocab_size);
    partial_scores = take(partials);
    partial_ids = reinterpret_cast<std::uint32_t*>(take(partials));
  }

  [[nodiscard]] std::size_t Bytes() const { return elements_ * sizeof(float); }

  float *hidden, *normed, *conv, *dynamic, *query, *key, *value, *attention;
  float *projected, *gate, *up, *down, *logits, *selector;
  std::uint32_t *tokens, *candidates, *partial_ids;
  float *uniforms, *probabilities, *confidences, *partial_scores;

private:
  std::size_t elements_{0};
};

void TraceBlock(const DFlashTrace& trace, std::string_view name,
                const float* source, std::size_t count, hipStream_t stream) {
  if (!trace)
    return;
  std::vector<float> values(count);
  HIP_CHECK(hipMemcpyAsync(values.data(), source, count * sizeof(float),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  trace(name, values);
}

void ProjectBlock(const models::QwenTensorRef& weight, const float* input,
                  float* output, std::size_t rows, std::size_t output_size,
                  std::size_t input_size, hipStream_t stream) {
  if (weight.type == core::GgmlType::kBF16) {
    LaunchExactBf16GEMMFp32SmallBatch(weight.data, input, output, rows,
                                      output_size, input_size, stream);
  } else if (weight.type == core::GgmlType::kQ8_0 ||
             detail::IsNativeWmmaQuant(weight.type)) {
    LaunchBatchedQuantGEMMFp32(weight.type, weight.data, input, output, rows,
                               output_size, input_size, stream);
  } else {
    LaunchBatchedGEMM(weight.data, false, input, output, rows, output_size,
                      input_size, stream);
  }
}

}  // namespace

std::size_t QwenDFlashGpuExecutor::BatchScratchBytes(
    const QwenDFlashGpuModel& model, std::size_t rows) {
  return BlockWorkspace(nullptr, std::bit_ceil(rows), model.GetConfig(),
                        model.GetDFlashConfig())
      .Bytes();
}

std::vector<speculative::DraftProposal>
QwenDFlashGpuExecutor::ForwardBlockBatch(
    std::span<const QwenDFlashBlockRequest> requests) {
  if (requests.empty() || requests.size() > 8 ||
      requests.front().executor == nullptr)
    throw std::invalid_argument(
        "DFlash2 block batch requires one to eight sessions");
  auto& coordinator = *requests.front().executor;
  const auto& config = coordinator.model_->GetConfig();
  const auto& draft = coordinator.model_->GetDFlashConfig();
  const auto& weights = coordinator.model_->GetWeights();
  std::array<std::size_t, 9> offsets{};
  std::array<std::uint32_t, 8> counts{};
  std::uint32_t max_count = 0;
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& request = requests[index];
    const auto* executor = request.executor;
    if (executor == nullptr || executor->model_ != coordinator.model_ ||
        request.anchor >= config.vocab_size ||
        draft.mask_token_id >= config.vocab_size ||
        request.position != executor->injected_context_len_ ||
        request.position >= executor->max_context_ ||
        !std::isfinite(request.temperature) || request.temperature < 0.0F)
      throw std::invalid_argument("DFlash2 block batch has an invalid request");
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (requests[previous].executor == executor)
        throw std::invalid_argument("DFlash2 block batch repeats a session");
    }
    counts[index] = std::min({request.draft_count, draft.block_size - 1U,
                              executor->max_context_ - request.position - 1U});
    if (request.temperature > 0.0F && request.uniforms.size() < counts[index])
      throw std::invalid_argument(
          "DFlash2 block batch is missing sampling draws");
    offsets[index + 1] = offsets[index] + counts[index] + 1U;
    max_count = std::max(max_count, counts[index]);
  }
  std::vector<speculative::DraftProposal> proposals(requests.size());
  // Keep the single-user path and uncommon topology handling unchanged.
  if (requests.size() == 1 || config.head_dim > 256 ||
      std::ranges::find(std::span(counts).first(requests.size()), 0U) !=
          std::span(counts).first(requests.size()).end()) {
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const auto& request = requests[index];
      auto& proposal = proposals[index];
      proposal.start_pos = request.position;
      proposal.candidates_per_token =
          request.temperature > 0.0F ? draft.selector_top_k : 0;
      proposal.tokens = request.executor->ForwardBlock(
          request.anchor, request.position, request.draft_count,
          request.temperature, request.uniforms, nullptr,
          request.temperature > 0.0F ? &proposal.candidate_ids : nullptr,
          request.temperature > 0.0F ? &proposal.candidate_probabilities
                                     : nullptr,
          request.trace);
    }
    return proposals;
  }
  const auto rows = offsets[requests.size()];
  const auto proposal_rows = rows - requests.size();
  const auto hidden_size = config.hidden_size;
  const auto intermediate_size = config.intermediate_size;
  const auto query_size = config.AttentionSize();
  const auto kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const auto dynamic_size =
      2U * draft.conv_kernel_size * (hidden_size / draft.conv_group_size);
  const auto capacity = std::bit_ceil(rows);
  const auto stream = coordinator.stream_;
  for (const auto& request : requests)
    HIP_CHECK(hipStreamSynchronize(request.executor->stream_));
  BatchCompletion completion{stream};
  const BlockWorkspace layout(nullptr, capacity, config, draft);
  if (layout.Bytes() > coordinator.batch_scratch_bytes_) {
    auto* allocation = detail::AllocateDevice(layout.Bytes());
    if (coordinator.d_batch_scratch_ != nullptr)
      (void)hipFree(coordinator.d_batch_scratch_);
    coordinator.d_batch_scratch_ = allocation;
    coordinator.batch_scratch_bytes_ = layout.Bytes();
  }
  const BlockWorkspace scratch(
      static_cast<float*>(coordinator.d_batch_scratch_), capacity, config,
      draft);
  std::vector<std::uint32_t> tokens(rows, draft.mask_token_id);
  std::vector<float> uniforms(proposal_rows, 0.0F);
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& request = requests[index];
    tokens[offsets[index]] = request.anchor;
    if (request.temperature > 0.0F)
      std::copy_n(request.uniforms.begin(), counts[index],
                  uniforms.begin() + offsets[index] - index);
  }
  HIP_CHECK(hipMemcpyAsync(scratch.tokens, tokens.data(),
                           rows * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                           stream));
  HIP_CHECK(hipMemcpyAsync(scratch.uniforms, uniforms.data(),
                           proposal_rows * sizeof(float), hipMemcpyHostToDevice,
                           stream));
  const auto emit = [&](std::string_view name, const float* source,
                        std::size_t width, bool proposals_only = false) {
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const auto offset = offsets[index] - (proposals_only ? index : 0);
      const auto count = counts[index] + (proposals_only ? 0U : 1U);
      TraceBlock(requests[index].trace, name, source + offset * width,
                 count * width, stream);
    }
  };
  LaunchBatchedEmbeddingLookup(weights.token_embedding.data,
                               weights.token_embedding.type, scratch.tokens,
                               scratch.hidden, rows, hidden_size, stream);
  emit("embedding", scratch.hidden, hidden_size);
  const auto project = [&](const models::QwenTensorRef& weight,
                           const float* input, float* output,
                           std::size_t output_size, std::size_t input_size) {
    ProjectBlock(weight, input, output, rows, output_size, input_size, stream);
  };
  const auto convolve = [&](const float* input, const float* base,
                            std::uint32_t direction) {
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const auto offset = offsets[index];
      kernels::LaunchDFlashGroupedDynamicConv(
          input + offset * hidden_size, scratch.dynamic + offset * dynamic_size,
          base, scratch.conv + offset * hidden_size, counts[index] + 1U,
          hidden_size, draft.conv_kernel_size, draft.conv_group_size, direction,
          stream);
    }
  };
  const bool tracing = std::ranges::any_of(
      requests, [](const auto& request) { return bool(request.trace); });
  for (std::size_t layer_index = 0; layer_index < draft.num_layers;
       ++layer_index) {
    const auto& block = weights.layers[layer_index];
    const auto& layer = block.transformer;
    const auto stage = [&](std::string_view name, const float* source,
                           std::size_t width) {
      if (tracing)
        emit("layer." + std::to_string(layer_index) + "." + std::string(name),
             source, width);
    };
    LaunchBatchedRMSNorm(
        scratch.hidden, static_cast<const float*>(layer.attn_norm.data),
        scratch.normed, nullptr, rows, hidden_size, 1e-6F, stream);
    stage("attn_norm", scratch.normed, hidden_size);
    project(block.attention_conv_projection, scratch.normed, scratch.dynamic,
            dynamic_size, hidden_size);
    convolve(scratch.normed,
             static_cast<const float*>(block.attention_conv_base.data), 0);
    stage("attn_conv_in", scratch.conv, hidden_size);
    project(layer.attn_q, scratch.conv, scratch.query, query_size, hidden_size);
    project(layer.attn_k, scratch.conv, scratch.key, kv_size, hidden_size);
    project(layer.attn_v, scratch.conv, scratch.value, kv_size, hidden_size);
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const auto offset = offsets[index];
      auto* query = scratch.query + offset * query_size;
      auto* key = scratch.key + offset * kv_size;
      LaunchBatchedFusedQKNormRoPEKvWrite(
          query, key, key, static_cast<const float*>(layer.attn_q_norm.data),
          static_cast<const float*>(layer.attn_k_norm.data), query, key,
          nullptr, nullptr, nullptr, nullptr, 0, requests[index].position,
          counts[index] + 1U, 0, config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, config.rotary_dim,
          config.rope_theta, 1e-6F, stream);
    }
    stage("q", scratch.query, query_size);
    stage("k", scratch.key, kv_size);
    stage("v", scratch.value, kv_size);
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const auto& request = requests[index];
      const auto& executor = *request.executor;
      const auto offset = offsets[index];
      kernels::LaunchDFlashNonCausalAttention(
          scratch.query + offset * query_size,
          executor.d_injected_k_[layer_index],
          executor.d_injected_v_[layer_index], scratch.key + offset * kv_size,
          scratch.value + offset * kv_size,
          scratch.attention + offset * query_size, request.position,
          executor.injected_context_len_, counts[index] + 1U,
          draft.sliding_window, config.num_attention_heads,
          config.num_key_value_heads, config.head_dim,
          1.0F / std::sqrt(static_cast<float>(config.head_dim)), stream,
          executor.history_capacity_);
    }
    stage("attention", scratch.attention, query_size);
    project(layer.attn_output, scratch.attention, scratch.projected,
            hidden_size, query_size);
    stage("attn_output", scratch.projected, hidden_size);
    convolve(scratch.projected,
             static_cast<const float*>(block.attention_conv_base.data), 1);
    stage("attn_conv_out", scratch.conv, hidden_size);
    LaunchBatchedResidualAdd(scratch.hidden, scratch.conv, scratch.hidden, rows,
                             hidden_size, stream);
    stage("attn_residual", scratch.hidden, hidden_size);
    LaunchBatchedRMSNorm(
        scratch.hidden, static_cast<const float*>(layer.ffn_norm.data),
        scratch.normed, nullptr, rows, hidden_size, 1e-6F, stream);
    stage("ffn_norm", scratch.normed, hidden_size);
    project(block.ffn_conv_projection, scratch.normed, scratch.dynamic,
            dynamic_size, hidden_size);
    convolve(scratch.normed,
             static_cast<const float*>(block.ffn_conv_base.data), 0);
    stage("ffn_conv_in", scratch.conv, hidden_size);
    project(layer.ffn_gate, scratch.conv, scratch.gate, intermediate_size,
            hidden_size);
    project(layer.ffn_up, scratch.conv, scratch.up, intermediate_size,
            hidden_size);
    kernels::LaunchDFlashSiLUMul(scratch.gate, scratch.up,
                                 rows * intermediate_size, stream);
    project(layer.ffn_down, scratch.gate, scratch.down, hidden_size,
            intermediate_size);
    stage("ffn_down", scratch.down, hidden_size);
    convolve(scratch.down, static_cast<const float*>(block.ffn_conv_base.data),
             1);
    stage("ffn_conv_out", scratch.conv, hidden_size);
    LaunchBatchedResidualAdd(scratch.hidden, scratch.conv, scratch.hidden, rows,
                             hidden_size, stream);
    stage("output", scratch.hidden, hidden_size);
  }
  LaunchBatchedRMSNorm(
      scratch.hidden, static_cast<const float*>(weights.output_norm.data),
      scratch.normed, nullptr, rows, hidden_size, 1e-6F, stream);
  emit("normalized", scratch.normed, hidden_size);
  // The vocabulary projection consumes proposals only. The old hidden buffer
  // is free after normalization, and holds these compacted input rows.
  for (std::size_t index = 0; index < requests.size(); ++index) {
    HIP_CHECK(
        hipMemcpyAsync(scratch.hidden + (offsets[index] - index) * hidden_size,
                       scratch.normed + (offsets[index] + 1U) * hidden_size,
                       counts[index] * hidden_size * sizeof(float),
                       hipMemcpyDeviceToDevice, stream));
  }
  ProjectBlock(weights.selector_hidden, scratch.hidden, scratch.selector,
               proposal_rows, draft.selector_rank, hidden_size, stream);
  emit("selector_hidden", scratch.selector, draft.selector_rank, true);
  ProjectBlock(weights.output, scratch.hidden, scratch.logits, proposal_rows,
               config.vocab_size, hidden_size, stream);
  emit("logits", scratch.logits, config.vocab_size, true);
  for (std::size_t step = 0; step < max_count; ++step) {
    for (std::size_t index = 0; index < requests.size(); ++index) {
      if (step >= counts[index])
        continue;
      const auto& request = requests[index];
      const auto row = offsets[index] - index + step;
      const auto candidate = row * draft.selector_top_k;
      kernels::LaunchDFlashSelectorStep(
          scratch.logits + row * config.vocab_size,
          scratch.selector + row * draft.selector_rank,
          weights.selector_predecessor.data, weights.selector_successor.data,
          scratch.tokens + offsets[index] + step,
          scratch.tokens + offsets[index] + step + 1U,
          scratch.confidences + row, scratch.partial_scores,
          scratch.partial_ids, request.temperature,
          request.temperature > 0.0F ? scratch.uniforms + row : nullptr,
          request.temperature > 0.0F ? scratch.candidates + candidate : nullptr,
          request.temperature > 0.0F ? scratch.probabilities + candidate
                                     : nullptr,
          config.vocab_size, draft.selector_rank, draft.selector_top_k, stream);
    }
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& request = requests[index];
    auto& proposal = proposals[index];
    proposal.start_pos = request.position;
    proposal.tokens.resize(counts[index]);
    HIP_CHECK(hipMemcpyAsync(proposal.tokens.data(),
                             scratch.tokens + offsets[index] + 1U,
                             counts[index] * sizeof(tokenization::TokenId),
                             hipMemcpyDeviceToHost, stream));
    if (request.temperature > 0.0F) {
      proposal.candidates_per_token = draft.selector_top_k;
      const auto count = counts[index] * draft.selector_top_k;
      const auto offset = (offsets[index] - index) * draft.selector_top_k;
      proposal.candidate_ids.resize(count);
      proposal.candidate_probabilities.resize(count);
      HIP_CHECK(hipMemcpyAsync(proposal.candidate_ids.data(),
                               scratch.candidates + offset,
                               count * sizeof(tokenization::TokenId),
                               hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipMemcpyAsync(proposal.candidate_probabilities.data(),
                               scratch.probabilities + offset,
                               count * sizeof(float), hipMemcpyDeviceToHost,
                               stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  completion.completed = true;
  return proposals;
}

}  // namespace gufo::hip
#endif
