#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/dflash_reference.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace gufo::hip {
namespace {

constexpr std::array<std::uint8_t, 8> kDFlashGpuPersistentMagic = {
    'G', 'D', 'F', 'G', 'P', 'U', '0', '1'};
constexpr std::uint32_t kDFlashGpuPersistentVersion = 1;
constexpr std::size_t kDFlashGpuPersistentHeaderBytes = 64;

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("DFlash GPU persistent header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
[[nodiscard]] T GetLittleEndian(std::span<const std::uint8_t> source,
                                std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument("DFlash GPU persistent header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

[[nodiscard]] std::size_t CheckedPersistentAdd(std::size_t left,
                                               std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("DFlash GPU persistent size overflows");
  }
  return left + right;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("DFlash GPU persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

template<typename T>
void AllocateBuffer(T*& pointer, std::size_t elements) {
  pointer = static_cast<T*>(detail::AllocateDevice(elements * sizeof(T)));
}

[[nodiscard]] bool DFlashDebugEnabled() noexcept {
  static const bool enabled = std::getenv("GUFO_DFLASH_DEBUG") != nullptr;
  return enabled;
}

[[nodiscard]] bool DFlashTraceEnabled() noexcept {
  static const bool enabled =
      DFlashDebugEnabled() || std::getenv("GUFO_DFLASH_TRACE") != nullptr;
  return enabled;
}

[[nodiscard]] bool DFlashPrewarmEnabled() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("GUFO_DFLASH_PREWARM");
    if (value == nullptr) {
      return true;
    }
    const std::string_view setting{value};
    return setting != "0" && setting != "false" && setting != "off";
  }();
  return enabled;
}

enum class DFlashGemmMode {
  kBaseline,
  kHipblas,
  kHipblasLt,
};

[[nodiscard]] DFlashGemmMode GetDFlashGemmMode() noexcept {
  static const DFlashGemmMode mode = [] {
    const char* value = std::getenv("GUFO_DFLASH_GEMM");
    if (value == nullptr) {
      return DFlashGemmMode::kHipblasLt;
    }
    const std::string_view setting{value};
    if (setting == "hipblaslt") {
      return DFlashGemmMode::kHipblasLt;
    }
    if (setting == "hipblas") {
      return DFlashGemmMode::kHipblas;
    }
    return DFlashGemmMode::kBaseline;
  }();
  return mode;
}

/// Widest batch the shared small-batch GEMM routes accept.
constexpr std::size_t kDFlashMaxSharedBatch = 8;

[[nodiscard]] bool DFlashTimingEnabled() noexcept {
  static const bool enabled = std::getenv("GUFO_DFLASH_TIMING") != nullptr;
  return enabled;
}

/// Wall-clock stage rollup for the draft graph, printed at process exit when
/// GUFO_DFLASH_TIMING is set. Each stage boundary synchronizes the draft
/// stream, so the accounting is only wired up under the environment flag.
struct DFlashStageTimings {
  double inject_ms{0.0};
  double embed_ms{0.0};
  double layer_norm_conv_ms{0.0};
  double layer_gemm_ms{0.0};
  double layer_attention_ms{0.0};
  double head_ms{0.0};
  double selector_ms{0.0};
  double download_ms{0.0};
  std::uint64_t inject_calls{0};
  std::uint64_t inject_tokens{0};
  std::uint64_t block_calls{0};

  ~DFlashStageTimings() {
    if (!DFlashTimingEnabled() || block_calls == 0) {
      return;
    }
    const double blocks = static_cast<double>(block_calls);
    const double total = embed_ms + layer_norm_conv_ms + layer_gemm_ms +
                         layer_attention_ms + head_ms + selector_ms +
                         download_ms;
    std::cerr << std::fixed << std::setprecision(3)
              << "\n[DFLASH_TIMING] blocks=" << block_calls
              << " inject_calls=" << inject_calls
              << " inject_tokens=" << inject_tokens << '\n'
              << "  inject          " << inject_ms << " ms\n"
              << "  block total     " << total << " ms  (" << (total / blocks)
              << " ms/block)\n"
              << "    embed         " << embed_ms << " ms  ("
              << (embed_ms / blocks) << ")\n"
              << "    norm+conv     " << layer_norm_conv_ms << " ms  ("
              << (layer_norm_conv_ms / blocks) << ")\n"
              << "    gemm          " << layer_gemm_ms << " ms  ("
              << (layer_gemm_ms / blocks) << ")\n"
              << "    attention     " << layer_attention_ms << " ms  ("
              << (layer_attention_ms / blocks) << ")\n"
              << "    lm head       " << head_ms << " ms  ("
              << (head_ms / blocks) << ")\n"
              << "    selector      " << selector_ms << " ms  ("
              << (selector_ms / blocks) << ")\n"
              << "    download      " << download_ms << " ms  ("
              << (download_ms / blocks) << ")\n";
  }
};

[[nodiscard]] DFlashStageTimings& StageTimings() {
  static DFlashStageTimings timings;
  return timings;
}

/// Scoped stage accumulator. Synchronizes on both ends so a stage total is the
/// device time of its dispatches rather than their launch cost.
class StageTimer {
public:
  StageTimer(double* sink, hipStream_t stream) : sink_(sink), stream_(stream) {
    if (!DFlashTimingEnabled()) {
      sink_ = nullptr;
      return;
    }
    (void)hipStreamSynchronize(stream_);
    start_ = std::chrono::steady_clock::now();
  }

  StageTimer(const StageTimer&) = delete;
  StageTimer& operator=(const StageTimer&) = delete;
  StageTimer(StageTimer&&) = delete;
  StageTimer& operator=(StageTimer&&) = delete;

  ~StageTimer() { Stop(); }

  /// Closes the stage early. Idempotent, so the destructor is a no-op after an
  /// explicit stop.
  void Stop() {
    if (sink_ == nullptr) {
      return;
    }
    (void)hipStreamSynchronize(stream_);
    *sink_ += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - start_)
                  .count();
    sink_ = nullptr;
  }

private:
  double* sink_{nullptr};
  hipStream_t stream_{nullptr};
  std::chrono::steady_clock::time_point start_{};
};

void DebugDeviceTensor(std::string_view name, const float* device,
                       std::size_t elements, hipStream_t stream) {
  if (!DFlashDebugEnabled() || device == nullptr || elements == 0) {
    return;
  }

  std::vector<float> host(elements);
  const auto copy_error =
      hipMemcpyAsync(host.data(), device, elements * sizeof(float),
                     hipMemcpyDeviceToHost, stream);
  if (copy_error != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) {
    std::cerr << "[DFLASH_DEBUG] " << name << " copy failed\n";
    return;
  }

  double sum = 0.0;
  double square_sum = 0.0;
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
  for (const float value : host) {
    sum += value;
    square_sum += static_cast<double>(value) * value;
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }

  std::cerr << std::setprecision(9) << "[DFLASH_DEBUG] " << name
            << " n=" << elements << " sum=" << sum
            << " l2=" << std::sqrt(square_sum) << " min=" << minimum
            << " max=" << maximum << " first=";
  for (std::size_t index = 0; index < std::min<std::size_t>(3, elements);
       ++index) {
    std::cerr << (index == 0 ? "" : ",") << host[index];
  }
  std::cerr << " last=";
  const std::size_t last_begin = elements > 3 ? elements - 3 : 0;
  for (std::size_t index = last_begin; index < elements; ++index) {
    std::cerr << (index == last_begin ? "" : ",") << host[index];
  }
  std::cerr << '\n';
}

}  // namespace

QwenDFlashGpuSnapshot::~QwenDFlashGpuSnapshot() {
  if (d_k_ != nullptr) {
    (void)hipFree(d_k_);
  }
  if (d_v_ != nullptr) {
    (void)hipFree(d_v_);
  }
}

std::size_t QwenDFlashGpuSnapshot::PersistentPayloadBytes() const {
  return CheckedPersistentAdd(kDFlashGpuPersistentHeaderBytes, payload_bytes_);
}

std::size_t QwenDFlashGpuSnapshot::SerializePersistent(
    std::span<std::uint8_t> destination) const {
  const std::size_t expected_bytes = PersistentPayloadBytes();
  if (destination.size() != expected_bytes) {
    throw std::invalid_argument(
        "DFlash GPU persistent destination size is invalid");
  }
  if (valid_context_ > max_context_ ||
      (kv_width_ != 0 &&
       valid_context_ > std::numeric_limits<std::size_t>::max() / kv_width_) ||
      elements_per_layer_ !=
          static_cast<std::size_t>(valid_context_) * kv_width_ ||
      (elements_per_layer_ != 0 &&
       num_layers_ >
           std::numeric_limits<std::size_t>::max() / elements_per_layer_)) {
    throw std::invalid_argument("DFlash GPU snapshot metadata is malformed");
  }
  const std::size_t total_elements =
      static_cast<std::size_t>(num_layers_) * elements_per_layer_;
  if (total_elements >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("DFlash GPU persistent size overflows");
  }
  const std::size_t bytes_per_kind = total_elements * sizeof(float);
  if (bytes_per_kind >
          std::numeric_limits<std::size_t>::max() - bytes_per_kind ||
      payload_bytes_ != bytes_per_kind + bytes_per_kind ||
      (bytes_per_kind != 0 && (d_k_ == nullptr || d_v_ == nullptr))) {
    throw std::invalid_argument("DFlash GPU snapshot payload is malformed");
  }

  std::fill(destination.begin(), destination.end(), std::uint8_t{0});
  std::copy(kDFlashGpuPersistentMagic.begin(), kDFlashGpuPersistentMagic.end(),
            destination.begin());
  PutLittleEndian<std::uint32_t>(destination, 8, kDFlashGpuPersistentVersion);
  PutLittleEndian<std::uint32_t>(
      destination, 12,
      static_cast<std::uint32_t>(kDFlashGpuPersistentHeaderBytes));
  PutLittleEndian<std::uint32_t>(destination, 16, num_layers_);
  PutLittleEndian<std::uint32_t>(destination, 20, kv_width_);
  PutLittleEndian<std::uint32_t>(destination, 24, max_context_);
  PutLittleEndian<std::uint32_t>(destination, 28, valid_context_);
  PutLittleEndian<std::uint64_t>(
      destination, 32, static_cast<std::uint64_t>(elements_per_layer_));
  PutLittleEndian<std::uint64_t>(destination, 40,
                                 static_cast<std::uint64_t>(payload_bytes_));
  PutLittleEndian<std::uint64_t>(destination, 48,
                                 static_cast<std::uint64_t>(expected_bytes));

  if (bytes_per_kind != 0) {
    HIP_CHECK(hipMemcpy(destination.data() + kDFlashGpuPersistentHeaderBytes,
                        d_k_, bytes_per_kind, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        destination.data() + kDFlashGpuPersistentHeaderBytes + bytes_per_kind,
        d_v_, bytes_per_kind, hipMemcpyDeviceToHost));
  }
  return destination.size();
}

QwenDFlashGpuExecutor::QwenDFlashGpuExecutor(
    std::shared_ptr<const QwenDFlashGpuModel> model, std::uint32_t max_context)
    : model_(std::move(model)),
      max_context_(max_context),
      h_target_features_(model_->GetDFlashConfig().target_layer_ids.size() *
                         model_->GetConfig().hidden_size),
      h_logits_(model_->GetDFlashConfig().block_size *
                model_->GetConfig().vocab_size) {
  try {
    Allocate();
    Reset();
  } catch (...) {
    Free();
    throw;
  }
}

QwenDFlashGpuExecutor::~QwenDFlashGpuExecutor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  Free();
}

std::unique_ptr<QwenDFlashGpuExecutor> QwenDFlashGpuExecutor::Create(
    std::shared_ptr<const QwenDFlashGpuModel> model, std::uint32_t max_context,
    std::string* error_msg) {
  if (model == nullptr || max_context == 0 ||
      max_context > model->GetConfig().context_length) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash GPU executor context is invalid";
    }
    return nullptr;
  }
  try {
    return std::unique_ptr<QwenDFlashGpuExecutor>(
        new QwenDFlashGpuExecutor(std::move(model), max_context));
  } catch (const std::exception& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
}

void QwenDFlashGpuExecutor::Allocate() {
  const auto& cfg = model_->GetConfig();
  const auto& df_cfg = model_->GetDFlashConfig();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t intermediate_size = cfg.intermediate_size;
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const std::size_t q_dim = cfg.AttentionSize();
  const std::size_t block_size = df_cfg.block_size;
  const std::size_t num_layers = df_cfg.num_layers;
  const std::size_t enc_in_dim = df_cfg.target_layer_ids.size() * hidden_size;
  const std::size_t dynamic_size =
      2U * df_cfg.conv_kernel_size * (hidden_size / df_cfg.conv_group_size);

  const auto error = hipStreamCreate(&stream_);
  if (error != hipSuccess) {
    throw std::runtime_error("DFlash hipStreamCreate failed");
  }
  HIPBLAS_CHECK(hipblasCreate(&hipblas_handle_));
  HIPBLAS_CHECK(hipblasSetStream(hipblas_handle_, stream_));
  hipblaslt_gemm_ = std::make_unique<HipblasLtGemm>();

  // Allocate KV caches per layer
  d_injected_k_.resize(num_layers, nullptr);
  d_injected_v_.resize(num_layers, nullptr);
  for (std::size_t i = 0; i < num_layers; ++i) {
    AllocateBuffer(d_injected_k_[i], max_context_ * kv_dim);
    AllocateBuffer(d_injected_v_[i], max_context_ * kv_dim);
  }

  // Scratch buffers for prompt injection & block drafting
  AllocateBuffer(d_target_features_, max_context_ * enc_in_dim);
  AllocateBuffer(d_fused_features_, max_context_ * hidden_size);
  AllocateBuffer(d_block_normed_, max_context_ * hidden_size);
  AllocateBuffer(d_k_block_, max_context_ * kv_dim);
  AllocateBuffer(d_v_block_, max_context_ * kv_dim);

  // Scratch buffers for block drafting (size = block_size)
  AllocateBuffer(d_block_hidden_, block_size * hidden_size);
  AllocateBuffer(d_conv_hidden_, block_size * hidden_size);
  AllocateBuffer(d_dynamic_coefficients_, block_size * dynamic_size);
  AllocateBuffer(d_q_, block_size * q_dim);
  AllocateBuffer(d_attn_out_, block_size * hidden_size);
  AllocateBuffer(d_ffn_gate_, block_size * intermediate_size);
  AllocateBuffer(d_ffn_up_, block_size * intermediate_size);
  AllocateBuffer(d_ffn_down_, block_size * hidden_size);
  AllocateBuffer(d_logits_, block_size * cfg.vocab_size);
  AllocateBuffer(d_selector_hidden_, block_size * df_cfg.selector_rank);
  const std::size_t selector_scratch_elements =
      kernels::DFlashSelectorScratchElements(cfg.vocab_size);
  AllocateBuffer(d_selector_partial_scores_, selector_scratch_elements);
  AllocateBuffer(d_selector_partial_ids_, selector_scratch_elements);
  AllocateBuffer(d_selector_candidate_ids_, block_size * df_cfg.selector_top_k);
  AllocateBuffer(d_selector_candidate_probabilities_,
                 block_size * df_cfg.selector_top_k);
  AllocateBuffer(d_selector_uniforms_, block_size);
  AllocateBuffer(d_confidences_, block_size);
  AllocateBuffer(d_out_token_, block_size);
  AllocateBuffer(d_bf16_input_, block_size * intermediate_size);

  PrewarmBlockGemms();
}

void QwenDFlashGpuExecutor::PrewarmBlockGemms() {
  if (!DFlashPrewarmEnabled() ||
      GetDFlashGemmMode() != DFlashGemmMode::kHipblasLt ||
      hipblaslt_gemm_ == nullptr) {
    return;
  }

  const auto& weights = model_->GetWeights();
  const auto& cfg = model_->GetConfig();
  const auto& df_cfg = model_->GetDFlashConfig();
  if (weights.layers.empty() || df_cfg.block_size < 2) {
    return;
  }

  const auto& layer = weights.layers.front();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t intermediate_size = cfg.intermediate_size;
  const std::size_t q_dim = cfg.AttentionSize();
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const std::size_t dynamic_size =
      2U * df_cfg.conv_kernel_size * (hidden_size / df_cfg.conv_group_size);

  HIP_CHECK(hipMemsetAsync(d_block_normed_, 0,
                           df_cfg.block_size * hidden_size * sizeof(float),
                           stream_));
  HIP_CHECK(hipMemsetAsync(d_conv_hidden_, 0,
                           df_cfg.block_size * hidden_size * sizeof(float),
                           stream_));
  HIP_CHECK(hipMemsetAsync(d_attn_out_, 0,
                           df_cfg.block_size * q_dim * sizeof(float), stream_));
  HIP_CHECK(hipMemsetAsync(
      d_ffn_gate_, 0, df_cfg.block_size * intermediate_size * sizeof(float),
      stream_));

  const auto prewarm = [&](const models::QwenTensorRef& weight,
                           const float* input, float* output,
                           std::size_t batch_size, std::size_t output_size,
                           std::size_t input_size) {
    if (weight.type == core::GgmlType::kBF16) {
      RunBlockGemm(weight, input, output, batch_size, output_size, input_size);
    }
  };

  for (std::size_t block_count = 2; block_count <= df_cfg.block_size;
       ++block_count) {
    prewarm(layer.attention_conv_projection, d_block_normed_,
            d_dynamic_coefficients_, block_count, dynamic_size, hidden_size);
    prewarm(layer.transformer.attn_q, d_conv_hidden_, d_q_, block_count, q_dim,
            hidden_size);
    prewarm(layer.transformer.attn_k, d_conv_hidden_, d_k_block_, block_count,
            kv_dim, hidden_size);
    prewarm(layer.transformer.attn_output, d_attn_out_, d_fused_features_,
            block_count, hidden_size, q_dim);
    prewarm(layer.transformer.ffn_gate, d_conv_hidden_, d_ffn_gate_,
            block_count, intermediate_size, hidden_size);
    prewarm(layer.transformer.ffn_down, d_ffn_gate_, d_ffn_down_, block_count,
            hidden_size, intermediate_size);
  }

  for (std::size_t draft_count = 1; draft_count < df_cfg.block_size;
       ++draft_count) {
    prewarm(weights.selector_hidden, d_block_normed_, d_selector_hidden_,
            draft_count, df_cfg.selector_rank, hidden_size);
    prewarm(weights.output, d_block_normed_, d_logits_, draft_count,
            cfg.vocab_size, hidden_size);
  }

  HIP_CHECK(hipStreamSynchronize(stream_));
}

void QwenDFlashGpuExecutor::RunBlockGemm(const models::QwenTensorRef& weight,
                                         const float* input, float* output,
                                         std::size_t batch_size,
                                         std::size_t output_size,
                                         std::size_t input_size) {
  if (weight.type == core::GgmlType::kQ8_0) {
    // The FP32-activation route streams each quantized weight row once for the
    // whole block and needs no BF16 activation round trip. It also beat the
    // matrix-core W8A8 route once the shared kernel read activations as float4
    // (11.3 vs 12.2 ms per draft block), so there is one route here.
    if (batch_size <= kDFlashMaxSharedBatch) {
      LaunchBatchedQuantGEMMFp32(weight.type, weight.data, input, output,
                                 batch_size, output_size, input_size, stream_);
      return;
    }
    LaunchFloatToBfloat16(input, d_bf16_input_, batch_size * input_size,
                          stream_);
    LaunchBatchedQuantGEMMBf16(weight.type, weight.data, d_bf16_input_, output,
                               batch_size, output_size, input_size, stream_);
    return;
  }

  const auto mode = GetDFlashGemmMode();
  if (mode == DFlashGemmMode::kBaseline ||
      weight.type != core::GgmlType::kBF16) {
    LaunchBatchedGEMM(weight.data, weight.type == core::GgmlType::kBF16, input,
                      output, batch_size, output_size, input_size, stream_);
    return;
  }

  LaunchFloatToBfloat16(input, d_bf16_input_, batch_size * input_size, stream_);
  if (mode == DFlashGemmMode::kHipblasLt && hipblaslt_gemm_ != nullptr &&
      hipblaslt_gemm_->RunBf16(weight.data, d_bf16_input_, output, batch_size,
                               output_size, input_size, stream_)) {
    return;
  }
  LaunchHipblasGEMMBF16(hipblas_handle_, weight.data, d_bf16_input_, output,
                        batch_size, output_size, input_size, stream_);
}

void QwenDFlashGpuExecutor::RunInjectGemm(const models::QwenTensorRef& weight,
                                          const float* input, float* output,
                                          std::size_t num_tokens,
                                          std::size_t output_size,
                                          std::size_t input_size) {
  if (weight.data == nullptr || num_tokens == 0) {
    return;
  }
  // The dense batched GEMM reads every weight row once per token because its
  // grid carries the batch on the y axis. Injection therefore streams the
  // encoder and K/V matrices `num_tokens` times. Feeding the shared small-batch
  // kernels in chunks reads them once per chunk instead, which is the same
  // arithmetic per row but up to eight times less weight traffic.
  const bool is_bf16 = weight.type == core::GgmlType::kBF16;
  if (!is_bf16 && weight.type != core::GgmlType::kQ8_0) {
    LaunchBatchedGEMM(weight.data, is_bf16, input, output, num_tokens,
                      output_size, input_size, stream_);
    return;
  }

  for (std::size_t offset = 0; offset < num_tokens;
       offset += kDFlashMaxSharedBatch) {
    const std::size_t chunk =
        std::min(kDFlashMaxSharedBatch, num_tokens - offset);
    const float* chunk_input = input + (offset * input_size);
    float* chunk_output = output + (offset * output_size);
    if (is_bf16) {
      LaunchExactBf16GEMMFp32SmallBatch(weight.data, chunk_input, chunk_output,
                                        chunk, output_size, input_size,
                                        stream_);
    } else {
      LaunchBatchedQuantGEMMFp32(weight.type, weight.data, chunk_input,
                                 chunk_output, chunk, output_size, input_size,
                                 stream_);
    }
  }
}

void QwenDFlashGpuExecutor::Free() noexcept {
  for (auto*& ptr : d_injected_k_) {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
      ptr = nullptr;
    }
  }
  for (auto*& ptr : d_injected_v_) {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
      ptr = nullptr;
    }
  }
  if (d_target_features_ != nullptr)
    (void)hipFree(d_target_features_);
  if (d_fused_features_ != nullptr)
    (void)hipFree(d_fused_features_);
  if (d_block_hidden_ != nullptr)
    (void)hipFree(d_block_hidden_);
  if (d_block_normed_ != nullptr)
    (void)hipFree(d_block_normed_);
  if (d_conv_hidden_ != nullptr)
    (void)hipFree(d_conv_hidden_);
  if (d_dynamic_coefficients_ != nullptr) {
    (void)hipFree(d_dynamic_coefficients_);
  }
  if (d_q_ != nullptr)
    (void)hipFree(d_q_);
  if (d_k_block_ != nullptr)
    (void)hipFree(d_k_block_);
  if (d_v_block_ != nullptr)
    (void)hipFree(d_v_block_);
  if (d_attn_out_ != nullptr)
    (void)hipFree(d_attn_out_);
  if (d_ffn_gate_ != nullptr)
    (void)hipFree(d_ffn_gate_);
  if (d_ffn_up_ != nullptr)
    (void)hipFree(d_ffn_up_);
  if (d_ffn_down_ != nullptr)
    (void)hipFree(d_ffn_down_);
  if (d_logits_ != nullptr)
    (void)hipFree(d_logits_);
  if (d_selector_hidden_ != nullptr)
    (void)hipFree(d_selector_hidden_);
  if (d_selector_partial_scores_ != nullptr) {
    (void)hipFree(d_selector_partial_scores_);
  }
  if (d_selector_partial_ids_ != nullptr) {
    (void)hipFree(d_selector_partial_ids_);
  }
  if (d_selector_candidate_ids_ != nullptr) {
    (void)hipFree(d_selector_candidate_ids_);
  }
  if (d_selector_candidate_probabilities_ != nullptr) {
    (void)hipFree(d_selector_candidate_probabilities_);
  }
  if (d_selector_uniforms_ != nullptr) {
    (void)hipFree(d_selector_uniforms_);
  }
  if (d_confidences_ != nullptr)
    (void)hipFree(d_confidences_);
  if (d_out_token_ != nullptr)
    (void)hipFree(d_out_token_);
  if (d_bf16_input_ != nullptr)
    (void)hipFree(d_bf16_input_);
  hipblaslt_gemm_.reset();
  if (hipblas_handle_ != nullptr) {
    (void)hipblasDestroy(hipblas_handle_);
    hipblas_handle_ = nullptr;
  }
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

std::size_t QwenDFlashGpuExecutor::StateBytes() const noexcept {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  return 2U * model_->GetDFlashConfig().num_layers * max_context_ * kv_width *
         sizeof(float);
}

std::size_t QwenDFlashGpuExecutor::SnapshotPayloadBytes() const noexcept {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  return 2U * model_->GetDFlashConfig().num_layers * injected_context_len_ *
         kv_width * sizeof(float);
}

std::unique_ptr<QwenDFlashGpuSnapshot> QwenDFlashGpuExecutor::SaveSnapshot()
    const {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  const std::size_t elements_per_layer =
      static_cast<std::size_t>(injected_context_len_) * kv_width;
  const std::size_t total_elements =
      model_->GetDFlashConfig().num_layers * elements_per_layer;
  const std::size_t bytes = total_elements * sizeof(float);

  auto snapshot =
      std::unique_ptr<QwenDFlashGpuSnapshot>(new QwenDFlashGpuSnapshot());
  snapshot->elements_per_layer_ = elements_per_layer;
  snapshot->num_layers_ = model_->GetDFlashConfig().num_layers;
  snapshot->kv_width_ = static_cast<std::uint32_t>(kv_width);
  snapshot->max_context_ = max_context_;
  snapshot->valid_context_ = injected_context_len_;
  snapshot->payload_bytes_ = SnapshotPayloadBytes();

  if (bytes == 0) {
    return snapshot;
  }
  HIP_CHECK(hipMalloc(&snapshot->d_k_, bytes));
  HIP_CHECK(hipMalloc(&snapshot->d_v_, bytes));
  auto* snapshot_k = static_cast<float*>(snapshot->d_k_);
  auto* snapshot_v = static_cast<float*>(snapshot->d_v_);
  for (std::size_t layer = 0; layer < snapshot->num_layers_; ++layer) {
    HIP_CHECK(hipMemcpyAsync(
        snapshot_k + layer * elements_per_layer, d_injected_k_[layer],
        elements_per_layer * sizeof(float), hipMemcpyDeviceToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(
        snapshot_v + layer * elements_per_layer, d_injected_v_[layer],
        elements_per_layer * sizeof(float), hipMemcpyDeviceToDevice, stream_));
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  return snapshot;
}

void QwenDFlashGpuExecutor::RestoreSnapshot(
    const QwenDFlashGpuSnapshot& snapshot) {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  const std::size_t expected_elements_per_layer =
      static_cast<std::size_t>(snapshot.valid_context_) * kv_width;
  if (snapshot.num_layers_ != model_->GetDFlashConfig().num_layers ||
      snapshot.kv_width_ != kv_width || snapshot.max_context_ != max_context_ ||
      snapshot.valid_context_ > max_context_ ||
      snapshot.elements_per_layer_ != expected_elements_per_layer) {
    throw std::invalid_argument(
        "DFlash snapshot is incompatible with the executor");
  }
  const std::size_t bytes =
      snapshot.elements_per_layer_ * snapshot.num_layers_ * sizeof(float);
  if (bytes != 0 && (snapshot.d_k_ == nullptr || snapshot.d_v_ == nullptr)) {
    throw std::invalid_argument("DFlash snapshot payload is incomplete");
  }
  const auto* snapshot_k = static_cast<const float*>(snapshot.d_k_);
  const auto* snapshot_v = static_cast<const float*>(snapshot.d_v_);
  for (std::size_t layer = 0; layer < snapshot.num_layers_; ++layer) {
    if (snapshot.elements_per_layer_ == 0) {
      continue;
    }
    HIP_CHECK(hipMemcpyAsync(d_injected_k_[layer],
                             snapshot_k + layer * snapshot.elements_per_layer_,
                             snapshot.elements_per_layer_ * sizeof(float),
                             hipMemcpyDeviceToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_injected_v_[layer],
                             snapshot_v + layer * snapshot.elements_per_layer_,
                             snapshot.elements_per_layer_ * sizeof(float),
                             hipMemcpyDeviceToDevice, stream_));
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  injected_context_len_ = snapshot.valid_context_;
}

void QwenDFlashGpuExecutor::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kDFlashGpuPersistentHeaderBytes ||
      !std::equal(kDFlashGpuPersistentMagic.begin(),
                  kDFlashGpuPersistentMagic.end(), payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) !=
          kDFlashGpuPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kDFlashGpuPersistentHeaderBytes ||
      GetLittleEndian<std::uint64_t>(payload, 56) != 0) {
    throw std::invalid_argument("DFlash GPU persistent header is invalid");
  }
  const std::uint32_t num_layers = GetLittleEndian<std::uint32_t>(payload, 16);
  const std::uint32_t kv_width = GetLittleEndian<std::uint32_t>(payload, 20);
  const std::uint32_t max_context = GetLittleEndian<std::uint32_t>(payload, 24);
  const std::uint32_t valid_context =
      GetLittleEndian<std::uint32_t>(payload, 28);
  const std::size_t elements_per_layer =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 32));
  const std::size_t payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48));

  const std::size_t expected_kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  if (expected_kv_width > std::numeric_limits<std::uint32_t>::max() ||
      num_layers != model_->GetDFlashConfig().num_layers ||
      kv_width != expected_kv_width || max_context != max_context_ ||
      valid_context > max_context_ ||
      (kv_width != 0 &&
       valid_context > std::numeric_limits<std::size_t>::max() / kv_width) ||
      elements_per_layer !=
          static_cast<std::size_t>(valid_context) * kv_width ||
      (elements_per_layer != 0 &&
       num_layers >
           std::numeric_limits<std::size_t>::max() / elements_per_layer)) {
    throw std::invalid_argument(
        "DFlash GPU persistent metadata is incompatible");
  }
  const std::size_t total_elements =
      static_cast<std::size_t>(num_layers) * elements_per_layer;
  if (total_elements >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("DFlash GPU persistent size overflows");
  }
  const std::size_t bytes_per_kind = total_elements * sizeof(float);
  if (bytes_per_kind >
          std::numeric_limits<std::size_t>::max() - bytes_per_kind ||
      payload_bytes != bytes_per_kind + bytes_per_kind ||
      total_bytes != payload.size() ||
      CheckedPersistentAdd(kDFlashGpuPersistentHeaderBytes, payload_bytes) !=
          payload.size()) {
    throw std::invalid_argument(
        "DFlash GPU persistent payload size is invalid");
  }

  for (std::size_t layer = 0; layer < num_layers; ++layer) {
    if (elements_per_layer == 0) {
      continue;
    }
    const std::size_t layer_bytes = elements_per_layer * sizeof(float);
    const std::size_t layer_offset = layer * layer_bytes;
    HIP_CHECK(hipMemcpyAsync(
        d_injected_k_[layer],
        payload.data() + kDFlashGpuPersistentHeaderBytes + layer_offset,
        layer_bytes, hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_injected_v_[layer],
                             payload.data() + kDFlashGpuPersistentHeaderBytes +
                                 bytes_per_kind + layer_offset,
                             layer_bytes, hipMemcpyHostToDevice, stream_));
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  injected_context_len_ = valid_context;
}

void QwenDFlashGpuExecutor::Reset() noexcept {
  injected_context_len_ = 0;
  const std::size_t kv_dim =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  for (auto* ptr : d_injected_k_) {
    if (ptr != nullptr) {
      (void)hipMemsetAsync(ptr, 0, max_context_ * kv_dim * sizeof(float),
                           stream_);
    }
  }
  for (auto* ptr : d_injected_v_) {
    if (ptr != nullptr) {
      (void)hipMemsetAsync(ptr, 0, max_context_ * kv_dim * sizeof(float),
                           stream_);
    }
  }
}

bool QwenDFlashGpuExecutor::InjectTargetContext(
    std::span<const float> target_features, std::uint32_t position,
    std::uint32_t num_tokens) {
  const auto& weights = model_->GetWeights();
  const auto& cfg = model_->GetConfig();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t enc_in_dim = GetTargetFeaturesSize();
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;

  if (target_features.size() != num_tokens * enc_in_dim ||
      position + num_tokens > max_context_) {
    return false;
  }

  StageTimer inject_timer(&StageTimings().inject_ms, stream_);
  if (DFlashTimingEnabled()) {
    ++StageTimings().inject_calls;
    StageTimings().inject_tokens += num_tokens;
  }

  // Upload target features
  const std::size_t bytes = target_features.size() * sizeof(float);
  const auto cpy_err =
      hipMemcpyAsync(d_target_features_, target_features.data(), bytes,
                     hipMemcpyHostToDevice, stream_);
  if (cpy_err != hipSuccess) {
    return false;
  }
  DebugDeviceTensor("encoder.inp_embd", d_target_features_,
                    num_tokens * enc_in_dim, stream_);

  // FC Projection & Norm
  RunInjectGemm(weights.fc_projection, d_target_features_, d_fused_features_,
                num_tokens, hidden_size, enc_in_dim);
  DebugDeviceTensor("encoder.fc_out", d_fused_features_,
                    num_tokens * hidden_size, stream_);
  LaunchBatchedRMSNorm(
      d_fused_features_, static_cast<const float*>(weights.fc_norm.data),
      d_block_normed_, nullptr, num_tokens, hidden_size, 1e-6F, stream_);
  DebugDeviceTensor("encoder.enc_norm_out", d_block_normed_,
                    num_tokens * hidden_size, stream_);

  // Project and store K / V for each draft layer
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    const auto& layer = weights.layers[i].transformer;
    if (layer.attn_k.data != nullptr && layer.attn_v.data != nullptr) {
      RunInjectGemm(layer.attn_k, d_block_normed_, d_k_block_, num_tokens,
                    kv_dim, hidden_size);
      RunInjectGemm(layer.attn_v, d_block_normed_, d_v_block_, num_tokens,
                    kv_dim, hidden_size);
    }

    if (layer.attn_k_norm.data != nullptr) {
      LaunchBatchedPerHeadRMSNorm(
          d_k_block_, static_cast<const float*>(layer.attn_k_norm.data),
          d_k_block_, num_tokens, cfg.num_key_value_heads, cfg.head_dim, 1e-6F,
          stream_);
    }

    // Apply RoPE across injected tokens
    for (std::size_t t = 0; t < num_tokens; ++t) {
      LaunchRoPE(nullptr, d_k_block_ + t * kv_dim, 0, cfg.num_key_value_heads,
                 cfg.head_dim, cfg.rotary_dim,
                 position + static_cast<std::uint32_t>(t), cfg.rope_theta,
                 stream_);
    }
    if (i == 0) {
      DebugDeviceTensor("encoder.Kcur_injected-0", d_k_block_,
                        num_tokens * kv_dim, stream_);
      DebugDeviceTensor("encoder.Vcur_injected-0", d_v_block_,
                        num_tokens * kv_dim, stream_);
    }

    // Copy to injected KV cache
    const std::size_t cache_bytes = num_tokens * kv_dim * sizeof(float);
    (void)hipMemcpyAsync(d_injected_k_[i] + position * kv_dim, d_k_block_,
                         cache_bytes, hipMemcpyDeviceToDevice, stream_);
    (void)hipMemcpyAsync(d_injected_v_[i] + position * kv_dim, d_v_block_,
                         cache_bytes, hipMemcpyDeviceToDevice, stream_);
  }

  injected_context_len_ =
      std::max(injected_context_len_, position + num_tokens);
  (void)hipStreamSynchronize(stream_);
  return true;
}

std::vector<tokenization::TokenId> QwenDFlashGpuExecutor::ForwardBlock(
    tokenization::TokenId anchor_token, std::uint32_t current_pos,
    std::uint32_t draft_count, float temperature,
    std::span<const float> sample_uniforms, std::vector<float>* out_confidences,
    std::vector<tokenization::TokenId>* out_candidate_ids,
    std::vector<float>* out_candidate_probabilities) {
  const auto& weights = model_->GetWeights();
  const auto& cfg = model_->GetConfig();
  const auto& df_cfg = model_->GetDFlashConfig();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t intermediate_size = cfg.intermediate_size;
  const std::size_t num_q_heads = cfg.num_attention_heads;
  const std::size_t num_kv_heads = cfg.num_key_value_heads;
  const std::size_t head_dim = cfg.head_dim;
  const std::size_t q_dim = cfg.AttentionSize();
  const std::size_t kv_dim = static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t vocab_size = cfg.vocab_size;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  const std::size_t dynamic_size =
      2U * df_cfg.conv_kernel_size * (hidden_size / df_cfg.conv_group_size);

  draft_count = std::min(draft_count,
                         df_cfg.block_size > 0 ? df_cfg.block_size - 1U : 0U);
  if (!std::isfinite(temperature) || temperature < 0.0F) {
    throw std::invalid_argument(
        "DFlash sampling temperature must be finite and nonnegative");
  }
  if (temperature > 0.0F && sample_uniforms.size() < draft_count) {
    throw std::invalid_argument(
        "DFlash sampled drafting requires one random value per proposal");
  }
  if (draft_count == 0) {
    if (out_confidences != nullptr) {
      out_confidences->clear();
    }
    if (out_candidate_ids != nullptr) {
      out_candidate_ids->clear();
    }
    if (out_candidate_probabilities != nullptr) {
      out_candidate_probabilities->clear();
    }
    return {};
  }
  const std::uint32_t block_count = draft_count + 1U;
  auto& timings = StageTimings();
  if (DFlashTimingEnabled()) {
    ++timings.block_calls;
  }

  {
    StageTimer embed_timer(&timings.embed_ms, stream_);
    // Slot zero is the committed anchor; all proposal slots are masks.
    if (weights.token_embedding.data != nullptr) {
      LaunchEmbeddingLookup(weights.token_embedding.data,
                            weights.token_embedding.type, anchor_token,
                            d_block_hidden_, hidden_size, stream_);
      for (std::size_t t = 1; t < block_count; ++t) {
        LaunchEmbeddingLookup(
            weights.token_embedding.data, weights.token_embedding.type,
            df_cfg.mask_token_id, d_block_hidden_ + t * hidden_size,
            hidden_size, stream_);
      }
    }
  }
  DebugDeviceTensor("decoder.inp_noise_embd", d_block_hidden_,
                    block_count * hidden_size, stream_);

  // Stage accounting is only wired up under GUFO_DFLASH_TIMING; otherwise each
  // StageTimer is inert and the dispatch order is unchanged.
  const auto timed = [&](double* sink, auto&& body) {
    StageTimer timer(sink, stream_);
    body();
  };

  // Five DFlash-2 decoder blocks.
  for (std::size_t i = 0; i < df_cfg.num_layers; ++i) {
    const auto& dflash_layer = weights.layers[i];
    const auto& layer = dflash_layer.transformer;

    timed(&timings.layer_norm_conv_ms, [&] {
      LaunchBatchedRMSNorm(
          d_block_hidden_, static_cast<const float*>(layer.attn_norm.data),
          d_block_normed_, nullptr, block_count, hidden_size, 1e-6F, stream_);
    });
    DebugDeviceTensor("decoder.noise_norm-" + std::to_string(i),
                      d_block_normed_, block_count * hidden_size, stream_);

    timed(&timings.layer_gemm_ms, [&] {
      RunBlockGemm(dflash_layer.attention_conv_projection, d_block_normed_,
                   d_dynamic_coefficients_, block_count, dynamic_size,
                   hidden_size);
    });
    timed(&timings.layer_norm_conv_ms, [&] {
      kernels::LaunchDFlashGroupedDynamicConv(
          d_block_normed_, d_dynamic_coefficients_,
          static_cast<const float*>(dflash_layer.attention_conv_base.data),
          d_conv_hidden_, block_count, hidden_size, df_cfg.conv_kernel_size,
          df_cfg.conv_group_size, 0, stream_);
    });
    DebugDeviceTensor("decoder.attn_conv_in-" + std::to_string(i),
                      d_conv_hidden_, block_count * hidden_size, stream_);

    timed(&timings.layer_gemm_ms, [&] {
      RunBlockGemm(layer.attn_q, d_conv_hidden_, d_q_, block_count, q_dim,
                   hidden_size);
      RunBlockGemm(layer.attn_k, d_conv_hidden_, d_k_block_, block_count,
                   kv_dim, hidden_size);
      RunBlockGemm(layer.attn_v, d_conv_hidden_, d_v_block_, block_count,
                   kv_dim, hidden_size);
    });

    timed(&timings.layer_norm_conv_ms, [&] {
      LaunchBatchedPerHeadRMSNorm(
          d_q_, static_cast<const float*>(layer.attn_q_norm.data), d_q_,
          block_count, num_q_heads, head_dim, 1e-6F, stream_);
      LaunchBatchedPerHeadRMSNorm(
          d_k_block_, static_cast<const float*>(layer.attn_k_norm.data),
          d_k_block_, block_count, num_kv_heads, head_dim, 1e-6F, stream_);

      for (std::size_t t = 0; t < block_count; ++t) {
        LaunchRoPE(d_q_ + t * q_dim, d_k_block_ + t * kv_dim, num_q_heads,
                   num_kv_heads, head_dim, cfg.rotary_dim,
                   current_pos + static_cast<std::uint32_t>(t), cfg.rope_theta,
                   stream_);
      }
    });
    DebugDeviceTensor("decoder.Qcur-" + std::to_string(i), d_q_,
                      block_count * q_dim, stream_);
    DebugDeviceTensor("decoder.Kcur-" + std::to_string(i), d_k_block_,
                      block_count * kv_dim, stream_);
    DebugDeviceTensor("decoder.Vcur-" + std::to_string(i), d_v_block_,
                      block_count * kv_dim, stream_);

    timed(&timings.layer_attention_ms, [&] {
      kernels::LaunchDFlashNonCausalAttention(
          d_q_, d_injected_k_[i], d_injected_v_[i], d_k_block_, d_v_block_,
          d_attn_out_, current_pos,
          std::min(current_pos, injected_context_len_), block_count,
          df_cfg.sliding_window, static_cast<std::uint32_t>(num_q_heads),
          static_cast<std::uint32_t>(num_kv_heads),
          static_cast<std::uint32_t>(head_dim), scale, stream_);
    });

    timed(&timings.layer_gemm_ms, [&] {
      RunBlockGemm(layer.attn_output, d_attn_out_, d_fused_features_,
                   block_count, hidden_size, q_dim);
    });
    timed(&timings.layer_norm_conv_ms, [&] {
      kernels::LaunchDFlashGroupedDynamicConv(
          d_fused_features_, d_dynamic_coefficients_,
          static_cast<const float*>(dflash_layer.attention_conv_base.data),
          d_conv_hidden_, block_count, hidden_size, df_cfg.conv_kernel_size,
          df_cfg.conv_group_size, 1, stream_);
      DebugDeviceTensor("decoder.attn_conv_out-" + std::to_string(i),
                        d_conv_hidden_, block_count * hidden_size, stream_);
      LaunchBatchedResidualAdd(d_block_hidden_, d_conv_hidden_, d_block_hidden_,
                               block_count, hidden_size, stream_);
      DebugDeviceTensor("decoder.ffn_inp-" + std::to_string(i), d_block_hidden_,
                        block_count * hidden_size, stream_);

      LaunchBatchedRMSNorm(
          d_block_hidden_, static_cast<const float*>(layer.ffn_norm.data),
          d_block_normed_, nullptr, block_count, hidden_size, 1e-6F, stream_);
    });
    DebugDeviceTensor("decoder.ffn_norm-" + std::to_string(i), d_block_normed_,
                      block_count * hidden_size, stream_);

    timed(&timings.layer_gemm_ms, [&] {
      RunBlockGemm(dflash_layer.ffn_conv_projection, d_block_normed_,
                   d_dynamic_coefficients_, block_count, dynamic_size,
                   hidden_size);
    });
    timed(&timings.layer_norm_conv_ms, [&] {
      kernels::LaunchDFlashGroupedDynamicConv(
          d_block_normed_, d_dynamic_coefficients_,
          static_cast<const float*>(dflash_layer.ffn_conv_base.data),
          d_conv_hidden_, block_count, hidden_size, df_cfg.conv_kernel_size,
          df_cfg.conv_group_size, 0, stream_);
    });
    DebugDeviceTensor("decoder.ffn_conv_in-" + std::to_string(i),
                      d_conv_hidden_, block_count * hidden_size, stream_);

    timed(&timings.layer_gemm_ms, [&] {
      RunBlockGemm(layer.ffn_gate, d_conv_hidden_, d_ffn_gate_, block_count,
                   intermediate_size, hidden_size);
      RunBlockGemm(layer.ffn_up, d_conv_hidden_, d_ffn_up_, block_count,
                   intermediate_size, hidden_size);
    });
    timed(&timings.layer_norm_conv_ms, [&] {
      kernels::LaunchDFlashSiLUMul(d_ffn_gate_, d_ffn_up_,
                                   block_count * intermediate_size, stream_);
    });
    timed(&timings.layer_gemm_ms, [&] {
      RunBlockGemm(layer.ffn_down, d_ffn_gate_, d_ffn_down_, block_count,
                   hidden_size, intermediate_size);
    });
    DebugDeviceTensor("decoder.ffn_out-" + std::to_string(i), d_ffn_down_,
                      block_count * hidden_size, stream_);
    timed(&timings.layer_norm_conv_ms, [&] {
      kernels::LaunchDFlashGroupedDynamicConv(
          d_ffn_down_, d_dynamic_coefficients_,
          static_cast<const float*>(dflash_layer.ffn_conv_base.data),
          d_conv_hidden_, block_count, hidden_size, df_cfg.conv_kernel_size,
          df_cfg.conv_group_size, 1, stream_);
      DebugDeviceTensor("decoder.ffn_conv_out-" + std::to_string(i),
                        d_conv_hidden_, block_count * hidden_size, stream_);
      LaunchBatchedResidualAdd(d_block_hidden_, d_conv_hidden_, d_block_hidden_,
                               block_count, hidden_size, stream_);
    });
    DebugDeviceTensor("decoder.l_out-" + std::to_string(i), d_block_hidden_,
                      block_count * hidden_size, stream_);
  }

  timed(&timings.layer_norm_conv_ms, [&] {
    LaunchBatchedRMSNorm(
        d_block_hidden_, static_cast<const float*>(weights.output_norm.data),
        d_block_normed_, nullptr, block_count, hidden_size, 1e-6F, stream_);
  });
  DebugDeviceTensor("decoder.result_norm", d_block_normed_,
                    block_count * hidden_size, stream_);

  std::vector<tokenization::TokenId> tokens(draft_count, 0);
  timed(&timings.head_ms, [&] {
    RunBlockGemm(weights.selector_hidden, d_block_normed_ + hidden_size,
                 d_selector_hidden_, draft_count, df_cfg.selector_rank,
                 hidden_size);

    if (GetDFlashGemmMode() == DFlashGemmMode::kBaseline) {
      for (std::size_t proposal = 0; proposal < draft_count; ++proposal) {
        LaunchGEMV(weights.output.data, weights.output.type,
                   d_block_normed_ + ((proposal + 1U) * hidden_size),
                   d_logits_ + (proposal * vocab_size), vocab_size, hidden_size,
                   stream_, models::qwen::QwenGemmMode::kHipMtp);
      }
    } else {
      RunBlockGemm(weights.output, d_block_normed_ + hidden_size, d_logits_,
                   draft_count, vocab_size, hidden_size);
    }
  });
  DebugDeviceTensor("decoder.result_output", d_logits_,
                    draft_count * vocab_size, stream_);

  const auto anchor_copy =
      hipMemcpyAsync(d_out_token_, &anchor_token, sizeof(anchor_token),
                     hipMemcpyHostToDevice, stream_);
  if (anchor_copy != hipSuccess) {
    throw std::runtime_error("DFlash GPU anchor upload failed");
  }
  if (temperature > 0.0F) {
    const auto uniform_copy = hipMemcpyAsync(
        d_selector_uniforms_, sample_uniforms.data(),
        draft_count * sizeof(float), hipMemcpyHostToDevice, stream_);
    if (uniform_copy != hipSuccess) {
      throw std::runtime_error("DFlash GPU sampling upload failed");
    }
  }
  StageTimer selector_timer(&timings.selector_ms, stream_);
  for (std::size_t proposal = 0; proposal < draft_count; ++proposal) {
    const std::size_t candidate_offset = proposal * df_cfg.selector_top_k;
    kernels::LaunchDFlashSelectorStep(
        d_logits_ + (proposal * vocab_size),
        d_selector_hidden_ + (proposal * df_cfg.selector_rank),
        weights.selector_predecessor.data, weights.selector_successor.data,
        d_out_token_ + proposal, d_out_token_ + proposal + 1U,
        d_confidences_ + proposal, d_selector_partial_scores_,
        d_selector_partial_ids_, temperature,
        temperature > 0.0F ? d_selector_uniforms_ + proposal : nullptr,
        temperature > 0.0F ? d_selector_candidate_ids_ + candidate_offset
                           : nullptr,
        temperature > 0.0F
            ? d_selector_candidate_probabilities_ + candidate_offset
            : nullptr,
        vocab_size, df_cfg.selector_rank, df_cfg.selector_top_k, stream_);
  }
  selector_timer.Stop();

  StageTimer download_timer(&timings.download_ms, stream_);
  const auto token_copy =
      hipMemcpyAsync(tokens.data(), d_out_token_ + 1U,
                     draft_count * sizeof(tokenization::TokenId),
                     hipMemcpyDeviceToHost, stream_);
  if (token_copy != hipSuccess) {
    throw std::runtime_error("DFlash GPU selector download failed");
  }

  if (out_confidences != nullptr) {
    out_confidences->resize(draft_count);
    const auto confidence_copy = hipMemcpyAsync(
        out_confidences->data(), d_confidences_, draft_count * sizeof(float),
        hipMemcpyDeviceToHost, stream_);
    if (confidence_copy != hipSuccess) {
      throw std::runtime_error("DFlash GPU confidence download failed");
    }
  }
  const std::size_t candidate_count =
      static_cast<std::size_t>(draft_count) * df_cfg.selector_top_k;
  if (out_candidate_ids != nullptr) {
    out_candidate_ids->resize(candidate_count);
    const auto candidate_copy =
        hipMemcpyAsync(out_candidate_ids->data(), d_selector_candidate_ids_,
                       candidate_count * sizeof(tokenization::TokenId),
                       hipMemcpyDeviceToHost, stream_);
    if (candidate_copy != hipSuccess) {
      throw std::runtime_error("DFlash GPU candidate download failed");
    }
  }
  if (out_candidate_probabilities != nullptr) {
    out_candidate_probabilities->resize(candidate_count);
    const auto probability_copy = hipMemcpyAsync(
        out_candidate_probabilities->data(),
        d_selector_candidate_probabilities_, candidate_count * sizeof(float),
        hipMemcpyDeviceToHost, stream_);
    if (probability_copy != hipSuccess) {
      throw std::runtime_error(
          "DFlash GPU candidate probability download failed");
    }
  }
  if (hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("DFlash GPU block synchronization failed");
  }
  if (DFlashTraceEnabled()) {
    std::cerr << "[DFLASH_DEBUG] selector anchor=" << anchor_token
              << " proposals=";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
      std::cerr << (index == 0 ? "" : ",") << tokens[index];
    }
    std::cerr << '\n';
  }

  return tokens;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
