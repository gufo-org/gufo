#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <stdexcept>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen/modules/moe.hpp"

#include "qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::hip {
namespace {

namespace routed = models::qwen38_flash_next::rocm;

/// Host side of the routed F16 expert path: the per-expert counts come back
/// once per layer to size the (expert, token tile) map, as in the Flash-Next
/// runner. The pinned buffers are reused across layers; each layer waits on
/// its own counts event, which the stream orders after the previous layer's
/// tile upload, so a buffer is never rewritten while a copy still reads it.
struct RoutedHostState {
  std::uint32_t* counts{nullptr};
  std::int32_t* tiles{nullptr};
  std::size_t count_capacity{0};
  std::size_t tile_capacity{0};
  hipEvent_t counts_ready{nullptr};

  void Reserve(std::size_t experts, std::size_t tiles_needed) {
    if (counts_ready == nullptr) {
      HIP_CHECK(hipEventCreateWithFlags(&counts_ready, hipEventDisableTiming));
    }
    if (experts > count_capacity) {
      if (counts != nullptr) {
        HIP_CHECK(hipHostFree(counts));
      }
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&counts),
                              experts * sizeof(std::uint32_t)));
      count_capacity = experts;
    }
    if (tiles_needed > tile_capacity) {
      if (tiles != nullptr) {
        HIP_CHECK(hipHostFree(tiles));
      }
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&tiles),
                              tiles_needed * sizeof(std::int32_t)));
      tile_capacity = tiles_needed;
    }
  }
};

RoutedHostState& GetRoutedHostState() {
  static thread_local RoutedHostState state;
  return state;
}

/// Flash-Next's routed F16 GEMM covers Q8_0 experts at any width that is a
/// whole number of 64-element blocks.
bool UseRoutedF16Experts(const models::qwen::MoeLayerView& view,
                         std::size_t hidden, std::size_t expert_ff) {
  return view.gate_exps.type == core::GgmlType::kQ8_0 &&
         view.up_exps.type == core::GgmlType::kQ8_0 &&
         view.down_exps.type == core::GgmlType::kQ8_0 && hidden % 64 == 0 &&
         expert_ff % 64 == 0;
}

/// Token rows per routed tile: the wide tile once the mean bucket fills
/// most of it (the weights are dequantized per tile), as in Flash-Next.
std::uint32_t RoutedTileRows(std::size_t slots, std::uint32_t n_experts) {
  return slots >= static_cast<std::size_t>(16) * n_experts ? 48U : 16U;
}

/// Batched routed MoE FFN (qwen35moe) for one prefill chunk. The router and
/// shared expert reuse the dense prefill GEMM routes; the routed experts run
/// the llama.cpp-derived tiled MMQ MoE kernels (Q8_0/Q6_K), BF16 experts (kept
/// by UD quants in a few layers) run the grouped WMMA GEMM, and any remaining
/// format uses the per-slot warp GEMV fallback.
/// Smallest prefill chunk that moves MoE-model BF16 dense projections from
/// the exact small-batch route to the WMMA GEMM.
constexpr std::size_t kBf16WmmaMinPrefillBatch = 128;

/// Bounds the routed MMQ grid for one chunk. Top-k experts are distinct, so no
/// expert receives more than n_tokens rows; without the hint the down
/// projection, which passes every slot as its own row, sizes its grid for all
/// n_tokens * n_used slots per expert and dispatches mostly empty blocks
/// (measured +8-10% prefill). Flash-Next's fitted column tile width on top of
/// this was no faster. The hint is process-wide state, so it is cleared when
/// the chunk's MoE is done.
class RoutedMmqHints {
public:
  explicit RoutedMmqHints(std::size_t n_tokens) {
    qfn_mmq_set_routed_max_expert_rows(static_cast<int>(n_tokens));
  }
  ~RoutedMmqHints() { qfn_mmq_set_routed_max_expert_rows(0); }
  RoutedMmqHints(const RoutedMmqHints&) = delete;
  RoutedMmqHints& operator=(const RoutedMmqHints&) = delete;
};

template<typename GemmWeight, typename ReadsQ8>
void ExecuteMoePrefillChunk(QwenGpuArena& arena, const QwenMoeScratch& moe,
                            const models::QwenLayerWeights& layer,
                            const core::ModelConfig& config,
                            std::size_t batch_size, GemmWeight& gemm_weight,
                            ReadsQ8& reads_q8_act) {
  const auto view = models::qwen::MakeMoeView(layer, config);
  const std::size_t hidden = config.hidden_size;
  const std::uint32_t n_experts = config.expert_count;
  const std::uint32_t n_used = config.expert_used_count;
  const std::size_t expert_ff = config.expert_ff_length;
  const std::size_t shared_ff = config.expert_shared_ff_length;
  const std::size_t slots = batch_size * n_used;
  const hipStream_t stream = arena.stream;

  // Router logits [B, E] and shared-expert gate [B]: F32 weights on the
  // hipBLAS FP32 route.
  LaunchHipblasGEMM(arena.hipblas_handle, view.router.data, false,
                    arena.d_normed, moe.router_logits.data(), batch_size,
                    n_experts, hidden, arena.d_scratch_bf16, stream);
  LaunchHipblasGEMM(arena.hipblas_handle, view.shexp_gate_inp.data, false,
                    arena.d_normed, moe.shexp_gate.data(), batch_size, 1,
                    hidden, arena.d_scratch_bf16, stream);
  LaunchMoeRouterTopK(moe.router_logits.data(), n_experts, moe.ids.data(),
                      moe.weights.data(), batch_size, n_experts, n_used,
                      stream);

  // Q8_0 experts take Flash-Next's routed F16 WMMA GEMMs. The per-expert
  // counts are downloaded now so the shared expert below hides the copy.
  const bool routed_f16 = UseRoutedF16Experts(view, hidden, expert_ff);
  auto* routed_counts =
      reinterpret_cast<std::uint32_t*>(moe.routed_counts.data());
  RoutedHostState& routed_host = GetRoutedHostState();
  if (routed_f16) {
    routed_host.Reserve(n_experts, RoutedTiles(slots, n_experts));
    routed::ExpertCounts(moe.ids.data(), routed_counts,
                         static_cast<std::uint32_t>(batch_size), n_experts,
                         n_used, stream);
    HIP_CHECK(hipMemcpyAsync(routed_host.counts, routed_counts,
                             n_experts * sizeof(std::uint32_t),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipEventRecord(routed_host.counts_ready, stream));
  }

  // Shared expert through the dense prefill FFN chain (width shared_ff). The
  // Q8_0 routes read the tiled Q8_1 activation, which the dense section would
  // have quantized from the FFN-normed BF16 staging buffer; do it here for
  // the MoE branch (d_scratch_bf16 still holds the FFN norm output).
  LaunchQuantizeActivationQ8_1(arena.d_scratch_bf16, arena.d_scratch_q8_act,
                               batch_size, hidden, stream);
  gemm_weight(view.shexp_gate, arena.d_scratch_bf16, arena.d_normed,
              arena.d_ffn_gate, shared_ff, hidden, arena.d_scratch_q8_act);
  gemm_weight(view.shexp_up, arena.d_scratch_bf16, arena.d_normed,
              arena.d_ffn_up, shared_ff, hidden, arena.d_scratch_q8_act);
  LaunchBatchedSwiGLUActivation(arena.d_ffn_gate, arena.d_ffn_up,
                                arena.d_ffn_act, arena.d_scratch_bf16,
                                batch_size * shared_ff, stream);
  if (reads_q8_act(view.shexp_down)) {
    LaunchQuantizeActivationQ8_1(arena.d_scratch_bf16, arena.d_scratch_q8_act,
                                 batch_size, shared_ff, stream);
  }
  gemm_weight(view.shexp_down, arena.d_scratch_bf16, arena.d_ffn_act,
              moe.shexp_out.data(), hidden, shared_ff,
              arena.d_scratch_q8_act);

  if (routed_f16) {
    // Tile map from the real bucket sizes: one entry per (expert, token tile)
    // of its 16-padded bucket, packed expert | tile << 16.
    HIP_CHECK(hipEventSynchronize(routed_host.counts_ready));
    const std::uint32_t tile_rows = RoutedTileRows(slots, n_experts);
    std::uint32_t n_tiles = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      const std::uint32_t padded = (routed_host.counts[e] + 15U) / 16U * 16U;
      for (std::uint32_t j = 0; j * tile_rows < padded; ++j) {
        routed_host.tiles[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
      }
    }
    HIP_CHECK(hipMemcpyAsync(moe.routed_tiles.data(), routed_host.tiles,
                             n_tiles * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream));
    routed::RoutedCompact(moe.ids.data(), routed_counts,
                          moe.routed_bounds.data(), moe.routed_cursors.data(),
                          moe.rows_token.data(), moe.rows_slot.data(),
                          static_cast<std::uint32_t>(batch_size), n_used,
                          n_experts, stream);
    routed::NarrowActivations(arena.d_normed, moe.x_half.data(), false,
                              batch_size * hidden, stream);
    const auto* x_half = reinterpret_cast<const __half*>(moe.x_half.data());
    // gate -> F32; up applies SwiGLU against it and writes F16; down reads
    // those F16 rows and writes F16 for the combine. Each output row is its
    // (token, slot) index.
    auto* act_half = reinterpret_cast<__half*>(moe.up_e.data());
    auto* down_half = reinterpret_cast<__half*>(moe.down_e.data());
    const bool ok =
        routed::RoutedF16Gemm(view.gate_exps.data, routed::WeightType::kQ8_0,
                              x_half, moe.routed_tiles.data(), n_tiles,
                              tile_rows, moe.routed_bounds.data(),
                              moe.rows_token.data(), moe.rows_slot.data(),
                              nullptr, moe.gate_e.data(), nullptr, expert_ff,
                              hidden, stream) &&
        routed::RoutedF16Gemm(view.up_exps.data, routed::WeightType::kQ8_0,
                              x_half, moe.routed_tiles.data(), n_tiles,
                              tile_rows, moe.routed_bounds.data(),
                              moe.rows_token.data(), moe.rows_slot.data(),
                              moe.gate_e.data(), nullptr, act_half, expert_ff,
                              hidden, stream) &&
        routed::RoutedF16Gemm(view.down_exps.data, routed::WeightType::kQ8_0,
                              act_half, moe.routed_tiles.data(), n_tiles,
                              tile_rows, moe.routed_bounds.data(),
                              moe.rows_slot.data(), moe.rows_slot.data(),
                              nullptr, nullptr, down_half, hidden, expert_ff,
                              stream);
    if (!ok) {
      throw std::runtime_error("routed F16 expert GEMM failed");
    }
    routed::MoeEpilogueVec4F16(down_half, moe.weights.data(),
                               moe.shexp_out.data(), moe.shexp_gate.data(), 1,
                               arena.d_ffn_out,
                               static_cast<std::uint32_t>(batch_size), n_used,
                               hidden, stream);
    LaunchBatchedResidualAdd(arena.d_hidden, arena.d_ffn_out, arena.d_hidden,
                             batch_size, hidden, stream);
    return;
  }

  const RoutedMmqHints mmq_hints(batch_size);

  // BF16 experts group their slots by expert once per layer, on first use.
  const MoeGroupedScratch grouped{.sorted_slots = moe.ids_dst.data(),
                                  .tiles = moe.ids_src1.data(),
                                  .expert_bounds = moe.expert_bounds.data()};
  bool slots_grouped = false;
  const auto grouped_bf16 = [&](const models::QwenTensorRef& weight,
                                std::size_t m, std::size_t k) {
    if (weight.type != core::GgmlType::kBF16 ||
        !IsMoeGroupedBf16GemmSupported(m, k, n_experts)) {
      return false;
    }
    if (!slots_grouped) {
      LaunchMoeGroupSlots(moe.ids.data(), static_cast<std::uint32_t>(slots),
                          n_experts, grouped, stream);
      slots_grouped = true;
    }
    return true;
  };

  // Routed gate/up share one activation quantization when both take the same
  // MMQ format; BF16 pairs take the grouped GEMM, and other mixed or
  // unsupported formats fall back to the fused per-slot SwiGLU GEMV.
  bool swiglu_done = false;
  if (view.gate_exps.type == core::GgmlType::kQ8_0 &&
      view.up_exps.type == core::GgmlType::kQ8_0) {
    if (qfn_mmq_q8_0_moe_pair(view.gate_exps.data, view.up_exps.data,
                              arena.d_normed, moe.ids.data(),
                              moe.gate_e.data(), moe.up_e.data(),
                              static_cast<int>(expert_ff),
                              static_cast<int>(hidden),
                              static_cast<int>(batch_size),
                              static_cast<int>(n_experts),
                              static_cast<int>(n_used), stream) != 0) {
      throw std::runtime_error("MoE gate/up expert GEMM failed");
    }
  } else if (view.gate_exps.type == core::GgmlType::kQ6_K &&
             view.up_exps.type == core::GgmlType::kQ6_K) {
    if (qfn_mmq_q6_K_moe_pair(view.gate_exps.data, view.up_exps.data,
                              arena.d_normed, moe.ids.data(),
                              moe.gate_e.data(), moe.up_e.data(),
                              static_cast<int>(expert_ff),
                              static_cast<int>(hidden),
                              static_cast<int>(batch_size),
                              static_cast<int>(n_experts),
                              static_cast<int>(n_used), stream) != 0) {
      throw std::runtime_error("MoE gate/up expert GEMM failed");
    }
  } else if (view.gate_exps.type == core::GgmlType::kBF16 &&
             view.up_exps.type == core::GgmlType::kBF16 &&
             grouped_bf16(view.gate_exps, expert_ff, hidden)) {
    LaunchMoeGroupedBf16Gemm(view.gate_exps.data, arena.d_normed,
                             moe.gate_e.data(), expert_ff, hidden,
                             static_cast<std::uint32_t>(slots), n_experts,
                             n_used, grouped, stream);
    LaunchMoeGroupedBf16Gemm(view.up_exps.data, arena.d_normed,
                             moe.up_e.data(), expert_ff, hidden,
                             static_cast<std::uint32_t>(slots), n_experts,
                             n_used, grouped, stream);
  } else {
    LaunchMoeSlotSwigluGemv(view.gate_exps.data, view.gate_exps.type,
                            view.up_exps.data, view.up_exps.type,
                            arena.d_normed, moe.ids.data(), moe.gate_e.data(),
                            expert_ff, hidden, static_cast<std::uint32_t>(slots),
                            n_used, stream);
    swiglu_done = true;
  }
  if (!swiglu_done) {
    LaunchBatchedSwiGLUActivation(moe.gate_e.data(), moe.up_e.data(),
                                  moe.gate_e.data(), nullptr,
                                  slots * expert_ff, stream);
  }

  // The down projection reads one activation row per (token, slot).
  if (view.down_exps.type == core::GgmlType::kQ8_0) {
    if (qfn_mmq_q8_0_moe_raw(view.down_exps.data, moe.gate_e.data(),
                             moe.ids.data(), moe.down_e.data(),
                             static_cast<int>(hidden),
                             static_cast<int>(expert_ff),
                             static_cast<int>(slots),
                             static_cast<int>(n_experts), 1, stream) != 0) {
      throw std::runtime_error("MoE down expert GEMM failed");
    }
  } else if (grouped_bf16(view.down_exps, hidden, expert_ff)) {
    LaunchMoeGroupedBf16Gemm(view.down_exps.data, moe.gate_e.data(),
                             moe.down_e.data(), hidden, expert_ff,
                             static_cast<std::uint32_t>(slots), n_experts,
                             1, grouped, stream);
  } else {
    LaunchMoeSlotGemv(view.down_exps.data, view.down_exps.type,
                      moe.gate_e.data(), moe.ids.data(), moe.down_e.data(),
                      hidden, expert_ff, static_cast<std::uint32_t>(slots), 1,
                      stream);
  }

  LaunchMoeEpilogue(moe.down_e.data(), moe.weights.data(),
                    moe.shexp_out.data(), moe.shexp_gate.data(),
                    arena.d_ffn_out, static_cast<std::uint32_t>(batch_size),
                    n_used, hidden, stream);
  LaunchBatchedResidualAdd(arena.d_hidden, arena.d_ffn_out, arena.d_hidden,
                           batch_size, hidden, stream);
}

bool UseQwen27bFp16Prefill(const models::QwenModelWeights& weights,
                           std::size_t batch) {
  const auto& config = weights.config;
  // Large Qwen27B low-bit prefill is qualified independently from the Q8
  // target and from the small-batch decode/verification kernels.
  if (batch < 1024 || config.hidden_size != 5120 ||
      config.intermediate_size != 17408 || config.num_layers != 64 ||
      config.AttentionSize() != 6144 || config.ssm_inner_size != 6144)
    return false;
  const auto supported = [](const models::QwenTensorRef& tensor) {
    return tensor.type == core::GgmlType::kQ8_0 ||
           detail::IsNativeWmmaQuant(tensor.type);
  };
  bool low_bit_ffn = false;
  for (const auto& layer : weights.layers) {
    if (!supported(layer.ffn_gate) || !supported(layer.ffn_up) ||
        !supported(layer.ffn_down))
      return false;
    low_bit_ffn |= detail::IsNativeWmmaQuant(layer.ffn_gate.type) ||
                   detail::IsNativeWmmaQuant(layer.ffn_up.type) ||
                   detail::IsNativeWmmaQuant(layer.ffn_down.type);
    if (layer.is_full_attention) {
      if (!supported(layer.attn_q) || !supported(layer.attn_k) ||
          !supported(layer.attn_v) || !supported(layer.attn_output))
        return false;
    } else if (!supported(layer.attn_qkv) || !supported(layer.attn_gate) ||
               !supported(layer.ssm_alpha) || !supported(layer.ssm_beta) ||
               !supported(layer.ssm_out))
      return false;
  }
  return low_bit_ffn;
}
}  // namespace

tokenization::TokenId QwenGpuExecutor::ForwardPromptChunk(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t batch_size = prompt_tokens.size();
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  const float eps = 1e-6F;

  if (batch_size == 0) {
    return 0;
  }
  if (batch_size > arena_.GetMaxBatch()) {
    throw std::length_error("prompt chunk exceeds the GPU batch length");
  }
  auto scratch = arena_.GetScratchView(batch_size);
  const auto attention_workspace =
      arena_.GetScratchView(arena_.GetMaxBatch()).ffn;
  const std::size_t target_layer_count = arena_.GetTargetLayerCapture().size();
  const std::size_t prompt_capture_offset = h_prompt_hidden_.size();
  if (capture_prompt_hidden_ && target_layer_count > 0) {
    h_prompt_hidden_.resize(prompt_capture_offset +
                            batch_size * target_layer_count * hidden_size);
  }

  // 1. Copy prompt token IDs to GPU
  std::vector<std::uint32_t> host_tokens(prompt_tokens.begin(),
                                         prompt_tokens.end());
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_tokens.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));

  // 2. Batched Embedding lookup: d_hidden [B, hidden_size]
  LaunchBatchedEmbeddingLookup(
      weights_.token_embd.data, weights_.token_embd.type,
      scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(),
      batch_size, hidden_size, arena_.stream);

  vision_input_.Inject(scratch.decode.hidden.data(), start_pos, batch_size,
                       hidden_size, 1, arena_.stream);

  const bool half_prefill = UseQwen27bFp16Prefill(weights_, batch_size);
  const auto gemm_weight = [&](const models::QwenTensorRef& w,
                               const void* bf16_input, const float* fp32_input,
                               float* output, std::size_t m, std::size_t k,
                               const void* q8_act = nullptr) {
    if (half_prefill) {
      LaunchBatchedQuantGEMMFp16(w.type, w.data, bf16_input, output, batch_size,
                                 m, k, arena_.stream);
      return;
    }
    const auto resolution = models::qwen::ResolveQwenGemmRoute(
        {.type = w.type,
         .batch_size = batch_size,
         .m = m,
         .k = k,
         .mode = models::qwen::QwenGemmMode::kHipPrefill});
    if (!resolution.accepted()) {
      std::abort();
    }
    switch (resolution.route) {
      case models::qwen::QwenGemmRoute::kHipPrefillBf16Fp32:
        // MoE targets have no exact cold/live continuation contract, and the
        // few BF16 dense projections UD quants keep (Qwen3.6-35B-A3B
        // UD-Q8_K_XL layer 1) cost ~9% of a 4K prefill on the exact route
        // below, so large chunks take the BF16 WMMA GEMM. Dense Qwen keeps the
        // exact route, which its continuation logits depend on.
        if (config.IsMoE() && batch_size >= kBf16WmmaMinPrefillBatch &&
            IsMoeGroupedBf16GemmSupported(m, k, 1)) {
          LaunchBf16WmmaGemm(w.data, fp32_input, output, batch_size, m, k,
                             arena_.stream);
          return;
        }
        // Preserve FP32 activation precision and a fixed reduction order.
        LaunchExactBf16GEMMFp32SmallBatch(w.data, fp32_input, output,
                                          batch_size, m, k, arena_.stream);
        return;
      case models::qwen::QwenGemmRoute::kHipPrefillF32Blas:
        LaunchHipblasGEMM(arena_.hipblas_handle, w.data, false, fp32_input,
                          output, batch_size, m, k, arena_.d_scratch_bf16,
                          arena_.stream);
        return;
      case models::qwen::QwenGemmRoute::kHipPrefillQuantDirect: {
        if (w.type == core::GgmlType::kQ8_0 ||
            detail::IsNativeWmmaQuant(w.type)) {
          // Route Q8_0 and every K-quant/IQ format the UD-Q4_K_XL shard uses
          // through the native W-quant x A8 WMMA matrix-core kernel with zero
          // scratch dequantization. Without the second half of this condition
          // the K-quants fell into the branch below, which dequantizes the
          // whole weight matrix to a BF16 scratch buffer and then runs a BF16
          // hipBLASLt GEMM -- paying the dequant AND giving up the 2x matrix
          // -core throughput of the int8 path.
          if (q8_act != nullptr) {
            LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, q8_act, output,
                                               batch_size, m, k, arena_.stream);
          } else {
            LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output,
                                   batch_size, m, k, arena_.stream);
          }
        } else if (batch_size > 1) {
          LaunchDequantizeToBf16(w.type, w.data, arena_.d_weights_bf16, m * k,
                                 arena_.stream);
          if (arena_.hipblaslt_gemm != nullptr && m >= 1024 && k >= 1024 &&
              arena_.hipblaslt_gemm->RunBf16(arena_.d_weights_bf16, bf16_input,
                                             output, batch_size, m, k,
                                             arena_.stream)) {
            // hipBLASLt executed successfully
          } else {
            LaunchHipblasGEMMBF16(arena_.hipblas_handle, arena_.d_weights_bf16,
                                  bf16_input, output, batch_size, m, k,
                                  arena_.stream);
          }
        } else {
          LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output, batch_size,
                                 m, k, arena_.stream);
        }
        return;
      }
      default:
        std::abort();
    }
  };

  // 3. Layer stack
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    PrefillLayerCheckpoint(l);
    const auto& layer = weights_.layers[l];
    const auto route_resolution = ResolveQwenLayerRouteWithReasons(
        policy_, QwenExecutionMode::kPrefill, layer.is_full_attention);
    const auto& route_plan = route_resolution.plan;
    detail::EmitQwenRouteResolution(
        "prefill", l, layer.is_full_attention ? "attention" : "ssm",
        route_plan.Fingerprint(),
        static_cast<std::uint32_t>(route_resolution.rejected));

    // opt-q4kxl: "reads the tiled Q8_1 activation" is the property every gate
    // below actually cares about, and it is no longer synonymous with Q8_0.
    // The K-quant/IQ formats in the UD-Q4_K_XL shard take the same
    // pre-quantized WMMA route, so they must drive the same fusion decisions --
    // a gate left at `== kQ8_0` would leave the activation buffer unwritten
    // under a route that reads it.
    const auto reads_q8_act = [&](const models::QwenTensorRef& w) {
      return !half_prefill && (w.type == core::GgmlType::kQ8_0 ||
                               detail::IsNativeWmmaQuant(w.type));
    };
    const auto is_q8 = reads_q8_act;
    // opt-c173-norm-quant: when every projection reading a norm is Q8_0 the
    // FP32 normed row and the BF16 staging copy are both dead, so the norm can
    // write the tiled Q8_1 activation directly and skip two round trips.
    const bool norm_feeds_q8_only =
        IsFusedRMSNormQuantizeQ8_1Supported(hidden_size) &&
        (layer.is_full_attention
             ? (is_q8(layer.attn_q) && is_q8(layer.attn_k) &&
                is_q8(layer.attn_v))
             : (is_q8(layer.attn_qkv) && is_q8(layer.attn_gate) &&
                is_q8(layer.ssm_alpha) && is_q8(layer.ssm_beta)));
    // Where these hold, the BF16 and FP32 buffers passed to gemm_weight below
    // are stale: the route for a Q8_0 tensor always reads the quantized
    // activation, and the flags are exactly the condition that every consumer
    // is Q8_0, so no route can reach them.
    const bool ffn_feeds_q8_only =
        IsFusedRMSNormQuantizeQ8_1Supported(hidden_size) &&
        is_q8(layer.ffn_gate) && is_q8(layer.ffn_up);

    // Pre-layer RMSNorm (generates BF16 into d_scratch_bf16 directly)
    if (half_prefill) {
      LaunchBatchedRMSNormFp16(arena_.d_hidden, nullptr,
                               static_cast<const float*>(layer.attn_norm.data),
                               nullptr, arena_.d_scratch_bf16, batch_size,
                               hidden_size, eps, arena_.stream);
    } else if (norm_feeds_q8_only) {
      LaunchBatchedFusedRMSNormQuantizeQ8_1(
          arena_.d_hidden, /*residual=*/nullptr,
          static_cast<const float*>(layer.attn_norm.data), /*sum_out=*/nullptr,
          arena_.d_scratch_q8_act, batch_size, hidden_size, eps, arena_.stream);
    } else {
      LaunchBatchedRMSNorm(arena_.d_hidden,
                           static_cast<const float*>(layer.attn_norm.data),
                           arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                           hidden_size, eps, arena_.stream);
    }

    if (layer.is_full_attention) {
      // The quantized activation is read only by Q8_0 projections; when q/k/v
      // are all BF16 (as in the Q8_K_XL artifact) nothing consumes it, and when
      // they are all Q8_0 the fused norm already wrote it.
      if (!norm_feeds_q8_only &&
          (reads_q8_act(layer.attn_q) || reads_q8_act(layer.attn_k) ||
           reads_q8_act(layer.attn_v))) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      gemm_weight(layer.attn_q, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_qkv, q_projection_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_k, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_k, kv_size, hidden_size, arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_v, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_v, kv_size, hidden_size, arena_.d_scratch_q8_act);

      LaunchBatchedUnpackQG(arena_.d_ssm_qkv, arena_.d_q, arena_.d_ssm_gate,
                            batch_size, config.num_attention_heads,
                            config.head_dim, arena_.stream);

      const std::size_t total_k = arena_.GetAttentionKvPlaneElements();
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;

      // QK-Norm + RoPE + KV-cache write fused into one kernel
      // (opt-c010-qk-rope-kv). The unfused chain stays wired behind the policy
      // toggle as the independent reference.
      const bool fused_qknorm_rope_kv =
          route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (fused_qknorm_rope_kv) {
        LaunchBatchedFusedQKNormRoPEKvWrite(
            arena_.d_q, arena_.d_k, arena_.d_v,
            static_cast<const float*>(layer.attn_q_norm.data),
            static_cast<const float*>(layer.attn_k_norm.data), arena_.d_q,
            arena_.d_k, arena_.d_kv_cache,
            OffsetIfPresent(arena_.d_kv_cache, total_k),
            arena_.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                total_k),
            attn_layer_idx, start_pos, batch_size, arena_.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.rotary_dim, config.rope_theta, eps,
            arena_.stream, vision_input_.rope());
      } else {
        if (!layer.attn_q_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),
              arena_.d_q, batch_size, config.num_attention_heads,
              config.head_dim, eps, arena_.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),
              arena_.d_k, batch_size, config.num_key_value_heads,
              config.head_dim, eps, arena_.stream);
        }

        LaunchBatchedRoPE(
            arena_.d_q, arena_.d_k, batch_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, config.rotary_dim,
            start_pos, config.rope_theta, arena_.stream, vision_input_.rope());
      }

      const std::size_t visible_context =
          static_cast<std::size_t>(start_pos) + batch_size;
      // One masked WMMA pass covers the visible prefix and causal diagonal.
      // Unsupported shapes use the tiled or scalar fallback below.
      // opt-c180-kv-resync: the fused QK-norm/RoPE kernel above already wrote
      // this chunk's K and V into the selected canonical cache plane. Earlier
      // chunks and all three decode paths maintain that same plane. The
      // attention launcher's own pack pass would therefore rewrite identical
      // bytes, and an FP16 prefix sync would reconvert a prefix that already
      // matches -- work that grows linearly with depth. The unfused fallback
      // below only applies RoPE in place and never touches the cache, so it
      // still needs the pack.
      const bool kv_already_written = fused_qknorm_rope_kv;
      const bool wmma_attention = LaunchQwenWmmaAttention(
          arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
          arena_.d_kv_cache, OffsetIfPresent(arena_.d_kv_cache, total_k),
          arena_.d_attention_kv_f16,
          OffsetIfPresent(
              static_cast<std::uint16_t*>(arena_.d_attention_kv_f16), total_k),
          arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
          arena_.GetMaxContext(), config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, arena_.stream,
          /*lse_out=*/nullptr, /*key_begin=*/0,
          /*skip_kv_write=*/kv_already_written, attention_workspace.gate,
          attention_workspace.up);
      if (wmma_attention) {
        detail::EmitAttentionDispatch("prefill_wmma", "");
      }

      if (!wmma_attention) {
        bool tiled_attention = false;
        detail::DispatchPrefillAttention(
            visible_context,
            [&] {
              tiled_attention = LaunchBatchedAttentionTile(
                  arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                  arena_.d_kv_cache,
                  OffsetIfPresent(arena_.d_kv_cache, total_k),
                  arena_.d_attention_kv_f16,
                  OffsetIfPresent(
                      static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                      total_k),
                  arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
                  arena_.GetMaxContext(), config.num_attention_heads,
                  config.num_key_value_heads, config.head_dim, arena_.stream);
              return tiled_attention;
            },
            [&] {
              LaunchBatchedAttention(
                  arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                  arena_.d_kv_cache,
                  OffsetIfPresent(arena_.d_kv_cache, total_k),
                  arena_.d_attention_kv_f16,
                  OffsetIfPresent(
                      static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                      total_k),
                  arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
                  arena_.GetMaxContext(), config.num_attention_heads,
                  config.num_key_value_heads, config.head_dim, arena_.stream,
                  fused_qknorm_rope_kv);
            });
        if (tiled_attention) {
          detail::EmitAttentionDispatch("prefill_tiled", "");
        } else {
          detail::EmitAttentionDispatch(
              "prefill_baseline",
              detail::ShouldAttemptOptimizedAttention(visible_context)
                  ? "prefill_tiled: rejected"
                  : "prefill_tiled: below_threshold");
        }
      }

      // opt-c172-fp32-quant: the Q8_0 route reads only the quantized
      // activation, so the BF16 staging buffer is dead. Quantizing straight
      // from FP32 drops one launch and one round trip (about 75 MB per layer
      // at batch 2048) and keeps the activation's full precision going into the
      // Q8_1 codes instead of rounding to BF16 first.
      if (half_prefill) {
        LaunchFloatToFp16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                          batch_size * attention_size, arena_.stream);
      } else if (reads_q8_act(layer.attn_output)) {
        LaunchQuantizeActivationQ8_1FromFp32(
            arena_.d_ssm_out, arena_.d_scratch_q8_act, batch_size,
            attention_size, arena_.stream);
      } else {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * attention_size, arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     attention_size, arena_.stream);
      }
      if (half_prefill) {
        LaunchBatchedQuantGEMMResidualFp16(
            layer.attn_output.type, layer.attn_output.data,
            arena_.d_scratch_bf16, arena_.d_hidden, batch_size, hidden_size,
            attention_size, arena_.stream);
      } else {
        gemm_weight(layer.attn_output, arena_.d_scratch_bf16, arena_.d_ssm_out,
                    arena_.d_attn_out, hidden_size, attention_size,
                    arena_.d_scratch_q8_act);
      }
    } else {
      if (!half_prefill && !norm_feeds_q8_only) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      gemm_weight(layer.attn_qkv, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_qkv, ssm_qkv_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_gate, ssm_inner_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.ssm_alpha, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_alpha_buf, time_step_rank, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.ssm_beta, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_beta_buf, time_step_rank, hidden_size,
                  arena_.d_scratch_q8_act);
      if (arena_.IsSsmReplayCaptureActive()) {
        LaunchCaptureBatchedSsmReplay(
            arena_.d_ssm_qkv, arena_.d_alpha_buf, arena_.d_beta_buf,
            arena_.GetSsmReplayCapture(), config.SsmLayerIndex(l), start_pos,
            batch_size, ssm_qkv_size, time_step_rank, arena_.stream);
      }

      // The row-split recurrence parallelizes independent state rows. Its
      // epilogue emits the activation consumed by ssm_out directly.
      const bool ssm_epilogue_q8 = reads_q8_act(layer.ssm_out) &&
                                   IsFusedSSMEpilogueQuantizeQ8_1Supported(
                                       config.SsmValueSize(), ssm_inner_size);
      const bool ssm_row_split = detail::ShouldUseSsmRowSplitRecurrence(
          IsDeltaNetRowSplitSupported(config.ssm_state_size,
                                      config.SsmValueSize()),
          arena_.d_ssm_kq_scales != nullptr &&
              arena_.d_ssm_alpha_beta != nullptr);
      if (ssm_row_split) {
        LaunchBatchedSSMConvRecurrenceRowSplit(
            arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
            arena_.d_ssm_conv_state, arena_.d_conv_out,
            arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
            static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
            arena_.d_ssm_out,
            ssm_epilogue_q8 ? arena_.d_scratch_q8_act : nullptr,
            arena_.d_ssm_kq_scales, arena_.d_ssm_alpha_beta,
            config.SsmLayerIndex(l), batch_size, ssm_qkv_size,
            config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena_.stream,
            arena_.GetRecurrentStateStorage(),
            half_prefill ? arena_.d_scratch_bf16 : nullptr);
      } else {
        LaunchBatchedSSMConvRecurrence(
            arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
            arena_.d_ssm_conv_state, arena_.d_conv_out,
            arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
            static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
            arena_.d_ssm_out, config.SsmLayerIndex(l), batch_size, ssm_qkv_size,
            config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena_.stream,
            arena_.GetRecurrentStateStorage());
      }

      if (half_prefill) {
        if (!ssm_row_split) {
          LaunchFloatToFp16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                            batch_size * ssm_inner_size, arena_.stream);
        }
      } else if (ssm_row_split && ssm_epilogue_q8) {
        // The recurrence epilogue already wrote the quantized activation.
      } else if (reads_q8_act(layer.ssm_out)) {
        LaunchQuantizeActivationQ8_1FromFp32(
            arena_.d_ssm_out, arena_.d_scratch_q8_act, batch_size,
            ssm_inner_size, arena_.stream);
      } else {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * ssm_inner_size, arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     ssm_inner_size, arena_.stream);
      }
      if (half_prefill) {
        LaunchBatchedQuantGEMMResidualFp16(
            layer.ssm_out.type, layer.ssm_out.data, arena_.d_scratch_bf16,
            arena_.d_hidden, batch_size, hidden_size, ssm_inner_size,
            arena_.stream);
      } else {
        gemm_weight(layer.ssm_out, arena_.d_scratch_bf16, arena_.d_ssm_out,
                    arena_.d_attn_out, hidden_size, ssm_inner_size,
                    arena_.d_scratch_q8_act);
      }
    }

    if (half_prefill) {
      LaunchBatchedRMSNormFp16(arena_.d_hidden, nullptr,
                               static_cast<const float*>(layer.ffn_norm.data),
                               nullptr, arena_.d_scratch_bf16, batch_size,
                               hidden_size, eps, arena_.stream);
    } else if (ffn_feeds_q8_only) {
      // The post-attention residual add folds into the norm: one pass reads the
      // hidden state and the attention output, writes the updated hidden state
      // for the next residual link, and emits the Q8_1 activation.
      LaunchBatchedFusedRMSNormQuantizeQ8_1(
          arena_.d_hidden, arena_.d_attn_out,
          static_cast<const float*>(layer.ffn_norm.data), arena_.d_hidden,
          arena_.d_scratch_q8_act, batch_size, hidden_size, eps, arena_.stream);
    } else {
      LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                               arena_.d_hidden, batch_size, hidden_size,
                               arena_.stream);

      // FFN RMSNorm (generates BF16 into d_scratch_bf16 directly)
      LaunchBatchedRMSNorm(arena_.d_hidden,
                           static_cast<const float*>(layer.ffn_norm.data),
                           arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                           hidden_size, eps, arena_.stream);
    }

    // Which buffer holds the Q8_1 form of the SwiGLU output ffn_down consumes.
    // The fused epilogue below moves it out of the shared activation scratch.
    const void* ffn_down_q8_act = arena_.d_scratch_q8_act;
    const void* ffn_down_input = arena_.d_scratch_bf16;

    if (config.IsMoE()) {
      ExecuteMoePrefillChunk(arena_, scratch.moe, layer, config, batch_size,
                             gemm_weight, reads_q8_act);
    } else {
    {
      if (!half_prefill && !ffn_feeds_q8_only) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      // opt-c192-swiglu-epilogue: when ffn_down also reads the Q8_1 activation,
      // the whole gate -> up -> SwiGLU -> quantize chain collapses into the two
      // projections, because the up projection can apply SwiGLU against the
      // already-written gate and emit the quantized activation from its own
      // accumulator. The FP32 up intermediate is then dead, so its allocation
      // is where the Q8_1 result goes -- reusing it keeps resident memory flat
      // and avoids writing the [batch, k] activation the same kernel is
      // reading.
      const bool fused_dual_swiglu =
          reads_q8_act(layer.ffn_gate) && reads_q8_act(layer.ffn_up) &&
          layer.ffn_gate.type == layer.ffn_up.type &&
          reads_q8_act(layer.ffn_down) &&
          IsFusedSwiGluGemmEpilogueSupported(
              layer.ffn_gate.type, batch_size, intermediate_size,
              batch_size * intermediate_size * sizeof(float));
      if (half_prefill) {
        const bool paired = TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
            layer.ffn_gate.type, layer.ffn_up.type, layer.ffn_gate.data,
            layer.ffn_up.data, arena_.d_scratch_bf16, arena_.d_ffn_up,
            batch_size, intermediate_size, hidden_size, arena_.stream);
        if (!paired) {
          gemm_weight(layer.ffn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                      arena_.d_ffn_gate, intermediate_size, hidden_size);
          LaunchBatchedQuantGEMMSwiGLUFp16(
              layer.ffn_up.type, layer.ffn_up.data, arena_.d_scratch_bf16,
              arena_.d_ffn_gate, arena_.d_ffn_up, batch_size, intermediate_size,
              hidden_size, arena_.stream);
        }
        ffn_down_input = arena_.d_ffn_up;
      } else if (fused_dual_swiglu) {
        LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(
            layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
            arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,
            batch_size, intermediate_size, hidden_size, arena_.stream);
        ffn_down_q8_act = arena_.d_ffn_up;
      } else if (reads_q8_act(layer.ffn_gate) && reads_q8_act(layer.ffn_up) &&
                 layer.ffn_gate.type == layer.ffn_up.type) {
        LaunchBatchedDualQuantGEMMPreQuantized(
            layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
            arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,
            batch_size, intermediate_size, hidden_size, arena_.stream);
      } else {
        gemm_weight(layer.ffn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_ffn_gate, intermediate_size, hidden_size,
                    arena_.d_scratch_q8_act);

        gemm_weight(layer.ffn_up, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_ffn_up, intermediate_size, hidden_size,
                    arena_.d_scratch_q8_act);
      }

      // opt-c164-swiglu-quant: when ffn_down reads Q8_0 the only consumer of
      // the activation is the quantized buffer, so SwiGLU can write it
      // directly. That drops the FP32 activation and BF16 scratch round trips
      // (about 500 MB per layer at batch 2048) and one kernel launch. Other
      // ffn_down formats still need the FP32/BF16 forms, so they keep the
      // unfused chain.
      if (half_prefill || fused_dual_swiglu) {
        // The up projection's epilogue already wrote the quantized activation.
      } else if (reads_q8_act(layer.ffn_down)) {
        LaunchBatchedFusedSwiGLUQuantizeQ8_1(
            arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_scratch_q8_act,
            batch_size, intermediate_size, arena_.stream);
      } else {
        LaunchBatchedSwiGLUActivation(arena_.d_ffn_gate, arena_.d_ffn_up,
                                      arena_.d_ffn_act, arena_.d_scratch_bf16,
                                      batch_size * intermediate_size,
                                      arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     intermediate_size, arena_.stream);
      }
    }

    if (half_prefill) {
      LaunchBatchedQuantGEMMResidualFp16(
          layer.ffn_down.type, layer.ffn_down.data, ffn_down_input,
          arena_.d_hidden, batch_size, hidden_size, intermediate_size,
          arena_.stream);
    } else {
      gemm_weight(layer.ffn_down, ffn_down_input, arena_.d_ffn_act,
                  arena_.d_ffn_out, hidden_size, intermediate_size,
                  ffn_down_q8_act);
      LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out,
                               arena_.d_hidden, batch_size, hidden_size,
                               arena_.stream);
    }
    }  // dense FFN

    if (capture_prompt_hidden_) {
      if (const auto tap_index = arena_.GetTargetLayerCaptureIndex(l);
          tap_index.has_value()) {
        float* destination = h_prompt_hidden_.data() + prompt_capture_offset +
                             (*tap_index * hidden_size);
        HIP_CHECK(hipMemcpy2DAsync(
            destination, target_layer_count * hidden_size * sizeof(float),
            scratch.decode.hidden.data(), hidden_size * sizeof(float),
            hidden_size * sizeof(float), batch_size, hipMemcpyDeviceToHost,
            arena_.stream));
      }
    }
  }

  if (capture_prompt_hidden_) {
    if (target_layer_count == 0) {
      const std::size_t old_size = h_prompt_hidden_.size();
      const std::size_t chunk_elements = batch_size * hidden_size;
      h_prompt_hidden_.resize(old_size + chunk_elements);
      HIP_CHECK(hipMemcpyAsync(h_prompt_hidden_.data() + old_size,
                               scratch.decode.hidden.data(),
                               chunk_elements * sizeof(float),
                               hipMemcpyDeviceToHost, arena_.stream));
    }
    HIP_CHECK(hipStreamSynchronize(arena_.stream));
  }
  last_hidden_offset_ = (batch_size - 1) * hidden_size;

  if (!compute_logits) {
    return 0;
  }

  // 4. Output Norm for final token
  const float* final_hidden =
      scratch.decode.hidden.data() + ((batch_size - 1) * hidden_size);
  LaunchRMSNorm(final_hidden,
                static_cast<const float*>(weights_.output_norm.data),
                scratch.decode.normed.data(), hidden_size, eps, arena_.stream);

  // 5. LM Head Logits GEMV on final token
  LaunchGEMV(weights_.output.data, weights_.output.type,
             scratch.decode.normed.data(), scratch.decode.logits.data(),
             vocab_size, hidden_size, arena_.stream);

  // 6. GPU Argmax. The sampled-token view enters its documented alias epoch
  // only after all SSM layer uses of alpha have completed.
  auto* d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUArgmax(scratch.decode.logits.data(), d_out_token, vocab_size,
                  arena_.stream);

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  return CheckedSampleToken(next_token_id, config.vocab_size);
}

std::vector<tokenization::TokenId> QwenGpuExecutor::ForwardVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_logits) {
  return ForwardDecodeEquivalentVerificationChunk(candidate_tokens, start_pos,
                                                  capture_logits);
}

void QwenGpuExecutor::CommitVerificationChunk(
    std::span<const tokenization::TokenId> committed_tokens,
    std::uint32_t start_pos) {
  if (committed_tokens.empty()) {
    return;
  }
  std::size_t replayed = 0;
  if (replaying_ssm_state_) {
    while (replayed < committed_tokens.size() &&
           arena_.CanReplaySsmPosition(start_pos +
                                       static_cast<std::uint32_t>(replayed)))
      ++replayed;
    if (replayed != 0)
      ReplaySsmState(start_pos, static_cast<std::uint32_t>(replayed));
  }
  replaying_ssm_state_ = false;
  if (replayed == committed_tokens.size())
    return;
  arena_.DisableSsmReplayCapture();

  const bool capture_hidden = capture_prompt_hidden_;
  capture_prompt_hidden_ = false;
  try {
    // Unrecorded rows must use the original decode arithmetic. Prefill
    // projections can round differently and leave a different recurrent state.
    for (std::size_t row = replayed; row < committed_tokens.size(); ++row)
      (void)ForwardToken(committed_tokens[row],
                         start_pos + static_cast<std::uint32_t>(row), false);
  } catch (...) {
    capture_prompt_hidden_ = capture_hidden;
    throw;
  }
  capture_prompt_hidden_ = capture_hidden;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
