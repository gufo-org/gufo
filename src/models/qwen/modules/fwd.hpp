#ifndef STRIX_MODELS_QWEN_MODULES_FWD_HPP_
#define STRIX_MODELS_QWEN_MODULES_FWD_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/ssm.hpp"
#include "src/models/qwen/state.hpp"

namespace strix::models::qwen {

/// CPU reference and HIP launch overloads are intentionally distinct. Backend
/// selection is a composition-time type choice, not a runtime tag branch.
void NormForward(const CpuModuleContext& ctx, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept;
void NormForward(const HipModuleContext& ctx, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept;

void RopeForward(const CpuModuleContext& ctx, const RopeLayerView& view,
                 std::span<float> q, std::span<float> k,
                 std::uint32_t pos) noexcept;

void AttnForward(const CpuLayerContext& ctx, const AttnLayerView& view,
                 std::span<const float> x, QwenKvCache& kv, std::uint32_t pos,
                 std::span<float> out) noexcept;

void SsmForward(const CpuLayerContext& ctx, const SsmLayerView& view,
                std::span<const float> x, QwenSsmCache& state,
                std::span<float> out) noexcept;

void FfnForward(const CpuModuleContext& ctx, const FfnLayerView& view,
                std::span<const float> x, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> out) noexcept;
void FfnForward(const HipModuleContext& ctx, const FfnLayerView& view,
                std::span<const float> x, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> out) noexcept;

void ResidualAdd(const CpuModuleContext& ctx, std::span<float> dst,
                 std::span<const float> src) noexcept;
void ResidualAdd(const HipModuleContext& ctx, std::span<float> dst,
                 std::span<const float> src) noexcept;

void QuantGemm(const CpuModuleContext& ctx, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept;
void QuantGemm(const HipModuleContext& ctx, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept;

void EmbedForward(const CpuModuleContext& ctx, std::uint32_t token_id,
                  const QwenTensorRef& token_embd, std::size_t hidden_size,
                  std::span<float> out) noexcept;

void UnembedForward(const CpuLayerContext& ctx,
                    const QwenTensorRef& output_norm,
                    const QwenTensorRef& output_weight,
                    std::span<const float> hidden,
                    std::span<float> logits_out) noexcept;

std::uint32_t SampleForward(const CpuModuleContext& ctx,
                            std::span<const float> logits) noexcept;

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_FWD_HPP_
