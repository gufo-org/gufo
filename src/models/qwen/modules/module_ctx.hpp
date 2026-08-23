#ifndef STRIX_MODELS_QWEN_MODULES_MODULE_CTX_HPP_
#define STRIX_MODELS_QWEN_MODULES_MODULE_CTX_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/model_config.hpp"
#include "src/models/qwen_state.hpp"

namespace strix::models::qwen {

/// Which compute backend drives a module call. The same module signature is
/// implemented twice — a CPU reference (oracle) and a HIP kernel path — and the
/// per-forward dispatch selects one via this tag. The plan's two-backend
/// contract; implementations land in Phase 2 extraction.
enum class Backend : std::uint8_t { Cpu = 0, Hip = 1 };

/// Opaque handle to the hipBLASLt plan cache.
///
/// NOTE: no `BlasPlanCache` type exists in this codebase yet (the refactor is
/// what introduces it). The name is forward-declared so `ModuleCtx` reserves
/// the slot the plan's contract specifies; the definition arrives with the
/// hipBLASLt plan-cache work. Until then callers pass nullptr.
struct BlasPlanCache;

/// Everything a module needs from the runtime, one object.
///
/// Modules are pure single-stage stateless functions: they read named slices
/// out of `arena` (via the accessors: hidden, normed, q, k, v, attn_scores,
/// ssm_qkv, ssm_gate, mlp_gate, ...) and exchange spans. Cross-module fusion
/// ownership lives in the composition layer (the per-layer `ForwardToken`
/// loop), NOT inside any one module.
struct ModuleCtx {
  core::ModelConfig const* config = nullptr;  ///< model/config dims (never owned)
  QwenScratchArena* arena = nullptr;          ///< named-slice scratch accessors
  BlasPlanCache* blas_plans = nullptr;        ///< hipBLASLt plan cache (placeholder)
  Backend backend = Backend::Cpu;             ///< selected call path (oracle/HIP)

  /// Opaque async-stream handle. nullptr on CPU builds. On HIP builds the real
  /// hipStream_t is a pointer type, so a HIP TU recovers it with:
  ///   hipStream_t s = static_cast<hipStream_t>(ctx.stream);
  /// Kept opaque so this header compiles in CPU-only translation units without
  /// dragging in <hip/hip_runtime.h>.
  void* stream = nullptr;

  std::uint32_t layer_idx = 0;  ///< index into QwenModelWeights::layers
  std::uint32_t pos = 0;        ///< sequence position (RoPE / KV write)
};

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_MODULE_CTX_HPP_
