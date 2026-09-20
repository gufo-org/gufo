#include "src/models/qwen3_asr/hip/blas.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace gufo::models::qwen3_asr::hip {
namespace {

constexpr float kAlpha = 1.0F;
constexpr float kBeta = 0.0F;
constexpr int kMaxAlgorithms = 16;

/// hipBLASLt workspace made available to a plan.
///
/// Zero reproduces the shared qwen wrapper, whose workspace defaults to zero
/// unless a tuning run asks for one. That matters for behaviour, not just
/// memory: a zero capacity makes every workspace-using algorithm unusable, so
/// the heuristic list the rank policy indexes into is a different, shorter
/// list. Raising it changes which kernel runs and therefore the accumulation
/// order.
constexpr std::size_t kWorkspaceBytes = 0;

/// Which of hipBLASLt's ranked heuristic results to install.
///
/// hipBLASLt returns its algorithms in predicted-goodness order, and on gfx1151
/// that prediction is poor for this model: the top-ranked entry is never the
/// fastest, and the shared qwen picker -- a rank policy calibrated on a
/// different model's shapes at batch 2048 -- lands on a mix that is 33 ms
/// slower end to end than a single measured choice.
///
/// Rank 6 is that measured choice, swept end to end over this model's own
/// shapes. It is not the fastest: rank 8 is 11 ms better (981 ms against
/// 992 ms). Rank 6 is retained because it is the only rank that is at least as
/// accurate as the reference route on every prefill boundary -- cosine
/// 0.999703 against 0.999692, relative L2 0.02444 against 0.02592 -- and this
/// model's contract is exact reproduction of the official token IDs, so a
/// measurable numerical regression is not worth 1% of the request.
///
/// This is an index into a library-generated list, so it is only valid for the
/// ROCm and hipBLASLt versions it was measured against. Re-sweep it after a
/// toolchain bump with tools/bench/asr_prefill_gemm_bench.hip. The guard falls
/// back to the last usable entry if the list is shorter than expected.
constexpr std::size_t kMeasuredHeuristicRank = 6;

[[nodiscard]] bool IsUsable(const hipblasLtMatmulHeuristicResult_t& result,
                            std::size_t workspace_capacity) {
  return result.state == HIPBLAS_STATUS_SUCCESS &&
         result.workspaceSize <= workspace_capacity;
}

struct Plan {
  hipblasLtMatmulDesc_t operation{nullptr};
  hipblasLtMatrixLayout_t a_layout{nullptr};
  hipblasLtMatrixLayout_t x_layout{nullptr};
  hipblasLtMatrixLayout_t y_layout{nullptr};
  hipblasLtMatmulAlgo_t algorithm{};
  std::size_t workspace_bytes{0};
  bool usable{false};

  ~Plan() {
    if (y_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(y_layout);
    }
    if (x_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(x_layout);
    }
    if (a_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(a_layout);
    }
    if (operation != nullptr) {
      (void)hipblasLtMatmulDescDestroy(operation);
    }
  }

  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
};

}  // namespace

void LaunchGemmBf16(hipblasHandle_t handle, const void* a_bf16,
                    const void* x_bf16, float* y, std::size_t batch_size,
                    std::size_t m, std::size_t k, hipStream_t stream) {
  if (handle == nullptr || a_bf16 == nullptr || x_bf16 == nullptr ||
      y == nullptr || batch_size == 0U || m == 0U || k == 0U) {
    throw std::invalid_argument("invalid Qwen3-ASR GEMM inputs");
  }
  if (batch_size > std::numeric_limits<int>::max() ||
      m > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max())
    throw std::length_error("Qwen3-ASR GEMM dimensions exceed library limits");
  if (stream != nullptr &&
      hipblasSetStream(handle, stream) != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Qwen3-ASR hipBLAS stream setup failed");
  const auto status = hipblasGemmEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(m),
      static_cast<int>(batch_size), static_cast<int>(k), &kAlpha, a_bf16,
      HIP_R_16BF, static_cast<int>(k), x_bf16, HIP_R_16BF, static_cast<int>(k),
      &kBeta, y, HIP_R_32F, static_cast<int>(m), HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT);
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Qwen3-ASR hipBLAS GEMM failed: " +
                             std::to_string(status));
}

struct GemmLt::Impl {
  hipblasLtHandle_t handle{nullptr};
  void* workspace{nullptr};
  std::map<std::tuple<std::size_t, std::size_t, std::size_t>,
           std::unique_ptr<Plan>>
      plans;

  ~Impl() {
    plans.clear();
    if (workspace != nullptr) {
      (void)hipFree(workspace);
    }
    if (handle != nullptr) {
      (void)hipblasLtDestroy(handle);
    }
  }

  /// Builds the descriptor, layouts, and algorithm for one shape. A plan with
  /// `usable == false` is cached too, so a shape hipBLASLt cannot serve is only
  /// probed once and then falls back for the rest of the run.
  /// Installs an algorithm and forces hipBLASLt's solution library to
  /// initialize. The name lookups are not diagnostics: hipBLASLt resolves the
  /// library lazily, and whether it is resolved changes the heuristic list
  /// returned for later shapes, so skipping them changes which kernels run.
  void Install(Plan* plan, const hipblasLtMatmulHeuristicResult_t& result) {
    plan->algorithm = result.algo;
    plan->workspace_bytes = result.workspaceSize;
    plan->usable = true;
    (void)hipblaslt_ext::getSolutionNameFromAlgo(handle, plan->algorithm);
    (void)hipblaslt_ext::getKernelNameFromAlgo(handle, plan->algorithm);
  }

  Plan* Resolve(std::size_t batch_size, std::size_t m, std::size_t k) {
    const auto key = std::make_tuple(batch_size, m, k);
    if (const auto found = plans.find(key); found != plans.end()) {
      return found->second.get();
    }
    auto plan = std::make_unique<Plan>();
    Plan* raw = plan.get();
    plans.emplace(key, std::move(plan));
    if (handle == nullptr) {
      return raw;
    }

    if (hipblasLtMatmulDescCreate(&raw->operation, HIPBLAS_COMPUTE_32F,
                                  HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&raw->a_layout, HIP_R_16BF, k, m, k) !=
            HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&raw->x_layout, HIP_R_16BF, k, batch_size,
                                    k) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&raw->y_layout, HIP_R_32F, m, batch_size,
                                    m) != HIPBLAS_STATUS_SUCCESS) {
      return raw;
    }
    // Only TRANSA is set. TRANSB defaults to HIPBLAS_OP_N, and setting it
    // explicitly changes the heuristic result list hipBLASLt returns, hence
    // which algorithm the rank policy lands on.
    const hipblasOperation_t transpose_a = HIPBLAS_OP_T;
    if (hipblasLtMatmulDescSetAttribute(
            raw->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose_a,
            sizeof(transpose_a)) != HIPBLAS_STATUS_SUCCESS) {
      return raw;
    }

    hipblasLtMatmulPreference_t preference = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&preference) !=
        HIPBLAS_STATUS_SUCCESS) {
      return raw;
    }
    const std::size_t capacity = workspace != nullptr ? kWorkspaceBytes : 0U;
    (void)hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &capacity,
        sizeof(capacity));

    std::vector<hipblasLtMatmulHeuristicResult_t> results(kMaxAlgorithms);
    int found = 0;
    const auto status = hipblasLtMatmulAlgoGetHeuristic(
        handle, raw->operation, raw->a_layout, raw->x_layout, raw->y_layout,
        raw->y_layout, preference, kMaxAlgorithms, results.data(), &found);
    (void)hipblasLtMatmulPreferenceDestroy(preference);
    if (status != HIPBLAS_STATUS_SUCCESS || found <= 0) {
      return raw;
    }

    const auto count = static_cast<std::size_t>(found);
    std::size_t rank = std::min(kMeasuredHeuristicRank, count - 1U);
    if (rank < count && IsUsable(results[rank], capacity)) {
      Install(raw, results[rank]);
      return raw;
    }
    for (std::size_t index = 0; index < count; ++index) {
      if (IsUsable(results[index], capacity)) {
        Install(raw, results[index]);
        return raw;
      }
    }
    return raw;
  }
};

GemmLt::GemmLt() : impl_(std::make_unique<Impl>()) {
  if (hipblasLtCreate(&impl_->handle) != HIPBLAS_STATUS_SUCCESS) {
    impl_->handle = nullptr;
    return;
  }
  if (kWorkspaceBytes > 0 &&
      hipMalloc(&impl_->workspace, kWorkspaceBytes) != hipSuccess) {
    impl_->workspace = nullptr;
  }
}

GemmLt::~GemmLt() = default;
GemmLt::GemmLt(GemmLt&&) noexcept = default;
GemmLt& GemmLt::operator=(GemmLt&&) noexcept = default;

bool GemmLt::Run(const void* a_bf16, const void* x_bf16, float* y,
                 std::size_t batch_size, std::size_t m, std::size_t k,
                 hipStream_t stream) {
  if (impl_ == nullptr || impl_->handle == nullptr || a_bf16 == nullptr ||
      x_bf16 == nullptr || y == nullptr || batch_size == 0U || m == 0U ||
      k == 0U) {
    return false;
  }
  Plan* plan = impl_->Resolve(batch_size, m, k);
  if (plan == nullptr || !plan->usable) {
    return false;
  }
  const auto status = hipblasLtMatmul(
      impl_->handle, plan->operation, &kAlpha, a_bf16, plan->a_layout, x_bf16,
      plan->x_layout, &kBeta, y, plan->y_layout, y, plan->y_layout,
      &plan->algorithm, impl_->workspace, plan->workspace_bytes, stream);
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Qwen3-ASR hipBLASLt GEMM failed: " +
                             std::to_string(status));
  return true;
}

}  // namespace gufo::models::qwen3_asr::hip

#endif
