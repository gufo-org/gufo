#include "blas.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace gufo::models::embedding_gemma2::hip {
namespace {

constexpr float kAlpha = 1.0F;
constexpr float kBeta = 0.0F;
constexpr int kMaxAlgorithms = 32;
constexpr int kTimedRuns = 3;

struct Plan {
  hipblasLtMatmulDesc_t operation{nullptr};
  hipblasLtMatrixLayout_t a_layout{nullptr};
  hipblasLtMatrixLayout_t x_layout{nullptr};
  hipblasLtMatrixLayout_t y_layout{nullptr};
  hipblasLtMatmulAlgo_t algorithm{};
  bool usable{false};

  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
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
};

}  // namespace

struct GemmLt::Impl {
  hipblasLtHandle_t handle{nullptr};
  std::map<std::tuple<std::size_t, std::size_t, std::size_t>,
           std::unique_ptr<Plan>>
      plans;

  ~Impl() {
    plans.clear();
    if (handle != nullptr) {
      (void)hipblasLtDestroy(handle);
    }
  }

  [[nodiscard]] hipblasStatus_t Launch(const Plan& plan, const void* weights,
                                       const void* inputs, float* output,
                                       hipStream_t stream) const {
    return hipblasLtMatmul(handle, plan.operation, &kAlpha, weights,
                           plan.a_layout, inputs, plan.x_layout, &kBeta, output,
                           plan.y_layout, output, plan.y_layout,
                           &plan.algorithm, nullptr, 0, stream);
  }

  /// Times one algorithm on the caller's operands, in milliseconds.
  [[nodiscard]] float Time(Plan* plan, const void* weights, const void* inputs,
                           float* output, hipStream_t stream) const {
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    const bool created = hipEventCreate(&begin) == hipSuccess &&
                         hipEventCreate(&end) == hipSuccess;
    if (!created) {
      if (begin != nullptr) {
        (void)hipEventDestroy(begin);
      }
      return -1.0F;
    }
    float milliseconds = -1.0F;
    // The first launch loads the kernel and is not timed.
    bool launched = Launch(*plan, weights, inputs, output, stream) ==
                    HIPBLAS_STATUS_SUCCESS;
    if (launched && hipEventRecord(begin, stream) == hipSuccess) {
      for (int run = 0; run < kTimedRuns && launched; ++run) {
        launched = Launch(*plan, weights, inputs, output, stream) ==
                   HIPBLAS_STATUS_SUCCESS;
      }
      if (launched && hipEventRecord(end, stream) == hipSuccess &&
          hipEventSynchronize(end) == hipSuccess) {
        (void)hipEventElapsedTime(&milliseconds, begin, end);
      }
    }
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    if (milliseconds < 0.0F) {
      // A failed launch or event leaves a sticky error that would fail a later
      // check.
      (void)hipGetLastError();
    }
    return milliseconds;
  }

  Plan* Resolve(std::size_t rows, std::size_t m, std::size_t k,
                const void* weights, const void* inputs, float* output,
                hipStream_t stream) {
    const auto key = std::make_tuple(rows, m, k);
    if (const auto found = plans.find(key); found != plans.end()) {
      return found->second.get();
    }
    auto owned = std::make_unique<Plan>();
    Plan* plan = owned.get();
    plans.emplace(key, std::move(owned));
    Tune(plan, rows, m, k, weights, inputs, output, stream);
    if (!plan->usable) {
      std::fprintf(stderr,
                   "EmbeddingGemma 2 hipBLASLt has no usable algorithm for "
                   "m=%zu k=%zu rows=%zu; using hipBLAS\n",
                   m, k, rows);
    }
    return plan;
  }

  void Tune(Plan* plan, std::size_t rows, std::size_t m, std::size_t k,
            const void* weights, const void* inputs, float* output,
            hipStream_t stream) {
    if (hipblasLtMatmulDescCreate(&plan->operation, HIPBLAS_COMPUTE_32F,
                                  HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&plan->a_layout, HIP_R_16BF, k, m, k) !=
            HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&plan->x_layout, HIP_R_16BF, k, rows, k) !=
            HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&plan->y_layout, HIP_R_32F, m, rows, m) !=
            HIPBLAS_STATUS_SUCCESS) {
      return;
    }
    const hipblasOperation_t transpose_a = HIPBLAS_OP_T;
    if (hipblasLtMatmulDescSetAttribute(
            plan->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose_a,
            sizeof(transpose_a)) != HIPBLAS_STATUS_SUCCESS) {
      return;
    }
    hipblasLtMatmulPreference_t preference = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&preference) !=
        HIPBLAS_STATUS_SUCCESS) {
      return;
    }
    std::vector<hipblasLtMatmulHeuristicResult_t> results(kMaxAlgorithms);
    int count = 0;
    const hipblasStatus_t status = hipblasLtMatmulAlgoGetHeuristic(
        handle, plan->operation, plan->a_layout, plan->x_layout, plan->y_layout,
        plan->y_layout, preference, kMaxAlgorithms, results.data(), &count);
    (void)hipblasLtMatmulPreferenceDestroy(preference);
    if (status != HIPBLAS_STATUS_SUCCESS || count <= 0) {
      return;
    }

    // No workspace is offered, so only algorithms that need none are usable.
    float best = -1.0F;
    hipblasLtMatmulAlgo_t best_algorithm{};
    for (int index = 0; index < count; ++index) {
      if (results[index].state != HIPBLAS_STATUS_SUCCESS ||
          results[index].workspaceSize != 0) {
        continue;
      }
      plan->algorithm = results[index].algo;
      const float milliseconds = Time(plan, weights, inputs, output, stream);
      if (milliseconds >= 0.0F && (best < 0.0F || milliseconds < best)) {
        best = milliseconds;
        best_algorithm = results[index].algo;
        plan->usable = true;
      }
    }
    if (plan->usable) {
      plan->algorithm = best_algorithm;
    }
  }
};

GemmLt::GemmLt() : impl_(std::make_unique<Impl>()) {
  if (hipblasLtCreate(&impl_->handle) != HIPBLAS_STATUS_SUCCESS) {
    impl_->handle = nullptr;
    std::fprintf(stderr,
                 "EmbeddingGemma 2 hipBLASLt is unavailable; using hipBLAS\n");
  }
}

GemmLt::~GemmLt() = default;

std::size_t GemmLt::Bucket(std::size_t tokens, std::size_t capacity_tokens) {
  // Up to 256 rows a GEMM is launch-bound and padding is free. Above that, a
  // 64-row step pads at most 24%, at 257 rows, and at most 12% from 513 rows
  // on.
  const std::size_t step = tokens <= 256 ? 32U : 64U;
  const std::size_t rounded = (tokens + step - 1U) / step * step;
  return std::min(rounded, capacity_tokens);
}

bool GemmLt::Run(const void* weight_bf16, const void* input_bf16, float* output,
                 std::size_t tokens, std::size_t m, std::size_t k,
                 std::size_t capacity_tokens, hipStream_t stream) {
  if (impl_->handle == nullptr || weight_bf16 == nullptr ||
      input_bf16 == nullptr || output == nullptr || tokens == 0U || m == 0U ||
      k == 0U || tokens > capacity_tokens) {
    return false;
  }
  const std::size_t rows = Bucket(tokens, capacity_tokens);
  Plan* plan =
      impl_->Resolve(rows, m, k, weight_bf16, input_bf16, output, stream);
  if (!plan->usable) {
    return false;
  }
  const hipblasStatus_t status =
      impl_->Launch(*plan, weight_bf16, input_bf16, output, stream);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("EmbeddingGemma 2 hipBLASLt GEMM failed: " +
                             std::to_string(static_cast<int>(status)));
  }
  return true;
}

void LaunchGemmBf16(hipblasHandle_t handle, const void* weight_bf16,
                    const void* input_bf16, float* output, std::size_t tokens,
                    std::size_t m, std::size_t k, hipStream_t stream) {
  if (handle == nullptr || weight_bf16 == nullptr || input_bf16 == nullptr ||
      output == nullptr || tokens == 0U || m == 0U || k == 0U) {
    throw std::invalid_argument("invalid EmbeddingGemma 2 GEMM inputs");
  }
  constexpr auto kLimit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (tokens > kLimit || m > kLimit || k > kLimit) {
    throw std::length_error("EmbeddingGemma 2 GEMM exceeds library limits");
  }
  if (hipblasSetStream(handle, stream) != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("EmbeddingGemma 2 hipBLAS stream setup failed");
  }
  // Row-major Y[tokens, m] is column-major Y^T[m, tokens] = W^T X^T.
  const hipblasStatus_t status = hipblasGemmEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(m),
      static_cast<int>(tokens), static_cast<int>(k), &kAlpha, weight_bf16,
      HIP_R_16BF, static_cast<int>(k), input_bf16, HIP_R_16BF,
      static_cast<int>(k), &kBeta, output, HIP_R_32F, static_cast<int>(m),
      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("EmbeddingGemma 2 hipBLAS GEMM failed: " +
                             std::to_string(static_cast<int>(status)));
  }
}

}  // namespace gufo::models::embedding_gemma2::hip

#endif  // defined(ENGINE_ENABLE_HIP)
