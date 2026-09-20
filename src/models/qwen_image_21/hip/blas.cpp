#include "src/models/qwen_image_21/hip/blas.hpp"

#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <stdexcept>
#include <tuple>

namespace gufo::models::qwen_image_21::hip {
namespace {

void Check(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Qwen-Image hipBLASLt failed: " +
                             std::to_string(status));
}

struct Plan {
  hipblasLtMatmulDesc_t op{};
  hipblasLtMatrixLayout_t weight{}, input{}, output{};
  hipblasLtMatmulPreference_t preference{};
  hipblasLtMatmulHeuristicResult_t selected{};
  bool usable{false};
  ~Plan() {
    if (preference)
      (void)hipblasLtMatmulPreferenceDestroy(preference);
    if (output)
      (void)hipblasLtMatrixLayoutDestroy(output);
    if (input)
      (void)hipblasLtMatrixLayoutDestroy(input);
    if (weight)
      (void)hipblasLtMatrixLayoutDestroy(weight);
    if (op)
      (void)hipblasLtMatmulDescDestroy(op);
  }
};

}  // namespace

struct Gemm::Impl {
  hipblasLtHandle_t handle{};
  std::map<std::tuple<int, int, int, bool, int, std::int64_t, std::int64_t,
                      std::int64_t>,
           std::unique_ptr<Plan>>
      plans;
  Impl() { Check(hipblasLtCreate(&handle)); }
  ~Impl() {
    plans.clear();
    if (handle)
      (void)hipblasLtDestroy(handle);
  }
  Plan& Resolve(int rows, int channels, int inner, bool float_output,
                int batches = 1, std::int64_t weight_stride = 0,
                std::int64_t input_stride = 0, std::int64_t output_stride = 0) {
    const auto key =
        std::make_tuple(rows, channels, inner, float_output, batches,
                        weight_stride, input_stride, output_stride);
    if (const auto found = plans.find(key); found != plans.end())
      return *found->second;
    auto plan = std::make_unique<Plan>();
    Check(hipblasLtMatmulDescCreate(&plan->op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    const hipblasOperation_t transpose = HIPBLAS_OP_T;
    Check(hipblasLtMatmulDescSetAttribute(
        plan->op, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
    Check(hipblasLtMatrixLayoutCreate(&plan->weight, HIP_R_16BF, inner,
                                      channels, inner));
    Check(hipblasLtMatrixLayoutCreate(&plan->input, HIP_R_16BF, inner, rows,
                                      inner));
    Check(hipblasLtMatrixLayoutCreate(&plan->output,
                                      float_output ? HIP_R_32F : HIP_R_16BF,
                                      channels, rows, channels));
    if (batches > 1) {
      for (auto [layout, stride] : {std::pair{plan->weight, weight_stride},
                                    {plan->input, input_stride},
                                    {plan->output, output_stride}}) {
        Check(hipblasLtMatrixLayoutSetAttribute(
            layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches,
            sizeof(batches)));
        Check(hipblasLtMatrixLayoutSetAttribute(
            layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
            sizeof(stride)));
      }
    }
    Check(hipblasLtMatmulPreferenceCreate(&plan->preference));
    const std::size_t workspace = 0;
    Check(hipblasLtMatmulPreferenceSetAttribute(
        plan->preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
        sizeof(workspace)));
    std::array<hipblasLtMatmulHeuristicResult_t, 16> candidates{};
    int count = 0;
    const auto status = hipblasLtMatmulAlgoGetHeuristic(
        handle, plan->op, plan->weight, plan->input, plan->output, plan->output,
        plan->preference, candidates.size(), candidates.data(), &count);
    if (status != HIPBLAS_STATUS_NOT_SUPPORTED)
      Check(status);
    // Measured on this model's 256/1024/4096-row DiT and 1024-row VAE shapes.
    // The default FP32-output choices are 4–6x slower on gfx1151. Keep a
    // deterministic shape policy; requalify after hipBLASLt updates.
    const int rank = float_output   ? (channels < 256 ? 5 : 4)
                     : rows <= 512  ? 12
                     : rows >= 4096 ? 0
                                    : 1;
    if (status == HIPBLAS_STATUS_SUCCESS && count > 0) {
      plan->selected = candidates[std::min(rank, count - 1)];
      plan->usable = plan->selected.state == HIPBLAS_STATUS_SUCCESS &&
                     plan->selected.workspaceSize == 0;
      if (plan->usable)
        (void)hipblaslt_ext::getSolutionNameFromAlgo(handle,
                                                     plan->selected.algo);
    }
    auto& result = *plan;
    plans.emplace(key, std::move(plan));
    return result;
  }
};

Gemm::Gemm() : impl_(std::make_unique<Impl>()) {}
Gemm::~Gemm() = default;

bool Gemm::Run(const void* weight, const void* input, void* output, int rows,
               int channels, int inner, bool float_output, hipStream_t stream) {
  // Small text/time embeddings retain the established vector-sized kernels.
  if (rows < 64)
    return false;
  auto& plan = impl_->Resolve(rows, channels, inner, float_output);
  if (!plan.usable)
    return false;
  const float alpha = 1, beta = 0;
  Check(hipblasLtMatmul(impl_->handle, plan.op, &alpha, weight, plan.weight,
                        input, plan.input, &beta, output, plan.output, output,
                        plan.output, &plan.selected.algo, nullptr, 0, stream));
  return true;
}

bool Gemm::RunBatched(const void* weight, const void* input, void* output,
                      int rows, int channels, int inner, int batches,
                      std::int64_t weight_stride, std::int64_t input_stride,
                      std::int64_t output_stride, hipStream_t stream) {
  if (rows < 64)
    return false;
  auto& plan = impl_->Resolve(rows, channels, inner, true, batches,
                              weight_stride, input_stride, output_stride);
  if (!plan.usable)
    return false;
  const float alpha = 1, beta = 0;
  Check(hipblasLtMatmul(impl_->handle, plan.op, &alpha, weight, plan.weight,
                        input, plan.input, &beta, output, plan.output, output,
                        plan.output, &plan.selected.algo, nullptr, 0, stream));
  return true;
}

}  // namespace gufo::models::qwen_image_21::hip
