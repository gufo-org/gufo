#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"

#include <array>
#include <hipblaslt/hipblaslt-ext.hpp>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

void AssignError(std::string* error, const char* message) {
  if (error != nullptr) {
    *error = message;
  }
}

}  // namespace

struct BlasLt::Plan {
  hipblasLtMatmulDesc_t operation{nullptr};
  hipblasLtMatrixLayout_t weights{nullptr};
  hipblasLtMatrixLayout_t input{nullptr};
  hipblasLtMatrixLayout_t output{nullptr};
  hipblasLtMatmulAlgo_t algorithm{};

  ~Plan() {
    if (output != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(output);
    if (input != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(input);
    if (weights != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(weights);
    if (operation != nullptr)
      (void)hipblasLtMatmulDescDestroy(operation);
  }
};

BlasLt::~BlasLt() {
  plans_.clear();
  if (handle_ != nullptr) {
    (void)hipblasLtDestroy(handle_);
  }
}

std::unique_ptr<BlasLt> BlasLt::Create(hipStream_t stream, std::string* error) {
  std::unique_ptr<BlasLt> blas(new BlasLt());
  blas->stream_ = stream;
  if (hipblasLtCreate(&blas->handle_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error, "hipblasLtCreate failed");
    return nullptr;
  }
  return blas;
}

std::unique_ptr<BlasLt::Plan> BlasLt::MakePlan(hipDataType type, int m, int n,
                                               int k,
                                               std::string* error) const {
  auto p = std::make_unique<Plan>();
  const hipblasOperation_t transpose = HIPBLAS_OP_T;
  const hipblasOperation_t normal = HIPBLAS_OP_N;
  if (hipblasLtMatmulDescCreate(&p->operation, HIPBLAS_COMPUTE_32F,
                                HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(
          p->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
          sizeof(transpose)) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(
          p->operation, HIPBLASLT_MATMUL_DESC_TRANSB, &normal,
          sizeof(normal)) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&p->weights, type, k, m, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&p->input, type, k, n, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&p->output, HIP_R_32F, m, n, m) !=
          HIPBLAS_STATUS_SUCCESS) {
    AssignError(error, "hipBLASLt descriptor creation failed");
    return nullptr;
  }

  const float one = 1.0F;
  const float zero = 0.0F;
  const auto usable = [&](hipblasLtMatmulAlgo_t& algorithm) {
    std::size_t workspace = 0;
    return hipblaslt_ext::matmulIsAlgoSupported(
               handle_, p->operation, &one, p->weights, p->input, &zero,
               p->output, p->output, algorithm,
               workspace) == HIPBLAS_STATUS_SUCCESS &&
           workspace == 0;
  };

  // Initialize the solution library and keep a deterministic fallback for
  // other tensor geometries. No trial GEMMs run on model activations.
  hipblasLtMatmulPreference_t preference = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&preference) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error, "hipBLASLt preference creation failed");
    return nullptr;
  }
  constexpr std::size_t workspace = 0;
  (void)hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
      sizeof(workspace));
  std::array<hipblasLtMatmulHeuristicResult_t, 16> candidates{};
  int count = 0;
  const auto status = hipblasLtMatmulAlgoGetHeuristic(
      handle_, p->operation, p->weights, p->input, p->output, p->output,
      preference, candidates.size(), candidates.data(), &count);
  (void)hipblasLtMatmulPreferenceDestroy(preference);

  if (status == HIPBLAS_STATUS_SUCCESS) {
    for (int i = 0; i < count; ++i) {
      if (candidates[i].state == HIPBLAS_STATUS_SUCCESS &&
          candidates[i].workspaceSize == 0 && usable(candidates[i].algo)) {
        p->algorithm = candidates[i].algo;
        return p;
      }
    }
  }
  AssignError(error, "no workspace-free hipBLASLt kernel for the GEMM shape");
  return nullptr;
}

bool BlasLt::Gemm(const void* weights, const void* input, float* out,
                  hipDataType type, int m, int n, int k, std::string* error) {
  if (m <= 0 || n <= 0 || k <= 0) {
    AssignError(error, "hipBLASLt dimensions must be positive");
    return false;
  }
  // Library edge tiles change the accumulation order when the same token
  // moves within a batch. Router and recurrent-gate errors then amplify across
  // layers. Keep one K reduction for every row and prefill chunk size: the
  // router, the recurrent gates, and a TP2 rank's half of the gates.
  if (type == HIP_R_16F && k == 2560 && (m == 96 || m == 48 || m == 513)) {
    return UnquantizedF16Gemm(weights, static_cast<const __half*>(input), out,
                              n, m, k, stream_);
  }
  // Exact dimensions prevent the first ragged request from determining
  // which kernel later requests in the same size bucket receive.
  const std::array<int, 4> key{static_cast<int>(type), m, n, k};
  auto& p = plans_[key];
  if (!p) {
    p = MakePlan(type, m, n, k, error);
    if (!p) {
      return false;
    }
  }
  const float one = 1.0F;
  const float zero = 0.0F;
  if (hipblasLtMatmul(handle_, p->operation, &one, weights, p->weights, input,
                      p->input, &zero, out, p->output, out, p->output,
                      &p->algorithm, nullptr, 0,
                      stream_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error, "hipBLASLt GEMM failed");
    return false;
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
