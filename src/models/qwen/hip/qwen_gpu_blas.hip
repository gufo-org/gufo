#if defined(ENGINE_ENABLE_HIP)
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <sstream>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/detail/hipblaslt_plan_database.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"

namespace strix::hip {

namespace {

constexpr std::size_t kHipblasLtMaxAlgorithms = 16;
constexpr std::size_t kHipblasLtWideFfnThreshold = 3;
constexpr std::size_t kHipblasLtLargeBatch = 256;
constexpr std::size_t kHipblasLtLargeWideRank = 4;
constexpr std::size_t kHipblasLtTallRank = 5;
constexpr std::size_t kMaxPersistedWorkspaceBytes = 256ULL * 1024 * 1024;
constexpr double kTuningPromotionRatio = 0.98;

struct HipblasLtPlan {
  std::size_t batch_size{0};
  std::size_t m{0};
  std::size_t k{0};
  hipblasLtMatmulDesc_t operation{nullptr};
  hipblasLtMatrixLayout_t a_layout{nullptr};
  hipblasLtMatrixLayout_t x_layout{nullptr};
  hipblasLtMatrixLayout_t y_layout{nullptr};
  hipblasLtMatmulAlgo_t algorithm{};
  std::size_t workspace_bytes{0};
  std::uint64_t median_nanoseconds{0};
  std::string solution_name;
  std::string kernel_name;
  std::string source{"heuristic"};
  std::string persistent_cache_status{"disabled"};
  double resolution_us{0.0};

  HipblasLtPlan() = default;
  HipblasLtPlan(const HipblasLtPlan&) = delete;
  HipblasLtPlan& operator=(const HipblasLtPlan&) = delete;

  ~HipblasLtPlan() {
    if (a_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(a_layout);
    }
    if (x_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(x_layout);
    }
    if (y_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(y_layout);
    }
    if (operation != nullptr) {
      (void)hipblasLtMatmulDescDestroy(operation);
    }
  }
};

[[nodiscard]] std::size_t SelectHipblasLtHeuristicRank(
    std::size_t batch_size, std::size_t m, std::size_t k,
    std::size_t result_count) noexcept {
  if (batch_size >= kHipblasLtLargeBatch) {
    if (m <= 1024) {
      return kHipblasLtLargeWideRank;
    }
    return k >= kHipblasLtWideFfnThreshold * m ? kHipblasLtTallRank
                                               : kHipblasLtLargeWideRank;
  }
  const std::size_t tall_m_threshold = batch_size >= 128 ? k : 2 * k;
  if (m > tall_m_threshold || k >= kHipblasLtWideFfnThreshold * m) {
    return kHipblasLtTallRank;
  }
  return result_count - 1;
}

[[nodiscard]] bool IsUsable(
    const hipblasLtMatmulHeuristicResult_t& result,
    std::size_t workspace_capacity) noexcept {
  return result.state == HIPBLAS_STATUS_SUCCESS &&
         result.workspaceSize <= workspace_capacity;
}

[[nodiscard]] std::unique_ptr<HipblasLtPlan> CreateHipblasLtDescriptorPlan(
    std::size_t batch_size, std::size_t m, std::size_t k) {
  auto plan = std::make_unique<HipblasLtPlan>();
  plan->batch_size = batch_size;
  plan->m = m;
  plan->k = k;

  if (hipblasLtMatmulDescCreate(&plan->operation, HIPBLAS_COMPUTE_32F,
                                HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  const hipblasOperation_t transpose = HIPBLAS_OP_T;
  if (hipblasLtMatmulDescSetAttribute(
          plan->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
          sizeof(transpose)) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  if (hipblasLtMatrixLayoutCreate(&plan->a_layout, HIP_R_16BF, k, m, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan->x_layout, HIP_R_16BF, k, batch_size,
                                  k) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan->y_layout, HIP_R_32F, m, batch_size,
                                  m) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  return plan;
}

[[nodiscard]] detail::HipblasLtPlanDatabaseKey CurrentPlanDatabaseKey(
    hipblasLtHandle_t handle) {
  hipDeviceProp_t properties{};
  int device = 0;
  int hip_runtime_version = 0;
  int hipblaslt_version = 0;
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess ||
      hipRuntimeGetVersion(&hip_runtime_version) != hipSuccess ||
      hipblasLtGetVersion(handle, &hipblaslt_version) !=
          HIPBLAS_STATUS_SUCCESS) {
    return {};
  }

  std::ostringstream fingerprint;
  fingerprint << properties.gcnArchName << '|' << properties.name
              << "|cu=" << properties.multiProcessorCount
              << "|wave=" << properties.warpSize
              << "|asic=" << properties.asicRevision;
  return {
      .hardware_fingerprint = fingerprint.str(),
      .hip_runtime_version =
          static_cast<std::uint32_t>(hip_runtime_version),
      .hipblaslt_version = static_cast<std::uint32_t>(hipblaslt_version),
  };
}

[[nodiscard]] std::string DatabaseLoadStatusName(
    detail::HipblasLtPlanDatabaseLoadStatus status) {
  switch (status) {
    case detail::HipblasLtPlanDatabaseLoadStatus::kLoaded:
      return "loaded";
    case detail::HipblasLtPlanDatabaseLoadStatus::kNotFound:
      return "not_found";
    case detail::HipblasLtPlanDatabaseLoadStatus::kInvalid:
      return "invalid";
    case detail::HipblasLtPlanDatabaseLoadStatus::kIncompatible:
      return "incompatible";
  }
  return "invalid";
}

[[nodiscard]] bool RestorePersistedAlgorithm(
    hipblasLtHandle_t handle, HipblasLtPlan& plan,
    const detail::HipblasLtPlanRecord& record,
    std::size_t workspace_capacity, bool* solution_library_ready) {
  static_assert(std::is_trivially_copyable_v<hipblasLtMatmulAlgo_t>);
  if (record.algorithm_blob.size() != sizeof(hipblasLtMatmulAlgo_t)) {
    plan.persistent_cache_status = "invalid_descriptor";
    return false;
  }

  hipblasLtMatmulAlgo_t algorithm{};
  if (*solution_library_ready) {
    std::memcpy(&algorithm, record.algorithm_blob.data(), sizeof(algorithm));
  } else {
    std::vector<int> indices{record.algorithm_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> results;
    if (hipblaslt_ext::getAlgosFromIndex(handle, indices, results) !=
            HIPBLAS_STATUS_SUCCESS ||
        results.empty()) {
      plan.persistent_cache_status = "algorithm_unavailable";
      return false;
    }
    algorithm = results.front().algo;
    *solution_library_ready = true;
  }
  if (hipblaslt_ext::getIndexFromAlgo(algorithm) != record.algorithm_id) {
    plan.persistent_cache_status = "algorithm_changed";
    return false;
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  std::size_t workspace_bytes = 0;
  if (hipblaslt_ext::matmulIsAlgoSupported(
          handle, plan.operation, &alpha, plan.a_layout, plan.x_layout, &beta,
          plan.y_layout, plan.y_layout, algorithm,
          workspace_bytes) != HIPBLAS_STATUS_SUCCESS) {
    plan.persistent_cache_status = "unsupported";
    return false;
  }
  if (workspace_bytes > workspace_capacity ||
      workspace_bytes != record.workspace_bytes) {
    plan.persistent_cache_status = "workspace_mismatch";
    return false;
  }

  const std::string solution_name =
      hipblaslt_ext::getSolutionNameFromAlgo(handle, algorithm);
  const std::string kernel_name =
      hipblaslt_ext::getKernelNameFromAlgo(handle, algorithm);
  if ((!record.solution_name.empty() &&
       solution_name != record.solution_name) ||
      (!record.kernel_name.empty() && kernel_name != record.kernel_name)) {
    plan.persistent_cache_status = "algorithm_changed";
    return false;
  }

  plan.algorithm = algorithm;
  plan.workspace_bytes = workspace_bytes;
  plan.median_nanoseconds = record.median_nanoseconds;
  plan.solution_name = solution_name;
  plan.kernel_name = kernel_name;
  plan.source = "persistent";
  plan.persistent_cache_status = "hit";
  return true;
}

[[nodiscard]] bool SelectHeuristicAlgorithm(hipblasLtHandle_t handle,
                                            HipblasLtPlan& plan,
                                            std::size_t workspace_capacity) {

  hipblasLtMatmulPreference_t preference = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&preference) != HIPBLAS_STATUS_SUCCESS) {
    return false;
  }
  const auto preference_status = hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_capacity, sizeof(workspace_capacity));

  std::vector<hipblasLtMatmulHeuristicResult_t> results(
      kHipblasLtMaxAlgorithms);
  int result_count = 0;
  const auto heuristic_status =
      preference_status == HIPBLAS_STATUS_SUCCESS
          ? hipblasLtMatmulAlgoGetHeuristic(
                handle, plan.operation, plan.a_layout, plan.x_layout,
                plan.y_layout, plan.y_layout, preference,
                static_cast<int>(results.size()), results.data(), &result_count)
          : preference_status;
  (void)hipblasLtMatmulPreferenceDestroy(preference);

  if (heuristic_status != HIPBLAS_STATUS_SUCCESS || result_count <= 0) {
    return false;
  }

  const auto result_span =
      std::span(results.data(), static_cast<std::size_t>(result_count));
  const auto preferred_rank =
      SelectHipblasLtHeuristicRank(plan.batch_size, plan.m, plan.k,
                                   result_span.size());
  if (preferred_rank < result_span.size() &&
      IsUsable(result_span[preferred_rank], workspace_capacity)) {
    plan.algorithm = result_span[preferred_rank].algo;
    plan.workspace_bytes = result_span[preferred_rank].workspaceSize;
    return true;
  }

  const auto fallback =
      std::ranges::find_if(result_span, [&](const auto& result) {
        return IsUsable(result, workspace_capacity);
      });
  if (fallback == result_span.end()) {
    return false;
  }
  plan.algorithm = fallback->algo;
  plan.workspace_bytes = fallback->workspaceSize;
  return true;
}

[[nodiscard]] std::unique_ptr<HipblasLtPlan> CreateHipblasLtPlan(
    hipblasLtHandle_t handle, std::size_t batch_size, std::size_t m,
    std::size_t k, const detail::HipblasLtPlanDatabase* database,
    std::string_view database_status, std::size_t workspace_capacity,
    bool* solution_library_ready) {
  const auto start = std::chrono::steady_clock::now();
  auto plan = CreateHipblasLtDescriptorPlan(batch_size, m, k);
  if (plan == nullptr) {
    return nullptr;
  }
  plan->persistent_cache_status = std::string(database_status);

  if (database != nullptr) {
    const auto* record = detail::FindHipblasLtPlanRecord(
        *database, batch_size, m, k,
        detail::HipblasLtPlanDataType::kBfloat16);
    if (record != nullptr &&
        RestorePersistedAlgorithm(handle, *plan, *record,
                                  workspace_capacity,
                                  solution_library_ready)) {
      plan->resolution_us =
          std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - start)
              .count();
      return plan;
    }
    if (record == nullptr) {
      plan->persistent_cache_status = "miss";
    }
  }

  if (!SelectHeuristicAlgorithm(handle, *plan, workspace_capacity)) {
    return nullptr;
  }
  *solution_library_ready = true;
  plan->solution_name =
      hipblaslt_ext::getSolutionNameFromAlgo(handle, plan->algorithm);
  plan->kernel_name =
      hipblaslt_ext::getKernelNameFromAlgo(handle, plan->algorithm);
  plan->resolution_us =
      std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - start)
          .count();
  return plan;
}

}  // namespace

struct HipblasLtGemm::Impl {
  struct PlanLookup {
    HipblasLtPlan* plan{nullptr};
    bool cache_hit{false};
  };

  hipblasLtHandle_t handle{nullptr};
  void* workspace{nullptr};
  std::size_t workspace_capacity{0};
  detail::HipblasLtPlanDatabaseKey database_key;
  detail::HipblasLtPlanDatabase database;
  bool database_loaded{false};
  bool solution_library_ready{false};
  std::string database_status{"disabled"};
  std::vector<std::unique_ptr<HipblasLtPlan>> plans;

  explicit Impl(const HipblasLtGemmOptions& options) {
    if (hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
      handle = nullptr;
      return;
    }
    database_key = CurrentPlanDatabaseKey(handle);

    std::string database_path = options.plan_database_path;
    if (database_path.empty() && !options.ignore_environment) {
      if (const char* value = std::getenv("STRIX_HIPBLASLT_PLAN_CACHE");
          value != nullptr) {
        database_path = value;
      }
    }
    std::size_t requested_workspace = options.tuning_workspace_bytes;
    if (!database_path.empty() && !database_key.hardware_fingerprint.empty()) {
      auto loaded =
          detail::LoadHipblasLtPlanDatabase(database_path, database_key);
      database_status = DatabaseLoadStatusName(loaded.status);
      if (loaded.status ==
          detail::HipblasLtPlanDatabaseLoadStatus::kLoaded) {
        database = std::move(loaded.database);
        database_loaded = true;
        for (const auto& record : database.records) {
          requested_workspace =
              std::max<std::size_t>(requested_workspace,
                                    record.workspace_bytes);
        }
      }
    }
    requested_workspace =
        std::min(requested_workspace, kMaxPersistedWorkspaceBytes);
    if (requested_workspace > 0 &&
        hipMalloc(&workspace, requested_workspace) == hipSuccess) {
      workspace_capacity = requested_workspace;
    }
  }

  ~Impl() {
    plans.clear();
    if (workspace != nullptr) {
      (void)hipFree(workspace);
    }
    if (handle != nullptr) {
      (void)hipblasLtDestroy(handle);
    }
  }

  [[nodiscard]] PlanLookup GetPlan(std::size_t batch_size, std::size_t m,
                                   std::size_t k) {
    const auto existing = std::ranges::find_if(plans, [&](const auto& plan) {
      return plan->batch_size == batch_size && plan->m == m && plan->k == k;
    });
    if (existing != plans.end()) {
      return {.plan = existing->get(), .cache_hit = true};
    }
    auto plan = CreateHipblasLtPlan(
        handle, batch_size, m, k, database_loaded ? &database : nullptr,
        database_status, workspace_capacity, &solution_library_ready);
    if (plan == nullptr) {
      return {};
    }
    auto* result = plan.get();
    plans.push_back(std::move(plan));
    return {.plan = result, .cache_hit = false};
  }
};

HipblasLtGemm::HipblasLtGemm(HipblasLtGemmOptions options)
    : impl_(std::make_unique<Impl>(options)) {}

HipblasLtGemm::~HipblasLtGemm() = default;

HipblasLtGemm::HipblasLtGemm(HipblasLtGemm&&) noexcept = default;

HipblasLtGemm& HipblasLtGemm::operator=(HipblasLtGemm&&) noexcept = default;

bool HipblasLtGemm::RunBf16(const void* a_bf16, const void* x_bf16, float* y,
                            std::size_t batch_size, std::size_t m,
                            std::size_t k, hipStream_t stream,
                            HipblasLtDispatchInfo* dispatch_info) {
  if (impl_ == nullptr || impl_->handle == nullptr || batch_size == 0 ||
      m == 0 || k == 0) {
    return false;
  }
  const auto lookup = impl_->GetPlan(batch_size, m, k);
  if (lookup.plan == nullptr) {
    return false;
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const bool succeeded =
      hipblasLtMatmul(impl_->handle, lookup.plan->operation, &alpha, a_bf16,
                      lookup.plan->a_layout, x_bf16, lookup.plan->x_layout,
                      &beta, y, lookup.plan->y_layout, y, lookup.plan->y_layout,
                      &lookup.plan->algorithm, impl_->workspace,
                      lookup.plan->workspace_bytes,
                      stream) == HIPBLAS_STATUS_SUCCESS;
  if (!succeeded) {
    return false;
  }

  if (dispatch_info != nullptr || detail::DispatchTelemetryEnabled()) {
    const int algorithm_id =
        hipblaslt_ext::getIndexFromAlgo(lookup.plan->algorithm);
    if (dispatch_info != nullptr) {
      *dispatch_info = {
          .algorithm_id = algorithm_id,
          .solution_name = lookup.plan->solution_name,
          .kernel_name = lookup.plan->kernel_name,
          .plan_source = lookup.plan->source,
          .persistent_cache_status =
              lookup.plan->persistent_cache_status,
          .workspace_bytes = lookup.plan->workspace_bytes,
          .plan_resolution_us = lookup.plan->resolution_us,
          .plan_cache_hit = lookup.cache_hit,
      };
    }
    detail::EmitHipblasLtDispatch(
        batch_size, m, k, algorithm_id, lookup.plan->solution_name,
        lookup.plan->kernel_name, lookup.plan->source,
        lookup.plan->persistent_cache_status, lookup.plan->workspace_bytes,
        lookup.cache_hit);
  }
  return true;
}

bool HipblasLtGemm::TuneBf16(
    const void* a_bf16, const void* x_bf16, float* y,
    std::size_t batch_size, std::size_t m, std::size_t k,
    const HipblasLtTuningOptions& options, HipblasLtTuningResult* result,
    hipStream_t stream) {
  if (impl_ == nullptr || impl_->handle == nullptr || a_bf16 == nullptr ||
      x_bf16 == nullptr || y == nullptr || batch_size == 0 || m == 0 ||
      k == 0 || options.repetitions == 0 || options.max_algorithms == 0) {
    return false;
  }

  auto plan = CreateHipblasLtDescriptorPlan(batch_size, m, k);
  if (plan == nullptr) {
    return false;
  }
  std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
  if (hipblaslt_ext::getAllAlgos(
          impl_->handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
          HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16BF, HIP_R_16BF, HIP_R_32F,
          HIP_R_32F, HIPBLAS_COMPUTE_32F,
          algorithms) != HIPBLAS_STATUS_SUCCESS) {
    return false;
  }
  impl_->solution_library_ready = true;

  hipEvent_t start_event = nullptr;
  hipEvent_t stop_event = nullptr;
  if (hipEventCreate(&start_event) != hipSuccess ||
      hipEventCreate(&stop_event) != hipSuccess) {
    if (start_event != nullptr) {
      (void)hipEventDestroy(start_event);
    }
    return false;
  }

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const auto measure_once =
      [&](hipblasLtMatmulAlgo_t& algorithm,
          std::size_t workspace_bytes) -> std::optional<double> {
    if (hipEventRecord(start_event, stream) != hipSuccess ||
        hipblasLtMatmul(
            impl_->handle, plan->operation, &alpha, a_bf16, plan->a_layout,
            x_bf16, plan->x_layout, &beta, y, plan->y_layout, y,
            plan->y_layout, &algorithm, impl_->workspace, workspace_bytes,
            stream) != HIPBLAS_STATUS_SUCCESS ||
        hipEventRecord(stop_event, stream) != hipSuccess ||
        hipEventSynchronize(stop_event) != hipSuccess) {
      return std::nullopt;
    }
    float elapsed_ms = 0.0F;
    if (hipEventElapsedTime(&elapsed_ms, start_event, stop_event) !=
        hipSuccess) {
      return std::nullopt;
    }
    return static_cast<double>(elapsed_ms) * 1000.0;
  };
  const auto warm_up = [&](hipblasLtMatmulAlgo_t& algorithm,
                           std::size_t workspace_bytes,
                           std::uint32_t warmups) {
    for (std::uint32_t warmup = 0; warmup < warmups; ++warmup) {
      if (hipblasLtMatmul(
              impl_->handle, plan->operation, &alpha, a_bf16, plan->a_layout,
              x_bf16, plan->x_layout, &beta, y, plan->y_layout, y,
              plan->y_layout, &algorithm, impl_->workspace, workspace_bytes,
              stream) != HIPBLAS_STATUS_SUCCESS) {
        return false;
      }
    }
    return hipStreamSynchronize(stream) == hipSuccess;
  };
  const auto median = [](std::vector<double> samples) {
    std::ranges::sort(samples);
    return samples[samples.size() / 2];
  };
  const auto measure = [&](hipblasLtMatmulAlgo_t& algorithm,
                           std::size_t workspace_bytes,
                           std::uint32_t repetitions) -> std::optional<double> {
    if (!warm_up(algorithm, workspace_bytes, options.warmup)) {
      return std::nullopt;
    }
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (std::uint32_t repetition = 0; repetition < repetitions;
         ++repetition) {
      const auto sample = measure_once(algorithm, workspace_bytes);
      if (!sample.has_value()) {
        return std::nullopt;
      }
      samples.push_back(*sample);
    }
    return median(std::move(samples));
  };

  std::size_t supported_algorithms = 0;
  std::size_t measured_algorithms = 0;
  double best_median_us = std::numeric_limits<double>::infinity();
  std::size_t best_workspace_bytes = 0;
  hipblasLtMatmulAlgo_t best_algorithm{};
  bool found = false;

  for (auto& candidate : algorithms) {
    auto algorithm = candidate.algo;
    std::size_t workspace_bytes = 0;
    if (hipblaslt_ext::matmulIsAlgoSupported(
            impl_->handle, plan->operation, &alpha, plan->a_layout,
            plan->x_layout, &beta, plan->y_layout, plan->y_layout, algorithm,
            workspace_bytes) != HIPBLAS_STATUS_SUCCESS ||
        workspace_bytes > impl_->workspace_capacity) {
      continue;
    }
    ++supported_algorithms;
    if (measured_algorithms >= options.max_algorithms) {
      continue;
    }

    const auto candidate_median =
        measure(algorithm, workspace_bytes, options.repetitions);
    if (!candidate_median.has_value()) {
      continue;
    }
    ++measured_algorithms;
    if (*candidate_median < best_median_us) {
      best_median_us = *candidate_median;
      best_workspace_bytes = workspace_bytes;
      best_algorithm = algorithm;
      found = true;
    }
  }
  if (!found) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }

  int heuristic_algorithm_id = -1;
  double heuristic_median_us = 0.0;
  bool retained_tuned_algorithm = false;
  auto heuristic_plan = CreateHipblasLtDescriptorPlan(batch_size, m, k);
  if (heuristic_plan != nullptr &&
      SelectHeuristicAlgorithm(impl_->handle, *heuristic_plan, 0)) {
    heuristic_algorithm_id =
        hipblaslt_ext::getIndexFromAlgo(heuristic_plan->algorithm);
    const int best_algorithm_id =
        hipblaslt_ext::getIndexFromAlgo(best_algorithm);
    if (best_algorithm_id != heuristic_algorithm_id) {
      const std::uint32_t verification_repetitions =
          std::max<std::uint32_t>(options.repetitions, 7);
      bool verified =
          warm_up(best_algorithm, best_workspace_bytes, options.warmup) &&
          warm_up(heuristic_plan->algorithm,
                  heuristic_plan->workspace_bytes, options.warmup);
      std::vector<double> best_samples;
      std::vector<double> heuristic_samples;
      best_samples.reserve(verification_repetitions);
      heuristic_samples.reserve(verification_repetitions);
      for (std::uint32_t repetition = 0;
           verified && repetition < verification_repetitions; ++repetition) {
        const bool best_first = (repetition % 2) == 0;
        const auto first = best_first
                               ? measure_once(best_algorithm,
                                              best_workspace_bytes)
                               : measure_once(heuristic_plan->algorithm,
                                              heuristic_plan->workspace_bytes);
        const auto second = best_first
                                ? measure_once(heuristic_plan->algorithm,
                                               heuristic_plan->workspace_bytes)
                                : measure_once(best_algorithm,
                                               best_workspace_bytes);
        verified = first.has_value() && second.has_value();
        if (verified) {
          best_samples.push_back(best_first ? *first : *second);
          heuristic_samples.push_back(best_first ? *second : *first);
        }
      }
      if (verified) {
        best_median_us = median(std::move(best_samples));
        heuristic_median_us = median(std::move(heuristic_samples));
        retained_tuned_algorithm =
            best_median_us < heuristic_median_us * kTuningPromotionRatio;
      } else {
        const auto fallback_median =
            measure(heuristic_plan->algorithm,
                    heuristic_plan->workspace_bytes, options.repetitions);
        if (fallback_median.has_value()) {
          heuristic_median_us = *fallback_median;
        }
      }
      if (!retained_tuned_algorithm) {
        best_algorithm = heuristic_plan->algorithm;
        best_workspace_bytes = heuristic_plan->workspace_bytes;
        if (heuristic_median_us > 0.0) {
          best_median_us = heuristic_median_us;
        }
      }
    } else {
      heuristic_median_us = best_median_us;
    }
  }
  (void)hipEventDestroy(stop_event);
  (void)hipEventDestroy(start_event);

  plan->algorithm = best_algorithm;
  plan->workspace_bytes = best_workspace_bytes;
  plan->median_nanoseconds =
      static_cast<std::uint64_t>(std::llround(best_median_us * 1000.0));
  plan->solution_name =
      hipblaslt_ext::getSolutionNameFromAlgo(impl_->handle, plan->algorithm);
  plan->kernel_name =
      hipblaslt_ext::getKernelNameFromAlgo(impl_->handle, plan->algorithm);
  plan->source = "tuned";
  plan->persistent_cache_status = "tuned";

  const auto existing =
      std::ranges::find_if(impl_->plans, [&](const auto& candidate) {
        return candidate->batch_size == batch_size && candidate->m == m &&
               candidate->k == k;
      });
  HipblasLtPlan* installed = plan.get();
  if (existing == impl_->plans.end()) {
    impl_->plans.push_back(std::move(plan));
  } else {
    *existing = std::move(plan);
    installed = existing->get();
  }

  if (result != nullptr) {
    *result = {
        .algorithm_id =
            hipblaslt_ext::getIndexFromAlgo(installed->algorithm),
        .heuristic_algorithm_id = heuristic_algorithm_id,
        .solution_name = installed->solution_name,
        .kernel_name = installed->kernel_name,
        .workspace_bytes = installed->workspace_bytes,
        .median_us = best_median_us,
        .heuristic_median_us = heuristic_median_us,
        .verified_speedup =
            heuristic_median_us > 0.0
                ? heuristic_median_us / best_median_us
                : 1.0,
        .supported_algorithms = supported_algorithms,
        .measured_algorithms = measured_algorithms,
        .retained_tuned_algorithm = retained_tuned_algorithm,
    };
  }
  return true;
}

bool HipblasLtGemm::SavePlans(const std::string& path,
                              std::string* error) const {
  if (impl_ == nullptr || impl_->database_key.hardware_fingerprint.empty()) {
    if (error != nullptr) {
      *error = "hipBLASLt hardware fingerprint is unavailable";
    }
    return false;
  }

  detail::HipblasLtPlanDatabase database{
      .key = impl_->database_key,
      .records = {},
  };
  database.records.reserve(impl_->plans.size());
  for (const auto& plan : impl_->plans) {
    const int algorithm_id =
        hipblaslt_ext::getIndexFromAlgo(plan->algorithm);
    if (algorithm_id < 0) {
      continue;
    }
    database.records.push_back({
        .batch_size = plan->batch_size,
        .m = plan->m,
        .k = plan->k,
        .data_type = detail::HipblasLtPlanDataType::kBfloat16,
        .algorithm_id = algorithm_id,
        .algorithm_blob =
            std::vector<std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(&plan->algorithm),
                reinterpret_cast<const std::uint8_t*>(&plan->algorithm) +
                    sizeof(plan->algorithm)),
        .workspace_bytes = plan->workspace_bytes,
        .median_nanoseconds = plan->median_nanoseconds,
        .solution_name = plan->solution_name,
        .kernel_name = plan->kernel_name,
    });
  }
  std::ranges::sort(database.records, [](const auto& left, const auto& right) {
    return std::tie(left.batch_size, left.m, left.k) <
           std::tie(right.batch_size, right.m, right.k);
  });
  return detail::SaveHipblasLtPlanDatabase(path, database, error);
}

void LaunchHipblasGEMM(hipblasHandle_t handle, const void* A, bool is_bf16,
                       const float* X, float* Y, std::size_t batch_size,
                       std::size_t M, std::size_t K, void* d_x_bf16_buf,
                       hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }

  if (stream != nullptr) {
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;

  if (is_bf16) {
    LaunchFloatToBfloat16(X, d_x_bf16_buf, batch_size * K, stream);
    HIPBLAS_CHECK(hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
        static_cast<int>(batch_size), static_cast<int>(K), &alpha, A,
        HIP_R_16BF, static_cast<int>(K), d_x_bf16_buf, HIP_R_16BF,
        static_cast<int>(K), &beta, Y, HIP_R_32F, static_cast<int>(M),
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  } else {
    HIPBLAS_CHECK(hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
        static_cast<int>(batch_size), static_cast<int>(K), &alpha, A, HIP_R_32F,
        static_cast<int>(K), X, HIP_R_32F, static_cast<int>(K), &beta, Y,
        HIP_R_32F, static_cast<int>(M), HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT));
  }
}

void LaunchHipblasGEMMBF16(hipblasHandle_t handle, const void* A_bf16,
                           const void* d_x_bf16, float* Y,
                           std::size_t batch_size, std::size_t M, std::size_t K,
                           hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }

  if (stream != nullptr) {
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;

  HIPBLAS_CHECK(hipblasGemmEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
      static_cast<int>(batch_size), static_cast<int>(K), &alpha, A_bf16,
      HIP_R_16BF, static_cast<int>(K), d_x_bf16, HIP_R_16BF,
      static_cast<int>(K), &beta, Y, HIP_R_32F, static_cast<int>(M),
      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
