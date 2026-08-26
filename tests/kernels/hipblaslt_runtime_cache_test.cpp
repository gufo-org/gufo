#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/hip/detail/hipblaslt_plan_database.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace {

std::uint16_t FloatToBf16Bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16);
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void Verify(const std::vector<float>& values, float expected) {
  for (const float value : values) {
    Expect(std::isfinite(value) && std::abs(value - expected) < 1e-2F,
           "persisted GEMM numerical output");
  }
}

}  // namespace

int main() {
  constexpr std::size_t batch_size = 32;
  constexpr std::size_t m = 256;
  constexpr std::size_t k = 64;
  constexpr std::size_t second_m = 64;
  const auto test_id =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto directory = std::filesystem::temp_directory_path() /
                         ("strix-hipblaslt-runtime-" + std::to_string(test_id));
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "plans.bin";
  const auto changed_path = directory / "changed.bin";

  std::vector<std::uint16_t> host_a(m * k, FloatToBf16Bits(1.5F));
  std::vector<std::uint16_t> host_x(batch_size * k, FloatToBf16Bits(2.0F));
  std::vector<float> host_y(batch_size * m, 0.0F);
  void* device_a = nullptr;
  void* device_x = nullptr;
  float* device_y = nullptr;
  HIP_CHECK(hipMalloc(&device_a, host_a.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&device_x, host_x.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&device_y, host_y.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_a, host_a.data(),
                      host_a.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_x, host_x.data(),
                      host_x.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));

  strix::hip::HipblasLtDispatchInfo original_info;
  strix::hip::HipblasLtGemm original({
      .plan_database_path = {},
      .tuning_workspace_bytes = 0,
      .ignore_environment = true,
  });
  Expect(original.RunBf16(device_a, device_x, device_y, batch_size, m, k,
                          nullptr, &original_info),
         "resolve original hipBLASLt plan");
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host_y.data(), device_y, host_y.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  Verify(host_y, 3.0F * static_cast<float>(k));

  strix::hip::HipblasLtDispatchInfo original_second_info;
  Expect(original.RunBf16(device_a, device_x, device_y, batch_size, second_m, k,
                          nullptr, &original_second_info),
         "resolve second hipBLASLt plan");
  Expect(original_second_info.algorithm_id != original_info.algorithm_id,
         "exercise a second hipBLASLt algorithm ID");
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host_y.data(), device_y,
                      batch_size * second_m * sizeof(float),
                      hipMemcpyDeviceToHost));
  Verify(std::vector<float>(host_y.begin(),
                            host_y.begin() + (batch_size * second_m)),
         3.0F * static_cast<float>(k));

  std::string error;
  Expect(original.SavePlans(database_path.string(), &error),
         "save resolved plan: " + error);

  strix::hip::HipblasLtDispatchInfo persisted_info;
  strix::hip::HipblasLtGemm persisted({
      .plan_database_path = database_path.string(),
      .tuning_workspace_bytes = 0,
      .ignore_environment = true,
  });
  Expect(persisted.RunBf16(device_a, device_x, device_y, batch_size, m, k,
                           nullptr, &persisted_info),
         "run reconstructed plan");
  Expect(persisted_info.plan_source == "persistent", "select persisted plan");
  Expect(persisted_info.persistent_cache_status == "hit",
         "report persistent cache hit");
  Expect(persisted_info.algorithm_id == original_info.algorithm_id,
         "preserve algorithm ID");
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host_y.data(), device_y, host_y.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  Verify(host_y, 3.0F * static_cast<float>(k));

  strix::hip::HipblasLtDispatchInfo persisted_second_info;
  Expect(persisted.RunBf16(device_a, device_x, device_y, batch_size, second_m,
                           k, nullptr, &persisted_second_info),
         "run second reconstructed plan");
  Expect(persisted_second_info.plan_source == "persistent",
         "select second persisted plan");
  Expect(persisted_second_info.persistent_cache_status == "hit",
         "report second persistent cache hit");
  Expect(
      persisted_second_info.algorithm_id == original_second_info.algorithm_id,
      "preserve second algorithm ID");
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host_y.data(), device_y,
                      batch_size * second_m * sizeof(float),
                      hipMemcpyDeviceToHost));
  Verify(std::vector<float>(host_y.begin(),
                            host_y.begin() + (batch_size * second_m)),
         3.0F * static_cast<float>(k));

  auto changed_database =
      strix::hip::detail::InspectHipblasLtPlanDatabase(database_path);
  Expect(changed_database.status ==
                 strix::hip::detail::HipblasLtPlanDatabaseLoadStatus::kLoaded &&
             changed_database.database.records.size() == 2,
         "inspect saved runtime database");
  const auto changed_record = std::ranges::find_if(
      changed_database.database.records, [](const auto& record) {
        return record.batch_size == batch_size && record.m == m &&
               record.k == k;
      });
  Expect(changed_record != changed_database.database.records.end(),
         "find primary persisted plan");
  ++changed_record->algorithm_id;
  Expect(strix::hip::detail::SaveHipblasLtPlanDatabase(
             changed_path, changed_database.database, &error),
         "save changed algorithm identity: " + error);

  strix::hip::HipblasLtDispatchInfo changed_info;
  strix::hip::HipblasLtGemm changed({
      .plan_database_path = changed_path.string(),
      .tuning_workspace_bytes = 0,
      .ignore_environment = true,
  });
  Expect(changed.RunBf16(device_a, device_x, device_y, batch_size, m, k,
                         nullptr, &changed_info),
         "fallback after algorithm identity change");
  Expect(changed_info.plan_source == "heuristic",
         "use heuristic after identity change");
  Expect(changed_info.persistent_cache_status == "algorithm_changed",
         "detect changed algorithm identity");
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host_y.data(), device_y, host_y.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  Verify(host_y, 3.0F * static_cast<float>(k));

  HIP_CHECK(hipFree(device_y));
  HIP_CHECK(hipFree(device_x));
  HIP_CHECK(hipFree(device_a));
  std::filesystem::remove_all(directory);
  std::cout << "PASS: hipBLASLt persisted runtime plan\n";
  return 0;
}
