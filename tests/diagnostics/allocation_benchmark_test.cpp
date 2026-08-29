#include "src/core/diagnostics/allocation_benchmark.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/diagnostics/artifact_validator.h"
#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/system_inventory.h"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

std::filesystem::path FindFixturesRoot() {
#ifdef TEST_FIXTURES_DIR
  if (std::filesystem::exists(TEST_FIXTURES_DIR)) {
    return TEST_FIXTURES_DIR;
  }
#endif
  const std::array<std::filesystem::path, 3> paths = {
      "tests/fixtures/diagnostics",
      "../tests/fixtures/diagnostics",
      "../../tests/fixtures/diagnostics",
  };
  for (const auto& path : paths) {
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  return "tests/fixtures/diagnostics";
}

void TestSizeParser() {
  std::vector<std::size_t> sizes;
  std::string error;
  Expect(
      gufo::diagnostics::ParseWorkingSetSizes("64,1024,52126", &sizes, &error),
      error);
  Expect(sizes.size() == 3, "three working-set sizes parsed");
  Expect(sizes[0] == 64ULL * 1024ULL * 1024ULL, "64 MiB parsed");
  Expect(sizes[1] == 1024ULL * 1024ULL * 1024ULL, "1 GiB parsed");
  Expect(!gufo::diagnostics::ParseWorkingSetSizes("64,invalid", &sizes, &error),
         "invalid size rejected");
  Expect(!gufo::diagnostics::ParseWorkingSetSizes("64,", &sizes, &error),
         "trailing separator rejected");
}

void TestStatistics() {
  gufo::diagnostics::AllocationPhaseResult phase;
  phase.raw_repetitions = {4.0, 1.0, 3.0, 2.0, 5.0};
  gufo::diagnostics::ComputeAllocationPhaseStatistics(&phase);
  Expect(phase.minimum == 1.0, "minimum computed");
  Expect(phase.median == 3.0, "median computed");
  Expect(phase.p95 == 5.0, "p95 computed");
  Expect(phase.maximum == 5.0, "maximum computed");
}

void TestBoundedDiagnostic() {
  const auto inventory = gufo::diagnostics::CollectSystemInventory();
  const auto fingerprint =
      gufo::diagnostics::GenerateMachineFingerprint(inventory);
  gufo::diagnostics::AllocationBenchmarkOptions options;
  options.working_set_bytes = {4ULL * 1024ULL * 1024ULL};
  options.warmup = 1;
  options.repetitions = 2;
  options.transfer_chunk_bytes = 1ULL * 1024ULL * 1024ULL;

  const auto report = gufo::diagnostics::RunHipAllocationBenchmark(
      options, inventory, fingerprint);
  Expect(report.paths.size() == 5, "all five allocation paths reported");
  Expect(report.all_requested_paths_reported,
         "requested path coverage is explicit");
  Expect(report.checksums_verified,
         "completed paths verify deterministic checksums");
  Expect(report.phases_separated, "allocation phases remain separate");
  for (const auto& path : report.paths) {
    Expect(path.status == "completed" || path.status == "unsupported" ||
               path.status == "unavailable",
           "path has a bounded terminal status");
    if (path.status == "completed") {
      Expect(path.checksum_verified, "completed path checksum verified");
      Expect(!path.phases.empty(), "completed path has phase measurements");
      const auto allocation = std::ranges::find_if(
          path.phases,
          [](const auto& phase) { return phase.name == "allocation"; });
      const auto teardown = std::ranges::find_if(
          path.phases,
          [](const auto& phase) { return phase.name == "teardown"; });
      Expect(allocation != path.phases.end(),
             "completed path reports allocation");
      Expect(teardown != path.phases.end(), "completed path reports teardown");
      Expect(allocation->raw_repetitions.size() == options.repetitions,
             "allocation latency has raw repetitions");
      Expect(teardown->raw_repetitions.size() == options.repetitions,
             "teardown latency has raw repetitions");
    } else {
      Expect(!path.reason.empty(), "non-completed path has a reason");
    }
  }

  auto serialization_report = report;
  gufo::diagnostics::AllocationPhaseResult serialized_phase;
  serialized_phase.name = "serialization_probe";
  serialized_phase.operation = "verify raw repetition encoding";
  serialized_phase.unit = "ms";
  serialized_phase.raw_repetitions = {1.0, 2.0};
  serialization_report.paths.front().phases.push_back(
      std::move(serialized_phase));
  const std::string serialization_json = serialization_report.ToJson();
  Expect(serialization_json.find("\"rawRepetitions\":") != std::string::npos,
         "raw repetitions serialized");

  const std::string json = report.ToJson();
  Expect(json.find("\"artifactType\": \"hipAllocation\"") != std::string::npos,
         "allocation artifact type serialized");
  Expect(json.find("\"pageSizeBytes\":") != std::string::npos,
         "page size serialized");
  const auto validation = gufo::diagnostics::ValidateArtifactContent(json);
  Expect(validation.is_valid, "allocation artifact validates");
}

void TestSchemaFixture() {
  const auto schema = FindFixturesRoot() / "allocation_benchmark_schema.json";
  Expect(std::filesystem::exists(schema),
         "allocation benchmark schema fixture exists");
  std::ifstream input(schema);
  std::string first_line;
  std::getline(input, first_line);
  Expect(!first_line.empty(), "allocation benchmark schema is non-empty");
}

}  // namespace

int main() {
  TestSizeParser();
  TestStatistics();
  TestBoundedDiagnostic();
  TestSchemaFixture();
  std::cout << "HIP allocation diagnostic tests passed.\n";
  return 0;
}
