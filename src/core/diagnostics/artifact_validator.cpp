#include "src/core/diagnostics/artifact_validator.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "src/core/diagnostics/fingerprint.h"
#include "src/core/hip/arch.hpp"
#include "src/core/json.hpp"

namespace gufo::diagnostics {

namespace {

constexpr std::size_t kMaxArtifactBytes = 16 * 1024 * 1024;

const json::Value& Object(const json::Value& parent, const std::string& name) {
  const auto* value = parent.find(name);
  if (!value || !value->is_object())
    throw std::invalid_argument("Expected object '" + name + "'");
  return *value;
}

std::string String(const json::Value& object, const std::string& name) {
  const auto* value = object.find(name);
  if (!value || !value->is_string())
    throw std::invalid_argument("Expected string '" + name + "'");
  return value->str();
}

std::uint64_t Uint(const json::Value& object, const std::string& name,
                   std::uint64_t maximum = 9007199254740991ULL) {
  const auto* value = object.find(name);
  if (!value || !value->is_number() || value->as_double() < 0 ||
      value->as_double() > static_cast<double>(maximum) ||
      std::floor(value->as_double()) != value->as_double())
    throw std::invalid_argument("Expected bounded unsigned integer '" + name +
                                "'");
  return static_cast<std::uint64_t>(value->as_double());
}

bool Boolean(const json::Value& object, const std::string& name) {
  const auto* value = object.find(name);
  if (!value || !value->is_bool())
    throw std::invalid_argument("Expected boolean '" + name + "'");
  return value->as_bool();
}

bool IsSha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

bool ContainsLocalMetadata(const json::Value& value) {
  if (value.is_string())
    return value.str().find("/home/") != std::string::npos;
  for (const auto& [name, member] : value.members())
    if (name == "hostname" || name == "username" || name == "secret" ||
        ContainsLocalMetadata(member))
      return true;
  for (const auto& item : value.items())
    if (ContainsLocalMetadata(item))
      return true;
  return false;
}

}  // namespace

ValidationResult ValidateArtifactContent(std::string_view content) {
  ValidationResult res;
  const auto fail = [&](std::string error) {
    res.is_valid = false;
    res.errors.push_back(std::move(error));
  };
  try {
    if (content.size() > kMaxArtifactBytes)
      throw std::invalid_argument("Artifact exceeds the 16 MiB limit");
    const auto root = json::parse(content);
    if (!root.is_object())
      throw std::invalid_argument("Artifact must be a JSON object");
    res.schema_version = String(root, "schemaVersion");
    if (res.schema_version != "1.0.0")
      fail("Unsupported schema version: " + res.schema_version);
    res.fingerprint_id = String(root, "fingerprintId");
    if (!IsSha256(res.fingerprint_id))
      fail("Invalid fingerprintId: expected a lowercase hexadecimal SHA-256");
    const auto check_architecture = [&](const json::Value& object) {
      if (object.contains("gpuArchitecture")) {
        const auto arch = String(object, "gpuArchitecture");
        if (!arch.empty() && !gufo::hip::IsRdna35(arch))
          fail("Incompatible GPU architecture: " + arch);
      }
    };
    check_architecture(root);
    if (root.contains("machine"))
      check_architecture(Object(root, "machine"));
    if (root.contains("canonical")) {
      const auto& block = Object(root, "canonical");
      CanonicalFingerprint canonical;
      canonical.schema_version = String(block, "schemaVersion");
      canonical.cpp_standard = String(block, "cppStandard");
      canonical.cpu_architecture = String(block, "cpuArchitecture");
      canonical.cpu_logical_cores = static_cast<std::uint32_t>(
          Uint(block, "cpuLogicalCores", UINT32_MAX));
      canonical.cpu_model = String(block, "cpuModel");
      canonical.cpu_physical_cores = static_cast<std::uint32_t>(
          Uint(block, "cpuPhysicalCores", UINT32_MAX));
      canonical.cxx_compiler = String(block, "cxxCompiler");
      canonical.gpu_architecture = String(block, "gpuArchitecture");
      canonical.gpu_compute_units = static_cast<std::uint32_t>(
          Uint(block, "gpuComputeUnits", UINT32_MAX));
      canonical.gpu_driver = String(block, "gpuDriver");
      canonical.gpu_name = String(block, "gpuName");
      canonical.gpu_pci_id = String(block, "gpuPciId");
      canonical.kernel_release = String(block, "kernelRelease");
      canonical.memory_total_bytes = Uint(block, "memoryTotalBytes");
      canonical.memory_type = String(block, "memoryType");
      canonical.rocm_version = String(block, "rocmVersion");

      if (canonical.schema_version != res.schema_version)
        fail("Canonical schemaVersion does not match the artifact");
      if (canonical.ComputeFingerprintId() != res.fingerprint_id)
        fail("Fingerprint ID mismatch over canonical fields");
      check_architecture(block);
    }
    const auto type =
        root.contains("artifactType") ? String(root, "artifactType") : "";
    if (type == "hipAllocation") {
      const auto& summary = Object(root, "summary");
      if (!Boolean(summary, "allRequestedPathsReported") ||
          !Boolean(summary, "checksumsVerified") ||
          !Boolean(summary, "phasesSeparated") ||
          String(summary, "status") != "completed")
        fail("HIP allocation diagnostic did not complete all paths and checks");
    }
    if (!type.empty() && type != "hipAllocation")
      fail("Unknown diagnostic artifactType: " + type);
    if (type.empty() && !root.contains("canonical")) {
      // Bandwidth artifacts reference a fingerprint rather than embedding it.
      // Recognize their structure; schemaVersion and an arbitrary hash alone
      // are not a diagnostic artifact.
      const auto& options = Object(root, "options");
      (void)Uint(options, "warmup");
      (void)Uint(options, "repetitions");
      (void)Uint(options, "durationMs");
      (void)Uint(options, "workingSetBytes");
      const auto* paths = root.find("paths");
      if (!paths || !paths->is_array())
        throw std::invalid_argument("Expected bandwidth paths array");
      for (const auto& path : paths->items()) {
        if (!path.is_object())
          throw std::invalid_argument("Expected bandwidth path object");
        (void)String(path, "backend");
        (void)String(path, "pathName");
        (void)Boolean(path, "sentinelVerified");
        (void)Uint(path, "workingSetBytes");
        const auto* samples = path.find("rawRepetitionsGbps");
        if (!samples || !samples->is_array())
          throw std::invalid_argument("Expected bandwidth samples array");
        for (const auto& sample : samples->items())
          if (!sample.is_number() || sample.as_double() < 0)
            throw std::invalid_argument("Invalid bandwidth sample");
      }
    }
    if (ContainsLocalMetadata(root))
      res.warnings.emplace_back(
          "Artifact contains unredacted local host metadata or paths");
  } catch (const std::exception& exception) {
    fail(exception.what());
  }
  return res;
}

ValidationResult ValidateArtifactFile(const std::filesystem::path& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    ValidationResult res;
    res.is_valid = false;
    res.errors.emplace_back("Could not open artifact file: " +
                            file_path.string());
    return res;
  }
  std::string content(kMaxArtifactBytes + 1, '\0');
  file.read(content.data(), static_cast<std::streamsize>(content.size()));
  content.resize(static_cast<std::size_t>(file.gcount()));
  if (file.bad()) {
    ValidationResult res;
    res.is_valid = false;
    res.errors.emplace_back("Could not read artifact file");
    return res;
  }
  return ValidateArtifactContent(content);
}

std::string ValidationResult::ToJson() const {
  auto value = json::Value::object();
  value["isValid"] = is_valid;
  value["schemaVersion"] = schema_version;
  value["fingerprintId"] = fingerprint_id;
  value["errors"] = json::Value::array();
  value["warnings"] = json::Value::array();
  for (const auto& error : errors)
    value["errors"].push_back(error);
  for (const auto& warning : warnings)
    value["warnings"].push_back(warning);
  return value.dump() + "\n";
}

std::string ValidationResult::ToHuman() const {
  std::ostringstream oss;
  oss << "=== Artifact Validation Result: [" << (is_valid ? "VALID" : "INVALID")
      << "] ===\n";
  oss << "Schema Version  : " << schema_version << "\n";
  oss << "Fingerprint ID  : " << fingerprint_id << "\n";

  if (!errors.empty()) {
    oss << "\n--- Validation Errors ---\n";
    for (const auto& err : errors) {
      oss << "  [ERROR] " << err << "\n";
    }
  }

  if (!warnings.empty()) {
    oss << "\n--- Validation Warnings ---\n";
    for (const auto& warn : warnings) {
      oss << "  [WARN] " << warn << "\n";
    }
  }

  return oss.str();
}

}  // namespace gufo::diagnostics
