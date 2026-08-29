#include "src/core/diagnostics/artifact_validator.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "src/core/diagnostics/fingerprint.h"

namespace gufo::diagnostics {

namespace {

std::string ExtractJsonStringField(std::string_view content,
                                   std::string_view field_name) {
  const std::string needle = "\"" + std::string(field_name) + "\"";
  const auto pos = content.find(needle);
  if (pos == std::string_view::npos) {
    return {};
  }
  const auto colon_pos = content.find(':', pos + needle.size());
  if (colon_pos == std::string_view::npos) {
    return {};
  }
  const auto quote_start = content.find('"', colon_pos + 1);
  if (quote_start == std::string_view::npos) {
    return {};
  }
  const auto quote_end = content.find('"', quote_start + 1);
  if (quote_end == std::string_view::npos) {
    return {};
  }
  return std::string(
      content.substr(quote_start + 1, quote_end - quote_start - 1));
}

std::uint64_t ExtractJsonUint64Field(std::string_view content,
                                     std::string_view field_name) {
  const std::string needle = "\"" + std::string(field_name) + "\"";
  const auto pos = content.find(needle);
  if (pos == std::string_view::npos) {
    return 0;
  }
  const auto colon_pos = content.find(':', pos + needle.size());
  if (colon_pos == std::string_view::npos) {
    return 0;
  }
  const auto num_start = content.find_first_of("0123456789", colon_pos + 1);
  if (num_start == std::string_view::npos) {
    return 0;
  }
  const auto num_end = content.find_first_not_of("0123456789", num_start);
  const std::string num_str =
      num_end == std::string_view::npos
          ? std::string(content.substr(num_start))
          : std::string(content.substr(num_start, num_end - num_start));
  return std::strtoull(num_str.c_str(), nullptr, 10);
}

bool ExtractJsonBoolField(std::string_view content, std::string_view field_name,
                          bool fallback) {
  const std::string needle = "\"" + std::string(field_name) + "\"";
  const auto pos = content.find(needle);
  if (pos == std::string_view::npos) {
    return fallback;
  }
  const auto colon_pos = content.find(':', pos + needle.size());
  if (colon_pos == std::string_view::npos) {
    return fallback;
  }
  const auto value_pos = content.find_first_not_of(" \t\r\n", colon_pos + 1);
  if (value_pos == std::string_view::npos) {
    return fallback;
  }
  if (content.substr(value_pos, 4) == "true") {
    return true;
  }
  if (content.substr(value_pos, 5) == "false") {
    return false;
  }
  return fallback;
}

std::string EscapeJsonString(std::string_view str) {
  std::ostringstream oss;
  for (const char character : str) {
    if (character == '"') {
      oss << "\\\"";
    } else if (character == '\\') {
      oss << "\\\\";
    } else if (character == '\n') {
      oss << "\\n";
    } else if (character == '\t') {
      oss << "\\t";
    } else {
      oss << character;
    }
  }
  return oss.str();
}

}  // namespace

ValidationResult ValidateArtifactContent(std::string_view content) {
  ValidationResult res;

  if (content.empty()) {
    res.is_valid = false;
    res.errors.emplace_back("Artifact is empty");
    return res;
  }

  // 1. Check schemaVersion
  res.schema_version = ExtractJsonStringField(content, "schemaVersion");
  if (res.schema_version.empty()) {
    res.is_valid = false;
    res.errors.emplace_back("Missing 'schemaVersion' field in artifact");
  } else if (res.schema_version != "1.0.0") {
    res.is_valid = false;
    res.errors.emplace_back("Unsupported schema version: " +
                            res.schema_version + " (expected 1.0.0)");
  }

  // 2. Check fingerprintId
  res.fingerprint_id = ExtractJsonStringField(content, "fingerprintId");
  if (res.fingerprint_id.empty()) {
    res.is_valid = false;
    res.errors.emplace_back("Missing 'fingerprintId' binding in artifact");
  } else if (res.fingerprint_id.size() != 64) {
    res.is_valid = false;
    res.errors.emplace_back(
        "Invalid 'fingerprintId' format: expected 64-char hex SHA-256");
  }

  // 3. Extract and validate canonical block if present
  const auto canonical_start = content.find("\"canonical\":");
  if (canonical_start != std::string_view::npos) {
    CanonicalFingerprint canonical;
    canonical.schema_version = ExtractJsonStringField(content, "schemaVersion");
    canonical.cpp_standard = ExtractJsonStringField(content, "cppStandard");
    canonical.cpu_architecture =
        ExtractJsonStringField(content, "cpuArchitecture");
    canonical.cpu_logical_cores = static_cast<std::uint32_t>(
        ExtractJsonUint64Field(content, "cpuLogicalCores"));
    canonical.cpu_model = ExtractJsonStringField(content, "cpuModel");
    canonical.cpu_physical_cores = static_cast<std::uint32_t>(
        ExtractJsonUint64Field(content, "cpuPhysicalCores"));
    canonical.cxx_compiler = ExtractJsonStringField(content, "cxxCompiler");
    canonical.gpu_architecture =
        ExtractJsonStringField(content, "gpuArchitecture");
    canonical.gpu_compute_units = static_cast<std::uint32_t>(
        ExtractJsonUint64Field(content, "gpuComputeUnits"));
    canonical.gpu_driver = ExtractJsonStringField(content, "gpuDriver");
    canonical.gpu_name = ExtractJsonStringField(content, "gpuName");
    canonical.gpu_pci_id = ExtractJsonStringField(content, "gpuPciId");
    canonical.kernel_release = ExtractJsonStringField(content, "kernelRelease");
    canonical.memory_total_bytes =
        ExtractJsonUint64Field(content, "memoryTotalBytes");
    canonical.memory_type = ExtractJsonStringField(content, "memoryType");
    canonical.npu_architecture =
        ExtractJsonStringField(content, "npuArchitecture");
    canonical.npu_driver = ExtractJsonStringField(content, "npuDriver");
    canonical.npu_firmware_version =
        ExtractJsonStringField(content, "npuFirmwareVersion");
    canonical.npu_identity = ExtractJsonStringField(content, "npuIdentity");
    canonical.npu_pci_id = ExtractJsonStringField(content, "npuPciId");
    canonical.rocm_version = ExtractJsonStringField(content, "rocmVersion");
    canonical.xrt_commit = ExtractJsonStringField(content, "xrtCommit");

    const std::string computed_id = canonical.ComputeFingerprintId();

    if (!res.fingerprint_id.empty() && res.fingerprint_id != computed_id) {
      res.is_valid = false;
      res.errors.emplace_back("Fingerprint ID mismatch: claimed '" +
                              res.fingerprint_id + "', but computed '" +
                              computed_id + "' over canonical fields");
    }
  }

  // 4. Validate architecture
  const std::string gpu_arch =
      ExtractJsonStringField(content, "gpuArchitecture");
  if (!gpu_arch.empty() && gpu_arch != "gfx1151") {
    res.is_valid = false;
    res.errors.emplace_back("Incompatible GPU architecture: " + gpu_arch +
                            " (expected gfx1151)");
  }

  const std::string npu_arch =
      ExtractJsonStringField(content, "npuArchitecture");
  if (!npu_arch.empty() && npu_arch != "XDNA2" && npu_arch != "AIE2P") {
    res.is_valid = false;
    res.errors.emplace_back("Incompatible NPU architecture: " + npu_arch +
                            " (expected XDNA2 or AIE2P)");
  }

  const std::string artifact_type =
      ExtractJsonStringField(content, "artifactType");
  if (artifact_type == "xrtSmoke") {
    const std::string program_hash =
        ExtractJsonStringField(content, "programSha256");
    if (program_hash.size() != 64) {
      res.is_valid = false;
      res.errors.emplace_back(
          "Invalid XRT smoke programSha256: expected 64-char SHA-256");
    }
    const auto iterations = ExtractJsonUint64Field(content, "iterations");
    const auto completed =
        ExtractJsonUint64Field(content, "completedIterations");
    if (iterations == 0 || completed != iterations) {
      res.is_valid = false;
      res.errors.emplace_back(
          "XRT smoke did not complete every requested iteration");
    }
    if (ExtractJsonStringField(content, "status") != "completed" ||
        ExtractJsonStringField(content, "completionStatus") != "completed") {
      res.is_valid = false;
      res.errors.emplace_back("XRT smoke completion status is not successful");
    }
    if (ExtractJsonBoolField(content, "quarantined", true)) {
      res.is_valid = false;
      res.errors.emplace_back("XRT smoke context was quarantined");
    }
  } else if (artifact_type == "hipAllocation") {
    if (!ExtractJsonBoolField(content, "allRequestedPathsReported", false)) {
      res.is_valid = false;
      res.errors.emplace_back(
          "HIP allocation diagnostic omitted one or more requested paths");
    }
    if (!ExtractJsonBoolField(content, "checksumsVerified", false)) {
      res.is_valid = false;
      res.errors.emplace_back(
          "HIP allocation diagnostic did not verify CPU/GPU checksums");
    }
    if (!ExtractJsonBoolField(content, "phasesSeparated", false)) {
      res.is_valid = false;
      res.errors.emplace_back(
          "HIP allocation diagnostic did not keep benchmark phases separate");
    }
    if (ExtractJsonStringField(content, "status") != "completed") {
      res.is_valid = false;
      res.errors.emplace_back(
          "HIP allocation diagnostic completion status is not successful");
    }
  }

  // 5. Redaction and sensitive field checks
  if (content.find("\"hostname\":") != std::string_view::npos ||
      content.find("\"username\":") != std::string_view::npos ||
      content.find("\"secret\":") != std::string_view::npos ||
      content.find("/home/") != std::string_view::npos) {
    res.warnings.emplace_back(
        "Artifact contains unredacted local host metadata or paths");
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
  std::stringstream buffer;
  buffer << file.rdbuf();
  return ValidateArtifactContent(buffer.str());
}

std::string ValidationResult::ToJson() const {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"isValid\": " << (is_valid ? "true" : "false") << ",\n";
  oss << "  \"schemaVersion\": \"" << EscapeJsonString(schema_version)
      << "\",\n";
  oss << "  \"fingerprintId\": \"" << EscapeJsonString(fingerprint_id)
      << "\",\n";

  oss << "  \"errors\": [";
  if (!errors.empty()) {
    oss << "\n";
    for (std::size_t i = 0; i < errors.size(); ++i) {
      oss << "    \"" << EscapeJsonString(errors[i]) << "\"";
      if (i + 1 < errors.size()) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ],\n";
  } else {
    oss << "],\n";
  }

  oss << "  \"warnings\": [";
  if (!warnings.empty()) {
    oss << "\n";
    for (std::size_t i = 0; i < warnings.size(); ++i) {
      oss << "    \"" << EscapeJsonString(warnings[i]) << "\"";
      if (i + 1 < warnings.size()) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ]\n";
  } else {
    oss << "]\n";
  }

  oss << "}\n";
  return oss.str();
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
