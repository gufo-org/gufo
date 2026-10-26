#include "src/core/diagnostics/compatibility.h"

#include <sstream>
#include <string>

namespace gufo::diagnostics {

namespace {

std::string EscapeJsonString(std::string_view str) {
  std::ostringstream oss;
  for (const char ch : str) {
    if (ch == '"') {
      oss << "\\\"";
    } else if (ch == '\\') {
      oss << "\\\\";
    } else if (ch == '\n') {
      oss << "\\n";
    } else {
      oss << ch;
    }
  }
  return oss.str();
}

}  // namespace

std::string_view ToString(CompatibilityVerdict verdict) {
  switch (verdict) {
    case CompatibilityVerdict::kSupported:
      return "supported";
    case CompatibilityVerdict::kUnsupported:
      return "unsupported";
    case CompatibilityVerdict::kUnknown:
      return "unknown";
  }
  return "unknown";
}

CompatibilityReport EvaluateCompatibility(const SystemInventory& inventory,
                                          std::string_view section) {
  CompatibilityReport report;
  CompatibilityVerdict overall = CompatibilityVerdict::kSupported;

  // 1. Platform Check
  if (section == "all" || section == "platform" || section == "inventory") {
    CompatibilityItem platform_item;
    platform_item.component = "platform";
    const std::string& target = inventory.toolchain.target_platform;
    platform_item.detected_value = target;
    platform_item.required_value = "x86_64-linux or x86_64-windows";
    platform_item.evidence = "CPU architecture / host OS";
    if (target == "x86_64-linux" || target == "x86_64-windows") {
      platform_item.verdict = CompatibilityVerdict::kSupported;
      platform_item.remediation_hint = "None";
    } else {
      platform_item.verdict = CompatibilityVerdict::kUnsupported;
      platform_item.remediation_hint =
          "gufo supports Linux x86-64 and Windows x86-64";
      overall = CompatibilityVerdict::kUnsupported;
    }
    report.items.push_back(std::move(platform_item));
  }

  // 2. GPU Architecture Check (gfx1151)
  if (section == "all" || section == "gpu" || section == "inventory") {
    CompatibilityItem gpu_item;
    gpu_item.component = "gpu";
    gpu_item.detected_value = inventory.gpu.architecture;
    gpu_item.required_value = "gfx1151";
    gpu_item.evidence = inventory.gpu.source;
    if (inventory.gpu.architecture == "gfx1151") {
      gpu_item.verdict = CompatibilityVerdict::kSupported;
      gpu_item.remediation_hint = "None";
    } else {
      gpu_item.verdict = CompatibilityVerdict::kUnsupported;
      gpu_item.remediation_hint =
          "gufo requires AMD Strix Halo gfx1151 GPU (Radeon 8060S / "
          "8050S). Detected: " +
          inventory.gpu.architecture;
      overall = CompatibilityVerdict::kUnsupported;
    }
    report.items.push_back(std::move(gpu_item));
  }

  // 3. Kernel Driver Check
  if (section == "all" || section == "inventory") {
    CompatibilityItem driver_item;
    driver_item.component = "kernel_driver";
    const bool linux_host = inventory.toolchain.os_name == "Linux";
    const std::string driver =
        linux_host ? std::string("amdgpu") : inventory.gpu.driver_name;
    driver_item.detected_value =
        driver + ": " + inventory.toolchain.amdgpu_status;
    driver_item.required_value = driver + ": loaded";
    driver_item.evidence =
        linux_host ? "/sys/class/drm" : "HIP device enumeration";
    if (inventory.toolchain.amdgpu_status == "loaded") {
      driver_item.verdict = CompatibilityVerdict::kSupported;
      driver_item.remediation_hint = "None";
    } else {
      driver_item.verdict = CompatibilityVerdict::kUnsupported;
      driver_item.remediation_hint =
          linux_host
              ? "Ensure Linux kernel >= 6.10 with the amdgpu driver loaded"
              : "Install an AMD graphics driver with HIP support";
      overall = CompatibilityVerdict::kUnsupported;
    }
    report.items.push_back(std::move(driver_item));
  }

  report.overall_verdict = overall;
  return report;
}

std::string CompatibilityReport::ToJson() const {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"overallVerdict\": \"" << ToString(overall_verdict) << "\",\n";
  oss << "  \"items\": [\n";

  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    oss << "    {\n";
    oss << "      \"component\": \"" << EscapeJsonString(item.component)
        << "\",\n";
    oss << "      \"verdict\": \"" << ToString(item.verdict) << "\",\n";
    oss << "      \"detectedValue\": \""
        << EscapeJsonString(item.detected_value) << "\",\n";
    oss << "      \"requiredValue\": \""
        << EscapeJsonString(item.required_value) << "\",\n";
    oss << "      \"evidence\": \"" << EscapeJsonString(item.evidence)
        << "\",\n";
    oss << "      \"remediationHint\": \""
        << EscapeJsonString(item.remediation_hint) << "\"\n";
    oss << "    }";
    if (i + 1 < items.size()) {
      oss << ",";
    }
    oss << "\n";
  }

  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

std::string CompatibilityReport::ToHuman() const {
  std::ostringstream oss;
  oss << "=== Compatibility Verdict: [" << ToString(overall_verdict)
      << "] ===\n\n";

  for (const auto& item : items) {
    oss << "  - " << item.component << " [" << ToString(item.verdict) << "]\n";
    oss << "      Detected   : " << item.detected_value << "\n";
    oss << "      Required   : " << item.required_value << "\n";
    if (item.verdict != CompatibilityVerdict::kSupported &&
        !item.remediation_hint.empty() && item.remediation_hint != "None") {
      oss << "      Remediation: " << item.remediation_hint << "\n";
    }
  }

  return oss.str();
}

}  // namespace gufo::diagnostics
