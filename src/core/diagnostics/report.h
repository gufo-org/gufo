#ifndef GUFO_CORE_DIAGNOSTICS_REPORT_H_
#define GUFO_CORE_DIAGNOSTICS_REPORT_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/diagnostics/compatibility.h"
#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/system_inventory.h"

namespace gufo::diagnostics {

enum class DiagnosticStatus : std::uint8_t {
  kPass,
  kWarn,
  kFail,
};

[[nodiscard]] std::string_view ToString(DiagnosticStatus status);

struct DiagnosticCheck {
  std::string name;
  DiagnosticStatus status{DiagnosticStatus::kPass};
  std::string message;
  std::map<std::string, std::string> details;
};

class DiagnosticReport {
public:
  DiagnosticReport();

  void AddCheck(DiagnosticCheck check);
  void AddError(std::string_view error);
  void SetInventory(SystemInventory inventory);
  void SetCompatibility(CompatibilityReport compatibility);
  void SetFingerprint(MachineFingerprint fingerprint);

  [[nodiscard]] std::string_view SchemaVersion() const {
    return schema_version_;
  }
  [[nodiscard]] std::string_view EngineRevision() const {
    return engine_revision_;
  }
  [[nodiscard]] std::string_view Timestamp() const { return timestamp_; }
  [[nodiscard]] DiagnosticStatus Status() const { return overall_status_; }
  [[nodiscard]] const std::vector<DiagnosticCheck>& Checks() const {
    return checks_;
  }
  [[nodiscard]] const std::vector<std::string>& Warnings() const {
    return warnings_;
  }
  [[nodiscard]] const std::vector<std::string>& Errors() const {
    return errors_;
  }
  [[nodiscard]] const std::optional<SystemInventory>& Inventory() const {
    return inventory_;
  }
  [[nodiscard]] const std::optional<CompatibilityReport>& Compatibility()
      const {
    return compatibility_;
  }
  [[nodiscard]] const std::optional<MachineFingerprint>& Fingerprint() const {
    return fingerprint_;
  }

  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;

private:
  void RecomputeStatus();

  std::string schema_version_{"1.0.0"};
  std::string engine_revision_{"0.1.0"};
  std::string timestamp_;
  DiagnosticStatus overall_status_{DiagnosticStatus::kPass};
  std::vector<DiagnosticCheck> checks_;
  std::vector<std::string> warnings_;
  std::vector<std::string> errors_;
  std::optional<SystemInventory> inventory_;
  std::optional<CompatibilityReport> compatibility_;
  std::optional<MachineFingerprint> fingerprint_;
};

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_REPORT_H_
