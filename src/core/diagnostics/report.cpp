#include "src/core/diagnostics/report.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "src/core/platform/os_info.hpp"

namespace gufo::diagnostics {

namespace {

std::string CurrentIso8601Utc() {
  const auto now = std::chrono::system_clock::now();
  const auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm tm_utc{};
  gufo::platform::UtcTime(time_t_now, &tm_utc);

  std::ostringstream oss;
  oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
  return oss.str();
}

std::string EscapeJson(std::string_view str) {
  std::ostringstream oss;
  for (const char character : str) {
    switch (character) {
      case '"':
        oss << "\\\"";
        break;
      case '\\':
        oss << "\\\\";
        break;
      case '\b':
        oss << "\\b";
        break;
      case '\f':
        oss << "\\f";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(character) < 0x20) {
          oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(character);
        } else {
          oss << character;
        }
        break;
    }
  }
  return oss.str();
}

}  // namespace

std::string_view ToString(DiagnosticStatus status) {
  switch (status) {
    case DiagnosticStatus::kPass:
      return "PASS";
    case DiagnosticStatus::kWarn:
      return "WARN";
    case DiagnosticStatus::kFail:
      return "FAIL";
  }
  return "UNKNOWN";
}

DiagnosticStatus FromString(std::string_view str) {
  if (str == "PASS" || str == "pass") {
    return DiagnosticStatus::kPass;
  }
  if (str == "WARN" || str == "warn") {
    return DiagnosticStatus::kWarn;
  }
  return DiagnosticStatus::kFail;
}

DiagnosticReport::DiagnosticReport() : timestamp_(CurrentIso8601Utc()) {}

void DiagnosticReport::SetEngineRevision(std::string_view revision) {
  engine_revision_ = std::string(revision);
}

void DiagnosticReport::SetTimestamp(std::string_view iso8601_utc) {
  timestamp_ = std::string(iso8601_utc);
}

void DiagnosticReport::AddCheck(DiagnosticCheck check) {
  checks_.push_back(std::move(check));
  RecomputeStatus();
}

void DiagnosticReport::AddWarning(std::string_view warning) {
  warnings_.emplace_back(warning);
  if (overall_status_ == DiagnosticStatus::kPass) {
    overall_status_ = DiagnosticStatus::kWarn;
  }
}

void DiagnosticReport::AddError(std::string_view error) {
  errors_.emplace_back(error);
  overall_status_ = DiagnosticStatus::kFail;
}

void DiagnosticReport::SetInventory(SystemInventory inventory) {
  inventory_ = std::move(inventory);
}

void DiagnosticReport::SetCompatibility(CompatibilityReport compatibility) {
  if (compatibility.overall_verdict == CompatibilityVerdict::kUnsupported) {
    overall_status_ = DiagnosticStatus::kFail;
  }
  compatibility_ = std::move(compatibility);
}

void DiagnosticReport::SetFingerprint(MachineFingerprint fingerprint) {
  fingerprint_ = std::move(fingerprint);
}

void DiagnosticReport::RecomputeStatus() {
  DiagnosticStatus overall_st = DiagnosticStatus::kPass;
  for (const auto& check : checks_) {
    if (check.status == DiagnosticStatus::kFail) {
      overall_st = DiagnosticStatus::kFail;
      break;
    }
    if (check.status == DiagnosticStatus::kWarn) {
      overall_st = DiagnosticStatus::kWarn;
    }
  }
  if (!errors_.empty() ||
      (compatibility_ &&
       compatibility_->overall_verdict == CompatibilityVerdict::kUnsupported)) {
    overall_st = DiagnosticStatus::kFail;
  } else if (!warnings_.empty() && overall_st == DiagnosticStatus::kPass) {
    overall_st = DiagnosticStatus::kWarn;
  }
  overall_status_ = overall_st;
}

std::string DiagnosticReport::ToJson() const {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"schemaVersion\": \"" << EscapeJson(schema_version_) << "\",\n";
  oss << "  \"engineRevision\": \"" << EscapeJson(engine_revision_) << "\",\n";
  oss << "  \"timestamp\": \"" << EscapeJson(timestamp_) << "\",\n";
  oss << "  \"status\": \"" << ToString(overall_status_) << "\",\n";

  if (fingerprint_) {
    oss << "  \"fingerprintId\": \"" << EscapeJson(fingerprint_->fingerprint_id)
        << "\",\n";
    oss << "  \"fingerprint\": " << fingerprint_->ToJson() << ",\n";
  }

  if (inventory_) {
    oss << "  \"inventory\": " << inventory_->ToJson() << ",\n";
  }

  if (compatibility_) {
    oss << "  \"compatibility\": " << compatibility_->ToJson() << ",\n";
  }

  // checks array
  oss << "  \"checks\": [\n";
  for (std::size_t i = 0; i < checks_.size(); ++i) {
    const auto& check_item = checks_[i];
    oss << "    {\n";
    oss << "      \"name\": \"" << EscapeJson(check_item.name) << "\",\n";
    oss << "      \"status\": \"" << ToString(check_item.status) << "\",\n";
    oss << "      \"message\": \"" << EscapeJson(check_item.message) << "\",\n";
    oss << "      \"details\": {";
    if (!check_item.details.empty()) {
      oss << "\n";
      std::size_t detail_idx = 0;
      for (const auto& [key, val] : check_item.details) {
        oss << "        \"" << EscapeJson(key) << "\": \"" << EscapeJson(val)
            << "\"";
        if (++detail_idx < check_item.details.size()) {
          oss << ",";
        }
        oss << "\n";
      }
      oss << "      }\n";
    } else {
      oss << "}\n";
    }
    oss << "    }";
    if (i + 1 < checks_.size()) {
      oss << ",";
    }
    oss << "\n";
  }
  oss << "  ],\n";

  // warnings array
  oss << "  \"warnings\": [";
  if (!warnings_.empty()) {
    oss << "\n";
    for (std::size_t i = 0; i < warnings_.size(); ++i) {
      oss << "    \"" << EscapeJson(warnings_[i]) << "\"";
      if (i + 1 < warnings_.size()) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ],\n";
  } else {
    oss << "],\n";
  }

  // errors array
  oss << "  \"errors\": [";
  if (!errors_.empty()) {
    oss << "\n";
    for (std::size_t i = 0; i < errors_.size(); ++i) {
      oss << "    \"" << EscapeJson(errors_[i]) << "\"";
      if (i + 1 < errors_.size()) {
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

std::string DiagnosticReport::ToHuman() const {
  std::ostringstream oss;
  oss << "=== Gufo Diagnostics ===\n";
  oss << "Engine Revision : " << engine_revision_ << "\n";
  oss << "Schema Version  : " << schema_version_ << "\n";
  oss << "Timestamp       : " << timestamp_ << "\n";
  oss << "Overall Status  : [" << ToString(overall_status_) << "]\n\n";

  if (fingerprint_) {
    oss << fingerprint_->ToHuman() << "\n";
  }

  if (inventory_) {
    oss << inventory_->ToHuman() << "\n";
  }

  if (compatibility_) {
    oss << compatibility_->ToHuman() << "\n";
  }

  oss << "--- Checks ---\n";
  for (const auto& check : checks_) {
    oss << "  [" << ToString(check.status) << "] " << check.name << ": "
        << check.message << "\n";
    for (const auto& [key, val] : check.details) {
      oss << "      - " << key << ": " << val << "\n";
    }
  }

  if (!warnings_.empty()) {
    oss << "\n--- Warnings ---\n";
    for (const auto& warning_msg : warnings_) {
      oss << "  [WARN] " << warning_msg << "\n";
    }
  }

  if (!errors_.empty()) {
    oss << "\n--- Errors ---\n";
    for (const auto& error_msg : errors_) {
      oss << "  [ERROR] " << error_msg << "\n";
    }
  }

  return oss.str();
}

}  // namespace gufo::diagnostics
