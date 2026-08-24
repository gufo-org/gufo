#ifndef STRIX_CORE_HIP_DETAIL_DISPATCH_TELEMETRY_HPP_
#define STRIX_CORE_HIP_DETAIL_DISPATCH_TELEMETRY_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>

namespace strix::hip::detail {

[[nodiscard]] inline bool DispatchTelemetryEnabled() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("STRIX_DISPATCH_TELEMETRY");
    if (value == nullptr) {
      return false;
    }
    const std::string_view text{value};
    return text != "0" && text != "false" && text != "OFF" && text != "off";
  }();
  return enabled;
}

inline std::mutex& DispatchTelemetryOutputMutex() {
  static std::mutex mutex;
  return mutex;
}

inline void WriteTelemetryField(std::ostringstream& output,
                                std::string_view name, std::string_view value) {
  output << ",\"" << name << "\":\"";
  for (const char character : value) {
    if (character == '"' || character == '\\') {
      output << '\\';
    }
    output << character;
  }
  output << '"';
}

template<typename Writer>
inline void EmitDispatchTelemetry(std::string_view event, Writer&& writer) {
  if (!DispatchTelemetryEnabled()) {
    return;
  }

  std::ostringstream output;
  output << "{\"component\":\"strix.dispatch\",\"event\":\"" << event << '"';
  writer(output);
  output << "}\n";

  const std::scoped_lock lock{DispatchTelemetryOutputMutex()};
  std::clog << output.str();
}

inline void EmitGemvDispatch(std::size_t m, std::size_t k, bool is_bf16,
                             std::string_view strategy) {
  EmitDispatchTelemetry("gemv", [&](std::ostringstream& output) {
    output << ",\"m\":" << m << ",\"k\":" << k << ",\"dataType\":\""
           << (is_bf16 ? "bf16" : "f32") << '"';
    WriteTelemetryField(output, "strategy", strategy);
  });
}

inline void EmitAttentionDispatch(std::string_view selected_backend,
                                  std::string_view rejected_fast_paths) {
  EmitDispatchTelemetry("attention", [&](std::ostringstream& output) {
    WriteTelemetryField(output, "selectedBackend", selected_backend);
    WriteTelemetryField(output, "rejectedFastPaths", rejected_fast_paths);
  });
}

inline void EmitHipblasLtDispatch(
    std::size_t batch_size, std::size_t m, std::size_t k, int algorithm_id,
    std::string_view solution_name, std::string_view kernel_name,
    std::string_view plan_source, std::string_view persistent_cache_status,
    std::size_t workspace_bytes, bool plan_cache_hit) {
  EmitDispatchTelemetry("hipblaslt", [&](std::ostringstream& output) {
    output << ",\"batchSize\":" << batch_size << ",\"m\":" << m
           << ",\"k\":" << k << ",\"algorithmId\":" << algorithm_id
           << ",\"workspaceBytes\":" << workspace_bytes
           << ",\"planCacheHit\":" << (plan_cache_hit ? "true" : "false");
    WriteTelemetryField(output, "solutionName", solution_name);
    WriteTelemetryField(output, "kernelName", kernel_name);
    WriteTelemetryField(output, "planSource", plan_source);
    WriteTelemetryField(output, "persistentCacheStatus",
                        persistent_cache_status);
  });
}

inline void EmitGraphDispatch(std::string_view cache_status,
                              std::uint64_t requested_execution_identity = 0,
                              std::uint64_t requested_workload_identity = 0,
                              std::uint64_t stored_execution_identity = 0,
                              std::uint64_t stored_workload_identity = 0) {
  EmitDispatchTelemetry("hip_graph", [&](std::ostringstream& output) {
    WriteTelemetryField(output, "cacheStatus", cache_status);
    output << ",\"requestedExecutionIdentity\":"
           << requested_execution_identity
           << ",\"requestedWorkloadIdentity\":"
           << requested_workload_identity
           << ",\"storedExecutionIdentity\":" << stored_execution_identity
           << ",\"storedWorkloadIdentity\":" << stored_workload_identity;
  });
}

inline void EmitQwenExecutionPolicy(std::uint64_t fingerprint) {
  EmitDispatchTelemetry("qwen_policy", [&](std::ostringstream& output) {
    output << ",\"fingerprint\":" << fingerprint;
  });
}

inline void EmitQwenRouteResolution(std::string_view mode,
                                    std::uint32_t layer_index,
                                    std::string_view layer_kind,
                                    std::uint64_t route_fingerprint,
                                    std::uint32_t rejection_mask) {
  EmitDispatchTelemetry("qwen_route", [&](std::ostringstream& output) {
    WriteTelemetryField(output, "mode", mode);
    output << ",\"layerIndex\":" << layer_index;
    WriteTelemetryField(output, "layerKind", layer_kind);
    output << ",\"routeFingerprint\":" << route_fingerprint
           << ",\"rejectionMask\":" << rejection_mask;
  });
}

inline void EmitQwenGraphEligibility(std::uint64_t policy_fingerprint,
                                     std::uint64_t workload_identity,
                                     std::uint32_t rejection_mask) {
  EmitDispatchTelemetry("qwen_graph_eligibility",
                        [&](std::ostringstream& output) {
                          output << ",\"policyFingerprint\":"
                                 << policy_fingerprint
                                 << ",\"workloadIdentity\":"
                                 << workload_identity
                                 << ",\"rejectionMask\":" << rejection_mask;
                        });
}

}  // namespace strix::hip::detail

#endif  // STRIX_CORE_HIP_DETAIL_DISPATCH_TELEMETRY_HPP_
