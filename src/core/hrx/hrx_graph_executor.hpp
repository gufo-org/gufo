#ifndef GUFO_CORE_HRX_HRX_GRAPH_EXECUTOR_HPP_
#define GUFO_CORE_HRX_HRX_GRAPH_EXECUTOR_HPP_

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string_view>

#include <hrx/hrx_runtime.h>
#include "src/core/hrx/hrx_utils.hpp"

namespace gufo::hrx {

struct HrxGraphCaptureKey {
  std::uint64_t execution_identity{0};
  std::uint64_t workload_identity{0};

  [[nodiscard]] friend constexpr bool operator==(
      const HrxGraphCaptureKey&, const HrxGraphCaptureKey&) noexcept = default;
};

class HrxGraphDecodeExecutor {
public:
  HrxGraphDecodeExecutor() {
    const char* env = std::getenv("GUFO_ENABLE_HRX_GRAPH");
    if (env != nullptr) {
      const std::string_view val(env);
      if (val == "0" || val == "false" || val == "OFF" || val == "off") {
        is_enabled_ = false;
      }
    }
  }

  ~HrxGraphDecodeExecutor() { Reset(); }

  HrxGraphDecodeExecutor(const HrxGraphDecodeExecutor&) = delete;
  HrxGraphDecodeExecutor& operator=(const HrxGraphDecodeExecutor&) = delete;
  HrxGraphDecodeExecutor(HrxGraphDecodeExecutor&&) = delete;
  HrxGraphDecodeExecutor& operator=(HrxGraphDecodeExecutor&&) = delete;

  void Reset() noexcept {
    if (exec_instance_ != nullptr) {
      hrx_graph_exec_release(exec_instance_);
      exec_instance_ = nullptr;
    }
    if (graph_ != nullptr) {
      hrx_graph_release(graph_);
      graph_ = nullptr;
    }
    capture_key_.reset();
    is_capturing_ = false;
    is_instantiated_ = false;
  }

  [[nodiscard]] bool IsEnabled() const noexcept { return is_enabled_; }
  [[nodiscard]] bool IsInstantiated() const noexcept {
    return is_instantiated_;
  }
  [[nodiscard]] bool IsInstantiatedFor(HrxGraphCaptureKey key) const noexcept {
    return is_instantiated_ && capture_key_.has_value() && *capture_key_ == key;
  }

  [[nodiscard]] hrx_graph_t Graph() const noexcept { return graph_; }
  [[nodiscard]] hrx_graph_exec_t ExecInstance() const noexcept {
    return exec_instance_;
  }

  bool BeginCapture(hrx_stream_t stream, HrxGraphCaptureKey key) {
    if (!is_enabled_ || stream == nullptr) {
      return false;
    }
    Reset();
    hrx_status_t status =
        hrx_stream_begin_capture(stream, HRX_CAPTURE_MODE_THREAD_LOCAL);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      return false;
    }
    capture_key_ = key;
    is_capturing_ = true;
    return true;
  }

  bool EndCapture(hrx_stream_t stream) {
    if (!is_capturing_ || stream == nullptr) {
      return false;
    }
    hrx_status_t status = hrx_stream_end_capture(stream, &graph_);
    is_capturing_ = false;
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      return false;
    }
    return Instantiate();
  }

  [[nodiscard]] bool IsCapturing(hrx_stream_t stream) const noexcept {
    if (!is_capturing_ || stream == nullptr) {
      return false;
    }
    return hrx_stream_is_capturing(stream);
  }

  bool InitializeGraph(hrx_device_t device, HrxGraphCaptureKey key) {
    Reset();
    if (!is_enabled_) {
      return false;
    }
    hrx_status_t status = hrx_graph_create(device, 0, &graph_);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      return false;
    }
    capture_key_ = key;
    return true;
  }

  bool AddEmptyNode(const hrx_graph_node_t* deps = nullptr,
                    size_t dep_count = 0,
                    hrx_graph_node_t* out_node = nullptr) {
    if (graph_ == nullptr) {
      return false;
    }
    hrx_status_t status =
        hrx_graph_add_empty_node(graph_, deps, dep_count, out_node);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      return false;
    }
    return true;
  }

  bool AddKernelNode(hrx_executable_t executable, uint32_t export_ordinal,
                     const hrx_dispatch_config_t& config,
                     const hrx_buffer_ref_t* bindings, size_t binding_count,
                     const void* constants = nullptr, size_t constants_size = 0,
                     const hrx_graph_node_t* deps = nullptr,
                     size_t dep_count = 0,
                     hrx_graph_node_t* out_node = nullptr) {
    if (graph_ == nullptr || executable == nullptr) {
      return false;
    }
    hrx_graph_kernel_node_attrs_t attrs{};
    attrs.executable = executable;
    attrs.export_ordinal = export_ordinal;
    attrs.config = config;
    attrs.constants = constants;
    attrs.constants_size = constants_size;
    attrs.bindings = bindings;
    attrs.binding_count = binding_count;
    attrs.flags = 0;

    hrx_status_t status =
        hrx_graph_add_kernel_node(graph_, deps, dep_count, &attrs, out_node);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      return false;
    }
    return true;
  }

  bool AddFillBufferNode(hrx_buffer_ref_t dst, uint32_t pattern,
                         size_t pattern_size,
                         const hrx_graph_node_t* deps = nullptr,
                         size_t dep_count = 0,
                         hrx_graph_node_t* out_node = nullptr) {
    if (graph_ == nullptr) {
      return false;
    }
    hrx_graph_fill_buffer_node_attrs_t attrs{};
    attrs.dst = dst;
    attrs.pattern = pattern;
    attrs.pattern_size = pattern_size;

    hrx_status_t status = hrx_graph_add_fill_buffer_node(
        graph_, deps, dep_count, &attrs, out_node);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      return false;
    }
    return true;
  }

  bool Instantiate() {
    if (graph_ == nullptr) {
      return false;
    }
    hrx_status_t status = hrx_graph_instantiate(graph_, 0, &exec_instance_);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      return false;
    }
    is_instantiated_ = true;
    return true;
  }

  bool Launch(hrx_stream_t stream) {
    if (!is_instantiated_ || exec_instance_ == nullptr) {
      return false;
    }
    hrx_status_t status = hrx_graph_exec_launch(exec_instance_, stream);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      return false;
    }
    return true;
  }

private:
  hrx_graph_t graph_{nullptr};
  hrx_graph_exec_t exec_instance_{nullptr};
  std::optional<HrxGraphCaptureKey> capture_key_{};
  bool is_enabled_{true};
  bool is_capturing_{false};
  bool is_instantiated_{false};
};

}  // namespace gufo::hrx

#endif  // GUFO_CORE_HRX_HRX_GRAPH_EXECUTOR_HPP_
