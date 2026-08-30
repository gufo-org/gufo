#include "src/core/hrx/hrx_backend.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/hrx/hrx_graph_executor.hpp"
#include "src/core/hrx/hrx_module_loader.hpp"
#include "src/core/hrx/hrx_owned_buffer.hpp"
#include "src/core/hrx/hrx_utils.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

void TestHrxBackendLifecycle() {
  auto& backend = gufo::hrx::HrxBackend::Instance();
  bool ok = backend.Initialize(0);
  Expect(ok, "HrxBackend::Initialize succeeds on device 0");
  Expect(backend.IsInitialized(), "HrxBackend::IsInitialized is true");
  Expect(backend.Device() != nullptr, "HrxBackend::Device is non-null");
  Expect(backend.Stream() != nullptr, "HrxBackend::Stream is non-null");

  backend.Shutdown();
  Expect(!backend.IsInitialized(), "HrxBackend::Shutdown cleans up state");
}

void TestHrxOwnedBuffer() {
  auto& backend = gufo::hrx::HrxBackend::Instance();
  Expect(backend.Initialize(0), "Backend initialize before owned buffer test");

  std::string error;
  Expect(!gufo::hrx::HrxOwnedBuffer::Allocate(backend.Stream(), 0, &error)
              .has_value(),
         "zero-length HRX allocation is rejected");
  Expect(!error.empty(), "rejected HRX allocation reports an error");

  auto allocation =
      gufo::hrx::HrxOwnedBuffer::Allocate(backend.Stream(), 64, &error);
  Expect(allocation.has_value(), "owned HRX allocation succeeds");
  Expect(allocation->IsValid(), "owned HRX allocation is valid");
  Expect(allocation->Size() == 64, "owned HRX allocation records size");
  Expect(allocation->Slice(16, 32).has_value(),
         "in-bounds HRX slice is accepted");
  Expect(!allocation->Slice(48, 32).has_value(),
         "out-of-bounds HRX slice is rejected");

  const std::array<std::uint32_t, 4> input{1, 2, 3, 4};
  std::array<std::uint32_t, 4> output{};
  const auto slice = allocation->Slice(16, sizeof(input));
  Expect(slice.has_value(), "copy test slice is valid");
  Expect(gufo::hrx::HrxCopyFromHost(backend.Device(), input.data(), *slice,
                                    sizeof(input), &error),
         "checked HRX H2D copy succeeds");
  Expect(gufo::hrx::HrxCopyToHost(backend.Device(), *slice, output.data(),
                                  sizeof(output), &error),
         "checked HRX D2H copy succeeds");
  Expect(input == output, "checked HRX copy roundtrip preserves data");
  Expect(setenv("GUFO_ENABLE_HRX_GRAPH", "0", 1) == 0,
         "optional graph kill switch is set for mandatory fill test");
  Expect(gufo::hrx::HrxFillBuffer(backend.Device(), backend.Stream(), *slice, 0,
                                  &error),
         "mandatory native HRX fill bypasses optional graph kill switch");
  Expect(unsetenv("GUFO_ENABLE_HRX_GRAPH") == 0,
         "optional graph kill switch is cleared after fill test");
  Expect(gufo::hrx::HrxCopyToHost(backend.Device(), *slice, output.data(),
                                  sizeof(output), &error),
         "filled HRX slice can be read back");
  Expect(output == std::array<std::uint32_t, 4>{},
         "native HRX graph fill zeroes the requested slice");

  auto moved = std::move(*allocation);
  Expect(moved.IsValid(), "moved HRX owner retains allocation");
  Expect(!allocation->IsValid(), "moved-from HRX owner is empty");
  Expect(gufo::hrx::kHrxNativeDeviceCopyAvailable,
         "artifact-backed native device copy is explicit");
  Expect(gufo::hrx::kHrxNativeDeviceFillAvailable,
         "graph-backed native device fill is available");
  moved.Reset();
  backend.Shutdown();
}

void TestHrxModuleLoader() {
  auto& backend = gufo::hrx::HrxBackend::Instance();
  Expect(backend.Initialize(0), "Backend initialize before loader");

  gufo::hrx::HrxModuleLoader loader;
  hrx_status_t status =
      loader.LoadFromFile(backend.Device(), "qwen_swiglu",
                          "/tmp/qwen_swiglu.fb", "amdgpu", "gfx1151");

  if (hrx_status_is_ok(status)) {
    hrx_executable_t exec = loader.GetExecutable("qwen_swiglu");
    Expect(exec != nullptr, "Loaded executable is non-null");
  } else {
    hrx_status_ignore(status);
  }

  loader.UnloadAll();
  backend.Shutdown();
}

void TestHrxGraphExecutor() {
  auto& backend = gufo::hrx::HrxBackend::Instance();
  Expect(backend.Initialize(0), "Backend initialize before graph test");

  gufo::hrx::HrxGraphDecodeExecutor executor;
  Expect(executor.IsEnabled(), "HrxGraphDecodeExecutor is enabled by default");

  gufo::hrx::HrxGraphCaptureKey key{1234, 5678};
  bool graph_ok = executor.InitializeGraph(backend.Device(), key);
  Expect(graph_ok, "InitializeGraph creates graph");
  Expect(executor.Graph() != nullptr, "Graph handle is non-null");

  // Add an empty node
  hrx_graph_node_t empty_node = nullptr;
  bool empty_ok = executor.AddEmptyNode(nullptr, 0, &empty_node);
  Expect(empty_ok, "AddEmptyNode succeeds");
  Expect(empty_node != nullptr, "Empty node handle is valid");

  // Allocate a buffer and add a FillBuffer node dependent on empty_node
  hrx_buffer_params_t params = {
      HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_MEMORY_ACCESS_ALL,
      HRX_BUFFER_USAGE_DEFAULT,
      0,
  };
  hrx_buffer_t buf = nullptr;
  hrx_status_t alloc_st = hrx_allocator_allocate_buffer(
      hrx_device_allocator(backend.Device()), params, 1024, &buf);
  Expect(hrx_status_is_ok(alloc_st), "Alloc buffer for graph fill test");

  hrx_buffer_ref_t dst_ref = {buf, 0, 1024};
  uint32_t pattern = 0xCAFEFACE;
  hrx_graph_node_t fill_node = nullptr;
  bool fill_ok = executor.AddFillBufferNode(dst_ref, pattern, sizeof(pattern),
                                            &empty_node, 1, &fill_node);
  Expect(fill_ok, "AddFillBufferNode succeeds with dependency");
  Expect(fill_node != nullptr, "Fill node handle is valid");

  bool inst_ok = executor.Instantiate();
  Expect(inst_ok, "Instantiate succeeds on multi-node graph");
  Expect(executor.IsInstantiated(), "IsInstantiated is true");
  Expect(executor.IsInstantiatedFor(key), "IsInstantiatedFor matches key");

  bool launch_ok = executor.Launch(backend.Stream());
  Expect(launch_ok, "Launch executes graph on stream");
  HRX_CHECK(hrx_stream_synchronize(backend.Stream()));

  hrx_buffer_release(buf);
  executor.Reset();
  Expect(!executor.IsInstantiated(), "Reset clears instantiated state");
  backend.Shutdown();
}

}  // namespace

int main() {
  std::cout << "Running hrx_backend_test...\n";
  TestHrxBackendLifecycle();
  TestHrxOwnedBuffer();
  TestHrxModuleLoader();
  TestHrxGraphExecutor();
  std::cout << "All hrx_backend_test assertions passed!\n";
  return 0;
}
