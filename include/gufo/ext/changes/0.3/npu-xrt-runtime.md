# XDNA NPU runtime on XRT

Experimental NPU runtime (`include/gufo/ext/experimental/npu.hpp`, namespace
`gufo::npu`) for runners that execute mlir-aie (IRON) designs on the AMD XDNA
NPU:
- `ProbeDevices()`: the `amdxdna` accel nodes in sysfs (`DeviceInfo`), without
  XRT or device access. `RuntimeAvailable()`: whether the build has XRT.
- `Device::Open(index)`: opens an XRT device; throws `npu::Error` without XRT
  or a usable NPU.
- `Program::Load(device, xclbin, kernel_prefix)`: one xclbin as a hardware
  context and its kernel. `Allocate`/`LoadBuffer` create `Buffer`s in a kernel
  argument's memory group, `AllocateInstructions`/`LoadInstructions` create
  `Instructions`, and `Start` begins an asynchronous `Run` with mlir-aie's
  opcode-3 convention (data buffers are kernel arguments 3 and up).
- `Buffer`: a mapped buffer object shared by the device's programs, with ranged
  `ToDevice`/`FromDevice` synchronization.
- `Instructions::Assign`/`Apply` and `DiffInstructions`: replace a stream, or
  patch words that change linearly with a run-time value.
- `DispatchLibrary`: loads an IRON design's `dispatch_generate` library (up to
  16 int32 scalars) and generates its instruction streams.

Public CMake contract: `GUFO_ENABLE_XRT` (Linux, default `OFF`) builds the
runtime on XRT's `XRT::xrt_coreutil`. Without it the API is unchanged, and
`Device::Open` and `Program::Load` throw `npu::Error`.

Compatibility: additive. Existing extensions need no change. The header names
no XRT type, so extensions build without XRT headers.
