# XDNA NPU runtime on XRT

Experimental NPU runtime (`include/gufo/ext/experimental/npu.hpp`, namespace
`gufo::npu`) for runners that execute mlir-aie (IRON) designs on the AMD XDNA
NPU:
- `ProbeDevices()`: the NPUs present (`DeviceInfo`), from the `amdxdna` accel
  nodes in sysfs on Linux and from XRT on Windows.
- `Device::Open(index)`: opens an XRT device; throws `npu::Error` without a
  usable NPU.
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
- `Q8Gemm` (`src/core/npu/q8_gemm.hpp`, included by the same header): Q8_0
  matrix products `y = W x` for any row count on the NPU. `Open(device, dir)`
  loads the IRON design in `src/core/npu/kernels/q8gemm` (`q8gemm.xclbin`,
  `q8gemm_dispatch.so`); `Upload` packs a matrix into the weight arena; `Run`
  executes one or several products in one dispatch. Rows are quantized to
  int16 per row and block scales are applied in integer arithmetic, so results
  match f32 activations to about 5e-5.

Speech and decision providers receive their declared options, so an NPU
switch such as `--npu` reaches them as it reaches text providers:
- `AsrRunnerOptions::provider_options` and `TtsRunnerOptions::provider_options`
  (`ParsedOptions`): the explicit values of the options the provider declares
  in its manifest for `serve asr` / `serve tts` (manifest commands `serve` or
  the full subcommand). `gufo serve asr` and `gufo serve tts --help` list them
  under "Provider options". Runners built through `create_with_options` get
  them; an option that was not given is absent.
- `DecisionRunnerOptions::provider_options`: likewise for `serve decision` and
  `decide`, whose `--help` lists them.

Public CMake contract: Gufo requires XRT (`XRT::xrt_coreutil`), found through
`XRT_DIR`, `XILINX_XRT` or `/opt/xilinx/xrt`.

Compatibility: additive. Existing extensions need no change (the new fields
default to empty). The header names
no XRT type, so extensions build without XRT headers.
