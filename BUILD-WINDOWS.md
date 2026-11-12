# Building Gufo (Windows / Strix Halo)

Windows x86-64 is a supported native build target for development, sharing
the source tree of the Linux production build. Production packages are cut
from the Linux build. The device toolchain differs: `clang-cl` and the HIP
runtime come from a TheRock SDK, MSVC supplies the C++ standard library and
the linker, and vcpkg supplies the libraries.

## Downloaded Windows releases

Extract the entire release ZIP before running `gufo.exe`, keeping its DLLs and
kernel directories beside it. Bundles include an LLVM OpenMP runtime built
from upstream LLVM 18.1.8 source under Apache-2.0 with LLVM exceptions, using
`libomp140.x86_64.dll` as required by the SDK. The license is included under
`llvm-openmp-LICENSE.txt`, with source provenance in `llvm-openmp-version.txt`.
The NPU runtime uses XRT from AMD's NPU driver, which the bundle does not
ship; without the driver, `gufo diagnose` warns `No XDNA NPU found` and the
GPU paths work as before.

The v0.24.0 ZIP omitted this DLL ([#801](https://github.com/gufo-org/gufo/issues/801)).
If it exits silently, including for `--help`, with status `0xC0000135`, the
Windows loader could not find a DLL. Use a subsequent fixed Gufo release when
available. For a local development workaround, install Microsoft's Visual
Studio Build Tools with the C++ workload and run Gufo from its x64 Native
Tools command prompt. Microsoft documents the LLVM OpenMP DLL under
`VC\Redist\MSVC\<version>\debug_nonredist\x64\Microsoft.VC*.OpenMP.LLVM`;
if necessary, copy that installed x64 DLL beside `gufo.exe` for local use.
Do not obtain DLLs from third-party download sites or replace system DLLs.
The ordinary VC++ Redistributable does not supply this LLVM runtime.
Microsoft's DLL is [not redistributable](https://learn.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170);
release packaging builds the upstream runtime instead.

## 1. Prerequisites

- **Hardware/driver:** AMD Strix Halo (gfx1151) with a current AMD graphics
  driver. Hosted builds without a GPU need none.
- **Visual Studio 2022 or 18** with the "Desktop development with C++"
  workload (MSVC x64).
- **Git**, **CMake ≥ 3.29**, **Ninja** (Visual Studio bundles both;
  `vs-env.cmd` puts them on `PATH`) and **Python 3**.
- **Rust 1.88 or newer** from [rustup](https://rustup.rs), with the
  `x86_64-pc-windows-msvc` target. Rustup reads the pinned version from
  `rust-toolchain.toml`; Cargo builds the Rust components.
- **vcpkg**, a full clone (the manifest baseline resolves through its
  history), with `VCPKG_ROOT` set to it:

  ```cmd
  git clone https://github.com/microsoft/vcpkg.git C:\vcpkg && C:\vcpkg\bootstrap-vcpkg.bat
  setx VCPKG_ROOT C:\vcpkg
  ```

- **TheRock 10.0.0 for gfx1151**, `therock-dist-windows-gfx1151-10.0.0.tar.gz`
  (sha256 `1293927b06b3b8d4bd7e0265823fb998bc9e0d83c68f33dcfa5d32663b30ce38`),
  extracted anywhere, with `HIP_PATH` set to that directory:

  ```cmd
  mkdir C:\TheRock\build
  tar -xzf therock-dist-windows-gfx1151-10.0.0.tar.gz -C C:\TheRock\build
  setx HIP_PATH C:\TheRock\build
  ```

  AMD's index no longer serves 10.0.0; CI downloads it from the
  [Windows port's mirror](https://github.com/endyjasmi/gufo/releases/tag/toolchain/therock-10.0.0).
  It is the only SDK validated at full speed on gfx1151: later builds
  (10.1.0 through at least the 10.2.0a nightlies of October 2026) compile
  correct kernels that stall on this GPU, with decode up to hundreds of times
  slower. The `windows-base` preset chain-loads
  `cmake/toolchains/windows-therock.cmake`, which takes `clang-cl` and the
  runtime from `HIP_PATH`.
- **XRT Windows SDK 2.21**, Xilinx's
  [`xrt_windows_sdk.zip`](https://github.com/Xilinx/XRT/releases/download/2.21.75/xrt_windows_sdk.zip)
  (sha256 `ccc244c2c423588972ade76142cdc01049477aaa39a35be97e782b97eb7c5295`),
  extracted to `C:\xrt`, where configure finds it:

  ```cmd
  mkdir C:\xrt
  tar -xf xrt_windows_sdk.zip -C C:\xrt
  ```

  The NPU runtime compiles against its headers and import library. At run
  time Gufo uses the `xrt_coreutil.dll` that AMD's NPU driver installs in
  System32 (the driver ships XRT 2.21 too); without the driver it runs on the
  GPU only. Elsewhere, pass `-DXRT_DIR=<dir>/xrt_sdk/xrt/share/cmake/XRT`. Do
  not set `XILINX_XRT`: the driver's XRT then looks for its device plugin
  there and finds no NPU.

## 2. Patch the SDK's HIP headers (once, and after every SDK reinstall)

MSVC 14.40+ conflicts with Clang's HIP math headers. The repository ships an
idempotent fixer, and configure stops with a pointer to it while the SDK is
unpatched:

```cmd
python tools\windows\patch_sdk_math.py %HIP_PATH%
```

## 3. Configure and build

```cmd
tools\windows\vs-env.cmd cmake --preset windows-release
tools\windows\vs-env.cmd cmake --build --preset windows-release
```

`vs-env.cmd` finds Visual Studio with `vswhere`, loads the x64 environment and
runs its arguments. From another `.bat` file, invoke it with `call`, or the
calling script stops after it. The first configure builds the dependencies
through vcpkg (ICU and OpenSSL take several minutes). The build produces
`build\windows-release\gufo.exe` and `gufo-server.exe`, with the TheRock
runtime DLLs and the `rocblas\` and `hipblaslt\` kernel libraries staged
beside them.

For the host test suite, which needs no GPU:

```cmd
tools\windows\vs-env.cmd cmake --preset windows-cpu-test
tools\windows\vs-env.cmd cmake --build --preset windows-cpu-test --target check-pr
```

Some host suites stay Linux-only (they simulate `RLIMIT_FSIZE`, fork
themselves or fault libc symbols); their targets are off on Windows and the
Linux CI runs them.

**Faster rebuilds:** with `ccache` on `PATH`, CMake compiles C, C++ and HIP
through it. Configure with embedded debug information, because ccache cannot
cache objects that share a `/Zi` PDB:

```cmd
tools\windows\vs-env.cmd cmake --preset windows-release -DCMAKE_POLICY_DEFAULT_CMP0141=NEW -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded
```

## 4. Verify

```cmd
build\windows-release\gufo.exe diagnose
```

On a Strix Halo it reports `x86_64-windows`, the gfx1151 GPU and ROCm 10.0.0.
Then run a real model (a GGUF from Hugging Face, for example an unsloth
UD-Q4_K_XL build):

```cmd
build\windows-release\gufo.exe bench -p 2048 -n 128 <path\to\model.gguf>
build\windows-release\gufo.exe serve --port 8080 llm --model <path\to\model.gguf> --context 4096
```

Then `curl http://127.0.0.1:8080/ready` and POST to `/v1/chat/completions`.

## CI

[`.github/workflows/windows-build.yml`](.github/workflows/windows-build.yml) builds this
recipe on a hosted `windows-latest` runner, with the SDK archive, the vcpkg
packages and the compiler cache restored from the Actions cache, and runs
`check-pr`. It also builds the upstream OpenMP runtime, extracts the packaged
bundle, checks every EXE/DLL's normal and delay imports against the bundle and
an explicit Windows system DLL list, and runs `--help`, `--version` and platform
diagnosis with a restricted `PATH`. The import check does not use the builder's
installed DLLs, so developer tools cannot conceal an omitted dependency.
On pull requests and manual dispatches, the
[Windows validation workflow](.github/workflows/devlab-windows.yml) downloads
the resulting bundle on Windows Strix Halo and runs `gufo diagnose`, requiring
the NPU check to pass, weight-free GPU tests, and one small model per family
through
[`tests/functional/family_smoke.py`](tests/functional/family_smoke.py):
EmbeddingGemma 2, Qwen3-ASR, Qwen3-TTS and Qwen3.8-27B.

## Gotchas

- If linking fails with `failed to write output 'gufo.exe': permission
  denied`, a `gufo serve` process is still running: stop it and rebuild.
- Keep `gufo.exe` with its folder. The staged runtime DLLs resolve relative
  to the executable, ahead of the older `amdhip64` the AMD driver installs in
  System32; copy the whole folder (`gufo.exe`, the DLLs, `rocblas\` and
  `hipblaslt\`).
- If configure fails with `Could not find a package configuration file
  provided by "XRT"`, extract the XRT SDK to `C:\xrt` (step 1) or pass
  `-DXRT_DIR`.
- If `gufo diagnose` warns `No XDNA NPU found` on a machine with the NPU
  driver, unset `XILINX_XRT` and remove any `xrt_coreutil.dll` beside
  `gufo.exe`.
- After reinstalling or upgrading the SDK, rerun the patch in step 2 and
  rebuild into a fresh build directory so the runtime stages again.

## NPU device test (developers)

Building and running Gufo needs no Python for the NPU. Only compiling the
in-tree NPU designs does: they are IRON (mlir-aie) programs, compiled by
Python packages. Install the Windows wheels of the versions
[`.devops/nix/iron`](.devops/nix/iron/default.nix) pins, then build and run
`npu_device_test`; aiecc needs the SDK's `xclbinutil` on `PATH`:

```cmd
py -3.12 -m venv C:\src\iron-venv
C:\src\iron-venv\Scripts\python -m pip install numpy ml-dtypes cloudpickle https://github.com/Xilinx/mlir-aie/releases/download/v1.4.5/mlir_aie-1.4.5-cp312-cp312-win_amd64.whl https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026111101%2B81af12b7-py3-none-win_amd64.whl
set PEANO_INSTALL_DIR=C:\src\iron-venv\Lib\site-packages\llvm-aie
set PATH=C:\xrt\xrt_sdk\xrt;%PATH%
tools\windows\vs-env.cmd cmake --preset windows-release -B build\npu-test -DBUILD_TESTING=ON -DGUFO_IRON_PYTHON=C:\src\iron-venv\Scripts\python.exe
tools\windows\vs-env.cmd cmake --build build\npu-test --target npu_device_test
build\npu-test\npu_device_test.exe build\npu-test\npu\q8gemm
```

It prints each product's error relative to a double reference (about 3e-5).

## External source extensions

Use the [extension distribution recipe](docs/internal/extension-distribution.md#native-windows-and-zip-recipe)
for vcpkg dependency preflight, external source selection, CMake installation,
license/data staging and fresh-ZIP runtime checks. Declared metadata dependencies
do not install additional vcpkg ports automatically.
