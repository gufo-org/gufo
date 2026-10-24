# Building Gufo (Windows / Strix Halo)

Windows x86-64 is a supported native build target sharing the same source
tree as the Linux production build. The device toolchain differs: the HIP
compiler comes from a TheRock SDK rather than Visual Studio. Production
packages are still cut from the Linux build.

## 1. Prerequisites

- **Hardware/driver:** AMD Strix Halo (gfx1151) with a recent AMD driver
  (current Adrenalin/PRO drivers work). GPU-less hosted checks skip this.
- **Visual Studio 2022/18** with the "Desktop development with C++"
  workload (MSVC x64 — the device compiler comes from TheRock below, not
  VS).
- **Git**, **CMake ≥ 3.29**, **Ninja** (VS bundles both; `vs-env.cmd` puts
  them on `PATH`), **Python 3**.
- **vcpkg**, with `VCPKG_ROOT` set to its directory:

  ```cmd
  git clone https://github.com/microsoft/vcpkg.git C:\vcpkg && C:\vcpkg\bootstrap-vcpkg.bat
  setx VCPKG_ROOT C:\vcpkg
  ```

- **TheRock HIP SDK** (the AMD ROCm development distribution built by the
  TheRock project) — a **Windows nightly distribution for gfx1151**,
  extracted anywhere, with `HIP_PATH` set to that directory (for example
  `setx HIP_PATH C:\TheRock\build`). The `windows-base` preset chain-loads
  `cmake/toolchains/windows-therock.cmake`, which takes `clang-cl` and the
  runtime from `HIP_PATH`. Use a **nightly** build — the stable ROCm 7.x SDK runtimes currently crash
  (`hipStreamCreateWithFlags` hangs/segfaults) with shipping drivers.

## 2. Patch the SDK's HIP headers (once, and after every SDK reinstall)

MSVC 14.40+ conflicts with Clang's HIP math headers. The repo ships an
idempotent fixer:

```cmd
python tools\windows\patch_sdk_math.py %HIP_PATH%
```

## 3. Configure + build

```cmd
tools\windows\vs-env.cmd cmake --preset windows-release
tools\windows\vs-env.cmd cmake --build --preset windows-release
```

(`vs-env.cmd` finds Visual Studio with `vswhere`. First configure downloads
dependencies via vcpkg. The build itself is a few minutes and produces
`build\windows-release\gufo.exe` plus the staged ROCm runtime DLLs and the
`rocblas\` / `hipblaslt\` kernel-library directories.)

For the hosted test suite without an AMD GPU, use the CPU preset instead:

```cmd
tools\windows\vs-env.cmd cmake --preset windows-cpu-test
tools\windows\vs-env.cmd cmake --build --preset windows-cpu-test
tools\windows\vs-env.cmd ctest --preset windows-cpu-test --output-on-failure
```

Some hosted suites stay Linux-only (they simulate `RLIMIT_FSIZE`, fork
themselves, or fault libc symbols); their targets are gated off on Windows
and run by the Linux CI.

**The binaries must stay in the build directory** — the staged runtime
DLLs are resolved relative to the executable, which must precede System32
on the DLL search path.

## 4. Verify

```cmd
build\windows-release\gufo.exe diagnose
```

Then run a real model (GGUF from Hugging Face, e.g. an unsloth Q4_K_XL
build):

```cmd
build\windows-release\gufo.exe bench -p 2048 -n 128 <path\to\model.gguf>
```

To serve:

```cmd
build\windows-release\gufo.exe serve --port 8080 llm --model <path\to\model.gguf> --context 4096
```

Then `curl http://127.0.0.1:8080/ready` and POST to `/v1/chat/completions`.

## Gotchas

- If linking fails with `failed to write output 'gufo.exe': permission
  denied`, a `gufo serve` process is still running — kill it and rebuild.
- Don't move or copy `gufo.exe` out of the build directory on its own;
  take the whole folder (`gufo.exe`, the DLLs, and the `rocblas\` and
  `hipblaslt\` directories).
- If you reinstall or upgrade the TheRock SDK, re-run steps 2's patch and
  rebuild into a fresh folder so the runtime stages again.
