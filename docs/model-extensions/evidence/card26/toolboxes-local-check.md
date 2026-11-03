# Local Toolboxes verification

Checked `/home/mixer/toolboxes` at `62d93f839ac3341e5228dbe64e14c4a6ddd27cca`
on 2026-11-02. Its working copy was clean and was not modified.
`nix flake check --no-build --offline` passed all derivation/app evaluations
(with missing app `meta` warnings); this is evaluation, not a new image build.
`bash -n deb/build-deb deb/verify-deb refresh-toolboxes.sh` passed.
Debian and Windows packages were not rebuilt in this check.

Published images were pulled and executed by digest:

- Runtime: `sha256:92ba47700a016a645775ecb61e228a782285ccaec5bbce94446d610ad5c9b67e`.
- Development: `sha256:c3bf4c5d03cd46e0943f38eeff281c54ab2eca568326bda1089cf2a12bcbe662`.

Runtime checks used `docker run --rm --network none --device /dev/kfd
--device /dev/dri --group-add 303 --ulimit memlock=-1 IMAGE /bin/bash -c ...`.
The commands were `gufo --version`, `gufo diagnose --section extensions --json`
and `gufo diagnose --json`. [Output](toolboxes-local-runtime-check.txt) confirms
Gufo `2e6aa96`, API 0.1.2, discovery mode and a passing gfx1151/ROCm 10.1 GPU
diagnostic. The extension inventory is empty at this historical engine pin;
this does not imply that legacy model executors are absent.
[Installed files](toolboxes-local-runtime-installed-files.txt) confirm the
root licenses, DeepSeek license and H3 source manifest exist.
An initial [asset probe](toolboxes-local-runtime-assets.txt) stopped at
`command -v ffmpeg`; subsequent explicit file checks completed. This check does
not establish video subprocess availability.

The development image exposes CMake, Ninja, hipcc and Git, but no `cc`, `c++`,
`gcc`, `g++`, `clang`, `clang++`, `pkg-config` or `python3` on PATH.
[Plain source configuration](toolboxes-local-dev-configure.txt) fails because
CMake cannot find C/C++ compilers. Selecting `$ROCM_PATH/llvm/bin/clang` and
`clang++` explicitly gets past compiler checks but
[fails at ICU development dependency discovery](toolboxes-local-dev-configure-explicit.txt).
Commands used `cmake -S /src -B /tmp/gufo-build -G Ninja
-DENGINE_ENABLE_HIP=OFF -DBUILD_TESTING=OFF -DGUFO_BUILD_TOOLS=OFF`, with the pinned
Gufo source mounted read-only; the second invocation also set both compiler
paths. No full Gufo or external-extension build completed in this image.

[hipcc smoke](toolboxes-local-dev-compiler.txt) succeeds compiling and running
`int main() { return 0; }` with `hipcc --offload-arch=gfx1151`; it establishes
compiler invocation/linking, not GPU kernel execution.

A [prompt attempt](toolboxes-local-runtime-prompt.txt) rejected a mistakenly
selected DFlash draft model (`layers=5`) as a main model. This is not evidence
of an inference regression or a successful inference smoke. No full model
quality or timing suite was run.

## Findings

1. Toolboxes selects `upstream.rocm10 or upstream.default`. Its release variant
   only overrides `releaseVersion`; neither variant requests
   `extensionsFromLock = true`. The runtime confirms discovery mode. This fails
   Card 26's edge lock-mode requirement.
2. The published development image is suitable for the small standalone hipcc
   smoke but does not provide a complete Gufo/extension source build environment.
   Compiler discovery and development dependencies need integration before its
   extension-builder contract can be qualified.

These results supersede the earlier statement that neither image had been
executed. Card 26 remains pending for the lock-mode runtime, a successful
published-builder extension build and native Windows validation.
