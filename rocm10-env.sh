# Spike environment: gufo built against the ROCm 10.0.0 TheRock gfx1151 dist.
# Sourced inside `nix develop`, which supplies host deps (gcc, cmake, ninja,
# ICU, CURL, python+triton, ffmpeg, aotriton) but ALSO injects ROCm 7.2.3
# -isystem paths. Those are stripped here so only ROCm 10 headers are visible.
S=/tmp/claude-1000/-home-mixer-gufo/82572f43-a7d3-4590-ab78-e6037443feed/scratchpad
R=$S/rocm10
AOT=/nix/store/zxjf0y681b6kfdrak9bfxjw0n6ybizzp-aotriton-0.11.1b

# Drop every -isystem pair that points at a ROCm 7.2.3 store path.
filter_rocm7() {
  printf '%s\n' $1 | paste -d' ' - - | grep -v -E \
    'clr-7\.2\.3|rocm-core-7\.2\.3|rocm-runtime-7\.2\.3|rocm-comgr|rocprofiler-sdk-7\.2\.3|hipblas|rocblas|miopen|composable[-_]kernel|rocwmma|hipcub|rocprim|hip-common|hipsolver|hipsparse' \
    | tr '\n' ' '
}
export NIX_CFLAGS_COMPILE="$(filter_rocm7 "$NIX_CFLAGS_COMPILE")"

# Same for the link line: nix develop adds -L/-rpath into clr-7.2.3, which also
# ships libamdhip64.so.7 and would otherwise win over the ROCm 10 runtime.
# Drops "-L<rocm7path>" tokens and "-rpath <rocm7path>" pairs together, so the
# stripped link line never leaves a dangling -rpath that swallows the next flag.
export NIX_LDFLAGS="$(printf '%s\n' $NIX_LDFLAGS | awk '
  BEGIN { skip = 0 }
  {
    if (skip) { skip = 0; next }
    if ($0 == "-rpath") { getline nxt; if (nxt ~ /7\.2\.3|rocm-comgr/) next; print $0; print nxt; next }
    if ($0 ~ /7\.2\.3|rocm-comgr/) next
    print $0
  }' | tr '\n' ' ')"
export NIX_LDFLAGS="-L$R/lib -rpath $R/lib $NIX_LDFLAGS"

# glibc headers for the unwrapped ROCm 10 amdclang++ device compiler.
GLIBC_INC=$(echo | g++ -E -Wp,-v - 2>&1 | grep -o '/nix/store/[^ ]*glibc[^ ]*-dev/include' | head -1)
# The nix g++ is a wrapper script, so resolve the real gcc prefix through the
# LTO wrapper path and swap libexec -> lib to get clang's --gcc-install-dir.
GCC_LTO=$(g++ -v 2>&1 | grep COLLECT_LTO_WRAPPER | cut -d= -f2)
GCC_INSTALL_DIR=$(dirname "$GCC_LTO" | sed 's|/libexec/gcc/|/lib/gcc/|')

export ROCM_PATH=$R
export GUFO_HIPCUB_ROOT=$R
export GUFO_ROCPRIM_ROOT=$R
export GUFO_ROCWMMA_ROOT=$R
export PATH=$R/bin:$R/llvm/bin:$PATH
# Link inputs too: the unwrapped amdclang++ has no nix crt/library search path.
GLIBC_LIB=$(dirname "$(g++ -print-file-name=Scrt1.o)")
GCC_LIB=$(dirname "$(g++ -print-file-name=libstdc++.so)")
export HIP_FLAGS="--gcc-install-dir=${GCC_INSTALL_DIR%/} -idirafter ${GLIBC_INC} -B${GLIBC_LIB} -L${GLIBC_LIB} -L${GCC_LIB} -Wl,-rpath,${GCC_LIB} -Wl,-rpath,${GLIBC_LIB}"
export AOT R S
