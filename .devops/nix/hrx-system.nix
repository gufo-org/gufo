{
  lib,
  stdenv,
  fetchFromGitHub,
  fetchzip,
  cmake,
  git,
  ninja,
  pkg-config,
  python3,
  libbacktrace,
  rocmPackages,
  zstd,
}:

let
  # Fixed-output dependency sources pinned in hrx-system MODULE.cmake.lock
  flatcc-src = fetchzip {
    url = "https://github.com/dvidelabs/flatcc/archive/9362cd00f0007d8cbee7bff86e90fb4b6b227ff3.tar.gz";
    sha256 = "sha256-umZ9TvNYDZtF/mNwQUGuhAGve0kPw7uXkaaQX0EzkBY=";
  };

  hsa-headers-src = fetchzip {
    url = "https://github.com/iree-org/hsa-runtime-headers/archive/4285513114a70f7cf4830c89279c8cfa57b901bb.tar.gz";
    sha256 = "sha256-Px1Erkg6T6z9KTuL3fc9dYR28zvuebxe7wgLfRIPhEQ=";
  };

  hip-headers-src = fetchzip {
    url = "https://github.com/iree-org/hip-build-deps/archive/c64ec391ab8eaf4c9872660b24a71fda8026b438.tar.gz";
    sha256 = "sha256-L571fRWGoH+ezaR/3JvvKIKfe1azG4ajerC/b4jVnKs=";
  };

  isa-xml-src = fetchzip {
    url = "https://gpuopen.com/download/AMD_GPU_MR_ISA_XML_2026_03_05.zip";
    sha256 = "sha256-yFuzMB6Wh2uVRNGpUIc5lTX7UnK9V+y4HPHnCWKYiE0=";
    stripRoot = false;
  };

in
stdenv.mkDerivation {
  pname = "hrx-system";
  version = "unstable-2026-08";

  # Pinned to the last revision before "[HAL/AMDGPU] Model ROCr AQL queue
  # execution modes" (0cc34d04, 2026-08-27), which queries
  # HSA_AMD_AGENT_INFO_PM4_EMULATION. The ROCr in rocmPackages 7.2.3 rejects
  # that attribute with HSA_STATUS_ERROR_INVALID_ARGUMENT, so the AMDGPU
  # accelerator comes up unavailable and every HRX backend init fails. Revisit
  # when the ROCm pin moves.
  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "hrx-system";
    rev = "bce2ba3789b3ac5e1843df99ab9ca25f2681f0a6";
    hash = "sha256-ZfOXRTQwwVIpuJf0dSj/U1/PC1FeYcpni0lkJaJH/Bw=";
  };

  nativeBuildInputs = [
    cmake
    # Upstream's locked-dependency helper hard-fails at configure time if any
    # dependency declares patches and git is absent, even when the source dir
    # is overridden and the patch step never runs. flatcc's only locked patch
    # is an MSVC restrict-qualifier fix, which this platform never applies.
    git
    ninja
    pkg-config
    python3
    rocmPackages.llvm.clang
  ];

  buildInputs = [
    libbacktrace
    rocmPackages.clr
    rocmPackages.rocm-runtime
    rocmPackages.aqlprofile
    rocmPackages.rocm-device-libs
    zstd
  ];

  preConfigure = ''
    mkdir -p build/_deps/amdgpu_isa_xml-src
    cp -r ${isa-xml-src}/* build/_deps/amdgpu_isa_xml-src/
  '';

  cmakeFlags = [
    "-DCMAKE_C_COMPILER=${rocmPackages.llvm.clang}/bin/clang"
    "-DCMAKE_CXX_COMPILER=${rocmPackages.llvm.clang}/bin/clang++"
    "-DFETCHCONTENT_SOURCE_DIR_FLATCC=${flatcc-src}"
    "-DFETCHCONTENT_SOURCE_DIR_HSA_RUNTIME_HEADERS=${hsa-headers-src}"
    "-DFETCHCONTENT_SOURCE_DIR_HIP_API_HEADERS=${hip-headers-src}"
    "-DFETCHCONTENT_SOURCE_DIR_LOOM_AMDGPU_ISA_XML=${isa-xml-src}"
    "-DIREE_DEPENDENCY_MODE=auto"
    "-DIREE_ROCM_DEPENDENCY_MODE=pinned"
    "-DCMAKE_PREFIX_PATH=${rocmPackages.aqlprofile};${zstd.dev or zstd}"
    "-DIREE_ROCM_PATH=${rocmPackages.clr}"
    "-DIREE_BUILD_TESTS=OFF"
    "-DIREE_BUILD_BENCHMARKS=OFF"
    "-DIREE_HAL_DRIVER_AMDGPU=ON"
    "-DIREE_HAL_DRIVER_HIP=ON"
    "-DLIBHRX_BUILD=ON"
    "-DLOOM_BUILD=ON"
    "-DLOOM_TARGET_AMDGPU=ON"
    "-DLOOM_TARGET_AMDGPU_TARGETS=gfx1151"
    "-DLOOM_TARGET_SPIRV=OFF"
    "-DLOOM_TARGET_WASM=OFF"
    "-DLOOM_TARGET_X86=ON"
    "-DLOOM_TARGET_IREE_VM=ON"
    "-DLOOM_TARGET_LLVMIR=ON"
    "-DCMAKE_INSTALL_RPATH=${lib.makeLibraryPath [ rocmPackages.rocm-runtime rocmPackages.clr zstd ]}"
    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
  ];

  postFixup = ''
    for f in $out/lib/*.so* $out/bin/*; do
      if [ -f "$f" ] && patchelf --print-rpath "$f" >/dev/null 2>&1; then
        patchelf --add-rpath "${lib.makeLibraryPath [ rocmPackages.rocm-runtime rocmPackages.clr zstd ]}" "$f"
      fi
    done
  '';

  meta = with lib; {
    description = "HRX (Hip Runtime Extended) system runtime components and AMDGPU/HIP driver";
    homepage = "https://github.com/ROCm/hrx-system";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
