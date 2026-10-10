{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  symlinkJoin,
  git,
  ninja,
  pkg-config,
  clr,
  rocm-cmake,
  rocm-runtime,
  rocprofiler-register,
  rocprof-trace-decoder,
  aqlprofile,
  rocm-comgr,
  rocmUpdateScript,
  python3,
  python3Packages,
  libdrm,
  elfutils,
  sqlite,
  otf2,
  zlib,
  zstd,
  xz,
  numactl,
  fmt,
  glog,
  gtest,
  fetchpatch,
  yaml-cpp,
  elfio,
  nlohmann_json,
  makeWrapper,
  gpuTargets ? (clr.localGpuTargets or clr.gpuTargets),
  buildTests ? false,
  buildSamples ? false,
}:

# FIXME: devendor remaining git submodules:
#   external/cereal     -> https://github.com/jrmadsen/cereal
#   external/gotcha     -> https://github.com/jrmadsen/GOTCHA
#   external/perfetto   -> https://github.com/google/perfetto
#   external/ptl        -> https://github.com/jrmadsen/PTL

# rocprofiler-sdk is the home of rocprofv3
let
  # ROCm 10 folded aqlprofile in (pm4_factory.cpp needs <libdrm/amdgpu_drm.h>)
  # and added the rocpd SQL layer (needs <sqlite3.h>). Neither subproject picks
  # up the wrapper's include paths, and cmakeFlags is a structuredAttrs array so
  # two -I flags cannot share one entry: merge both dev trees into one prefix.
  extraIncludes = symlinkJoin {
    name = "rocprofiler-sdk-extra-includes";
    paths = [
      (lib.getDev libdrm)
      (lib.getDev sqlite)
    ];
  };
in
stdenv.mkDerivation (finalAttrs: {
  pname = "rocprofiler-sdk";
  version = "10.0.0";

  outputs = [
    "out"
    "dev"
  ];

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "rocm-systems";
    rev = "therock-10.0";
    hash = "sha256-N83wTDI6R66pOnDdxYIS9SZy64o1fyHzj7awRJKXNp0=";
    fetchSubmodules = true;
    sparseCheckout = [
      "projects/rocprofiler-sdk"
    ];
  };
  sourceRoot = "${finalAttrs.src.name}/projects/rocprofiler-sdk";

  strictDeps = true;

  nativeBuildInputs = [
    # ROCm 10's rocprofiler-sdk-tool does find_package(Git REQUIRED)
    git
    cmake
    ninja
    pkg-config
    clr
    python3
    makeWrapper
  ];

  buildInputs = [
    clr
    rocm-cmake
    rocm-runtime
    rocprofiler-register
    # ROCm 10's rocprofiler-sdk-tool does find_package(rocprof-trace-decoder REQUIRED)
    rocprof-trace-decoder
    aqlprofile
    rocm-comgr
    libdrm
    # ROCm 10 folded aqlprofile in (needs <libdrm/amdgpu_drm.h>) and added the
    # rocpd SQL layer (needs <sqlite3.h>); both live in the dev outputs.
    (lib.getDev libdrm)
    (lib.getDev sqlite)
    elfutils
    sqlite
    otf2
    zlib
    zstd
    xz
    numactl
    fmt
    glog
    gtest
    yaml-cpp
    elfio
    nlohmann_json
    python3Packages.pybind11
  ];

  # The three upstream commits nixpkgs backports (system libraries, fmt memory
  # header, comgr linkage in the pc-sampling test) are already in therock-10.0.
  # The local PR 4721 series is still required: without it the build falls back
  # to FetchContent and tries to download elfio/otf2/json during the build.
  patches = [
    # 0001 only renamed the consumer test so the rest of the series applied to
    # 7.2.3; therock-10.0 already carries that rename.
    # 0002 only added <array>/<memory>/<fstream>/<iomanip> includes that
    # therock-10.0 already has.
    ./0003-rocprofiler-sdk-add-missing-rocprofiler-sdk-rocprofi.patch
    ./0004-rocprofiler-sdk-fix-find_package-dependency-scoping-.patch
    ./0005-rocprofiler-sdk-allow-using-system-elfio-dependency.patch
    ./0006-rocprofiler-sdk-allow-using-system-otf2-dependency.patch
    ./0007-rocprofiler-sdk-Allow-using-system-json-dependency.patch
    # 0008 only touched test targets, so nothing of it survives with tests off.
  ];

  postPatch = ''
    # ROCm 10 still misses these STL includes (nixpkgs' 0002 patch no longer
    # applies as a whole, but these three files genuinely need them).
    sed -i '0,/#include/s//#include <fstream>\n#include/' \
      source/lib/rocprofiler-sdk-rocpd/sql.cpp
    sed -i '0,/#include/s//#include <fstream>\n#include/' \
      source/lib/rocprofiler-sdk/counters/metrics.cpp
    sed -i '0,/#include/s//#include <array>\n#include <memory>\n#include/' \
      source/lib/att-tool/att_lib_wrapper.hpp
    sed -i '0,/#include/s//#include <fstream>\n#include/' \
      source/lib/rocprofiler-sdk-rocattach/auxv.cpp
    # NixOS' ROCm distribution does not support libomptarget yet
    substituteInPlace samples/CMakeLists.txt \
      --replace-fail 'add_subdirectory(openmp_target)' '# add_subdirectory(openmp_target)'
    substituteInPlace tests/CMakeLists.txt \
      --replace-fail 'add_subdirectory(openmp-tools)' '# add_subdirectory(openmp-tools)'
    substituteInPlace tests/bin/CMakeLists.txt \
      --replace-fail 'add_subdirectory(openmp)' '# add_subdirectory(openmp)'

    # This requires building perfetto's trace-processor-shell via a weird external script
    substituteInPlace tests/CMakeLists.txt \
      --replace-fail 'add_subdirectory(pytest-packages)' ""

    patchShebangs source/libexec/rocprofiler-sdk/rocprofiler-sdk-launch-compiler/rocprofiler-sdk-launch-compiler.sh
  '';

  cmakeFlags = [
    (lib.cmakeBool "ROCPROFILER_BUILD_GHC_FS" false)
    (lib.cmakeBool "ROCPROFILER_BUILD_FMT" false)
    (lib.cmakeBool "ROCPROFILER_BUILD_GLOG" false)
    (lib.cmakeBool "ROCPROFILER_BUILD_GTEST" false)
    (lib.cmakeBool "ROCPROFILER_BUILD_PYBIND11" false)
    (lib.cmakeBool "ROCPROFILER_BUILD_YAML_CPP" false)
    (lib.cmakeBool "ROCPROFILER_BUILD_ELFIO" false)
    (lib.cmakeBool "ROCPROFILER_BUILD_OTF2" false)
    (lib.cmakeBool "ROCPROFILER_BUILD_JSON" false)
    (lib.cmakeBool "ROCPROFILER_BUILD_TESTS" buildTests)
    (lib.cmakeBool "ROCPROFILER_BUILD_SAMPLES" buildSamples)
    (lib.cmakeBool "ROCPROFILER_BUILD_BENCHMARK" false)
    (lib.cmakeBool "ROCPROFILER_BUILD_WERROR" false)
    # rocprofiler-sdk's CMake file doesn't add this dependency properly.
    "-DCMAKE_HIP_FLAGS=-I${rocm-runtime}/include"
    "-DCMAKE_CXX_FLAGS=-I${extraIncludes}/include"
    "-DCMAKE_INSTALL_BINDIR=bin"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_INSTALL_INCLUDEDIR=include"
    # Required for rocprofiler-sdk-launch-compiler.sh
    "-DCMAKE_INSTALL_LIBEXECDIR=libexec"
    "--debug-find-pkg=amd_comgr"
    "--trace-source=tests/pc_sampling/CMakeLists.txt"
    "--trace-expand"
  ]
  ++ lib.optionals (buildTests || buildSamples) [
    # rocprofiler-sdk normally doesn't depend on which GPU is in the system, only when
    # building tests or samples.
    (lib.cmakeFeature "GPU_TARGETS" (lib.concatStringsSep ";" gpuTargets))
  ];

  doCheck = false; # Requires GPU

  postFixup = ''
    patchelf $out/lib/*.so \
      --add-rpath ${aqlprofile}/lib \
      --add-needed libhsa-amd-aqlprofile64.so

    wrapProgram $out/bin/rocprofv3 --add-flags "--att-library-path ${
      lib.makeLibraryPath [ rocprof-trace-decoder ]
    }"
  '';

  postInstall = ''
    mkdir -p $dev/lib $dev/share/rocprofiler-sdk
    mv $out/lib/cmake $dev/lib/
    mv $out/share/rocprofiler-sdk/{samples,tests} $dev/share/rocprofiler-sdk/
  '';

  passthru.updateScript = rocmUpdateScript { inherit finalAttrs; };

  meta = {
    description = "ROCm GPU performance analysis SDK";
    homepage = "https://github.com/ROCm/rocprofiler-sdk";
    license = lib.licenses.mit;
    platforms = lib.platforms.linux;
  };
})
