{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  rocm-cmake,
  rocm-smi,
  pkg-config,
  clr,
  gfortran,
  git,
  gtest,
  boost,
  llvm,
  msgpack-cxx,
  amd-blis,
  libxml2,
  python3,
  python3Packages,
  openmp,
  hipblas-common,
  amdsmi,
  lapack-reference,
  ncurses,
  ninja,
  libffi,
  jemalloc,
  zlib,
  zstd,
  rocmUpdateScript,
  buildTests ? false,
  buildSamples ? false,
  # hipblaslt supports only devices with MFMA or WMMA
  gpuTargets ? (clr.localGpuTargets or clr.gpuTargets),
}:

let
  # hipblaslt is extremely particular about what it will build with
  # so intersect with a known supported list and use only those
  supportedTargets = (
    lib.lists.intersectLists gpuTargets [
      "gfx908"
      "gfx90a"
      "gfx942"
      "gfx950"
      "gfx1100"
      "gfx1101"
      "gfx1150"
      "gfx1151"
      "gfx1200"
      "gfx1201"
    ]
  );
  supportsTargetArches = supportedTargets != [ ];
  py = python3.withPackages (ps: [
    ps.pyyaml
    ps.setuptools
    ps.packaging
    ps.nanobind
    ps.msgpack
    # required by ROCm 10's tensilelite/requirements.txt
    ps.joblib
    ps.simplejson
    ps.ujson
  ]);
  # workaround: build for one working target if no targets are supported
  # a few CXX files are still build for the device
  gpuTargets' =
    if supportsTargetArches then (lib.concatStringsSep ";" supportedTargets) else "gfx1200";
  compiler = "amdclang++";
  # no-switch due to spammy warnings on some cases with fixme messages
  # FIXME(LunNova@): cmake files need patched to include this properly or
  # maybe we improve the toolchain to use config files + assemble a sysroot
  # so system wide include assumptions work
  cFlags = "-Wno-switch -fopenmp -I${lib.getDev zstd}/include -I${amd-blis}/include/blis/ -I${lib.getDev msgpack-cxx}/include";
in
stdenv.mkDerivation (finalAttrs: {
  pname = "hipblaslt${clr.gpuArchSuffix}";
  version = "10.0.0";

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "rocm-libraries";
    rev = "therock-10.0";
    hash = "sha256-oQO6/vrDjo6CLTPlqY6uF/yl4jYlf1jG1krfr5WwliI=";
    sparseCheckout = [
      "projects/hipblaslt"
      "shared"
      # ROCm 10's shared/stinkytofu includes modules from the repo-root cmake dir
      "cmake"
    ];
    # Compress the 5ish GiB of yaml files so this .src is under output size limit
    postFetch = ''
      find $out -name '*.yaml' -path '*/Tensile/Logic/*' -exec ${lib.getExe zstd} --rm {} \;
    '';
  };
  sourceRoot = "${finalAttrs.src.name}/projects/hipblaslt";
  env.CXX = compiler;
  env.CFLAGS = cFlags;
  env.CXXFLAGS = cFlags;
  # messagepack-compression-support.patch pulls zstd into the msgpack loader, so
  # every consumer (including the bench clients) has to link it.
  env.LDFLAGS = "-L${lib.getLib zstd}/lib -lzstd";
  env.ROCM_PATH = "${clr}";
  env.TENSILE_ROCM_ASSEMBLER_PATH = lib.getExe' clr "amdclang++";
  env.TENSILE_GEN_ASSEMBLY_TOOLCHAIN = lib.getExe' clr "amdclang++";
  env.LD_PRELOAD = "${jemalloc}/lib/libjemalloc.so";
  env.MALLOC_CONF = "background_thread:true,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:10000";
  requiredSystemFeatures = [ "big-parallel" ];

  __structuredAttrs = true;
  strictDeps = true;

  outputs = [
    "out"
    # benchmarks are non-optional
    "benchmark"
  ]
  ++ lib.optionals buildTests [
    "test"
  ]
  ++ lib.optionals buildSamples [
    "sample"
  ];

  patches = [
    # parallel-buildSourceCodeObjectFile.diff is applied through postPatch below:
    # ROCm 10 moved the compiler call into a timing_context block, so the diff's
    # context no longer matches.
    # Support loading zstd compressed .dat files, required to keep output under
    # hydra size limit
    ./messagepack-compression-support.patch
    # ./TensileCreateLibrary-refactor.patch and ./Tensile-interning.patch are
    # dropped: rocm-libraries#2073 was never merged, and ROCm 10 restructured
    # TensileCreateLibrary enough that the diffs no longer apply. ROCm 10 still
    # uses joblib, which is supplied through the python env below.
  ];

  preConfigure = ''
    # ROCm 10's tensilelite caches generated helpers under $HOME/.tensile, which
    # is /homeless-shelter in the nix sandbox.
    export HOME=$TMPDIR
    find . -name '*.yaml.zst' -path '*/Tensile/Logic/*' -exec zstd -d --rm {} \;
  '';

  postPatch = ''
    # Upstream issue asking for parallel-jobs to be specified properly:
    # https://github.com/ROCm/rocm-libraries/issues/1242
    substituteInPlace tensilelite/Tensile/Toolchain/Source.py \
      --replace-fail \
        '    objPath = str(tmpObjDir / objFilename)' \
        '    objPath = str(tmpObjDir / objFilename)
    compiler.default_args += ["-parallel-jobs=8"]'

    # messagepack-compression-support.patch needs <zstd.h>; ROCm 10 reordered the
    # includes in this file so the patch's first hunk no longer applies.
    substituteInPlace tensilelite/src/msgpack/MessagePack.cpp \
      --replace-fail \
        '#include <Tensile/msgpack/Loading.hpp>' \
        '#include <Tensile/msgpack/Loading.hpp>

#include <zstd.h>'

    # ROCm 10 dropped cmake/dependencies.cmake and no longer does
    # find_package(Git REQUIRED), so nothing to strip here any more.
    substituteInPlace CMakeLists.txt \
      --replace-fail " LANGUAGES CXX" " LANGUAGES CXX C ASM"
  '';

  doCheck = false;
  doInstallCheck = true;

  nativeBuildInputs = [
    cmake
    # ROCm 10's shared/origami does find_package(Git REQUIRED)
    git
    rocm-cmake
    py
    clr
    gfortran
    pkg-config
    ninja
    rocm-smi
    zstd
  ];

  buildInputs = [
    llvm.llvm
    clr
    rocm-cmake
    hipblas-common
    # ROCm 10's hipblaslt does find_package(amd_smi REQUIRED)
    amdsmi
    amd-blis
    rocm-smi
    openmp
    libffi
    ncurses
    lapack-reference

    # Tensile deps - not optional, building without tensile isn't actually supported
    msgpack-cxx
    libxml2
    python3Packages.msgpack
    zlib
    zstd
  ]
  ++ lib.optionals buildTests [
    gtest
  ];

  cmakeFlags = [
    (lib.cmakeFeature "Boost_INCLUDE_DIR" "${lib.getDev boost}/include") # msgpack FindBoost fails to find boost
    (lib.cmakeFeature "GPU_TARGETS" gpuTargets')
    (lib.cmakeBool "BUILD_TESTING" buildTests)
    (lib.cmakeBool "HIPBLASLT_ENABLE_BLIS" true)
    (lib.cmakeBool "HIPBLASLT_BUILD_TESTING" buildTests)
    (lib.cmakeBool "HIPBLASLT_ENABLE_SAMPLES" buildSamples)
    (lib.cmakeBool "HIPBLASLT_ENABLE_DEVICE" supportsTargetArches)
    # FIXME: Enable for ROCm 7.x
    (lib.cmakeBool "HIPBLASLT_ENABLE_ROCROLLER" false)
    "-DCMAKE_C_COMPILER=amdclang"
    "-DCMAKE_HIP_COMPILER=${compiler}"
    "-DCMAKE_CXX_COMPILER=${compiler}"
    "-DROCM_FOUND=ON" # hipblaslt tries to download rocm-cmake if this isn't set
    "-DBLIS_ROOT=${amd-blis}"
    "-DBLIS_LIB=${amd-blis}/lib/libblis-mt.so"
    "-DBLIS_INCLUDE_DIR=${amd-blis}/include/blis/"
    "-DBLA_PREFER_PKGCONFIG=ON"
    "-DFETCHCONTENT_SOURCE_DIR_NANOBIND=${python3Packages.nanobind.src}"
    # Manually define CMAKE_INSTALL_<DIR>
    # See: https://github.com/NixOS/nixpkgs/pull/197838
    "-DCMAKE_INSTALL_BINDIR=bin"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_INSTALL_INCLUDEDIR=include"
    "-DHIPBLASLT_ENABLE_MARKER=Off"
  ];

  postInstall =
    # Compress msgpack .dat files to stay under hydra output size limit
    # Relies on messagepack-compression-support.patch
    ''
      for file in $out/lib/hipblaslt/library/*.dat; do
        zstd -19 --long -f "$file" -o "$file.tmp" && mv "$file.tmp" "$file"
      done
    ''
    # Move binaries to appropriate outputs and delete leftover /bin
    + ''
      mkdir -p $benchmark/bin
      # ROCm 10 also installs hipblaslt-cotenant{,-kernel} and hipblaslt-perf.
      mv $out/bin/hipblaslt-{api-overhead,bench*,cotenant*,perf} $out/bin/*.yaml $out/bin/*.py $benchmark/bin
      ${lib.optionalString buildTests ''
        mkdir -p $test/bin
        mv $out/bin/hipblas-test $test/bin
      ''}
      ${lib.optionalString buildSamples ''
        mkdir -p $sample/bin
        mv $out/bin/example-* $sample/bin
      ''}
      rmdir $out/bin
    '';

  installCheckPhase =
    # Verify compression worked and .dat files aren't huge
    ''
      runHook preInstallCheck
      find "$out" -type f -name "*.dat" -size "+2M" -exec sh -c '
          echo "ERROR: oversized .dat file, check for issues with install compression: {}" >&2
          exit 1
      ' {} \;
      echo "Verified .dat files in $out are not huge"
      runHook postInstallCheck
    '';

  # If this is false there are no kernels in the output lib
  # supporting the target device
  # so if it's an optional dep it's best to not depend on it
  # Some packages like torch need hipblaslt to compile
  # and are fine ignoring it at runtime if it's not supported
  # so we have to support building an empty hipblaslt
  passthru.supportsTargetArches = supportsTargetArches;
  passthru.updateScript = rocmUpdateScript { inherit finalAttrs; };
  meta = {
    description = "Library that provides general matrix-matrix operations with a flexible API";
    homepage = "https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt";
    license = lib.licenses.mit;
    teams = [ lib.teams.rocm ];
    platforms = lib.platforms.linux;
  };
})
