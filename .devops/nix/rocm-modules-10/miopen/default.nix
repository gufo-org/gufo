{
  lib,
  stdenv,
  callPackage,
  fetchFromGitHub,
  fetchpatch,
  fetchurl,
  runCommand,
  pkg-config,
  python3,
  cmake,
  rocm-cmake,
  rocblas,
  rocmlir,
  rocrand,
  rocm-runtime,
  hipblas-common,
  hipblas,
  hipblaslt,
  clr,
  llvm,
  composable_kernel,
  frugally-deep,
  rocm-docs-core,
  half,
  boost,
  sqlite,
  symlinkJoin,
  bzip2,
  lbzip2,
  nlohmann_json,
  texliveSmall,
  doxygen,
  sphinx,
  zlib,
  gtest,
  eigen,
  rocm-comgr,
  roctracer,
  python3Packages,
  gpuTargets ? clr.localGpuTargets or clr.gpuTargets,
  buildDocs ? false, # Needs internet because of rocm-docs-core
  buildTests ? false,
  withComposableKernel ? true,
}:

let
  # FIXME: cmake files need patched to include this properly
  # <ciso646> is deprecated in C++20 and one of MIOpen's vendored headers still
  # includes it; clang 23 turns that #warning into an error under -Werror.
  cFlags = "-Wno-error=#warnings -Wno-documentation-pedantic --offload-compress -isystem ${hipblas-common}/include -isystem ${hipblas}/include -isystem ${roctracer}/include -isystem ${nlohmann_json}/include -isystem ${sqlite.dev}/include -isystem ${rocrand}/include";
  version = "10.0.0";

  # Targets outside this list will get
  # error: use of undeclared identifier 'CK_BUFFER_RESOURCE_3RD_DWORD'
  supportedTargets = lib.intersectLists [
    "gfx900"
    "gfx906"
    "gfx908"
    "gfx90a"
    "gfx942"
    "gfx950"
    "gfx1030"
    "gfx1031"
    "gfx1100"
    "gfx1101"
    "gfx1102"
    "gfx1150"
    "gfx1151"
    "gfx1200"
    "gfx1201"
  ] gpuTargets;

  latex = lib.optionalAttrs buildDocs (
    texliveSmall.withPackages (
      ps: with ps; [
        latexmk
        tex-gyre
        fncychap
        wrapfig
        capt-of
        framed
        needspace
        tabulary
        varwidth
        titlesec
      ]
    )
  );

  # for hiprtcCompileProgram (dropout kernels require rocrand in -I at runtime)
  hiprtcCompileRocmPath = symlinkJoin {
    name = "miopen-hiprtc-compile-rocm-path";
    paths = [
      clr
      rocrand
    ];
  };

  # Kernel databases moved from Git LFS to DVC (anonymous s3 bucket s3://therock-dvc/rocm-libraries)
  fetchKdb =
    name:
    { url, hash }:
    runCommand "miopen-${name}.kdb" { preferLocalBuild = true; } ''
      ${lbzip2}/bin/lbzip2 -ckd ${
        fetchurl {
          inherit url hash;
          name = "${name}.kdb.bz2";
        }
      } > $out
    '';

  kdbs = lib.mapAttrs fetchKdb (
    lib.filterAttrs (name: _: lib.elem name supportedTargets) (import ./kdbs.nix)
  );

  linkKDBsTo =
    targetPath:
    lib.concatStringsSep "" (
      lib.mapAttrsToList (name: kdb: ''
        ln -sf ${kdb} ${targetPath}/${name}.kdb
      '') kdbs
    );
in
stdenv.mkDerivation (finalAttrs: {
  inherit version;
  pname = "miopen";

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "rocm-libraries";
    rev = "therock-10.0";
    sparseCheckout = [
      "projects/miopen"
      "shared"
    ];
    fetchSubmodules = true;
    hash = "sha256-8Ksw1v/nkSSS8ecWJ7CaRnK4JX1Kn6+EMg8GrTzSzhA=";
  };
  sourceRoot = "${finalAttrs.src.name}/projects/miopen";

  env.CFLAGS = cFlags;
  env.CXXFLAGS = cFlags;

  # Find zstd and add to target. Mainly for torch.
  patches = [
    ./skip-preexisting-dbs.patch
    # Both upstream MIOpen commits (e608b432, 3413d2da) are already contained in
    # therock-10.0. The ISA-compatibility patch is dropped for the same reason as
    # rocBLAS's: kernels here are built for gfx1151 exactly.
  ];

  outputs = [
    "out"
  ]
  ++ lib.optionals buildDocs [
    "doc"
  ]
  ++ lib.optionals buildTests [
    "test"
  ];
  enableParallelBuilding = true;
  env.ROCM_PATH = clr;
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [ rocm-runtime ];

  nativeBuildInputs = [
    pkg-config
    # ROCm 10's MIOpen does find_package(Python3 REQUIRED) at configure time
    python3
    cmake
    rocm-cmake
    clr
  ];

  buildInputs = [
    hipblas
    hipblas-common
    rocblas
    rocmlir
    half
    boost
    sqlite
    bzip2
    nlohmann_json
    # ROCm 10's cmake/ThirdParty.cmake find_package()s these and falls back to
    # FetchContent (network) when they are missing.
    gtest
    eigen
    frugally-deep
    roctracer
    rocrand
    hipblaslt
  ]
  ++ lib.optionals withComposableKernel [
    composable_kernel
  ]
  ++ lib.optionals buildDocs [
    latex
    doxygen
    sphinx
    rocm-docs-core
    python3Packages.sphinx-rtd-theme
    python3Packages.breathe
    python3Packages.myst-parser
  ]
  ++ lib.optionals buildTests [
    gtest
    zlib
  ];

  cmakeFlags = [
    "-DAMDGPU_TARGETS=${lib.concatStringsSep ";" supportedTargets}"
    "-DGPU_TARGETS=${lib.concatStringsSep ";" supportedTargets}"
    "-DGPU_ARCHS=${lib.concatStringsSep ";" supportedTargets}"
    "-DCMAKE_VERBOSE_MAKEFILE=ON"
    "-DCMAKE_MODULE_PATH=${clr}/hip/cmake"
    "-DCMAKE_BUILD_TYPE=Release"

    # needs to stream to stdout so bzcat rather than bunzip2
    "-DUNZIPPER=${bzip2}/bin/bzcat"

    "-DCMAKE_C_COMPILER=amdclang"
    "-DCMAKE_CXX_COMPILER=amdclang++"
    "-DROCM_PATH=${clr}"
    # ROCm 10's cmake/ClangToolChain.cmake assumes TheRock's single-prefix layout
    # (${ROCM_PATH}/lib/llvm/bin). Point it at the split nix outputs instead: the
    # ROCM_CMAKE_PATH branch wants <prefix>/bin to exist, and it honours an
    # explicit ROCM_CMAKE_LLVM_BIN_PATH for the compiler directory.
    "-DROCM_CMAKE_PATH=${llvm.rocm-toolchain}"
    "-DROCM_CMAKE_LLVM_BIN_PATH=${llvm.rocm-toolchain}/bin"
    "-DHIP_ROOT_DIR=${clr}"
    (lib.cmakeBool "MIOPEN_USE_ROCBLAS" true)
    (lib.cmakeBool "MIOPEN_USE_HIPBLASLT" true)
    (lib.cmakeBool "MIOPEN_USE_COMPOSABLEKERNEL" withComposableKernel)
    (lib.cmakeBool "MIOPEN_USE_HIPRTC" true)
    (lib.cmakeBool "MIOPEN_USE_COMGR" true)
    # ROCm 10's MIOpen pulls gtest through FetchContent when testing is enabled,
    # which cannot work in the sandbox. Tests are off for this build anyway.
    (lib.cmakeBool "BUILD_TESTING" buildTests)
    "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${clr}"
    # Manually define CMAKE_INSTALL_<DIR>
    # See: https://github.com/NixOS/nixpkgs/pull/197838
    "-DCMAKE_INSTALL_BINDIR=bin"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_INSTALL_INCLUDEDIR=include"
    "-DMIOPEN_BACKEND=HIP"
  ]
  ++ lib.optionals buildTests [
    "-DBUILD_TESTS=ON"
    "-DMIOPEN_TEST_ALL=ON"
  ];

  postPatch = ''
    substituteInPlace cmake/ClangTidy.cmake \
      --replace-fail 'macro(enable_clang_tidy)' 'macro(enable_clang_tidy)
      endmacro()
      macro(enable_clang_tidy_unused)' \
      --replace-fail 'function(clang_tidy_check TARGET)' 'function(clang_tidy_check TARGET)
      return()'

    # src/composable_kernel is gone in ROCm 10 (and CK is disabled for this build).
    patchShebangs test fin utils install_deps.cmake

    # ROCm 10's comgr.cpp no longer hardcodes "/opt/rocm": it reads the ROCM_PATH
    # env var (MIOPEN_DECLARE_ENV_VAR_STR(ROCM_PATH), src/comgr.cpp:73), which the
    # consumer already sets.
  ''
  + linkKDBsTo "src/kernels"
  + ''
    mkdir -p build/share/miopen/db/
  ''
  + linkKDBsTo "build/share/miopen/db";

  # Unfortunately, it seems like we have to call make on these manually
  postBuild =
    lib.optionalString buildDocs ''
      python -m sphinx -T -E -b html -d _build/doctrees -D language=en ../docs _build/html
    ''
    + lib.optionalString buildTests ''
      make -j$NIX_BUILD_CORES check
    '';

  postInstall = ''
    rm $out/libexec/miopen/install_precompiled_kernels.sh
  ''
  + linkKDBsTo "$out/share/miopen/db"
  + lib.optionalString buildDocs ''
    mv ../doc/html $out/share/doc/miopen-hip
  ''
  + lib.optionalString buildTests ''
    mkdir -p $test/bin
    mv bin/test_* $test/bin
    patchelf --set-rpath $out/lib:${
      lib.makeLibraryPath (
        finalAttrs.buildInputs
        ++ [
          clr
          rocm-comgr
        ]
      )
    } $test/bin/*
  '';

  requiredSystemFeatures = [ "big-parallel" ];

  passthru.impureTests = {
    # bash $(nix-build -A rocmPackages.miopen.passthru.impureTests.conv) etc
    bnorm = callPackage ./test-runtime-compilation.nix {
      miopen = finalAttrs.finalPackage;
      name = "bnorm";
      testScript = "MIOpenDriver bnorm -n 16 -c 16 -H 512 -W 512 -m 1 -F 1 -s 1 -r 1";
    };
    conv = callPackage ./test-runtime-compilation.nix {
      miopen = finalAttrs.finalPackage;
      name = "conv";
      testScript = "MIOpenDriver conv -n 1 -c 1 -H 4 -W 4 -k 1 -y 3 -x 3 -p 0 -q 0 -V 0";
    };
    dropout = callPackage ./test-runtime-compilation.nix {
      miopen = finalAttrs.finalPackage;
      name = "dropout";
      testScript = "MIOpenDriver dropout -d 64,32,14,14";
    };
    pool = callPackage ./test-runtime-compilation.nix {
      miopen = finalAttrs.finalPackage;
      name = "pool";
      testScript = "MIOpenDriver pool -W 1x1x4x4 -y 2 -x 2 -p 0 -q 0 -F 1 -V 0";
    };
  };
  passthru.tests = {
    # Ensure all .tn.model files can be loaded by whatever version of frugally-deep we have
    # This is otherwise hard to verify as MIOpen will only use these models on specific,
    # expensive Instinct GPUs
    # If MIOpen stops embedding .tn.model files the test will also fail, and can be deleted,
    # likely along with the frugally-deep dependency
    can-load-models = callPackage ./test-frugally-deep-model-loading.nix {
      inherit (finalAttrs) src version;
      inherit frugally-deep nlohmann_json;
    };
  };
  passthru.updateScript = ./update.sh;

  meta = {
    description = "Machine intelligence library for ROCm";
    homepage = "https://github.com/ROCm/rocm-libraries/tree/develop/projects/miopen";
    license = lib.licenses.mit;
    teams = [ lib.teams.rocm ];
    platforms = lib.platforms.linux;
  };
})
