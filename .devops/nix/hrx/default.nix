# Experimental: HRX (https://github.com/ROCm/hrx-system), AMD's alternative HIP
# runtime, built for gfx1151 on its native libamdf driver. It installs a
# drop-in `libamdhip64.so.7` plus the `hrx-info` and `iree-profile` tools.
# Not part of the default package, hosted checks or releases; see
# docs/internal/hrx-migration.md for the measurements behind the patches.
{
  lib,
  fetchFromGitHub,
  stdenvNoCC,
  cacert,
  cmake,
  curl,
  git,
  ninja,
  python3,
  rocmPackages,
  # Also build Loom's compiler and serving tools with AMD GPU (gfx1151) and
  # XDNA NPU targets, plus `loom-fetch`.
  withLoom ? false,
}:

let
  # Exact commit of gufo-org/hrx-system's gufo-staging branch. Individual
  # fixes remain on independent branches for upstream review; see README.md.
  revision = "2ebcaaece8fa731cf2e14e978c9f6242540c3341";

  src = fetchFromGitHub {
    owner = "gufo-org";
    repo = "hrx-system";
    rev = revision;
    hash = "sha256-da6x7wU4whbgVNSIaUrQsbIl0x529gyMiUQHEkTjJwc=";
  };

  # Every download pinned in MODULE.cmake.lock and Loom's ISA XML descriptor,
  # fetched once so the configure step can run without network access.
  deps = stdenvNoCC.mkDerivation {
    pname = "hrx-system-deps";
    version = "0-unstable-2026-11-11";
    inherit src;
    nativeBuildInputs = [
      cacert
      curl
      python3
    ];
    dontConfigure = true;
    buildPhase = ''
      runHook preBuild
      python3 ${./hrx-deps.py} fetch . $out
      runHook postBuild
    '';
    dontInstall = true;
    dontFixup = true;
    outputHashMode = "recursive";
    outputHashAlgo = "sha256";
    outputHash = "sha256-yIMgpPVc9lnSaGnkGgOIeUUiweN04uj+sHVm49WWAH8=";
  };

  llvm = rocmPackages.llvm;
  loomTools = [
    "loom-serve"
    "loom-compile"
    "loom-import-cxx"
    "loom-format"
    "loom-benchmark"
    "loom-test"
  ];
in
llvm.rocmClangStdenv.mkDerivation {
  pname = if withLoom then "hrx-system-loom" else "hrx-system";
  version = "0-unstable-2026-11-11";
  inherit src;

  patches = [
    # HRX_DEBUG_CAPTURE=1 prints a backtrace where a graph capture is
    # invalidated, naming the HIP call that broke it.
    ../patches/hrx-debug-capture.patch
    # IREE_PROFILE_ALLOW_OPEN_SESSION=1 lets iree-profile read a trace whose
    # process exited without ending the profiling session, as Gufo does.
    ../patches/hrx-profile-open-session.patch
  ];

  postPatch = ''
    python3 ${./hrx-deps.py} rewrite . ${deps}
  '';

  # The pinned ROCm toolchain already includes lld. Depending on the separate
  # llvm.lld derivation needlessly rebuilds LLVM instead of using that linker.
  nativeBuildInputs = [
    cmake
    ninja
    git
    python3
    llvm.clang
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DIREE_ROCM_DEPENDENCY_MODE=pinned"
    "-DIREE_HAL_AMDGPU_TARGETS=gfx1151"
    "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=llvm-tools"
    "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR=${llvm.clang}/bin"
    "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_DEVICE_LIB_PATH=${rocmPackages.rocm-device-libs}/amdgcn/bitcode"
    "-DIREE_BUILD_TESTS=OFF"
    "-DIREE_ENABLE_WERROR_FLAG=OFF"
    "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld"
    "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld"
    "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld"
  ]
  ++ lib.optionals withLoom [
    "-DLOOM_TARGET_ALL_AMD=ON"
    "-DLOOM_ARCH_AMDGPU_TARGETS=gfx1151"
    "-DLOOM_IMPORT_CXX=ON"
  ];

  ninjaFlags = [
    "libhrx/src/binding/hip/libamdhip64.so"
    "hrx-info"
    "iree-profile"
  ]
  ++ lib.optionals withLoom (map (tool: "loom/src/loom/tools/${tool}/${tool}") loomTools);

  installPhase = ''
    runHook preInstall
    install -Dm755 libhrx/src/binding/hip/libamdhip64.so.7.9999.0 \
      $out/lib/libamdhip64.so.7.9999.0
    ln -s libamdhip64.so.7.9999.0 $out/lib/libamdhip64.so.7
    ln -s libamdhip64.so.7 $out/lib/libamdhip64.so
    install -Dm755 libhrx/tools/hrx-info $out/bin/hrx-info
    install -Dm755 runtime/src/iree/tools/iree-profile/iree-profile \
      $out/bin/iree-profile
  ''
  + lib.optionalString withLoom ''
    for tool in ${lib.concatStringsSep " " loomTools}; do
      install -Dm755 loom/src/loom/tools/$tool/$tool $out/bin/$tool
    done
    # loom-fetch is the pure-Python model acquisition tool.
    mkdir -p $out/share/loom
    cp -r $src/loom/py $out/share/loom/py
    cat > $out/bin/loom-fetch <<EOF
    #!${python3}/bin/python3
    import sys
    sys.path.insert(0, "$out/share/loom/py")
    from loom.model.fetch_main import main
    sys.exit(main())
    EOF
    chmod +x $out/bin/loom-fetch
  ''
  + ''
    runHook postInstall
  '';

  passthru = { inherit deps revision; };

  meta = {
    description = "HRX HIP runtime${lib.optionalString withLoom " and Loom tools"} (experimental Gufo integration)";
    homepage = "https://github.com/ROCm/hrx-system";
    license = with lib.licenses; [
      asl20
      llvm-exception
    ];
    platforms = [ "x86_64-linux" ];
  };
}
