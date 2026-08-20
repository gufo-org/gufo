{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  libuuid,
  rocmPackages,
  aie-qwen-mtp-eh-proj,
  aie-qwen-mtp-rmsnorm,
  aie-smoke,
  xrt,
  xrt-plugin-amdxdna,
  config,
  version,

  # Overridable feature flags
  rocmSupport ? config.rocmSupport or false,
  rocmGpuTargets ? (lib.optionals rocmSupport rocmPackages.clr.gpuTargets),
  # Wire the XRT NPU shim + amdxdna plugin into the build.
  xrtSupport ? true,
}:

let
  sourceRoot = ../..;
  productionSource = lib.cleanSourceWith {
    src = sourceRoot;
    filter =
      path: _type:
      let
        root = toString sourceRoot;
        pathString = toString path;
        relativePath = lib.removePrefix "${root}/" pathString;
      in
      pathString == root
      || relativePath == "CMakeLists.txt"
      || relativePath == "cmake"
      || lib.hasPrefix "cmake/" relativePath
      || relativePath == "src"
      || lib.hasPrefix "src/" relativePath
      || relativePath == "tools"
      || relativePath == "tools/tune_hipblaslt.cpp"
      || relativePath == "tools/benchmark_ssm_replay.cpp";
  };
in
stdenv.mkDerivation (finalAttrs: {
  pname = "strix";
  inherit version;
  src = productionSource;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ]
  ++ lib.optional rocmSupport rocmPackages.clr;

  buildInputs = lib.optionals rocmSupport [
    rocmPackages.clr
    rocmPackages.hipblas
    rocmPackages.hipblaslt
    rocmPackages.rocblas
    rocmPackages.composable_kernel
    rocmPackages.rocprofiler-sdk
  ]
  ++ lib.optionals xrtSupport [
    aie-qwen-mtp-eh-proj
    aie-qwen-mtp-rmsnorm
    aie-smoke
    xrt
    xrt-plugin-amdxdna
    libuuid
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DBUILD_TESTING=OFF"
    "-DSTRIX_VERSION=${version}"
  ]
  ++ lib.optional rocmSupport "-DENGINE_ENABLE_HIP=ON"
  ++ lib.optional rocmSupport "-DCMAKE_HIP_COMPILER=${rocmPackages.llvm.clang}/bin/clang"
  ++ lib.optional rocmSupport "-DGPU_TARGETS=${lib.concatStringsSep ";" rocmGpuTargets}"
  ++ lib.optional xrtSupport "-DENGINE_ENABLE_XRT=ON"
  ++ lib.optional xrtSupport "-DSTRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR=${placeholder "out"}/share/strix/aie/qwen-mtp-eh-proj"
  ++ lib.optional xrtSupport "-DSTRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR=${placeholder "out"}/share/strix/aie/qwen-mtp-rmsnorm"
  ++ lib.optional xrtSupport "-DSTRIX_AIE_SMOKE_PROGRAM_DIR=${placeholder "out"}/share/strix/aie/smoke";

  env = lib.optionalAttrs rocmSupport {
    ROCM_PATH = "${rocmPackages.clr}";
  }
  // lib.optionalAttrs xrtSupport {
    STRIX_AIE_QWEN_MTP_EH_PROJ_ROOT = "${aie-qwen-mtp-eh-proj}";
    STRIX_AIE_QWEN_MTP_RMSNORM_ROOT = "${aie-qwen-mtp-rmsnorm}";
    STRIX_AIE_SMOKE_ROOT = "${aie-smoke}";
    XRT_PATH = "${xrt}/opt/xilinx/xrt";
    # Combined NPU lib dir so XRT can discover the amdxdna plugin at runtime.
    LD_LIBRARY_PATH = "${xrt}/opt/xilinx/xrt/lib:${xrt-plugin-amdxdna}/opt/xilinx/xrt/lib";
  };

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    cp strix $out/bin/strix
    cp strix-server $out/bin/strix-server
    if [ -f strix-bench ]; then
      cp strix-bench $out/bin/strix-bench
    fi
    if [ -f strix-kernel-bench ]; then
      cp strix-kernel-bench $out/bin/strix-kernel-bench
    fi
    if [ -f tune_hipblaslt ]; then
      cp tune_hipblaslt $out/bin/tune_hipblaslt
    fi
    if [ -f benchmark_ssm_replay ]; then
      cp benchmark_ssm_replay $out/bin/benchmark_ssm_replay
    fi
    if [ -d ${aie-smoke} ]; then
      mkdir -p $out/share/strix/aie/smoke
      cp ${aie-smoke}/smoke.xclbin ${aie-smoke}/smoke.insts.elf \
        ${aie-smoke}/smoke.insts.bin ${aie-smoke}/smoke.pdi \
        ${aie-smoke}/smoke.aie-partition.json \
        ${aie-smoke}/manifest.json ${aie-smoke}/SHA256SUMS \
        $out/share/strix/aie/smoke/
    fi
    if [ -d ${aie-qwen-mtp-rmsnorm} ]; then
      mkdir -p $out/share/strix/aie/qwen-mtp-rmsnorm
      cp ${aie-qwen-mtp-rmsnorm}/qwen_mtp_rmsnorm.xclbin \
        ${aie-qwen-mtp-rmsnorm}/qwen_mtp_rmsnorm.insts.elf \
        ${aie-qwen-mtp-rmsnorm}/qwen_mtp_rmsnorm.insts.bin \
        ${aie-qwen-mtp-rmsnorm}/qwen_mtp_rmsnorm.pdi \
        ${aie-qwen-mtp-rmsnorm}/qwen_mtp_rmsnorm.aie-partition.json \
        ${aie-qwen-mtp-rmsnorm}/manifest.json \
        ${aie-qwen-mtp-rmsnorm}/SHA256SUMS \
        $out/share/strix/aie/qwen-mtp-rmsnorm/
    fi
    if [ -d ${aie-qwen-mtp-eh-proj} ]; then
      mkdir -p $out/share/strix/aie/qwen-mtp-eh-proj
      cp ${aie-qwen-mtp-eh-proj}/qwen_mtp_eh_proj.xclbin \
        ${aie-qwen-mtp-eh-proj}/qwen_mtp_eh_proj.insts.elf \
        ${aie-qwen-mtp-eh-proj}/qwen_mtp_eh_proj_insts.bin \
        ${aie-qwen-mtp-eh-proj}/qwen_mtp_eh_proj.pdi \
        ${aie-qwen-mtp-eh-proj}/qwen_mtp_eh_proj.aie-partition.json \
        ${aie-qwen-mtp-eh-proj}/manifest.json \
        ${aie-qwen-mtp-eh-proj}/SHA256SUMS \
        $out/share/strix/aie/qwen-mtp-eh-proj/
    fi
    chmod +x $out/bin/*

    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck

    $out/bin/strix-server --version
    $out/bin/strix-server --help >/dev/null
    if [ -x $out/bin/strix-kernel-bench ]; then
      $out/bin/strix-kernel-bench --help >/dev/null
    fi
    if [ -x $out/bin/tune_hipblaslt ]; then
      $out/bin/tune_hipblaslt --help >/dev/null
    fi

    runHook postInstallCheck
  '';

  passthru = {
    inherit rocmPackages xrt xrt-plugin-amdxdna;
    toolchain = {
      targetPlatform = "x86_64-linux";
      targetGpu = "gfx1151";
      targetNpu = "XDNA2/AIE2P";
      cxxCompiler = stdenv.cc.name;
      rocmVersion = rocmPackages.clr.version;
      hipClangVersion = rocmPackages.llvm.clang.version;
      xrtCommit = xrt.src.rev;
      xrtPluginCommit = xrt-plugin-amdxdna.src.rev;
      xrtPluginVersion = xrt-plugin-amdxdna.pluginVersion;
      aiebuRevision = aie-smoke.passthru.aiebu.src.rev;
      llvmAieVersion = aie-smoke.passthru."llvm-aie".version;
      mlirAieVersion = aie-smoke.passthru."mlir-aie".version;
    };
  };

  meta = with lib; {
    description = "Strix Engine — local inference runtime for AMD Strix Halo (gfx1151 GPU + XDNA2 NPU)";
    homepage = "https://github.com/";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strix";
  };
})
