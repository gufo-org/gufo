{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  python313,
  icu,
  ffmpeg-headless,
  libuuid,
  rocmPackages,
  aie-qwen-mtp-eh-proj,
  aie-qwen-mtp-rmsnorm,
  aie-smoke,
  xrt,
  xrt-plugin-amdxdna,
  hrx-system,
  config,
  version,

  # Overridable feature flags
  rocmSupport ? config.rocmSupport or false,
  rocmGpuTargets ? (lib.optionals rocmSupport rocmPackages.clr.gpuTargets),
  # Wire the XRT NPU shim + amdxdna plugin into the build.
  xrtSupport ? true,
  # Build the native HRX/Loom runtime variant. The default package stays on
  # the established ROCm/HIP route.
  hrxSupport ? false,
}:

let
  sourceRoot = ../..;
  h3TritonCompiler = python313.withPackages (ps: [ ps.triton ]);
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
      || relativePath == "tools/bench"
      || relativePath == "tools/bench/tune_hipblaslt.cpp"
      || relativePath == "tools/bench/benchmark_ssm_replay.cpp"
      || relativePath == "tools/bench/wmma_layout_test.hip"
      || relativePath == "tools/quant"
      || relativePath == "tools/quant/gguf_dump_types.cpp"
      || relativePath == "tools/gufo"
      || relativePath == "tools/gufo/compile_h3_attention.py"
      || relativePath == "tools/gufo/h3_attention_kernel.py"
      || relativePath == "tools/loom"
      || lib.hasPrefix "tools/loom/" relativePath
      || relativePath == "tools/gufo/hrx-wrapper.sh";
  };
in
stdenv.mkDerivation (finalAttrs: {
  pname = "gufo";
  inherit version;
  src = productionSource;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ]
  ++ lib.optionals rocmSupport [
    rocmPackages.clr
    h3TritonCompiler
  ];

  buildInputs = [
    icu
    ffmpeg-headless
  ]
  ++ lib.optionals rocmSupport [
    rocmPackages.clr
    rocmPackages.hipblas
    rocmPackages.hipblaslt
    rocmPackages.hipcub
    rocmPackages.rocprim
    rocmPackages.rocwmma
    rocmPackages.rocblas
    rocmPackages.composable_kernel
    rocmPackages.aotriton
    rocmPackages.rocprofiler-sdk
  ]
  ++ lib.optionals xrtSupport [
    aie-qwen-mtp-eh-proj
    aie-qwen-mtp-rmsnorm
    aie-smoke
    xrt
    xrt-plugin-amdxdna
    libuuid
  ]
  ++ lib.optional hrxSupport hrx-system;

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DBUILD_TESTING=OFF"
    "-DGUFO_VERSION=${version}"
    "-DGUFO_FFMPEG_EXECUTABLE=${ffmpeg-headless}/bin/ffmpeg"
    "-DGUFO_FFPROBE_EXECUTABLE=${ffmpeg-headless}/bin/ffprobe"
  ]
  ++ lib.optional rocmSupport "-DENGINE_ENABLE_HIP=ON"
  ++ lib.optional rocmSupport "-DCMAKE_HIP_COMPILER=${rocmPackages.llvm.clang}/bin/clang"
  ++ lib.optional rocmSupport "-DGPU_TARGETS=${lib.concatStringsSep ";" rocmGpuTargets}"
  ++ lib.optional rocmSupport "-DHIPCUB_INCLUDE_DIR=${rocmPackages.hipcub}/include"
  ++ lib.optional rocmSupport "-DROCPRIM_INCLUDE_DIR=${rocmPackages.rocprim}/include"
  ++ lib.optional rocmSupport "-DROCWMMA_INCLUDE_DIR=${rocmPackages.rocwmma}/include"
  ++ lib.optional xrtSupport "-DENGINE_ENABLE_XRT=ON"
  ++ lib.optional hrxSupport "-DENGINE_ENABLE_HRX=ON"
  ++ lib.optional hrxSupport "-DHRX_ROOT=${hrx-system}"
  ++ lib.optional xrtSupport "-DENGINE_ENABLE_XRT=ON"
  ++ lib.optional xrtSupport "-DGUFO_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR=${placeholder "out"}/share/gufo/aie/qwen-mtp-eh-proj"
  ++ lib.optional xrtSupport "-DGUFO_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR=${placeholder "out"}/share/gufo/aie/qwen-mtp-rmsnorm"
  ++ lib.optional xrtSupport "-DGUFO_AIE_SMOKE_PROGRAM_DIR=${placeholder "out"}/share/gufo/aie/smoke";

  env = lib.optionalAttrs rocmSupport {
    ROCM_PATH = "${rocmPackages.clr}";
    GUFO_HIPCUB_ROOT = "${rocmPackages.hipcub}";
    GUFO_ROCPRIM_ROOT = "${rocmPackages.rocprim}";
    GUFO_ROCWMMA_ROOT = "${rocmPackages.rocwmma}";
  }
  // lib.optionalAttrs xrtSupport {
    GUFO_AIE_QWEN_MTP_EH_PROJ_ROOT = "${aie-qwen-mtp-eh-proj}";
    GUFO_AIE_QWEN_MTP_RMSNORM_ROOT = "${aie-qwen-mtp-rmsnorm}";
    GUFO_AIE_SMOKE_ROOT = "${aie-smoke}";
    XRT_PATH = "${xrt}/opt/xilinx/xrt";
    # Combined NPU lib dir so XRT can discover the amdxdna plugin at runtime.
    LD_LIBRARY_PATH = "${xrt}/opt/xilinx/xrt/lib:${xrt-plugin-amdxdna}/opt/xilinx/xrt/lib";
  };

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    cp gufo $out/bin/gufo
    if [ "${if hrxSupport then "1" else "0"}" = 1 ]; then
      mkdir -p $out/share/gufo/kernels
      cp -R share/gufo/kernels/. $out/share/gufo/kernels/
      mv $out/bin/gufo $out/bin/.gufo-hrx-real
      cp $src/tools/gufo/hrx-wrapper.sh $out/bin/gufo
      substituteInPlace $out/bin/gufo \
        --replace-fail '@HRX_HIP_LIBRARY@' '${hrx-system}/lib/libamdhip64.so'
    fi
    mkdir -p $out/share/gufo/models/qwen3_tts
    cp $src/src/models/qwen3_tts/reference/run_official.py \
      $out/share/gufo/models/qwen3_tts/
    mkdir -p $out/share/gufo/models/minimax_h3
    cp $src/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json \
      $out/share/gufo/models/minimax_h3/
    if [ -f gufo-kernel-bench ]; then
      cp gufo-kernel-bench $out/bin/gufo-kernel-bench
    fi
    if [ -f tune_hipblaslt ]; then
      cp tune_hipblaslt $out/bin/tune_hipblaslt
    fi
    if [ -f benchmark_ssm_replay ]; then
      cp benchmark_ssm_replay $out/bin/benchmark_ssm_replay
    fi
    if [ -d ${aie-smoke} ]; then
      mkdir -p $out/share/gufo/aie/smoke
      cp ${aie-smoke}/smoke.xclbin ${aie-smoke}/smoke.insts.elf \
        ${aie-smoke}/smoke.insts.bin ${aie-smoke}/smoke.pdi \
        ${aie-smoke}/smoke.aie-partition.json \
        ${aie-smoke}/manifest.json ${aie-smoke}/SHA256SUMS \
        $out/share/gufo/aie/smoke/
    fi
    if [ -d ${aie-qwen-mtp-rmsnorm} ]; then
      mkdir -p $out/share/gufo/aie/qwen-mtp-rmsnorm
      cp ${aie-qwen-mtp-rmsnorm}/qwen_mtp_rmsnorm.xclbin \
        ${aie-qwen-mtp-rmsnorm}/qwen_mtp_rmsnorm.insts.elf \
        ${aie-qwen-mtp-rmsnorm}/qwen_mtp_rmsnorm.insts.bin \
        ${aie-qwen-mtp-rmsnorm}/qwen_mtp_rmsnorm.pdi \
        ${aie-qwen-mtp-rmsnorm}/qwen_mtp_rmsnorm.aie-partition.json \
        ${aie-qwen-mtp-rmsnorm}/manifest.json \
        ${aie-qwen-mtp-rmsnorm}/SHA256SUMS \
        $out/share/gufo/aie/qwen-mtp-rmsnorm/
    fi
    if [ -d ${aie-qwen-mtp-eh-proj} ]; then
      mkdir -p $out/share/gufo/aie/qwen-mtp-eh-proj
      cp ${aie-qwen-mtp-eh-proj}/qwen_mtp_eh_proj.xclbin \
        ${aie-qwen-mtp-eh-proj}/qwen_mtp_eh_proj.insts.elf \
        ${aie-qwen-mtp-eh-proj}/qwen_mtp_eh_proj_insts.bin \
        ${aie-qwen-mtp-eh-proj}/qwen_mtp_eh_proj.pdi \
        ${aie-qwen-mtp-eh-proj}/qwen_mtp_eh_proj.aie-partition.json \
        ${aie-qwen-mtp-eh-proj}/manifest.json \
        ${aie-qwen-mtp-eh-proj}/SHA256SUMS \
        $out/share/gufo/aie/qwen-mtp-eh-proj/
    fi
    chmod +x $out/bin/*

    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck

    $out/bin/gufo --version
    $out/bin/gufo --help >/dev/null
    $out/bin/gufo serve --help >/dev/null
    if [ -x $out/bin/gufo-kernel-bench ]; then
      $out/bin/gufo-kernel-bench --help >/dev/null
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
    description = "Gufo Engine — local inference runtime for AMD Strix Halo (gfx1151 GPU + XDNA2 NPU)";
    homepage = "https://github.com/";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "gufo";
  };
})
