{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  python313,
  icu,
  curl,
  libpng,
  libjpeg,
  openssl,
  ffmpeg-headless,
  libuuid,
  rocmPackages,
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
      || relativePath == "tests"
      || relativePath == "tests/models"
      || relativePath == "tests/models/deepseek_v4_flash"
      || relativePath == "tests/models/deepseek_v4_flash/fixtures"
      || relativePath == "tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json"
      || relativePath == "tools"
      || relativePath == "tools/bench"
      || relativePath == "tools/bench/tune_hipblaslt.cpp"
      || relativePath == "tools/bench/benchmark_ssm_replay.cpp"
      || relativePath == "tools/bench/wmma_layout_test.hip"
      || relativePath == "tools/quant"
      || relativePath == "tools/quant/gguf_dump_types.cpp"
      || relativePath == "tools/gufo"
      || relativePath == "tools/gufo/compile_h3_attention.py"
      || relativePath == "tools/gufo/h3_attention_kernel.py";
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
    curl
    libpng
    libjpeg
    openssl
    ffmpeg-headless
  ]
  ++ lib.optionals rocmSupport [
    rocmPackages.clr
    rocmPackages.hipblas
    rocmPackages.hipblaslt
    rocmPackages.hipcub
    rocmPackages.miopen
    rocmPackages.rocprim
    rocmPackages.rocwmma
    rocmPackages.rocblas
    rocmPackages.composable_kernel
    rocmPackages.aotriton
    rocmPackages.rocprofiler-sdk
  ]
  ++ lib.optionals xrtSupport [
    aie-smoke
    xrt
    xrt-plugin-amdxdna
    libuuid
  ];

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
  ++ lib.optional xrtSupport "-DGUFO_AIE_SMOKE_PROGRAM_DIR=${placeholder "out"}/share/gufo/aie/smoke";

  env = lib.optionalAttrs rocmSupport {
    ROCM_PATH = "${rocmPackages.clr}";
    GUFO_HIPCUB_ROOT = "${rocmPackages.hipcub}";
    GUFO_ROCPRIM_ROOT = "${rocmPackages.rocprim}";
    GUFO_ROCWMMA_ROOT = "${rocmPackages.rocwmma}";
  }
  // lib.optionalAttrs xrtSupport {
    GUFO_AIE_SMOKE_ROOT = "${aie-smoke}";
    XRT_PATH = "${xrt}/opt/xilinx/xrt";
    # Combined NPU lib dir so XRT can discover the amdxdna plugin at runtime.
    LD_LIBRARY_PATH = "${xrt}/opt/xilinx/xrt/lib:${xrt-plugin-amdxdna}/opt/xilinx/xrt/lib";
  };

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    cp gufo $out/bin/gufo
    ln -sf gufo $out/bin/gufo-server
    mkdir -p $out/share/gufo/models/qwen3_tts
    cp $src/src/models/qwen3_tts/reference/run_official.py \
      $out/share/gufo/models/qwen3_tts/
    mkdir -p $out/share/gufo/models/minimax_h3
    cp $src/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json \
      $out/share/gufo/models/minimax_h3/
    mkdir -p $out/share/gufo/eval
    cp $src/tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json $out/share/gufo/eval/
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
    chmod +x $out/bin/*

    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck

    $out/bin/gufo --version
    $out/bin/gufo --help >/dev/null
    $out/bin/gufo-server --version
    $out/bin/gufo-server --help >/dev/null
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
