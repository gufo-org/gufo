{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  icu,
  curl,
  libpng,
  libjpeg,
  openssl,
  ffmpeg-headless,
  rocmPackages,
  rdma-core ? null,
  enableTp2Rdma ? false,
  releaseVersion ? null,
  revision,
}:

let
  versionLabel = if releaseVersion == null then "development" else releaseVersion;
  packageVersion =
    if releaseVersion == null then "0-unstable-${revision}" else releaseVersion;
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
      || builtins.elem relativePath [ "CMakeLists.txt" "LICENSE" "NOTICE" "THIRD_PARTY_NOTICES.md" ]
      || relativePath == "licenses"
      || lib.hasPrefix "licenses/" relativePath
      || relativePath == "cmake"
      || lib.hasPrefix "cmake/" relativePath
      || relativePath == "src"
      || lib.hasPrefix "src/" relativePath
      || relativePath == "tests"
      || relativePath == "tests/models"
      || relativePath == "tests/models/deepseek_v4_flash"
      || relativePath == "tests/models/deepseek_v4_flash/fixtures"
      || relativePath == "tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json"
      || relativePath == "tests/models/qwen38_flash_next"
      || lib.hasPrefix "tests/models/qwen38_flash_next/" relativePath
      || relativePath == "tests/cli/tp_control_test.cpp";
  };
in
stdenv.mkDerivation {
  pname = "gufo";
  version = packageVersion;
  src = productionSource;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    rocmPackages.clr
  ];

  buildInputs = [
    icu
    curl
    libpng
    libjpeg
    openssl
    ffmpeg-headless
    rocmPackages.clr
    rocmPackages.hipblas
    rocmPackages.hipblaslt
    rocmPackages.hipcub
    rocmPackages.rocprim
    rocmPackages.rocwmma
    rocmPackages.rocblas
  ] ++ lib.optionals enableTp2Rdma [ rdma-core rdma-core.dev ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DBUILD_TESTING=OFF"
    "-DGUFO_RELEASE_VERSION=${versionLabel}"
    "-DGUFO_REVISION=${revision}"
    "-DGUFO_FFMPEG_EXECUTABLE=${ffmpeg-headless}/bin/ffmpeg"
    "-DGUFO_FFPROBE_EXECUTABLE=${ffmpeg-headless}/bin/ffprobe"
    "-DCMAKE_HIP_COMPILER=${rocmPackages.llvm.clang}/bin/clang"
  ] ++ lib.optionals enableTp2Rdma [ "-DGUFO_ENABLE_TP2_RDMA=ON" ];

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck

    test "$($out/bin/gufo --version)" = "gufo version ${versionLabel} (${revision})"
    $out/bin/gufo --help >/dev/null
    test "$($out/bin/gufo-server --version)" = "gufo version ${versionLabel} (${revision})"
    $out/bin/gufo-server --help >/dev/null
    test ! -e $out/bin/gufo-kernel-bench
    test -f $out/share/licenses/gufo/LICENSE
    test -f $out/share/licenses/gufo/third-party/LICENSE.ds4
    ${lib.optionalString enableTp2Rdma ''
      test -e ${rdma-core.dev}/lib/libibverbs.so
      test -f $out/share/licenses/gufo/third-party/rdma-core/COPYING.GPL2
    ''}

    runHook postInstallCheck
  '';

  passthru = {
    inherit releaseVersion revision rocmPackages;
    toolchain = {
      targetPlatform = "x86_64-linux";
      targetGpu = "gfx1151";
      cxxCompiler = stdenv.cc.name;
      rocmVersion = rocmPackages.clr.version;
      hipClangVersion = rocmPackages.llvm.clang.version;
    };
  };

  meta = with lib; {
    description = "Gufo Engine — local inference runtime for AMD Strix Halo (gfx1151 GPU)";
    homepage = "https://github.com/gufo-org/gufo";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "gufo";
  };
}
