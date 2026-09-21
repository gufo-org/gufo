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
  version,
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
      || relativePath == "tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json";
  };
in
stdenv.mkDerivation {
  pname = "gufo";
  inherit version;
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
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DBUILD_TESTING=OFF"
    "-DGUFO_VERSION=${version}"
    "-DGUFO_FFMPEG_EXECUTABLE=${ffmpeg-headless}/bin/ffmpeg"
    "-DGUFO_FFPROBE_EXECUTABLE=${ffmpeg-headless}/bin/ffprobe"
    "-DCMAKE_HIP_COMPILER=${rocmPackages.llvm.clang}/bin/clang"
  ];

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck

    $out/bin/gufo --version
    $out/bin/gufo --help >/dev/null
    $out/bin/gufo-server --version
    $out/bin/gufo-server --help >/dev/null
    test ! -e $out/bin/gufo-kernel-bench
    test -f $out/share/licenses/gufo/LICENSE
    test -f $out/share/licenses/gufo/third-party/LICENSE.ds4

    runHook postInstallCheck
  '';

  passthru = {
    inherit rocmPackages;
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
