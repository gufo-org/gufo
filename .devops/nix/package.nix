{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  makeWrapper,
  icu,
  curl,
  libpng,
  libjpeg,
  libwebp,
  libsndfile,
  soxr,
  openssl,
  ffmpeg-headless,
  ccache,
  rocmPackages,
  releaseVersion ? null,
  revision,
  sourceRoot ? ../..,
}:

let
  # Keep SVM available for managed model/state allocations, but prevent
  # compaction of ROCr's queue save areas and HIP's runtime-owned buffers from
  # suspending inference.
  rocmRuntime = rocmPackages.rocm-runtime.overrideAttrs (old: {
    patches = (old.patches or [ ]) ++ [ ./patches/rocr-fixed-cwsr-rocm10.patch ];
  });
  versionLabel = if releaseVersion == null then "development" else releaseVersion;
  packageVersion =
    if releaseVersion == null then "0-unstable-${revision}" else releaseVersion;
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
assert lib.versions.majorMinor rocmPackages.clr.version == "10.1";
stdenv.mkDerivation {
  pname = "gufo";
  version = packageVersion;
  src = productionSource;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    makeWrapper
    rocmPackages.clr
  ];

  buildInputs = [
    icu
    curl
    libpng
    libjpeg
    libwebp
    libsndfile
    soxr
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
    "-DGUFO_RELEASE_VERSION=${versionLabel}"
    "-DGUFO_REVISION=${revision}"
    "-DGUFO_FFMPEG_EXECUTABLE=${ffmpeg-headless}/bin/ffmpeg"
    "-DGUFO_FFPROBE_EXECUTABLE=${ffmpeg-headless}/bin/ffprobe"
    "-DCMAKE_HIP_COMPILER=${rocmPackages.llvm.clang}/bin/clang"
  ];

  # CI exposes a persistent /tmp/gufo-ccache through extra-sandbox-paths for
  # C/C++ sources. Keep HIP uncached: Clang emits a bundled preprocessor output
  # whose included headers ccache 4.13.6 does not record. A header-only change
  # can then reuse a stale HIP object, including one from the timing baseline.
  preConfigure = ''
    ccache_dir=/tmp/gufo-ccache
    if [[ -d "$ccache_dir" && -w "$ccache_dir" ]]; then
      export CCACHE_DIR="$ccache_dir"
      export CCACHE_COMPILERCHECK=content
      export CCACHE_MAXSIZE=4G
      export CCACHE_NOHASHDIR=true
      export CCACHE_UMASK=000
      cmakeFlagsArray+=(
        "-DGUFO_CCACHE_PROGRAM=${ccache}/bin/ccache"
      )
      ${ccache}/bin/ccache --zero-stats
    fi
  '';
  postBuild = ''
    if [[ -n "''${CCACHE_DIR:-}" ]]; then
      ${ccache}/bin/ccache --show-stats
    fi
  '';

  doInstallCheck = true;
  postFixup = ''
    # HIP loads ROCr transitively. Select the patched ABI-compatible runtime
    # without rebuilding the unchanged HIP/BLAS compiler and kernel libraries.
    wrapProgram "$out/bin/gufo" \
      --prefix LD_LIBRARY_PATH : "${rocmRuntime}/lib"
  '';
  installCheckPhase = ''
    runHook preInstallCheck

    test "$($out/bin/gufo --version)" = "gufo version ${versionLabel} (${revision})"
    $out/bin/gufo --help >/dev/null
    test "$($out/bin/gufo-server --version)" = "gufo version ${versionLabel} (${revision})"
    $out/bin/gufo-server --help >/dev/null
    test ! -e $out/bin/gufo-kernel-bench
    test -f $out/share/licenses/gufo/LICENSE
    test -f $out/share/licenses/gufo/third-party/LICENSE.ds4

    runHook postInstallCheck
  '';

  passthru = {
    inherit releaseVersion revision rocmPackages rocmRuntime;
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
