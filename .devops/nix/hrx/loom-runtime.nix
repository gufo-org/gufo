# The Loom runtime of the loom extension (src/models/loom): libloom_text_embed,
# Loom's text model layer behind the C API in src/models/loom/embed, built from
# the hrx-system release that loom-fun packages target (hrx-loom 0.2.0). It
# drives AMD GPUs through Loom's own driver (libamdf): no ROCm runtime or
# device toolchain is involved, and no HIP library is installed.
#
# Separate from the HRX staging package in default.nix, which pins a Gufo fork
# for the HIP runtime experiments; loom-fun packages follow Loom SDK releases.
# Keep `version` in step with LOOM_TEXT_EMBED_HRX_LOOM and runtime.cmake.
{
  lib,
  fetchFromGitHub,
  stdenvNoCC,
  clangStdenv,
  cacert,
  cmake,
  curl,
  git,
  ninja,
  python3,
}:

let
  # hrx-system v0.2.0.
  version = "0.2.0";
  revision = "0cd370f2baaa0cfef47cc23007f1a109906f8a52";

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "hrx-system";
    rev = revision;
    hash = "sha256-t/R1Qq00lpeS/elqLIltGJgHstNsWdgQWmpBMS3axkw=";
  };

  embed = ../../../src/models/loom/embed;

  # Every download pinned in MODULE.cmake.lock and Loom's ISA XML descriptor,
  # fetched once so the configure step can run without network access.
  deps = stdenvNoCC.mkDerivation {
    pname = "loom-runtime-deps";
    inherit version src;
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
    outputHash = "sha256-kUc4b7acjbrnrG6KHIpJPTRkRvZHJKeRX7WnyAUR7ug=";
  };
in
clangStdenv.mkDerivation {
  pname = "loom-runtime";
  inherit version src;

  postPatch = ''
    python3 ${./hrx-deps.py} rewrite . ${deps}
  '';

  nativeBuildInputs = [
    cmake
    ninja
    git
    python3
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    # Loom's static libraries link into the shared embedding library.
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DIREE_ENABLE_POSITION_INDEPENDENT_CODE=ON"
    "-DIREE_BUILD_TESTS=OFF"
    "-DIREE_ENABLE_WERROR_FLAG=OFF"
    "-DLIBHRX_BUILD=OFF"
    "-DIREE_HAL_DRIVER_AMD_XDNA=OFF"
    "-DLOOM_TARGET_ALL_AMD_GPU=ON"
    # HRX's CMake includes this file once its targets exist.
    "-DIREE_CMAKE_TRY_FILE=${embed}/loom_text_embed.cmake"
  ];

  ninjaFlags = [ "loom_text_embed" ];

  installPhase = ''
    runHook preInstall
    install -Dm755 libloom_text_embed.so $out/lib/libloom_text_embed.so
    install -Dm644 ${embed}/loom_text_embed.h $out/include/loom_text_embed.h
    install -Dm644 $src/LICENSE $out/share/licenses/loom-runtime/LICENSE
    mkdir -p $out/lib/pkgconfig
    cat > $out/lib/pkgconfig/loom-text-embed.pc <<EOF
    prefix=$out
    Name: loom-text-embed
    Description: Loom text model layer for embedding
    Version: ${version}
    Libs: -L$out/lib -lloom_text_embed
    Cflags: -I$out/include
    EOF
    runHook postInstall
  '';

  passthru = { inherit deps revision; };

  meta = {
    description = "Loom ${version} text model runtime for Gufo's loom extension";
    homepage = "https://github.com/ROCm/hrx-system";
    license = with lib.licenses; [
      asl20
      llvm-exception
    ];
    platforms = [ "x86_64-linux" ];
  };
}
