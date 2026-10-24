# Windows test binaries for the OS leaves and the CPU cache, cross-built
# with the pinned MinGW-w64 toolchain. CI runs them on a Windows host.
{
  lib,
  stdenv,
  ccache,
  cmake,
  ninja,
}:

stdenv.mkDerivation {
  pname = "gufo-windows-platform-tests";
  version = "0";

  src = lib.fileset.toSource {
    root = ../..;
    fileset = lib.fileset.unions [
      ../../CMakeLists.txt
      ../../cmake
      ../../src/cache
      ../../src/core/platform
      ../../tests/cache
      ../../tests/core/platform_fs_test.cpp
      ../../tests/core/platform_mapped_file_test.cpp
    ];
  };

  nativeBuildInputs = [
    cmake
    ninja
  ];

  cmakeFlags = [
    "-DGUFO_PLATFORM_ONLY=ON"
    "-DBUILD_TESTING=ON"
    # Static runtimes: the binaries run on a bare Windows host.
    "-DCMAKE_EXE_LINKER_FLAGS=-static"
  ];

  # Same compiler cache policy as the Linux package: CI exposes a persistent
  # /tmp/gufo-ccache through extra-sandbox-paths; without it the compiler
  # runs directly.
  preConfigure = ''
    ccache_dir=/tmp/gufo-ccache
    if [[ -d "$ccache_dir" && -w "$ccache_dir" ]]; then
      export CCACHE_DIR="$ccache_dir"
      export CCACHE_BASEDIR="$NIX_BUILD_TOP"
      export CCACHE_COMPILERCHECK=content
      export CCACHE_MAXSIZE=1G
      export CCACHE_NOHASHDIR=true
      export CCACHE_UMASK=000
      cmakeFlagsArray+=("-DGUFO_CCACHE_PROGRAM=${ccache}/bin/ccache")
      ${ccache}/bin/ccache --zero-stats
    fi
  '';
  postBuild = ''
    if [[ -n "''${CCACHE_DIR:-}" ]]; then
      ${ccache}/bin/ccache --show-stats
    fi
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin
    find . -name '*_test.exe' -exec cp {} $out/bin/ \;
    runHook postInstall
  '';
}
