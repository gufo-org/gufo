{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  ninja,
  pkg-config,
  git,
  python3,
  boost,
  rapidjson,
  libdrm,
  elfutils,
  libuuid,
  libsystemtap,
  xrt,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "xrt-plugin-amdxdna";
  version = "202610.2.21.21";

  # Exported for use in symlinkJoin postBuild
  pluginVersion = "2.21.0";

  src = fetchFromGitHub {
    owner = "amd";
    repo = "xdna-driver";
    rev = "2.21.75";
    hash = "sha256-bBiI42bwap6O59MQdIylX7uz+fLUF75RTyNWTJfAFds=";
    fetchSubmodules = true;
  };

  # Fetch XRT source for internal headers (must match XRT version for ABI compatibility)
  xrtSrc = fetchFromGitHub {
    owner = "Xilinx";
    repo = "XRT";
    rev = "202610.2.21.21";
    hash = "sha256-Foj33/U6waL81EzJ0ah66xCXEGWEkvhwmurKobfCevE=";
    fetchSubmodules = true;
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    git
    python3
  ];

  buildInputs = [
    boost
    rapidjson
    libdrm
    elfutils
    libuuid
    libsystemtap
    xrt
  ];

  # Build from source root with our wrapper CMakeLists.txt
  cmakeDir = "..";

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_INSTALL_PREFIX" (placeholder "out"))
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "XRT_VERSION_STRING" finalAttrs.pluginVersion)
    # Point to XRT installation for libraries
    (lib.cmakeFeature "XRT_INSTALL_DIR" "${xrt}/opt/xilinx/xrt")
    # Point to XRT source for internal headers
    (lib.cmakeFeature "XRT_SOURCE_DIR" "${finalAttrs.xrtSrc}/src")
  ];

  postPatch = ''
    # Disable Werror
    find . -name "CMakeLists.txt" -exec sed -i 's/-Werror//g' {} \; || true

    # Fix firmware path (kernel provides firmware via linux-firmware)
    sed -i 's|/usr/lib/firmware/amdnpu|${placeholder "out"}/share/firmware/amdnpu|g' CMakeLists.txt || true

    # Remove install commands for XRT targets (they don't exist in upstream mode)
    sed -i '/install(TARGETS.*XRT_CORE_TARGET/d' src/shim/CMakeLists.txt
    sed -i '/install(TARGETS.*XRT_COREUTIL_TARGET/d' src/shim/CMakeLists.txt
    sed -i '/install(TARGETS.*XDNA_TARGET.*DESTINATION.*XDNA_BIN_DIR/d' src/shim/CMakeLists.txt

    # Create a wrapper CMakeLists.txt that defines IMPORTED targets for XRT
    cat > CMakeLists.txt << 'CMAKEOF'
cmake_minimum_required(VERSION 3.18)
project(xrt-plugin-amdxdna)

# XRT upstream mode
set(XRT_UPSTREAM ON)

# Create IMPORTED targets for XRT libraries
add_library(xrt_core SHARED IMPORTED)
set_target_properties(xrt_core PROPERTIES
  IMPORTED_LOCATION "''${XRT_INSTALL_DIR}/lib/libxrt_core.so"
  INTERFACE_INCLUDE_DIRECTORIES "''${XRT_INSTALL_DIR}/include"
)

add_library(xrt_coreutil SHARED IMPORTED)
set_target_properties(xrt_coreutil PROPERTIES
  IMPORTED_LOCATION "''${XRT_INSTALL_DIR}/lib/libxrt_coreutil.so"
  INTERFACE_INCLUDE_DIRECTORIES "''${XRT_INSTALL_DIR}/include"
)

# Create gen directory for version.h
file(MAKE_DIRECTORY ''${CMAKE_BINARY_DIR}/gen)

# Create version.h stub
file(WRITE ''${CMAKE_BINARY_DIR}/gen/version.h
"#pragma once
#define XRT_VERSION \"''${XRT_VERSION_STRING}\"
#define XRT_VERSION_MAJOR 2
#define XRT_VERSION_MINOR 21
#define XRT_VERSION_PATCH 0
")

# Set XRT_BINARY_DIR for gen includes
set(XRT_BINARY_DIR ''${CMAKE_BINARY_DIR})

# Plugin version
set(XRT_PLUGIN_VERSION_STRING ''${XRT_VERSION_STRING})
set(XRT_SOVERSION 2)

# Package installation directories
set(XDNA_PKG_LIB_DIR "opt/xilinx/xrt/lib")
set(XDNA_COMPONENT "xdna")
set(XDNA_BIN_DIR "''${CMAKE_BINARY_DIR}/bin")

add_subdirectory(src/shim)
CMAKEOF
  '';

  postInstall = ''
    # Move plugin library to XRT's expected location
    mkdir -p $out/opt/xilinx/xrt/lib
    if [ -d $out/opt/xilinx/lib ]; then
      mv $out/opt/xilinx/lib/* $out/opt/xilinx/xrt/lib/ || true
    fi

    # Create top-level symlinks
    mkdir -p $out/lib
    for f in $out/opt/xilinx/xrt/lib/*.so*; do
      ln -sf $f $out/lib/ 2>/dev/null || true
    done

    # Create pkg-config file
    mkdir -p $out/lib/pkgconfig
    cat > $out/lib/pkgconfig/xrt-amdxdna.pc << EOF
prefix=$out/opt/xilinx/xrt
exec_prefix=\''${prefix}
libdir=\''${exec_prefix}/lib
includedir=\''${prefix}/include

Name: XRT-AMDXDNA
Description: AMD XDNA Plugin for Xilinx Runtime
Version: ${finalAttrs.pluginVersion}
Requires: xrt
Libs: -L\''${libdir} -lamdxdna
EOF
  '';

  meta = {
    description = "AMD XDNA driver plugin for XRT (Ryzen AI NPU support)";
    homepage = "https://github.com/amd/xdna-driver";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
})
