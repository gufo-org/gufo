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
  opencl-headers,
  opencl-clhpp,
  ocl-icd,
  rapidjson,
  protobuf,
  elfutils,
  libdrm,
  systemd,
  curl,
  openssl,
  libuuid,
  libxcrypt,
  ncurses,
  libsystemtap,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "xrt";
  version = "202610.2.21.21";

  src = fetchFromGitHub {
    owner = "Xilinx";
    repo = "XRT";
    rev = finalAttrs.version;
    hash = "sha256-Foj33/U6waL81EzJ0ah66xCXEGWEkvhwmurKobfCevE=";
    fetchSubmodules = true;
  };

  # Python with packages needed by spec_tool.py during build
  pythonEnv = python3.withPackages (ps: [
    ps.pyyaml
    ps.markdown
    ps.jinja2
    ps.pybind11
  ]);

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    git
    finalAttrs.pythonEnv
  ];

  buildInputs = [
    boost
    opencl-headers
    opencl-clhpp
    ocl-icd
    rapidjson
    protobuf
    elfutils
    libdrm
    systemd
    curl
    openssl
    libuuid
    libxcrypt
    ncurses
    libsystemtap
  ];

  cmakeDir = "../src";

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_INSTALL_PREFIX" "${placeholder "out"}/opt/xilinx/xrt")
    (lib.cmakeFeature "XRT_INSTALL_PREFIX" "${placeholder "out"}/opt/xilinx/xrt")
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeBool "DISABLE_WERROR" true)
    # Build for NPU only, disable Alveo PCIe components (avoids /usr/src DKMS installs)
    (lib.cmakeBool "XRT_NPU" true)
    (lib.cmakeBool "XRT_ALVEO" false)
    # Disable kernel module building (we use mainline amdxdna)
    (lib.cmakeFeature "XRT_DKMS_DRIVER_SRC_BASE_DIR" "")
    # Redirect DKMS install dir to output (prevents /usr/src install attempt)
    (lib.cmakeFeature "XRT_DKMS_INSTALL_DIR" "${placeholder "out"}/share/xrt-dkms")
    # XRT_UPSTREAM_DEBIAN enables XRT_UPSTREAM which propagates to AIEBU_UPSTREAM
    # This disables static linking in aiebu tools
    (lib.cmakeBool "XRT_UPSTREAM_DEBIAN" true)
    # Override install dirs to relative paths
    (lib.cmakeFeature "CMAKE_INSTALL_LIBDIR" "lib")
    (lib.cmakeFeature "CMAKE_INSTALL_BINDIR" "bin")
    (lib.cmakeFeature "CMAKE_INSTALL_INCLUDEDIR" "include")
    # Enable Python bindings (pyxrt)
    (lib.cmakeBool "XRT_ENABLE_PYXRT" true)
    (lib.cmakeFeature "Python3_EXECUTABLE" "${finalAttrs.pythonEnv}/bin/python3")
    (lib.cmakeFeature "Python3_INCLUDE_DIR" "${finalAttrs.pythonEnv}/include/python${python3.pythonVersion}")
    (lib.cmakeFeature "Python3_LIBRARY" "${finalAttrs.pythonEnv}/lib/libpython${python3.pythonVersion}.so")
    (lib.cmakeFeature "PYTHON_EXECUTABLE" "${finalAttrs.pythonEnv}/bin/python3")
  ];

  postPatch = ''
    # Fix Python3 detection for pybind11/pyxrt build
    substituteInPlace src/python/pybind11/CMakeLists.txt \
      --replace-quiet '/usr/bin/python3' "${finalAttrs.pythonEnv}/bin/python3" || true

    # Remove kernel module references
    substituteInPlace src/CMakeLists.txt \
      --replace-quiet 'add_subdirectory(runtime_src/core/pcie/driver)' '#add_subdirectory(runtime_src/core/pcie/driver)' || true

    # Fix /etc/os-release access - create a fake one for the build
    mkdir -p $TMPDIR/etc
    echo 'ID=nixos' > $TMPDIR/etc/os-release
    echo 'VERSION_ID="25.11"' >> $TMPDIR/etc/os-release

    # Patch CMake scripts that try to read /etc/os-release
    find . -name "*.cmake" -o -name "CMakeLists.txt" | xargs sed -i \
      -e 's|/etc/os-release|'$TMPDIR'/etc/os-release|g' || true

    # Fix hardcoded /usr/src DKMS install paths - redirect to output
    for f in src/CMake/version.cmake src/CMake/dkms.cmake src/CMake/dkms-edge.cmake; do
      substituteInPlace "$f" \
        --replace-fail 'set (XRT_DKMS_INSTALL_DIR "/usr/src/xrt-''${XRT_VERSION_STRING}")' \
                       'set (XRT_DKMS_INSTALL_DIR "${placeholder "out"}/share/xrt-dkms")' || true
    done
    substituteInPlace src/CMake/dkms-aws.cmake \
      --replace-fail 'set(XRT_DKMS_AWS_INSTALL_DIR "/usr/src/xrt-aws-''${XRT_VERSION_STRING}")' \
                     'set(XRT_DKMS_AWS_INSTALL_DIR "${placeholder "out"}/share/xrt-aws-dkms")' || true

    # Fix OpenCL ICD installation path
    substituteInPlace src/CMake/icd.cmake \
      --replace-fail 'set(OCL_ICD_INSTALL_PREFIX "/etc/OpenCL/vendors")' \
                     'set(OCL_ICD_INSTALL_PREFIX "${placeholder "out"}/etc/OpenCL/vendors")'

    # Fix hardcoded /usr/local/bin install paths for xbflash tools
    substituteInPlace src/runtime_src/core/pcie/tools/xbflash.qspi/CMakeLists.txt \
      --replace-fail 'set(XBFLASH_INSTALL_DEST "/usr/local/bin")' \
                     'set(XBFLASH_INSTALL_DEST "''${CMAKE_INSTALL_PREFIX}/bin")'
    substituteInPlace src/runtime_src/core/tools/xbflash2/CMakeLists.txt \
      --replace-fail 'set(XBFLASH_INSTALL_DEST "/usr/local/bin")' \
                     'set(XBFLASH_INSTALL_DEST "''${CMAKE_INSTALL_PREFIX}/bin")'

    # Disable Werror globally
    find . -name "CMakeLists.txt" -exec sed -i 's/-Werror//g' {} \; || true

    # Create stub markdown_graphviz_svg.py module to avoid network download
    cat > src/runtime_src/core/common/aiebu/specification/markdown_graphviz_svg.py << 'PYEOF'
# Stub implementation of markdown_graphviz_svg for Nix build
from markdown.extensions import Extension

class GraphvizBlocksExtension(Extension):
    """Stub extension - graphviz rendering is disabled in Nix build"""
    def extendMarkdown(self, md):
        pass

GraphvizExtension = GraphvizBlocksExtension

def makeExtension(**kwargs):
    return GraphvizBlocksExtension(**kwargs)
PYEOF

    # Replace wget command in specification CMakeLists.txt
    find . -name "CMakeLists.txt" -exec grep -l "wget" {} \; | while read f; do
      sed -i 's|COMMAND wget|COMMAND true # wget|g' "$f"
      sed -i 's|COMMAND powershell wget|COMMAND true # powershell wget|g' "$f"
    done

    # Disable spec generation targets during install
    specCmake="src/runtime_src/core/common/aiebu/specification/aie2ps/CMakeLists.txt"
    if [ -f "$specCmake" ]; then
      cat > "$specCmake" << 'STUBCMAKE'
# SPDX-License-Identifier: MIT
# Disabled for Nix build - spec generation causes issues
message(STATUS "Skipping aie2ps spec generation (Nix build)")
STUBCMAKE
    fi

    # Fix shebangs for Python scripts
    patchShebangs --build src/runtime_src/core/common/aiebu/specification/
    patchShebangs --build src/runtime_src/core/common/aiebu/src/python/ || true
  '';

  postInstall = ''
    # Create convenience symlinks at top level
    mkdir -p $out/bin $out/lib $out/include

    # Link binaries
    for bin in $out/opt/xilinx/xrt/bin/*; do
      if [ -f "$bin" ] || [ -L "$bin" ]; then
        ln -sf "$bin" $out/bin/
      fi
    done

    # Link libraries
    for f in $out/opt/xilinx/xrt/lib/*.so*; do
      if [ -f "$f" ] || [ -L "$f" ]; then
        ln -sf "$f" $out/lib/
      fi
    done

    # Copy setup script
    cp $out/opt/xilinx/xrt/setup.sh $out/ || true

    # Create pkg-config file
    mkdir -p $out/lib/pkgconfig
    cat > $out/lib/pkgconfig/xrt.pc << EOF
prefix=$out/opt/xilinx/xrt
exec_prefix=\''${prefix}
libdir=\''${exec_prefix}/lib
includedir=\''${prefix}/include

Name: XRT
Description: Xilinx Runtime for AMD NPU
Version: ${finalAttrs.version}
Libs: -L\''${libdir} -lxrt_coreutil
Cflags: -I\''${includedir}
EOF
  '';

  meta = {
    description = "Xilinx Runtime (XRT) for AMD Ryzen AI NPU";
    homepage = "https://github.com/Xilinx/XRT";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
})
