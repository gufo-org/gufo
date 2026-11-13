# The in-tree NPU designs (src/core/npu/kernels), compiled with IRON on their
# own (cmake/npu-designs). Its source is only the kernels and the CMake that
# builds them, so this derivation, and the Cachix entry for it, stay the same
# until a design changes; Gufo's package installs the result. The designs
# compile on any x86 host, no NPU needed; qwen3pre alone takes many minutes.
{
  lib,
  stdenv,
  cmake,
  ninja,
  iron,
  xrt,
  sourceRoot,
}:

let
  kept = [
    "cmake/GufoNpuDesigns.cmake"
    "cmake/GufoNpuDesignList.cmake"
    "cmake/npu-designs"
    "src/core/npu/kernels"
  ];
in
stdenv.mkDerivation {
  pname = "gufo-npu-designs";
  version = "0";
  src = lib.cleanSourceWith {
    name = "gufo-npu-designs-source";
    src = sourceRoot;
    filter =
      path: _type:
      let
        relativePath = lib.removePrefix "${toString sourceRoot}/" (toString path);
      in
      toString path == toString sourceRoot
      || builtins.any (
        keep: relativePath == keep || lib.hasPrefix "${keep}/" relativePath || lib.hasPrefix "${relativePath}/" keep
      ) kept;
  };
  cmakeDir = "../cmake/npu-designs";
  nativeBuildInputs = [ cmake ninja xrt ];
  cmakeFlags = [
    "-DGUFO_IRON_PYTHON=${iron}/bin/python3"
    "-DXRT_DIR=${xrt}/opt/xilinx/xrt/share/cmake/XRT"
  ];
  preConfigure = ''
    export PEANO_INSTALL_DIR=${iron.peano}
  '';
  # Writes $out/share/gufo/npu/<name>/.
  dontFixup = true;
}
