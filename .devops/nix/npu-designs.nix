# The in-tree NPU designs (src/core/npu/kernels), compiled with IRON on their
# own (cmake/npu-designs), one derivation per design. Changing one kernel
# preserves cached outputs for the others; Gufo installs their combined result.
# Compilation needs no NPU; qwen3pre alone takes many minutes.
{
  lib,
  stdenv,
  cmake,
  ninja,
  iron,
  hostCxx,
  xrt,
  sourceRoot,
  runCommand,
}:

let
  # Read the canonical CMake list rather than maintaining a second design list.
  declarations = builtins.filter builtins.isList (
    builtins.split "gufo_npu_design\\(([a-z0-9_]+)" (
      builtins.readFile (sourceRoot + "/cmake/GufoNpuDesignList.cmake")
    )
  );
  names = map builtins.head declarations;
  buildDesign =
    name:
    let
      kept = [
        "cmake/GufoNpuDesigns.cmake"
        "cmake/GufoNpuDesignList.cmake"
        "cmake/npu-designs"
        "src/core/npu/kernels/${name}"
      ];
    in
    stdenv.mkDerivation (
      {
        pname = "gufo-npu-design-${name}";
        version = "0";
        src = lib.cleanSourceWith {
          name = "gufo-npu-design-${name}-source";
          src = sourceRoot;
          filter =
            path: _type:
            let
              relativePath = lib.removePrefix "${toString sourceRoot}/" (toString path);
            in
            toString path == toString sourceRoot
            || builtins.any (
              keep:
              relativePath == keep
              || lib.hasPrefix "${keep}/" relativePath
              || lib.hasPrefix "${relativePath}/" keep
            ) kept;
        };
        cmakeDir = "../cmake/npu-designs";
        nativeBuildInputs = [
          cmake
          ninja
          xrt
        ];
        cmakeFlags = [
          "-DGUFO_NPU_DESIGN=${name}"
          "-DGUFO_NPU_HOST_CXX=${hostCxx}/bin/clang++"
          "-DGUFO_IRON_PYTHON=${iron}/bin/python3"
          "-DXRT_DIR=${xrt}/opt/xilinx/xrt/share/cmake/XRT"
        ];
        preConfigure = ''
          export PEANO_INSTALL_DIR=${iron.peano}
        '';
        dontFixup = true;
      }
      //
        lib.optionalAttrs
          (
            !(lib.hasInfix "if(GUFO_NPU_DESIGN AND" (
              builtins.readFile (sourceRoot + "/cmake/GufoNpuDesigns.cmake")
            ))
          )
          {
            # Timing baselines may predate the per-design CMake selector.
            postPatch = ''
              substituteInPlace cmake/GufoNpuDesigns.cmake \
                --replace-fail 'function(gufo_npu_design name)' 'function(gufo_npu_design name)
                if(GUFO_NPU_DESIGN AND NOT name STREQUAL GUFO_NPU_DESIGN)
                  return()
                endif()'
            '';
          }
    );
  designs = map buildDesign names;
in
runCommand "gufo-npu-designs" { passthru = { inherit designs; }; } (
  "mkdir -p $out/share/gufo/npu\n"
  + lib.concatMapStringsSep "\n" (design: ''
    ln -s ${design}/share/gufo/npu/* $out/share/gufo/npu/
  '') designs
)
