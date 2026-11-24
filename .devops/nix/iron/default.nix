# The IRON toolchain for NPU designs (src/core/npu/kernels): mlir-aie's Python
# package (IRON, aiecc) and Peano (llvm-aie, the AIE compiler) as their
# released wheels, in one Python. mlir-aie 1.4.5 ships for CPython 3.13.
#
# llvm-aie publishes nightly wheels only; the pinned one may leave the
# `nightly` release, then update `peanoVersion` and its hash together.
{
  lib,
  stdenv,
  fetchurl,
  python313,
  autoPatchelfHook,
  zlib,
  zstd,
  libxml2,
  ncurses,
}:

let
  py = python313.pkgs;
  wheel =
    {
      pname,
      version,
      url,
      hash,
    }:
    py.buildPythonPackage {
      inherit pname version;
      format = "wheel";
      src = fetchurl { inherit url hash; };
      nativeBuildInputs = [ autoPatchelfHook ];
      buildInputs = [
        stdenv.cc.cc.lib
        zlib
        zstd
        libxml2
        ncurses
      ];
      # The wheels bundle their own LLVM and MLIR libraries.
      autoPatchelfIgnoreMissingDeps = [ "libpython3*" ];
      dontStrip = true;
      doCheck = false;
    };
  peanoVersion = "22.0.0.2026111101+81af12b7";
  mlir-aie =
    (wheel {
      pname = "mlir_aie";
      version = "1.4.5";
      url = "https://github.com/Xilinx/mlir-aie/releases/download/v1.4.5/mlir_aie-1.4.5-cp313-cp313-manylinux_2_35_x86_64.whl";
      hash = "sha256-63cUgzoqUIlYQFdp7t5YuI6fzvcUs3nmfxHjXHzynB4=";
    }).overridePythonAttrs
      (o: {
        dependencies = [
          py.numpy
          py.ml-dtypes
          py.cloudpickle
        ];
      });
  llvm-aie = wheel {
    pname = "llvm_aie";
    version = peanoVersion;
    url = "https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-${
      lib.replaceStrings [ "+" ] [ "%2B" ] peanoVersion
    }-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-ogcEezpbyYd2axcjBxPSsMmsBcBoJ4O8cosjRQ5pv2g=";
  };
  python = python313.withPackages (_: [
    mlir-aie
    llvm-aie
    py.numpy
    py.ml-dtypes
    py.cloudpickle
  ]);
in
python.overrideAttrs (o: {
  passthru = (o.passthru or { }) // {
    inherit mlir-aie llvm-aie;
    # Where aiecc finds Peano.
    peano = "${llvm-aie}/${python313.sitePackages}/llvm-aie";
  };
})
