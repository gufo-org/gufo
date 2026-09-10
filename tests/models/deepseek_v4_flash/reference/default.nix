# Build an unmodified upstream model with the same locked ROCm toolchain.
# The caller verifies the checkout revision and cleanliness before building.
{ upstream }:
let
  lock = builtins.fromJSON (builtins.readFile ../../../../flake.lock);
  source = builtins.fetchTree lock.nodes.nixpkgs.locked;
  pkgs = import source.outPath { system = "x86_64-linux"; };
  rocm = pkgs.rocmPackages;
in
pkgs.stdenv.mkDerivation {
  pname = "ds4-independent-reference";
  version = "6289c516";
  src = pkgs.lib.cleanSource (builtins.toPath upstream);
  nativeBuildInputs = [ pkgs.cmake pkgs.ninja rocm.clr ];
  buildInputs = [
    rocm.clr rocm.hipblas rocm.hipblaslt rocm.rocblas
    rocm.hipcub rocm.rocprim rocm.rocwmma
  ];
  postPatch = ''
    cp ${./CMakeLists.txt} CMakeLists.txt
    cp ${./frontiers.c} frontiers.c
  '';
  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_HIP_COMPILER=${rocm.llvm.clang}/bin/clang"
    "-DCMAKE_HIP_ARCHITECTURES=gfx1151"
    "-DHIPCUB_INCLUDE_DIR=${rocm.hipcub}/include"
    "-DROCPRIM_INCLUDE_DIR=${rocm.rocprim}/include"
    "-DROCWMMA_INCLUDE_DIR=${rocm.rocwmma}/include"
  ];
  env.ROCM_PATH = "${rocm.clr}";
  enableParallelBuilding = true;
}
