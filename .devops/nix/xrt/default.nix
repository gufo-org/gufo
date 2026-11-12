# XRT and its amdxdna plugin as a matched pair: xdna-driver main and the XRT
# commit it pins as a submodule. The derivations started from robcohen/nixpkgs
# (`vitis-ai` branch, MIT); the pins and the plugin patch are Gufo's.
#
# Gufo links `combined` on Linux (docs/extension-api.md#npu-programs).
{
  lib,
  callPackage,
  fetchFromGitHub,
}:

let
  xrtRev = "06c43a7f93fee129d32def1d372e17a8e69c05d2";
  xdnaRev = "010b7d2fbcb10b5412cdf829e0a53728f43de01d";
  xrtSrc = fetchFromGitHub {
    owner = "Xilinx";
    repo = "XRT";
    rev = xrtRev;
    fetchSubmodules = true;
    hash = builtins.readFile ./xrt.hash;
  };
  xdnaSrc = fetchFromGitHub {
    owner = "amd";
    repo = "xdna-driver";
    rev = xdnaRev;
    fetchSubmodules = false;
    hash = builtins.readFile ./xdna.hash;
  };
  xrt = (callPackage ./xrt.nix { }).overrideAttrs (o: {
    version = "2.26-${builtins.substring 0 7 xrtRev}";
    src = xrtSrc;
  });
  plugin = (callPackage ./xrt-plugin-amdxdna.nix { inherit xrt; }).overrideAttrs (o: {
    version = "2.26-${builtins.substring 0 7 xdnaRev}";
    pluginVersion = "2.26.0";
    src = xdnaSrc;
    inherit xrtSrc;
    # The 64 MiB device heap is mapped MAP_LOCKED, which needs an unlimited
    # memlock limit (8 MiB by default, root to raise). The driver pins the
    # pages itself; populate them instead.
    postPatch = (o.postPatch or "") + ''
      substituteInPlace src/shim/buffer.cpp --replace-fail "MAP_SHARED | MAP_LOCKED | flags" "MAP_SHARED | MAP_POPULATE | flags"
    '';
  });
in
{
  inherit xrt plugin;
  combined = callPackage ./xrt-amdxdna.nix {
    inherit xrt;
    xrt-plugin-amdxdna = plugin;
  };
}
