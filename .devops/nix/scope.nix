{
  lib,
  newScope,
  releaseVersion ? null,
  revision,
  rocmPackages,
  rustToolchain,
}:

lib.makeScope newScope (self: {
  # XRT with its amdxdna plugin: the XDNA NPU runtime (src/core/npu).
  xrt-amdxdna = (self.callPackage ./xrt { }).combined;
  gufo = self.callPackage ./package.nix {
    inherit releaseVersion revision rocmPackages rustToolchain;
    xrt = self.xrt-amdxdna;
  };
  mkServe = self.callPackage ./mk-serve.nix { };
  extensionModules = self.callPackage ./extension-modules.nix { };
})
