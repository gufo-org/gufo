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
  # mlir-aie (IRON) and Peano for NPU designs (src/core/npu/kernels).
  iron = self.callPackage ./iron { };
  # libloom_text_embed for the loom extension (src/models/loom).
  loom-runtime = self.callPackage ./hrx/loom-runtime.nix { };
  gufo = self.callPackage ./package.nix {
    inherit releaseVersion revision rocmPackages rustToolchain;
    xrt = self.xrt-amdxdna;
    inherit (self) iron;
    loomRuntime = self.loom-runtime;
  };
  mkServe = self.callPackage ./mk-serve.nix { };
  extensionModules = self.callPackage ./extension-modules.nix { };
})
