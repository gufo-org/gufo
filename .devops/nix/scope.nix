{
  lib,
  newScope,
  releaseVersion ? null,
  revision,
  rocmPackages,
  rustToolchain,
}:

lib.makeScope newScope (self: {
  gufo = self.callPackage ./package.nix { inherit releaseVersion revision rocmPackages rustToolchain; };
  mkServe = self.callPackage ./mk-serve.nix { };
  extensionModules = self.callPackage ./extension-modules.nix { };
})
