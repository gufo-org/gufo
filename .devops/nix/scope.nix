{
  lib,
  newScope,
  releaseVersion ? null,
  revision,
  rocmPackages,
}:

lib.makeScope newScope (self: {
  gufo = self.callPackage ./package.nix { inherit releaseVersion revision rocmPackages; };
  mkServe = self.callPackage ./mk-serve.nix { };
})
