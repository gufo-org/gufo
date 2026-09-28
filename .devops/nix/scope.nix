{
  lib,
  newScope,
  releaseVersion ? null,
  revision,
}:

lib.makeScope newScope (self: {
  gufo = self.callPackage ./package.nix { inherit releaseVersion revision; };
  mkServe = self.callPackage ./mk-serve.nix { };
})
