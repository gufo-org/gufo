{
  lib,
  newScope,
  pkgs,
  releaseVersion ? null,
  revision,
}:

lib.makeScope newScope (self: {
  gufo = self.callPackage ./package.nix { inherit releaseVersion revision; };
  tp2-rdma = self.callPackage ./package.nix {
    inherit releaseVersion revision;
    enableTp2Rdma = true;
    rdma-core = pkgs.rdma-core;
  };
  mkServe = self.callPackage ./mk-serve.nix { };
})
