{
  lib,
  newScope,
  pkgs,
  version,
}:

lib.makeScope newScope (self: {
  gufo = self.callPackage ./package.nix { inherit version; };
  tp2-rdma = self.callPackage ./package.nix {
    inherit version;
    enableTp2Rdma = true;
    rdma-core = pkgs.rdma-core;
  };
  mkServe = self.callPackage ./mk-serve.nix { };
})
