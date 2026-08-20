{
  lib,
  newScope,
  version,
}:

lib.makeScope newScope (self: {
  aie-qwen-mtp-eh-proj = self.callPackage ./aie-qwen-mtp-eh-proj.nix {
    inherit (self)
      aiebu
      llvm-aie
      mlir-aie
      xrt
      ;
  };
  aie-qwen-mtp-rmsnorm = self.callPackage ./aie-qwen-mtp-rmsnorm.nix {
    inherit (self)
      aiebu
      llvm-aie
      mlir-aie
      xrt
      ;
  };
  aie-smoke = self.callPackage ./aie-smoke.nix {
    inherit (self)
      aiebu
      llvm-aie
      mlir-aie
      xrt
      ;
  };
  aiebu = self.callPackage ./aiebu.nix { };
  strix = self.callPackage ./package.nix {
    inherit version;
    inherit (self) aie-qwen-mtp-eh-proj aie-qwen-mtp-rmsnorm aie-smoke;
  };
  xrt = self.callPackage ./xrt.nix { };
  xrt-plugin-amdxdna = self.callPackage ./xrt-plugin-amdxdna.nix {
    inherit (self) xrt;
  };
  llvm-aie = self.callPackage ./llvm-aie.nix { };
  mlir-aie = self.callPackage ./mlir-aie.nix { };
  hrx-system = self.callPackage ./hrx-system.nix { };
  hyperloom = self.callPackage ./hyperloom.nix { };
})
