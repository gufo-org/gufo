{ pkgs, defaultPackage }:
let
  inherit (pkgs) lib;
  known = map (model: model.id) defaultPackage.extensionMetadata.models;
  mk = pkgs.callPackage ./mk-model-sync.nix { gufo = defaultPackage; };
  wrapped = pkgs.callPackage ./mk-model-sync.nix {
    gufo = pkgs.writeShellScriptBin "gufo" "";
  };
  fails = args: !(builtins.tryEval (builtins.deepSeq (mk args).name true)).success;
  os = import (pkgs.path + "/nixos/lib/eval-config.nix") {
    inherit pkgs;
    system = pkgs.stdenv.hostPlatform.system;
    modules = [
      (import ./model-sync-module.nix {
        self.packages.${pkgs.stdenv.hostPlatform.system}.default = defaultPackage;
      })
      {
        boot.isContainer = true;
        system.stateVersion = "26.05";
        services.gufo-model-sync = {
          enable = true;
          models = [ "qwen27-q4" ];
          onCalendar = "weekly";
        };
      }
    ];
  };
  service = os.config.systemd.services.gufo-model-sync;
  timer = os.config.systemd.timers.gufo-model-sync;
in
assert builtins.elem "qwen27-q4" known;
assert lib.all (id: builtins.elem id known) [ "qwen3-tts" "minimax-h3" ];
assert (mk { models = [ "qwen3-tts" "minimax-h3" ]; }).name == "gufo-model-sync";
assert fails { models = [ "typo-model" ]; };
assert fails {
  models = [ "qwen27-q4" ];
  directory = "relative";
};
# A package without build metadata cannot prove which models it offers.
assert !(builtins.tryEval (builtins.deepSeq (wrapped { models = [ "qwen27-q4" ]; }).name true)).success;
assert (mk { models = [ "qwen27-q4" ]; }).name == "gufo-model-sync";
assert service.wantedBy == [ ];
assert !service.restartIfChanged && !service.stopIfChanged;
assert service.serviceConfig.Type == "oneshot";
assert !(service.serviceConfig.RemainAfterExit or false);
assert timer.wantedBy == [ "timers.target" ];
assert timer.timerConfig.OnCalendar == "weekly";
assert !timer.timerConfig.Persistent;
assert !(timer.timerConfig ? OnBootSec);
assert !(timer.timerConfig ? OnStartupSec);
assert lib.hasSuffix "/bin/gufo-model-sync" service.serviceConfig.ExecStart;
assert lib.all (a: a.assertion) os.config.assertions;
true
