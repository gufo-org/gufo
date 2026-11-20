{ self }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.gufo-model-sync;
  sync = pkgs.callPackage ./mk-model-sync.nix { gufo = cfg.package; } {
    inherit (cfg) models directory;
  };
in
{
  options.services.gufo-model-sync = {
    enable = lib.mkEnableOption "scheduled Gufo model synchronization";
    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      description = "Gufo package whose extensions offer the models; also used to install and update them.";
    };
    models = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Model IDs to track; each must be offered by an extension built into the package.";
    };
    directory = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/gufo/models";
      description = "Absolute directory shared by model sync and inference processes.";
    };
    user = lib.mkOption {
      type = lib.types.str;
      default = "gufo";
      description = "Existing service user, or gufo to create a dedicated system account.";
    };
    group = lib.mkOption {
      type = lib.types.str;
      default = "gufo";
      description = "Group that can read synchronized models.";
    };
    onCalendar = lib.mkOption {
      type = lib.types.str;
      default = "daily";
      description = "Systemd calendar expression for synchronization.";
    };
    randomizedDelaySec = lib.mkOption {
      type = lib.types.str;
      default = "30m";
      description = "Maximum random delay after the scheduled time.";
    };
  };
  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.models != [ ];
        message = "services.gufo-model-sync.models must contain at least one model ID";
      }
    ];
    users.users = lib.mkIf (cfg.user == "gufo") {
      gufo = {
        isSystemUser = true;
        group = cfg.group;
        home = "/var/lib/gufo";
      };
    };
    users.groups = lib.mkIf (cfg.group == "gufo") { gufo = { }; };
    systemd.services.gufo-model-sync = {
      description = "Synchronize declared Gufo models";
      # Only the timer starts this unit. A switch must not trigger downloads.
      restartIfChanged = false;
      stopIfChanged = false;
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      environment = {
        GUFO_MODELS_DIR = cfg.directory;
        HOME = config.users.users.${cfg.user}.home;
      };
      serviceConfig = {
        Type = "oneshot";
        User = cfg.user;
        Group = cfg.group;
        StateDirectory = "gufo";
        StateDirectoryMode = "0750";
        UMask = "0027";
        ExecStart = lib.getExe sync;
        TimeoutStartSec = "infinity";
      };
    };
    systemd.timers.gufo-model-sync = {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.onCalendar;
        RandomizedDelaySec = cfg.randomizedDelaySec;
        Persistent = false;
        Unit = "gufo-model-sync.service";
      };
    };
  };
}
