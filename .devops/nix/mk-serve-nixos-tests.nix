# Evaluate real NixOS service units with the actual default/release packages.
# This checks module integration without realizing either GPU closure.
{
  pkgs,
  defaultPackage,
  releasePackage,
}:
let
  inherit (pkgs) lib;
  evaluate =
    package:
    let
      command =
        import ./mk-serve.nix
          {
            inherit lib;
            gufo = package;
          }
          {
            model = "/var/lib/gufo/model.gguf";
            provider = "flash-next-q4";
            speculative = "mtp";
            mtpModel = "/var/lib/gufo/draft.gguf";
            port = 8080;
          };
      os = import (pkgs.path + "/nixos/lib/eval-config.nix") {
        inherit pkgs;
        system = pkgs.stdenv.hostPlatform.system;
        modules = [
          {
            boot.isContainer = true;
            system.stateVersion = "26.05";
            systemd.services.gufo = {
              wantedBy = [ "multi-user.target" ];
              serviceConfig = {
                ExecStart = command;
                User = "gufo";
                Restart = "on-failure";
              };
            };
            users.users.gufo = {
              isSystemUser = true;
              group = "gufo";
            };
            users.groups.gufo = { };
          }
        ];
      };
      unit = os.config.systemd.services.gufo;
    in
    unit.serviceConfig.ExecStart == command
    && lib.hasPrefix "${package}/bin/gufo serve --port 8080 llm" command
    && lib.hasInfix "--mtp-model /var/lib/gufo/draft.gguf" command
    && unit.serviceConfig.User == "gufo"
    && builtins.elem "multi-user.target" unit.wantedBy;
in
assert lib.assertMsg (evaluate defaultPackage) "default mkServe NixOS unit evaluation failed";
assert lib.assertMsg (evaluate releasePackage) "release mkServe NixOS unit evaluation failed";
assert lib.assertMsg (
  !defaultPackage.extensionsFromLock && releasePackage.extensionsFromLock
) "default/release extension selection changed";
assert lib.assertMsg (
  defaultPackage.extensionMetadata.selectionMode == "discovery"
  && releasePackage.extensionMetadata.selectionMode == "lock"
) "serving metadata selection changed";
assert lib.assertMsg (
  defaultPackage.releaseVersion == null && releasePackage.releaseVersion != null
) "default/release version policy changed";
true
