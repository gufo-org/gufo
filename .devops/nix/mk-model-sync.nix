{
  lib,
  writeShellApplication,
  python3,
  util-linux,
  coreutils,
  gufo,
}:
{
  models,
  directory ? "/var/lib/gufo/models",
}:
let
  # Only models the package itself ships can be synchronized: the service never
  # installs extensions, so a registry-only model would fail on every run.
  known = map (model: model.id) (
    gufo.extensionMetadata.models
      or (throw "gufo model sync: the Gufo package exposes no extensionMetadata; build it with extensions instead of wrapping it")
  );
  unknown = lib.subtractLists known models;
  specification = builtins.toJSON (lib.unique models);
in
assert lib.assertMsg (lib.all builtins.isString models) "gufo model sync: models must be strings";
assert lib.assertMsg (
  unknown == [ ]
) "gufo model sync: the Gufo package offers no models ${lib.concatStringsSep ", " unknown}; it offers ${lib.concatStringsSep ", " known}";
assert lib.assertMsg (lib.hasPrefix "/" directory) "gufo model sync: directory must be absolute";
writeShellApplication {
  name = "gufo-model-sync";
  runtimeInputs = [
    python3
    util-linux
    coreutils
  ];
  text = ''
    export GUFO_MODELS_DIR=${lib.escapeShellArg directory}
    mkdir -p "$GUFO_MODELS_DIR"
    exec 9>"$GUFO_MODELS_DIR/.sync.lock"
    flock 9
    python3 ${./model-sync.py} ${lib.escapeShellArg "${gufo}/bin/gufo"} ${lib.escapeShellArg specification}
  '';
}
