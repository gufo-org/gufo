# Offline Cargo setup for the Rust components. The vendored crates follow the
# lockfile of the source being built, so a baseline built from another revision
# (tools/ci/linux-baseline.nix) gets its own pinned crates, or none before the
# Rust components existed.
{
  lib,
  rustPlatform,
  lockFile ? ../../Cargo.lock,
}:

let
  vendor = rustPlatform.importCargoLock { inherit lockFile; };
in
lib.optionalString (builtins.pathExists lockFile) ''
  export CARGO_HOME="$TMPDIR/cargo-home"
  mkdir -p "$CARGO_HOME"
  cat > "$CARGO_HOME/config.toml" <<CARGO_CONFIG
  [source.crates-io]
  replace-with = "vendored-sources"
  [source.vendored-sources]
  directory = "${vendor}"
  [net]
  offline = true
  CARGO_CONFIG
''
