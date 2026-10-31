{ rustPlatform }:

let
  vendor = rustPlatform.importCargoLock {
    lockFile = ../../src/core/png_decode/Cargo.lock;
  };
in
''
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
