# Build baseline source with the candidate's package definition and toolchain.
{ source, revision }:
let
  candidate = builtins.getFlake "git+file://${toString ../..}";
  package = candidate.packages.${builtins.currentSystem}.default;
in
assert builtins.match "[0-9a-f]{40}" revision != null;
package.override {
  sourceRoot = builtins.toPath source;
  revision = builtins.substring 0 7 revision;
}
