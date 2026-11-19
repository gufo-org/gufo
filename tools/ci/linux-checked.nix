# The candidate's checked package (assertions on) with the test executables
# of the given comma-separated CTest labels bundled for Devlab.
{ labels ? "" }:
let
  candidate = builtins.getFlake "git+file://${toString ../..}";
  package = candidate.packages.${builtins.currentSystem}.checked;
  selected = builtins.filter (label: label != "") (
    builtins.filter builtins.isString (builtins.split "," labels));
in
package.override { ctestLabels = selected; }
