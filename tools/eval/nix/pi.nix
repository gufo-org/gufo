# Pi coding agent for gufo-agent-eval.
#
# Taken from nixpkgs, so the version is pinned by the flake lock. Pi's
# revision is part of the benchmark identity, so the runner records the
# resolved version in the result rather than assuming it is stable.
#
# Nothing downloads at run time: the jail has no network, and the runner sets
# PI_OFFLINE=1, PI_SKIP_VERSION_CHECK=1, and PI_TELEMETRY=0 so Pi does not
# contact pi.dev even when run outside a jail during development.
{ pkgs ? import <nixpkgs> { } }:

pkgs.pi-coding-agent
