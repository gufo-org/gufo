# Pi coding agent for gufo-agent-eval.
#
# Pi has to exist before the jail starts: the jail runs with --unshare-net, so
# nothing can be fetched once it is inside. This derivation is how it gets
# materialized -- the flake app and the dev shell both build it and export
# GUFO_EVAL_PI, so the packaged and development paths run the same agent.
#
# Taken from nixpkgs, so the version is pinned by the flake lock. Pi's
# revision is part of the benchmark identity, so the runner records the
# resolved version in the result rather than assuming it is stable.
{ pkgs ? import <nixpkgs> { } }:

pkgs.pi-coding-agent
