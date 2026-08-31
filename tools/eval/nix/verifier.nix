# Shared verifier-closure builder for gufo-agent-eval.
#
# Replaces the upstream Terminal-Bench `tests/test.sh` bootstrap, which ran
# `apt-get install curl` and piped `astral.sh/uv/install.sh` into a shell
# before invoking `uvx pytest`. That needs network at verify time and pins
# nothing reproducibly.
#
# Verifier dependencies are per task: `sparql-university` needs rdflib,
# `mailman` needs mailman, most tasks need neither. Each task therefore has
# its own `tests/verifier.nix` calling this builder.
#
# The observable contract is unchanged from upstream:
#   - CTRF JSON written to /logs/verifier/ctrf.json
#   - reward written to /logs/verifier/reward.txt
#   - reward is exactly 1 on pass, 0 otherwise
{ pkgs }:

{
  # Task name, used for the derivation name only.
  name,
  # Function from a python package set to the task's extra dependencies.
  pythonDeps ? (_ps: [ ]),
}:

let
  # Not in nixpkgs. Upstream Terminal-Bench pins 0.3.5 and the CTRF report is
  # part of the result contract, so it is packaged here rather than dropped.
  pytest-json-ctrf =
    ps:
    ps.buildPythonPackage rec {
      pname = "pytest-json-ctrf";
      version = "0.3.5";
      pyproject = true;
      build-system = [ ps.setuptools ];
      src = pkgs.fetchPypi {
        pname = "pytest_json_ctrf";
        inherit version;
        sha256 = "1zypkdvz6p4rm2w3v9mc38j59493z8fpn89i63vnkyyidpy3mz1y";
      };
      propagatedBuildInputs = [ ps.pytest ];
      doCheck = false;
    };

  python = pkgs.python313.withPackages (
    ps:
    [
      ps.pytest
      (pytest-json-ctrf ps)
    ]
    ++ pythonDeps ps
  );
in

pkgs.writeShellApplication {
  name = "gufo-agent-eval-verify-${name}";
  runtimeInputs = [ python pkgs.coreutils ];
  text = ''
    # Runs inside the verifier jail. The agent's workspace is mounted at its
    # task workdir; the hidden fixtures and this task's tests are at /tests.
    # The agent jail never has /tests in its mount namespace.
    mkdir -p /logs/verifier

    set +e
    pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
    status=$?
    set -e

    if [ $status -eq 0 ]; then
      echo 1 > /logs/verifier/reward.txt
    else
      echo 0 > /logs/verifier/reward.txt
    fi

    # The verifier itself succeeded in producing a verdict; a failed task is
    # reward 0, not a crashed verifier. The runner distinguishes the two.
    exit 0
  '';
}
