# Provenance — `mailman`

| Field | Value |
| --- | --- |
| Task | `terminal-bench/mailman` |
| Suite | Terminal-Bench 2.1, via [terminal-bench-mini](https://github.com/kyuz0/terminal-bench-mini) |
| Revision | `5c8eadf1f393183288fa08b8f73ca9a469cc5e00` |
| Licence | Apache-2.0 |

Upstream files carry the Terminal-Bench canary string, preserved verbatim and
added to files created during the port. Do not remove it.

## Byte-identical to upstream

`instruction.md`, the task fixtures, the hidden test fixtures, and
`solution/`.

## Changed by the port

| Upstream | Here | Why |
| --- | --- | --- |
| `environment/Dockerfile` | `environment/env.nix` | No OCI runtime; the rootfs is a Nix derivation. |
| `tests/test.sh` | `tests/verifier.nix` | The upstream bootstrap apt-installs curl and pipes `astral.sh` into a shell, which needs network at verify time. |
| `allow_internet = true` | `network = "none"` | Dependencies move from image-build time to Nix build time. |

## Blocked

Upstream installs `postfix`, `mailutils` and `mailman3` from apt and expects
them to run as managed services. The jail unshares the pid namespace and runs
the agent as pid 1 with no init, so nothing supervises daemons, and Postfix
in particular expects a running master process, a spool directory it owns,
and privilege separation across several uids.

The verifier compounds this. Upstream installs `mailman==3.3.8` from PyPI as
an importable library. nixpkgs carries `mailman` 3.3.10, but as a top-level
application rather than a member of the Python package set, so it is not
importable by the verifier's interpreter as written, and the version differs
from the pin.

Making this work means either running an init inside the jail or rewriting the
task to check configuration rather than a live mail flow. Both are task
changes, not ports, so the task is excluded from the `known-good` tier and
kept in `full` only to keep the gap visible.

The verifier contract is unchanged: CTRF at `/logs/verifier/ctrf.json`, reward
at `/logs/verifier/reward.txt`, pass only when reward is exactly 1.

## Not an official score

Gufo-owned suite running a different agent (Pi) in a different sandbox
(bubblewrap) than upstream. Not comparable to published Terminal-Bench
results.
