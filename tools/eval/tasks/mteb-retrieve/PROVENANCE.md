# Provenance — `mteb-retrieve`

| Field | Value |
| --- | --- |
| Task | `terminal-bench/mteb-retrieve` |
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

This task cannot pass, and the blocker is a dependency conflict rather than
missing effort.

The instruction requires the agent to use `mteb` version 1.36.8. That release
requires `datasets<3.0.0,>=2.19.0`, while nixpkgs ships `datasets 4.5.0` --
a major version beyond what mteb accepts. Pinning an older `datasets` would
cascade through `transformers` and `sentence-transformers`, which nixpkgs also
carries at newer versions. `mteb` additionally requires
`pytrec-eval-terrier`, a C++ extension not packaged in nixpkgs.

Separately, the task downloads the `bge-small-zh-v1.5` model at revision
`7999e1d3359715c523056ef9478215996d62a620` at run time. The offline sandbox
does not allow that, so the model would need vendoring as a fixed-output
derivation. That part is tractable; the dependency conflict is not, without
building a separate pinned Python set for this one task.

The task is therefore excluded from the `known-good` tier, so a harness
limitation is not scored as a model failure.

The verifier contract is unchanged: CTRF at `/logs/verifier/ctrf.json`, reward
at `/logs/verifier/reward.txt`, pass only when reward is exactly 1.

## Not an official score

Gufo-owned suite running a different agent (Pi) in a different sandbox
(bubblewrap) than upstream. Not comparable to published Terminal-Bench
results.
