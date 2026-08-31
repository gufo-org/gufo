# Provenance — `openssl-selfsigned-cert`

| Field | Value |
| --- | --- |
| Task | `terminal-bench/openssl-selfsigned-cert` |
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



The verifier contract is unchanged: CTRF at `/logs/verifier/ctrf.json`, reward
at `/logs/verifier/reward.txt`, pass only when reward is exactly 1.

## Not an official score

Gufo-owned suite running a different agent (Pi) in a different sandbox
(bubblewrap) than upstream. Not comparable to published Terminal-Bench
results.
