# Card 25 evidence: metadata-driven Nix serving

The implementation derives provider options, speculative modes, modality
availability, required draft paths and named-voice support from manifests.
Host network and systemd policy remain caller-owned. Existing canonical draft
argument names are preserved without aliases or public argument renames.

## Retained checks

| Check | Result | Evidence |
| --- | --- | --- |
| Nix command rendering and metadata evaluation | PASS: 41 positive/negative cases | `nix-checks.txt`; `.devops/nix/mk-serve-tests.nix` |
| Default/release NixOS service evaluation | PASS: actual package metadata, executable selection and systemd service configuration | `nix-checks.txt`; `.devops/nix/mk-serve-nixos-tests.nix` |
| Production CMake metadata contracts | PASS: 13 tests, including draft ownership/type/command validation | `metadata-tests.txt` |
| Independent JSON Schema validation | PASS | `schema-tests.txt` |
| Extension API policy | PASS, including guidance symlink exclusion and descriptive-record normalization | `api-policy-tests.txt` |
| Candidate GPU `gufo` target | PASS; final reconfiguration needs no recompilation | `gpu-build.txt`; `gpu-build-final.txt` |
| Production default and release packages | PASS: builds, provenance diagnostics and matching current output paths | `packages.txt`; `package-paths.json`; `default-version.txt`; `release-version.txt`; `default-extensions.json`; `release-extensions.json` |
| Generated-command systemd/HTTP smoke on the NixOS GPU host | PASS: external LFM3 provider, model listing and chat completion | `systemd-smoke.json`; replay in `systemd-smoke.py` |

Commands:

```sh
nix build --offline .#default .#release --no-link --max-jobs 1 --cores 4
nix build --offline .#checks.x86_64-linux.mk-serve .#checks.x86_64-linux.mk-serve-nixos --no-link
nix develop --offline -c python3 tests/extensions/metadata_test.py
nix develop --offline -c python3 tests/extensions/metadata_test.py --schema
nix develop --offline -c python3 tests/tools/check_extension_api_test.py
nix develop --offline -c cmake --build --preset gpu-test --target gufo --parallel 4
python3 docs/model-extensions/evidence/card25/systemd-smoke.py
python3 tools/ci/generate-infrastructure.py --check
python3 tools/ci/check-extension-api.py --base HEAD --labels '["ext-api"]'
nix develop --offline -c python3 tools/ci/check-docs.py
```

The GPU build uses the existing discovery configuration with
`GUFO_EXTENSION_DIRS=/home/mixer/gufo/lfm3-gufo-extension`. The smoke report records
the actual external revision, candidate binary hash, rendered command, service
journal and HTTP output. Its temporary service is stopped after the check.
The initial 32-token smoke request exhausted its budget in reasoning, leaving
final content empty; `systemd-smoke-initial.json` retains that result. The
128-token rerun produced final content. This is a startup/API smoke check, not
numerical or timing qualification. Compiler warnings in the retained build log
are not represented as a warning-free build.

The initial direct CMake test attempts lacked CMake, then Make/OpenSSL in the
editing shell. The retained successful run uses the pinned `nix develop`
environment. A production package build from the preceding numbered-record
snapshot was interrupted when the record-format requirement changed; final
package results are recorded separately.

## Boundaries

`provider` validates metadata at evaluation time; runtime artifact probing still
selects the model. The actual serving parser retains its existing canonical
flags and speculative modes. A newly declared runtime control needs its own CLI
integration; Nix rendering alone does not implement it.

The NixOS check evaluates service definitions; the live smoke uses a transient
user service on the NixOS GPU host. No VM boot or all-model correctness/speed
sweep is claimed. No inference implementation or numerical contract changed.
