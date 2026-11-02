# Card 08 — pending evidence (PENDING, never recorded as passes)

Per the run plan, Card 08 runs unit, identity, CPU, hosted and static gates
only. Functional-harness runs, `prompt_lookup.py`, main-control matched
timings and speed benchmarks are deferred to the single qualification window
after Card 14, on the candidate with main as the matched control. Each row is
the exact command to run there; none of it was executed for this card.

## Deferred functional and performance rows

| ID | Command (verbatim) | Why it is deferred |
| --- | --- | --- |
| P1 | `nix develop -c python3 tests/functional/run.py --binary build/card08-release/gufo --model models/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf --suite all` | text-API qualification for the shared chat-template/comparison paths; functional window |
| P2 | `nix develop -c python3 tests/functional/run.py --binary build/card08-release/gufo --model models/deepseek-v4-flash --suite all --cache` | DeepSeek text support now comes from `gufo_deepseek_v4_flash_text` in HIP builds; `long-context` and `cache` are required because the sampler moved |
| P3 | `nix develop -c python3 tests/functional/continuation.py --binary build/card08-release/gufo` | continuation behaviour over the moved sampler |
| P4 | `nix develop -c python3 tests/functional/prompt_lookup.py --binary build/card08-release/gufo` | explicitly out-of-band for this card |
| P5 | `nix develop -c python3 tests/functional/pi_agent.py --binary build/card08-release/gufo --model <pinned agent model>` | agent-loop replay; retain the sessions and inspect actual tool results |
| P6 | `result/bin/gufo bench …` per [docs/BENCHMARKS.md](../../../BENCHMARKS.md) on the candidate and on a matched main control, per-request phase timings at 5 % and 10 ms | `gufo bench` is the shipped consumer of `gufo_compare`; timing belongs to the qualification window |
| P7 | ~~PENDING~~ **RUN by the orchestrator after the closing commit** (2026-10-31): `nix build .#checks.x86_64-linux.pr --no-link --print-out-paths` exit 0 with freshly built `check-docs`/`check-format`/`check-pr-tests`/`check-pr`; `nix eval --raw .#packages.x86_64-linux.release.src` shows the H3 tier, the core fixture present, the old model path gone and exactly one `tests/` file admitted. Evidence: [`hosted-post-commit.txt`](hosted-post-commit.txt). The pre-commit row in `gates.md` stays non-authoritative. | Hosted coverage of the committed revision established |

## Blocked on assets on this host (PENDING, not passes)

| ID | Command (verbatim) | Blocking condition |
| --- | --- | --- |
| P8 | `GUFO_H3_MODEL_ROOT=<h3 weights> GUFO_H3_SOURCE_MANIFEST=$PWD/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json nix develop -c ctest --test-dir build/card08-gpu -R '^minimax_h3_(runtime_hip_test|prompt_encoder_real|dit_block0_real|denoiser_smoke|denoiser_oracle|denoiser_production_shape_oracle|denoiser_reuse_smoke|video_vae_real|audio_vae_real)$' --output-on-failure` | the nine H3 rows skipped with exit 77: `SKIP: set GUFO_H3_MODEL_ROOT and GUFO_H3_SOURCE_MANIFEST`; no `models/minimax-*` weights exist here |
| P9 | `nix build .#packages.x86_64-linux.windows-platform-tests` | fails on the parent revision too: `cmake/PlatformTests.cmake` registers `platform_pages_test` from `tests/core/platform_pages_test.cpp`, but the `src` fileset of `.devops/nix/windows-platform-tests.nix` admits only `platform_fs_test.cpp` and `platform_mapped_file_test.cpp`. Neither file is touched by Card 08; the Windows fileset needs its own fix, and the full Windows build/CTest gate stays with `.github/workflows/windows-build.yml` |

## What already covers the moved paths without waiting

- `eval_http_test` (`LABELS cpu;quality;eval;http;cli;server`) runs
  `tests/tools/test_eval_http.py --gufo $<TARGET_FILE:gufo> --data
  tests/fixtures/eval/antirez-ds4.json`, i.e. the real `gufo` executable
  against the moved core fixture. It passed on `build/card08-cpu`.
- `openai_chat_test`, `ds4.template`, `ds4.sampling`, `ds4.cli`, `ds4.eval` and
  `ds4.dataset` exercise the single DeepSeek text implementation and the moved
  dataset from the test side, and passed on both candidate configurations.
- `tools/prof/card08_parity.py --mode device` plus
  `tools/ci/check-kernel-resources.py` prove the moved H3 HIP tier is
  byte-identical and uses the same scratch, which bounds what P8 could reveal.

## Distribution and integration gates

- P10: `nix build .#packages.x86_64-linux.release --no-link` — PENDING:
  full HIP package build, distribution-level gate; previously only path-evaluated.
  Local `build/card08-release/gufo` is the in-card production artifact; the
  earlier source-filter evaluation records the install-rule dataset path, not
  an end-to-end package build.
- Sandbox-blocked integration reruns and exact commands are authoritative in
  [integration-reruns.txt](integration-reruns.txt). No blocked command is a pass.
