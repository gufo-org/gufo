# Card 10 evidence — provenance

Repository `/home/mixer/gufo`, jj. Parent revision `klwmvoxr a761679a`
("model-extensions | chore(extensions): merge main and resolve integration conflicts").
The Card 10 work is the entire uncommitted working copy; nothing else was touched.
`docs/model-extensions/RFC.md` stayed byte-identical and uncommitted.

Host: Strix Halo (gfx1151), Linux x86_64. Nix toolchain with `nix develop -c` and
host python interpreter for static repository checks.

Agents (three distinct agents, sequential, then orchestrator closure):

| Role | Agent | Run ID | Verdict / Outcome |
| --- | --- | --- | --- |
| Implementation | `worker` | `6d5263a1-9f0d-46ef-a092-76804e8f1003` | Complete, handed off to reviewer |
| Independent read-only review | `reviewer` | `922a7582-1ace-4ef7-b37c-e926d5df7c5b` | PASS (0 findings, clean conformance) |
| Review integration | `codex-exec-writer` | `daeda205-c3f4-4d9f-a035-6467a64d8706` | Re-ran affected gates clean, 10/10 CTests pass, drafted execution record |
| Closure | `orchestrator` | Pending orchestrator session | Closure and commit |

Builds used:
`build/cpu-test` (BUILD_TESTING=ON, ENGINE_ENABLE_HIP=OFF),
`build/gpu-test` (RelWithDebInfo, BUILD_TESTING=ON, ENGINE_ENABLE_HIP=ON).

Files in this evidence directory:

- `input-brief-worker.md`, `input-brief-reviewer.md`, `input-brief-integration.md` —
  the briefs given to each agent.
- `RESUME-NOTES.md` — the implementation and integration handoff notes.
- `header-classification.md` — inventory and classification of all 41 model-used headers into stable, platform, experimental, host wrapper, and private.
- `SURFACE-metrics.md` — detailed measurement of the 97 paths in `include/gufo/ext/SURFACE` (LoC, token count, zero serving/model leaks).
- `review-findings.md` — independent review report (clean PASS, 0 findings).
- `build-and-tests.txt` — implementation agent's compilation, gate, and test logs.
- `pending-commands.txt` — verbatim commands for deferred functional and benchmark runs per the 2026-10-31 project decision.
- `integration-reruns.txt` — integration agent's rerun ledger for all affected static, isolation, policy, format, and CTest gates.
- `README.md` — this provenance document.

Scratch created during the card: `/tmp/card10-logs`, `/tmp/card10-review`, `/tmp/card10-integration`.
No functional test server cache was created (functional harness deferred to post-Card-14 window).
