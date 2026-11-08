# Card 11 evidence — provenance

Repository `/home/mixer/gufo`, jj. Parent revision `xppvypvw 8d8ef356`
("feat(extensions): publish and measure public extension api surface").
The Card 11 work is the entire uncommitted working copy; nothing else was touched.
`docs/model-extensions/RFC.md` stayed strictly byte-identical and uncommitted.

Host: Strix Halo (gfx1151), Linux x86_64. Nix toolchain with `nix develop -c` and
host python interpreter for static repository checks.

Agents (three distinct agents, sequential, then orchestrator closure):

| Role | Agent | Run ID | Verdict / Outcome |
| --- | --- | --- | --- |
| Implementation | `worker` | `1c0adb86-f704-460b-9694-93aa312ac1cd` | Complete, handed off to reviewer |
| Independent read-only review | `reviewer` | `70a872ba-869f-42b1-80e6-b71880841633` | PASS (1 Nit: F1 test reporting language) |
| Review integration | `codex-exec-writer` | `dd111c9e-4200-4e2c-bc44-4c549b28b08d` | Resolved F1, re-ran affected gates clean, 4/4 CPU & GPU CTests pass, drafted execution record |
| Closure | `orchestrator` | Current session | Re-ran full gate set clean, ran Codex exec review, resolved P2 finding on `tools/bench/build.sh`, verified PR checks, closing commit |

Builds used:
`build/cpu-test` (BUILD_TESTING=ON, ENGINE_ENABLE_HIP=OFF),
`build/gpu-test` (RelWithDebInfo, BUILD_TESTING=ON, ENGINE_ENABLE_HIP=ON).

Files in this evidence directory:

- `input-brief-worker.md`, `input-brief-reviewer.md`, `input-brief-integration.md` —
  the briefs given to each agent.
- `RESUME-NOTES.md` — the implementation and integration handoff notes.
- `model-include-inventory.md` — inventory of migrated includes across 8 models (152 includes across 95 files) and catalog of preserved own/family coupling.
- `isolation-ratchet.md` — specification of `--fail-on-model-core-includes` and its gate integration.
- `build-and-tests.txt` — implementation agent's compilation, gate, and test logs.
- `pending-commands.txt` — verbatim commands for deferred functional and benchmark runs per the 2026-10-31 project decision.
- `review-findings.md` — independent review report (PASS, 1 Nit: F1).
- `integration-reruns.txt` — integration agent's rerun ledger for all affected static, isolation, policy, format, and CTest gates.
- `orchestrator-gates.txt` — orchestrator closure transcript with re-run gates, Codex review output, and resolution proof.
- `README.md` — this provenance document.

Scratch created during the card: `/tmp/card11-logs`, `/tmp/card11-review`, `/tmp/card11-integration`.
No functional test server cache was created (functional harness deferred to post-Card-14 window).
