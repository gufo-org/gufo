# Card 09 evidence — provenance

Repository `/home/mixer/gufo`, jj. Parent revision `uytorytn c03aaa23`
("refactor(extensions): extract h3, eval and comparison targets from core", Card 08);
`main` was `192b75a8` at the start. The Card 09 work was the entire uncommitted
working copy; nothing else of the user's was present or touched.
`docs/model-extensions/RFC.md` stayed byte-identical and uncommitted, sha256
`2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448`.
`HANDOFF.md` was never read (deleted; the decisions live in `CARDS.md` and
`cards/README.md`).

Host: Strix Halo (gfx1151), 125 GiB RAM, 32 cores, `llama-swap.service` left
running throughout. `cmake`/`ctest`/`ninja`/`nix` only through `nix develop -c`;
the `tools/ci` python checks also run on the host interpreter. ccache warm
(66% hits), so a from-scratch configure stayed cheap.

Agents (three distinct pi sub-agents, sequential, then the orchestrator):

| Role | Agent | Run ID |
| --- | --- | --- |
| Implementation | `worker` | `8e9ab18d-6117-4e80-b033-e5521056fbdc` |
| Independent read-only review | `reviewer` | `c9a6d209-325e-4f13-9dba-884f29d56fc1` (verdict BLOCK) |
| Review integration | `codex-exec-writer` | `5cb5551d-5bba-458e-b260-a86101cedec8` (no Nix daemon in its sandbox; every dependent gate handed over as `PENDING(orchestrator)`) |
| Closure, own pinned re-run, `codex exec review` and its fixes, plus the O0 walker fix | orchestrator | this session |

Builds used (all still present, gitignored scratch, not evidence):
`build/card09-gpu` (gpu-test: RelWithDebInfo, BUILD_TESTING=ON, HIP ON, tools ON),
`build/card09-cpu` (cpu-test: HIP OFF), `build/card09-release` (discovery:
BUILD_TESTING=OFF, `GUFO_EXTENSIONS_FROM_LOCK=OFF`),
`build/card09-lock` (`GUFO_EXTENSIONS_FROM_LOCK=ON`).
Post-closure binary hashes: `build/card09-release/gufo`
`822b71f9004604003dc16…`, `build/card09-lock/gufo` `d8b27e8512257b10c5419…`.
The Card 08 artifacts under `build/card08-*` were used only as the matched
`gufo diagnose` control.

Files here:

- `input-brief-worker.md`, `input-brief-reviewer.md`, `input-brief-integration.md` —
  the instructions given to each agent (not results).
- `RESUME-NOTES.md` — the implementation agent's handoff, including its scratch list.
- `review-findings.md` — the independent review verbatim (23 KB, BLOCK verdict).
- `integration-reruns.txt` — the integration agent's commands, exits and denials.
- `orchestrator-gates.txt` — authoritative closure transcript: the O0 defect, F1/F2/F3
  proofs, the two negative proofs, the `codex exec review` findings and their fixes,
  static gates, production artifacts and the working-copy integrity audit.
- `build-and-tests.txt`, `moved-source-flags.txt`, `negative-proofs.txt` — the
  implementation agent's evidence; `flag-audit.py` reproduces the parity audit from
  two `compile_commands.json` files.
- `logs/` — complete 43-line verdict lists per configuration, the two flag diffs, the
  gate outputs before/after each injected violation, the selection regex and the
  `diagnose` diffs. Small by design: no raw build logs, no caches, no model blobs.
- `pending-commands.txt` — every deferred requirement with its verbatim command.

Scratch created during the card, all deleted at closure: `/tmp/card09-logs`,
`/tmp/card09-backup`, `/tmp/card09-review`, `/tmp/card09-integration`. No functional
server cache was created because no functional harness ran (user decision
2026-10-31).
