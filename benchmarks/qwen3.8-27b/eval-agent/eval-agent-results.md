# gufo-agent-eval — Qwen3.8-27B on `gufo serve`

Results from the [gufo-agent-eval](../../../tools/eval/README.md) harness
([#153](https://github.com/gufo-org/gufo/issues/153)). Twelve of the twenty
suite tasks, run sequentially against a local `gufo serve` endpoint over about
three hours.

> Not an official Terminal-Bench score. This suite runs a different agent (Pi)
> in a different sandbox (bubblewrap) than upstream Terminal-Bench, so these
> numbers are not comparable to published Terminal-Bench results. The
> reference column exists only to give the timings a scale; it is a different
> agent on a different server.

## Setup

| | |
| --- | --- |
| Model | `Qwen3.8-27B-UD-Q4_K_XL.gguf`, served as `qwen3.8-27b` |
| Server | `gufo serve llm`, `--context 32768 --max-tokens 8192` |
| Agent | Pi 0.84.2 |
| Attempts | 1 per task (pass@1) |
| Timeout | 1800 s per attempt |
| Thinking | enabled (server default) |
| Suite | `gufo-agent-eval`, seeded from Terminal-Bench 2.1 at `5c8eadf1` |
| Date | 2026-09-01 |

## Results

**3 of 12 passed (pass@1 0.25).** Total 3.09 h.

| Task | Result | Min | Ref | Turns | Tool calls | In tok | Out tok | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `build-cython-ext` | fail | 30.0 | 36 | 0 | 0 | 0 | 0 | timeout, see caveat |
| `cobol-modernization` | fail | 30.0 | 40 | 0 | 0 | 0 | 0 | timeout, see caveat |
| `headless-terminal` | fail | 21.1 | 10 | 12 | 11 (4 failed) | 9,112 | 8,192 | output cap |
| `configure-git-webserver` | fail | 19.6 | 26 | 20 | 19 (3 failed) | 12,125 | 563 | |
| `pypi-server` | fail | 19.5 | 3 | 21 | 21 (6 failed) | 14,056 | 383 | |
| `regex-log` | fail | 13.3 | 10 | 1 | 0 | 1,603 | 8,192 | output cap |
| `sparql-university` | fail | 13.1 | 14 | 3 | 2 | 6,481 | 8,192 | output cap |
| `fix-git` | **pass** | 12.8 | 4 | 12 | 11 (0 failed) | 10,637 | 527 | |
| `nginx-request-logging` | fail | 8.9 | 5 | 16 | 15 (1 failed) | 8,525 | 409 | |
| `break-filter-js-from-html` | fail | 8.3 | 11 | 3 | 2 | 2,652 | 4,716 | |
| `openssl-selfsigned-cert` | **pass** | 5.0 | 2 | 8 | 7 (0 failed) | 4,537 | 364 | |
| `git-leak-recovery` | **pass** | 3.8 | 4 | 9 | 8 (0 failed) | 5,015 | 262 | |

Reference is the same model and quantisation from
[terminal-bench-mini](https://github.com/kyuz0/terminal-bench-mini)'s
`Qwen3.8-27B-UD-Q4_K_XL` run on llama.cpp with Terminus-2, where 19 of 20
passed at pass@2.

No verifier crashed. Time to first response was 262-268 ms throughout, so
endpoint latency is not what makes these runs long.

## Three failures are an output-budget problem, not a capability one

`headless-terminal`, `regex-log` and `sparql-university` each report
`context_failures: 1`, meaning generation stopped on `length` after exhausting
the 8,192-token output cap.

`regex-log` is the clearest case: **one turn, zero tool calls, 8,192 output
tokens, thirteen minutes of generation that never produced an action.** The
model reasoned until it ran out of budget.

With thinking enabled, reasoning and answer share one output budget, and 8,192
tokens is evidently too tight for some tasks. These three should be re-run with
a larger `--max-tokens` before being read as failures. The reference run passed
all three.

## Passing runs made no failed tool calls

Every passing task had zero failed tool calls; every completed failure had
either failed tool calls or a truncated response.

| | Passes | Failures |
| --- | --- | --- |
| Failed tool calls | 0, 0, 0 | 4, 3, 6, 1, and 0 for truncated runs |

Twelve tasks is too few to lean on, but the split is clean enough to be worth
watching.

## `cache_read` is zero on every task

Every task reports `cache_read: 0`, including `pypi-server` at 14,056 input
tokens over 21 turns. Continuation prefix reuse never hits for this workload,
so each turn re-prefills its whole prompt, and each request also pays a 2.36 GB
snapshot that is then discarded.

Tracked as [#224](https://github.com/gufo-org/gufo/issues/224). This is the
main reason the runs sit above the reference times, and it means these timings
are an upper bound rather than a settled measurement of the server.

Measured ratio against the reference, over the tasks that ran to completion:
median about 2.5x, range 2.1-5.6x. Projecting the full twenty-task suite at
that ratio gives roughly 8 h at a 30 min cap (with 11 tasks cut off), 12.9 h at
60 min, or 24.7 h at the harness default of 3 h (3 cut off).

## Caveat: the two 30-minute timeouts are not trustworthy

`build-cython-ext` and `cobol-modernization` both recorded 0 turns, 0 tool
calls and 0 tokens after hitting the 1800 s cap. That is inconsistent with the
server log, which shows requests arriving throughout both runs, and with a
120 s reproduction of `build-cython-ext` that completed 5 turns and 6 tool
calls normally.

So the agent was working and the result document failed to record it. The
cause has not been isolated. Treat both rows as measurement failures rather
than model failures. Tracked in
[tools/eval/TODO.md](../../../tools/eval/TODO.md).

## Raw artifacts

Under `raw/`:

| File | Contents |
| --- | --- |
| `<task>.json` | Sanitized result document: per-task metrics, aggregate, identity |
| `<task>.log` | Full runner output, including preflight and the verifier's pytest output |

The JSON is written through the harness sanitizer and carries no credentials,
endpoint addresses, or user paths; the endpoint appears only as a stable hash
label. The `.log` files are raw runner output and are **not** sanitized.

## Reproducing

```sh
gufo serve llm --model models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --served-model-name qwen3.8-27b --context 32768 --max-tokens 8192 --port 9999

nix run .#eval-agent -- run <task> \
  --base-url http://127.0.0.1:9999/v1 \
  --agent-timeout 1800 \
  --output result.json -v
```

## Caveats

- One attempt per task. Upstream defaults to two and reports pass@2, so the
  0.25 pass rate here is not comparable to its 19/20.
- Twelve of twenty tasks. `mteb-retrieve` and `mailman` cannot pass yet for
  reasons recorded in their `PROVENANCE.md`; the rest were not run.
- The 1800 s cap is below the harness default of three hours and below the
  reference times for `build-cython-ext` (36 min) and `cobol-modernization`
  (40 min), so those two could not have completed regardless.
- The output cap of 8,192 tokens truncated three runs, as above.
- Prefix reuse is broken (#224), so every timing includes redundant prefill.
