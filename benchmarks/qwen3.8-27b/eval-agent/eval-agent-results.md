# gufo-agent-eval — Qwen3.8-27B on `gufo serve`

First recorded results from the [gufo-agent-eval](../../../tools/eval/README.md)
harness ([#153](https://github.com/gufo-org/gufo/issues/153)). Three tasks,
run sequentially against a local `gufo serve` endpoint.

> Not an official Terminal-Bench score. This suite runs a different agent (Pi)
> in a different sandbox (bubblewrap) than upstream Terminal-Bench, so these
> numbers are not comparable to published Terminal-Bench results. The
> reference column below is included only to give the timings a scale, and is
> itself a different agent and server.

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

| Task | Result | Wall time | Reference | Turns | Tool calls | Input tok | Output tok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `openssl-selfsigned-cert` | **pass** | 299 s | 2 min | 8 | 7 (0 failed) | 4,537 | 364 |
| `pypi-server` | fail | 1,173 s | 3 min | 21 | 21 (6 failed) | 14,056 | 383 |
| `fix-git` | **pass** | 770 s | 4 min | 12 | 11 (0 failed) | 10,637 | 527 |

**2 of 3 passed (pass@1 0.67).** Total 2,242 s (37 min).

Reference is the same model and quantisation from
[terminal-bench-mini](https://github.com/kyuz0/terminal-bench-mini)'s
`Qwen3.8-27B-UD-Q4_K_XL` run on llama.cpp with Terminus-2. Different agent and
server, so it bounds the scale rather than being a like-for-like baseline.

Time to first response was 262–268 ms across all three, so endpoint latency is
not what makes these runs long.

No task hit the timeout, and no verifier crashed. `pypi-server` failed on
merit after 21 turns with 6 failed tool calls.

## `cache_read` is zero everywhere

Every task reports `cache_read: 0`, including `pypi-server` at 14,056 input
tokens over 21 turns. That is not a reporting gap in the harness: continuation
prefix reuse never hits for this workload, so each turn re-prefills its whole
prompt.

Tracked as [#224](https://github.com/gufo-org/gufo/issues/224). Two
independent causes: the cache entry stores prompt + generated tokens so it is
always longer than the prompt that produced it, and reasoning tokens are never
echoed back by the client so the sequences diverge at the first generated
token. Each request also pays a 2.36 GB snapshot that is then discarded.

This is the main reason these runs sit well above the reference times, and it
means these timings should be treated as an upper bound rather than a settled
measurement of the server.

## Raw artifacts

Under `raw/`:

| File | Contents |
| --- | --- |
| `<task>.json` | Sanitized result document: per-task metrics, aggregate, identity |
| `<task>.log` | Full runner output, including preflight and the verifier's pytest output |

The JSON is written through the harness sanitizer, so it carries no
credentials, endpoint addresses, or user paths; the endpoint appears only as a
stable hash label. The `.log` files are raw runner output and are **not**
sanitized — they were kept verbatim for inspection.

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

- One attempt per task. Upstream Terminal-Bench-Mini defaults to two and
  reports pass@2, so these are not comparable to its published rates.
- Three tasks out of twenty, chosen for short reference times. Not a
  capability measurement.
- Timeout was 1800 s rather than the harness default of three hours, to bound
  the run.
- Output-token counts are low (364–527 across 8–21 turns). Whether that
  reflects terse tool-driven behaviour or thin generations is not yet
  established; note that with thinking enabled, reasoning tokens are counted
  in `output` by the provider but are not visible in the transcript.
