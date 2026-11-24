# Functional test rules

These rules apply to the suites under `tests/functional/`. They keep the suites
deterministic, independent of each other and cheap enough to run on every PR.
[README.md](README.md) describes each suite and the Devlab workflow.

## When to add or change a functional suite

- Add a functional case only for behavior a real model on the GPU shows end to
  end: cache restores and checkpoints, scheduling and batching, speculative
  decoding, model output under grammars or sampling, model-specific framing.
- Parsing, serialization, HTTP shapes and validation belong in the C++ tests
  under `tests/cli/` with deterministic fixtures; hosted CI runs those.
- Prefer a case in an existing suite with the same setup over a new suite.
- Name the regression a case guards (issue or PR number) in its comment or
  docstring. A case that guards nothing is a candidate for removal.
- When a fix changes a functional behavior, add or update the case that would
  have caught it, and confirm it fails without the fix.

## Registering a suite

1. Write `check_<name>(client, model, checks, chat_result, ...)` in its own
   module or next to related checks.
2. Add the name to `SUITES` in [suites.py](suites.py), and to `MIN_CONTEXT`,
   `NEEDS_CACHE_RAM`, `NOT_IN_ALL`, `VISION` or `INCLUDED_IN` when it needs them.
3. Dispatch it from the `suites` table in `openai_sdk.py`; `test_impact` fails
   when a registered suite has no entry.
4. Add its `impact.core.json` entry (`covers`, `uses`, `configuration`,
   `vision_required`, optional `profiles` and `isolation`), then run
   `python3 tools/ci/generate-infrastructure.py --write`.
5. Add a unit test with a fake client in `test_runner.py` that pins its
   requests, and a row in the README suite table.
6. Decide which models run it. Flash-Next has no `suites` list and runs every
   suite. Qwen27's and DeepSeek's lists (`src/models/*/functional/devlab.json`)
   hold only the suites their own runtime, cache, speculative decoding,
   template or native tool syntax change. A suite whose code is shared server
   plumbing (validation, accounting, streaming events, schema shapes) runs on
   Flash-Next only.
7. Measure it and refresh the cost table:
   `python3 tests/functional/devlab_costs.py <results>`.

## Cost

- Size prompts to the smallest history that crosses the threshold the case
  tests, and assert that threshold (`assert prompt_tokens >= 16384`).
- Bound `max_tokens` by what the assertion needs.
- Every warm request asserts its exact reply. Replay one uncached control per
  scenario, the deepest history, and compare it last.
- Vary endpoints, streams and tool choices with `transports.rotate` and
  `transports.streaming_modes`: every option, not every combination.
- Never sleep for a fixed time; poll the state with a deadline.
- Report the suite's seconds per model in the PR. A suite over 60 s on
  Flash-Next needs a reason.

## Independence

- A suite must pass alone and after any other suite, in any order. Devlab
  shuffles suites with the plan's `order_seed`; `run.py --order-seed N`
  reproduces an order.
- Start each conversation with a prefix unique to the suite (for example
  `cache_depth_main`) so other suites' cache entries never match it.
- Read server counters as deltas across the suite's own requests.
- A suite that depends on server-wide cache state (RAM budget, disk turnover,
  scheduling headroom) and cannot measure it in isolation gets
  `"isolation": "server"`: its own server. A suite that fills that state for
  others joins the `"cache-fill"` pool, which shares one server apart from the
  rest. Prefer making the suite independent.

## Validation

Run the smallest set that covers the change:

```sh
python3 -m unittest discover -s tests/functional -p 'test_*.py'
python3 tests/functional/run.py --output /tmp/one --sampling-preset qwen38 \
  --expected-input-modalities text,image --record-baseline --suite <name> \
  -- ./result/bin/gufo serve llm --model <weights> [model options]
```

Then run the suite in its configuration group with two `--order-seed` values,
on every model whose list includes it.
