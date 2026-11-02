# Pending post-Card-14 commands — NOT execution evidence

Orchestrator-only qualification recipe. Bind Q4/Q8 to verified Qwen27 target
GGUFs, FN to the first Flash-Next target shard, DF to the DFlash draft, MTP to
the Flash-Next shared draft, and MTP27 to a compatible currently missing 27B
MTP draft. Verify assets and binary/harness/toolchain hashes before execution.
`BIN=build/card06-release/gufo`; matched control
`MAIN=/tmp/gufo-main-ctl/build/card06-main-release/gufo`.
For lock-mode qualification repeat with `BIN=build/release/gufo` after the
lock build; keep separate evidence. These are parameterized commands, not
claims that today's historical controls remain reusable.

Each row selects `MODEL`, `MODE`, `DRAFT_FLAG`, `DRAFT`, `TAG`:

| TAG | MODEL | MODE | DRAFT_FLAG | DRAFT |
| --- | --- | --- | --- | --- |
| qwen27-q4-off | $Q4 | off | --dflash-model | $DF |
| qwen27-q4-dflash2 | $Q4 | dflash2 | --dflash-model | $DF |
| qwen27-q8-off | $Q8 | off | --dflash-model | $DF |
| qwen27-q8-dflash2 | $Q8 | dflash2 | --dflash-model | $DF |
| flash-next-q4-off | $FN | off | --mtp-model | $MTP |
| flash-next-q4-mtp | $FN | mtp | --mtp-model | $MTP |

For every row, use suites in `functional-impact-plan-pr.txt` plus `metrics`;
retain standard, single-session and affected cache-pressure/cache-stalls
settings per harness README. `$SUITES` is the ordered repeated `--suite NAME`
argument list, including long-context and cache. `$SERVER_SETTINGS` contains
the selected concurrency/context configuration. Resolve these parameters and
record literal expanded commands before qualification; missing settings remain
pending, never a pass. Disk caches must use a dedicated /tmp directory and be
deleted on success/failure/interruption. Keep every verdict before cleanup.

```sh
nix develop -c python3 tests/functional/run.py --output "/tmp/card06-$TAG-pr" --sampling-preset qwen38 $SUITES -- "$BIN" serve llm --model "$MODEL" --speculative "$MODE" "$DRAFT_FLAG" "$DRAFT" $SERVER_SETTINGS
# Timing control: same selected histories/settings, replace BIN with MAIN and
# output with /tmp/card06-$TAG-main; avoid duplicate full correctness on main.
nix develop -c python3 tests/functional/compare.py --pair "/tmp/card06-$TAG-main" "/tmp/card06-$TAG-pr" --output "/tmp/card06-$TAG-compare.json"
nix develop -c python3 tests/functional/prompt_lookup.py --output /tmp/card06-lookup-fn -- "$BIN" serve llm --model "$FN" --speculative mtp --mtp-model "$MTP"
nix develop -c python3 tests/functional/prompt_lookup.py --output /tmp/card06-lookup-q27 -- "$BIN" serve llm --model "$Q4" --speculative dflash2 --dflash-model "$DF"
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-27b --gguf "$Q4" --dflash "$DF" run --target gufo --table "$TABLE"
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-flash-next --gguf "$FN" --mtp "$MTP" run --target gufo --table "$TABLE"
```

Repeat Qwen benchmark for Q8. TABLE is each affected table ID from the model's
artifacts/bench.json; preserve its numerical quality commands from QUALITY.md
and standard benchmark settings. DeepSeek runtime and benchmark remain skipped
by user decision until explicitly scheduled; exact runtime command is retained
in deepseek-static-analysis.md. Benchmark recipe when authorized:
`nix develop -c python3 tools/bench/model-bench.py --model deepseek-v4-flash --gguf "$DS" --dspark "$DSS" run --target gufo --table "$TABLE"`.
