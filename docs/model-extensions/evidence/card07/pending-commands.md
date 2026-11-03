# Pending commands — Card 07 (NOT execution evidence)

Everything below is un-run. The 2026-10-31 user decision excludes functional-harness
runs, main-control matched timings and speed benchmarks from cards 06–14; they execute
once, in the single post-Card-14 qualification window, scheduled solely by the
orchestrator. Nothing here is a pass, and no historical control from Card 06 is assumed
reusable.

Assets (verified present today):

```sh
Q4=$PWD/models/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf
Q8=$PWD/models/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q8_K_XL.gguf
MMPROJ=$PWD/models/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/mmproj-BF16.gguf
DF=$PWD/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
FN=$PWD/models/qwen38-flash-next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
MTP=$PWD/models/qwen38-flash-next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
MTP27=            # MISSING: no 27B-compatible MTP draft under ./models
BIN=build/card07-release/gufo
MAIN=             # a main-control production build from the same pinned dev shell
```

## 1. Functional matrix (Qwen 27B and Flash-Next; deferred, PENDING)

Card 07 moved shared tokenization/framing, GGUF config extraction, the vision stack and
every shared Qwen kernel, so all four text profiles are affected. Rows select
`MODEL`, `MODE`, `DRAFT_FLAG`, `DRAFT`:

| TAG | MODEL | MODE | DRAFT_FLAG | DRAFT |
| --- | --- | --- | --- | --- |
| qwen27-q4-off | $Q4 | off | --dflash-model | $DF |
| qwen27-q4-dflash2 | $Q4 | dflash2 | --dflash-model | $DF |
| qwen27-q8-off | $Q8 | off | --dflash-model | $DF |
| qwen27-q8-dflash2 | $Q8 | dflash2 | --dflash-model | $DF |
| flash-next-q4-off | $FN | off | --mtp-model | $MTP |
| flash-next-q4-mtp | $FN | mtp | --mtp-model | $MTP |

`$SUITES` is the ordered repeated `--suite NAME` list selected for these paths, and it
MUST include `long-context` and `cache` (continuation code and the cache identity inputs
were touched indirectly by the config/token moves), plus `metrics`. `$SERVER_SETTINGS`
carries the chosen sessions/context configuration, recorded literally before execution.
Verify loaded modes and actual drafts, not just requested flags.

```sh
# candidate correctness
nix develop -c python3 tests/functional/run.py --output "/tmp/card07-$TAG-pr" \
  --sampling-preset qwen38 $SUITES -- \
  "$BIN" serve llm --model "$MODEL" --mmproj "$MMPROJ" --speculative "$MODE" "$DRAFT_FLAG" "$DRAFT" $SERVER_SETTINGS
# matched main control (same histories/settings; no duplicate full correctness on main)
nix develop -c python3 tests/functional/run.py --output "/tmp/card07-$TAG-main" \
  --sampling-preset qwen38 $SUITES -- \
  "$MAIN" serve llm --model "$MODEL" --mmproj "$MMPROJ" --speculative "$MODE" "$DRAFT_FLAG" "$DRAFT" $SERVER_SETTINGS
nix develop -c python3 tests/functional/compare.py --pair "/tmp/card07-$TAG-main" \
  "/tmp/card07-$TAG-pr" --output "/tmp/card07-$TAG-compare.json"
```

Acceptance is per request and per phase at 5% and 10 ms, never pooled averages; retain
every row, including failures and inconclusive noise. Server disk caches always live in
the dedicated `/tmp/card07-<row>` output directory and are deleted afterwards on success,
failure and interruption alike (only small verdict artifacts move into this directory).

## 2. Prompt-lookup continuity (PENDING)

```sh
nix develop -c python3 tests/functional/prompt_lookup.py --output /tmp/card07-lookup-fn -- \
  "$BIN" serve llm --model "$FN" --speculative mtp --mtp-model "$MTP"
nix develop -c python3 tests/functional/prompt_lookup.py --output /tmp/card07-lookup-q27 -- \
  "$BIN" serve llm --model "$Q4" --speculative dflash2 --dflash-model "$DF"
```

## 3. CLI `prompt`/`bench` controls (PENDING — required because `src/cli/prompt/prompt.cpp`
now calls `ExtractQwenConfig`, a routing change under the shared gates)

```sh
# matched CLI output and timings, candidate vs main, same prompt/seed/settings
"$BIN"  prompt --model "$Q4" --mmproj "$MMPROJ" --max-tokens 64 --seed 7 --verbose --prompt "<fixed prompt>"
"$MAIN" prompt --model "$Q4" --mmproj "$MMPROJ" --max-tokens 64 --seed 7 --verbose --prompt "<fixed prompt>"
"$BIN"  bench  -p 512 -n 128 "$Q4"
"$MAIN" bench  -p 512 -n 128 "$Q4"
```

Byte-identical rendered prompts and phase timings within 5% and 10 ms per request/phase.

## 4. Standard speed benchmark (PENDING)

Table ids come from `docs/models/<model>/artifacts/bench.json`.

```sh
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-27b --gguf "$Q4" --dflash "$DF" run --target gufo --table "$TABLE_27B"
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-flash-next --gguf "$FN" --mtp "$MTP" run --target gufo --table "$TABLE_FN"
```

## 5. Shipped lock-mode HIP production artifact (PENDING)

```sh
nix develop -c cmake --preset release -B build/card07-lock -DGUFO_EXTENSIONS_FROM_LOCK=ON
nix develop -c cmake --build build/card07-lock --parallel 4
# then repeat sections 1-4 with BIN=build/card07-lock/gufo and retain the lock cache,
# compile_commands.json and binary provenance separately from the discovery results.
```

Note: `extensions.lock` currently holds `"extensions": []`, so lock mode selects nothing
yet; the gate becomes meaningful once models register (Cards 11-14+). A CPU lock build or
the collision gate proves neither numerical quality nor shipped performance.

## 6. Windows native build/run of the affected targets (PENDING — no Windows host here)

From `BUILD-WINDOWS.md` on a Windows Strix Halo with the pinned TheRock 10.0.0 tree:

```
tools\windows\vs-env.cmd cmake --preset windows-release
tools\windows\vs-env.cmd cmake --build --preset windows-release
build\windows-release\gufo.exe diagnose
tools\windows\vs-env.cmd cmake --preset windows-cpu-test
tools\windows\vs-env.cmd cmake --build --preset windows-cpu-test --target check-pr
```

Hosted CI equivalents: `.github/workflows/windows-build.yml` and
`.github/workflows/devlab-windows.yml`. A workflow edit is not Windows execution evidence.

## 7. Asset-blocked unit tests (PENDING, not passes)

```sh
# needs a 27B-compatible MTP draft (today: Missing Qwen MTP tensor: token_embd.weight)
GUFO_QWEN27B_MODEL="$Q4" GUFO_QWEN27B_MTP_MODEL="$MTP27" \
  nix develop -c ctest --test-dir build/card07-gpu -R '^(qwen_mtp_gpu_test|qwen_mtp_reference_test)$' --output-on-failure
# needs GUFO_QWEN27B_{MODEL,DFLASH_MODEL,MTP_MODEL}; the sampling scope already passed
GUFO_QWEN27B_MODEL="$Q4" GUFO_QWEN27B_DFLASH_MODEL="$DF" GUFO_QWEN27B_MTP_MODEL="$MTP27" \
  nix develop -c ctest --test-dir build/card07-gpu -R '^qwen27b\.cli$' --output-on-failure
# needs MODEL DRAFT IMAGES_DIR MODE arguments (an image fixture, none registered)
build/card07-gpu/tests/models/qwen27b/qwen_vision_serving_test "$Q4" - <images-dir> --disk-only
```

## 8. Stale kernel-scratch baseline (owner decision, PENDING)

`kernel_resources_test` reports the same 30 over-budget kernels on the parent revision's
own binary and on this card's build (byte-identical lists; see `kernel-resources.txt`).
Either refresh the baseline deliberately with

```sh
nix develop -c python3 tools/ci/check-kernel-resources.py build/card07-gpu/gufo \
  --write-baseline tools/ci/kernel-resources.json --bundler "$(dirname "$(grep -m1 '^CMAKE_HIP_COMPILER:' build/card07-gpu/CMakeCache.txt | cut -d= -f2)")/clang-offload-bundler"
```

or fix the spills. Do not silently widen the allowance inside this card.

## 9. Hosted checks (orchestrator runs these at closure, not the implementation agent)

```sh
nix build .#checks.x86_64-linux.pr
```
