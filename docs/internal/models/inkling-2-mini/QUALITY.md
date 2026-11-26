# Inkling 2 Mini quality

Strix Halo (gfx1151, ROCm 10.1), Unsloth `UD-Q4_K_XL` revision `8078d149`,
Gufo `94ef8903` and later (November 24, 2026) unless noted.

## Logits against llama.cpp

Teacher forcing over the first 9,300 WikiText-2 test tokens against llama.cpp
`94f3af3d0` on the CPU backend, same GGUF (llama.cpp's `inkling2_loop`
layout). KL over the reference's top 64 plus a tail bucket; lower is better.

| Engine | mean KL | max KL | top-1 |
| --- | --- | --- | --- |
| Gufo prefill path | 1.593e-3 | 0.179 | 97.96% |
| Gufo decode path (one token per call) | 1.609e-3 | | 98.11% |
| llama.cpp HIP backend, same file | 1.832e-3 | | 97.87% |

| Positions | Gufo prefill | Gufo decode | llama.cpp HIP |
| --- | --- | --- | --- |
| 0–1,023 | 1.134e-3 | 1.049e-3 | 1.469e-3 |
| 1,024–4,095 | 1.526e-3 | 1.585e-3 | 1.602e-3 |
| 4,096–8,191 | 1.676e-3 | 1.635e-3 | 2.001e-3 |
| 8,192–9,299 | 1.893e-3 | 2.095e-3 | 2.181e-3 |

The chunk-invariant prefill (below) changed the prefill path from 1.554e-3
(max 0.254, top-1 98.02%); the ranges move both ways, so it reads as rounding
noise. The router and loop-controller decode copies are exact BF16; the
imported extension's F16 copies scored the same within noise.

```sh
inkling2_oracle --model GGUF --text wiki.test.raw --n 9300 --tokenize-only --ids-out ids.bin
llama_dump GGUF ids.bin ref.bin 16 512            # tests/oracle/llama_dump.cpp, CPU, ~20 min
inkling2_oracle --model GGUF --ids ids.bin --ref ref.bin --n 9300 --out ours.bin [--single]
python3 src/models/inkling2/tests/oracle/compare_dumps.py ref.bin ours.bin 1024 4096 8192
```

## Tokenizer and template

The whole WikiText-2 test set tokenizes to the same 312,991 ids as
`llama-tokenize`. `inkling2_text_test` passes 8,354 tokenizer goldens (HF
tokenizer) and chat-template goldens (`transformers`). The GGUF's template is
byte-identical to the official `chat_template.jinja`.

## Exactness guarantees

Each holds bit for bit and is checked by a test:

| Guarantee | Check |
| --- | --- |
| A prompt split into any chunks, or a restored prefix plus a tail, equals one prefill | `inkling2_cache_test`: 22 chunkings of 1,853 tokens and 26 of 5,000 (1- to 1,025-token tails, splits at 2,047 to 3,000, thirds, 37-token chunks) and restores, logits and 4 decode steps; `inkling2_oracle` dumps of 9,300 tokens equal for 2,048-, 3,000-, 4,096- and 8,192-token passes |
| A restore continues as the uninterrupted session, also into a used slot | `inkling2_cache_test`, 48 tokens after 8,792 |
| Layered prefill, with decodes or prefills in its yields, equals an unsplit run | `inkling2_cache_test`, `inkling2_spec_test` (7 sessions decoding in an 8th's yields) |
| Batched decode of 2–8 sessions equals each session alone | `inkling2_spec_test`, `inkling2_bench --verify` |
| DSpark verification rows (2–64) equal one-token decodes; greedy and seeded sampled DSpark output equals AR | `inkling2_spec_test --dspark`, `inkling2_bench --spec`; HTTP hashes |
| The drafter's proposals after a restore equal those of a session that prefilled | `inkling2_cache_test --dspark` |

Chunk invariance needs two kernel rules on gfx1151. First, prefill attention's
16-key softmax steps start at absolute multiples of 16. Second,
`v_wmma_f32_16x16x16_f16` is not exact for zero products: the A entries that
meet a zero B entry change the result (37% of outputs in a standalone test),
while a whole zero column leaves C exact. So each row's own causal step and
its window's first step are summed on the VALU with masked V zeroed. A
one-token prompt chunk takes the prompt path; decoding uses `Engine::Decode`.

## Functional suites

`tests/functional/run.py --sampling-preset provider --thinking-off-reasons`,
with DSpark, vision and a 32,768-token context: all 23 suites in
`functional/devlab.json` pass (`cache-bridge` with `--cache-ram-bytes
2147483648`). Responses are identical with and without DSpark, and across the
round's builds (85 of 85 matched requests).

Model behaviour, not Gufo: the template has no thinking-off switch, so
`enable_thinking: false` selects level 0, where the model still writes a 29- to
345-token thought (as with llama.cpp). Suites outside the Devlab list that
assert no reasoning, or exact text with thinking off, fail on that. Pi's edit
task loops on `bash` in about 1 of 12 sampled runs, as often as the
distributions predict (llama.cpp's match within 0.5 nats).

## Benchmark method

[Benchmarks](BENCHMARKS.md) come from `tools/bench/model-bench.py` with
[`artifacts/bench.json`](artifacts/bench.json): HTTP on both sides, greedy,
seed 1, thinking off. For Gufo that means `--think off`, where the model may
still think briefly; llama.cpp gets `--reasoning off`.

| | Gufo | llama.cpp |
| --- | --- | --- |
| Build | `result/bin/gufo` (`nix build`) at `1d61903a` (`.gufo-wrapped` SHA-256 `49f10edb…`), November 26, 2026 | `4c92c5bfb` (master with #32694, #32704, #32799, #32833), ROCm 10.1 build, SHA-256 `be1e917f…`, November 24, 2026 |
| Server | `serve llm --think off --max-pending-per-client 8`, DSpark `--speculative dspark --dspark-model` | `-ngl 999 -fa on --cache-reuse 0 --cache-ram 0 --jinja --reasoning off` |

Model files: `Inkling-2-Mini-UD-Q4_K_XL.gguf` (SHA-256 `e6427665…`) and
`dspark-Inkling-2-Mini-Q8_0.gguf` (`3da00cf7…`), revision `8078d149`. Platform
profile `performance`. Completion hashes cover the streamed reasoning, because
Inkling spends the timed tokens thinking. Greedy DSpark output equals AR at all
eight depths and five concurrencies.

The pinned `.#llama-cpp-reference` (b11069) predates Inkling. No llama.cpp
build loads the DSpark drafter (`joint_proj.weight` is 5,120 × 768 in the
file, 2,816 × 768 in llama.cpp's DSpark loader; the Inkling drafter's support,
llama.cpp #32218, is not merged). Its DSpark and loading cells are N/A
([`artifacts/unavailable.json`](artifacts/unavailable.json)).

```sh
FILES="--gguf Inkling-2-Mini-UD-Q4_K_XL.gguf --dspark dspark-Inkling-2-Mini-Q8_0.gguf"
python3 tools/bench/model-bench.py --model inkling-2-mini --gufo result/bin/gufo $FILES \
  run --target gufo --fresh --drop-caches "<POSIX_FADV_DONTNEED on both files>"
python3 tools/bench/model-bench.py --model inkling-2-mini $FILES run --target reference \
  --reference-binary llama.cpp/build-rocm/bin/llama-server --table single-ar,multi-ar,memory --fresh
python3 tools/bench/model-bench.py --model inkling-2-mini render
```
