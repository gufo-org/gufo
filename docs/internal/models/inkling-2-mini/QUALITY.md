# Inkling 2 Mini quality

Measured November 22–23, 2026 on Strix Halo (gfx1151, ROCm 10.1).

## Logits against llama.cpp

Reference: llama.cpp `94f3af3d0` (includes the Inkling 2 merges #32694,
#32704 and #32799), CPU backend, the `Mainline_Support` UD-Q4_K_XL file (same
tensors as the root file), teacher forcing over the first 9,300 tokens of the
WikiText-2 test set. KL is over the reference's top 64 plus a tail bucket.

| Engine | mean KL | top-1 agreement |
| --- | --- | --- |
| Gufo, batched prefill path | 1.554e-3 | 98.02% |
| Gufo, one token per call (decode path) | 1.609e-3 | 98.11% |
| llama.cpp HIP backend, same file | 1.832e-3 | 97.87% |

Per position range, Gufo prefill / decode / llama.cpp HIP:

| Positions | Gufo prefill | Gufo decode | llama.cpp HIP |
| --- | --- | --- | --- |
| 0–1,023 | 1.151e-3 | 1.049e-3 | 1.469e-3 |
| 1,024–4,095 | 1.497e-3 | 1.585e-3 | 1.602e-3 |
| 4,096–8,191 | 1.514e-3 | 1.635e-3 | 2.001e-3 |
| 8,192–9,299 | 2.232e-3 | 2.095e-3 | 2.181e-3 |

Gufo stays closer to the CPU reference than llama.cpp's own GPU backend in
every range. Positions past 8,192, where the global layers' log-scaled
queries start, are harder for both GPU engines alike.

Reproduce:

```sh
inkling2_oracle --model ROOT.gguf --text wiki.test.raw --n 9300 --tokenize-only --ids-out ids.bin
llama_dump MAINLINE.gguf ids.bin ref.bin 16 512            # CPU, about 20 minutes
inkling2_oracle --model ROOT.gguf --ids ids.bin --ref ref.bin --n 9300 --out ours.bin [--single]
python3 src/models/inkling2/tests/oracle/compare_dumps.py ref.bin ours.bin 1024 4096 8192
```

## Exactness of the optimisations

The batched prefill path is byte-identical to the imported extension at every
position. The decode path differs only through the router and loop-controller
copies: exact BF16 here, F16-rounded in the extension. On the first 4,096
positions the two decode builds score 1.451e-3 and 1.416e-3 KL against the CPU
reference (98.34% and 98.51% top-1), the same within noise. All later kernel
changes are byte-identical in both paths, and batched decode of 2, 3 and 4
sessions equals single-session decode exactly (`inkling2_bench --verify`).

## Tokenizer and template

Tokenizing the whole WikiText-2 test set gives the same 312,991 ids as
llama.cpp's `llama-tokenize` (with `ignore_merges`, #32799). `inkling2_text_test`
passes its 8,354 checks: tokenizer goldens from the HF tokenizer and
chat-template goldens from `transformers`' `apply_chat_template`.

## State and cache

`inkling2_cache_test` on 5,000 tokens: restoring a checkpoint (341.9 MiB, 99
components) continues with bit-identical logits for 48 tokens, also into a used
slot, and layered prefill with interleaved decodes matches uninterrupted runs.

## Functional suites

See the PR for the run on the final build. Known model behaviour, also
documented by the extension: with thinking off, the model often deliberates in
its answer, so `reasoning-separator` (thinking-off copy case), `tools`
(`auto-tools` with thinking off) and `batch` fail on their exact-text
assertions. No request fails at the HTTP level.

## DSpark

DSpark only changes how many tokens a step produces; the tokens are those of
plain decoding:

- `inkling2_bench --spec 2,3,4` replays a greedy reference through
  `Engine::Verify` with one proposal corrupted at random per step: every kept
  row's logits equal the one-token decode's byte for byte (0 mismatches of
  about 65 rows per width at contexts 64 and 2,040), and so does a plain decode
  after the last commit.
- `inkling2_bench --dspark` decodes 128 to 256 greedy tokens with real
  proposals: the token sequence equals plain decoding in every run (contexts
  64 to 12,000, depths 1 to 3).
- Over HTTP, three greedy and two seeded sampled requests (temperature 0.7 and
  0.8, top-p 0.95) return the same reasoning and content with and without
  `--dspark-model`.
- `inkling2_bench --batch 2,4 --verify` and the oracle dumps (batched and
  one-token paths) are unchanged.
- `inkling2_spec_test --dspark` repeats the first two checks in 8 seconds.
- Functional suites (`tests/functional/run.py`, the runner's default
  `--sessions 4 --context 32768`, with disk-cache restarts and cancellations):
  discovery, responses, stops, conversation, structured, tool-history,
  tool-choice-cache, state-edges, prefill-scheduling, cache-edits,
  cache-depth, logprobs, sampling-ranges, batch and cache. AR and DSpark pass
  the same suites (batch fails for both: with thinking off the model argues
  instead of calling the tool), and all 432 recorded responses have the same
  output hash, with 2,062 drafted cycles in the DSpark run. Repeated after the
  verification widening and round 3 (8-row verification, low-rank proposal
  logits): 428 of 428 matched responses identical (1,836 drafted cycles), and
  identical to the first run's baseline; the only unmatched requests are the
  failing `batch` suite's concurrent ones, whose indices follow arrival order.
