# Inkling 2 Mini quality

Measured November 22–23, 2026 on Strix Halo (gfx1151, ROCm 10.1).

## Logits against llama.cpp

Reference: llama.cpp `94f3af3d0` (includes the Inkling 2 merges #32694,
#32704 and #32799), CPU backend, the UD-Q4_K_XL file of revision `8078d149` (the
same file Gufo reads), teacher forcing over the first 9,300 tokens of the
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

Rerun on revision `8078d149` (llama.cpp's layout, read through the shared
GGUF reader): the same numbers to every printed digit, for both paths and
every range.

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

### Chunk invariance (November 24)

A restored prefix plus a tail prefill now equals one prefill of the whole
prompt, bit for bit. Before, restoring 1,809 cached tokens and prefilling the
last 44 changed the greedy output, and `system-injection`, `cache-depth` and
`cache` (`disk-spacing-disk`) failed on it. Three causes:

- A one-token prompt chunk ran the decode step. `Engine::Forward` now always
  takes the prompt arithmetic; decoding calls `Engine::Decode` or `Step`.
- Window-layer prefill attention started its 16-key softmax steps at the
  query block's first key, which moves with the chunk's start. The steps now
  start at multiples of 16.
- gfx1151's `v_wmma_f32_16x16x16_f16` does not ignore a zero product: the A
  entries that meet a zero B entry change the result (37% of outputs in a
  standalone test with tiny A values). It does keep a column exact when the
  whole B column is zero, and rows and columns are independent. A row's own
  16-key step (later keys masked) and its window's first step (earlier keys
  masked) held V rows that differ with the chunking: later tokens, the last
  row repeated at the chunk's end, or overwritten ring rows. Those two steps
  are now summed on the VALU (`fdot2`) with the masked keys' V zeroed, and the
  row's WMMA column is zero.

`inkling2_cache_test` checks it first: 22 chunkings (tails of 1 to 1,025
tokens, thirds, 37-token chunks) and a checkpoint restore plus a 1- or
44-token tail, each against one prefill of 1,853 tokens (logits and four
decode steps). Also passes at 5,000 tokens (`--chunk-prompt 5000`: ubatch
boundaries, ring wraps, 128-query blocks). The three functional suites pass.

The prefill path's arithmetic changed in those steps: KL against the CPU
reference 1.554e-3 → 1.593e-3, top-1 98.02% → 97.96%, max KL 0.254 → 0.179
(the position ranges move both ways: 0–1,023 1.134e-3, 1,024–2,047
1.690e-3, 2,048–8,191 1.599e-3, 8,192–9,299 1.893e-3). The decode path is
unchanged. Prefill speed in `inkling2_bench`: pp2048 2,221 → 2,212,
pp8192 2,119 → 2,109, pp32000 1,934 → 1,917 tokens/s.

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

## Concurrent requests and the tool suites (November 23, round 4)

- `inkling2_spec_test` gains three checks, all byte for byte against each
  session's lone one-token decode: batched decode of sessions in different
  attention buckets (36 rows), batched verification of several sessions with
  mixed widths and corruptions (149 rows), and one drafter pass for several
  sessions against lone proposals (tokens and probabilities). Before the
  per-row attention split, the first two reported 24 of 36 and 37 of 54
  mismatched rows: a request's output depended on what ran beside it once a
  neighbour passed 4K tokens.
- Eight HTTP requests (four greedy, four seeded with the default sampling) at
  C = 1, 2 and 4, AR and DSpark: all 48 outputs byte-identical.
- Reasoning budgets (`reasoning_budget_tokens`) were ignored: the host's
  budget waits for the thinking opener as the first generated token, and the
  model's first token is the assistant header. The prompt now ends with that
  header (the model writes it with probability ~1), greedy device-argmax rows
  take the token the budget forces, and no second thinking block opens after
  a forced close. Budgets 0, 8, 16, 32 and 64 now cap the reasoning exactly.
- Functional suites on DSpark, `--sessions 4`, with vision: discovery,
  responses, stops, conversation, structured, tool-history, messages-tools,
  tool-choice-cache, tool-type-unions, tool-schema-edges, sampling-defaults,
  sampling-ranges, logprobs, state-edges, prefill-scheduling, cache-edits,
  cache-depth, image-inputs and the cache suite (disk restarts and
  cancellations) pass.
- Failing, with causes:
  - With thinking off (`enable_thinking: false`, most tool suites) the model
    puts nearly all its probability on opening a thinking block (the text
    header is at log-probability -24.5; the model card says the minimum
    effort "does not fully disable reasoning"). The runner forbids that block,
    and the model then sometimes writes its deliberation as the answer, or
    declines to call a tool ("I can't call tools directly from here"):
    tools/auto-tools, tool-agent, tool-untyped, tool-mixed,
    structured-limits, batch, reasoning-separator (thinking off case),
    tool-edges (writes the requested literal, then ends instead of calling).
    Letting the model think at level 0 and not streaming the thought was
    measured and rejected: the suites' 16 to 128-token limits ran out inside
    the thought (stops, structured, conversation, cache-edits and tools then
    failed), and the next turn's re-render no longer matched the generated
    tokens (tool-choice-cache lost its frontier).
  - tool-parameter-order: the chat template writes call arguments with sorted
    keys (`dictsort`), so the model emits them alphabetically, not in schema
    order.
  - tool-native-schemas (`parity_no_parameters_root_one_of`), tool-reasoning
    (malformed `oldText`), tool-streaming (with thinking on, the model called
    the tool without a thought): model output.
  - reasoning-budget: one fixture assumption. Its uncapped control is cut at
    192 output tokens, and the first of them is the model's thinking opener,
    so it counts 191 reasoning tokens; a 192-token budget correctly allows
    192.

## Round 6: sampler, selection and copy drafts (November 23)

The engine is unchanged; the host side is checked against the round-5 build:

- `sampling_test`: the whole-row temperature draw equals
  `Distribution(logits).Sample()` in token and RNG state for 1,200 rows
  (penalties, min-p, masked entries, rows with a single surviving token,
  200,058-entry rows).
- Functional suites (DSpark, `--sessions 4`): responses, stops, conversation,
  sampling-defaults, sampling-ranges, logprobs, batch, structured,
  tool-history, state-edges, prefill-scheduling and cache-edits pass or fail
  as before (`batch` fails on both builds, see above), and 446 of the 447
  recorded requests return the same output; the last is a request the
  harness sizes from the byte length of an earlier response (2,307 against
  2,306 tokens asked).
- HTTP: 64 concurrent-workload responses (eight prompts, AR and DSpark, C =
  1, 2, 4, 8), 28 single requests (greedy and seeded) and 8 copy requests
  are byte-identical between the builds and between AR and DSpark.

### Resolved: a stall with eight sessions and RAM checkpoints

`tests/functional/run.py --suite cache-concurrency` with `--sessions 8
--context 32768` used to stall on an uncached repeat after the groups of
four requests that share a prefix: the scheduler waited for device work that
never finished. On `53c31dd1` (the branch rebased on main `38a1589c`) it
passes 4 of 4 runs (58.0 to 59.0 s, one of them with DSpark), as it does on
a second machine and on Windows. It needed RAM checkpoints, whose K/V pages
the sessions share; main changed that path meanwhile (eager clears finished
before a state region is published, linear-time backing reservation, session
pages kept for checkpoints and restored by mapping).

## Round 11: serving (November 24)

Every engine and kernel output is unchanged: `inkling2_oracle` dumps of
both paths (9,300 WikiText-2 positions) are byte-identical to the round-10
build's, and `inkling2_cache_test` passes (restores and layered prefill
bit-exact).

- **Decoding inside prefill yields.** `inkling2_spec_test` now starts with
  seven sessions decoding in the yields of an eighth's 5,000-token layered
  prefill on a fresh engine (batches of 2 to 7, every third yield a batched
  verification): 11 yields, 59 decoded rows, 0 mismatched against each
  session's lone decode, and the yielded prefill's logits equal an unsplit
  prefill's.
- **DSpark on a real prompt.** The drafter check now decodes a chat prompt
  (a C coding request): 128 tokens equal to plain decoding, 82 of 315
  proposals kept (26.0%); the test requires 20%.
- **Reasoning off** keeps Gufo's framing (no thought). The official template's
  level 0, which llama.cpp uses for thinking off, still thinks: 29 to 345
  tokens on eight short prompts (greedy), 29 to 513 at the default 0.9. With
  it, 14 of the first 16 functional suites failed their no-reasoning
  assertions. Required calls with thinking on are now masked like those with
  thinking off (a thought, then one declared name and the invoke token).
- **Host memory**: a loaded server with DSpark holds 0.38 GB RSS against
  18.1 GB before on Linux (17.9 GB of file pages).
- **Cancellation**: after a client disconnects mid-prefill, the next
  request's one-token reply takes 42 to 60 ms against 0.09 to 1.74 s before;
  the 34K-token prefill throughput is unchanged within 0.3%.
- **Functional suites** (DSpark and vision, `--sessions 4 --context
  32768`; 22 suites plus the cache suite's disk and cancel cases) on this
  build and on `53c31dd1`: the same suites pass and fail (the failures are
  the model behaviour above: tools, tool-edges, tool-agent, tool-streaming,
  tool-reasoning, reasoning-separator, reasoning-budget, batch), and all 486
  matched requests return the same output; the one unmatched request is a
  `batch` request indexed by arrival order.
