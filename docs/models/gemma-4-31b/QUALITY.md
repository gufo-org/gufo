# Gemma 4 31B quality

**Decode matches the scalar reference; greedy MTP matches autoregressive
output.** Unsloth UD-Q4_K_XL target and Q8_0 `gemma4-assistant` drafter;
[identities](artifacts/model-identities.json). The arbiter is Gufo's scalar
CPU implementation of the llama.cpp `gemma4` graph over the same GGUF
(FP32 activations, binary16 KV, `src/models/gemma4/reference.cpp`); these are
not unquantized-model or GGUF-conversion checks. Measured September 26, 2026.

| Check | Result |
| --- | --- |
| GPU decode (FP32 activations, split-K attention) vs scalar reference, 22-token chat prompt | Mean KL 1.1e-6 (limit 1e-5), top-1 22/22 (9.7e-7 before 2026-10-05, when split attention began adding each row's own key last) |
| Eight-row verification vs scalar reference | Mean KL 1.4e-6, top-1 22/22 (8.4e-7 before the own-key step) (WMMA projections since 2026-10-05; double-stage FP32 GEMV before: 1.5e-6); verification rows reproduce single-token decode with the drafter loaded bit for bit (WMMA projection widths 1–16 in `gemma4.projection_ops`, attention rows ≤16) |
| GPU prefill (binary16 activations and WMMA projections/attention) vs scalar reference | Mean KL 5.5e-5 (limit 0.015), max 5.4e-4, top-1 22/22 (Q8_1 activations until 2026-10-05: 0.0055); llama.cpp b11069 on the same prompt: 0.038, top-1 21/22 |
| Bulk prefill vs exact rows, 1542-token conversation past the window and ring (`tests/models/gemma4/fixtures/long_conversation.txt`) | Mean KL 0.0045 (limit 0.15), top-1 1534/1542 (Q8_1 activations: 0.056–0.076, 1444/1542) |
| Bulk prefill vs exact rows, 1353-token repetitive prompt (reported, not a gate) | Mean KL 0.124, top-1 1279/1353 with binary16 activations; with Q8_1 activations 1.64, top-1 1016/1353; llama.cpp prefill vs the same rows 2.93, 920/1353. Many near-tied predictions make this prompt ill-conditioned for any binary16/Q8_1 prefill: rounding-only changes moved it between 1.33 and 1.64, so it no longer gates changes (replaced by the conversation above on 2026-09-27; its former limit was 1.6). The exact rows match the reference (KL 3.5e-6 over the first 96 tokens). |
| Long context vs llama.cpp, 16K templated turn (15.3K-token prose prompt, 640-token model reply) | KL(llama.cpp ‖ Gufo) 0.0092, top-1 628/641, true-token NLL 0.097 (llama.cpp 0.112); with the full global K cache: 0.0093, 628/641, 0.097 |
| Greedy MTP vs single-token decode with the drafter loaded, three prompts × 64 tokens, four drafts | Identical token IDs |
| Greedy MTP with sibling drafts vs single-token decode with the drafter loaded | After every cycle of three 96-token prompts the session's logits (a chain row's or an accepted sibling's) equal decode's bit for bit, 23 of 189 siblings accepted; the keys an accepted sibling leaves behind continue identically (`gemma4.target`) |
| Greedy MTP vs an AR-only server | Not equal in general: AR-only decode uses the faster split-K GEMV, whose FP32 summation order differs from the verification kernels, so long greedy completions can diverge (the pp2048 prose completion does). Both are FP32 decodes within the KL limits above. |
| Batched decoding | Up to eight sessions share a forward; with the drafter loaded each row reproduces its own session's decode bit for bit (`gemma4.target`: three sessions, plain and speculative greedy steps equal AR). Without the drafter, batches use the verification kernels while one session uses the faster AR GEMV, so a greedy completion can depend on concurrency (the multi-user AR tables matched their C1 outputs; bench.json's `"ar_exactness": "reported"` records a near-tie's count instead of failing the table, as on the [26B-A4B UD-Q6_K_XL](../gemma-4-26b-a4b/QUALITY.md#benchmark-method)). Multi-user MTP tables check every completion against Gufo's own MTP C1 (`exactness_reference: self` in bench.json), since greedy MTP does not equal an AR-only server |
| Session state | Prefix extension, rewind and snapshot restore after a sliding-ring wrap (typed and serialized) continue bit for bit; snapshots sharing position blocks hold the same payload as fresh copies, also after rewritten or ring-wrapped blocks; truncated snapshots are rejected |
| Tokenizer | Hugging Face `tokenizers` and llama.cpp agree on the 19 valid-UTF-8 entries of a 20-entry corpus (whitespace/newline runs, CJK, emoji, special tokens, 20k-character line); Gufo matches llama.cpp on all 20 |
| Chat template | Compiled renderer byte-identical to Jinja2 on 19 cases (system, thinking on/off, tools, tool history, reasoning history) |
| Global-layer keys | K and V share one projection, so the cache keeps V and only the 64 rotated rope pairs of K; the other key dims are V times `k_norm`, folded into the query. Attention with these keys matches FP64 like stored keys (split ≤5e-6, WMMA ≤0.007) |
| Kernels | Attention vs FP64 on window, ring, key-limit and 32K shapes (split ≤5e-6, WMMA ≤0.007 absolute), repeated launches bit-identical; every decode projection within 2e-6 of an FP64 dot relative to Σ\|wx\| (the WMMA verification projections within 1.6e-7, including activation blocks of zeros, 1e-6 and past the binary16 range) |
| Serving | Streaming, `reasoning_content`, Gemma tool calls and tool-result turns, multi-turn prompt reuse and disk-cache restore across a restart |
| Vision encoder | GPU vs the scalar FP64-accumulating reference (`vision::Reference`): embedding relative RMS 0.27% at 260 soft tokens, 0.41% at 1,107 (binary16 projection inputs; BF16 inputs gave 1.6% and 2.6%; the patch stage matches to 5e-7). llama.cpp's own encoder is 5.9% off at 260. `gemma4.vision_encoder` gates every stage at 1e-2 |
| Image prompts | Teacher-forced chart description (96 positions from `<image\|>`), text model fed Gufo's encoder vs fed the reference's embeddings: mean KL 0.0012 (260 tokens) and 0.020 (1,107; top-1 95/96). Against the scalar reference text model on the reference's embeddings (260): Gufo 0.0018 (answer positions 0.0003), llama.cpp with `clip.use_gelu` 0.0123 (0.0079). Stock llama.cpp runs the tower with GELU-quick (its converter omits `clip.use_gelu`) and lands at 0.224 on an earlier transcript |
| Image sessions | `gemma4.vision_session`: a red and a blue image behind identical tokens give different answers (KL 32); changing, rewinding, extending, snapshot-restoring and chunk-splitting image prompts all match fresh sessions. Over HTTP with `--cache-disk`, a follow-up after a restart restored 272 of 316 prompt tokens; the same conversation with a colour-inverted copy of the image (identical tokens) restored none |

Greedy MTP drafts the drafter's argmax and keeps it while the target's own
choice agrees. Sampled MTP samples each draft from the drafter's top-64
logits under the request's sampler (q) and accepts it with probability
min(1, p/q) against the target's filtered distribution p, drawing a rejected
position from the residual max(0, p − q); emitted tokens follow the target
distribution exactly (the rule is shared with Flash-Next and unit-tested
there). How many drafts a cycle verifies depends on the drafter's outputs,
earlier verification outcomes and a fixed cost table, never on timings, so
greedy equality holds under every `--draft-policy`. With `--draft-calibration
request`, a seed replays the same tokens (`gemma4.target`), also beside other
sessions (a sampled cycle then prices its drafts as if alone), though they consume
different random draws than an AR run with the same seed; with the default
shared calibration, what earlier requests taught the policy changes later
draft counts, so a repeated seed draws a different, equally distributed
sample. Prompt-lookup
copies are verified like drafts (sampled: as point-mass proposals under the
same rule); `gemma4.target` checks greedy equality and seeded replay on a
prompt that copies a paragraph, with accepted copies.

## Q8 quant

UD-Q8_K_XL, measured September 30, 2026 with `gemma4.target` (same prompts
and limits as above, Q8_0 drafter); prefill rows remeasured October 5 after
every projection moved to binary16 prefill activations. Its F16 projections
run on the binary16 kernels (decode FP32 activations; prefill binary16
activations with FP32 accumulation).

| Check | Result |
| --- | --- |
| GPU decode vs scalar reference | Mean KL 6.3e-7 (limit 1e-5), max 6.2e-6, top-1 22/22 |
| Eight-row verification | Bit-identical to single-token decode; the binary16 GEMV rounds every width from 1 to 16 rows identically (`gemma4.projection_ops`) |
| GPU prefill vs scalar reference | Mean KL 4.7e-6 (limit 0.015), max 2.7e-5, top-1 22/22 (Q8_1 Q8_0 inputs: 0.0025) |
| Bulk prefill vs exact rows, 1542-token conversation | Mean KL 0.00012 (limit 0.15), top-1 1539/1542 (before: 0.029, 1474/1542) |
| Bulk prefill vs exact rows, 1353-token repetitive prompt (reported) | Mean KL 0.19, top-1 1249/1353 (before: 1.41, 1018/1353) |
| Greedy MTP vs single-token decode, batched sessions, session state | Identical token IDs; bit-for-bit continuation (`gemma4.target`) |
| Image sessions | `gemma4.vision_session` passes |

## Reproduce

```sh
export GUFO_GEMMA4_MODEL=/path/to/gemma-4-31B-it-UD-Q4_K_XL.gguf
export GUFO_GEMMA4_MTP_MODEL=/path/to/mtp-gemma-4-31B-it-Q8_0.gguf
nix develop -c cmake --build --preset gpu-test --target gemma4_target_test \
  gemma4_attention_ops_test gemma4_projection_ops_test gemma4_kernel_ops_test
nix develop -c ctest --preset gpu-full -R '^gemma4\.' --output-on-failure
```

`gemma4.target` loads the model twice (without and with the drafter) and
takes about a minute; without the environment variables it exits 77, which
is a skip, not a pass. For teacher-forced comparisons with llama.cpp:
`tools/gemma4/build_llama_logits.sh`, then `gemma4_gpu_probe --tokens T.i32
--logits-out G.g4lg` and `llama_logits --tokens T.i32 --output L.g4lg`, and
`python3 tools/gemma4/compare_logits.py L.g4lg G.g4lg`.
Image prompts: `gemma4_vision_generate_probe --image FILE --out-dir D`
(`--image-tokens`, `--continuation`, `--embeddings`) and `llama_vision
--image D/image0.rgb W H --tokens D/tokens.i32 --logits-out L.g4lg`, whose
`--mmproj` should be a copy of the sidecar with `clip.use_gelu = true`
(see [VISION.md](VISION.md)); `gemma4_vision_probe` compares encoders
(`--output` writes the reference's embeddings) and `gemma4_reference_probe
--tokens-in D/tokens.i32 --image-embd E.g4ve` runs the scalar text model on
them.

## Benchmark method

Gufo October 17, 2026 (revision a9ff98e, `nix build`, binary SHA-256 prefix
d92d910545862f6b), llama.cpp September 26–27 (not re-measured); one warmed sample per point,
greedy, thinking off. MTP drafts up to 15 tokens under Gufo's calibrated
policy (the default) and up to four on llama.cpp and the fork. Single-user uses pp2048/tg128 after a cached
prefix of the stated depth. Loading: cold files (page cache dropped with
`echo 3 > /proc/sys/vm/drop_caches`), C1, MTP, capacity 262144. Memory: C1, AR,
capacity 133121, peak device-global HIP allocation. Loading and memory run
with the target in a directory without `mmproj-BF16.gguf`, since Gufo
prepares a sidecar found beside the target at load. The memory server
receives only the measured requests: the driver calibrates its synthetic
text once per variant and process, so memory runs after a sampled table in
the same process (a calibration probe on the memory server leaves 1.8 GiB of
prompt snapshots resident on the Q4 peak). Sampled MTP: five seeds on both
sides. The multi-user tables compare against Gufo AR C1 hashes regenerated
October 17; the greedy text has changed since September. The llama.cpp reference is
`b11069` (ROCm) from `flake.nix`. The third-party Vulkan fork
halo-box/strix-llama.cpp (`8c1c282ec`, Vulkan RADV,
`GGML_VK_MMV_NO_SPLIT=1 -b 2048 -ub 512`) was measured with the same driver; its results
are in [artifacts/fork](artifacts/fork). Commands, counts and server flags are
recorded per row in the artifacts.
Against the October 7 build: prefill +3.5% at d0 to +22% at 128K (Q4 AR),
AR decode within 0.6%, MTP decode +3.5–9% (Q4 mixed), memory +0.06 GiB.

UD-Q8_K_XL tables: Gufo October 17 (same build), llama.cpp September 30, 2026, a reduced grid
under the same driver, workloads and server flags: single-user depths 0, 4096,
16384 and 32768, sampled MTP at depth 0 (five seeds), one and two users,
memory and loading. Gufo drafts under its calibrated policy. Every Gufo
multi-user completion matches its C1 hash (AR against AR C1, MTP against MTP
C1); the AR C1 hashes llama.cpp's MTP rows are compared with were qualified
once with an isolated C1 AR run (`multi-*-q8-gufo-ar.json`).

The 2026-10-19 DPP reduction change preserves every existing attention
operator output byte for byte against 04d70a14, including each verification
row’s single-row evaluation. A separate A/B covers rows 1–16 at
0/4K/16K/32K/64K, both global head layouts, and normal/10×/30× queries;
all outputs are finite and bit-identical. The FP64 attention and kernel
operator checks pass. `gemma4.target` with the UD-Q4_K_XL target and Q8_0
assistant also passes: decode KL 1.11695e-6, eight-row KL 1.35631e-6,
prefill KL 5.46016e-5 (all 22/22 top-1), long-conversation KL 0.004864.
Greedy equivalence, siblings (23/189 accepted), sampled replay and snapshot
restore checks pass. [Timing and calibration evidence](artifacts/mtp-verification-2026-10-19.json).

## Coalesced attention qualification (2026-10-20)

`519dfe9c` (source-equivalent to `eb17a526`) versus `422fbe4d`, production
CMake `release` binaries, unchanged compiler/linker fingerprints. Candidate
SHA-256 `bb1dd37daf0459c8…` matches `build/release/gufo`; baseline
`a8038e8407374cae…`. The supplied 31B Q8_0 assistant is SHA-256
`f4f0edc6cca78252…`; the full target/drafter identities are recorded in the
[new artifact](artifacts/mtp-attention-e2e-2026-10-20.json). Historical tables
retain their original drafter identities.

The supplied seven-round operator A/B covers all 160 depth/head-layout/row
shapes (rows 1–16, depths 0/4K/16K/32K/64K, four and two KV heads).
Normal, 10× and 30× queries have finite, byte-identical outputs. Rebuilt
`gemma4.attention_ops` (independent FP64 oracle and row comparisons),
`gemma4.kernel_ops` and `gemma4.draft_policy` pass on the current source:

```sh
nix develop -c cmake --build --preset gpu-test --parallel 4 --target \
  gemma4_attention_ops_test gemma4_kernel_ops_test gemma4_draft_policy_test
nix develop -c ctest --preset gpu-full \
  -R '^gemma4[.](attention_ops|kernel_ops|draft_policy)$' --output-on-failure
```

The supplied candidate `gemma4.target` log is reused: decode/eight-row/prefill
mean KL 1.11695e-6 / 1.35631e-6 / 5.46016e-5, all 22/22 top-1;
long-conversation mean KL 0.004864. Greedy equivalence, siblings, sampled
replay and snapshot checks pass in that log. The full numerical output,
including reported repetitive-prefill limits, is retained without changing
tolerances or goldens.

Timing method: native pp2048/tg128 at d16384 then d32768, one fresh process
per ABBA arm, calibrated policy, cap 15, seed 1, `--verbose`. Prefill has
an untimed warmup; decode follows depth preparation without an extra decode
warmup. Native phase times are reconstructed from the printed 0.01-tok/s
rates. HTTP: `serve llm --sessions 1 --context 40960 --think off`, greedy,
seed 1, a short 16-token warmup per fresh server. Frozen-document summary
and essay request 400 tokens; code edit requests 512. The separate
`--draft-policy fixed --draft-tokens 11` control warms the same code prompt
with 128 tokens, then repeats it for 512 with all 31,161 prompt tokens cached.
This cap allows ragged lookup/tail rows. No loading-time measurement or
page-cache reset is included.

Every HTTP completion hash and prompt/cache/output count matches within its
four-arm case. Summary is cold (31,290 tokens); essay reuses 22,528 of
31,297 tokens, code edit 22,528 of 31,161. HTTP hashes establish text agreement
for these matched histories. Independent model accuracy remains covered by
the teacher comparisons above. Each arm's server phase metrics and client
wall time are retained; overlapping stages are not summed into a headline.
Paired comparisons require both >5% and >3 ms in both repeats. Prefill,
decode, TTFT and request wall remain within those thresholds. Essay maximum
inter-token gap fails: 154.02/153.92 → 167.80/167.33 ms. Summary/code snapshot
flags are mixed and inconclusive; the candidate is not blanket latency-qualified.

Separate fixed-11 rocprof traces isolate the final 512-token request using
client/SDK monotonic timestamps. Affected global verification attention is
20.8% of baseline GPU kernel time, 1,665 → 1,449 ms in the two profiles;
400 candidate launches use the new `WPH=4` route (four waves per head,
1024 threads per block, one row group per KV head/chunk) for 11/12-row
launches in the 9–12-row group. These instrumented profiles explain scope and dispatch;
unprofiled ABBA supplies the throughput results.

The cost-only diagnostic changes exactly the runtime 320-byte dense
verification table in a copy of the candidate; original binaries and source
remain unchanged. The duplicate constexpr data in DWARF is not modified.
It reruns the same fresh-server ABBA summary→essay history with the same
warmup, model files and settings. Texts and prompt/cache/output counts match;
previous prices restore baseline proposal/cycle counts and remove the essay
maximum-gap increase (candidate 152.36/152.48, baseline 152.10/153.94 ms).
The diagnostic summary snapshot phase still flags slower in both pairs;
identical unchanged-baseline controls span 86.56–99.49 ms, encompassing its
97.45–97.84 ms. The flag remains in the artifact and its cause is inconclusive.
This diagnostic does not clear the original candidate's failed gap check or
qualify a new production configuration.

A per-cycle trace with identical text in all four arms locates the failed
gap check in one essay cycle: the new prices verify two extra sibling rows
(164.9 / 167.4 ms instead of 152.1 / 152.3 ms) and that cycle emits two
tokens instead of one. No other essay cycle exceeds ~152.7 ms under either
table ([trace](artifacts/mtp-cycle-trace-2026-10-20.json)).

The rejected paired-head wave64 probe (October 20) preserves output bits across
64 global-attention cells (rows 1–16, depths 16K/32K, both target head layouts),
also at 10× and 30× query scales. Eight independent FP64 cases, including
derived keys and deep verification, have maximum error 4.91e-6 against the
unchanged 2e-5 limit; each row also matches its single-row evaluation. The
selective production-binary probe preserves responses and draft/acceptance
counts on six HTTP requests across 31B and 26B, including seeded sampling.
These are probe results; production attention remains unchanged.
[Raw evidence](artifacts/wave64-global-verification-2026-10-20.json).
