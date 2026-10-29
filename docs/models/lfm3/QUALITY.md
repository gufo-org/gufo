# LFM3 numerical qualification

October 28, 2026, Strix Halo `gfx1151`, GCC 16.2 / ROCm 10.1 HIP Clang 24.
Original BF16 and official Q8_0 GGUFs are pinned to Hugging Face revision
`b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf`.

## Independent checks

The native BF16 path is compared against llama.cpp
`8642541b255bfb5dc39cd351a178be09013a55d6`, using losslessly expanded F32
weights, FP32 K/V caches and FP32 inference (flash attention disabled). This isolates graph/operator differences from
llama.cpp's activation quantization. Compare identical token histories and
all 128000 logits, not independently generated continuations.

Four histories cover plain text, Unicode/numbers, chat framing and 515 tokens
crossing both chunk and sliding-window boundaries. All select the same next
token. BF16 versus teacher KL is below `1.2e-12`, total variation below `4.7e-7`;
maximum absolute logit error is below `3.1e-5`. The tolerance gates are KL
`1e-5` and total variation `0.002`, with matching top-1.

Q8_0 versus BF16 also agrees on top-1 for all four histories. KL ranges from
approximately `0.000080` to `0.00252`, below the `0.02` qualification gate.
This is a bounded quantization check, not evidence of equal downstream task
accuracy. F16 compatibility on the five-token plain-text history differs
from BF16 by at most `1.5e-5` in logits.

The scalar Gufo oracle independently agrees with the GPU Q8 forward on the
five-token history (maximum absolute logit difference `0.00034`). The separate
operator checks use FP64 formulas for convolution and gated attention, and
an independently decoded Q8 matrix product. Projection tolerance is `1e-4`;
attention tolerance is `2e-6`. Tests include 1/8/9/32/33/128/512 rows, ragged output
rows, high RoPE positions, zero-tail frequencies and sliding-cache wrap.

[Retained metrics](artifacts/quality.json). Full logits and process logs are
kept locally under ignored `artifacts/lfm3/logits-wired-fp32kv/`. Reproduction commands are
in [tools/lfm3](../../../tools/lfm3/README.md).

## State and API checks

The explicit `lfm3_session_test MODEL.gguf` covers 513-token chunked versus
serial execution, reset, independent sessions, invalid-token/context rejection
without state mutation and continuation. Prompt and decode attention now use
different FP32 summation orders, so batched and serial logits differ by at most
`0.0000067` (BF16) and `0.000015` (Q8, previously `0.000038`).
Reset reproduces the same path's logits exactly.

Tokenizer IDs match independently generated Hugging Face `tokenizers` goldens
for ASCII, contractions, Unicode, whitespace, numbers and control framing.
Fixed template byte goldens are independent of production constants.

The focused [HTTP suite](../../../tests/functional/lfm3.py) passes greedy and
streaming equality, seeded sampled replay, multi-turn fresh replay, a prompt crossing the 512-token
prefill boundary and two-session output isolation. Every response and its per-phase
latency are retained in local `artifacts/lfm3/http-optimized.json` (historical
smoke evidence). New session tests also check compact immutable snapshots,
restoration after dirty state, partial-prefill frontiers, malformed payload
rejection before mutation, and real multi-session C2/C3/C4/C8 decode. Full logits
match serial execution exactly for Q8 and BF16 at mixed depths 1/33/257/513,
including sliding-ring wrap. Independent native-tool ID goldens and a complete
136-token tool history agree with Hugging Face tokenization. Literal special
spellings remain user text without splitting ordinary whitespace BPE merges.

## Limits

No maximum-depth or perplexity qualification has been performed. Tools,
structured output and persistent snapshots are implemented, but full downstream
API qualification fails as documented below. Speculative decoding and multimodal
inputs remain unsupported. Full native
256K-context quality and performance remain unqualified. These focused checks
are opt-in and stay outside hosted model CI; only the small metadata/template
check joins the hosted PR suite.

## Full release text API matrix

The October 28 production-release Q8_0 run failed full API qualification:
**4 of 40 suites passed, 36 failed**. Passing suites were discovery, text-only
Messages image validation, sampling ranges and stream start. The dedicated
LFM3 smoke suite still passed. The full matrix retained 309 HTTP request records
and 280 check records; recorded checks are not all successful assertions.

Failures cover missing structured output/tools, effort/preserve-thinking controls,
custom stops and prefix reuse. Disk-cache startup rejects the unsupported mode.
A concurrent long task also produced numeric suffixes and literal `</think>` text
instead of the required code words; its cause remains unclassified.

Two fixtures initially exceeded the 8192-token server context. Focused reruns
retained those failures and raised capacity: the 10102-token shared-prefix case
failed exact output at 16K capacity; the deep-history case returned correct BETA
at 17481 and 18145 tokens but failed required reuse at 32K capacity. Full API
release qualification remains failed. Per-request timings are observations,
without a matched baseline speed claim. Evidence is retained locally under
`artifacts/lfm3/release-functional-20261028/`, including `SUMMARY.md`, raw SDK
reports, server logs and both focused context follow-ups.

## Serving integration follow-up

The full Q8 release run in `artifacts/lfm3/wired-release-final/` passed **30 of
46 checks**, including physical batched execution, reasoning budgets, native
tool history, Messages tools, schema edges, sampling controls, long context,
deep continuation and restart-safe disk recovery. It still failed 16 checks;
this is not a release quality pass. Raw requests, replies, logs and individual
phase timings remain retained. No matched HTTP speedup is claimed.

That run exposed a missing stable checkpoint before the assistant opening.
The final runner retains that boundary through the existing two-checkpoint
policy, preserving cancellation recovery as well as complete-prompt retries.
Focused replay in `artifacts/lfm3/wired-stable-prefix/` passes conversation,
sampling defaults and every cancellation/restart check (10 of 11 labels).
The remaining `cache-edits` failure restores 2048 tokens and reproduces its
cold control, but the model returns a sentence instead of the required `BETA`.
The prior full run also encountered bounded cache pressure on intermediate
history; focused evidence does not retroactively turn that run into a pass.

Remaining failures include unfinished reasoning within fixture token limits,
incorrect tool arguments/selections, and exact-answer failures. Two failures
were checked independently on their actual fixed token histories:

* The structured Nostradamus answer produces `1498` instead of the required
  `1503`. At the first birth-year token, both upstream F32 and upstream Q8
  prefer `149`; Gufo BF16 agrees with the F32 teacher (max logit error
  `9.5e-5`, KL `2.4e-10`).
* The 6601-token concurrent task produces `ALPHA.59</think>ALPHA.59`. After
  `ALPHA`, both upstream F32 and Gufo BF16 strongly prefer the incorrect period
  over end-of-turn; max logit error is `0.00025`, KL `4.1e-13`. Standalone Q8
  reproduces the same bad answer without concurrent scheduling or cache reuse.

These establish checkpoint failures at those decisions; other failed fixtures
remain unqualified. Details and full logits are retained in
`artifacts/lfm3/wiring-diagnosis/`. No test expectations, numerical tolerances
or timing margins were relaxed. The shared JSON constraint fix gives omitted
array `items` its JSON Schema meaning and preserves finite const/enum arrays;
independent tests cover JSON, Qwen and DeepSeek tool formats.

The real Pi 0.87.0 replay passes both selected workflows: creating/readback of
`hello.txt` and reading/editing/verifying `src/calc.py`. Each makes three real
tool calls with no tool errors. Tool results and final fixture files were
inspected (`bonjour`, and `return a + b`, respectively). The sessions, exact
HTTP/SSE and request timings are retained in
`artifacts/lfm3/wired-pi-absolute/replay/`. A preceding harness invocation used
a relative output path and sent no requests; that failure remains retained in
`artifacts/lfm3/wired-pi/` and is not model-quality evidence.

The shared compiler is covered by CPU tests for all three native tool formats.
GPU functional evidence here covers LFM3; the other four model profiles were
not rerun and no cross-model release pass is claimed.

## October 29 optimization pass

Every changed operator kept or lowered its error against the unchanged FP64
operator formulas. Maximum errors: Q8 projection at 512 rows `8.6e-7`
(three-digit encoding `1.13e-6`), prompt attention at 512 rows `6.0e-7`
(`8.6e-7`), decode attention `1.5e-7` (`3.0e-7`). A new 40-row prompt over a
17000-token history passes the `2e-6` gate split into key spans (`1.4e-6`);
the same case unsplit fails it (`2.7e-6`). Decode cases at position 4999
cover multi-tile key spans.

Against the scalar Gufo FP32 oracle on the same random token histories
(Q8 weights, all 128000 logits of the last position):

| Tokens | Before KL | After KL | Before max logit error | After max logit error |
| --: | --: | --: | --: | --: |
| 600 | `1.44e-10` | `1.24e-10` | `8.4e-5` | `5.7e-5` |
| 1100 | `7.81e-10` | `5.38e-10` | `2.3e-4` | `2.0e-4` |

Top-1 matches in all cases. Layout-only steps (shared splits, fused SwiGLU,
fused residual/norm/split, 16-byte stores, staged down-projection splits)
produce byte-identical logits at 3, 36, 1100 and 9000 tokens, BF16 included.
The llama.cpp teacher comparison was not repeated in this pass.
