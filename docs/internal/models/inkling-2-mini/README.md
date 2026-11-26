# Inkling 2 Mini implementation notes

Developer detail behind the [public model page](../../../models/inkling-2-mini.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Forward pass](FORWARD.md)

Supported target: `unsloth/Inkling-2-Mini-GGUF` revision `8078d149286e`,
`Inkling-2-Mini-UD-Q4_K_XL.gguf`, in the layout mainline llama.cpp loads
(architecture `inkling2_loop`, 24 physical blocks). Unsloth's earlier layout
of the same tensors (architecture `inkling2`, 48 layer visits; the root file
before `8078d149`, now in `Old_Support/`) is not supported. Optional vision
projector: Unsloth's `mmproj-BF16.gguf`.

## Origin

The engine, kernels, tokenizer, chat template, tool dialect and vision encoder
come from the MIT-licensed `headbouyJB/inkling2-gufo` extension at `8bb6cd5`
(0.1.2), built against Gufo 0.35.0. `src/models/inkling2/LICENSE.inkling2`
and `THIRD_PARTY.md` carry its license and attributions. Changes made here:

- Session state is a `mem::StateDeclaration`: the 16 global-attention K/V
  buffers are demand-mapped context regions charged to the session's state
  account, the 80 window rings, convolution histories and slot table fixed
  buffers. The core allocates them (`mem::SessionState`). Every writer backs
  rows first (`Session::BackRows`, restores `EnsureRows`).
- Checkpoints keep the global K/V pages instead of copying them
  (`BorrowRowsView`; writes call `PrepareWrite`/`SettleRemaps`, restores
  `Region::Restore`).
- Up to eight checkpoints are captured inside one prefill pass
  (`Engine::Forward` with `checkpoints`). The window rings (window + ubatch
  rows) still hold a boundary's last 1,024 rows at the end of its ubatch and
  the global rows are append-only, so each visit copies only its convolution
  history at the boundary as it computes it, the head writes the boundary
  row's logits, and the drafter's Markov context comes from the same row.
  `CanCaptureInPass` refuses a boundary whose window rows the ubatch would
  overwrite; the host then ends the pass there.
- The runner claims demand-mapped state and exposes the state account, so
  `--sessions` share memory when they do not all fit at full context. The
  default context is the model's native 1,048,576 tokens.
- No environment variables: the tuning, profiling and debug switches were
  removed with the code paths only they reached (MALL prefetch, second decode
  stream, alternative MoE-down kernel, tap printing, slow generic prefill
  paths).
- Prefill and decode read the router and loop-controller weights as exact
  BF16 copies of the file's BF16-valued F32 tensors (the extension rounded
  them to F16, which changes every router value and 99% of the controller
  values). The device checks at load that every value is BF16-exact; a tensor
  that is not stays F32.
- Faster exact prefill and decode kernels; see [Benchmarks](BENCHMARKS.md).

## DSpark

`dspark.{hpp,cpp}` and `hip/dspark.{hpp,hip}` run Unsloth's
`dspark-Inkling-2-Mini-Q8_0.gguf` (architecture `dflash`): the checkpoint's
3-layer block drafter, written from llama.cpp PR #32218's description of it.
It reads llama.cpp's layout of the drafter (the root file since `8078d149`):
per-layer `ctx_mix` already softmaxed, `ctx_norm`, `markov_w1` over the
200,058 tokenizer rows, and `markov_w2` plus bias, the Markov output folded
through the BF16 unembedding in FP64. Unsloth's earlier layout of the same
weights (`dspark_fuse`, `dflare_norm`, padded `markov_w1`, `hidden_out`; now
`dspark-Inkling-2-Mini-Q8_0_Old.gguf`) is not supported.

- **Context.** After every prefill ubatch, decode step and verification
  commit, the engine copies the target's hidden states at visit slots 5, 29
  and 47 (the inputs of visits 5, 23 and 41) and the final normed output into
  a feature buffer; `Drafter::Inject` mixes them per drafter layer (`ctx_mix`)
  into context K/V rows (K normed per head and rotated, NeoX
  RoPE, theta 1e6) and the Markov context `f` of the last token. Sessions
  declare the drafter's rings (layers 0 and 2, 1,024-token windows) and its
  global layer (a demand-mapped region) with the rest of their state, and
  checkpoints hold them too (six row components and the Markov vector; the
  metadata records the context's bounds), so a restored prompt keeps its
  proposals. A context that does not reach the frontier restarts there.
- **Proposals.** `Drafter::Propose` runs the 8-row block [anchor, mask x 7]
  through the three layers (non-causal over the block plus the context), then
  the gated RNN Markov head proposes greedily, one row at a time. Proposals
  come from the 65,536 LM-head rows with the largest `markov_w1` norms (a
  Q8_0 copy; the Loom package's choice; the acceptance counts did not
  change). A row's logits are W_U h_i + `markov_w2` lat + bias: one 8-row
  pass over the head per draft gives every W_U h_i, and each Markov step
  reads only those tokens' 65,536 x 256 `markov_w2` rows (int8, one scale per
  row, 16.8 MB) instead of the 160 MB head. The drafter's Q8_0 matrix
  products quantize their F16 activations to int8 (one scale per 32 values)
  and run on int8 WMMA; this changes only the proposals, never the verified
  tokens.
- **Verification.** `Engine::Verify` evaluates the anchor and up to seven
  proposals of one session in one batched-decode graph (decode steps are 8
  rows wide; batches of sessions stay at 4). On global visits up to five rows
  share one attention launch (15 of its 16 query slots), more run as two
  interleaved groups (adjacent blocks share the K/V tiles through L2). The
  three-input projections (q / gate, k, v) give each block the rows of one
  input, so up to five tokens take one weight pass (six to eight: two).
  Every row has the
  arithmetic of a one-token decode at its position: the conv kernels read the
  history of rows 0..b-1 from the step itself (the stored history before row
  0), write no history, and capture their inputs; `CommitVerify` sets each
  history to the last three inputs of the kept rows. A verification never
  crosses an attention-split bucket (`VerifyRows`).
- **Selection.** `Inkling2Runner::DecodeStep` selects each verified row's
  token exactly as `SelectNext` would after the tokens before it (masks, the
  request's sampler on a working copy; greedy requests take the device
  argmax and leave the logits on the device until something reads them; with
  a grammar, a closed reasoning budget or a required call the argmax is kept
  when the grammar and the runner's masks allow it, else that row's logits
  are read) and
  keeps a proposal while it equals that selection. The first selection that
  differs becomes the next cycle's anchor (`pending`). The emitted tokens and
  draws are therefore those of AR decoding.
- **Coupled proposals.** A request that samples with temperature only (no
  top-k or top-p, the model's default) gets proposals drawn from the
  drafter's own soft-capped distribution with the uniform its sampler will
  use for that row (read in advance from a copy of the request's RNG). The
  sampler's linear draw walks tokens in id order, as the proposal head's rows
  are stored, so the proposal equals the target's draw whenever the two
  distributions agree around the uniform. Other sampled requests (top-k,
  top-p) get greedy proposals and keep a proposal only when the request's own
  draw selects it, so their tokens are those of AR decoding. When every draw
  takes exactly one uniform (temperature 0.1 or more, no min-p, penalties,
  constraint or reasoning budget), the device samples each row with its
  uniform known in advance and the logits stay on the device; with logprobs
  the rows' host samplings run in parallel instead.
- **Copy drafts.** A sampled request whose last generated tokens repeat an
  earlier stretch of its context (`speculative::PromptLookup`: 24 tokens, 5
  where the last accepted copy left off) proposes the tokens that followed
  that stretch instead of the drafter's; a copy that keeps fewer than half
  of its tokens skips the next opportunities. Greedy requests do not copy.
- **Several requests.** `DecodeBatch` (2 to 8 requests) drafts for all of
  them in one drafter pass and verifies their rows in one step of up to 16
  rows; from 6 rows the step groups the MoE pairs by expert, since
  consecutive rows of a session share most of their experts. Without
  proposals it runs one batched one-token step of their anchors. Each next token is selected as the request's next anchor, so a
  request moves between single and batched steps with the same draws. Every
  eight drafted cycles, a batch width whose cycles emitted fewer tokens per
  millisecond than its measured plain steps stops drafting for a while.
- **Selection.** Greedy rows take the device argmax and device-sampled rows
  their device token. Other sampled rows come back through a pinned buffer;
  up to three are selected one after another on the calling thread, more on
  persistent threads.
- **Width.** `dspark_policy.hpp`: the drafter's probability for each proposal,
  through a calibration of acceptance per probability bin that keeps learning,
  gives the survival of each prefix; the width maximises expected tokens per
  measured cycle cost (verifying 2/4/8 rows costs about 26/36/55 ms against
  20 ms for one); greedy and coupled proposals keep separate calibrations.
  Drafting stops early once a longer width cannot pay. Every eight drafted
  cycles, a request whose cycles emitted fewer tokens per millisecond than
  its measured plain steps decodes plainly for a while (8 to 64 steps) before
  probing again.

## Serving

- **Loading.** The target and drafter weights stream to the device through
  `hip::WeightUpload`, which reads the files without faulting their mappings.
  The router, loop controllers, bias tables and padded sink projections are
  converted on the device from their uploaded copies, and the drafter's
  `markov_w1` row norms are computed there; only the selected `markov_w2`
  rows and bias are read through the mapping, whose pages are released once
  the engine is built (`platform::ReleaseMappedPages`).
  A loaded server with DSpark holds 0.38 GB of host memory (18.1 GB before
  on Linux, 17.6 GB on Windows, where a mapping's touched pages stay in the
  working set and the RAM prompt cache, sized from free memory, got 0.53 GB
  against 10.7 GB for Qwen 27B on the same binary).
  Weights and scratch are `hipMalloc`ed only: no managed fallback, which on
  Windows would be host RAM outside the GPU carve-out.
- **Cancellation.** A cancelled request's prefill stops at the next
  cancellation point: the end of a layer visit and, in passes over 2,048
  tokens, also between a visit's attention and expert halves. While a
  cancellation check is set, the pass keeps one point queued ahead of the
  device (an event per point), so the check runs while the device works
  instead of milliseconds after the whole pass was queued. After a client
  disconnects mid-prefill the next request starts within 42 to 64 ms (up to
  1.7 s before; 63 to 103 ms with one point per visit of a 4,096-token pass);
  a 34K-token cold prefill runs at 1,972 tokens/s against 1,978 without the
  waits.
- **Pass width.** A prompt that no other request can join prefills in passes
  of up to 4,096 tokens (`Engine::kUbatch`). While a session is free for an
  arrival, or another one decodes, passes stay at 2,048 tokens
  (`kSharedPassTokens`, the scheduling profile's `pass_tokens`), so an
  arrival still waits for at most one 0.9 s pass. Every pass costs about
  115 ms plus 0.38 ms per token: each visit dequantises every routed expert
  once, however few tokens it serves. As of November 26, 2026,
  `inkling2_bench` prefills 2,090 tokens at
  2,292 tokens/s (2,033 with 2,048-token passes, which needed a second pass
  for 42 tokens), 4,096 at 2,404 (2,227) and 8,192 at 2,323 (2,177); 8,192-token
  passes would add 4% at 8,192 tokens and double the scratch again. Against
  2,048-token passes the server holds 0.5 GiB more GPU memory with one
  session and 1.8 GiB more with four (33.7 and 36.2 GiB after two 8,192-token
  prompts): every session's window rings grow by 336 MiB, and the area a
  pass spills its live rows to while another session prefills in its yield
  (0.5 GiB) is allocated only when the server has more than one session.
- **Layered prefill.** A pass yields to the decoders between up to 16 visit
  groups, and every decoding session steps in each yield
  (`max_yield_decode_width = Engine::kMaxDecodeBatch`): batched decode and
  verification inside a yield equal decoding alone, byte for byte
  (`inkling2_spec_test`, first check).
- **Images.** `ImageCache` keeps decoded images (128 MiB of host memory,
  matched on the uploaded bytes and detail) and encoder rows (256 MiB of
  device memory, matched on the SHA-256 of the pixels), so a conversation
  that sends its images again decodes and encodes each once. `Idle()` frees
  the encoder's workspace, which grows to the largest image seen (up to
  about 1 GiB).

## Reasoning and tool calls

The prompt is the official `chat_template.jinja` (byte-identical to the one
in the GGUF, effort levels none 0, minimal 0.1, low 0.2, medium 0.7, high
0.9 by default, xhigh and max 0.99) followed by the assistant header the
model writes first. The template has no off switch: at level 0 (`none`) the
model thinks for 29 to 345 tokens on short prompts. Gufo renders thinking off
as level 0 like llama.cpp and lets the model think: the runner reports
`InitialOutputState::kAuto` for every request, so the parsers and the grammar
accept a thought that the model opens itself. Forcing it off (an answer block
or an empty thought in the prompt, the thinking token masked) moved the model
off its own distribution: with `tool_choice: "auto"`, after the mask it wrote
the call's arguments as text (`tool-schema-edges` text-only case), where
llama.cpp thinks briefly and calls.
The functional suites ask for thinking off to get short fixed answers;
Inkling's Devlab entry sets `thinking_off_reasons`, so they bound that thought
with a reasoning budget and accept it (`tests/functional/thinking.py`).

A required call (`tool_choice` required or a named function) masks the
tokens after a header to a thought (unless a budget closed one) or one
declared function name and the invoke token, so the generated call is the
tokens the next turn's render writes; the host's grammar constrains the
streamed call (`stream_dialect.hpp`). The template writes call arguments
with sorted keys, so the model emits them alphabetically.

The template differs from Thinking Machines' renderer (`tml-renderers`
0.5.0, `renderers.Inkling2()`) in how it spells numbers, and Gufo follows the
renderer, as vLLM and SGLang do. Floats are positional for decimal exponents
in [-5, 16) and otherwise use unpadded exponents: tool declarations write
`1e+20`, call arguments `1e20`, and `1e-5` is `0.00001` in both (Python's
`repr` writes `1e-05`). A numeric effort is rounded to thousandths, then to
hundredths, halves to even (0.125 is "0.12"). The reference refuses integers
over 64 bits; Gufo keeps them, as the template does. The four render goldens
that hold such numbers are marked `"spelling": "tml-renderers"`. As of
November 26, 2026, every other checked prompt matched the renderer's token
IDs, apart from the header Gufo appends. Across 92 agent prompts the model
writes that header first with probability at least 1 - 1.1e-8.

vLLM and SGLang add a `tool_choice` instruction that `tml-renderers` does
not write. The text is the same in both: a required call, a named function
and `parallel_tool_calls: false` each append one sentence to a leading system
message, or else to the effort directive. Gufo keeps `tool_choice` out of the
prompt, so a conversation that switches it stays cached (#619); the grammar
forces the call. Sampled without a grammar, the model called a tool 0 times
in 12 for "Hi" and "17 × 23" without the sentence and 12 times in 12 with it.
Under the grammar alone, its thought decided against a call, and 3 of 6
forced calls on the arithmetic question asked for the weather in some city.
With the sentence, every forced call was a search, but thoughts before a
required call grew (greeting 41-151 tokens without it, 324-757 with it).

A turn is a list of blocks, and the model can think again after an answer
block: text, then a thought, then text. A block-boundary probe over 184 agent
replies found this only where the model doubted a finished answer, at 21% in
that context and about 0 elsewhere. After a call, only further calls followed.
The reasoning markup sets `repeats`: the parsers move each later thought to
the reasoning in order, as llama.cpp, vLLM and SGLang return it, so Chat
Completions joins the thoughts and the text. Responses and Messages stream
the blocks as separate items. `ChatMessage::parts` replays them in place,
from Chat `thinking` parts, Messages blocks or consecutive Responses items,
and each text part renders as its own block, as in the reference.
`tml-renderers` itself refuses to write a thought after text in OpenAI Chat
form ("a thinking part after content has no OpenAI Chat form"); its
`bridge_to_next_turn` keeps the sampled tokens.

The prompt context declares itself independent of the tokenization when it
holds no images. The host then aligns a replayed answer to the frontier's
sampled tokens (#1154). The model sometimes samples splits that differ from
the tokenizer's own, for example " cha", "î", "nes" for "chaînes".

## Numerical contract

Every optimisation kept the per-output arithmetic: the same dequantisation,
WMMA operands and ascending-k accumulation, and the same rounding steps (the
fused MoE tail turns off FMA contraction where the kernels it replaced could
not contract). The checks are byte comparisons of `inkling2_oracle` dumps
(logsumexp and top-64 of all 9,300 positions) against the pre-optimisation
build, for the batched prefill path and the one-token decode path, and
`inkling2_bench --verify`, which compares batched decode of 2 to 8 sessions with
single-session decode.

## Tests and tools

| Target | What it checks |
| --- | --- |
| `inkling2_probe_test` | Probe on synthetic GGUFs (CPU, CTest `inkling2.probe`) |
| `inkling2_text_test [--model GGUF]` | Chat-template text and error goldens from `transformers`, JSON, tool dialect and reasoning controls (CPU, CTest `inkling2.text`); with a model also the tokenizer goldens from the HF tokenizer and the template's tokens |
| `inkling2_cache_test --model GGUF --ids IDS [--chunk-prompt P] [--chunks-only] [--dspark DRAFT]` | GPU: chunk invariance (a prompt split into calls, or a restored prefix plus a tail, equals one prefill), checkpoint capture and restore, restore into a used slot (with a drafter, its proposals after a restore), layered prefill; all bit-exact |
| `inkling2_oracle` | Teacher-forced dumps; `tests/oracle/compare_dumps.py` scores KL and top-1 against a reference dump |
| `tests/oracle/llama_dump.cpp` | The llama.cpp side of the same dump (CPU backend, or `NGL` layers on the GPU) |
| `inkling2_spec_test --model GGUF [--dspark DRAFT]` | GPU: decoding inside a layered prefill's yields (batches of 2 to 7, batched verification) and verification rows of 2 to 64 tokens byte-identical to one-token decoding; with a drafter, greedy DSpark on a real chat prompt equal to plain decoding with at least 20% of proposals kept (26% measured) |
| `inkling2_bench` | Prefill and decode speed, batched decode, `--verify`; `--spec R` verification timing, `--dspark` speculative speed, acceptance and calibration |

The GPU tests need the model and run by hand; hosted CI builds the model
(Linux and Windows) and runs the two CPU tests.

## Possible improvements

Exact, not done yet:

- **Required calls through the grammar.** With reasoning on, the prompt ends
  in a header and the shared grammar writes its call opener before every
  call, so it cannot describe the call; the runner's mask (Select) does.
  For a required call the shared grammar could replace the mask only once
  it can name the function twice.

Kept on purpose:

- **Unicode tables.** The pre-tokenizer's tables (`text/unicode_tables.cpp`)
  are Unicode 16.0, as the tokenizer goldens. ICU (Unicode 17.0), which the
  Qwen tokenizers use, classifies 4,700 code points differently (newly
  assigned ones, and U+0295 as Lo instead of Ll), so switching would change
  tokenization.

Not exact: see [Experiments](EXPERIMENTS.md#not-exact-not-taken).
