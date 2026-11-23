# Inkling 2 Mini implementation notes

Developer detail behind the [public model page](../../../models/inkling-2-mini.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Forward pass](FORWARD.md)

Supported target: `unsloth/Inkling-2-Mini-GGUF` revision `45fd08764f2b`,
`Inkling-2-Mini-UD-Q4_K_XL.gguf` (root layout, architecture `inkling2`, 48
layer visits). The `Mainline_Support` copy (architecture `inkling2_loop`, 24
physical blocks, the layout mainline llama.cpp loads) holds the same tensors
and loads too. Optional vision projector: Unsloth's `mmproj-BF16.gguf`.

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
- The runner claims demand-mapped state and exposes the state account, so
  `--sessions` share memory when they do not all fit at full context. The
  default context is the model's native 1,048,576 tokens.
- No environment variables: the tuning, profiling and debug switches were
  removed with the code paths only they reached (MALL prefetch, second decode
  stream, alternative MoE-down kernel, tap printing, slow generic prefill
  paths).
- Decode reads the router and loop-controller weights as exact BF16 copies of
  the file's BF16-valued F32 tensors (the extension rounded them to F16, which
  changes every router value and 99% of the controller values).
- Faster exact prefill and decode kernels; see [Benchmarks](BENCHMARKS.md).

## DSpark

`dspark.{hpp,cpp}` and `hip/dspark.{hpp,hip}` run Unsloth's
`dspark-Inkling-2-Mini-Q8_0.gguf` (architecture `dflash`): the checkpoint's
3-layer block drafter, written from llama.cpp PR #32218's description of it.

- **Context.** After every prefill ubatch, decode step and verification
  commit, the engine copies the target's hidden states at visit slots 5, 29
  and 47 (the inputs of visits 5, 23 and 41) and the final normed output into
  a feature buffer; `Drafter::Inject` mixes them per drafter layer (softmax of
  `dspark_fuse`) into context K/V rows (K normed per head and rotated, NeoX
  RoPE, theta 1e6) and the Markov context `f` of the last token. Sessions
  declare the drafter's rings (layers 0 and 2, 1,024-token windows) and its
  global layer (a demand-mapped region) with the rest of their state. A cache
  restore drops the context; it restarts at the restored position.
- **Proposals.** `Drafter::Propose` runs the 8-row block [anchor, mask x 7]
  through the three layers (non-causal over the block plus the context), then
  the gated RNN Markov head proposes greedily, one row at a time. Two ideas
  from the Loom package: the head reads `h_i + W_o latent + b_o` in one GEMV
  (instead of the LM head of each term), and proposals come from the 65,536
  LM-head rows with the largest `markov_w1` norms (a Q8_0 copy, 160 MB per
  step instead of 490 MB; the acceptance counts did not change).
- **Verification.** `Engine::Verify` evaluates the anchor and up to three
  proposals of one session in one batched-decode graph. Every row has the
  arithmetic of a one-token decode at its position: the conv kernels read the
  history of rows 0..b-1 from the step itself (the stored history before row
  0), write no history, and capture their inputs; `CommitVerify` sets each
  history to the last three inputs of the kept rows. A verification never
  crosses an attention-split bucket (`VerifyRows`).
- **Selection.** `Inkling2Runner::DecodeStep` selects each verified row's
  token exactly as `SelectNext` would after the tokens before it (masks, the
  request's sampler on a working copy; plain greedy requests take the device
  argmax) and keeps a proposal while it equals that selection. The first
  selection that differs becomes the next cycle's anchor (`pending`). The
  emitted tokens and draws are therefore those of AR decoding.
- **Width.** `dspark_policy.hpp`: the drafter's probability for each proposal,
  through a calibration of acceptance per probability bin that keeps learning,
  gives the survival of each prefix; the width maximises expected tokens per
  measured cycle cost (verifying 2/3/4 rows costs about 28/33/38 ms against
  22 ms for one). Drafting stops early once a longer width cannot pay, and a
  request whose cycles stop paying decodes plainly for a while (4 to 32
  steps) before probing again.

## Numerical contract

Every optimisation kept the per-output arithmetic: the same dequantisation,
WMMA operands and ascending-k accumulation, and the same rounding steps (the
fused MoE tail turns off FMA contraction where the kernels it replaced could
not contract). The checks are byte comparisons of `inkling2_oracle` dumps
(logsumexp and top-64 of all 9,300 positions) against the pre-optimisation
build, for the batched prefill path and the one-token decode path, and
`inkling2_bench --verify`, which compares batched decode of 2 to 4 sessions with
single-session decode.

## Tests and tools

| Target | What it checks |
| --- | --- |
| `inkling2_probe_test` | Probe on synthetic GGUFs (CPU, CTest `inkling2.probe`) |
| `inkling2_text_test --model GGUF` | Tokenizer and chat-template goldens from the HF tokenizer and `transformers` (CTest `inkling2.text` skips without a model) |
| `inkling2_cache_test --model GGUF --ids IDS` | GPU: checkpoint capture and restore, restore into a used slot, layered prefill; all bit-exact |
| `inkling2_oracle` | Teacher-forced dumps; `tests/oracle/compare_dumps.py` scores KL and top-1 against a reference dump |
| `tests/oracle/llama_dump.cpp` | The llama.cpp side of the same dump (CPU backend, or `NGL` layers on the GPU) |
| `inkling2_bench` | Prefill and decode speed, batched decode, `--verify` |
