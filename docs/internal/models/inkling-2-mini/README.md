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

Not supported: DSpark speculative decoding (no implementation exists to port).

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
