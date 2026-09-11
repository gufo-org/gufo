# Qwen27B quality checks

Production targets: UD-Q4_K_XL and UD-Q8_K_XL. BF16 is an optional
quantization reference. A Gufo BF16 run is not independent proof that the
entire target model matches the original checkpoint.

## Maintained checks

Run on gfx1151 inside Nix. Model-specific tests live in
`tests/models/qwen27b`; shared quantization/operator controls remain in
`tests/models/qwen` and are selected by the same runner.

| Suite | Contract |
| --- | --- |
| `fast` | DFlash metadata/layout validation, NPU packing, strict result reporting. |
| `kernels` | Quantized GEMM versus independent/decode controls; DFlash causal grouped convolution, windowed attention and sparse selector probabilities versus CPU equations. |
| `model` | Target full-logit replay; MTP committed-feature alignment; DFlash ring/snapshot/restore; prompt and multi-turn GPU chat token-ID parity across AR/MTP/DFlash. |
| `serving` | Direct versus served tokens, sampled DFlash, bounded prefill, cache forks, persistent restore, concurrency, cancellation and reclamation. |
| `reference` | Teacher-forced target versus optional BF16: KL, total variation, top-1 agreement, RMSE and NLL difference. Informational quantization measurements. |

```sh
nix develop -c python3 tools/qwen27b/check.py fast
nix develop -c python3 tools/qwen27b/check.py kernels
nix develop -c python3 tools/qwen27b/check.py model \
  --model "$MODEL" --mtp-model "$MTP" --dflash-model "$DRAFT"
nix develop -c python3 tools/qwen27b/check.py serving \
  --model "$MODEL" --dflash-model "$DRAFT"
# Optional; load the two targets sequentially, never simultaneously.
nix develop -c python3 tools/qwen27b/check.py reference \
  --model "$MODEL" --reference-model "$BF16_REFERENCE"
```

The GPU correctness preset uses optimized code with symbols; test assertions
remain enabled. Performance measurements always use Nix release binaries.
Artifact variables `GUFO_QWEN27B_*_MODEL` are test inputs, not execution switches.

## Current evidence

- Q4 and Q8: three fixed 24-token prefixes, four forced continuation tokens
  each. Repeated prefill and snapshot continuation are byte-identical. All
  twelve batched-verifier logit rows per target equal scalar decode byte for
  byte; rejected-prefix replay also preserves subsequent logits. Prefill and
  scalar decode retain the same top-1 choice on these prefixes.
- MTP: feedback replay equals a fresh teacher-forced committed prefix. The
  regression fails when replay uses the newest target hidden row for the old
  proposal anchor.
- DFlash2: FP32 history is bounded by its 2,048-token attention window: 80 MiB
  at a 262,144-token logical context. Ring wrap, persistent restore and replay
  retain proposals and confidence values. Scratch ingestion is bounded to
  256 rows.
- DFlash operators: CPU equations cover both convolution coefficient planes,
  attention window boundaries and partial-top-k partitions. Top-k 1, 7 and 16
  check greedy selection and sampled probabilities, including poisoned unused
  scratch slots. This catches the former unwritten-slot merge for top-k <16.
- Release companion comparison: both targets × both draft quants × three
  chat prompts × two repetitions: all 24 cases match all 128 target token IDs.
  See [the measurement record](draft-selection.json). This is a development
  corpus, not a capability evaluation or proof across arbitrary contexts.
- `prompt` and `chat` share GPU/speculative setup. Their first-turn tokens
  match, and two chat turns reproduce AR with both draft backends. Verbose
  generation emits a SHA-256 over little-endian token IDs; the corpus rejects
  missing/incomplete traces and text-only matches.
- Removed an unused 170 MiB target scratch allocation per session. Inactive
  fusion/prefetch branches and their kernels/tests are deleted; production
  arithmetic and independent operator controls remain.

Run the affected operator check first, then model replay on both target quants.
Require complete speculative corpus results before comparing release speed.
Compare token IDs and full logits, not just decoded text or acceptance.
Use `tools/qwen27b/drafts.py` for matched Q4/Q8 companion comparisons; do not
rank draft acceptance across different target-generated continuations.

## Original DFlash2 source

Pinned upstream: `z-lab/dflash` commit
`07ebd93db9f472af339b644bb70221ad8428328a`, `dflash/model_mlx.py`.
Its SHA-256 is
`2f8598eaca4cb814e63ea69e791c1bdf55ba82280bfd299f9141906356b7cb87`.
The upstream README identifies MLX as the Qwen implementation.

| Operation | Source contract checked |
| --- | --- |
| Target taps | Concatenate selected target layer outputs; normalize GGUF layer-input indices once on load. |
| Context injection | Shared feature projection/norm, per-layer K/V projection, per-head K norm, absolute-position RoPE. |
| Attention | All proposal keys visible; historical keys satisfy query minus key < window; 32 query heads, 8 KV heads, dimension 128. |
| Dynamic convolution | Causal zero padding; static and dynamic grouped coefficients; separate prepare/finish planes. |
| Selector | Unary top-k then predecessor × projected hidden × successor score; selected token becomes the next predecessor; report the probability actually sampled. |

Gufo uses FP32 draft activations and packed weights; upstream may use BF16.
Equivalent formulas do not imply identical draft probability distributions.
The target verifier must preserve the target distribution independently of
proposal quality. Full original-target equivalence, an independent MTP source
audit, the optional BF16 comparison, and the longer capability/depth corpus
remain TODO.

Experiment: removing symmetric-format activation corrections, including explicit
rounded multiply/add, failed bitwise GEMM verification. Rejected; the production
reduction remains unchanged.

Short TG profile: exact quantized verifier projections account for about 73%
of kernel time, recurrent updates about 9%. Prioritize those before minor
launches such as draft RoPE (0.1% in that trace).
