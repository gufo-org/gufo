# Gemma 4 31B vision: implementation plan

Image input for the native Gemma 4 engine (`src/models/gemma4`), served through
the same `--mmproj` sidecar option as Qwen. Text-only behavior, speed and
snapshots stay unchanged when no image is present.

## Sources

- Vision sidecar: `unsloth/gemma-4-31B-it-GGUF` `mmproj-BF16.gguf`, SHA-256
  `7a4601b12ec680f706a7e0c7e1f78f579b1db64d485a9e352ee87d4b9daa45e4`
  (1,200,726,496 bytes, downloaded 2026-09-28 next to the Unsloth target).
- Graph semantics: llama.cpp `tools/mtmd/models/gemma4v.cpp`, `clip.cpp`
  (`PROJECTOR_TYPE_GEMMA4V`), `mtmd.cpp` and `src/models/gemma4.cpp`; the repo reference
  `.#llama-cpp-reference` (b11069) ships `clip_graph_gemma4v` and
  `llama-mtmd-cli`.
- Preprocessing and prompt: `google/gemma-4-31B-it` `processor_config.json`,
  `config.json`, `chat_template.jinja`, and transformers `models/gemma4`
  `processing_gemma4.py` / `image_processing_gemma4.py`.

## Architecture facts

**Sidecar** (`clip`, projector `gemma4v`, 356 tensors): 27 ViT blocks, width
1152, 16 heads of 72, FFN 4304, eps 1e-6, patch 16, `projection_dim` 5376.
Matrices are BF16; norms, `v.patch_embd` (16×16×3→1152, no bias),
`v.position_embd` ([2][10240][1152]) and `v.std_bias`/`v.std_scale` are F32.
No tensor has a bias, and this checkpoint has no clamp scalars
(`use_clipped_linears: false`).

**Encoder**, per image of `W×H` pixels (both multiples of 48):
1. `p = 2·(rgb/255) − 1`, then a 16×16 patch convolution. Patches are row-major:
   `x = i mod (W/16)`, `y = i div (W/16)`.
2. `h = patch + pos[0][x] + pos[1][y]` (learned tables, no interpolation).
3. For each block:
   - `a = rms(h)·ln1`.
   - `Q = rope2d(rms(Wq a)·q_norm)`, `K = rope2d(rms(Wk a)·k_norm)`, `V = rms_noweight(Wv a)`.
   - Full bidirectional attention with **scale 1.0**.
   - `h += rms(Wo·att)·attn_post_norm`.
   - `h += rms(Wd(gelu_tanh(Wg b)·(Wu b)))·ffn_post_norm`, with `b = rms(h)·ln2`.
4. `rope2d`: NEOX on each half of the head. Dims [0,36) rotate by `x` and
   [36,72) by `y`, each as pairs (i, i+18), θ = 100.
5. Pool: 3×3 average over the patch grid, giving `(W/48)·(H/48)` rows, then
   multiply by √1152.
6. Standardize: `(h − std_bias)·std_scale`.
7. Embed: `rms_noweight(h)`, then `mm.input_projection` 1152→5376.

**Text side** (`llama.cpp src/models/gemma4.cpp`, `llama-kv-cache.cpp`):
- Image rows take the encoder output **unscaled**; only token embeddings are
  multiplied by √5376. The first-layer `attn_norm` applies to both.
- Positions stay 1D and consecutive. There is no M-RoPE.
- The 31B uses `LLAMA_NON_CAUSAL_TYPE_SWA_ONLY` for image chunks (E2B/E4B are
  always causal). Every row of one image may attend to every key of that same
  image in **sliding** layers; the ordinary window still masks older keys.
  Global layers stay causal. mtmd decodes each image as one non-causal batch,
  so an image never splits across forwards. transformers agrees:
  `modeling_gemma4.py` builds sliding masks as `window AND (causal OR
  same-image block)` over image soft tokens only, and global masks as causal.

**Prompt:**
- The chat template emits `<|image|>` for each image part, in content order.
- The processor replaces it with `<|image>` (255999), N × `<|image|>` (258880)
  and `<image|>` (258882). No newlines are added.
- llama.cpp wraps the embeddings with the same `<|image>`/`<image|>` pair.

**Resize:**
- HF (`get_aspect_ratio_preserving_size`): scale by `sqrt(max_patches·256/(H·W))`,
  where `max_patches = budget·9`, then floor each side to a multiple of 48.
  A side that floors to 0 becomes 48 and the other side is capped. The image is
  then resized with antialiased bicubic.
- The processor default is 280 soft tokens (valid budgets: 70, 140, 280, 560,
  1120). llama.cpp instead clamps to [70, 1120] tokens.

## Decisions

- **Budget:** follow the HF processor, filling the budget up or down.
  `--image-tokens` (serve, prompt, chat) selects one of the checkpoint's
  budgets (70, 140, 280, 560, 1120; default 280, the processor's). The budget
  is part of every image identity, so cached prefixes never mix budgets.
- **Resampler:** move Qwen's PIL-exact fixed-point bicubic resampler to
  `src/core/image` so both models share it. Qwen output stays bit-identical.
- **Code placement:** the Gemma encoder lives in `src/models/gemma4/vision/`. It
  shares the BF16 hipBLASLt GEMM and image decoding, but not the Qwen ViT: that
  graph differs in norms, positions, attention and the merger.
- **Activation:** tanh GELU, as transformers configures the vision MLP
  (`gelu_pytorch_tanh`). llama.cpp's converter writes no `clip.use_gelu` for
  `gemma4v`, so stock llama.cpp runs GELU-quick. llama.cpp is compared through
  a copy of the sidecar with `clip.use_gelu = true`.
- **Oracle:** the CPU FP32 encoder reference and llama.cpp. llama.cpp is fed
  images already resized to 48-multiples within its token range, so both see
  identical pixels. Text quality is checked with teacher-forced logits on image
  prompts.

## Milestones

**V1 — Prompt and preprocessing (CPU, hosted-CI testable).**
- Shared resampler.
- Gemma resize sizing.
- Template image parts.
- Token expansion with boi/eoi.
- Per-image prefix identities (SHA-256 of pixels, grid, offset and encoder
  identity), matching Qwen's `cache_prefixes` contract.
- *Exit:* template goldens with image parts match jinja2. Sizing matches the HF
  formula on edge cases (tiny, extreme aspect, exact fit).

**V2 — Encoder.**
- CPU FP32 reference, then the HIP BF16 encoder with stage observers.
- *Exit:* the GPU encoder matches the reference at each stage within declared
  tolerances. Final embeddings match llama.cpp's `gemma4v` output for the same
  pixels.

**V3 — Engine.**
- `Executor::Forward` takes optional image rows (unscaled embeddings) and image
  spans. Sliding attention extends each span's key limit to its end.
- Prefill chunks never split a span.
- `Session::Sync` limits prefix reuse to the longest prefix whose images also
  match.
- Snapshots record span identities.
- *Exit:* text-only outputs and snapshots are bitwise unchanged. Image prompts
  give top-1/KL against teacher-forced llama.cpp logits within the text-model
  limits.

**V4 — Serving and CLI.**
- `--mmproj` for Gemma in `gufo serve llm` and `gufo prompt --image`.
- `Gemma4TextRunner::PreparePrompt`/`SetPromptContext`.
- RAM/disk cache identities.
- *Exit:* OpenAI image_url requests work, including a multi-turn replay that
  reuses the image prefix. A changed image never reuses cached KV.

**V5 — Performance and docs.**
- Encoder and image-prefill timings against llama.cpp.
- README/QUALITY/BENCHMARKS rows, plus EXPERIMENTS entries for kernel work.

## Progress

**2026-09-28 — Plan.** Merged `origin/main` (JSON Schema output) into
`feat/gemma`. Structured output stays Qwen-only: Gemma does not build a
constraint vocabulary. Downloaded and verified the BF16 sidecar.

**2026-09-28 — V1 and V2.**
- Prompt builder:
  - The template renders `<|image|>` per image part. jinja2 goldens cover image-only, interleaved and multi-turn cases.
  - Sizing reproduces transformers on 12 edge cases.
  - Prompts expand to `<|image>` + N×258880 + `<image|>` with per-image prefix identities.
- Encoder:
  - `vision::Reference` is the scalar FP64-accumulating oracle. `vision::Encoder` is the HIP version: FP32 patch GEMM, BF16 hipBLASLt projections, FP32 residual, tiled GEMM attention.
- Measured on a 144×96 synthetic image (6 rows):

  | Comparison | Embedding rel RMS |
  |---|---:|
  | Reference vs llama.cpp (`use_gelu`) | 1.4% |
  | Reference vs stock llama.cpp (GELU-quick) | 7.1% |
  | Reference vs GPU | 2.1% |

  - The GPU patch stage matches to 5e-7. Layer errors stay near 3e-4 until
    layer 20 and then grow through the last layers, as with BF16 activations.
- Measured on a 624×960 photo (260 rows, 2,340 patches):

  | Comparison | Rel RMS | Worst row cosine |
  |---|---:|---:|
  | GPU vs reference | 1.6% | — |
  | llama.cpp (`use_gelu`) vs reference | 5.9% | 0.81 |

  - GPU encode takes 402 ms (unoptimized).
- Tools:
  - `tools/gemma4/llama_vision.cpp`: llama.cpp embeddings and stage dumps; teacher-forced image-prompt logits.
  - `tests/models/gemma4/vision_probe.cpp`: prepare, reference and GPU comparison.
  - `tools/gemma4/compare_stages.py`.
- `gemma4.vision_encoder` gates the GPU encoder against the reference
  (`GUFO_GEMMA4_MMPROJ`).

**2026-09-28 — V3 engine.**
- `Executor::Forward` takes `ImageRows`: unscaled embeddings, plus per-row
  key ends so sliding layers see the whole image (`AttentionArgs::key_ends`,
  in the WMMA, single-pass and split kernels).
- `Session::Sync(prompt, images, embed)`:
  - Stops reuse at the first changed image and never resumes inside one.
  - Asks for embeddings only for images it evaluates.
  - Ends prefill chunks before an image that does not fit.
- Snapshots record image spans (payload v3).
- Teacher-forced on the 624×960 photo prompt (greedy 80-token description, 96 text positions from `<image|>` on):

  | Comparison | Mean KL | Max KL | Top-1 |
  |---|---:|---:|---:|
  | Gufo vs llama.cpp (`use_gelu` sidecar) | 0.0151 | 0.34 | 96/96 |
  | Gufo vs stock llama.cpp (GELU-quick) | 0.475 | 43.8 | 92/96 |

  - The 16K text-only figure is 0.0092.
  - The two llama.cpp activations differ from each other by 0.21.
- Gufo's description is correct: the "Test Images" grid plus the "Grayscales" and "Primary Colors" charts.
- Timings: 598 ms to encode, 872 ms to prefill 281 tokens.
- `gemma4.vision_session` (model-backed) checks the reuse and chunking rules:
  - A red and a blue image behind identical tokens answer differently (KL 32).
  - A changed image, rewind, extension, snapshot restore and a prefill chunk
    ending inside an image all match fresh sessions inside the prefill
    envelope.
- `gemma4.attention_ops` covers image rows across the window, ring wrap and the split path.

**2026-09-28 — V4 serving and CLI.**
- `gufo serve llm` and `gufo prompt`/`chat` load `mmproj-BF16.gguf` beside the model, or the one passed with `--mmproj`.
- `Gemma4TextRunner`:
  - Prepares image prompts with per-image cache prefixes.
  - Encodes each image at most once per request state.
  - Never ends a prefill step inside an image.
- Measured over HTTP (OpenAI `image_url` data URLs):
  - The 624×960 chart photo: first turn 285 prompt tokens in 5.5 s (cold, including sidecar upload). The follow-up reuses 278 cached tokens (1.35 s) and answers "24 photographs" correctly.
  - The Gufo logo: "a pixel art illustration of an owl", brown with orange eyes.

**2026-09-28 — V5 performance, budgets and benchmarks.**
- Fused WMMA attention, then one softmax update per 64 keys: encode 402 → 164 ms at 260 soft tokens and 1,025 ms at 1,107. Both steps also brought the encoder closer to the reference (EXPERIMENTS.md).
- `--image-tokens` selects Google's budgets (70–1120) on serve, prompt and chat.
- Against llama.cpp b11069 over HTTP:
  - Cold image requests: 1.4–2.0× faster time to first token.
  - Follow-up turns: 1.7–2.2× faster.
  - Decode: 13–36% faster.
  - Details in BENCHMARKS.md#image-requests.
- llama.cpp aborts on 1,120-token images unless its ubatch holds the whole image.


**2026-09-29 — binary16 encoder GEMMs and the 26B-A4B.**
- The 26B-A4B's sidecar (`unsloth/gemma-4-26B-A4B-it-GGUF` `mmproj-BF16.gguf`,
  SHA-256 `41926ed5f1403cf5add23b0684992805ea6f97253096132e769e65646b8cef9d`,
  1,194,828,256 bytes) has the same tower and processor as the 31B's
  (`config.json` `vision_config` and `processor_config.json` are identical);
  only `projection_dim` differs (2816, the 26B's width). The encoder takes the
  width from the target, and image rows reach the expert layers like token
  rows.
- `vision::Reference`-level checks there showed BF16 GEMM inputs costing more
  than on the 31B: 1.16% embedding error and mean KL 0.112 from the scalar
  text model on the reference's embeddings (`Reference` now takes image rows).
  The encoder's projections now take binary16 inputs through Flash-Next's
  binary16 WMMA GEMM (weights converted on upload): 0.27% and KL 0.029, the
  same as the GPU text model fed the reference's embeddings (0.028). On the
  31B the embedding error drops 1.6% → 0.27% (260 rows) and 2.6% → 0.41%
  (1,107 rows); encode time is unchanged at 260 rows and 13% longer at 1,107
  (EXPERIMENTS.md).
- llama.cpp is not an oracle for the 26B: its Q8_1 activations already put its
  text logits at KL 0.42 from the reference (26B-A4B QUALITY.md). On the chart
  prompt its `use_gelu` run sits at 0.251 from the reference, Gufo's at 0.029.
