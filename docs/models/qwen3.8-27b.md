# Qwen3.8 27B

Qwen3.8 27B is a dense model that reads text and images and writes text. Gufo
runs its Q4 and Q8 GGUF files, optionally sped up with the DFlash2 draft model.

## Files

| File | Size | Needed for |
| --- | --- | --- |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 16.35 GiB | Text, Q4 target |
| `Qwen3.8-27B-UD-Q8_K_XL.gguf` | 29.30 GiB | Text, Q8 target |
| `mmproj-BF16.gguf` | | Images |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | | DFlash2, with either target |

Pick one target. Each request also needs memory for its context, and the draft
and image projector add their own weights.

```sh
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --local-dir models/qwen3.8-27b
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --revision 2d9571f8ce46e151f61c6499c99dee6079e1d610 \
  --local-dir models/qwen3.8-27b
```

## Run

```sh
MODEL=models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
DRAFT=models/qwen3.8-27b/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
gufo chat --model "$MODEL"
gufo serve llm --model "$MODEL" --speculative dflash2 \
  --dflash-model "$DRAFT" --sessions 2 --context 32768
```

## Decoding modes

| Mode | Options | What happens |
| --- | --- | --- |
| AR | none | The target produces one token per step. |
| DFlash2 | `--speculative dflash2 --dflash-model PATH` | The draft proposes several tokens; the target checks them in one pass and keeps the ones it agrees with. |

DFlash2 adapts how many tokens it proposes. `--draft-tokens` caps the number,
and `--draft-policy fixed` always proposes that many, for comparisons.

When a request decodes alone, Gufo also proposes copies of earlier text from
the same conversation and checks up to 64 of them in one pass. Copies help
most when the answer repeats code or text the model has already seen.

With greedy decoding, DFlash2 produces exactly the AR output. With sampling,
both modes follow the same distribution, but the same seed gives different
text. Thinking and sampling defaults are set per request; see
[reasoning controls](../server.md#reasoning-controls).

## Images

Gufo finds `mmproj-BF16.gguf` next to the target; pass `--mmproj PATH` to use
another location. Repeat `--image` to attach several images:

```sh
gufo prompt --model "$MODEL" --image photo.jpg \
  --prompt "Describe this image." --temperature 0
```

Over HTTP, send images as `image_url` parts of a chat message; the
[server guide](../server.md#images) lists the accepted formats and limits.
Images count toward the context. `detail: low` scales an image to fit 512 × 512
pixels and uses at most 256 image tokens; `auto` (the default), `high` and
`original` allow 64 to 16,384.

To number images as `Picture 1:`, `Picture 2:` and so on, start the server with
`--add-vision-id` or send `chat_template_kwargs: {"add_vision_id": true}`.

## Performance

As of October 19, 2026, on Strix Halo with the Q4 target:

| Workload | Speed |
| --- | --- |
| Prefill, AR | 648.89 tokens/s |
| Decode, one user, DFlash2 | up to 71.55 tokens/s |
| Decode, 8 concurrent users, DFlash2 | 128.42 tokens/s in total |

The [full benchmarks](../internal/models/qwen3.8-27b/BENCHMARKS.md) compare
with llama.cpp at every depth and include Q8. The
[quality report](../internal/models/qwen3.8-27b/QUALITY.md) covers text and
vision accuracy.
