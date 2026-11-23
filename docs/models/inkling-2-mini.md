# Inkling 2 Mini

Inkling 2 Mini is Thinking Machines' 26B mixture-of-experts model that reads
text and images and writes text. Gufo runs Unsloth's UD-Q4_K_XL GGUF, which
keeps the checkpoint's native NVFP4 experts.

## Files

| File | Needed for |
| --- | --- |
| `Inkling-2-Mini-UD-Q4_K_XL.gguf` | Text |
| `mmproj-BF16.gguf` | Images |

```sh
gufo model install inkling-2-mini
```

## Run

```sh
gufo serve llm --model inkling-2-mini
```

Pass `--mmproj PATH` to accept images. Requests name the model
`inkling-2-mini`. Without `--context`, a session gets the model's native
context of 1,048,576 tokens; state for the eight full-attention layer visits
takes 32 KiB per token and is mapped only as a conversation grows.

## Thinking

Thinking is on by default. `reasoning_effort` and
`chat_template_kwargs.enable_thinking` follow the
[reasoning controls](../server.md#reasoning-controls). With thinking off, the
model sometimes writes its deliberation into the answer instead of a thinking
block.

## Performance

As of November 23, 2026, on Strix Halo, compared with llama.cpp's HIP backend on
the same weights:

| Workload | Gufo | llama.cpp HIP |
| --- | --- | --- |
| Prefill, 2,048-token prompt | 2,210 tokens/s | 383 tokens/s |
| Decode, one user, short context | 47.0 tokens/s | 24.3 tokens/s |
| Decode, 4 users, short context | 93.4 tokens/s in total | not measured |

The [benchmarks](../internal/models/inkling-2-mini/BENCHMARKS.md) give every
measurement, and the [quality report](../internal/models/inkling-2-mini/QUALITY.md)
compares the output with llama.cpp's CPU reference.
