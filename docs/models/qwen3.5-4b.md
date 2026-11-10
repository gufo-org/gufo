# Qwen3.5 4B

Qwen3.5 4B is a small text model. Gufo runs its Q4 GGUF file with the same
hybrid (Gated DeltaNet + full attention) runtime as Qwen3.8 27B, so no extra
extension or draft model is needed.

## Files

| File | Size | Needed for |
| --- | --- | --- |
| `Qwen3.5-4B-Q4_K_M.gguf` | 2.6 GiB | Text |

```sh
hf download unsloth/Qwen3.5-4B-GGUF \
  Qwen3.5-4B-Q4_K_M.gguf \
  --revision e87f176479d0855a907a41277aca2f8ee7a09523 \
  --local-dir models/qwen3.5-4b
```

## Run

```sh
MODEL=models/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf
gufo chat --model "$MODEL"
gufo serve llm --model "$MODEL" --sessions 2 --context 32768
```

Gufo recognizes the file by its `qwen35` metadata and serves it under the
`/v1/models` ID `qwen35-4b`. The artifact declares a 262,144-token context;
`serve` and `bench` accept any `--context` up to that length.

## Modes and limits

| Capability | Status |
| --- | --- |
| Chat, prompt, serve, bench | Yes, text only. |
| Images | No. This artifact has no vision sidecar, and `--image`/`--mmproj` are rejected. |
| Speculative decoding | No. The 4B file carries no native MTP layer, and no draft model is published for it. |

Sampling defaults follow the general text preset; the thinking and reasoning
controls work as described in the [server guide](../server.md#reasoning-controls).

## Implementation notes

The 4B shape (32 layers, 2560 hidden size, 9216 intermediate size, SSM time-
step rank 32) shares the Qwen HIP kernels with the 27B target. Its 16-query-
head attention (vs. the 27B's 24) gets its own WMMA and tiled attention
variants (`attention_wmma_16.hip`, `attention_tile_16.hip`), so long-context
prefill uses the same flash-attention-style path as the 27B instead of the
baseline O(n²) kernel. Measured prefill: ~3140 tok/s at 12k context on Strix
Halo (release build).
