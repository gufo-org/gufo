# DeepSeek V4 Flash

DeepSeek V4 Flash is a large mixture-of-experts text model. Gufo runs the
antirez Flash 0731 GGUF, a mix of 2-bit and 8-bit weights, optionally sped up
with the DSpark support model.

## Files

| File | Size | Needed for |
| --- | --- | --- |
| `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` | 80.76 GiB | Text |
| `DeepSeek-V4-Flash-DSpark-support-0731.gguf` | | DSpark |

The weights alone take 80.76 GiB, so you need a 128 GiB system with room left
for each request's context.

```sh
hf download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --revision 1cd7b564460821938add0475a60b942c409295e0 \
  --local-dir models/deepseek-v4-flash
hf download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --revision e7f04037032990db0346398d249baf9fb9df1ccc \
  --local-dir models/deepseek-v4-flash
```

## Run

```sh
MODEL=/path/to/target.gguf
gufo chat --model "$MODEL"
gufo serve llm --model "$MODEL" --context 32768 --sessions 1
```

## Decoding modes

| Mode | Options | What happens |
| --- | --- | --- |
| AR | none | The model produces one token per step. |
| DSpark | `--dspark-model PATH` | The support model proposes tokens; the target checks them and keeps only the ones it agrees with. |

How much DSpark helps depends on how many proposals the target accepts, which
varies with the text and the context length.

## Thinking

Thinking is on by default with `high` effort, as in the
[DeepSeek API](https://api-docs.deepseek.com/guides/thinking_mode). Use
`--think off` to turn it off, or `--reasoning-effort low` for shorter
reasoning. The [reasoning controls](../server.md#reasoning-controls) set both
per request.

## Limits

- Text only: images aren't supported.
- As of October 2026, Gufo's arithmetic hasn't been fully compared with the
  original model, so treat agreement between AR and DSpark as consistency, not
  as proof of quality. The
  [quality report](../internal/models/deepseek-v4-flash/QUALITY.md) lists what
  is checked.

## Performance

As of October 30, 2026, on Strix Halo:

| Workload | Speed |
| --- | --- |
| Prefill, AR | 474.87 tokens/s |
| Decode, one user, DSpark | up to 29.57 tokens/s |
| Decode, 8 concurrent users, AR | 77.12 tokens/s in total |

The [full benchmarks](../internal/models/deepseek-v4-flash/BENCHMARKS.md)
compare with antirez/ds4.
