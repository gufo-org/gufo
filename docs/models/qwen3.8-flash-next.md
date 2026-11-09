# Qwen3.8 Flash-Next

Qwen3.8 Flash-Next is a mixture-of-experts model that reads text and images and
writes text. Gufo runs its Q4 GGUF, optionally sped up with its own
multi-token prediction (MTP) head.

## Files

| File | Needed for |
| --- | --- |
| `UD-Q4_K_XL/*` (four shards) | Text |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | MTP |
| `mmproj-BF16.gguf` | Images |

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision d96201194f103c906994f283f504209dc8d2a10c \
  --include "UD-Q4_K_XL/*" \
  --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --include "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
```

## Run

Pass the first shard; Gufo finds the others.

```sh
MODEL=/path/to/first-shard.gguf
MTP=models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
gufo chat --model "$MODEL"
gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

## Decoding modes

| Mode | Options | What happens |
| --- | --- | --- |
| AR | none | The model produces one token per step. |
| MTP | `--speculative mtp --mtp-model PATH` | The MTP head proposes up to seven tokens; the model checks them in one pass and keeps the ones it agrees with. |

MTP adapts how many tokens it proposes, up to seven. When
proposals keep failing, a request falls back to AR for a while.

MTP also proposes copies of earlier text from the same conversation, which
speeds up answers that repeat code, tool output or a previous reply. A request
decoding alone can check up to 64 copied tokens in one pass. The `timings`
object in a response reports copies as `lookup_n` and `lookup_n_accepted`.

With greedy decoding, MTP produces exactly the AR output. With sampling, both
modes follow the same distribution, but the same seed gives different text.

## Thinking

Thinking is on by default, with `xhigh` effort, and earlier reasoning is kept
in the conversation. Change this per request with the
[reasoning controls](../server.md#reasoning-controls).

## Long context

The model's native context is 262,144 tokens. To go further, start the server
with `--enable-yarn` and a larger `--context`; without the flag, Gufo refuses a
`--context` above 262,144. The flag is an error at or below 262,144, where it
would change nothing.

**YaRN** stretches the model's position encoding to cover more tokens. Requests
that stay within 262,144 tokens run unchanged; longer ones use the smallest
whole stretch factor that fits, up to context / 262,144 (1.5625 for
`--context 409600`). Larger factors change the output more. As of October 2026,
multiple-choice accuracy stayed within noise from factor 1.5625 to 3.875.

Memory grows with the context in use: about 27.5 KiB per token with MTP. Gufo
reserves the configured capacity when it starts.

## Images

Gufo finds `mmproj-BF16.gguf` next to the target; pass `--mmproj PATH` to use
another location. Images work the same way as for
[Qwen3.8 27B](qwen3.8-27b.md#images), with AR and MTP.

## Performance

As of October 18, 2026, on Strix Halo:

| Workload | Speed |
| --- | --- |
| Prefill, AR | 1,801.01 tokens/s |
| Decode, one user, MTP | up to 116.93 tokens/s |
| Decode, 8 concurrent users, MTP | 245.81 tokens/s in total |

The [full benchmarks](../internal/models/qwen3.8-flash-next/BENCHMARKS.md)
compare with llama.cpp. The
[quality report](../internal/models/qwen3.8-flash-next/QUALITY.md) covers
accuracy against the reference and between modes.
