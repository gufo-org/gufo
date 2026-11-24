# Inkling 2 Mini

Inkling 2 Mini is Thinking Machines' 26B mixture-of-experts model that reads
text and images and writes text. Gufo runs Unsloth's UD-Q4_K_XL GGUF, which
keeps the checkpoint's native NVFP4 experts.

## Files

| File | Needed for |
| --- | --- |
| `Inkling-2-Mini-UD-Q4_K_XL.gguf` | Text |
| `mmproj-BF16.gguf` | Images |
| `dspark-Inkling-2-Mini-Q8_0.gguf` | DSpark |

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

## Decoding modes

| Mode | Options | What happens |
| --- | --- | --- |
| AR | none | The model produces one token per step. |
| DSpark | `--dspark-model PATH` | The model's own drafter proposes up to seven tokens (fewer when it is unsure); one step checks them and keeps those the model agrees with. |

`gufo model install` fetches the DSpark file, so `--model inkling-2-mini`
decodes with DSpark. DSpark returns exactly the tokens AR returns, for greedy
and sampled requests alike; it only changes the speed. How much it helps
depends on how predictable the text is. Requests decoding at the same time
(`--sessions` above 1) draft together and are checked in one shared step, which
gives each request its share of the rows where its drafts are likely to be
kept; their output is the same as when each runs alone.

DSpark also proposes copies of earlier text from the same conversation, which
speeds up answers that repeat code, a file being edited or a quoted passage. A
request decoding alone checks up to 64 copied tokens in one step; the width
grows while whole copies are kept and drops back after a miss. The `timings`
object in a response reports copies as `lookup_n` and `lookup_n_accepted`.

## Sampling

The defaults are the upstream `generation_config.json`'s: temperature 1 and no
top-k, top-p or min-p truncation. Requests can override each value.

## Thinking

Thinking is on by default at effort `high`. `reasoning_effort` and
`chat_template_kwargs.enable_thinking` follow the
[reasoning controls](../server.md#reasoning-controls); each effort selects the
chat template's level:

| Effort | `none` (thinking off) | `minimal` | `low` | `medium` | `high` | `xhigh`, `max` |
| --- | --- | --- | --- | --- | --- | --- |
| Level | 0 | 0.1 | 0.2 | 0.7 | 0.9 | 0.99 |

The template itself can't switch thinking off: at level 0 the model still
writes a short thought, 29 to 345 tokens on short prompts as of November 2026.
With thinking off, Gufo therefore opens the answer (or, with tools, closes an
empty thought) in the prompt and doesn't let a thought start; the model then
sometimes writes its deliberation into the answer instead.

With thinking on, `tool_choice: "required"` and named functions let the model
think first, then force a call.

## Performance

As of November 23, 2026, on Strix Halo, compared with llama.cpp's HIP backend on
the same weights:

| Workload | Gufo | llama.cpp HIP |
| --- | --- | --- |
| Prefill, 2,048-token prompt | 2,268 tokens/s | 383 tokens/s |
| Decode, one user, short context | 47.0 tokens/s | 24.3 tokens/s |
| Decode, one user, DSpark, code (greedy / default sampling) | 78.0 / 65.7 tokens/s | not measured |
| Decode, one user, DSpark, prose | 48 to 92 tokens/s | not measured |
| Decode, one user, DSpark, rewriting a file (default sampling) | 90.1 tokens/s | not measured |
| Decode, 2 users, DSpark, mixed greedy and sampled | 73.9 tokens/s in total (AR 62.3) | not measured |
| Decode, 4 / 8 users, mixed greedy and sampled | 91.1 / 114.1 tokens/s in total | not measured |

The [benchmarks](../internal/models/inkling-2-mini/BENCHMARKS.md) give every
measurement, and the [quality report](../internal/models/inkling-2-mini/QUALITY.md)
compares the output with llama.cpp's CPU reference.
