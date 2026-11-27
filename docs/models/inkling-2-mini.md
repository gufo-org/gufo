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

| Effort | `none` | `minimal` | `low` | `medium` | `high` | `xhigh`, `max` |
| --- | --- | --- | --- | --- | --- | --- |
| Level | 0 | 0.1 | 0.2 | 0.7 | 0.9 | 0.99 |

Inkling can't switch thinking off. At level 0 (`none`, or
`enable_thinking: false`) the model still usually writes a short thought, 29 to
345 tokens on short prompts as of November 2026, as the model card says. Gufo
follows the template, as llama.cpp does: `none` lowers the effort, and the
reply may still carry `reasoning_content`.

`tool_choice: "required"`, named functions and `response_format` let the
model think first, then force the call or the JSON.
A required call gets one thought: a model that decided against the call would
otherwise keep reopening thoughts until the token limit.

The prompt matches Thinking Machines' renderer (`tml-renderers`). The model can
think again after part of its answer. Responses and Messages return one item or
block per thought and answer text, in order; Chat Completions joins the
thoughts in `reasoning_content`. Send them back in order to keep the
conversation cached: in Chat Completions, as `{"type": "thinking", "thinking":
"..."}` content parts between the text parts (other models read them joined).

## Tools and structured output

Tool arguments are generated with their keys in sorted order, the order the
template declares tools and re-renders calls in. The model writes them that way
(`{"action":…,"id":…,"status":…}`), so with the schema's own order it could
lose a property declared before one it had already written. Nested objects are
sorted too; the template re-renders those as they were written.

A grammar sees each token's own bytes, so it describes the thought as the model
writes it after the prompt's header: `<|content_thinking|>`, then
`<|end_message|><|message_model|><|content_text|>` before the answer. When the
model answers without a thought, the grammar can't let it open the answer
block, and it writes the JSON right after the header; Gufo streams that as the
answer.

## Performance

As of November 26, 2026, on Strix Halo with greedy decoding and thinking off,
compared with llama.cpp's HIP backend on the same weights:

| Workload | Gufo | llama.cpp HIP |
| --- | --- | --- |
| Prefill, 2,048-token prompt | 2,296 tokens/s | 347 tokens/s |
| Decode, one user | 47.4 tokens/s | 22.0 tokens/s |
| Decode, one user, 131,072 tokens of context | 24.3 tokens/s | 7.7 tokens/s |
| Decode, one user, DSpark, prose | 59.4 tokens/s | N/A |
| Decode, 8 users, in total | 189.7 tokens/s | 68.3 tokens/s |
| Cold start to ready, with DSpark | 3.9 s | N/A |

llama.cpp can't load the DSpark drafter yet. DSpark gains most on predictable
text: a prose summary whose prompt comes back from the cache decodes at 77
tokens/s.

The [benchmarks](../internal/models/inkling-2-mini/BENCHMARKS.md) give every
measurement, and the [quality report](../internal/models/inkling-2-mini/QUALITY.md)
compares the output with llama.cpp's CPU reference.
