# DeepSeek V4 Flash model implementation

This directory contains Gufo's model-private DeepSeek V4 Flash 0731 ROCm
runtime for Linux x86-64 on AMD Strix Halo (`gfx1151`). The imported numerical
implementation and update policy are recorded in [UPSTREAM.md](UPSTREAM.md).

## Chat template and reasoning

Gufo uses the compiled `deepseek-v4-flash-0731-compiled-v2` formatter derived
from the official 0731 encoder. The official encoder provenance and deployed
artifact Jinja are stored under [`reference/`](reference/). Model loading
rejects missing or unrecognized `tokenizer.chat_template` hashes. It does not
execute Jinja or another template language from the model artifact.

Chat Completions accepts top-level `reasoning_effort` and these
`chat_template_kwargs`: `enable_thinking`, `reasoning_effort`,
`preserve_thinking`, and the DeepSeek alias `thinking_mode` (`thinking` or
`chat`). Pi's native DeepSeek shape, `thinking: {"type": "enabled"}` or
`thinking: {"type": "disabled"}`, is also accepted.

Thinking defaults to off in the server and CLI. DeepSeek's native effort set is
`low`, `high`, and `max`; Pi's levels map as follows:

| Requested | DeepSeek level |
|---|---|
| `minimal`, `low` | `low` |
| `medium`, `high` | `high` |
| `xhigh`, `max` | `max` |

The native `low` level adds no instruction. `high` and `max` prepend the pinned
0731 effort instructions. Thinking generation begins after
`<｜Assistant｜><think>`; chat mode begins after
`<｜Assistant｜></think>`.

In thinking mode, historical reasoning is dropped by default while visible
assistant content is retained. `preserve_thinking=true` replays it, and tool
conversations force preservation. In chat mode, the standard encoder never
replays reasoning: assistant history is reconstructed as
`<｜Assistant｜></think>{content}`. Consecutive tool results and user follow-ups
are merged into one user turn, and tool calls use the official
`<｜DSML｜...>` markers.

The HTTP adapter returns generated thought bytes as `reasoning_content` and the
final answer as `content` in both streaming and non-streaming responses.

The CLI controls are `--think`, `--reasoning-effort`, and
`--preserve-thinking`. There is no custom Jinja override and no
reasoning-token-budget enforcement.

Pi should configure this model with `thinkingFormat: "deepseek"`. Both
thinking-on and thinking-off prompts are deterministic, so identical repeated
requests can reuse continuation-cache snapshots independently for each mode.

Rendered-byte and token-ID SHA goldens generated with the pinned official
encoder/tokenizer are stored in
[`tests/fixtures/chat_template_hf_goldens.json`](../../../tests/fixtures/chat_template_hf_goldens.json).
The template tests cover chat mode, all native effort levels, history
drop/preserve, strict artifact validation, and DSML tool loops. The
model-backed `ds4.target` check compares complete token-ID sequences with these
goldens while reusing the model loaded for target quality checks. All DS4 checks
are registered in `tests/models/deepseek_v4_flash/CMakeLists.txt`; see the
[quality workflow](../../../benchmarks/deepseek-v4-flash/README.md).

## Runtime boundary

`engine.*` owns model loading, tokenization, sessions, snapshots, logits, and
cancellation. `chat_template.*` owns deterministic chat rendering. `runtime/`
and `kernels/rocm/` remain model-private and do not share Qwen tensor layouts
or mutable state.
