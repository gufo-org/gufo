# OpenAI-Compatible Server

Status: implemented subset, 2026-09-25

## Purpose

The server exposes a focused OpenAI-compatible API while keeping inference,
scheduling, and device execution in the C++ process. Compatibility is a
versioned contract: supported fields behave as documented, and unsupported
fields return explicit errors.

Chat Completions is the main API, including streaming, images and tools.
Responses supports text, images and function tools with optional streaming.
Anthropic Messages exposes a synchronous text subset.
The reference protocols are:

- https://developers.openai.com/api/reference/resources/responses/methods/create/
- https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/
- https://developers.openai.com/api/reference/resources/models/methods/list/
- https://platform.claude.com/docs/en/api/handling-stop-reasons
- https://developers.openai.com/api/docs/guides/streaming-responses

The local server does not need to reproduce OpenAI-hosted storage, billing,
organization, or account behavior.

The only supported production platform is Linux x86-64 on Strix Halo.

## Implementation

The C++20 runtime uses bounded HTTP/1.1 requests, one connection worker per
request, and a separate inference scheduler. Connections close after each
response. Chat Completions and Responses support Server-Sent Events; socket closure
propagates cancellation to the scheduler. JSON parsing rejects duplicate keys,
invalid numbers and nesting beyond 128 containers.

Python may be used for development tools and API conformance tests, but it must
not be required to serve requests.

## Single Executable and Model Configuration

The deployed product is one `gufo` executable. Supported model graphs,
tokenizers and HIP kernels are compiled into it.

The executable provides subcommands rather than separate inference binaries:

```text
gufo serve
gufo chat
gufo prompt
gufo video
gufo transcribe
```

`chat` and `prompt` are transport adapters over the same scheduler and may
either load the runtime directly or connect to a running server. `video`
executes the direct MiniMax H3 route used by the asynchronous video worker.
`transcribe` executes the native Qwen3-ASR-1.7B route used by the synchronous
audio transcription endpoint.
Their detailed contracts are defined in [Command-Line Interface](CLI.md) and
[MiniMax H3 upstream contract](models/minimax-h3/QUALITY.md).

### HIP execution

Qwen HTTP requests share one immutable `QwenGpuModel` containing the mapped
weights and tokenizer. Each request leases a preallocated `QwenGpuExecutor`
with independent KV, recurrent, graph, activation, and logit state.
`--sessions N` controls the bounded execution session pool, not the number of
remembered conversations. Checkpoints have separate entry and memory limits;
see [KV cache](KV-CACHE.md#limits). The scheduler batches ready
requests when the model runner supports their execution mode.

Text serving defaults match llama.cpp for context and generation length:

| Setting | Default |
| --- | --- |
| `--context` | `0`: native context from model metadata, per session |
| `--max-tokens` | `-1`: until EOS or remaining context is exhausted |
| `--sessions` | `1` |
| Thinking / reasoning effort | Enabled; Qwen `xhigh`, DeepSeek `high` |

Qwen chat prompts are bounded by the session context, not a fixed size: the
rendered template may use up to 128 bytes per context token (at least 1 MiB).

Clients can set a positive `max_tokens` / `max_completion_tokens` (Chat
Completions) or `max_output_tokens` (Responses). These include reasoning tokens.
Omitting the field uses the server default. A response cannot exceed remaining
context; exhaustion reports `length` or `incomplete`, without discarding earlier
conversation tokens. Reduce `--context` or `--sessions` if their state exceeds
available memory.

Text sampling follows each model's recommended defaults in `serve`, `prompt`
and `chat`. Explicit request values override explicit server options, which
otherwise inherit the effective thinking preset. Null request values inherit
where supported; explicit zero values are preserved (`top_p` must remain positive).
The startup sampling log reports server defaults; requests can override each
setting independently, including when changing thinking mode.

| Model / mode | Temperature | Top-p | Top-k | Presence penalty |
| --- | --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 (agentic) | 1.0 | 0.95 | 0 | 0 |
| Qwen3.8 27B / Flash-Next, thinking | 1.0 | 0.95 | 20 | 0 |
| Qwen3.8 27B / Flash-Next, thinking off | 0.7 | 0.8 | 20 | 1.5 |

Min-p and frequency penalty default to zero; repetition penalty is 1.0.
AR and DFlash2/MTP/DSpark use the same target defaults.
Use `--temperature 0` or request `"temperature": 0` for greedy
output; `bench` remains greedy by default. Audio and image generation retain
their own settings.
Use `--think off` or request `"reasoning_effort": "none"` to disable thinking.
DeepSeek maps `minimal`/`low` to `low`, `medium`/`high`/`xhigh` to `high`,
and `max` to `max`.
The [functional suite](../tests/functional/README.md) with
`--suite sampling-defaults --sampling-preset qwen38` (or
`deepseek4`) compares omitted and explicit settings, including C2 replay.

Sources: [Qwen27B](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices),
[Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices),
[DeepSeek 0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md),
[DeepSeek thinking defaults](https://api-docs.deepseek.com/guides/thinking_mode).
DeepSeek's 0.95 top-p is its agentic recommendation; neutral penalties and
disabled unspecified filters are Gufo defaults.

```sh
nix build

./result/bin/gufo serve \
  --host 127.0.0.1 \
  --port 8080 \
  --sessions 1 \
  llm \
  --model models/qwen.gguf \
  --context 4096
```

Cancellation preserves successfully executed state for continuation; failed
model operations invalidate mutable state. Model replacement is transactional,
so active requests retain their original model until their lease ends.
Generation responses include:

```text
Server-Timing: ttft;dur=<milliseconds>, inter_token;dur=<milliseconds>
```

When `stream_options.include_usage` is enabled, the terminal Chat Completions
usage chunk also includes a namespaced `usage.gufo` object. It reports
privacy-safe scheduler and stage metrics used by
`tools/serving/gufo-serving-bench.py`: queue, prefill, decode, TTFT and ITL timing;
actual prefill work; cache use; logical concurrency; physical execution width;
and the executed plan. Prompts, generated text, local paths, request IDs, and
token IDs are excluded.

DeepSeek V4 Flash batches up to eight active requests. Each request owns its
attention caches, compressor state, position, and sampling state. Dense and
expert projections share weight reads across the batch; one request uses the
single-session path.

With `--dspark-model <support.gguf>`, DSpark is selected automatically unless
`--speculative off` is explicit. Greedy and sampled requests keep independent
samplers, cache state and draft policies. C1 uses point-mass proposals and
retains seeded AR token identity; filtered sampled C>1 can use compact
probabilistic proposals with exact p/q acceptance and residual correction.
That route preserves the target distribution with its own seeded trace.
Request decoding leaves EOS and later speculative tokens out of the reusable
checkpoint. Fixed-length benchmarks may continue through EOS.
See the [DS4 benchmark and quality contract](models/deepseek-v4-flash/BENCHMARKS.md).

Qwen3.8-Flash-Next uses the same scheduler, with independent recurrent, KV,
indexer, sampling and MTP state per session. Prefix snapshots retain complete
state; `--cache-disk` adds restart-safe reuse. MTP draft length adapts within the
configured ceiling. Sampled MTP proposals use p/q acceptance and residual
correction; greedy verification follows target argmax. Draft and verification
work can batch across ready requests. See the
[Flash-Next benchmark and quality contract](models/qwen3.8-flash-next/BENCHMARKS.md).

Each model chooses its prefill chunk. `--prefill-chunk` limits prompt work
between active decode rounds without changing a lone request's kernel policy.

`--cache-ram-bytes 0` (the default) selects an automatic snapshot budget capped
at 32 GiB and half the available host RAM after model/state allocation, respecting
container limits. 27B also checks HIP free memory. A positive value sets a byte
cap that may exceed the automatic budget, up to the available host RAM minus
4 GiB; the startup line reports both as `automatic_bytes` and `max_bytes`.
Disk staging and temporary disk-save buffers are separate from this RAM budget.
The 128 checkpoint records are independent of `--sessions`; more than one can
belong to a conversation.
Payloads are allocated only when captured. Under pressure, optional copies
give way before another conversation's last useful checkpoint.

Prompt reuse is enabled by default; how the cache finds, retains and
evicts that state is described in [the KV cache](KV-CACHE.md).
`cache_prompt: false` on
`/v1/chat/completions` or `/v1/responses` bypasses memory and disk lookup for
that request; the result can still populate the cache. DeepSeek and Qwen tool
requests retain a checkpoint before the assistant-generation suffix, including
when a client drops the interrupted assistant and appends `"."` after a tool result. DeepSeek
also accounts for tokenization changes where adjacent user/tool turns join.
Qwen requests retain this checkpoint with thinking enabled or disabled.
Warm continuations preserve the reused frontier before prefill and save the
new stable boundary too. A second full-prompt checkpoint enables exact retries
without prefill; all copies share the same snapshot-memory budget.
Exact live continuations reuse generated tokens. The server reports cached
and newly processed tokens separately; resuming from the checkpoint processes
the short suffix. System instructions, tool definitions and image identities
must match the retained prefix.

Qwen image identity is checked only for images consumed before each checkpoint.
Appending an image reuses the preceding text/image state in RAM or on disk;
changing, removing or moving an earlier image invalidates checkpoints after it.
New images get a checkpoint before assistant framing so later turns do not
encode or prefill them again.
With `preserve_thinking=false`, a new user turn removes reasoning from the
preceding tool cycle. Gufo retains the state before that cycle and processes
its changed suffix again.

`SIGINT` and `SIGTERM` cancel active requests and drain accepted disk writes
before exiting. `--cache-disk DIR` defaults to 8 GiB retained on disk.
`--cache-disk-staging-bytes 0` (the default) selects the smallest of 1 GiB,
one eighth of available host RAM after model/session loading (including cgroup
limits), and the disk budget. This bounds queued captures/writes and each disk
read separately; it allocates nothing upfront. Live model state and retained
RAM snapshots have separate budgets.

Snapshots that exceed either limit are skipped with their required size and
available budget logged; live conversation reuse remains available. Existing
files that exceed the current staging limit are preserved subject to disk LRU
eviction and can be reused after restarting with sufficient staging.
For Flash-Next/MTP at full 262K context, explicitly set
`--cache-disk-staging-bytes 8589934592` (8 GiB) if RAM permits.
Qwen27B at full context needs larger staging and `--cache-disk-bytes` limits;
use the required size reported in the skip log.

For a focused cancellation check, run
`python3 tests/functional/continuation.py --output /tmp/cache-check.json`
against a private server named `cache-test` on port 5815.
It checks interruption during reasoning and visible output, with and without
reasoning replay, greedy/seeded sampling, and explicit cache bypass. Use
`--tools --discard-assistant` to exercise interrupted agent tool turns; add
`--prefix-repetitions 5500` for a roughly 50K-token prefix.
For persistence, enable `--cache-disk` before the check, restart the same server,
and add `--restore /tmp/cache-check.json`.
Use `--image /path/to/image.png` for Qwen image conversations.
Add `--append-image` to introduce the image after a cached text turn, and
`--reasoning-effort high` to check a specific thinking effort.
Each case continues for a third turn; repeat `--case NAME` to select only the
cases needed for a change.
The check requires exact snapshot and matched-history replay. It separately
reports equality to a fresh full prefill, whose different matrix shapes and
prefill/decode history can change rounding; that comparison is not silently
counted as an exact cache replay.

### Hardware compute queues

The gfx1151 command processor keeps eight compute queues mapped at once,
counted across every process on the device. Past that total the firmware
scheduler time-slices them even when all of them are empty: the GPU then
reports a busy engine at its top shader clock and draws about 26 W above idle
for as long as the processes live. Queues are claimed on a process's first HIP
dispatch and never released — neither `hipStreamDestroy` nor `hipDeviceReset`
gives one back — so the count is fixed at startup, and the first dispatch costs
two queues whatever the configuration. Four HIP processes is therefore the
ceiling on one device, gufo or otherwise.

Each server reads the queues already in use from
`/sys/class/kfd/kfd/proc/*/queues/*/type`, which is world-readable and so
includes processes gufo does not own, then exports `GPU_MAX_HW_QUEUES` before
loading a model and logs a `queue_budget` event. That event is INFO-tier and
follows `--log-level` like the rest of the startup diagnostics; the
`queue_budget_exceeded` warning below is WARN-tier:

| Server | `GPU_MAX_HW_QUEUES` | Resident compute queues |
| --- | ---: | ---: |
| `serve llm` | 2 | 3 |
| `serve tts` | 1 | 2 |
| `serve asr` | 1 | 2 |
| `serve image`, `serve video` | runtime default | up to 5 |

Text and audio caps are measured to cost no throughput: `serve llm` is flat from
the runtime default down to a single queue at one and at four concurrent
requests, and both audio servers are flat within run-to-run noise. Image and
video are unmeasured, so they keep the runtime default unless the device is too
busy to hold it.

The cap is only ever lowered to fit the free slots, never raised to fill them,
so a server's throughput does not depend on the order the servers started. A
`GPU_MAX_HW_QUEUES` set by the operator is always left alone, but it is still
checked: a cap of N resolves to at most N + 1 resident queues, so a value that
will not fit is reported even though it is honoured. When a server cannot fit,
it logs a `queue_budget_exceeded` warning and starts anyway.

### Reasoning controls

`--think auto` uses the model's default. Qwen27B and Flash-Next match the
official Jinja: thinking enabled, `xhigh` effort, prior reasoning preserved.
Use `--think off` or `chat_template_kwargs.enable_thinking=false` for direct
answers. DeepSeek defaults to thinking with `high` effort. Quality comparisons
must use the same reasoning mode and effort.

`POST /v1/chat/completions` accepts top-level `reasoning_effort` (`none`,
`minimal`, `low`, `medium`, `high`, `xhigh`, or `max`) and Pi/llama.cpp-style
`chat_template_kwargs`:

```json
{
  "reasoning_effort": "high",
  "chat_template_kwargs": {
    "enable_thinking": true,
    "reasoning_effort": "high",
    "preserve_thinking": false
  }
}
```

Pi's native DeepSeek request shape is also accepted:

```json
{
  "thinking": {"type": "enabled"},
  "reasoning_effort": "high"
}
```

`thinking.type` accepts `enabled` or `disabled`. DeepSeek also accepts
`thinking_mode` (`thinking`, `chat`, or `auto`) in `chat_template_kwargs`.
Conflicting controls return `invalid_reasoning`. In template kwargs,
`enable_thinking=false` suppresses the configured effort, matching Qwen's
Jinja; a non-off top-level `reasoning_effort` explicitly enables reasoning.
Generated reasoning is returned as `reasoning_content` in ordinary and
streaming Chat Completions responses. Per-model effort mappings and history
policies are documented in the model cards under `docs/models/`.

The server uses compiled model-specific formatters and validates recognized
artifact template hashes during model loading. It does not accept custom Jinja
or claim to enforce a reasoning-token budget.

Diagnostic telemetry is limited to status, token counts, timing, and
cancellation state. It must not contain prompt text, model paths, machine
identity, request IDs, or token IDs.

Pass the supported GGUF or safetensors directory through `--model`, with the
matching projector or speculative sidecar where needed. Loaders validate
tensor inventory, dimensions, quantization and tokenizer metadata. Model
artifacts supply weights and metadata; executable kernels and templates are
compiled into Gufo. See [model cards](models/README.md) for concrete paths
and supported modes.

## Process and State Ownership

HTTP workers submit requests through `TextGenerationBackend`. The text
scheduler owns request state and serializes model execution. Model runners own
weights, tokenization, prefill/decode, speculative verification and complete
cache snapshots. HTTP handlers do not implement model kernels.

## Endpoint Set

### Text endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | List loaded model aliases and capabilities |
| `POST` | `/v1/responses` | Text/images, structured output, optional SSE streaming |
| `POST` | `/v1/chat/completions` | Main chat, streaming, image and tool API |
| `POST` | `/v1/completions` | Optional legacy text completion adapter |
| `GET` | `/health` | Process liveness and GPU context (aliases: `/v1/health`, `/healthz`) |
| `GET` | `/ready` | Model and backend readiness (aliases: `/v1/ready`, `/readyz`) |
| `GET` | `/metrics` | Prometheus-format operational metrics (text LLM serving only) |

### Other model services

| Method | Path | Condition |
| --- | --- | --- |
| `POST` | `/v1/embeddings` | Not implemented (501) |
| `POST` | `/v1/audio/transcriptions` | An STT implementation is compiled and loaded |
| `POST` | `/v1/audio/speech` | A TTS implementation is compiled and loaded |
| `POST` | `/v1/images/generations` | Qwen-Image-2.1 image serving is configured |
| `POST` | `/v1/images/edits` | Qwen-Image-2.1 image serving is configured |
| `POST` | `/v1/videos` | A validated operator-supplied MiniMax H3 checkpoint is configured |
| `GET` | `/v1/videos/{id}` | MiniMax H3 video serving is configured |
| `GET` | `/v1/videos/{id}/content` | The requested MiniMax H3 job completed |
| `DELETE` | `/v1/videos/{id}` | MiniMax H3 video serving is configured |

Use `gufo serve image --model SNAPSHOT_DIR` for
[Qwen-Image-2.1](models/qwen-image-2.1/README.md). Generation accepts JSON;
editing accepts multipart PNG/JPEG references. Both return base64 PNG data
and work through llama-swap's image routes.

Qwen3-TTS serving is enabled with a dedicated model process. All three 12Hz
1.7B variants are supported; the variant is detected from the checkpoint's
`tts_model_type` and determines both the advertised model id and the request
fields that are required:

```sh
./result/bin/gufo serve --port 8080 tts \
  --model /persist/models/audio/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Start ASR separately with `gufo serve asr --model DIR`. Both speech
commands accept `--model`, `--context`, and `--served-model-name`, plus the
shared HTTP options. Context defaults are 4096 for TTS and 1024 per chunk
for ASR. Route separate model processes through llama-swap when one public
API URL should offer both synthesis and transcription.

| Variant | Model id | Voices | Additional required fields |
| --- | --- | --- | --- |
| CustomVoice | `qwen3-tts-12hz-1.7b-customvoice` | `talker_config.spk_id` names | `voice` |
| VoiceDesign | `qwen3-tts-12hz-1.7b-voice-design` | `voice-design` | `instructions` |
| Base | `qwen3-tts-12hz-1.7b-base` | `voice-clone` | `reference_audio`, plus `reference_text` unless `voice_clone_mode` is `speaker_embedding_only` |

`GET /v1/audio/voices` lists the advertised voices. CustomVoice exposes the
speaker names in `talker_config.spk_id`, while VoiceDesign and Base expose a
single placeholder name because their timbre comes from `instructions` or
`reference_audio` per request.

A Base checkpoint can additionally advertise operator-registered named voices,
so clients select a speaker by name instead of uploading a reference clip on
every request:

```sh
./result/bin/gufo serve tts \
  --model /persist/models/audio/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator_eng=/persist/models/audio/clear-english-voice.wav \
  --voice narrator_ita=/persist/models/audio/clear-italian-voice.wav
```

`--voice NAME=PATH` is repeatable. The reference transcript comes from
`--voice-text NAME=<transcript or path>`, which is also repeatable and may
appear before or after its matching `--voice`:

```sh
./result/bin/gufo serve tts \
  --model /persist/models/audio/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator_ita=/persist/models/audio/clear-italian-voice.wav \
  --voice-text "narrator_ita=Questo racconto e' cresciuto..." \
  --voice narrator_eng=/persist/models/audio/clear-english-voice.wav \
  --voice-text narrator_eng=/persist/models/audio/clear-english-voice.txt
```

A `--voice-text` value that names an existing file is read for its contents;
anything else is used as the transcript itself. Resolution order is
`--voice-text`, then a `.txt` sidecar beside the WAV
(`clear-english-voice.txt` for the example above), and with neither the preset
falls back to `speaker_embedding_only` cloning, which needs no transcript.

`--voice-lang NAME=LANGUAGE` binds a language to a voice. It supplies the
default when a request omits `language`, so a caller naming an Italian voice
does not have to repeat it -- without this the request would fall back to the
global `english` default and synthesize the voice in the wrong language. An
explicit request `language` still wins, so a voice can be driven in another
language deliberately.

A transcript that does not match its reference audio is worse than no
transcript: synthesis runs to the `max_new_tokens` cap, so a short input can
return several minutes of unusable audio. Preset names join
`voice-clone` in `/v1/audio/voices`, and a request naming a preset must not
also send `reference_audio`, `reference_text`, or `voice_clone_mode` --- the
preset already supplies them. Presets require a Base checkpoint; CustomVoice
selects a trained embedding and VoiceDesign is driven by `instructions`, so
neither has anything to apply them to.

`POST /v1/audio/speech` accepts `model`, `input`, `voice`, `response_format`,
`speed`, `language`, `instructions`, `seed`, `max_new_tokens`, `greedy`,
`stream_format`, talker `temperature`/`top_k`/`top_p`/`repetition_penalty`, and
predictor `subtalker_dosample`/`subtalker_temperature`/`subtalker_top_k`/`subtalker_top_p`.
Base additionally accepts `reference_audio`, `reference_text`, `voice_clone_mode`.
Unknown fields are rejected. `response_format` supports buffered `wav` and
streaming `pcm`; `stream_format: "sse"` emits OpenAI audio events with base64
PCM. `speed` supports `1.0` only; input is capped at 16384 UTF-8 bytes and
`max_new_tokens` at 8192 (default 3000). Output is 24 kHz mono 16-bit PCM.

`GET /v1/audio/speech/stream` upgrades to the vLLM-Omni incremental-text
WebSocket protocol. [TTS streaming examples](models/qwen3-tts/README.md#sampling-and-streaming)
cover per-session configuration, audio events and optional sentence/clause
segmentation. Output streaming does not change model prompt construction.
HTTP/1.1 streams use chunked transfer encoding; failed generation omits the
terminal chunk so clients can detect truncated audio. SSE also reports an error
event, while WebSocket speech reports an error without `audio.done`.

Qwen3-ASR serving uses its own model process:

```sh
./result/bin/gufo serve asr \
  --model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/<revision>
```

`POST /v1/audio/transcriptions` accepts OpenAI-compatible multipart fields
`file`, `model`, `language`, `prompt`, `response_format`, `temperature`, and
`max_tokens`, and `stream`. The native route is deterministic (`temperature=0`)
and supports `json`, `text`, and `verbose_json`. `stream=true` with JSON emits
OpenAI transcript delta/done events. Long uploads are split within model
capacity; `max_tokens` applies per chunk. Timestamp requests are rejected.

`GET /v1/realtime?intent=transcription` upgrades to OpenAI's manual-commit
transcription WebSocket interface. [ASR streaming examples](models/qwen3-asr/README.md#streaming)
describe PCM format, append/commit/clear, event IDs and limits. WebSockets use
the same bearer authentication and connection limits as HTTP. Messages are
bounded to 4 MiB, queued input to 8 MiB/128 messages, and unconsumed output to
a five-second socket timeout. Native generation observes close/disconnect;
no per-server streaming flag or extra model is required.

The MiniMax H3 subset follows the asynchronous OpenAI-style video resource
shape and is versioned independently as `gufo.video-api.v1`. Its supported
fields, frozen presets, queue behavior, and deliberate conditioning
omissions are documented in the [H3 upstream contract](models/minimax-h3/QUALITY.md).
Create requests accept both `application/json` and OpenAI-client-compatible
`multipart/form-data`; duplicate form fields, malformed boundaries, unsupported
media types, and reference-image parts fail explicitly.
H3 accepts `seconds` as a JSON number or string (including multipart), at
`512x512` or `1344x768`. Durations from 5 seconds are converted to 24-fps frames
and rounded up to the official VAE grid (`17n + 5`), within upstream's
15-second ceiling: 5 → 124 frames, 10 → 243, 14 → 345. The maximum is
`14.375` seconds (345 frames); `15` would round to 362 frames and is rejected,
as upstream does. An explicit `gufo.frames` must match the aligned duration.
The one-second/22-frame diagnostic extension remains available.
The encoder accepts at most 4,096 tokens after tokenization and normalization.
There is no 4,096-byte prompt limit. HTTP request bodies remain bounded by
`--max-request-bytes`; token-limit violations fail the asynchronous job before
prompt-encoder weights or activations are allocated.
The bounded worker retains prompt text only in volatile queued or active
request memory, persists only its SHA-256 digest, and wipes both source and
active string storage after transfer and completion. Completed MP4s are probed
for H.264/AAC codec, geometry, rates, duration, and A/V synchronization before
atomic publication.

## Internal Request Model

Text endpoints use `ChatRequest` and `SamplingConfig` at the
[backend boundary](../src/cli/serve/text_generation_backend.hpp).
Chat messages retain roles, reasoning, tool calls and image content until the
model's tokenizer and template turn them into model input. The scheduler owns
request limits, cancellation, cache accounting and completion state.

## Responses API Subset

`POST /v1/responses` accepts `model`, `input` as text or message arrays,
`instructions`, `max_output_tokens`, `stream`, `reasoning.effort`,
`text.format`, `tools`, `tool_choice`, `parallel_tool_calls`, and the shared
sampling controls. Message content supports
`input_text` and `input_image` with an `image_url` (HTTPS or a data URL).
Image uploads accept PNG, JPEG and WebP. Base64 data URLs also accept
`image/jpg`, mixed-case media types/`base64`, and parameters such as `name=`.
Clients supply the complete conversation, including prior Gufo `output` items
when retaining reasoning. Replay `function_call` items with their `call_id`,
then supply `function_call_output` items using the same ID. Function tools use
the flat `{type:"function",name,parameters,strict}` shape.
`store` and `background` must be false when present; server-side conversations
and `previous_response_id` remain unsupported.

Responses report `incomplete` with reason `max_output_tokens` when generation
hits its limit. Otherwise they report `completed`. `stream: true` sends typed
SSE events with consecutive `sequence_number` values: lifecycle, output items,
text/reasoning deltas and terminal status. Local reasoning is exposed as
`reasoning` items with `summary_text`; visible answers use `output_text`.
Disconnects cancel generation through the same scheduler as Chat Completions.

The official [OpenAI Python SDK](https://github.com/openai/openai-python) is
included in `nix develop`. Use the model name from `/v1/models`
(for example, `qwen` with `--served-model-name qwen`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
for event in client.responses.create(
    model="qwen", input="Hello", store=False, stream=True
):
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.failed":
        raise RuntimeError(event.response.error.message)
```

The other compatibility routes are deliberately limited:

| Route | Supported request | Output limit |
| --- | --- | --- |
| `/v1/completions` | One prompt string, buffered or SSE completion | `max_tokens` |
| `/v1/messages` | Text messages and optional text system instructions | `max_tokens` |
| `/completion` | One prompt string, non-streaming completion | `n_predict` |

All four routes validate the loaded model, positive integer limits and shared
sampling controls. Messages and `/completion` reject streaming; all reject
multiple candidates. Responses and Messages honor the server's thinking defaults.
Messages accepts `thinking.type` (`enabled` or `disabled`); `budget_tokens`
has no native equivalent and keeps the server's effort. Reasoning is returned
as a `thinking` block before the `text` block, with an empty `signature`.
Replay assistant `thinking` blocks unchanged so later turns reuse the cached
prompt. Messages rejects tools and `output_config`; use Chat Completions for
tool and reasoning-effort controls. Completions routes accept `stop`;
Messages accepts `stop_sequences`. Responses has no stop-sequence field.
`/infill` and `/v1/messages/count_tokens` return 501: suffix-conditioned infill
and template-aware message counting are not implemented.

Raw Completions accepts `stream: true` and
`stream_options: {"include_usage": true}`. The final usage event reports prompt,
cached and completion token counts before the `[DONE]` sentinel. Local benchmark
clients can set `ignore_eos: true` to generate exactly `max_tokens`; context
capacity remains the hard limit. Chat Completions, Responses, Messages and
`/completion` reject `ignore_eos`: Raw Completions owns the fixed-length
contract. `GET /v1/models` reports that configured limit as `context_length` on
the loaded text model.

`ignore_eos` measures decode past the model's stop tokens, which speculative
backends do not accelerate: verification ends its accepted run at every EOS it
emits, so each crossing clips the draft window. Expect decode near the
autoregressive rate in that region, not the rate the same model reaches on an
ordinary continuation.

## Chat Completions Adapter

`POST /v1/chat/completions` accepts the common compatibility subset:

- `model`
- `messages`
- `max_tokens` or `max_completion_tokens`
- `temperature`
- `top_p`
- `seed`
- `stop`: null, one string, or an array of up to four strings
- `stream`
- `stream_options.include_usage`
- `return_progress` (see [Prompt progress](#prompt-progress))
- `tools` and `tool_choice` when supported. Function tools accept the nested
  Chat Completions shape and the flat Responses-style `{type,name,parameters}`
  shape. Missing or null `parameters` become `{}`. `parametersJsonSchema` is
  accepted as an alias for `parameters`. Flat definitions retain all function
  fields, including `strict`. A function name holds 1-64 printable ASCII
  characters and may use any of them except a space, `<`, `>`, `"` and `\`,
  which frame a rendered call; dotted and namespaced names such as
  `github.create_issue` are accepted. OpenAI itself documents a narrower set
  for this field, so a name outside `[A-Za-z0-9_-]` is portable to gufo but not
  to every OpenAI-compatible service. Historical names in assistant
  `tool_calls` and Responses `function_call` items are preserved verbatim,
  including Unicode and names absent from the current tools. This does not
  authorize new calls to them. History still requires a non-empty string name and
  JSON-object arguments; embedded NUL names are unsupported. Malformed history
  and invalid declarations return 400 before generation.
- shared top-k, min-p, repeat, frequency and presence sampling controls

Streaming objects use `chat.completion.chunk` and end with the compatibility
sentinel expected by common clients.

### Prompt progress

Streaming `/v1/chat/completions`, `/v1/completions` and `/v1/responses` requests accept
llama-server's `return_progress: true`. Before the first token, the stream
carries chunks with an empty delta (Chat) or empty text (Completions) and a
top-level `prompt_progress` object:

```json
{"prompt_progress": {"total": 4096, "cache": 1024, "processed": 2048, "time_ms": 850}}
```

`processed` includes cached tokens; a full cache hit reports `processed == total`.
`time_ms` measures prompt-processing wall time. Updates follow prefill chunks;
a slow client receives only the newest pending update. Responses uses
`response.in_progress` events. Omitted, false or null disables progress;
other non-boolean values return 400. Buffered requests ignore the flag.
Disabled progress adds no progress queue updates or consumer wakeups.

The adapter must not implement a second inference path. It converts messages
into the same prompt and sampling structures used by `/v1/responses`.

Frequency and presence penalties count every committed token generated in the
current request, excluding its prompt. Repetition penalties use `repeat_last_n`
and may include prompt tokens. Setting that window to zero disables only the
repetition penalty. Speculative rejection discards tentative counts; seeded
sampling replay retains independent request histories.

API ranges: temperature `[0, 2]`, top-p/min-p `[0, 1]`, frequency/presence
penalties `[-2, 2]`. Top-p zero keeps the highest-probability token (or
`min_keep` tokens). Setting temperature to zero retains configured penalties;
omitted controls keep their model/CLI defaults.

Tool calls are emitted only for declared functions when `tool_choice` allows
calling tools. With `auto`, ordinary text and reasoning remain allowed; once a
call starts, decoding constrains its name and argument format. Non-strict tools
keep optional arguments optional. Open nested objects retain native syntax and
declared requirements/types, including nested fields; unsupported schema
keywords remain guidance. Unsupported property-admitting rules, including
conditional branches, leave those objects open without discarding declared
requirements. Qwen wildcard fields and ambiguous string/null unions use JSON
to preserve types. Constrained JSON keys follow schema order, with additional
keys last. Impossible non-strict schemas fall back to JSON-object arguments;
impossible strict schemas are rejected before generation.
`tool_choice: "required"` and named choices constrain decoding to a declared
call. Extended schemas retain compact JSON on this path, avoiding extra
native framing tokens; ordinary native calls keep their existing format. Where
the backend cannot constrain sampling, an unmet `required` choice still returns
`tool_choice_unsatisfied` (HTTP 502, or an SSE error after streaming starts).
Stops and token limits terminate normally without emitting incomplete calls.
With no tools or `tool_choice:"none"`, tool markers are ordinary text and do
not interrupt reasoning or delay streaming.

Stop sequences match accepted output bytes, including reasoning and tool
markup, before streaming or response parsing. Partial prefixes are buffered;
matched sequences and subsequent text are excluded. OpenAI reports
`finish_reason: "stop"`; Messages reports `stop_reason: "stop_sequence"` and
the matched `stop_sequence`. EOS and length limits flush unmatched prefixes.
Interrupted tool calls are omitted; complete preceding calls are retained.
Usage includes the token completing the match. Each request has independent
matching state, including speculative batches; caches retain only correctly
labelled executed model state. Stops must be nonempty, at most 4 KiB each and
16 KiB combined. Messages allows up to 64 sequences.

Nullable Chat Completions defaults retain server settings, including sampling
and token limits. `logprobs: false`, empty `logit_bias`,
`response_format: {"type":"text"}` and `modalities: ["text"]` are accepted.
Actual log probabilities, token biases and audio output remain unsupported
and return explicit errors.

### Structured output

Chat Completions accepts `response_format: {"type":"json_object"}` or
`{"type":"json_schema","json_schema":{"name":"Reply","strict":true,"schema":…}}`.
Responses uses `text: {"format":{"type":"json_schema","name":"Reply","strict":true,"schema":…}}`.
The official SDK supports typed parsing and streaming through
`client.chat.completions.parse/stream(response_format=Model)` and
`client.responses.parse/stream(text_format=Model)`.

Constraints apply before target sampling in AR, DFlash2, MTP and DSpark, including
streaming, images and concurrent requests. Reasoning stays separate from JSON
and counts toward the output budget. Changing the schema changes the cache prefix.

For constrained tool or JSON output, only `</think>` ends the initial reasoning
phase. Literal tool markers such as `<tool_call>` quoted during reasoning remain
reasoning data; they do not start a call or move reasoning into visible content.
This boundary is identical for buffered responses and SSE deltas in Chat
Completions and Responses. Tool parsing starts after the reasoning delimiter,
and markers inside tool argument strings remain argument data.

Parse the returned content: leading whitespace is valid JSON, and stops or token
limits can leave it incomplete. `finish_reason: "stop"` includes matched stop
sequences and does not guarantee complete JSON. Token limits return `"length"`
in Chat or `status: "incomplete"` with reason `max_output_tokens` in Responses.

Supported: objects, arrays, nullable types, `enum`/`const`, `anyOf`, recursive local
`$ref`/`$defs`, numeric bounds/`multipleOf`, string patterns/lengths/formats and
array length bounds. Formats include date/time, duration, email, hostname,
IP addresses and UUID.

The root must resolve to an object. Objects require `additionalProperties:false`;
strict schemas require every property (use null for optional values).
Unsupported keywords, external references and unusable cycles are rejected.
Schemas are limited to 2 MiB, 5,000 properties, 1,000 enum values and 120,000
characters in names and enum/const strings. Patterns use ECMA-262 Unicode
semantics; lookbehind, backreferences, inline flags and unbounded repetition
of assertions are unsupported.

Function `strict:true` constrains tool arguments independently of the response
schema. Chat Completions defaults to non-strict tools. Responses attempts strict
schema normalization when `strict` is omitted, falling back to `strict:false`
when unsupported; explicit `strict:false` keeps best-effort arguments.
With `tool_choice: "auto"`, the model may call a tool or give a final answer;
the response schema constrains the latter. Use `"none"` for JSON answers only,
`"required"` to force a call, or select a named function.
`"required"` and a named function constrain decoding regardless of tool
strictness; `parallel_tool_calls:false` allows at most one call and constrains
it likewise. Interrupted calls are omitted.

References: [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[function calling](https://developers.openai.com/api/docs/guides/function-calling),
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[JSON Schema patterns](https://json-schema.org/draft/2020-12/json-schema-validation#name-pattern)
and [llama.cpp grammar sampling](https://github.com/ggml-org/llama.cpp/blob/68d9053afd4f4d0752ced6187585f862355a40be/common/sampling.cpp).
Verify with `tests/functional/openai_sdk.py --suite tools`,
`--suite structured` or `--suite structured-limits`
(add `--vision` for an image-capable server).

## Model Discovery

`GET /v1/models` lists the configured text model and ready audio/video
services. Entries provide `id`, `object`, `created` and `owned_by`; the text
entry also provides `context_length` and `architecture.input_modalities`: `["text"]`
for text-only serving or `["text", "image"]` when a vision encoder is loaded.
Image support reflects the loaded projector, not the model name; launching a
vision-capable model without its projector advertises text only. Audio/video
entries describe their capability. Send the returned model ID in requests.
Text requests naming another model return 404.

## Errors

Return an OpenAI-style error object:

```json
{
  "error": {
    "message": "Unsupported field: logprobs",
    "type": "invalid_request_error",
    "param": "logprobs",
    "code": "unsupported_parameter"
  }
}
```

Use appropriate HTTP status codes:

- `400` invalid request or unsupported field
- `401` missing or invalid configured bearer token
- `404` unknown model or endpoint object
- `409` incompatible session state
- `413` request or prompt too large
- `429` admission queue full or rate limit exceeded
- `499` internally recorded client cancellation
- `500` internal failure
- `503` model or backend unavailable, or GPU context lost (`device_lost`)

Generation stops at the request budget or context capacity and reports a
length finish reason when either limit is reached.

## Authentication and Exposure

The server binds to `127.0.0.1` by default. Set `--api-key` to require:

```text
Authorization: Bearer <local-token>
```

Authentication covers every route, including health, metrics, audio, and video.
CORS preflight (`OPTIONS`) does not require credentials. Select the listen
address with `--host`; it must be an IPv4 address. TLS termination belongs in
a reverse proxy.

Request bodies, prompts, generated text and credentials are not logged.

Tool definitions and generated tool calls are treated as data. `gufo`
never executes tools, shell commands, URLs, or generated code.

The HTTP transport currently uses a wildcard CORS origin and the same listener
and API key for inference and metrics.

`gufo eval` and `tools/serving/gufo-serving-bench.py` send `OPENAI_API_KEY`
as the bearer credential when it is set.

## Lifecycle and limits

The HTTP transport bounds connection count and request-body size. The scheduler
bounds admission and output buffering and propagates client cancellation to
model runners. An in-flight GPU operation may finish before its request retires.
See [CLI.md](CLI.md) for supported configuration flags.

Admission groups text requests by the socket peer's IP address across chat and
compatibility endpoints. Caller-provided identity headers do not affect quotas;
clients behind the same proxy or NAT share a peer quota.

Cancellation retains the last successfully executed conversation frontier and
the immutable prompt snapshot. It does not execute a selected but unfinished
token to populate the cache. A failed model operation invalidates its mutable
state; the prompt snapshot remains available for safe replay. Image identity,
positions and speculative state participate in restoration and cache isolation.

`GET /health` reports process liveness. `GET /ready` returns 503 until a model
service is ready, then reports `status` and the active model. Neither performs
GPU work on each poll. HTTP model replacement and persistent Responses
conversations are not implemented.

A GPU reset (for example `amdgpu` recovering from a MES hang) permanently
invalidates the process's HIP context. Loss is detected when a text generation
fails: an idle server with a dead GPU stays healthy until its next request.
After such a failure the scheduler runs a bounded device probe (a 4-byte
memset on a private stream, at most 5 s), before invalidating request state.
Only a hard HIP error from the probe
marks the device lost. A probe still pending after 5 s may be queued behind
long kernels, so it logs `event=device_probe_timeout` and counts as usable. A
usable device leaves the original failure unchanged. A lost device logs
`event=device_lost remedy=restart reason=<driver error>` once and makes the
loss permanent for the process:

- the failing request reports `device_lost`: a 503 with
  `GPU context lost; restart required` and no `Retry-After` before the
  response starts, or the same code in the stream's terminal error event
  after it started. `/v1/responses` streams end with `response.failed`
  carrying the Responses `server_error` code, since that code enum is closed;
  the message names the device loss;
- `/health`, `/ready` and their aliases return 503 with
  `{"status":"device_lost","error":{...,"code":"device_lost"}}`;
- every text-backend `POST` route, including unimplemented stubs that would
  otherwise answer 501, returns 503 `device_lost` without reaching the device;
- `gufo serve` logs `event=device_lost_shutdown`, runs the `SIGTERM` shutdown
  and exits with status 75 (`EX_TEMPFAIL`), also when an external `SIGTERM`
  stops it after the loss. Teardown can block on the dead device, so the
  process exits with status 75 after 10 s regardless.

The listener observes the sticky loss independently of response writes, so
a blocked streaming client cannot delay arming that watchdog. Failed resident
requests retain their state and snapshot workers until process exit, avoiding
HIP resets, frees and transfer joins on the lost context. The watchdog is armed
before logging or shutdown; the fatal exit bypasses model destructors and log
flushing. The operating system reclaims those resources.

Run the server under a supervisor that restarts on failure, such as systemd
`Restart=on-failure` or a container `restart: always`/`on-failure` policy;
a restart is the only recovery.

## Metrics

`/metrics` exposes total prompt/generated tokens and the latest prompt/decode
speeds. The token counters advance as each prefill chunk and generated token
executes, so their rates show live throughput; prompt tokens exclude cache
hits. `llamacpp:requests_processing` counts admitted requests, including cache
preparation and cleanup; `llamacpp:requests_deferred` counts requests waiting
for a session. The speed gauges retain the latest nonzero request rates.
When the scheduler finishes or cancels a request, whether or not the client
reads the result, it adds the request's cached prompt tokens
(`llamacpp:prompt_tokens_cached_total`), prefill and decode seconds
(`llamacpp:prompt_seconds_total`, `llamacpp:tokens_predicted_seconds_total`),
speculative verification rounds (`llamacpp:spec_decode_num_drafts_total`),
proposed and accepted draft tokens
(`llamacpp:spec_decode_num_draft_tokens_total`,
`llamacpp:spec_decode_num_accepted_tokens_total`), and raises
`llamacpp:n_tokens_max` to its prompt plus generated tokens. A request counts
even if the client disconnects before reading it; requests that fail in the
scheduler, such as on a deadline or a runner error, add none of these.
`llamacpp:kv_cache_usage_ratio` is the in-flight prompt and generated tokens
over sessions times context; retained cache entries are not counted.
`Server-Timing`, generation `timings`, and Chat Completions `usage.gufo`
provide request-level measurements.

`GET /slots` (alias `/v1/slots`) lists one llama-server slot per `--sessions`
entry. A session is processing from admission, including cache preparation,
until the request ends, so the processing slots match
`llamacpp:requests_processing`; queued requests do not appear.
`id_task` is the request id, or -1 when idle. `n_prompt_tokens` is the whole
prompt, `n_prompt_tokens_cache` the tokens restored from cache, and
`n_prompt_tokens_processed` the uncached tokens prefilled so far, as in
`timings.prompt_n`. `next_token[0].n_decoded` counts generated tokens and
`n_remain` the remaining token budget. Idle slots report zero counts and
`n_remain` -1. `task_id` and `state` (0 idle, 1 processing) remain for older
clients; `prompt` is always empty. `/props` metadata is a placeholder.
Slots and metrics are separate snapshots; requests can advance between polls.

Streaming terminal chunks always include llama.cpp-compatible `timings`, even
without `stream_options.include_usage`. `prompt_n` counts newly processed
tokens; `cache_n` counts reused tokens. llama-swap uses these fields on every
turn. Gufo-specific details stay in `usage.gufo`.

TODO: retained-cache KV metrics, a validated administrative reload/drain
interface, and in-process recovery after device reset or suspend/resume.

## Troubleshooting logs

Server lifecycle and request logs go to stderr. Each request gets an
`X-Request-ID` response header matching its `request=rN` log entries. Inference
requests log receipt and completion; streaming completion is logged after the
stream ends. Successful health/metrics and video-status polls are quiet at the
default level.

Set verbosity with `--log-level <error|warn|info|debug>` (default `info`);
`-v`/`--verbose` is shorthand for `--log-level=debug` and cannot combine with
the canonical spelling — passing both is a usage error rather than a precedence
to guess at. Each tier is a threshold, not only an addition:
`--log-level error` shows ERROR lines alone, so a 5xx response or stream
failure still logs while a 4xx refusal, which logs at WARN (a 429 admission
refusal included), is filtered out.

The threshold also covers the lifecycle lines. `event=listening` (the bound
address, auth mode and connection limits) and `event=shutdown_requested` (the
signal that asked for a stop) are INFO-tier, so `--log-level=warn` and
`--log-level=error` start and stop with no output at all. Keep the default
`info` for systemd units and CI that read the startup banner, or confirm boot
with `GET /health` or `GET /ready`.

The `debug` tier adds:

- the resolved server options as `event=options` before the model opens, with
  `api_key=set` rather than the key. This line and the `event=listening` banner
  are separate on purpose: the banner only appears once the model has loaded and
  the listener is accepting, so a load that fails or hangs leaves
  `event=options` as the sole record of what was asked for;
- completion logs for the polls that are quiet by default, and `event=received`
  for GET requests;
- scheduler decisions: `event=admitted` with the pending queue depth and the
  per-client depth the request joined, `event=admission_refused` naming the limit
  that rejected it and the client, and `event=backpressure` once per stream
  naming which output budget refused a token (`request_buffer` or
  `total_buffer`) with the byte counts;
- cache candidate detail behind the summarised lines: `event=candidate_skip`
  gives the entry index, its token length and the first guard that failed
  (`unavailable`, `input_identity`, `stable_prefix_boundary` or
  `token_prefix`), and `event=capture_evicts` names the checkpoint a capture is
  about to overwrite. A request that finds every slot busy emits its skip
  records once, so a long wait does not repeat them on every 10ms poll.

Loader phases stay at INFO: the weight-mapping and session-preallocation work is
HIP-only code, so deeper sub-phases there need a GPU build to verify and are not
part of this tier. The imported DeepSeek V4 Flash runtime logs through the same
threshold rather than writing to stderr directly, so its informational startup
lines (the ROCm model-cache and managed-KV lines, DSpark attachment, the shared
batch workspace) are suppressed by `--log-level=warn`/`error` too.

Prompt text, message bodies and API keys stay unlogged at every level, and debug
lines use the same escaping and redaction as the rest of the log. Client
identity is the exception: scheduler admission lines name a client by the peer
address of its socket (`client_id=127.0.0.1` on the default loopback bind), so a
public `--host` writes client IP addresses into the debug tier.

Pass `--log-progress` to `gufo serve llm` to log each prefill chunk, each
50-token decode boundary, and the final decode remainder. Each line includes the
request ID, completed and total tokens, percentage, current speed and average
speed. Speculative requests also include accepted and proposed drafts. Progress
lines are INFO-tier: `--log-level=warn` or `--log-level=error` would discard
them, so the server rejects that combination at startup instead of ignoring the
flag.

Text completion logs include stop/length/cancellation, queue and first-token
latency, prefill/decode speed, execution width, memory/disk cache hits and reused
tokens, plus accepted/proposed drafts and acceptance percentage. These metrics
are logged even when a streaming client does not request a usage chunk.
Errors include a stable error code; disconnects and stream failures are marked.
On a cache miss, `cache_miss_reason` distinguishes a missing checkpoint, changed
token prefix, changed image input, and explicit cache bypass. Common-prefix and
nearest-checkpoint token counts report token agreement even when image
identity differs; matching image-placeholder tokens do not imply matching
pixels. Prompt text is not logged. Adding tools or editing system/developer
instructions near the beginning invalidates the later state: keep those inputs
stable during an agent conversation. A disconnected SSE stream may never
deliver its terminal usage chunk; proxy counters can then show zero despite
generated tokens.
The server's cancellation log retains the actual token counts.

Routine cache replacement is quiet; failed captures, disk corruption and cache
capacity refusals produce warnings. Cache capacity is a budget, not allocated
memory.

Loading logs report elapsed time, model, context and session capacity,
speculative mode and memory. `rss_mib` is process resident memory;
`host_available_mib` is available system memory. At load completion,
`gpu_device_used_mib` is device-wide HIP usage. These overlap on unified memory
and must not be added together.
With disk caching enabled, `phase=artifact_identity` identifies full-file
SHA-256 work. Digests are cached against the open file's identity, size and
modification/change timestamps; unchanged artifacts avoid another scan.
Startup qualification measures process launch through first prefill and first
token, including work deferred until the first request.

Audio summaries include model, duration and generation/transcription timings.
Video requests log a job ID linking queue, start, throttled phase progress and
completion/failure events. Video startup validates inventory; weights load
lazily in the worker. Control characters are escaped in log lines.

## Focused checks

- `json_test`: number precision, Unicode escapes, malformed input and depth limits.
- `http_server_test`: transport framing, authentication, compatibility validation,
  sampling forwarding, request logs at the default and debug tiers, and streaming
  failures without loading a model.
- `serve_cli_test`: executable help, argument wiring and rejected configurations,
  including that `--log-level` and `-v` change the emitted log rather than only
  parsing.
- `openai_chat_test`: chat parsing, streaming, images, tools and sampling controls.
- Per-model serving tests: greedy/sampled decoding, batching, cache reuse and
  cancellation. Use the affected model's benchmark README for commands and limits.

For session-boundary changes, also run a six-turn conversation: remember a fact,
force one token-limit stop, recall the fact, update it and recall the update.
Compare greedy AR/speculative token traces and repeat each sampled mode with the
same seed. Mix EOS and token-limit stops rather than testing only fixed-length
generation. This is a session/replay check, not a capability or distribution test.

2026-09-19: six-turn greedy AR/speculative traces matched for DS4/DSpark,
Flash-Next/MTP and Qwen27B Q4/Q8 with the Q4 DFlash2 draft. Speculative runs
repeated exactly at temperature 0.6, seed 7, with a 32-token turn limit.
Two simultaneous Qwen27B Q4/DFlash2 HTTP conversations also retained independent
facts through six turns, with streaming and memory-cache reuse.

DSpark's short boundary check is available without the complete serving suite:

```sh
# Set GUFO_DEEPSEEK_V4_FLASH_MODEL and GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL first.
nix develop -c build/gpu-test/tests/models/deepseek_v4_flash/ds4_serving_test --dspark-eos
```
