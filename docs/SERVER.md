# OpenAI-Compatible Server

Status: implemented subset, 2026-09-25

## Purpose

The server exposes a focused OpenAI-compatible API while keeping inference,
scheduling, and device execution in the C++ process. Compatibility is a
versioned contract: supported fields behave as documented, and unsupported
fields return explicit errors.

Chat Completions is the main API, including streaming, images and tools.
Responses supports text with optional streaming; Anthropic Messages exposes a
synchronous text subset.
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
`--sessions N` controls the bounded session pool. The scheduler batches ready
requests when the model runner supports their execution mode.

Text serving defaults match llama.cpp for context and generation length:

| Setting | Default |
| --- | --- |
| `--context` | `0`: native context from model metadata, per session |
| `--max-tokens` | `-1`: until EOS or remaining context is exhausted |
| `--sessions` | `1` |
| Thinking / reasoning effort | Model template defaults |

Clients can set a positive `max_tokens` / `max_completion_tokens` (Chat
Completions) or `max_output_tokens` (Responses). These include reasoning tokens.
Omitting the field uses the server default. A response cannot exceed remaining
context; exhaustion reports `length` or `incomplete`, without discarding earlier
conversation tokens. Reduce `--context` or `--sessions` if their state exceeds
available memory.

Gufo retains greedy sampling by default. llama.cpp instead defaults to temperature
0.8, top-k 40, top-p 0.95 and min-p 0.05; set these explicitly to match its sampler.
Both leave repetition, frequency and presence penalties disabled.
Reference: [llama.cpp parameters](https://github.com/ggml-org/llama.cpp/blob/68d9053afd4f4d0752ced6187585f862355a40be/common/common.h)
and [server options](https://github.com/ggml-org/llama.cpp/blob/68d9053afd4f4d0752ced6187585f862355a40be/tools/server/README.md).

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

Prompt reuse is enabled by default. `cache_prompt: false` on
`/v1/chat/completions` bypasses both memory and disk lookup for that request;
the result can still populate the cache. DeepSeek and Qwen tool requests retain
a checkpoint before the assistant-generation suffix, including when a client
drops the interrupted assistant and appends `"."` after a tool result. DeepSeek
also accounts for tokenization changes where adjacent user/tool turns join.
Qwen requests that remove previous reasoning or enable thinking retain this
checkpoint too.
Exact live continuations reuse generated tokens. The server reports cached
and newly processed tokens separately; resuming from the checkpoint processes
the short suffix. System instructions, tool definitions and image identities
must match the retained prefix.

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
`python3 tools/serving/check-continuation.py --output /tmp/cache-check.json`
against a private server named `cache-test` on port 5815.
It checks interruption during reasoning and visible output, with and without
reasoning replay, greedy/seeded sampling, and explicit cache bypass. Use
`--tools --discard-assistant` to exercise interrupted agent tool turns; add
`--prefix-repetitions 5500` for a roughly 50K-token prefix.
For persistence, enable `--cache-disk` before the check, restart the same server,
and add `--restore /tmp/cache-check.json`.
Use `--image /path/to/image.png` for Qwen image conversations.
Each case continues for a third turn; repeat `--case NAME` to select only the
cases needed for a change.
The check requires exact snapshot and matched-history replay. It separately
reports equality to a fresh full prefill, whose different matrix shapes and
prefill/decode history can change rounding; that comparison is not silently
counted as an exact cache replay.

### Reasoning controls

`--think auto` uses the model's default. Qwen27B and Flash-Next match the
official Jinja: thinking enabled, `xhigh` effort, prior reasoning preserved.
Use `--think off` or `chat_template_kwargs.enable_thinking=false` for direct
answers. DeepSeek retains its chat-mode default. Quality comparisons must use
the same reasoning mode and effort.

`POST /v1/chat/completions` accepts top-level `reasoning_effort` (`off`,
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
| `POST` | `/v1/responses` | Text, optional SSE streaming |
| `POST` | `/v1/chat/completions` | Main chat, streaming, image and tool API |
| `POST` | `/v1/completions` | Optional legacy text completion adapter |
| `GET` | `/health` | Process liveness (aliases: `/v1/health`, `/healthz`) |
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

`POST /v1/responses` accepts `model`, `input` as text or text-message arrays,
`instructions`, `max_output_tokens`, `stream`, and the shared sampling controls.
Clients supply the complete conversation, including prior Gufo `output` items
when retaining reasoning. `store` and `background` must be false
when present. Images, tools, structured output, server-side conversations and
`previous_response_id` are rejected on this route. Use Chat Completions for images
and tools.

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
| `/v1/completions` | One prompt string, one non-streaming completion | `max_tokens` |
| `/v1/messages` | Text messages and optional text system instructions | `max_tokens` |
| `/completion` | One prompt string, non-streaming completion | `n_predict` |

All four routes validate the loaded model, positive integer limits and shared
sampling controls. The three routes above reject streaming; all reject multiple
candidates. Responses and Messages honor the server's thinking defaults.
Native Messages rejects tools, `thinking`, and `output_config`; use Chat
Completions for tool/reasoning controls. Completions routes accept `stop`;
Messages accepts `stop_sequences`. Responses has no stop-sequence field.
`/infill` and `/v1/messages/count_tokens` return 501: suffix-conditioned infill
and template-aware message counting are not implemented.

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
- `tools` and `tool_choice` when supported. Function tools accept the nested
  Chat Completions shape and the flat Responses-style `{type,name,parameters}`
  shape. Missing or null `parameters` become `{}`. `parametersJsonSchema` is
  accepted as an alias for `parameters`. Flat definitions retain all function
  fields, including `strict`. Unsupported tool types, malformed entries and
  non-object parameters return 400 `invalid_tools` before generation.
- shared top-k, min-p, repeat, frequency and presence sampling controls

Streaming objects use `chat.completion.chunk` and end with the compatibility
sentinel expected by common clients.

The adapter must not implement a second inference path. It converts messages
into the same prompt and sampling structures used by `/v1/responses`.

Frequency and presence penalties count every committed token generated in the
current request, excluding its prompt. Repetition penalties use `repeat_last_n`
and may include prompt tokens. Setting that window to zero disables only the
repetition penalty. Speculative rejection discards tentative counts; seeded
sampling replay retains independent request histories.

Tool calls are emitted only for declared functions when `tool_choice` allows
calling tools. An unmet `required` choice returns `tool_choice_unsatisfied`
(HTTP 502, or an SSE error after streaming starts), unless a requested stop
sequence interrupted generation first.

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
Actual log probabilities, token biases, structured outputs and audio output
remain unsupported and return explicit errors.

Admission groups text requests by the socket peer's IP address across chat and
compatibility endpoints. Caller-provided identity headers do not affect quotas;
clients behind the same proxy or NAT share a peer quota.

## Model Discovery

`GET /v1/models` lists the configured text model and ready audio/video
services. Entries provide `id`, `object`, `created` and `owned_by`; audio/video
entries also describe their capability. Send the returned model ID in requests.
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
- `503` model or backend unavailable

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

Cancellation retains the last successfully executed conversation frontier and
the immutable prompt snapshot. It does not execute a selected but unfinished
token to populate the cache. A failed model operation invalidates its mutable
state; the prompt snapshot remains available for safe replay. Image identity,
positions and speculative state participate in restoration and cache isolation.

`GET /health` reports process liveness. `GET /ready` returns 503 until a model
service is ready, then reports `status` and the active model. It does not expose
a GPU health matrix. HTTP model replacement and persistent Responses
conversations are not implemented.

## Metrics

`/metrics` exposes total prompt/generated tokens and the latest prompt/decode
speeds. `Server-Timing`, generation `timings`, and Chat Completions
`usage.gufo` provide request-level measurements. The legacy KV-utilization
metric and `/slots`/`/props` metadata are placeholders; do not use them for
capacity or admission decisions.

Streaming terminal chunks always include llama.cpp-compatible `timings`, even
without `stream_options.include_usage`. `prompt_n` counts newly processed
tokens; `cache_n` counts reused tokens. llama-swap uses these fields on every
turn. Gufo-specific details stay in `usage.gufo`.

TODO: real slot/KV metrics, a validated administrative reload/drain interface,
and automatic recovery after device reset or suspend/resume.

## Troubleshooting logs

Server lifecycle and request logs go to stderr. Each request gets an
`X-Request-ID` response header matching its `request=rN` log entries. Inference
requests log receipt and completion; streaming completion is logged after the
stream ends. Successful health/metrics and video-status polls are quiet.

Pass `--log-progress` to `gufo serve llm` to log each prefill chunk, each
50-token decode boundary, and the final decode remainder. Each line includes the
request ID, completed and total tokens, percentage, current speed and average
speed. Speculative requests also include accepted and proposed drafts.

Text completion logs include stop/length/cancellation, queue and first-token
latency, prefill/decode speed, execution width, memory/disk cache hits and reused
tokens, plus accepted/proposed drafts and acceptance percentage. These metrics
are logged even when a streaming client does not request a usage chunk.
Errors include a stable error code; disconnects and stream failures are marked.
On a cache miss, `cache_miss_reason` distinguishes a missing checkpoint, changed
token prefix, changed image input, and explicit cache bypass. Common-prefix and
nearest-checkpoint token counts explain how far the inputs agree, without
logging prompt text. Adding tools or editing system/developer instructions near
the beginning invalidates the later state: keep those inputs stable during an
agent conversation. A disconnected SSE stream may never deliver its terminal
usage chunk; proxy counters can then show zero despite generated tokens.
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
  sampling forwarding, request logs and streaming failures without loading a model.
- `openai_chat_test`: chat parsing, streaming, images, tools and sampling controls.
- `serve_cli_test`: executable help, argument wiring and rejected configurations.
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
