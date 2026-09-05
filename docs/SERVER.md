# OpenAI-Compatible Server

Status: active implementation contract, 2026-08-22

## Purpose

The server exposes a focused OpenAI-compatible API while keeping inference,
scheduling, and device execution in the C++ process. Compatibility is a
versioned contract: supported fields behave as documented, and unsupported
fields return explicit errors.

The primary API is the Responses API. Chat Completions is an adapter over the
same internal request representation. The implementation should follow the
official OpenAI API reference for object names and streaming event shapes:

- https://developers.openai.com/api/reference/resources/responses/methods/create/
- https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/
- https://developers.openai.com/api/reference/resources/models/methods/list/

The local server does not need to reproduce OpenAI-hosted storage, billing,
organization, or account behavior.

The only supported production platform is Linux x86-64 on Strix Halo.

## Implementation

Use C++20 for the complete runtime:

- Asynchronous HTTP/1.1 with keep-alive.
- Server-Sent Events for streaming.
- Separate I/O, scheduling, and device-execution threads.
- Incremental JSON parsing with bounded request sizes.
- Backpressure-aware response writers.
- Cancellation propagated from socket closure to the scheduler.

Boost.Asio/Beast is a reasonable initial transport. Keep HTTP and JSON types
outside the inference core so the transport can be replaced without changing
request scheduling.

Python may be used for development tools and API conformance tests, but it must
not be required to serve requests.

## Single Executable and Model Configuration

The deployed product is one `gufo` executable. Supported model graphs,
tokenizers, HIP kernels, and AIE programs are compiled into it.

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
[MiniMax H3 Integration Boundary](MINIMAX_H3.md).

### Current Single-Request HIP Path

Qwen HTTP requests share one immutable `QwenGpuModel` containing the mapped
weights and tokenizer. Each request leases a preallocated `QwenGpuExecutor`
with independent KV, recurrent, graph, activation, and logit state.
`--sessions N` controls the bounded session pool; it does not enable
continuous batching.

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

Cancellation and exceptions return a reset session to the pool. Model
replacement is transactional, so active requests retain their original model
until their lease ends. Generation responses include:

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

DeepSeek V4 Flash uses native layer-synchronous session batches when DSpark is
not attached. The scheduler advertises every physical width from 2 through 8
and chooses the exact runnable width. Every request keeps its own attention
caches and position; dense projections, attention output, FFN/MoE, and the LM
head run over the concurrent rows together. Width one stays on the existing
serial decode path. Set `GUFO_DEEPSEEK_SESSION_BATCH=0` before starting the
server to disable this route. DSpark sessions remain serial.

### Reasoning controls

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
Conflicting controls return `invalid_reasoning`. Thinking is off by default
unless the request or `gufo serve llm --think on` enables it.
Generated reasoning is returned as `reasoning_content` in ordinary and
streaming Chat Completions responses. Per-model effort mappings and history
policies are documented in the model cards under `src/models/`.

The server uses compiled model-specific formatters and validates recognized
artifact template hashes during model loading. It does not accept custom Jinja
or claim to enforce a reasoning-token budget.

Diagnostic telemetry is limited to status, token counts, timing, and
cancellation state. It must not contain prompt text, model paths, machine
identity, request IDs, or token IDs.

The user provides weight artifacts and configuration:

```toml
bind = "127.0.0.1:8080"
memory_policy = "guaranteed"

[[models]]
alias = "qwen-current-27b"
kind = "QWEN38_27B_TEXT"
weights = "/models/qwen-current/gufo-manifest.json"
enable_gpu = true
enable_npu = true

[[models]]
alias = "deepseek-flash"
kind = "DEEPSEEK_V4_FLASH"
weights = "/models/deepseek-flash/gufo-manifest.json"
enable_gpu = true
enable_npu = true
```

`kind` resolves through a closed compiled-in enum. The server verifies that the
manifest, tensor inventory, dimensions, quantization, tokenizer, and state
contract match that implementation.

Qwen3.8-27B is the first production model; Qwen3.5-0.8B is the rapid-iteration
fixture. The first production artifact is text-only, excludes the vision
encoder, and rejects image/video input.

Model artifacts cannot provide:

- Host shared libraries.
- HIP or generic GPU code objects.
- AIE overlays or `ctrlcode`.
- Tokenizer plugins.
- Executable chat templates.

Adding another model architecture requires rebuilding `gufo`.
Changing weights for an already supported kind does not.

## Process and State Ownership

Use separate execution domains:

```text
HTTP I/O threads
tokenization workers
single-owner scheduler
GPU submit/completion worker
NPU submit/completion worker
background storage workers
```

Only the scheduler mutates logical request, session, KV-ownership, speculative,
and commit state. Other threads send immutable commands or completion records.

Every completion record contains:

- Request and execution IDs.
- Allocation and state generations.
- Backend route and program/kernel identity.
- Completion status and bounded result metadata.

Stale completions are rejected by generation before they can alter request
state. I/O threads never wait synchronously for device completion.

## Endpoint Set

### Milestone 1

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | List loaded model aliases and capabilities |
| `POST` | `/v1/responses` | Primary text generation API |
| `POST` | `/v1/chat/completions` | Chat Completions compatibility |
| `POST` | `/v1/completions` | Optional legacy text completion adapter |
| `GET` | `/health` | Process liveness (aliases: `/v1/health`, `/healthz`) |
| `GET` | `/ready` | Model and backend readiness (aliases: `/v1/ready`, `/readyz`) |
| `GET` | `/metrics` | Prometheus-format operational metrics (text LLM serving only) |

### Later, capability-gated

| Method | Path | Condition |
| --- | --- | --- |
| `POST` | `/v1/embeddings` | An embedding implementation is compiled and loaded |
| `POST` | `/v1/audio/transcriptions` | An STT implementation is compiled and loaded |
| `POST` | `/v1/audio/speech` | A TTS implementation is compiled and loaded |
| `POST` | `/v1/images/generations` | An image implementation is compiled and loaded |
| `POST` | `/v1/videos` | A validated operator-supplied MiniMax H3 checkpoint is configured |
| `GET` | `/v1/videos/{id}` | MiniMax H3 video serving is configured |
| `GET` | `/v1/videos/{id}/content` | The requested MiniMax H3 job completed |
| `DELETE` | `/v1/videos/{id}` | MiniMax H3 video serving is configured |

Qwen3-TTS serving is enabled with a dedicated model process. All three 12Hz
1.7B variants are supported; the variant is detected from the checkpoint's
`tts_model_type` and determines both the advertised model id and the request
fields that are required:

```sh
./result/bin/gufo serve --port 8080 audio \
  --model /persist/models/audio/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

A single audio server can host Qwen3-TTS synthesis, Qwen3-ASR transcription,
or both, since `/v1/audio/speech` and `/v1/audio/transcriptions` dispatch from
independent services. Name each checkpoint explicitly to run both in one
process:

```sh
./result/bin/gufo serve --port 8080 audio \
  --tts-model /persist/models/audio/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --asr-model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/<revision>
```

At least one of `--tts-model` or `--asr-model` is required. A bare `--model`
and `--context` are backward-compatible aliases for `--tts-model` and
`--tts-context`. `--tts-context` (default 4096) and `--asr-context`
(default 1024) size each service independently. Both checkpoints load eagerly at startup, so running
them co-resident costs the sum of their weights.

| Variant | Model id | Voices | Additional required fields |
| --- | --- | --- | --- |
| CustomVoice | `qwen3-tts-12hz-1.7b-customvoice` | `talker_config.spk_id` names | `voice` |
| VoiceDesign | `qwen3-tts-12hz-1.7b-voice-design` | `voice-design` | `instruct` |
| Base | `qwen3-tts-12hz-1.7b-base` | `voice-clone` | `reference_audio`, plus `reference_text` unless `voice_clone_mode` is `speaker_embedding_only` |

`GET /v1/audio/voices` lists the advertised voices. CustomVoice exposes the
speaker names in `talker_config.spk_id`, while VoiceDesign and Base expose a
single placeholder name because their timbre comes from `instruct` or
`reference_audio` per request.

A Base checkpoint can additionally advertise operator-registered named voices,
so clients select a speaker by name instead of uploading a reference clip on
every request:

```sh
./result/bin/gufo serve audio \
  --tts-model /persist/models/audio/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator_eng=/persist/models/audio/clear-english-voice.wav \
  --voice narrator_ita=/persist/models/audio/clear-italian-voice.wav
```

`--voice NAME=PATH` is repeatable. The reference transcript comes from
`--voice-text NAME=<transcript or path>`, which is also repeatable and may
appear before or after its matching `--voice`:

```sh
./result/bin/gufo serve audio \
  --tts-model /persist/models/audio/Qwen3-TTS-12Hz-1.7B-Base \
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
selects a trained embedding and VoiceDesign is driven by `instruct`, so
neither has anything to apply them to.

`POST /v1/audio/speech` accepts `model`, `input`, `voice`, `response_format`,
`speed`, `language`, `instruct`, `seed`, `max_new_tokens`, `greedy`, and the
Base-only `reference_audio`, `reference_text`, and `voice_clone_mode`. Any
other field is rejected. `response_format` supports `wav` only and `speed`
supports `1.0` only; `input` is capped at 16384 UTF-8 bytes and
`max_new_tokens` at 8192 (default 3000). Output is 24 kHz mono 16-bit PCM.

Qwen3-ASR serving uses the same audio server, naming only the ASR checkpoint:

```sh
./result/bin/gufo serve audio \
  --asr-model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/<revision>
```

`POST /v1/audio/transcriptions` accepts OpenAI-compatible multipart fields
`file`, `model`, `language`, `prompt`, `response_format`, `temperature`, and
`max_tokens`. The native route is deterministic (`temperature=0`) and supports
`json`, `text`, and `verbose_json`; streaming and timestamp granularities are
rejected explicitly.

The MiniMax H3 subset follows the asynchronous OpenAI-style video resource
shape and is versioned independently as `gufo.video-api.v1`. Its supported
fields, frozen presets, queue behavior, and deliberate conditioning
omissions are documented in [MINIMAX_H3.md](MINIMAX_H3.md).
Create requests accept both `application/json` and OpenAI-client-compatible
`multipart/form-data`; duplicate form fields, malformed boundaries, unsupported
media types, and reference-image parts fail explicitly.
The production H3 contract supports both the rapid `512x512`, one-second MP4
route and the released `1344x768`, five-second exact route (124 aligned frames
at 24 fps).
The bounded worker retains prompt text only in volatile queued or active
request memory, persists only its SHA-256 digest, and wipes both source and
active string storage after transfer and completion. Completed MP4s are probed
for H.264/AAC codec, geometry, rates, duration, and A/V synchronization before
atomic publication.

## Internal Request Model

All public endpoints translate into one scheduler request:

```cpp
struct GenerationRequest {
    RequestId id;
    ModelId model;
    Prompt prompt;
    SamplingConfig sampling;
    OutputConstraints constraints;
    ToolConfig tools;
    Priority priority;
    Deadline deadline;
    bool stream;
};
```

Transport adapters may not directly call a model backend.

The prompt representation must preserve:

- System, developer, user, assistant, and tool roles where supported.
- Text and structured tool-call items.
- Model-specific chat-template version.
- Exact token IDs after tokenization.
- Request-level stop conditions.

## Responses API Subset

`POST /v1/responses` initially accepts:

- `model`
- `input` as text or a supported item array
- `instructions`
- `max_output_tokens`
- `temperature`
- `top_p`
- `stream`
- `text` for supported text and structured-output configuration
- `tools` and `tool_choice` for model implementations with validated tool support
- `metadata`, retained only for logging if enabled

The first release is stateless by default:

- `store` is accepted only when it is false; `store: true` is rejected.
- Server-side conversations and `previous_response_id` are rejected.
- Clients provide all context required for the request.

Non-streaming responses return one completed response object. Streaming uses
SSE and emits a stable subset of Responses streaming events, including:

```text
response.created
response.output_item.added
response.content_part.added
response.output_text.delta
response.output_text.done
response.completed
```

Errors terminate the stream with an error event when possible.

## Chat Completions Adapter

`POST /v1/chat/completions` accepts the common compatibility subset:

- `model`
- `messages`
- `max_tokens` or `max_completion_tokens`
- `temperature`
- `top_p`
- `seed`
- `stop`
- `stream`
- `stream_options.include_usage`
- `tools` and `tool_choice` when supported
- common frequency and presence penalties when implemented by the sampler

Streaming objects use `chat.completion.chunk` and end with the compatibility
sentinel expected by common clients.

The adapter must not implement a second inference path. It converts messages
into the same prompt and sampling structures used by `/v1/responses`.

## Model Discovery

`GET /v1/models` returns only usable model aliases. Each model entry includes
the standard identifier fields plus optional local metadata under a namespaced
extension:

```json
{
  "id": "qwen-current-27b",
  "object": "model",
  "created": 0,
  "owned_by": "local",
  "gufo": {
    "artifact_id": "qwen-27b-shq-t16-4p42bpw",
    "context_length": 131072,
    "quantization": "SHQ-T16",
    "bits_per_weight": 4.42,
    "tensor_encodings": ["SHQ4-T16", "SHQ8-T16", "BF16"],
    "backends": ["gfx1151", "xdna2"],
    "capabilities": ["responses", "chat", "tools"]
  }
}
```

Clients must not need the extension to use the model.

When several sizes of the same model are loaded, their public aliases must be
distinct and should expose the measured size, for example:

```text
qwen-27b-4.4bpw
qwen-27b-5.2bpw
qwen-27b-6.1bpw
```

Aliases are configuration choices; compatibility depends on the immutable
artifact ID returned in the `gufo` metadata.

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

Do not silently clamp context or output limits without returning the effective
limits in response metadata.

## Authentication and Exposure

Default binding should be loopback-only. Optional bearer-token authentication
uses:

```text
Authorization: Bearer <local-token>
```

Remote binding requires an explicit configuration flag. TLS should normally be
terminated by a local reverse proxy, though native TLS may be added later.

Prompt and generated text logging is disabled by default.

Tool definitions and generated tool calls are treated as data. `gufo`
never executes tools, shell commands, URLs, or generated code.

Browser access requires an explicit CORS origin allowlist. Do not enable a
wildcard origin on a remotely reachable authenticated server.

Metrics and future administrative endpoints may use a separate loopback
listener or a distinct administrative bearer token. They are not implicitly
exposed merely because the inference listener is remote.

## Backpressure and Cancellation

- Bound accepted request count, queued tokens, and active KV pages.
- Reject before tokenization when the admission queue is full.
- Stop scheduling new decode work immediately after cancellation.
- Reclaim provisional KV and speculative state transactionally.
- Permit an in-flight device step to finish if it cannot be safely interrupted.
- Never block an I/O thread waiting for a device event.

## Startup and Readiness

Process states are:

```text
STARTING
WARMING
READY
DEGRADED
DRAINING
FAILED
```

Startup performs:

1. Parse configuration and reject unknown fields.
2. Discover and fingerprint the Strix Halo GPU and NPU.
3. Initialize the allocation broker and safety reserves.
4. Initialize compiled backend programs and kernel registries.
5. Load each configured weight artifact into unpublished state.
6. Validate tensors, checksums, tokenizer, quantization, and state contracts.
7. Prefault required memory and create graph-stable allocations.
8. Run backend visibility, kernel, and model smoke tests.
9. Warm required single-request and configured batch routes.
10. Publish usable model aliases and become ready.

`GET /health` reports whether the process event loops and control plane are
alive. It does not imply that a model is usable.

`GET /ready` succeeds when at least one advertised model alias can admit work.
Its body reports:

- Process state.
- Loaded aliases.
- GPU and NPU availability.
- Disabled or degraded routes.
- Memory-pressure state.

If NPU initialization or recovery fails but a model has a validated GPU route,
the process enters `DEGRADED` and remains ready for GPU-only serving. A model
that requires an unavailable backend is omitted from `/v1/models`.

## Configuration

Configuration precedence is:

```text
compiled safe defaults
< configuration file
< explicitly supported command-line flags
```

Secrets such as bearer tokens may be read from protected files or environment
variables, but their values are never rendered by configuration diagnostics.

Reject:

- Unknown configuration keys.
- Duplicate model aliases.
- Unsupported model kinds.
- Relative or escaping model paths where policy forbids them.
- Unsafe memory budgets below the platform minimum without an explicit
  development override.
- Remote binding without the remote-exposure flag.

The effective non-secret configuration is available in startup logs and
diagnostic output.

## Model Replacement

Weight replacement for a supported model kind is transactional:

1. Resolve and validate the candidate artifact.
2. Reserve its complete load and warmup budget.
3. Load it under an unpublished internal ID.
4. Run required smoke and compatibility tests.
5. Atomically move the public alias to the candidate.
6. Drain sessions using the previous artifact.
7. Release previous weights after the final reference retires.

The server never changes the weights underneath an active session. Rollback is
available only when memory for the previous artifact was deliberately retained.

Replacement is initiated through a local administrative mechanism, not the
public OpenAI API.

## Graceful Shutdown

On `SIGTERM` or an administrative shutdown:

1. Enter `DRAINING`.
2. Stop accepting new inference requests.
3. Finish or cancel queued work according to configured grace time.
4. Let uninterruptible device operations retire.
5. Commit only transitions completed before cancellation.
6. Flush bounded response and metric buffers.
7. Finish or abort snapshot writes transactionally.
8. Destroy graphs, NPU contexts, queues, and allocations in dependency order.

A second termination signal or expired hard deadline performs an immediate
best-effort shutdown without publishing partial snapshots or state.

After system suspend/resume or a backend reset, the server recreates affected
contexts and imported allocations, reruns visibility and smoke tests, and keeps
the route disabled until validation succeeds.

## Metrics

At minimum expose:

- Time to first token.
- Inter-token latency.
- Prompt and generated tokens per second.
- Queue and admission delay.
- Active, queued, cancelled, and failed requests.
- GPU-only, NPU-only, and heterogeneous route counts.
- Scheduler batch width and padding.
- KV pages used, shared, evicted, dumped, and restored.
- Speculative proposed, verified, accepted, and committed tokens.
- Per-backend execution and synchronization time.

Metrics labels must use bounded model and route identifiers. Do not place
request IDs or prompt text in labels.

## Conformance Tests

- Golden JSON request/response fixtures for every supported field.
- Official-client smoke tests against a local base URL.
- Streaming chunk ordering and disconnect tests.
- Chat Completions and Responses token parity.
- Tool-call and structured-output fixtures.
- Unknown and unsupported parameter tests.
- Queue saturation, timeout, and cancellation tests.
- Direct runtime versus HTTP exact-token tests for greedy generation.
- Startup failure at every initialization stage.
- GPU-only degraded readiness after NPU failure.
- Model replacement, drain, alias switch, and rollback.
- Graceful and forced shutdown with active requests.
- Suspend/resume and backend-context reconstruction.
- Scheduler stale-completion rejection.
- Configuration and secret-redaction tests.
