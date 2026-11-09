# Server

`gufo serve` exposes a model over HTTP with the OpenAI API, so existing clients
and SDKs work by changing their base URL. It also speaks Anthropic Messages and
the llama-server routes that tools such as llama-swap rely on.

## Start a server

```sh
gufo serve --port 8080 llm --model "$MODEL" --served-model-name qwen
curl -s http://127.0.0.1:8080/ready
```

While the model loads, every route answers 503 with `Retry-After: 1`; `/ready`
returns 200 once requests can run. Point a client at the server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen", messages=[{"role": "user", "content": "Hello"}])
print(reply.choices[0].message.content)
```

Gufo ignores API keys. Supported fields behave as documented here; fields Gufo
can't honor return 400 instead of being silently dropped. The
[server reference](internal/server.md) lists every field.

## Endpoints

| Route | Use |
| --- | --- |
| `POST /v1/chat/completions` | Chat, streaming, images, tools and structured output. The main API. |
| `POST /v1/responses` | The Responses API: text, images, function tools, streaming. |
| `POST /v1/messages` | Anthropic Messages: text, images, thinking and custom tools. |
| `POST /v1/completions`, `/completion` | Plain text completion from one prompt. |
| `POST /v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/v1/messages/count_tokens` | Count a prompt's tokens without running it. |
| `POST /tokenize` | Get token IDs for raw text. |
| `GET /v1/models` | The served model, its context length and whether it accepts images. |
| `GET /health`, `GET /ready` | Liveness and readiness. |
| `GET /metrics`, `/slots`, `/props` | Prometheus metrics and llama-server status. |

Each other modality has its own routes; see [other model services](#other-model-services).

## Chat Completions

Chat Completions accepts the common fields: `messages`, `max_tokens` or
`max_completion_tokens`, `temperature`, `top_p`, `seed`, `stop` (up to four
strings), `stream`, `stream_options.include_usage`, `tools`, `tool_choice`,
`response_format` and `logprobs`, plus llama.cpp's `top_k`, `min_p` and
repetition penalties.

`system` and `developer` messages can appear anywhere in the conversation, as
agent clients such as Codex send them. Responses stream as
`chat.completion.chunk` events, and the last chunk carries llama.cpp-style
`timings`: `prompt_n` tokens newly processed and `cache_n` tokens reused from
the [cache](cache.md). Set `return_progress: true` to receive prefill progress
while a long prompt is processed.

## Responses and Messages

The Responses API takes the whole conversation in every request: Gufo stores
nothing, so `store` and `background` must be false and `previous_response_id`
isn't supported. Replay earlier output items, including `function_call` and
`reasoning` items, as they came back. Hosted tools such as `web_search` are
skipped so the function tools still reach the model.

Anthropic Messages supports `thinking` blocks, custom tools and images. Replay
`thinking` blocks unchanged so the next turn reuses the cached prompt. Server
tools such as `web_search` are rejected.

## Reasoning controls

Models that think before answering have a default; `--think` and
`--reasoning-effort` change it for the server, and a request changes it for
itself:

| Control | Chat Completions | Responses | Messages |
| --- | --- | --- | --- |
| Thinking off | `reasoning_effort: "none"` or `chat_template_kwargs.enable_thinking: false` | `reasoning.effort: "none"` | `thinking.type: "disabled"` |
| Effort | `reasoning_effort` | `reasoning.effort` | `output_config.effort` |
| Token budget | `reasoning_budget_tokens` | `reasoning_budget_tokens` | `thinking.budget_tokens` |

Effort is one of `minimal`, `low`, `medium`, `high`, `xhigh` and `max`; it is an
instruction to the model, not a limit. The budget forces the end of reasoning
after that many tokens; `--reasoning-budget` sets the server default.

Reasoning comes back separately from the answer: as `reasoning_content` in
Chat Completions, `reasoning` items in Responses and `thinking` blocks in
Messages. Keep the effort the same across a conversation: it's part of the
prompt, so changing it means processing the whole conversation again.

## Tools and structured output

Function tools follow the Chat Completions or Responses shape. With
`tool_choice: "auto"` the model chooses between a call and an answer;
`"required"` or a named function forces a call, and Gufo constrains decoding so
the call is well formed. `strict: true` on a function also constrains its
arguments to the schema. Gufo never runs tools: calls are returned to your
client.

`response_format` (Chat) or `text.format` (Responses) set to `json_object` or
`json_schema` constrains the answer to JSON, in every decoding mode and with
streaming. The official SDK's `parse()` helpers work. Most of JSON Schema is
supported, including `$ref`, `anyOf`, string formats and patterns; unsupported
keywords return 400. Always parse the result: a token limit or stop sequence can
cut the JSON short.

## Images

Vision models accept images as `image_url` parts in Chat Completions,
`input_image` in Responses and `image` blocks in Messages, also inside tool
results. Send PNG, JPEG or WebP as a data URL or a public HTTPS URL; Gufo
refuses private and loopback addresses.

All images in a request share 20 MiB of encoded data and 15 seconds of
downloading, and each can have up to 32 megapixels. There's no fixed number of
images, but their tokens must fit the context. `detail` (`low`, `high`,
`original` or `auto`) trades detail for tokens where the model supports it; see
the model's page.

## Embeddings

`gufo serve embedding` answers `POST /v1/embeddings`, and llama.cpp's
`POST /embeddings` for clients that use it. Inputs, limits and task prefixes
are on the [EmbeddingGemma 2](models/embeddinggemma-2.md) page.

## Videos

Video generation is asynchronous, as in the OpenAI video API:

1. `POST /v1/videos` creates a job and returns its `id`.
2. `GET /v1/videos/{id}` reports its status and progress.
3. `GET /v1/videos/{id}/content` downloads the MP4 once the job completes,
   with HTTP range requests.
4. `DELETE /v1/videos/{id}` cancels the job and deletes its files.

Jobs and finished videos survive a server restart until `--ttl` expires them.
Request fields and limits are on the [MiniMax H3](models/minimax-h3.md) page.

## Other model services

| Command | Routes | Model page |
| --- | --- | --- |
| `gufo serve asr` | `POST /v1/audio/transcriptions`, WebSocket `/v1/realtime` | [Qwen3-ASR](models/qwen3-asr.md) |
| `gufo serve tts` | `POST /v1/audio/speech`, `GET /v1/audio/voices`, WebSocket `/v1/audio/speech/stream` | [Qwen3-TTS](models/qwen3-tts.md) |
| `gufo serve image` | `POST /v1/images/generations`, `POST /v1/images/edits` | [Qwen-Image-2.1](models/qwen-image-2.1.md) |
| `gufo serve decision` | `POST /v1/systemone` | Defined by the model |

## Model names

`GET /v1/models` lists the IDs the server answers to. Repeat
`--served-model-name` to add aliases; the first is the primary name. A request
for any other model returns 404, or 400 `invalid_model` on audio and video
routes.

## Errors

Errors use the OpenAI error object:

```json
{"error": {"message": "Unsupported field: logit_bias", "type": "invalid_request_error",
           "param": "logit_bias", "code": "unsupported_parameter"}}
```

| Status | Meaning |
| --- | --- |
| 400 | Invalid request or unsupported field |
| 404 | Unknown model or object |
| 413 | Request or prompt too large |
| 429 | Queue full; retry later |
| 503 | Model loading, or the GPU was lost (`device_lost`) |

A stream that fails after it started ends with an error event instead.

## Health and restarts

`/health` says the process is alive and `/ready` that the model can serve.
Neither does GPU work when polled.

When the GPU driver resets, the process can't recover. Gufo answers 503
`device_lost`, logs `event=device_lost`, and exits with status 75. Run it under
a supervisor that restarts on failure, such as systemd's
`Restart=on-failure` or a container restart policy. `--stall-timeout-ms` turns
a hung GPU, which fails nothing, into the same exit.

## Metrics

`/metrics` exports llama-server's Prometheus counters: tokens processed and
generated, cached prompt tokens, speeds, queue depth and speculative
acceptance. `/slots` shows what each session is doing, `/props` which model,
build and settings are running, and `POST /apply-template` the prompt a chat
request becomes. [Troubleshooting](troubleshooting.md) covers the logs.

## llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) starts Gufo on demand
and puts several models behind one URL. Use `/ready` as its check:

```yaml
models:
  gufo-model:
    aliases: [gufo-fast]
    cmd: >
      /path/to/gufo serve --host 127.0.0.1 --port ${PORT} llm
      --model /path/to/model.gguf
      --served-model-name gufo-model --served-model-name gufo-fast
    checkEndpoint: /ready
```

Pass one `--served-model-name` per llama-swap name or alias, or set
llama-swap's `useModelName`. Raise `healthCheckTimeout` for large models,
because loading and reading the disk cache take a while.

## Exposure

> **Warning:** Gufo has no authentication. Anyone who can reach the port can
> run the model and read its metrics. It listens on `127.0.0.1` by default;
> before changing `--host`, put a reverse proxy with TLS and access control in
> front.

Gufo never logs prompts or replies, and never runs tool calls, commands or
generated code. Only the opt-in [`--trace`](troubleshooting.md#content-trace)
file records content.
