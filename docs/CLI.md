# Command-Line Interface

Status: design draft, 2026-08-11

## Purpose

The `strix` executable also provides terminal interfaces for:

- Interactive conversations.
- One-shot prompt testing.
- Piped and scripted generation.
- Testing a locally loaded runtime without HTTP.
- Testing an already running OpenAI-compatible server.

The CLI is a transport adapter over the same request, scheduler, tokenizer,
sampling, and model implementations used by the HTTP server. It must not
contain a second inference path.

## Commands

```text
strix serve
strix chat
strix prompt
strix diagnose
```

`serve` starts the OpenAI-compatible server. `chat` maintains an interactive
conversation. `prompt` executes one request and exits. `diagnose` runs
non-interactive system and hardware diagnostics.

All commands support `--help` and `--version`. Unknown options and invalid
combinations return an error instead of being ignored.

## Execution Modes

`chat` and `prompt` support two mutually exclusive execution modes.

### Direct mode

Direct mode loads the configured model in the current process:

```bash
strix prompt \
  --config strix.toml \
  --model qwen-current-27b \
  --prompt "Explain wave32 in three sentences."
```

The CLI constructs the same internal `GenerationRequest` used by the server and
submits it to the same scheduler. Direct mode is useful for kernel development,
logit checks, and isolating HTTP from inference failures.

### Client mode

Client mode sends a request to a running Strix-Halo.cpp server:

```bash
strix prompt \
  --connect http://127.0.0.1:8080 \
  --model qwen-current-27b \
  --prompt "Explain wave32 in three sentences."
```

Client mode uses `/v1/responses` by default. It validates the public API,
streaming behavior, cancellation, and usage accounting.

`--config` and `--connect` are mutually exclusive. A configured default may be
used, but the effective mode must be printed by verbose diagnostics.

## One-Shot Prompts

The prompt may be supplied as a named option:

```bash
strix prompt --config strix.toml --model qwen-current-27b \
  --prompt "Write a JSON object with the keys name and value."
```

It may also be the final positional argument:

```bash
strix prompt --config strix.toml --model qwen-current-27b \
  "Summarize how paged KV caches work."
```

The two forms are mutually exclusive.

Additional input forms:

```bash
strix prompt --config strix.toml --model qwen-current-27b \
  --prompt-file prompt.txt

printf '%s\n' "Explain INT4 zero points." |
  strix prompt --config strix.toml --model qwen-current-27b --stdin
```

Exactly one of the following may provide the user prompt:

- `--prompt`
- One positional prompt
- `--prompt-file`
- `--stdin`

An empty prompt is rejected unless `--allow-empty-prompt` is explicitly used
for a model-specific test.

## Interactive Chat

Start a direct interactive session:

```bash
strix chat --config strix.toml --model qwen-current-27b
```

Or connect to a running server:

```bash
strix chat \
  --connect http://127.0.0.1:8080 \
  --model qwen-current-27b
```

The terminal displays a small prompt marker and streams assistant text as it is
generated. The complete conversation history is submitted on subsequent turns
unless the selected server capability provides an explicitly compatible
session mechanism.

Initial interactive commands:

| Command | Behavior |
| --- | --- |
| `/help` | Show available terminal commands |
| `/clear` | Clear conversation history and reset local session state |
| `/system TEXT` | Replace the system instruction for future turns |
| `/stats` | Show timing, token, route, and cache statistics for the last turn |
| `/model` | Show the selected model and capabilities |
| `/save PATH` | Save the conversation as UTF-8 JSON |
| `/load PATH` | Load a compatible saved conversation |
| `/cancel` | Cancel an active generation when input handling permits it |
| `/exit` | End the session cleanly |

`Ctrl-C` cancels the current generation. A second `Ctrl-C` exits. End-of-file
exits after flushing the terminal without adding a synthetic user message.

Terminal commands are handled only when entered as the first non-whitespace
content of a new input turn. User text can escape a leading slash with `//`.

## Common Generation Options

Both `chat` and `prompt` support:

```text
--model ALIAS
--system TEXT
--max-output-tokens N
--temperature VALUE
--top-p VALUE
--top-k N
--seed N
--stop TEXT
--greedy
--json-schema PATH
--no-stream
```

Rules:

- `--greedy` selects the canonical greedy sampling contract and rejects
  conflicting sampling options.
- Repeated `--stop` options create multiple stop sequences.
- `--json-schema` uses the same bounded structured-output implementation as
  the server.
- Unsupported model capabilities fail before generation.
- Defaults come from the same validated configuration types as HTTP requests.

The CLI does not execute model-generated tool calls. It may display their
structured representation.

## Output Modes

### Human terminal

When stdout is an interactive terminal:

- Stream assistant text by default.
- Keep diagnostics and statistics on stderr.
- Use color only when enabled and supported.
- Never allow ANSI styling to enter saved conversations or model input.
- Finish generated text with one terminal newline without changing the
  underlying token output.

### Plain text

`--output text` writes only generated text to stdout. This is the default when
stdout is redirected.

### JSON

`--output json` emits one complete JSON object containing:

```json
{
  "model": "qwen-current-27b",
  "text": "Generated output",
  "finish_reason": "stop",
  "usage": {
    "input_tokens": 12,
    "output_tokens": 8
  },
  "strix": {
    "route": "GPU_ONLY",
    "time_to_first_token_ms": 18.2,
    "tokens_per_second": 21.4
  }
}
```

`--output jsonl` emits bounded streaming events as one JSON object per line.
Machine-readable output never contains progress bars, terminal control
sequences, or human diagnostics.

## Diagnostics

Optional flags:

```text
--verbose
--stats
--show-token-ids
--show-logprobs N
--trace-output PATH
```

Diagnostics are sent to stderr or the explicitly selected trace file.
`--show-logprobs` is enabled only when the selected implementation exposes the
required logits without changing the normal sampling result.

Direct mode reports:

- Model artifact and implementation IDs.
- GPU, NPU, or heterogeneous route.
- Prompt evaluation time.
- Time to first token.
- Inter-token latency and output tokens per second.
- Peak and current memory by relevant class.
- Prefix-cache and graph hits.
- Speculative acceptance statistics when applicable.

## System Diagnostics

`strix diagnose` executes non-interactive platform, system, toolchain,
GPU, and NPU diagnostics without starting a server or loading a model:

```bash
strix diagnose
strix diagnose --json
strix diagnose --json --section inventory
strix diagnose --fingerprint --json --output /tmp/strix-fingerprint.json
strix diagnose --validate-artifact /tmp/strix-fingerprint.json
strix diagnose --benchmark bandwidth --backends cpu,hip,xrt --output /tmp/strix-bandwidth.json
strix diagnose --smoke xrt --iterations 100 --timeout-ms 30000 \
  --json --output /tmp/strix-xrt-smoke.json
```

### Machine Fingerprint and Artifact Validation

- `--fingerprint`: Emits a canonical machine fingerprint object with a 64-character SHA-256 identity hash computed over CPU topology, gfx1151 GPU identity, XDNA2 NPU identity, kernel drivers, and pinned toolchain versions.
- `--validate-artifact <path>`: Validates a benchmark or diagnostic JSON artifact against schema v1.0.0, verifies fingerprint SHA-256 integrity, and checks architecture requirements (`gfx1151`, `XDNA2`).
- `--output <path>`: Writes command output to the specified file path.
- `--benchmark <name>`: Runs a diagnostic benchmark suite (`bandwidth`).
- `--backends <csv>`: Comma-separated list of backends to benchmark (`cpu`, `hip`, `xrt`).
- `--warmup <n>`: Number of warmup iterations (default: `3`).
- `--repetitions <n>`: Number of benchmark repetitions (default: `10`).
- `--duration-ms <ms>`: Target duration per test in milliseconds (default: `2000`).
- `--smoke xrt`: Executes the packaged deterministic XDNA2 program.
- `--iterations <n>`: Number of command/completion cycles (default: `100`).
- `--timeout-ms <n>`: Per-command timeout; a timeout fails and quarantines the context.

### JSON Output Schema (v1.0.0)

When `--json` is supplied, `diagnose` emits structured JSON output to stdout:

```json
{
  "schemaVersion": "1.0.0",
  "engineRevision": "0.1.0",
  "timestamp": "2026-08-17T20:00:00Z",
  "status": "PASS",
  "fingerprintId": "e124845f98b4db093157486a779156beabb0dd32438945d5aa0e0c4c0baf89e5",
  "inventory": {
    "cpu": {
      "modelName": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
      "architecture": "x86_64",
      "logicalCores": 32,
      "physicalCores": 16
    },
    "gpu": {
      "name": "AMD Radeon 8060S Graphics",
      "architecture": "gfx1151",
      "computeUnits": 40,
      "driverName": "amdgpu"
    },
    "npu": {
      "identity": "AMD XDNA2 NPU",
      "architecture": "XDNA2",
      "pciDeviceId": "1022:17f0",
      "driverName": "amdxdna"
    }
  },
  "compatibility": {
    "overallVerdict": "supported",
    "items": []
  },
  "checks": [
    {
      "name": "platform",
      "status": "PASS",
      "message": "Supported architecture: Linux x86-64",
      "details": {
        "targetArchitecture": "x86_64-linux"
      }
    }
  ],
  "warnings": [],
  "errors": []
}
```

The top-level `status` reflects overall health: `PASS` when all checks pass,
`WARN` when non-fatal hardware/device warnings exist, and `FAIL` when a critical
platform/system failure occurs.

Diagnostic sections can be selected using `--section <name>` (`all`, `inventory`,
`platform`, `gpu`, `npu`).

## Conversation Files

Saved conversations use versioned UTF-8 JSON and contain logical messages, not
raw KV data:

```json
{
  "format": "strix-conversation-v1",
  "model": "qwen-current-27b",
  "chat_template_id": "qwen38-chat-v1",
  "system": "You are concise.",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello."}
  ]
}
```

Loading validates:

- Format version.
- Roles and content limits.
- Model and chat-template compatibility.
- UTF-8 validity.
- Maximum file and conversation size.

Conversation files are untrusted input. They cannot contain executable
templates, tools, plugins, device programs, or filesystem directives.

KV snapshots remain governed by the separate KV persistence format. A
conversation JSON file is portable but requires prompt re-evaluation.

## Exit Codes

```text
0   completed successfully
2   command-line or configuration error
3   model or capability unavailable
4   request rejected by admission control
5   generation or backend failure
6   API, connection, or authentication failure
7   incompatible or corrupt input artifact
130 cancelled by interrupt
```

Partial generated text may already have been written before a streaming
failure. JSON output includes an error object when it can still remain valid;
otherwise the nonzero exit status is authoritative.

## Tests

- Named, positional, file, and stdin prompt input.
- Rejection of multiple prompt sources.
- Direct CLI versus direct runtime exact-token parity.
- Client CLI versus HTTP fixture parity.
- Greedy direct mode versus client mode parity.
- Interactive history and `/clear`.
- System-instruction replacement.
- UTF-8 split across streaming chunks.
- Stop strings spanning token boundaries.
- `Ctrl-C` cancellation and resource reclamation.
- TTY, redirected text, JSON, and JSONL output.
- Broken pipe handling when a downstream process exits.
- Conversation save/load validation and corruption.
- Stable exit codes.
- Secret and prompt redaction in diagnostics.
- Single-user latency equivalent to the server fast path.
