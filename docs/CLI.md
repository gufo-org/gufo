# Command-Line Interface

The `gufo` executable provides terminal interfaces for:

- Interactive conversations.
- One-shot prompt testing.
- Piped and scripted generation.
- Testing a locally loaded runtime without HTTP.
- Testing an already running OpenAI-compatible server.

Terminal generation and HTTP serving share model runtimes, tokenizers, and
sampling. The HTTP scheduler also manages concurrent requests and persistent
prompt caching.

## Functionalities

This is an exhaustive list of functionalities and features offered by the gufo cli, the mapping between features and flags is not documented here and should be discovered using the `--help` command to avoid divergences between the implementation and the documentation.

The supported modalities are LLM, image, video, and audio (TTS, ASR).
`gufo serve image` provides [Qwen-Image-2.1 generation and editing](models/qwen-image-2.1/README.md).

### LLM or text

**Model identity**

- `servedModelName` — the name the server reports to clients (for example in
  `/v1/models`). In practice: what your client's "model" field should say.
  Purely cosmetic, it does not change which weights are loaded.

**Size and memory**

- `context` — maximum context capacity; memory use depends on the model and
  the number of resident sessions.
- `prefillChunk` — prompt tokens processed between active decode steps in
  HTTP serving. Model runtimes choose their own internal prefill batch sizes.
- `maxTokens` — maximum number of tokens a request may generate. In practice:
  stops a model that rambles forever.

**Sampling (how the next token is picked)**

- `temperature` — randomness of the choice. Low (for example
  0.2) gives predictable, repetitive answers; high (1.0+) gives varied,
  creative ones. Greedy decoding is the low extreme.
- `topP` — only consider the smallest set of tokens whose probabilities add
  up to P (nucleus sampling). In practice: a "keep only plausible tokens"
  filter; lower means safer output.
- `topK` — only consider the K most likely tokens. The same idea as `topP` but
  with a fixed count instead of a probability mass.
- `minP` — discard tokens whose probability is below a fraction of the best
  token's probability. A modern alternative to `topP`/`topK` that stays
  consistent across models.
- `minKeep` — always keep at least N candidates after those filters. In
  practice: prevents the filters from ever leaving just one choice, so the
  sampler keeps working as intended.
- `seed` — fixed random seed. Same seed plus same prompt gives the same output,
  which makes runs reproducible.
- `repeatPenalty` — makes tokens that already appeared less likely. In
  practice: the anti-loop knob, stops "the the the the".
- `repeatLastN` — how far back the repeat penalty looks. Larger windows catch
  loops that build up over long outputs.
- `frequencyPenalty` — a penalty that grows the more often a token has already
  been used. Reduces word-for-word repetition.
- `presencePenalty` — a flat penalty for any token used at least once. Pushes
  the model toward new words and topics even if it used each only once.

**Reasoning models**

- `think` (`"on"` / `"off"` / `"auto"`) — whether the model produces its
  visible thinking before answering. In practice: off for quick chats, on
  for hard problems, auto lets the model decide per request.
- `reasoningEffort` (`"auto"`, `"minimal"`, `"low"`, `"medium"`, `"high"`,
  `"xhigh"`, `"max"`) — how long the model is allowed to reason before
  answering. More effort usually means better answers on hard tasks, at the
  cost of slower responses and more generated tokens.
- `preserveThinking` (`"on"` / `"off"` / `"auto"`) — whether earlier turns'
  thinking is kept in the conversation history. In practice: keeping it can
  help follow-up questions that build on the previous reasoning, at the cost
  of eating context.

**Disk cache**

- `cacheDisk` — enable an on-disk cache of computed prompt state, reused
  across requests and restarts. In practice: repeated or shared prompts get
  much faster; costs disk space.
- `cacheDiskBytes` — the maximum size of that cache on disk.
- `cacheDiskStagingBytes` — scratch space used while writing new cache
  entries. Must fit alongside `cacheDiskBytes`.

**Speculative decoding**

A draft model proposes tokens for the target model to verify. Greedy decoding
must reproduce the target's tokens; sampled decoding must preserve the target's
distribution, but need not match an autoregressive run's sequence for the same
seed. Speed depends on acceptance and verification cost.

- `speculative` — `dflash2` for Qwen3.8-27B, `dspark` for DeepSeek V4 Flash,
  `mtp` for models with a supported MTP head, or `off`. HTTP serving
  supports `dflash2`, `dspark`, and Flash-Next `mtp`.
- `dflashModel` — path to the DFlash2 companion draft.
- `dsparkModel` — path to the DSpark support GGUF.
- `mtpModel` — path to the MTP draft model.
- `draftTokens` — maximum tokens drafted per step. Wider drafts go faster
  when the guesses are accepted and waste work when they are rejected.
- `draftPolicy` (`"fixed"` / `"adaptive"`) — DFlash2 block-length policy;
  defaults to adaptive.
- `minDraftTokens` — minimum draft length where supported. DFlash2, DSpark,
  and Flash-Next MTP require the default of one.

**Server limits (protecting the machine from clients)**

- `maxPending` — maximum requests waiting in the queue. Over the limit,
  clients wait or are refused rather than exhausting memory.
- `maxPendingPerClient` — the same cap but per client, so one client cannot
  hog the whole queue.
- `requestTimeoutMs` — requests are killed after this many milliseconds. In
  practice: prevents stuck requests from holding GPU sessions forever.
- `maxOutputBytes` — maximum response size per request.
- `maxBufferedOutputBytes` — maximum generated-but-not-yet-delivered output
  buffered for one request; protects against slow clients.
- `maxBufferedOutputTotal` — the same buffer budget across all requests.

**Hardware**

- `cpu` — CPU reference execution for supported models in `prompt` and
  `chat`; unavailable in HTTP serving.

### Video

TODO

### Audio (TTS and ASR)

Use `gufo serve tts --model DIR --context 4096` for synthesis or
`gufo serve asr --model DIR --context 1024` for transcription. Each process
loads one checkpoint; llama-swap can route both services under one API URL.
The Nix helper uses the same `modality`, `model`, `context`, and
`servedModelName` fields.

`--context` reserves the model's token capacity. ASR applies that capacity
per audio chunk, so long files do not require one enormous context. The
standalone `gufo transcribe` command also uses `--model` and `--context`;
`--prompt` supplies optional transcription hints.

**Voices**

- `voices` — named voice presets built from reference WAVs. Each entry is
  either a bare reference WAV (its transcript is read from a `.txt` sidecar
  file beside it) or the `--voice` / `--voice-text` / `--voice-lang` triple:
  the reference WAV, its transcript (inline text or a path to a file holding
  it), and an optional default language for that voice. In practice: clients
  pick a voice by name in the request, for example `{"voice":
"narrator_ita"}`, without shipping audio; the optional language means the
  caller does not need to repeat it. Voices require a TTS checkpoint.

```sh
gufo serve tts \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator_eng=/audio/clear-english-voice.wav \
  --voice-text narrator_eng=/audio/clear-english-voice.txt \
  --voice narrator_ita=/audio/clear-italian-voice.wav \
  --voice-text narrator_ita="Questo racconto e' cresciuto..." \
  --voice-lang narrator_ita=italian
```

### Server options (all modalities)

These apply to every modality, before or after the subcommand:

- `host` / `port` — where to listen (`-i` / `-p`). In practice: set
  `--host 0.0.0.0` if you serve from inside a container, or the published
  port will not answer.
- `sessions` (`-j`) — preallocated GPU request sessions. In practice: more
  sessions means more requests can compute at once, at the cost of memory
  per session. This option applies to LLM serving; image and speech servers
  use their own bounded queues.
- `maxConnections` — maximum simultaneous HTTP connections. In practice:
  clients beyond the limit queue or are refused instead of piling up.
- `maxRequestBytes` — maximum request body size. In practice: matters mostly
  for ASR, since requests carry whole audio files (the audio benchmark used
  32 MiB).
- `apiKey` — if set, requests require `Authorization: Bearer <key>`, including
  health checks. Browser CORS preflight (`OPTIONS`) remains unauthenticated.
- `verbose` — chattier logs.

## Commands

```text
gufo serve
gufo chat
gufo prompt
gufo eval
gufo bench
gufo video
gufo transcribe
gufo diagnose
```

`serve` starts the OpenAI-compatible server. `chat` maintains an interactive
conversation. `prompt` executes one request and exits. `eval` grades the pinned
DS4 capability questions through an already running OpenAI-compatible server.
`bench` measures model execution, `video` runs MiniMax H3 generation,
`transcribe` runs native Qwen3-ASR-1.7B speech recognition, and `diagnose` runs
non-interactive system and hardware diagnostics.

All commands support `--help` and `--version`. Unknown options and invalid
combinations return an error instead of being ignored.

### Server mode

Gufo can be run as a server exposing OpenAI-compatible API HTTP endpoints, activated based on the model served.

```sh
gufo serve --host 0.0.0.0 --port 8080 llm --model /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf
# or
gufo serve tts \
  --model ./models/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator_ita=./audio/clear-italian-voice.wav \
  --voice-text "narrator_ita=Questo racconto e' cresciuto..." \
  --voice narrator_eng=./audio/clear-english-voice.wav \
  --voice-text narrator_eng=./audio/clear-english-voice.txt
```

Several flags are offered by the command, they won't be documented here to avoid having the out of sync. If you need more information please use the help command

```sh
gufo serve --help
# or
gufo serve llm --help
```

### Diagnose

Gufo can self detect if everything is correctly configured to be able to run, such as drivers, groups, permissions, etc.

### Benchmarks and Evaluations

Two utilities are shipped with gufo to quickly verify the speed of a model (`bench`) and the accuracy of it (`eval`).
