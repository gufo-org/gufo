# Capability Evaluation

## Purpose

`gufo eval` runs a fixed, answer-graded capability suite against an already
running OpenAI-compatible text server. It is a regression harness for complete
model responses, not a leaderboard runner and not an official GPQA,
SuperGPQA, AIME, or COMPSEC score.

The evaluator is only an HTTP client. It does not load a model, initialize a
Gufo backend, or provide a second inference path.

## Pinned Question Set

The committed data is [the unified DS4 fixture](../tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json).
It was imported from `antirez/ds4`:

| Field | Identity |
| --- | --- |
| Repository | `https://github.com/antirez/ds4` |
| Revision | `84cc882352757baf628a1776badf7cc54d584e28` |
| File | `ds4_eval.c` |
| Git blob | `7aed5d5c5b5cdc74d1b3aa0310161e2b437c1d08` |

The 75 redistributable rows preserve DS4's exact interleaved order:

1. GPQA Diamond row.
2. Audited SuperGPQA row.
3. AIME 2025 row.
4. Repeat until each source contributes 25 rows.

Every row records its DS4 array index, source ID, domain, title, answer form,
question, choices, expected answer, dataset URL, license, audit note, and a
canonical source-record hash. The first four expected answers are `B`, `C`,
`70`, and `C`.

DS4's final 17 COMPSEC prompts remain uncommitted while their redistribution
provenance is reviewed. The fixture retains metadata-only records containing
their DS4 indices, IDs, domains, and accepted line sets.

To reproduce the import from a DS4 checkout:

```sh
python3 tools/ds4/import-eval.py \
  --ds4-repository /path/to/ds4 \
  --output tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json
```

## Request Contract

The evaluator discovers the model with `GET /v1/models`. Exactly one model
must be returned; zero or multiple models fail before generation.

For each selected case, `gufo eval` sends one independent
`POST /v1/chat/completions` request, sequentially, with:

- DS4's exact benchmark system message.
- DS4's exact rendered user message.
- The discovered model ID.
- `max_completion_tokens: 16000`.
- `stream: false`.

By default the request omits `temperature`, so sampling and thinking behavior
come from the server configuration. `--greedy` adds exactly
`temperature: 0`. The initial evaluator sends no reasoning-effort or
thinking-mode parameter.

The evaluator does not:

- Force-close a thinking block.
- Preflight context capacity.
- Clear prefix caches between cases.
- Introduce concurrent users.
- Apply a benchmark wall-clock generation cutoff.

Connection and transport failures remain infrastructure errors.

## Command

```sh
gufo eval \
  --base-url http://127.0.0.1:8080/v1 \
  --output /tmp/gufo-eval.json

gufo eval \
  --questions 4 \
  --greedy \
  --output /tmp/gufo-eval-greedy.json
```

Initial options:

| Option | Behavior |
| --- | --- |
| `--base-url URL` | API root; defaults to `http://127.0.0.1:8080/v1` |
| `--questions N` | Run the first `N` pinned rows; defaults to all 75 |
| `--greedy` | Send `temperature: 0`; otherwise omit `temperature` |
| `--output PATH` | Required sanitized JSON result |

There is no initial `--model` or `--suite` option. An API key may be supplied
through `OPENAI_API_KEY` or `GUFO_EVAL_API_KEY`; it is never written to the
result.

## Grading

The pure text extractors support:

- `mcq`: a final option letter, including DS4's wrapped and rejection forms;
- `integer`: a normalized integer, including leading zeros and final
  arithmetic expressions;
- `linespec`: a line number, list, or range.

No answer and genuinely ambiguous alternatives fail closed. For COMPSEC-style
line sets, every submitted line must belong to the audited accepted set.

Wrong answers are normal benchmark results. They do not make the evaluator
process fail. Transport, HTTP, context-rejection, and malformed-response errors
are execution failures and produce a nonzero exit status after the report is
written.

## Result Artifact

The JSON report records:

- evaluator commit and clean/dirty state;
- pinned DS4 repository, revision, path, and blob;
- sanitized command and request policy;
- server label/commit/configuration and model artifact/revision when supplied;
- exact submitted system and user messages;
- visible response text and separated `reasoning_content`;
- extracted answer, expected answer, and verdict;
- finish reason and execution classification;
- prompt, completion, cached-token, queue, and timing numbers when published;
- aggregate pass, failure, execution-error, length-finish, and generated-token
  counts.

Only allowlisted model metadata and numeric telemetry are copied. Credentials,
base URLs, private addresses, home paths, and local user paths are excluded or
redacted. Optional run identities can be provided with:

```sh
export GUFO_EVAL_SERVER_LABEL=gufo
export GUFO_EVAL_SERVER_COMMIT=<git-commit>
export GUFO_EVAL_SERVER_CONFIG='gufo serve --context 32768 llm --model <artifact>'
export GUFO_EVAL_MODEL_REVISION=<model-revision>
export GUFO_EVAL_ARTIFACT_ID=<artifact-hash-or-id>
```

The Gufo repository commit is the benchmark-data identity; there is no
separate manually maintained suite version.

## First Regression Baseline

The canonical small run uses the first four DS4 cases against the real
`gufo serve` OpenAI-compatible route:

```sh
gufo eval --questions 4 --output /tmp/gufo-eval.json
```

It uses server-default sampling and thinking behavior. A separate `--greedy`
run may be retained for diagnosis, but it does not replace the default-mode
baseline.

The retained report is a Gufo regression baseline. Score variance across
repeated runs is reported rather than hidden, and no serving-throughput claim
is inferred from the sequential single-user run.

The first retained DeepSeek V4 Flash result and its independent repeat are
documented in
[the benchmark baseline](../benchmarks/deepseek-v4-flash/eval/README.md).

## Tests

The Nix test suite verifies:

- source revision, blob, licenses, row order, row audit hashes, and COMPSEC
  metadata-only policy;
- the first four keys;
- MCQ, integer, and line-spec extraction, including no-answer and ambiguous
  output;
- exact request messages and ordering;
- omitted default temperature and exact greedy temperature;
- the fixed 16,000-token completion budget;
- single-model discovery;
- stop, length, context rejection, malformed response, and transport failure;
- artifact redaction and telemetry allowlisting.

Pi and coding-agent evaluation remain deferred to issue #153.
