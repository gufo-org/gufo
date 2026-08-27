# Antirez DS4 First-Four Regression Baseline

Run date: 2026-08-27.

This is a Gufo regression baseline, not an official dataset score. Both runs
used the pinned DeepSeek V4 Flash artifact through the real OpenAI-compatible
`gufo serve` route. Requests were sequential and independent, with server
default sampling and thinking behavior. The evaluator omitted `temperature`
and sent `max_completion_tokens: 16000`.

| Run | Passed | Failed | Execution errors | Length finishes | Completion tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Default](antirez-ds4-first4-default.json) | 4 | 0 | 0 | 0 | 939 |
| [Default repeat](antirez-ds4-first4-default-repeat.json) | 4 | 0 | 0 | 0 | 939 |

The extracted answer sequence was `B`, `C`, `70`, `C` in both runs. Complete
visible responses, reasoning content, grades, and run identities were
identical across the repeat. The server reported `cache_hit: false` and zero
cached tokens for every request, so no cache reuse occurred in this pair.

The retained JSON uses sanitized endpoint, server-configuration, and artifact
identities. It contains no private endpoint, credential, machine identifier,
or local model path. These single-user runs make no serving-throughput claim.
