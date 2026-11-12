# Troubleshooting

Start with `gufo diagnose`, then read the server log. This page explains both,
the common failures, and what to include in a bug report.

## Check the setup

```sh
gufo diagnose
gufo diagnose --section gpu        # platform, gpu, inventory or extensions
gufo diagnose --json               # for scripts and bug reports
```

`diagnose` checks the driver, device permissions, memory and the extensions
compiled in, without loading a model. It runs as a separate process, so it
can't see the state of a server that is already running; use `/health` and
`/ready` for that.

## The GPU isn't found

- Your user must be in the `video` and `render` groups; log out and back in
  after adding them. Check with `id` and `ls -l /dev/kfd /dev/dri/renderD*`.
- In rootless Podman, `--group-add keep-groups` needs the `crun` runtime.
  Groups named `video` or `render` inside the container are not your host
  groups.
- On Fedora and other SELinux hosts, ROCm can report "Memory critical" when
  SELinux blocked mapping `/dev/kfd`. Look for denials on the **host**:

  ```sh
  sudo ausearch -m avc -ts recent | grep -E '/dev/kfd|hsa_device_t'
  ```

  Podman's documented fix, `sudo setsebool -P container_use_devices true`,
  lets every container on the host use the devices passed to it; review that
  scope before enabling it.

## The server stops with `device_lost`

A GPU reset leaves the process unable to use the GPU again. Gufo returns 503
`device_lost`, logs `event=device_lost remedy=restart`, and exits with status
75. Run it under a supervisor that restarts it; see
[health and restarts](server.md#health-and-restarts). If generation hangs
instead of failing, `--stall-timeout-ms 120000` turns the hang into the same
exit.

## Requests are slow

- **Every turn processes the whole conversation:** check `timings.cache_n` in
  the responses. If it's near zero, something early in the prompt changes
  between turns; see [what breaks reuse](cache.md#what-breaks-reuse).
- **Requests wait:** each `--sessions` slot runs one request; others queue.
  `/slots` shows what each session is doing.
- **Prefill seems stuck:** start the server with `--log-progress` to log each
  prefill chunk and every 50 decoded tokens, with speeds.

## Client notes

- **Codex with Qwen models** sends developer messages in the middle of a
  conversation. Qwen's template allows one system message at the start, so
  Gufo moves them there; the request that adds one processes the conversation
  again, and later requests reuse it.
- **Claude Code with Qwen models** isn't supported: it adds a new system
  message on every turn, so every request processes the whole history again.

## Logs

The server logs to stderr. Each request gets an `X-Request-ID` response header
that matches its `request=rN` log lines. The completion line reports the
finish reason, queue and first-token time, speeds, reused tokens and, on a
cache miss, `cache_miss_reason`.

`--log-level` sets the detail:

| Level | Adds |
| --- | --- |
| `error` | Server errors only. |
| `warn` | Refused requests (4xx), cache and capacity warnings. |
| `info` (default) | Startup, loading, and one line per request. |
| `debug` (`-v`) | Resolved options, queue decisions, and why each cache checkpoint was skipped. |

At `warn` and `error` the startup line is hidden too, so check `/ready` to
confirm the server started. No level logs prompts or replies, but `debug` logs
client IP addresses.

## Content trace

When the problem is in the text itself, such as tool-call markup leaking into
the answer, start the server with `--trace trace.jsonl`. Every text request
writes three JSON lines: the request as received, the prompt as the model saw
it with the raw output, and the response as sent. To read one request:

```sh
jq 'select(.request == "r12")' trace.jsonl
```

> **Warning:** The trace holds whole conversations, tool results included.
> Treat it like the conversations themselves and delete it when you're done;
> Gufo never rotates or truncates it.

## Reporting a bug

Open an issue at [gufo-org/gufo](https://github.com/gufo-org/gufo/issues)
with:

- `gufo --version` and `gufo diagnose --json`;
- the full command line, and the model files with their revisions;
- the server log around the failure, at `--log-level debug` if you can
  reproduce it;
- for wrong output, the trace records of the request, after removing anything
  private.
