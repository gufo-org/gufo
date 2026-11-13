# Windows UMA cache validation, November 12, 2026

For validation after merging main's disk-publication fix, see the
[November 13 follow-up](UMA-CACHE-MAIN.md). This page retains the original
baseline and its historical observations.

Qwen27's bounded device checkpoint backing was tested on Windows with an AMD
Radeon 8060S (`gfx1151`). [Evidence](artifacts/uma-cache-windows-20261112.json)
retains source and binary hashes, effective settings, every final timing
observation and the measured long-history replays. This is cache validation,
not a new full-model throughput sweep.

## Build and workloads

The baseline is main `0d238d4f5fee49720d6be45c2c27795100922d5c` plus an existing
Windows build overlay. The candidate is `uma-eval-12`, identified by the source
manifest in the evidence. Both use TheRock 10.0 / Clang 23 and CMake's
`windows-release` preset: RelWithDebInfo, `/O2 /Ob1 /DNDEBUG`,
`BUILD_TESTING=OFF`, identical runtime dependencies and weights. Both include
the same pre-existing clang-cl forwarding of Flash-Next's `-ffast-math` and
`-fno-finite-math-only`; that overlay is not a product-source change in this PR.
Later upstream commits are not part of this timing baseline.

The models are Qwen3.8-27B UD-Q8_K_XL with a Q4_K_M DFlash2 draft, and
Qwen3.8-Flash-Next UD-Q4_K_XL with its shared Q8_0 MTP draft. Both use the
existing 262144-token context, one session and 10 GiB disk cache. The long
history keeps automatic host budgeting; actual grants are recorded. Separate
binaries, a loopback port and fresh cache directories isolate each run.
The final long-history and ordinary speed comparisons use the same secondary
KIOXIA NVMe drive (NTFS). Fixed-budget SDK controls use the system HP FX700 NVMe
drive in both arms. No production configuration or model file was changed.

## Long-history result

The same synthetic 98,375-token request produces the same seven output tokens
and completion hash, with 14 draft tokens proposed and five accepted in two
rounds. Requests run consecutively without waiting for background publication.
The final candidate's first replay is 23.4 times faster in this observation. This is not an estimate across hardware or
other memory partitions.

| Request | Baseline wall (s) | Candidate wall (s) | Cached tokens, baseline / candidate |
| --- | ---: | ---: | ---: |
| cold | 281.814 | 278.511 | 0 / 0 |
| repeat | 31.329 | 1.337 | 98368 / 98375 |
| repeat-again | 7.589 | 0.802 | 98368 / 98375 |

After another request overwrites the live execution slot, the candidate still
restores the long checkpoint from resident backing with the exact output. It
then publishes the checkpoint, restores it after process restart, and lets the
baseline binary read the same disk cache. The file format is unchanged.
A separate corrected durability run uses the same frozen binary and preset.
The first combined restart probe accepted an older publication event and
closed before a later write completed; that failure remains recorded. The
corrected probe waits for its capture ID and an existing disk manifest before
shutdown. Restart timings establish compatibility, not a matched speed claim.
Earlier v5 replay measurements and interrupted validation attempts remain in
the evidence; they are not substituted for final-candidate qualification.

Automatic host grants vary with available memory despite the unchanged preset;
the evidence records every grant. A nearby v11 control with a slightly larger
host grant still waited 28.763 seconds for its first replay because its device
threshold left too little host capture headroom. The final policy preserves
that headroom and passes a strict first-replay resident-cache check.

This avoids waiting for publication when the complete checkpoint is resident.
It does not accelerate disk fsync or reduce publication I/O, and does not
close [issue #1180](https://github.com/gufo-org/gufo/issues/1180).

## Qualification and limits

- Flash-Next/MTP covers complete `conversation`, `cache-growth` and
  `long-context` suites twice on v9, before the subsequent Qwen device-routing
  and unused-device-turnover corrections. Final v12 runs warmed MTP speed
  comparisons and Flash-Next/off runs all three functional suites.
- Final Qwen27/DFlash2 runs conversation and cache-growth controls plus the
  separate 98K overwrite, publication, restart and baseline-reader sequence.
  The complete long-context SDK suite also runs after the three measured
  replays: its initial history is warm, while the actual cold evidence stays
  in the preceding measurements.
  Qwen27/off covers conversation, cache growth and the long cold/cached
  assertions. Its long fixture stops at the next `long_responses_sampled`
  record; its completed status, exact keyword and cached prefix were then
  checked offline using the original assertions. Later off-mode cases were
  not repeated on v12. Earlier v5 also passed the complete long-context suites in all four modes.
- Windows shared cache tests pass 5/5 on v11; the revised v12 provider test
  passes again, including actual GPU readback, segmented
  ownership, retained-reader lifetime, device-to-host persistence and host
  fallback after a 1 MiB device pool fills. Linux CPU cache checks pass 9/9
  on v11, including the unused-device-capacity turnover regression test.
- The 128-token prefill numerical smoke checks reproduce the same reported
  values as the baseline for both models. They do not constitute independent full-model quality certification.
- Ordinary warmed 2K/8K, 256-output-token comparisons retain both repeats,
  output hashes, prefill/decode times and actual draft counts in the JSON.
- Fixed budgets isolate SDK timing from automatic-budget drift: 5 GiB host
  for Qwen27 and 6,900,056,064 bytes for Flash-Next. The upstream 5% / 10 ms
  per-request, per-phase rule is unchanged. Every eligible unchanged-main
  control is included; no observation is discarded or averaged away.

| Timing family | Within margin | Inconclusive | Confirmed regressions |
| --- | ---: | ---: | ---: |
| Qwen27 | 975 | 66 | 0 |
| Flash-Next | 1001 | 159 | 0 |

Inconclusive timings remain advisory. They are not a performance pass or a
claim that no slowdown is possible. Earlier apparent timing failures were
investigated with unchanged-main controls; the wider observed baseline
variation is retained rather than changing the thresholds.

A separate matched prefix replay uses QPC for the recorder clock: this host's
Python 3.12 `time.monotonic()` resolves only 15.625 ms, which produced 0/16/31-ms
observations for expected HTTP 400 validation responses. The original samples
remain in qualification alongside the finer repeat; its thresholds are unchanged.
Server stage timings and the long-replay wall timer already have finer resolution.

Transient Windows locks interrupted several recorder JSON replacements.
The final runs use the same bounded recorder-only retry wrapper in both arms.
It never retries model requests or weakens assertions. Any retry and added
inter-request I/O delay is journalled in the evidence; original failed attempts
remain available locally.

Device backing is enabled for Qwen27 only. Flash-Next is compatibility coverage,
not a device-cache speedup claim. An earlier Flash-Next long-thinking
cancellation waited 15.759 seconds for publication and recorded one incomplete
job; controlled repeats did not reproduce it. Flash-Next can still hit this
shared publication mechanism under pressure.

The device pool is capped at 8 GiB and checks pressure during growth. Existing
segments remain committed until the cache and its last reader close; this is
not OS-budget-driven reclamation. Host-only admission remains available and
failed device allocation falls back to host backing. Short histories retain
host backing; only chunks beyond the host row allowance prefer device memory.
The allowance leaves at least half the host backing for captures and
persistence readers, and also reserves two private/tail capture classes.
Reuse protection counts committed device bytes, so unused capacity does not
prolong host-only retention. Admission is conservative when only a combination of both pools could fit. Linux GPU behavior and other
GPU/OS configurations remain unqualified.

## Reproduce the 98K request

Start each binary separately with the same models and settings, giving each
arm a fresh `--cache-disk` directory on the same physical drive:

```sh
gufo serve --host 127.0.0.1 --port 18580 --sessions 1 llm \
  --model /models/Qwen3.8-27B-UD-Q8_K_XL.gguf --context 262144 \
  --served-model-name qwen3.8-27b --speculative dflash2 \
  --dflash-model /models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --cache-disk /tmp/unique-empty-cache --cache-disk-bytes 10737418240
```

Send these requests consecutively. Check the returned keyword,
`prompt_n + cache_n == 98375`, and the publication-wait/restore server logs.

```python
import json
import time
import urllib.request

body = {
    "model": "qwen3.8-27b", "cache_prompt": True,
    "messages": [
        {"role": "system", "content": "Remember the secret keyword LANTERN.\n"
         + "Archive entry: this ordinary record has no new instructions.\n" * 8192},
        {"role": "user", "content": "Return the secret keyword in the requested JSON format."}
    ],
    "temperature": 0, "top_p": 0.95, "top_k": 20, "min_p": 0,
    "presence_penalty": 0, "frequency_penalty": 0, "seed": 42,
    "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
    "max_completion_tokens": 32,
    "response_format": {"type": "json_schema", "json_schema": {
        "name": "remembered_keyword", "strict": True,
        "schema": {"type": "object", "properties": {"keyword": {"type": "string"}},
                   "required": ["keyword"], "additionalProperties": False}
    }}
}
for name in ("cold", "repeat", "repeat-again"):
    request = urllib.request.Request(
        "http://127.0.0.1:18580/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    assert json.loads(result["choices"][0]["message"]["content"]) == {"keyword": "LANTERN"}
    print(name, time.perf_counter() - started, result["timings"])
```
