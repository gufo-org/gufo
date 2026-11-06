# Card 03 retained evidence

The implementation lives exclusively in throwaway jj change `ovmkszut`, exact
source `50ba836fc4ab2fe7003371f3f290bdd48ea68b0a`, parent `79930b75`.
[spike.patch](spike.patch) reproduces its source diff. The main workspace contains
only this evidence, the [spike report](../../SPIKE.md) and card ledger changes;
the user's RFC is unchanged.

## Production binaries

| Arm | Path | SHA-256 |
| --- | --- | --- |
| Baseline | `/home/mixer/gufo/build/release/gufo` | `04a885b0d7847c5567c233cb1aeecafd1de3ca4862cec3034c505e0ce6dad8b6` |
| Candidate | `/tmp/gufo-card03-spike/build/release/gufo` | `0bf5c8653f071884f4f575dff3d51ccc1a3b640900ea293b72fc65b91c2b0a9d` |

Both use production `release` preset, `BUILD_TESTING=OFF`, `RelWithDebInfo`,
`NDEBUG`, GCC 16.2, HIP Clang 24 and ROCm 10.1, the same `flake.nix`/`flake.lock`
and dependency store paths. Both default embedded revisions are `unknown`;
explicit source revisions above take precedence for evidence. Candidate initial
build was followed by a ten-object rebuild after formatting; only the final hash
above is used in controlled candidate runs. The earlier unmeasured candidate
hash `aecfc884cb8b7e7083c6587fe042eece774e381c0ad57288d04aa6746401cfb5`
is not a timing baseline.

Configure from each source root:

```sh
nix develop -c cmake --preset release -DCMAKE_HIP_COMPILER=/nix/store/92bpbb2rmrkszhw83x63x7qzj9jsz324-rocm-toolchain/bin/clang
nix develop -c cmake --build --preset release --target gufo --parallel 4
```

Baseline additionally used `--fresh` to remove a stale old-ROCm cache before its
successful build. `diagnose` passed gfx1151/ROCm 10.1 on both final binaries.

## Focused tests

From the spike workspace:

```sh
nix develop -c cmake --preset cpu-test
nix develop -c cmake --build --preset cpu-test --target qwen27b_continuation_identity_test qwen38_flash_next_continuation_identity_test deepseek_v4_flash_continuation_identity_test text_model_runner_test text_generation_scheduler_test qwen38_flash_next_mtp_sampling_test --parallel 4
nix develop -c ctest --test-dir build/cpu-test -R '^(qwen27b_continuation_identity_test|qwen38_flash_next_continuation_identity_test|deepseek_v4_flash_continuation_identity_test|text_model_runner_test|text_generation_scheduler_test|qwen38_flash_next.mtp_sampling)$' --output-on-failure
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py --fix
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
nix develop -c c++ -std=c++20 -I. -fsyntax-only /tmp/card03-header-probe.cpp
```

[CPU tests](cpu-tests.txt) pass 6/6 with zero skips. The MTP sampling test retains
independent proposal/rejection/residual/RNG/stop numerical checks. The three
identity tests pin card 02's exact compatibility bytes, including DeepSeek
executed locally without a model-skip path. [Formatting](format-check.txt) passes
716 files. The [standalone provider-header probe](header-probe.cpp.txt) compiles
without `ENGINE_ENABLE_HIP` or model headers ([output](header-probe.txt)).

## Functional and timing captures

[run-functional.py](run-functional.py) records exact commands and executes each
baseline/candidate arm sequentially with explicitly selected `state-edges`,
`long-context`, `tool-reasoning` and `cache` suites, PR tier, sessions 4, context
8192 and actual projector. The candidate runs the real checks. Baseline executes
only the same affected histories needed for matched request/phase timing.
Every raw report/request/response/server log remains under
`artifacts/model-extensions/card03/`; disk cache payloads stay local.

[attempts.json](attempts.json) preserves the first diagnostic arm's timing
contamination: candidate production/CPU compilation overlapped early requests.
Its timings are excluded from comparisons. Subsequent `*-idle` arms start after
all builds/tests finish, with no competing GPU workload or root build jobs.

## Retained controlled results and review disposition

[summary.json](summary.json) retains every nonpassing phase and all qualification
counts. Original `mtp-qualified.json` in the archive records three reproduced
flags on one cold-schema request. Exact-history focused replays did not
reproduce them; the combined result is inconclusive. Off ends at 1,640 passed /
23 inconclusive / zero reproduced failures; MTP with focused controls ends at
1,604 passed / 59 inconclusive / zero reproduced failures, each out of 1,663
phases. Performance parity and the initial stall cause remain unproven.
The original failed MTP qualification is retained unchanged. Third-agent
integration dispositions it as inconclusive across the expanded controls: initial
delays remain unexplained and parity remains unproven, with no confirmed failure
across the combined ranges. No thresholds were widened and no arithmetic fix was
applied. Later production migrations retain all their own quality/timing gates.

[functional-captures.tar.xz](functional-captures.tar.xz) retains all 560 raw
report/request/body/log/invocation/comparison files from every attempt, excluding
only generated multi-gigabyte disk-cache payloads. [raw-capture-sha256.json](raw-capture-sha256.json)
pins each archived file. To restore them into a new empty directory:

```sh
tar -xJf docs/model-extensions/evidence/card03/functional-captures.tar.xz -C /path/to/new/empty/directory
```

[run-controls.py](run-controls.py) reproduces the alternating full histories and
unchanged-main comparisons. [run-cold-controls.py](run-cold-controls.py) keeps the
original suite arguments and adds `--through-case long-context:long_schema_cold`.
The runner computes disk policy before truncating those suites; exact
request/history hash matching is also checked by the existing strict qualifier.
The focused pairs cover 182 phases each and change no output/work counters.

[build-audit.json](build-audit.json) records 281 matching normalized compilation
commands. Raw debug-containing bundle bytes differ in all 74 images;
[device-code-audit.json](device-code-audit.json) extracts actual gfx1151 images:
all 74 resource notes match, 71 `.text` sections match, three MMQ `.text`
sections and seven constant sections differ. Source-root identifiers differ,
but their complete relationship to those instruction/constant differences is
not established here. These are explicit audit limits, not universal ISA
parity evidence. The [build](audit-builds.py) and [device](audit-device-code.py)
auditors reproduce the inventories after GPU timing jobs stop.

## Integration checks

The implementation/review/integration agents are `/root/card03_implement`,
`/root/card03_review` and `/root/card03_integrate`. Integration executed:

```sh
python3 docs/model-extensions/evidence/card03/run-speed.py
python3 docs/model-extensions/evidence/card03/analyze-speed.py
nix develop -c c++ -std=c++20 -x c++ -I/tmp/gufo-card03-spike docs/model-extensions/evidence/card03/metadata-probe.cpp.txt -o /tmp/card03-metadata-probe
/tmp/card03-metadata-probe
python3 docs/model-extensions/evidence/card03/capture-provenance.py
```

The [standard-speed capture wrapper](capture-standard-speed.py) retains each
actual `RequestObservation`, request/history hash and actual server command.
[speed-captures.tar.xz](speed-captures.tar.xz) contains all 39 files from these
runs; [speed-capture-sha256.json](speed-capture-sha256.json) verifies every raw
file, including both comparisons and all invocations/logs. It restores with the
same tar command as above into an empty directory. No builds/tests ran during
timed jobs; the metadata fixture compiled only after the final GPU job exited.

All eight HTTP observations (including calibration/warmup) have identical
request/history hashes and output/cache/work/draft counts; all 70 available
individual phase comparisons pass 5% and 10 ms (two unavailable fields remain
unmeasured). Standard measured C1 pp2048/tg128
cells use AR and MTP prose/repetitive workloads, depth 0 and one repetition.
Each CLI mode's pp2048, tg128 and short-prompt decode phase passes the same
margins, with exact bench/prompt output hashes. CLI durations derive from printed
rounded rates, without averaging; prompt exposes decode timing only. These
bounded speed results do not turn the full functional qualification into a pass.

Candidate GPU prefill validation passes existing bounds at 16 tokens: finite,
winner rank 1, RMSE 0.389394, cosine 0.990998, maximum error 2.20532. It compares
GPU batched and sequential paths. The independently numerical evidence is the
retained `qwen38_flash_next.mtp_sampling` CPU test's proposal/rejection/residual,
RNG and stop checks; no independent full-model GPU forward oracle was run or
claimed. Exact matched serving output/work plus unchanged arithmetic bodies
complete the throwaway spike's scoped semantic evidence. Model migrations retain
their own independent numerical/API/performance obligations.

The [executable metadata fixture](metadata-probe.cpp.txt) and its
[output](metadata-probe.txt) validate HIP-free shape/representability: artifact
key, both canonical option names, speculative value, all five typed option
fields and the factory signature. Its mapping is fixture code. Production does
not yet consume metadata for selection/parsing/CLI registration, so those remain
dependent-card requirements. Actual typed options and factory execute on serving.

[provenance.json](provenance.json) records actual per-arm source/binary hashes,
production cache flags, pinned compiler paths and model hashes. Main/MTP hashes
come from stat-validated production identity cache entries; projector is hashed
after timing. Raw model-bench artifacts autodetect editing-host GCC 15.3 and root
revision `79930b75`, including when supplied the candidate binary. That metadata
does not describe the supplied executable: authoritative audited provenance is
GCC 16.2/HIP Clang 24/ROCm 10.1 and the source/hash table above. Raw artifacts
remain unchanged. No pending or inconclusive gate is a pass.
