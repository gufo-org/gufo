# Card 02 identity baseline

Captured on 2026-10-30 on the local Linux x86-64 host, from parent revision
`d1304fd6` (`wkwulkks`), before production identity writers moved. The original
writers were literal copies of `inference_backend.cpp`'s three identity writers,
`IsSha256Hex`, and state ABI constants/function, compiled with their actual
production headers/constants. The frozen capture inputs are retained beside
this file; `SHA256SUMS` records both inputs and every exact fixture byte file.
These files document the original baseline; do not update them alongside a
migration or generate expected strings from the new writer.

The initial capture used the then-current pinned Nix development environment
(`nix develop -c c++`, GCC 16.2.0, C++20), with HIP disabled in this standalone
translation unit. No model was opened and no GPU inference ran. DeepSeek's
actual writer ran locally with its actual chat-template/reference constants;
its test has no missing-model skip path. Reproduce with the original revision's
headers/toolchain in a separate source checkout, copying these retained inputs
and running from that checkout's root:

```sh
cp /path/to/evidence/original-writers.cpp.txt /tmp/card02_original_writers.cpp
cp /path/to/evidence/cases.hpp.txt /tmp/card02_cases.hpp
cp /path/to/evidence/record.cpp.txt /tmp/card02_record.cpp
nix develop -c c++ -std=c++20 -I. /tmp/card02_record.cpp -o /tmp/card02_record
/tmp/card02_record
```

The driver writes only the named model fixtures under `tests/models/`; run it
in the disposable original checkout, then compare against committed goldens.
The capture inputs intentionally do not include the extracted production seam.
The corresponding implementation tests only read fixture files.

Coverage: 20 Qwen 27B cases include the four attention-KV/recurrent storage
combinations, with/without DFlash, context and artifact changes, fixed/adaptive
policy, token bounds, rolling window, float precision, adaptive/batched toggles,
and the currently identity-neutral frontier-logit/fusion flags. Eleven
Flash-Next cases include with/without MTP, context, explicit YaRN policy bytes,
concurrency, token bounds and artifact changes. Eight DeepSeek cases include
DSpark, target-only support omission, constructor-reachable token limits,
context/artifact changes and writer-level zero-limit/empty-support edge cases.
The writer itself validates only DeepSeek's main fingerprint; DSpark support
validation remains in the runner constructor, unchanged. Fixtures do not claim
to test every runner constructor or model-load path.

Process-local Qwen/Flash-Next identities are tested separately using literal
field names/order plus the current standard-library pointer spelling. Addresses
vary across processes and libraries, including null spelling, so no persistent
pointer golden is meaningful. Tests check model/draft pointer distinctions and
context/settings distinctions. Qwen's process-local identity omits speculative
settings/storage; Flash-Next includes MTP/token-limit/concurrency even without
MTP. These are existing decisions, preserved rather than redesigned here.
DeepSeek has no corresponding component identity fallback: without an artifact
fingerprint its persistence descriptor remains absent.

The temporary `src/cli/serve/continuation_identity.*` seam stays with serving
until the model runner migrations (cards 20–22); then each writer and its tests
follow their model. It is not a public extension API. No numerical routes,
execution scheduling, tensor data, cache payload serialization or key field
values changed, so this card needs identity/compilation checks rather than a
new numerical/HTTP performance matrix.
