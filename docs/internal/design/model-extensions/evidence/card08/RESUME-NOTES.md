# Card 08 resume notes

Reconstructed at handoff; the worker deleted its original /tmp notes.

1. Worker extracted H3/eval/compare and the single DeepSeek text tier, moved
   the eval fixture, added ownership manifests/gate and ran local CPU/HIP,
   identity, flags/object/device and resource checks. Its initial dataset
   sibling-path failure was fixed; asset skips and parent resource drift were
   retained. See the five implementation evidence files.
2. Independent reviewer inspected the tree read-only and reported F1–F9;
   no edits were made by review. Reviewer run ID and source artifact are in
   the card execution record.
3. Integration corrected CPU text optimization, Qwen links, manifest direction,
   symmetric claim validation and parity fail-closed behavior; added a negative
   fixture and corrected provenance/documentation. The Nix daemon probe was
   denied. Pinned build/parity/format reruns remain PENDING; local static checks
   use an existing store Python without asserting a pinned-shell pass.
4. No functional harness, timings, benchmark, commit, push or jj mutation ran.
   No live jobs remain. The original RFC is unchanged.

Resume with exact commands from [integration-reruns.txt](integration-reruns.txt),
then [pending-commands.md](pending-commands.md). Fresh sanitizer configure,
build and selected tests must use /tmp/card08-sanitizer with cleanup on every
exit. Do not write parent baselines. Rebuild CPU artifacts after the flags fix;
rebuild the four Qwen targets after the link fix. Re-run flags/object/device
once in the pinned shell. P7 is authoritative only after the closing commit
and includes source-snapshot proof; production package build P10 remains
unbuilt. The orchestrator owns final closure and post-Card-14 qualification.
