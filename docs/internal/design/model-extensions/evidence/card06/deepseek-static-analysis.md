# Card 06 — DeepSeek static correctness analysis (not runtime serving evidence)

Scope: every DeepSeek/DSpark-reachable path touched by the Card 06 working-copy
diff (jj `@` vs parent `cfc9802f`). Cross-checked against the main-control
checkout `/tmp/gufo-main-ctl` @ `5d66de8c`.

**Status: static analysis only.** DeepSeek runtime serving/timing rows were
SKIPPED BY USER DECISION (host capacity, 2026-10-31) and are not claimed as
passes. Re-verify command (post-Card-14 qualification window, requires the
verified `models/deepseek-v4-flash` target + DSpark sidecar):
`nix develop -c python3 tests/functional/run.py --output /tmp/card06-ds4-<tag> --sampling-preset deepseek4 --suite discovery --suite stops --suite state-edges --suite batch --suite metrics --suite long-context --suite cache -- build/card06-release/gufo serve llm --model models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf --speculative dspark --dspark-model models/deepseek-v4-flash/DeepSeek-V4-Flash-DSpark-support-0731.gguf --sessions 2 --context 32768` (plus the `--speculative off` arm keeping the sidecar flag, and the matched main-control baseline/`compare.py` pair per the harness README).

## Per-path verdicts

1. **`src/core/speculative/draft_backend.hpp` — `sampling::TokenId` swap.**
   `tokenization::TokenId` (qwen/tokenizer.hpp:22) and `sampling::TokenId`
   (core/sampling.hpp:16) are both `using … = std::uint32_t;` in both trees
   (verified in candidate and `/tmp/gufo-main-ctl`). All 31 changed lines are
   this alias rename plus the include swap
   (`src/models/qwen/tokenizer.hpp` → `src/core/sampling.hpp`); the
   forward-declaration block for `sampling::SamplerState` was dropped because
   the full header is now included. Same-width, same-signedness integer: every
   ABI-visible struct layout (`DraftProposal`, `DraftTargetContext`,
   `DraftProposalRequest`) is unchanged. **Verdict: identical semantics.**

2. **`src/core/speculative/speculative_verifier.{hpp,cpp}` engine.**
   Normalizing `tokenization::TokenId`→`sampling::TokenId` and excluding the
   moved adapter/constructor lines, the candidate core `.cpp` diffs **empty**
   against main control (`diff` of filtered trees → no output): VerifyStep,
   VerifyBatch, PrepareStep, ProcessVerificationChunk, SaveState/RestoreState,
   snapshot/persistent restore, `IsStopToken` stop handling and all
   sampler/RNG/constraint-state copies are byte-identical. The two
   `hip::QwenGpuExecutor&` constructors became
   `std::unique_ptr<ISpeculativeTargetExecutor>` constructors with the same
   member-initialization order and `ConfigureAdaptiveDraftPolicy()` body; the
   third (non-owning `ISpeculativeTargetExecutor&`) constructor is untouched.
   `SpeculativeOptions::dflash_policy` was **never read by the engine**
   (main-tree grep: only header decl + `src/cli/serve/` consumers); removal is
   engine-inert. **Verdict: identical semantics; core has zero
   `src/models/**` includes.**

3. **Moved adapter — `src/models/qwen/hip/qwen_speculative_verifier.{hpp,cpp}`.**
   `awk`-extracted `QwenGpuSpeculativeTarget` class from main
   `speculative_verifier.cpp` vs the new file, after the alias rename, differs
   only in two clang-format line wraps (verified with `diff`). Every override
   still forwards 1:1 to `executor_`. The two new
   `hip::CreateQwenGpuSpeculativeVerifier` factories construct the same
   adapter (`make_unique<QwenGpuSpeculativeTarget>` → same object, now built
   model-side) and pass it to the identical verifier constructor.
   **Verdict: verbatim relocation, no behavior change.**

4. **`src/core/speculative/prompt_lookup.hpp` — untouched.** `jj diff -- …`
   is empty. Still the core/host primitive. **Verdict: unchanged.**

5. **DSpark wiring — `src/cli/serve/inference_backend.cpp`.**
   `jj diff | grep -ci "dspark|deepseek"` = 0: no hunk touches the DeepSeek
   path. The DSpark `DeepSeekTextRunner` (persistence descriptor, snapshot,
   session, cancellation, stop handling) never included `core/speculative`
   (grep: `src/models/deepseek_v4_flash/` has zero `core/speculative` includes
   before and after) — DSpark speculation runs inside the engine and is
   completely absent from this diff. The changed hunks are Qwen-only: verifier
   built via the model factory with identical arguments
   (`hip::CreateQwenGpuSpeculativeVerifier(*executor_, *draft_backend_,
   options)`) and the Qwen DFlash-policy flow re-plumbed:
   `speculative_config.dflash_policy` no longer copied into
   `SpeculativeOptions` (field removed) but passed as the explicit
   `QwenTextRunner` ctor parameter (same default `kAdaptive`), then to
   `QwenCompatibilityIdentity(..., dflash_policy_)` and
   `QwenDFlashGpuDraftConfig{.policy = dflash_policy_}` — same value, same
   default, same consumers, one fewer hidden carrier.
   `serve.cpp` (where the CLI parses the policy) is not in the diff.
   **Verdict: DeepSeek untouched; Qwen re-plumbing value-identical.**

6. **`src/cli/serve/continuation_identity.{hpp,cpp}`.**
   Whole-file diff = 6 changed lines, none matching `deepseek|dspark`:
   only `QwenCompatibilityIdentity` gained the explicit
   `speculative::DFlashDraftPolicy draft_policy` parameter used at
   `draft_policy=<DFlashDraftPolicyName>-v3` (main emitted
   `speculative_options.dflash_policy` at the same position with the same
   spelling). `DeepSeekCompatibilityIdentity` and `kDeepSeekStateAbi` bodies
   are character-identical. Machine proof: the three identity golden tests
   (Qwen27B, Flash-Next, DeepSeek persistence descriptor, incl. draft
   variants) pass with exact bytes — `unit-identity-goldens.txt` 3/3, and
   re-confirmed after these doc edits (item F). **Verdict: identity bytes
   unchanged.**

7. **`IDraftBackendSnapshot` persistence.** Overrides/uses
   (`PersistentPayloadBytes`/`SerializePersistent`) exist only in core
   (`draft_backend.hpp`, verifier), serving (`text_model_runner.*`,
   `inference_backend.cpp`) and Qwen DFlash
   (`src/models/qwen/hip/dflash/*`). None of the DeepSeek files implement or
   call them; the core-side declarations changed only via the alias rename
   (item 1), and the verifier snapshot/restore code is byte-identical
   (item 2). DeepSeek native snapshot/persistence continues through its own
   unchanged `TextRunnerSnapshot`/descriptor path. **Verdict: unaffected.**

8. **Cancellation and stop handling.** Inside the byte-identical engine core
   (item 2) and the unchanged DeepSeek engine/runner (item 5).
   `bench.cpp`/`prompt.cpp` diffs contain zero `dspark|deepseek` lines; their
   hunks are the Qwen factory construction and
   `speculative::GenerationOptions` for `SpeculativeVerifier::Generate`
   (neutral struct carrying the same `{max_new_tokens, sampling}` pair; the
   direct-generator path still uses `models::GenerationOptions`).
   **Verdict: unchanged.**

## Overall verdict

Every DeepSeek/DSpark-reachable path in this diff is a pure type-alias rename,
verbatim relocation, or value-identical re-plumbing, machine-corroborated by
(empty) normalized source diffs and the passing exact-byte identity goldens.
This is **static** evidence; DeepSeek serving, DSpark numerical quality
(`ds4.target`/`ds4.dspark`/`ds4.serving`) and timing remain PENDING/SKIPPED-BY-
USER-DECISION for the post-Card-14 qualification window with the command above.
