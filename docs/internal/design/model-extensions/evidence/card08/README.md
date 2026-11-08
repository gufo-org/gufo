# Card 08 evidence and provenance

Candidate: worker checkpoint `@ uytorytn 1cac04bf`, parent Card 07
`zwllvvnv ec59bed0`; integration is an uncommitted edit on that same working
copy. No jj mutations, commits, pushes or staging by integration.
Agents: worker `3f4f4c60-7afa-4d25-9f02-14cd8b7070d6`, reviewer
`e7a49787-2db0-4a73-a81a-83e24f3259ac`, codex-exec-writer
`<INTEGRATION-RUN-ID>`.

Pinned nixpkgs revision: `3ec636362e1816bc2f0888fa0fa2c69f1bc9e504`.
Cache C++ compiler: GCC wrapper 16.2.0,
`/nix/store/r5ri78axx1qggkjq3vawywlzglg70b5w-gcc-wrapper-16.2.0/bin/g++`.
HIP compiler: `/nix/store/65dxwr554b4ij9f4aviz07f9v8jgai0a-hipClang/bin/clang++`;
ROCm toolchain: `/nix/store/92bpbb2rmrkszhw83x63x7qzj9jsz324-rocm-toolchain`;
architecture `gfx1151`. These identities are read from existing caches, not
fresh shell execution. Integration's pinned-shell identity query was denied by
the Nix daemon socket sandbox (exit 1); live compiler/ROCm versions remain
PENDING rather than inferred from other installed versions.

| Candidate directory | Type | Tests | HIP | Sanitizers | Lock selection |
| --- | --- | --- | --- | --- | --- |
| `build/card08-release` | RelWithDebInfo | OFF | ON | OFF | OFF |
| `build/card08-gpu` | RelWithDebInfo | ON | ON | OFF | OFF |
| `build/card08-cpu` | RelWithDebInfo | ON | OFF | OFF | OFF |

All three caches predate integration edits. In particular CPU text flags and
Qwen link changes require the pending fresh rebuild. Production is a local
RelWithDebInfo NDEBUG discovery artifact, not a built shipped Nix package.
`build/card06-*` and `build/card07-*` baselines are read-only controls throughout
integration; no configure/build command targeted them.

Host observed at integration: Linux 7.2.9 x86-64, total RAM 131161760 kB,
available 118150812 kB at inspection. No GPU workloads were run, and no
llama-swap services were changed; live service state was not queried.
`models/` contains `deepseek-v4-flash`, `qwen38-flash-next`, `embeddinggemma-2`,
Qwen3.8-27B and Qwen3.6-35B snapshot directories and DFlash2 Q4/Q8 GGUF files;
this is an availability inventory, not a weight-integrity check. No H3 model
asset directory is present; the nine H3 oracle rows remain asset-blocked.

- [Integration reruns](integration-reruns.txt): exact commands, exits, denials, pending gates.
- [Gates](gates.md), [source parity](moved-source-parity.md),
  [ownership](ownership-inventory.md), [unit evidence](unit-and-identity-tests.md):
  worker results with integration corrections; historical runs are not fresh reruns.
- [Pending commands](pending-commands.md): deferred and distribution gates.
- [Resume notes](RESUME-NOTES.md): reconstructed phases and remaining work.
- [Input brief](input-brief-worker.md): planned instructions, never results.
