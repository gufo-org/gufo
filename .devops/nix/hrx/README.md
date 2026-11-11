# HRX staging

Gufo's experimental HRX packages use
[`gufo-org/hrx-system`](https://github.com/gufo-org/hrx-system). The source
revision in `default.nix` is an exact commit from
[`gufo-staging`](https://github.com/gufo-org/hrx-system/tree/gufo-staging),
so advancing that branch does not silently change a Gufo build.

Each upstream candidate starts from upstream `main` on its own `fix/<topic>`
branch, containing only that fix and its regression coverage. Combine validated
candidates on `gufo-staging`; keep the fix branches independent so upstream can
review and merge them separately. Staging merges the independent candidates
without adding unrelated changes.

| Fix branch | Purpose |
| --- | --- |
| [`fix/cached-kernel-arguments`](https://github.com/gufo-org/hrx-system/tree/fix/cached-kernel-arguments) ([upstream PR #3035](https://github.com/ROCm/hrx-system/pull/3035)) | Give PM4 argument generations GPU-cached host-coherent backing independently of native queue control cells. |
| [`fix/upload-cache-state`](https://github.com/gufo-org/hrx-system/tree/fix/upload-cache-state) ([upstream PR #3036](https://github.com/ROCm/hrx-system/pull/3036)) | Use native transfers for synchronous uploads into device allocations, preserving GPU read performance. Experimental; small-copy preparation can cost more. |

The staging base includes upstream's loader search-path and queue-policy fixes,
so Gufo no longer applies its corresponding patches. The two remaining local
patches enable optional capture diagnostics and unfinished profile-session
reading; neither changes normal runtime behavior.

After staging changes:

1. Update `revision` and the source `hash` in `default.nix` to the tested staging
   commit. Keep the commit immutable once Gufo pins it.
2. If dependency pins changed, temporarily set the dependency `outputHash` to
   `lib.fakeHash`, build `.#hrx.passthru.deps`, and retain the reported hash.
3. Build `.#gufo-hrx`, check correctness and matched timings for affected
   workloads, and record the revisions and results in
   `docs/internal/hrx-migration.md`.
4. When upstream incorporates a fix, merge the updated upstream base into
   staging, resolve any duplicated implementation, and repeat qualification.
   Do not add another local copy of an upstreamed fix.
