# CI workflows

What runs on a pull request, on `main` and at release, and which checks gate a
merge. All workflow files live in `.github/workflows/`. Job and step names start
with a verb; a reusable workflow's jobs appear as `caller / called job`.

| File | Display name | Job | Runner | When |
| --- | --- | --- | --- | --- |
| `checks.yml` | Repository checks | Run CPU and repository checks | GitHub Ubuntu | PRs to main, pushes to main, manual. Bounded CPU, repository and functional-harness tests. |
| `checks.yml` | Repository checks | Warm Linux package cache | GitHub Ubuntu, via `linux-build.yml` | Main and manual only. Builds the production package to fill the shared cache. |
| `warm-caches.yml` | Compiler cache warming | Warm check compiler caches | GitHub Ubuntu | Pushes to main, nightly, manual. Compiles the CPU and Rust checks without running them to fill `main`'s compiler caches. Blocks nothing. |
| `linux-build.yml` | Linux build | Linux package, Linux checked package | GitHub Ubuntu | Reusable. Main cache warming, candidate and baseline packages for comparisons, and on request the candidate's assertion-enabled `.#checked` package. |
| `windows-build.yml` | Windows build | Windows package | GitHub Windows | Reusable. Builds the native application, runs CPU contracts and verifies the bundle. Called by Windows validation and by Toolboxes releases. |
| `devlab-windows.yml` | Windows validation | Build, then Run Windows model checks | GitHub Windows, then Devlab Windows Strix Halo | PRs to main, pushes to main, manual. GPU diagnostics and model smoke checks on the same run's bundle. |
| `devlab-linux.yml` | Linux validation | Authorize and select checks | GitHub Ubuntu | Non-draft PR updates and manual runs. Pins revisions and plans the affected checks. |
| `devlab-linux.yml` | Linux validation | Build | GitHub Ubuntu, via `linux-build.yml` | When the plan selects work. Builds the candidate and `main` packages, and the checked candidate that correctness runs on. |
| `devlab-linux.yml` | Linux validation | Run Linux model checks | Devlab Linux Strix Halo | Prepares while the hosted build runs, then tests once the packages arrive. |
| `devlab-linux.yml` | Linux validation | Publish results | GitHub Ubuntu | After testing, failures included. Publishes the report and the `devlab/functional` status. |
| `devlab-comment.yml` | Functional reporting | Update PR comment | GitHub Ubuntu | When Linux validation starts or ends. Keeps the PR comment current, also for forks, without running PR code. |
| `caches.yml` | Actions caches | Prune caches | GitHub Ubuntu | Reusable after each cache-saving workflow, on PR close, nightly and manual. Deletes closed PRs' caches, superseded per-run entries and, over budget, other refs' large idle entries. |
| `pr-title.yml` | PR title | Validate PR title | GitHub Ubuntu | PR opened, edited, reopened or updated. |
| `docs-website.yml` | Website docs | Request website development docs | GitHub Ubuntu | Pushes to main that change public docs, manual. Sends `gufo-docs` so gufo.blog refreshes its `development` docs. |
| `release-please.yml` | Release automation | Prepare, merge and publish release | GitHub Ubuntu | Manual. Waits for the required checks, creates the draft release and sends the tag to Toolboxes and the website. Toolboxes publishes the draft. |

## Required checks

The **Required CI** ruleset requires these contexts on `main`:

| Context | From |
| --- | --- |
| Run CPU and repository checks | `checks.yml` |
| Validate PR title | `pr-title.yml` |
| Build / Windows package | `devlab-windows.yml` calling `windows-build.yml` |
| Run Windows model checks | `devlab-windows.yml` |
| `devlab/functional` | Commit status from `devlab-linux.yml` |

A required context is the caller and leaf job name, not the workflow display
name, so renaming a job means updating the ruleset, `release-please.yml`'s check
names and `devlab-comment.yml`'s `workflow_run` subscription in the same change.
Toolboxes calls `windows-build.yml@main` by path.

`devlab/functional` fails on correctness failures and on checked phases that
are consistently slower than `main`. Missing or incomparable evidence is an
error. Noisy but complete comparisons pass with a visible warning, on ordinary
and release PRs alike. See [testing](testing.md#text-api-functional-tests) for
how checks are selected.

## Caches

Compiler and dependency caches restore and save in separate steps. Saves run
even when the build fails, is cancelled or times out, so a cold run seeds the
cache for its rerun. Long steps have their own time limits below the job's,
which leaves time for the saves.

The Actions cache holds 10 GB per repository and evicts the least recently used
entries. A pull request restores only its own entries and `main`'s, so PRs
avoid saving copies of `main`'s caches, and `caches.yml` keeps the total below
`main`'s eviction point:

- After each run, only the newest per-run entry (keys ending in
  `-<run id>-<attempt>`) of each key per branch or PR remains.
- Then entries of refs other than `main` are deleted, largest and longest
  idle first (size times idle time), until the cache is below 7 GiB. Thousands
  of tiny Magic Nix Cache entries go last. `main`'s entries are never deleted
  this way.
- A closed PR's entries are deleted. Nightly, every ref is swept again, and
  entries of other refs that no run has used for 2 days are deleted.
- An entry created or restored in the last 15 minutes is never pruned, since a
  concurrent run may be downloading it.
- One run deletes at most 100 entries, one per second; the next run or the
  nightly sweep continues. A rate limit (HTTP 429, or a 403 reporting one) is
  retried after `Retry-After`, or `X-RateLimit-Reset` once the hourly quota is
  spent, else after 60 s, doubling. Retries take up to 4 minutes in total, then
  the prune stops with a warning. Any other failed delete, such as
  a missing permission, fails the job; an entry already gone does not.

`tools/ci/prune_caches.py` implements this; its tests stub `gh`.

Which runs save each cache:

| Cache | Key | Pushes to `main`, schedule, dispatch | Pull requests |
| --- | --- | --- | --- |
| TheRock SDK, IRON, compiled NPU designs, Loom runtime, Linux NPU designs | Exact content | On a miss | On a miss, which means the PR changed the inputs |
| `gufo-vcpkg` | Manifest, TheRock SDK and toolchain file, per run | Always | When the run built a new archive, or without a measurement, when nothing for these inputs was restored |
| `gufo-windows-extension-vcpkg` | Manifest, per run | Always | Only when nothing for this `vcpkg.json` was restored |
| Windows NPU design cache (`gufo-npu-windows`) | IRON pins, per run | Always | Never; the compiled designs above cover the rerun |
| Check compiler caches (`gufo-ccache`, `gufo-rust`) | Toolchain, per run | `warm-caches.yml`, when the run added at least 32 MiB or nothing was restored | `checks.yml`, when the run added at least 32 MiB |
| Other compiler caches (`gufo-package-ccache`, `gufo-checked-ccache`, `gufo-windows-ccache`) | Toolchain, per run | Always | When the run added at least 32 MiB |

`tools/ci/cache_growth.py` measures what a run added: new files, or files whose
size changed, since the restore. A failed measurement saves. Outside a pull
request, a Linux build saves only when it built the calling ref's own revision:
a dispatched Devlab run builds a PR from `main`'s ref and must not fill `main`'s
cache. Toolboxes calls `windows-build.yml` from `repository_dispatch`,
`schedule` and `workflow_dispatch`, so its runs save as `main` does, in the
Toolboxes cache.

- **Repository checks** use the compiler cache, Cachix and the Magic Nix Cache.
  On `main` every check comes from Cachix and compiles nothing, so
  `warm-caches.yml` fills `main`'s check compiler caches instead. It builds
  `.#check-tests-compile` and `.#check-rust-compile`, which compile what
  `check-tests` and `check-rust` compile, with the same compilers, flags, cache
  settings and sandbox paths, and run no tests. Their objects are therefore the
  cache entries a PR's checks look up. The workflow pulls from Cachix and never
  pushes, and a newer push to `main` cancels its run after saving what it
  compiled. The Rust cache only ever gains objects, so past 1 GiB the workflow
  clears it before building, and a PR keeps its restored Rust cache instead of
  saving.
- **Linux package builds** use the package compiler cache and Cachix. Main's
  warming job exists because PR builds don't push to Cachix.
- **Windows builds** cache the SDK archive, IRON, NPU designs, the Loom runtime,
  the compiler cache and the vcpkg binary cache.
- **Devlab runners** download the verified bundles that the hosted builds
  produce; Windows uses the bundle from the same workflow run.

Native Windows `check-pr` runs every Windows test from the platform-only build
with assertions enabled. A MinGW Nix package remains for local use only.
