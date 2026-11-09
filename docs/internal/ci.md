# CI workflows

What runs on a pull request, on `main` and at release, and which checks gate a
merge. All workflow files live in `.github/workflows/`. Job and step names start
with a verb; a reusable workflow's jobs appear as `caller / called job`.

| File | Display name | Job | Runner | When |
| --- | --- | --- | --- | --- |
| `checks.yml` | Repository checks | Run CPU and repository checks | GitHub Ubuntu | PRs to main, pushes to main, manual. Bounded CPU, repository and functional-harness tests. |
| `checks.yml` | Repository checks | Warm Linux package cache | GitHub Ubuntu, via `linux-build.yml` | Main and manual only. Builds the production package to fill the shared cache. |
| `linux-build.yml` | Linux build | Linux package | GitHub Ubuntu | Reusable. Main cache warming, and candidate and baseline packages for comparisons. |
| `windows-build.yml` | Windows build | Windows package | GitHub Windows | Reusable. Builds the native application, runs CPU contracts and verifies the bundle. Called by Windows validation and by Toolboxes releases. |
| `devlab-windows.yml` | Windows validation | Build, then Run Windows model checks | GitHub Windows, then Devlab Windows Strix Halo | PRs to main, pushes to main, manual. GPU diagnostics and model smoke checks on the same run's bundle. |
| `devlab-linux.yml` | Linux validation | Authorize and select checks | GitHub Ubuntu | Non-draft PR updates and manual runs. Pins revisions and plans the affected checks. |
| `devlab-linux.yml` | Linux validation | Build | GitHub Ubuntu, via `linux-build.yml` | When the plan selects work. Builds the candidate and `main` packages. |
| `devlab-linux.yml` | Linux validation | Run Linux model checks | Devlab Linux Strix Halo | Prepares while the hosted build runs, then tests once the packages arrive. |
| `devlab-linux.yml` | Linux validation | Publish results | GitHub Ubuntu | After testing, failures included. Publishes the report and the `devlab/functional` status. |
| `devlab-comment.yml` | Functional reporting | Update PR comment | GitHub Ubuntu | When Linux validation starts or ends. Keeps the PR comment current, also for forks, without running PR code. |
| `pr-title.yml` | PR title | Validate PR title | GitHub Ubuntu | PR opened, edited, reopened or updated. |
| `docs-website.yml` | Website docs | Request website development docs | GitHub Ubuntu | Pushes to main that change public docs, manual. Sends `gufo-docs` so gufo.blog refreshes its `development` docs. |
| `release-please.yml` | Release automation | Prepare, merge and publish release | GitHub Ubuntu | Manual. Waits for the required checks, publishes the release and sends the tag to Toolboxes and the website. |

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

- **Repository checks** use the compiler cache, Cachix and the Magic Nix Cache.
  Each run saves under a fresh key, so the Actions cache stays current.
- **Linux package builds** use the package compiler cache and Cachix. Main's
  warming job exists because PR builds don't push to Cachix.
- **Windows builds** cache the SDK archive, the compiler cache and the vcpkg
  binary cache.
- **Devlab runners** download the verified bundles that the hosted builds
  produce; Windows uses the bundle from the same workflow run.

Native Windows `check-pr` runs every Windows test from the platform-only build
with assertions enabled. A MinGW Nix package remains for local use only.
