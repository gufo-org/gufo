# CI workflows

Workflow names identify the platform and purpose. Job and step names start with
an action. Reusable workflow callers appear as `caller / called job` in GitHub.
All workflow files below live in `.github/workflows/`.

| Workflow file | Display name | Jobs | Runner | Activation and connections |
| --- | --- | --- | --- | --- |
| `checks.yml` | Repository checks | Run CPU and repository checks | GitHub Ubuntu | PRs to main, pushes to main, manual. Runs bounded CPU/repository and functional harness tests. |
| `checks.yml` | Repository checks | Warm Linux package cache | GitHub Ubuntu through `linux-build.yml` | Main/manual only. Builds the production package and populates its shared cache. |
| `linux-build.yml` | Linux build | Linux package | GitHub Ubuntu | Reusable only. Called for main cache warming or candidate/baseline comparison packages. |
| `windows-build.yml` | Windows build | Windows package | GitHub Windows | Reusable only; called by Windows validation and Toolboxes releases. Builds the native application, runs CPU contracts, and verifies bundles. |
| `devlab-windows.yml` | Windows validation | Build | Through `windows-build.yml` | PRs to main, pushes to main and manual runs. Calls the reusable native builder. |
| `devlab-windows.yml` | Windows validation | Run Windows model checks | Devlab Windows Strix Halo | After the build on Gufo PR/manual runs. Downloads the same run’s Windows bundle and runs GPU diagnostics and model smoke checks. |
| `devlab-linux.yml` | Linux validation | Authorize and select checks | GitHub Ubuntu | Non-draft PR updates/readiness and manual runs. Pins revisions and produces the selection plan. |
| `devlab-linux.yml` | Linux validation | Build | GitHub Ubuntu through `linux-build.yml` | After authorization when selected work exists. Builds candidate and main packages. |
| `devlab-linux.yml` | Linux validation | Run Linux model checks | Devlab Linux Strix Halo | Starts preparation after authorization, overlaps hosted compilation, and waits for comparison artifacts before running tests. |
| `devlab-linux.yml` | Linux validation | Publish results | GitHub Ubuntu | After authorization/testing, including failure paths when a head revision is available. Publishes the report and `devlab/functional` status. |
| `devlab-comment.yml` | Functional reporting | Update PR comment | GitHub Ubuntu | Linux validation PR runs starting/completing. Maintains placeholders and publishes fork reports without executing PR code. |
| `pr-title.yml` | PR title | Validate PR title | GitHub Ubuntu | PR open/edit/reopen/synchronize events targeting main. |
| `release-please.yml` | Release automation | Prepare, merge and publish release | GitHub Ubuntu | Daily/manual. Waits for repository checks, PR title and `devlab/functional`, then merges/publishes and requests Toolboxes artifacts. |

`devlab/functional` is an aggregate commit status rather than a runner job.
Correctness failures and consistently slower checked phases fail it. Missing or
incomparable evidence is an error. Complete noisy comparisons succeed with a
visible warning on both ordinary and release PRs.

## Cache ownership

- Repository checks retain the compiler cache restore prefix, Cachix, and Magic
  Nix Cache. Fresh per-run save keys refresh the immutable Actions cache rather
  than retaining an initially empty archive for the entire week.
- Linux package builds retain the package compiler cache and Cachix. Main's
  warming invocation remains because functional PR runs do not push to Cachix.
- Windows builds retain the SDK archive, compiler cache, and vcpkg binary cache,
  including their existing keys, restore prefixes, and save conditions.
- Devlab consumers download verified bundles; the Linux consumer also reuses its
  persistent runtime cache. Both validation workflows call reusable builders; Windows consumes its
  bundle in the same workflow run.
- The removed MinGW jobs duplicated native Windows platform/cache tests. Native
  `check-pr` includes all Windows tests from the platform-only build, with test
  assertions enabled. The optional local MinGW Nix package remains available.

## Rename rollout

The release workflow's explicit check-name references and the reporting
workflow's `workflow_run` subscription are updated alongside the workflow names.
Toolboxes must use `gufo-org/gufo/.github/workflows/windows-build.yml@main` after
the renamed workflow reaches Gufo main; its caller change must not land first.

GitHub's live **Required CI** ruleset (25949571) has been migrated to the
renamed check contexts:

| Existing required context | Replacement |
| --- | --- |
| CPU contracts & repository checks | Run CPU and repository checks |
| Conventional Commit title | Validate PR title |
| Windows build | Build / Windows package |
| Windows Devlab checks | Run Windows model checks |
| devlab/functional | Unchanged |

The Required CI ruleset is migrated with the job names. Reusable check contexts
include the caller and leaf job names; workflow display names are not part of
the required context. Linux and Windows retain distinct check contexts.
