# Releasing

Gufo uses squash merges and Conventional Commit pull request titles to keep
`main` machine-readable. A pull request may contain arbitrary work-in-progress
commits; its title becomes the single commit retained on `main`.

## Pull request titles

Titles have this form:

```text
<type>[(optional-scope)][!]: <description>
```

The accepted types are:

| Type | Purpose | Release effect |
| --- | --- | --- |
| `fix` | Correct user-visible behavior | Patch |
| `perf` | Improve user-visible performance | Patch |
| `feat` | Add backward-compatible functionality | Minor |
| `docs` | Change documentation only | None |
| `test` | Change tests only | None |
| `build` | Change the build system or dependencies | None by default |
| `ci` | Change automation only | None |
| `chore` | Repository maintenance | None |
| `refactor` | Restructure code without changing behavior | None |
| `style` | Change formatting without changing behavior | None |
| `revert` | Revert an earlier change | Determined from the reverted change |

Use a short, lowercase scope when it adds useful ownership context. Examples:

```text
fix(cache): preserve image prefixes after cancellation
perf(qwen): make prompt tokenization linear
feat(server): support constrained JSON output
docs: explain the release policy
feat(server)!: remove the legacy completion endpoint
```

Breaking changes must include `!` before the colon. Do not rely only on a pull
request body footer: the title is the canonical release signal retained by a
single-line squash merge.

## Version policy

Release versions follow Semantic Versioning. The compatibility surface is the
documented command-line interface, HTTP API and public Nix flake outputs. A
change is breaking when an existing documented use must be changed to keep
working.

Before `1.0.0`, Gufo uses this policy:

- `fix` and `perf` changes increment the patch version.
- `feat` changes increment the minor version.
- breaking changes increment the minor version.
- `1.0.0` is an explicit stability decision and is never selected implicitly.

After `1.0.0`, breaking changes increment the major version. A release uses the
largest required increment among all changes since the previous release.
Ordinary pull requests do not bump the version; release automation owns version
and changelog updates.

Git tags use `vX.Y.Z`. A released version is immutable. Development builds keep
their exact Git revision so diagnostics can identify the source independently
of the release version.

## Release flow

`version.txt` is the canonical release version. A daily GitHub workflow reads
the Conventional Commit history after the latest release and opens or updates
one release pull request. That pull request updates `version.txt`,
`CHANGELOG.md` and the release manifest. It receives the same CI as any other
change, plus the complete Devlab functional suite on Strix Halo (other pull
requests run its smaller PR tier). The workflow waits up to 160 minutes for the
pull request title, the repository checks and `devlab/functional`. If any of
them fails, errors or is still pending, it leaves the pull request open without
publishing a release and comments on it to notify the maintainers.

After the checks pass, the workflow squash-merges the release pull request and
immediately creates the immutable tag and GitHub Release. A manually dispatched
run follows the same process and is available for recovery. Do not edit the
version, changelog or release manifest in an ordinary feature or fix pull
request.

Nix keeps release identity and source identity separate. The default package is
a development build and reports `gufo version development (<revision>)`.
Official artifacts build the `release` package from a `vX.Y.Z` tag and report
`gufo version X.Y.Z (<revision>)`. Both package outputs retain the exact flake
revision; only the explicitly selected release output reads the semantic version
from `version.txt`.

After Release Please creates a GitHub Release, the same workflow sends its exact
`vX.Y.Z` tag to the Toolboxes repository. Toolboxes builds the release package
from that tag and publishes OCI images tagged `X.Y.Z`, `X.Y` and `latest`.
Rolling Toolboxes images remain separate: they follow Gufo `main` and use `edge`
and `sha-<revision>` tags.

In parallel with the OCI builds, Toolboxes calls Gufo's reusable Windows
workflow from `main`, with the same exact release tag as its source checkout.
It uses the existing Windows toolchain and host contract suite, sets the
release version and source revision explicitly, and checks the packaged
`gufo.exe --version` and platform diagnosis with only the bundled runtime on
`PATH`. This does not run the manual Windows Devlab GPU/model checks.

The called workflow uploads its artifact to the Toolboxes workflow run.
Toolboxes then attaches `gufo-vX.Y.Z-windows-x86_64.zip` and its `.sha256`
checksum to the existing Gufo GitHub Release. The ZIP includes `gufo.exe`,
the HIP and other runtime DLLs, kernel directories and notices. Extract the
entire archive before running the executable; use `gufo.exe serve` for the
HTTP server.

Merge the reusable Windows workflow in Gufo before enabling its Toolboxes
caller. Toolboxes needs `RELEASE_APP_CLIENT_ID` and
`RELEASE_APP_PRIVATE_KEY` available to its release workflow. The app must be
installed on Gufo with Contents write permission; Toolboxes' own
`GITHUB_TOKEN` cannot upload assets to Gufo. The upload job creates a fresh app
token after the build and does not overwrite existing release assets.
Toolboxes' manual release workflow accepts a tag for recovery runs.


## Extension API and build selection

Gufo's CLI, HTTP API and existing public flake outputs retain the release
compatibility rules above. The separately versioned source extension API and
the future experimental flake `extensions` argument are outside that surface.
A source extension API break alone does not require `!`; a CLI/HTTP/public flake
break made during migration does. Ordinary PRs may update the extension API
version and [its change log](../include/gufo/ext/CHANGES.md), independently of
release automation's ownership of Gufo's version/changelog.

Before 1.0, the API version is `0.MINOR.REVISION`: a minor bump may break source
compatibility, while revision bumps are additive. Extensions request a revision
floor within one minor series. From 1.0, minor bumps are additive and breaking
changes increment the major; retain an adapter from the preceding major for at
least 60 days where feasible and report deprecations at configure time. Fix
in-tree extensions in the same PR. Compatibility means rebuilding extension
source against a compatible Gufo release; there is no binary ABI promise.
Experimental hooks have no stability guarantee and graduate only by explicit
review after their own 60 unchanged days. API 1.0 remains gated by all RFC
criteria, including the measured text surface, external pilot, independently
pinned external text compatibility fixture, five additive API features and an
actual 60-day period. In-tree migrations alone do not establish external
compatibility. No 1.0 stability claim is made today.

Every declaration/contract change in the generated
[SURFACE](../include/gufo/ext/SURFACE) transitive closure, including forwarded
core/cache headers, and every public CMake/schema contract change requires the
`ext-api` PR label, an increased extension API version and a matching change-log
entry. The PR workflow runs `tools/ci/check-extension-api.py --base <base-sha>
--labels <json-label-list>`; the hosted suite checks inventory drift and policy
fixtures. The initial token check conservatively includes inline bodies,
initializers and CMake/schema statements; comments/whitespace do not trigger it.
Card10 will expand classification of stable/platform/experimental/private
headers and measure declarations; the initial inventory is no claim that all
future text interfaces are already stable.

`nix build` / `.#default`, every normal CMake preset and both Devlab build paths
use discovery: all supported in-tree extensions are selected, including ones
not in [extensions.lock](../extensions.lock). `.#release` and the release-tag
Windows ZIP use `GUFO_EXTENSIONS_FROM_LOCK=ON`. Toolboxes can select the same
policy through `gufo.override { extensionsFromLock = true; }`; publishing actual
Toolboxes images remains an external distribution deliverable. The lock is an
exact name list, including every family dependency, not an implicit dependency
closure. An unavailable platform or missing dependency in the lock is an error
in every configure mode, including plain discovery, so a stale lock also fails
development builds. `GUFO_EXTENSIONS` is also an exact semicolon list; `*`
selects discovery and an explicitly empty list selects none. An explicit list
and lock mode conflict, and an empty entry from a stray `;` is rejected.

Bootstrap: the current lock contains no migrated extensions. Existing legacy
models still build and serve through their existing routes in both modes; they
are not falsely reported as extension providers. Lock mode controls discovered
extension manifests only. The pilot and later model cards populate this list as
providers move, while Card14 adds metadata for remaining models before code
migration. `gufo --version` preserves the version line first, then reports API,
selection and extension provenance. `gufo diagnose --section extensions
[--json]` inspects provenance without model loads/HIP probes; ordinary human
`diagnose` also lists extensions. Existing diagnostic JSON schema is unchanged.

Hosted CPU checks build a real lock-mode executable, report unshipped
extensions and reject shipped/unshipped artifact-key, option-name and model-ID
collisions. They do not establish HIP release behavior or performance parity.
Run affected functional, numerical and performance gates on the actual shipped
HIP lock artifact as well as the default artifact when providers migrate;
include dialect selection/fallback and forced tool choice. Keep unshipped
discovery evidence separate. The RFC's strict-subset/default-Devlab inference
is not accepted as release qualification.
