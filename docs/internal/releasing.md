# Releasing

How pull request titles drive versions, how a release is cut and published, and
how the extension API is versioned alongside. What a version promises to users
is in [versions](../versions.md).

## Pull request titles

Gufo squash-merges every pull request, so its title becomes the one commit on
`main` and the input to release automation. Titles have this form:

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

Use a short, lowercase scope when it adds ownership context. Examples:

```text
fix(cache): preserve image prefixes after cancellation
perf(qwen): make prompt tokenization linear
feat(server): support constrained JSON output
docs: explain the release policy
feat(server)!: remove the legacy completion endpoint
```

Mark a breaking change with `!` before the colon; a footer in the body is lost
in the squash. `tools/ci/check-pr-title.py` enforces this table, and its test
reads the table and examples from this page.

## Versions

Gufo follows Semantic Versioning over the documented CLI, HTTP API and public
Nix flake outputs. Before 1.0.0, `fix` and `perf` bump the patch, and `feat` and
breaking changes bump the minor. After 1.0.0, breaking changes bump the major;
1.0.0 itself is an explicit decision, never automatic. A release takes the
largest bump among its changes.

Ordinary pull requests never edit `version.txt`, `CHANGELOG.md` or the release
manifest; release automation owns them. Tags are `vX.Y.Z` and never move.

## Cutting a release

1. A maintainer runs **Release automation** (`release-please.yml`) on `main`
   from the Actions tab; it has no schedule.
2. It opens or updates the release pull request, which bumps `version.txt`,
   `CHANGELOG.md` and the release manifest from the Conventional Commit history.
   That pull request runs the normal CI plus the full Devlab functional tier.
3. The workflow waits up to 160 minutes for the title check, repository checks
   and `devlab/functional`. On any failure, error or timeout it leaves the pull
   request open and comments on it.
4. When everything passes, it squash-merges, creates the tag and the GitHub
   Release, and sends the tag as a `gufo-release` dispatch to Toolboxes and to
   the [website](https://github.com/gufo-org/website).

Nix keeps release and source identity apart. The default package reports
`gufo version development (<revision>)`; the `release` package, built from a
tag, reads `version.txt` and reports `gufo version X.Y.Z (<revision>)`.

## Publishing artifacts

Toolboxes' `publish_release.yml` handles the dispatch:

- **Container images:** builds from the tag and pushes `X.Y.Z`, `X.Y` and
  `latest`. Rolling images follow `main` as `edge` and `sha-<revision>`.
- **Windows ZIP:** calls Gufo's reusable `windows-build.yml@main` with the tag
  as source, sets the version and revision, and checks `gufo.exe --version` and
  diagnostics with only the bundled runtime on `PATH`. It attaches
  `gufo-vX.Y.Z-windows-x86_64.zip` and its `.sha256` to the Gufo release. This
  doesn't run the Windows Devlab GPU checks.
- **Debian package:** builds `gufo_<version>-1_amd64.deb` for Debian 13,
  verifies it in a fresh container, and attaches it with its checksum.

- **Documentation:** the website's `sync-docs.yml` copies the public docs at
  the tag into a new version of [gufo.blog/docs](https://gufo.blog/docs),
  keeping the 10 newest releases, and deploys the site. Between releases,
  `docs-website.yml` refreshes the site's `development` version whenever a
  push to `main` changes the public docs.

Uploading to the Gufo release needs the release GitHub App, installed on Gufo
with Contents write, through `RELEASE_APP_CLIENT_ID` and
`RELEASE_APP_PRIVATE_KEY` in Toolboxes. Uploads never overwrite existing
assets; the workflow's manual inputs rebuild one release, or only its Windows or
Debian assets, for recovery.

## Extension API versioning

The extension API is versioned separately and sits outside Gufo's release
surface: an extension API break alone doesn't need `!`, while a CLI, HTTP or
public flake break does. The flake's experimental `extensions` argument is
outside the surface too.

The version is `0.MINOR.REVISION`. A revision only adds; a minor bump may break
source compatibility. Extensions request a floor within one minor series. After
1.0, minor bumps are additive and breaking changes bump the major, with an
adapter for the previous major for at least 60 days where feasible and
configure-time deprecation reports. Compatibility means rebuilding from source;
there is no binary ABI promise. Experimental hooks graduate only by explicit
review after 60 unchanged days. API 1.0 stays gated by the RFC criteria (the
measured text surface, an external pilot, a pinned external compatibility
fixture, five additive features and a real 60-day period); no 1.0 claim is made
today.

**Change records.** Every declaration or contract change in the
[SURFACE](../../include/gufo/ext/SURFACE) closure (forwarded core and cache
headers included), the manifest schema or the public CMake contracts needs the
`ext-api` label and one new, descriptively named record under
`include/gufo/ext/changes/<major>.<minor>/`, as the
[record instructions](../../include/gufo/ext/changes/README.md) describe.
Records are never edited, renamed or removed. In-tree extensions are fixed in
the same pull request.

**Revision.** [api-series.json](../../include/gufo/ext/api-series.json) sets
`major`, `minor` and `revision_base`; the revision is `revision_base` plus the
number of records in the active series, excluding `baseline.md`. CMake
generates the version header from it, also without Git metadata. Two parallel
additive pull requests each add a record, and merging both advances the revision
twice with no shared counter to reconcile. Branch revisions are provisional: an
extension that needs an unmerged API pins that Gufo revision and picks its floor
after the merge. A breaking change explicitly advances the series in
`api-series.json` and adds a record in the new series; that transition is
reviewed, never inferred.

```sh
python3 tools/ci/check-extension-api.py --write        # after header or include changes
python3 tools/ci/check-extension-api.py --changelog /tmp/gufo-extension-api-changes.md
```

In pull requests, CI runs `check-extension-api.py --base <base-sha> --labels
<labels>`. Its token check counts inline bodies, initializers and CMake and
schema statements; comments and whitespace don't trigger it.

## Extension selection in builds

| Build | Selection |
| --- | --- |
| `nix build` (`.#default`), CMake presets, Devlab | Discovery: every supported in-tree extension |
| `.#release`, the release-tag Windows ZIP | `GUFO_EXTENSIONS_FROM_LOCK=ON`: exactly [extensions.lock](../../extensions.lock) |
| Toolboxes images | The default package with the release version; `gufo.override { extensionsFromLock = true; }` would select the lock |

The lock is an exact name list, family dependencies included, validated in
every configure. Hosted CPU checks build a real lock-mode executable and reject
artifact-key, option-name and model-ID collisions between shipped and unshipped
extensions. They don't establish HIP behavior or performance: when a change
affects providers, run the functional, numerical and performance gates on the
shipped lock-mode HIP build as well. [Extension build](extension-build.md)
describes selection in full.
