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
change. The workflow waits for the pull request title and repository checks,
and leaves the pull request open without publishing a release if either check
fails.

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
