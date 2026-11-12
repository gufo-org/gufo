# Versions

Gufo follows Semantic Versioning, so a version number tells you whether an
upgrade can break your setup. This page explains what counts as breaking and
where releases are published.

## What a version promises

The promise covers the documented command line, HTTP API and public Nix flake
outputs. A change is **breaking** when something these pages document stops
working without a change on your side.

Before 1.0.0:

- fixes and speedups raise the patch version: 0.31.0 to 0.31.1;
- new features and breaking changes raise the minor version: 0.31.1 to 0.32.0.

After 1.0.0, breaking changes raise the major version. The
[changelog](../CHANGELOG.md) lists every release and marks breaking changes.

Extensions have a separate version; see the
[extension API](extension-api.md#versions).

## Where releases go

| Channel | Tags | Contents |
| --- | --- | --- |
| [GitHub releases](https://github.com/gufo-org/gufo/releases) | `vX.Y.Z` | Source, the Debian package and the Windows ZIP |
| Container images | `X.Y.Z`, `X.Y`, `latest` | Releases; `latest` is the newest |
| Container images | `edge`, `sha-<revision>` | Development builds, updated continuously |
| Nix flake | `github:gufo-org/gufo/vX.Y.Z` | Any release or revision |

A released version never changes. Development builds report their exact Git
revision in `gufo --version`, so a bug report says what ran.

## Which documentation to read

[gufo.blog/docs](https://gufo.blog/docs) keeps two kinds of versions, and the
version picker switches between them:

- the **ten newest releases**, each frozen as it was released; `/docs` opens
  the newest one;
- a **development** version that follows `main`, like the repository's `docs/`
  folder, and can describe features no release has yet.
