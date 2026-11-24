# Security policy

Report vulnerabilities privately, not in public issues, pull requests or
discussions.

## Report a vulnerability

Open a private report from the repository's
[Security tab](https://github.com/gufo-org/gufo/security/advisories/new).
Include the Gufo version (`gufo --version`), how to reproduce the problem and
what an attacker gains. The maintainers reply in the report, agree a fix and
disclosure date with you, and credit you in the advisory unless you ask
otherwise.

## Supported versions

Fixes go into `main` and ship in the next release. Older releases don't get
backports, so update to the latest release.

## Scope

In scope:

- crashes, memory corruption or code execution from a crafted request to
  `gufo serve`;
- crashes, memory corruption or code execution from a crafted model file;
- official extensions, which ship with Gufo.

Out of scope, because these are documented limits:

- anyone who can reach the server's port using it: Gufo has no
  authentication; see [the server warning](docs/server.md);
- community extensions doing harm: an extension is native code that runs with
  your permissions; see [extensions](docs/extensions.md). Report those to the
  extension's author.
