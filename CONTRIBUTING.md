# Contributing to Gufo

Gufo is maintained in the maintainers' spare time. This page says what belongs
in Gufo, what belongs in an extension, and what helps a pull request land.
Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md); report
vulnerabilities as the [security policy](SECURITY.md) describes.

## What belongs where

The maintainers focus on the core inference engine: caching, scheduling, state
memory, the server and the extension runtime. Models are extensions, and Gufo
keeps one or two official models per use case. The full rule is in
[core or extension](docs/writing-extensions.md#core-or-extension).

- **Core fixes and improvements** are welcome. For a large change, such as a
  new subsystem, a new dependency or an extension API change, start a thread
  in [Discussions](https://github.com/gufo-org/gufo/discussions/categories/ideas)
  first. It tells the community what you plan and gathers feedback before you
  build; the
  [Rust host layer RFC](https://github.com/gufo-org/gufo/discussions/864) is a
  good example. An issue, or a draft pull request to discuss in review, works
  too.
- **A new model** starts as an extension in your own repository. Follow
  [writing an extension](docs/writing-extensions.md); the
  [catalogue](https://gufo.blog/extensions) lists it once it's public.
  Becoming official follows the
  [support policy](docs/internal/extension-distribution.md#support-states).
- **Changes to an official model** are welcome with the quality and speed
  evidence that model's tests and reports ask for.

## Why a pull request waits

An open pull request usually waits for one of these:

- **An outside dependency.** It needs a change in a project Gufo doesn't own,
  and that project has to ship it first.
- **Missing evidence.** Quality, functional or timing results are missing or
  show a regression.
- **Review time.** No maintainer has had time to review it. Maintainers pick
  work by size and readiness, not only importance, so a small self-contained
  pull request can land before a larger one.
- **Scope.** It adds to the core something that belongs in an extension.

To help one move, work on its blocker, add the missing evidence, or test it on
your hardware and post the results. You can push commits to a branch in your
fork and propose them to the pull request's author, as you would for your own
work.

## Before you open a pull request

1. Read [development](docs/internal/development.md) and
   [testing](docs/internal/testing.md). [AGENTS.md](AGENTS.md) holds the rules
   that people and coding agents follow.
2. Run the smallest checks that cover your change, and the format check for
   C++:

   ```sh
   nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
   ```

3. Title the pull request as a Conventional Commit, following the
   [release policy](docs/internal/releasing.md).
