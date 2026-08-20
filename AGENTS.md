# AGENTS.md

## Supported Platform

Linux x86-64 on AMD Strix Halo (`gfx1151` GPU and XDNA2 NPU) is the only
supported production target.

## Build & Test (Nix)

Build and test with Nix only. Direct host builds and Makefiles are unsupported.

```sh
nix build                          # build default package (gfx1151 + XRT)
./result/bin/strix                 # run hardware probe
./result/bin/strix-server          # run server
nix build .#checks.x86_64-linux.pr # canonical PR test command (all gates)
nix develop                        # dev shell
```

- `git add` before `nix build` — Nix sees only tracked files.

## Development

- Use `gh` CLI to retrieve and update issues content. Verify if installed and configured with `gh auth status`, and use it unless the user explicitly says otherwise.
- Follow Conventional Commits format with single-line commit messages (e.g., `feat(scope): summary (#issue)`, `fix(scope): summary (#issue)`).
- Prefer Jujutsu (`jj`) over Git when possible. Verify that it is available by running `jj version`; fall back to Git if it is unavailable or if the user explicitly requires Git.

