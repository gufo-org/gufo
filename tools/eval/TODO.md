# gufo-agent-eval — what is missing

Against [plan.md](../../plan.md) and issue #153.

## Blocked on work outside this harness

- [ ] **`mteb-retrieve` cannot pass.** `mteb` 1.36.8 requires
      `datasets<3.0.0`; nixpkgs ships `datasets 4.5.0`, and pinning an older
      one cascades through transformers and sentence-transformers. It also
      needs `pytrec-eval-terrier`, a C++ extension absent from nixpkgs. The
      model would additionally need vendoring as a fixed-output derivation.
      Excluded from the `known-good` tier.
- [ ] **`mailman` cannot pass.** Postfix and Mailman expect an init system;
      the jail runs the agent as pid 1 with nothing supervising daemons. The
      verifier also needs an importable `mailman` library, and nixpkgs
      carries it only as a top-level application. Needs a task rewrite, not a
      port. Excluded from the `known-good` tier.
- [ ] **Endpoint preflight convergence.** A minimal preflight exists,
      covering what Pi actually exercises. #153 specifies the #203 fixture,
      which does not exist; converge when it lands.

## Needs a long run to answer

- [ ] **Token accounting.** Usage reads zero on truncated runs, because Pi
      reports cumulative usage only at completion. Confirm it is correct on a
      full attempt before trusting the aggregate figures.
- [ ] **Repeat-run stability and tolerance.** Requires several full runs
      (12-26h each on this hardware) before a tolerance can be declared.

## Deliberately not done

- [ ] **Split jails.** Pi and the workspace share one jail, so on an
      `endpoint-only` task the agent's shell can also reach the endpoint. A
      real split needs a Pi extension proxying every tool -- bash, read,
      write, edit -- through a bridge, since those run in-process. A shell
      wrapper alone would look like a boundary without being one. The current
      boundary is pinned by a test: the shell reaches the endpoint and
      nothing else.

## Decisions for a human

- [ ] Whether `plan.md` stays at the repo root, moves under `docs/`, or is
      dropped now that the README carries the durable reasoning.
- [ ] Whether the task suite eventually moves to its own repository.
- [ ] Whether publishing `solution/` and hidden fixtures in a public repo is
      acceptable. Apache-2.0 permits it; the canary string signals upstream
      would rather it did not spread.

## Done

- [x] Sandbox: bubblewrap, copied rootfs, namespace isolation
- [x] `endpoint-only` networking via a unix-socket bridge
- [x] Pi packaging, per-run config generation, trajectory parsing
- [x] All 20 tasks ported; rootfs and verifier closures build; all start
- [x] `fix-ocaml-gc` unblocked by vendoring its broken checkout
- [x] Apache-2.0 compliance for the redistributed suite
- [x] Versioned result documents, written by `run --output`
- [x] Sanitizer over every artifact, with tests
- [x] Suite tiers: smoke, known-good, full
- [x] `compare`, with comparability checked rather than assumed
- [x] Minimal endpoint preflight; unsupported endpoints abort
- [x] cgroup v2 resource limits, recorded as unenforced where unavailable
- [x] Attempt budget via `--attempts`, reported as pass@N
- [x] Three-hour default timeout, matching the reference runs
- [x] Sandbox negative tests: 16 cases, all passing
- [x] PR gate: `nix build .#checks.x86_64-linux.eval-agent`
- [x] `nix run .#eval-agent`, `nix develop`, help and task listings
- [x] Cleanup on kill: PDEATHSIG on the forwarder, stale scratch sweep
