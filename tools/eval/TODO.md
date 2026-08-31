# gufo-agent-eval — what is missing

Against [plan.md](../../plan.md) and issue #153. Ordered by what blocks the
most.

## Blocking a meaningful run

- [ ] **Result file.** `run` prints to the terminal and discards everything.
      #153's CLI is `run --output result.json`. Needs `result.py`: versioned
      schema, per-task and aggregate, written to `--output`.
- [ ] **Sanitizer.** No credentials, endpoint addresses, secrets, or user
      paths in any emitted artifact. Acceptance criterion; nothing enforces it.
- [ ] **Suites / tiers.** `--suite coding-smoke` is in the spec; `run` takes
      one task name only. Needs `suite.py`, a smoke tier, and a full tier.
- [ ] **`compare`.** Paired task-level diff between two result files. The
      whole point is comparing `gufo serve` against llama.cpp.
- [ ] **Timeout policy.** `agent.timeout_sec = 900` is below the *median*
      task duration of the comparable reference run, so a full run would
      measure timeouts, not capability. Reference runs used 3 h.

## Correctness and safety

- [ ] **Sandbox negative tests.** Host file read, credential read, write
      outside the workspace, external network, orphaned process survival.
      None written.
- [ ] **cgroup v2 resource limits.** `cpus`, `memory_mb`, `storage_mb` are
      recorded in manifests but not enforced. A runaway agent shell is
      bounded only by wall-clock.
- [ ] **Token accounting.** Usage reads `in=0` on most tasks. Pi reports
      cumulative usage only at completion, so truncated runs lose it. Verify
      on a full run before trusting the numbers.
- [ ] **Tests in the PR gate.** `tests/eval/` runs in no flake check, so the
      manifest guard can rot silently.

## Fidelity to #153

- [ ] **Endpoint preflight.** Blocked: the #203 fixtures do not exist. Decide
      between a minimal preflight here and reopening #203.
- [ ] **Attempt budget.** One attempt, no retry. Upstream defaults to two and
      reports pass@2.
- [ ] **Repeat-run stability.** Measure, then declare a tolerance.
- [ ] **Split jails.** Pi and the workspace share one jail, so the agent
      shell can reach the endpoint. Needs an exec bridge to close.

## Tasks that start but cannot pass

- [ ] **`fix-ocaml-gc`** — vendor the broken OCaml checkout as a fixed-output
      derivation.
- [ ] **`mteb-retrieve`** — vendor the embedding model; `mteb` is not in
      nixpkgs.
- [ ] **`mailman`** — expects an init system the jail does not provide.
      Likely a task rewrite rather than a port.

## Housekeeping

- [ ] Decide whether `plan.md` stays at the repo root, moves under `docs/`,
      or is dropped now that the README carries the durable reasoning.
- [ ] Update the PR description; it still lists as absent several things that
      have since landed.
- [ ] Decide whether the suite eventually moves to its own repository.

## Done

- [x] Sandbox: bubblewrap, copied rootfs, namespace isolation
- [x] `endpoint-only` networking via a unix-socket bridge
- [x] Pi packaging, per-run config generation, trajectory parsing
- [x] All 20 tasks ported; 20/20 rootfs and verifier closures build
- [x] 20/20 tasks start the agent against a live server
- [x] Apache-2.0 compliance for the redistributed suite
- [x] `nix run .#eval-agent`, `nix develop`, help and task listings
- [x] Cleanup on kill: PDEATHSIG on the forwarder, stale scratch sweep
