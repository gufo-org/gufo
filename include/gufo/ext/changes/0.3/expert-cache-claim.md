# Expert cache sized by serving

Additive, experimental: serving sizes a streamed model's expert cache together
with the RAM prompt cache.

- `TextRunnerResourceClaim::expert_cache` (`TextRunnerExpertCacheClaim`)
  names the runner's `expert_stream::Store`, the host memory to leave free
  after the expert cache (`headroom_bytes`), and an optional automatic
  prompt-cache budget per session while experts stream
  (`prompt_cache_share_per_session_bytes`). `AutomaticRetainedBytes(claim,
  sessions)` is the automatic prompt-cache budget the claim allows.
- `cache::kPreparedBackingBytes` names the backing `commit_backing`
  prepares before readiness, at most; `ComponentCacheResources::
  committed_backing_bytes` (optional, set by the HIP resources) reports what
  it committed.
- Once the sessions exist, serving commits the prompt cache's whole backing
  (beyond `kPreparedBackingBytes` through `prepare_capture`, where the
  resources provide it), then grants the store what it holds plus the host
  memory available, less the headroom and any budget still to commit, calls
  `Store::Fit(grant)`, and logs both budgets with the backing actually
  committed.
- Host pressure (#1275) is measured against the store's floor
  (`Store::FloorBytes()`, the larger of it and the free CMA pages) instead of
  the host reserve while experts stream: the expert cache fills the host down
  to that floor on purpose. Below it, the idle scheduler releases experts
  first (`Store::Release`, logged as `event=expert_cache_release`), then
  prompt-cache memory; once what it gave back is free again (the floor plus
  `Store::ReleasedBytes()` and 256 MiB available),
  `Store::Restore` lets the expert cache regrow to its grant
  (`event=expert_cache_restore`). New requests are refused below half the
  floor.
- `HostAvailableBytes()` (`gufo/ext/host_budget.hpp`) is the host memory
  available after cgroup limits and promised session state.

Migration: none.
