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
- Once the sessions exist and the prompt cache committed its prepared
  backing, serving grants the store what it holds plus the host memory
  available, less the headroom and the prompt-cache budget beyond the
  prepared backing, calls `Store::Fit(grant)`, and logs both budgets with
  the backing actually committed.
- `Store::Fit` replaces `Store::SetReserve`: it takes the prefetch stage from
  the grant, shrinks the cache to it and seeds the room left with the ranking
  last given to `Seed`.
- `HostAvailableBytes()` (`gufo/ext/host_budget.hpp`) is the host memory
  available after cgroup limits and promised session state. It replaces the
  store's `expert_stream::AvailableMemory()`.

Migration: a runner that called `SetReserve` once its sessions existed
declares `expert_cache` instead; callers of `expert_stream::AvailableMemory()`
call `HostAvailableBytes()`.
