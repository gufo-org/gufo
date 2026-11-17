# Shared session state

Additive: sessions may share their demand-mapped state when the configured
sessions do not all fit at full context (#715).

- `TextRunnerResourceClaim::demand_mapped_state_bytes` (`src/ext/text.hpp`)
  reports the part of `per_request_state_bytes` held in demand-mapped regions.
  When the configured sessions do not fit at full context but their fixed
  state does, with room for one session at full context, the host loads them
  sharing that memory (`mem::FitSessions`) instead of refusing the
  configuration. Missing or zero keeps the full-context check.
- `TextRunnerState::state_account()` returns the session's
  `mem::StateAccount`, whose regions hold that state; the default returns
  null. The host claims a request's tokens on it before the request writes
  them, sets it to the live frontier afterwards and trims an idle session's
  pages when another request needs them.
- `gufo/ext/experimental/state_memory.hpp` now also provides
  `mem::StateAccount`, `mem::StateReservation`, `mem::FitSessions`,
  `mem::RegionPageBytes`, `mem::DemandMapped()` and the ledger's promise cap
  and charges. `mem::Region::Create` takes an optional account and the tokens
  each row holds; `Region::TrimTo` releases pages past a row count, and
  `Region::AheadLimit` bounds map-ahead to the session's need.
  `mem::RegionShape` moved to `src/mem/account.hpp`, which the region header
  includes.

Migration: none. A runner with demand-mapped state that wants sessions to
share it creates one `mem::StateAccount` per session (its margin the rows it
backs past a frontier), passes it to `Region::Create`, returns it from
`state_account()` and reports the regions' bytes in
`demand_mapped_state_bytes`.
