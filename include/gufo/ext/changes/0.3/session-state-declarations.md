# Session state declarations

Additive: a model declares the state one session holds, and the core
allocates it and gives the bytes the runner claims from the same list.

- `gufo/ext/state.hpp` (new) forwards `src/mem/state_declaration.hpp`, which
  builds without HIP:
  - `mem::StateKind`: `kContext` (rows that grow with the context, at a fixed
    address, mapped as written and charged to the session's
    `mem::StateAccount`), `kWhole` (a region backed whole at creation) and
    `kFixed` (a zeroed device allocation of fixed size).
  - `mem::StateBuffer`, `mem::StateBufferId` and `mem::StateBytes`
    (`kSize`: a region's size; `kPages`: its whole 2 MiB pages, what it
    reserves and its account charges where state is demand-mapped).
  - `mem::StateDeclaration(margin_tokens)`: `Context(shape, tokens_per_row)`,
    `Whole(shape)` and `Fixed(bytes)` return each buffer's id in declaration
    order; `buffers`, `buffer`, `margin_tokens`, `has_context`,
    `Bytes(count)`, `Bytes(id, count)` and `ContextBytes(count)`. Invalid
    shapes (no rows, no row bytes, no segments, overlapping segments) throw
    `std::invalid_argument`.
  - `mem::RegionSpanBytes(shape)`: the bytes a region allocates without
    demand mapping.
- `gufo/ext/experimental/state_memory.hpp` also forwards
  `src/core/hip/session_state.hpp`: `mem::SessionState::Create(declaration,
  error)` allocates every buffer (`kContext` buffers as regions charged to a
  new account with the declaration's margin, `kWhole` buffers as regions
  backed whole, `kFixed` buffers through `hip::AllocateDevice`, cleared and
  synchronized with the null stream before it returns), and exposes `data`,
  `region`, `account` and `allocated_bytes`. Destruction releases every
  buffer; call it where no work that uses them is in flight.

A text runner that allocates its sessions this way returns the session's
`SessionState::account()` from `TextRunnerState::state_account()` and fills
`TextRunnerResourceClaim::per_request_state_bytes` and
`demand_mapped_state_bytes` from the declaration (`Bytes` plus any state it
allocates itself, and `ContextBytes` where `mem::DemandMapped()`, zero
otherwise). Counting `kPages` matches what the ledger charges.

In-tree, Qwen3.8 Flash-Next, Qwen3.8 27B and DeepSeek V4 Flash declare their
session state this way. Their claims are unchanged: Flash-Next and 27B count
sizes and DeepSeek V4 Flash pages, as before.

Migration: none; `mem::Region::Create` and `mem::StateAccount` stay as they
were.
