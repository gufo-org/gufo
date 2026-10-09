# 06 · Slot leases and the mutation guard

**Milestone:** Common package · **Depends on:** 03, 04 · **Size:** L (split
candidate: leases, then guard) · **Affects:** nothing at runtime ·
**Status:** agreed

## Goal

Lease execution slots to requests, and make every overwrite of borrowed rows
safe and transactional.

## Scope

- **Lease lifecycle:** acquire, reuse the live frontier or restore, execute,
  publish checkpoints or commit, release; invalidate on any failure.
- **Slot generation counter:** bumped on every reassignment and reset, and
  checked by any pin on borrowed rows.
- **Mutation guard** (`MutationGuard::BeforeOverwrite`), in this order:
  1. find the checkpoints and readers that need the rows;
  2. preserve them into reserved backing, or retire eligible checkpoints;
  3. wait for transfers and reader pins on those rows;
  4. allow the overwrite.

  Preservation is synchronous here; card 09 moves most of it to idle time.
- **Paths covered:** prefill, decode, rewind, speculative rollback, reset,
  reassignment and destruction.
- **Cancellation:** at every step it releases leases, pins and reservations.
- **Lock scope:** metadata changes under a short lock; copies and waits
  outside it.

## Not in this PR

Device copies (the fake adapter copies host memory). Idle spill (card 09).

## Test first

With the fake adapter:

- each mutation path preserves needed rows before overwrite;
- an injected allocation or copy failure before mutation leaves published
  checkpoints valid and nothing overwritten;
- cancellation at each step leaks nothing (ledger back to zero);
- an adapter path that forgets the guard is detected in assertion builds by a
  generation mismatch on read.

## Step baseline

Guard cost per mutation when no checkpoint needs the rows (the common fast
path), and lease acquire and release cost.

## Done when

- [ ] Tests above pass, including under ThreadSanitizer for concurrent leases.

## Review focus

- The transactional sequence: what happens if step 3 waits on a pin held by a
  slow restore?
- What is the right split if this is too big for one review?

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [llama.cpp server slots and context checkpoints](https://github.com/ggml-org/llama.cpp/blob/41abbfd59/tools/server/server-context.cpp): how checkpoints are created and restored around slot reuse.
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM, SOSP 2023): copy-on-write before overwriting shared blocks.

## RFC

[Budget accounting and preservation before mutation](../RFC.md#budget-accounting-and-preservation-before-mutation) ·
[Capture and borrowing](../RFC.md#capture-and-borrowing)

## Review notes

