# Qwen3.8-Flash-Next TP=2

Two-host tensor-parallel serving over InfiniBand RDMA. It serves Q4 and full
Q8, AR and MTP, images, concurrent requests and history reuse. It needs
libibverbs and the build option `GUFO_ENABLE_TP2_RDMA` (off by default). Its
measurements are in [EXPERIMENTS.md](EXPERIMENTS.md#tp2), not in the published
one-host tables.

## Build

```sh
nix develop --inputs-from .#tp2-rdma -c cmake --preset gpu-tp2
nix develop --inputs-from .#tp2-rdma -c cmake --build --preset gpu-tp2 --target gufo

# Release binary with TP2
cmake --preset release -DGUFO_ENABLE_TP2_RDMA=ON
cmake --build --preset release --parallel 4
```

## Run

Start rank 0 first, and rank 1 within 30 s. `RANK0_ADDRESS` is where rank 1
reaches rank 0's TCP bootstrap. Both ranks need the same model files and the
same `--tp-control-token`, `--context`, `--prefill-chunk`, `--sessions`, MTP
settings and `--mmproj` use; the handshake names any setting that differs.

```sh
# rank 0: public HTTP server
build/gpu-tp2/gufo serve llm --model "$MODEL" --speculative mtp --mtp-model "$MTP" \
  --tp-world-size 2 --tp-rank 0 --tp-bootstrap-port 18515 \
  --tp-control-port 18516 --tp-control-token SHARED_TOKEN --sessions 4

# rank 1: worker only, no public HTTP port
build/gpu-tp2/gufo serve llm --model "$MODEL" --speculative mtp --mtp-model "$MTP" \
  --tp-world-size 2 --tp-rank 1 --tp-bootstrap-host RANK0_ADDRESS \
  --tp-bootstrap-port 18515 --tp-control-port 18516 \
  --tp-control-token SHARED_TOKEN --sessions 4
```

Sampling, streaming, stop sequences, tool calls, images (`--mmproj` on both
ranks), cancellation, `--request-timeout-ms`, history reuse and the disk cache
(`--cache-disk` on both ranks, each on its own disk) work as on one host.
Cancellations and timeouts take effect between model calls, so a long prefill
stops at its next chunk. Up to `--sessions` requests run at once, their
decoders batched; more wait up to `--max-pending`, the rest get HTTP 429.

Limits:

- A lost peer or a disagreement fails the request with HTTP 500, and a lost
  peer fails every later request, but the process keeps running and `/ready`
  stays green: restart both ranks.
- Each session holds a full-context state on each rank. Full Q8 (six `Q8_0`
  shards, which do not fit one host) at 262,144 tokens uses 87 GB of each
  host's 127 GB GPU memory with four sessions and 103 GB with eight.

## Design

**Partition.** Both hosts hold the whole checkpoint and each loads its share of
the split weights. Every routed and shared expert's intermediate dimension is
halved on quantization-block boundaries (a routed down projection in a 256-value
super-block type is refused at load); every rank computes all selected experts
on its half. Each GDN layer's 16 key heads and the 48 value heads that read them
(value head h reads key head h % 16) and each attention layer's two KV heads
with their query heads are split, and each rank keeps only its heads' KV cache
and recurrent state. The indexer, draft block, HC, embedding, PLE, LM head and
vision encoder are replicated.

**Exchange.** Each mixer block and each MoE ends in one exchange of partial
sums, 10 KiB per decode row. A stage kernel copies the partial to host memory
and raises a flag; a communicator thread writes it into one of the peer's three
receive windows with an RDMA write with immediate data, whose arrival is the
readiness signal; a one-wave kernel holds the stream until it arrives, and the
GPU adds the peer's partial straight from the window. The host queues a whole
forward without waiting. The rotating windows need no acknowledgement provided
the add of exchange k is queued before exchange k+2 starts. Each exchange
header carries the call's scope, an ordinal and the byte count; a mismatch
poisons the communicator. A sum of two operands keeps both ranks bit-identical;
TP2 and one host differ by reduction order. HIP graphs are off under TP2.

**Prefill overlap.** A prefill step of up to 4096 tokens runs as two trunk
batches a layer apart (one batch under 384 tokens), so each batch's exchange
crosses the link while the other computes; the results equal two steps of
those sizes bit for bit.

**Rank 1 as executor.** Rank 0 runs the ordinary scheduler, runner pool and
continuation cache through `TpMirroredRunner`. Just before each call that
changes model state, it sends the call to rank 1 over an authenticated TCP
control channel; rank 1's `TpExecutor` makes the same call on the same state,
and both ranks meet in the call's exchanges. Token selection reads only logits
and stays on rank 0. A request opens with `kSingle` (prompt, sampling
configuration and prepared images) and closes with `kEnd`. Mirrored calls:

- prefill chunks, single and batched advances, and single and batched
  multi-token MTP cycles. A cycle carries rank 0's sampler draw state so both
  ranks draw alike; a batch also carries the draft count rank 0 chose from its
  own cycle timings;
- state resets, and binding a state to its request's prompt context (images
  can reset the state);
- snapshot, restore, prefix reuse and cancellation preparation, each
  acknowledged by rank 1 before rank 0 continues; snapshot bytes stay on each
  host, and the budget is the smaller host's.

Every model call binds the communicator's scope to its instruction's index on
both ranks. Rules the compiler does not enforce: every state-changing call must
pass through the wrapper; a request's cancellation check is never passed to the
model session, which could strand rank 1 inside an exchange; TP2 must not
change the cache's boundaries.

**Disk cache.** Each rank keeps its own half of a snapshot on its own disk.
Rank 0's continuation disk store decides what is saved, restored and evicted,
as on one host. Its file holds rank 0's half behind a header that names a
random file key; when it writes the file, rank 1 persists its half under that
key in the background. Restoring a file restores both halves, rank 1's from
its own directory, acknowledged before rank 0 continues. Rank 1 bounds its
directory by least-recent use within its own `--cache-disk-bytes`; a missing,
corrupt or foreign half fails the restore on rank 1, which rank 0's store
treats as a miss (the state is reset on both ranks and prefilled). TP2 files
carry the rank in their cache identity, so neither a one-host server nor the
other rank restores them. Within one process, a file of a request rank 1
rejected is a miss, and a restore before the verdict depends on it.

**Agreement.** Both ranks digest every call and its result; `kEnd` carries rank
0's digest and call count, and rank 1 fails the request on a difference. Under
greedy decoding rank 1 also checks its own argmax against every token rank 0
advances. A capture carries the request's digest so far, and rank 1 captures
only a state both ranks reached the same way. Snapshots and released states
are reused at once, as on one host; a request that reuses what an unjudged
request left fails too if rank 1 later rejects that request, and a rejected
request's states and snapshots are not reused.

## Tests and probes

- `tp_control_test`, `tp_executor_test` (label `tp2`, in the CPU suite): the
  protocol, and the real scheduler and pool on rank 0 against an executor on
  rank 1 with a toy model, for every request kind, divergence and reuse rule.
- `qwen38_flash_next.tp_partition` and the attention, projection and routed
  operator tests: the split against the full kernels.
- `qwen38_flash_next_tp_probe` (`gpu-tp2`): prompt, decode and logit
  comparison on both ranks; `--split N` checks that a prefill does not depend
  on its steps (every layer's MoE input and output, and the logits, bit for
  bit), `--decode-tail K` compares decoding the last K tokens with
  prefilling them.
- `qwen38_flash_next_tp_batched_probe` (`gpu-tp2`): batched against serial
  decoding on both ranks without the serving stack; `--allreduce-bench N`
  times the exchange.
- `tools/bench/model-bench.py` launches both ranks through a `gufo.tp2`
  overlay in `bench.json` (`remote_host`, `bootstrap_host`, ports, token,
  container and binary); it runs single-user tables only.

## Qualifying a change

On both hosts, from one commit (record it and the binary hash):

1. The format check on a fresh copy of the tree, and the CPU suite and TP2
   tests above with no skips.
2. Two hosts, Q4 and Q8, AR and MTP, greedy: outputs repeat, AR and MTP agree,
   and prompts longer than one prefill chunk work.
3. Seeded sampling reproduces; streaming, stop sequences, a tool call and
   client disconnects during decode and prefill behave as on one host.
4. A multi-turn cached chat restores its whole prompt on replay; cached token
   counts match one host (outputs may differ after a few turns).
5. Two, four and eight concurrent requests, AR and MTP, each equal their
   one-at-a-time output, seeded sampled ones too; the queue bound returns 429;
   a long prefill runs beside decoders.
6. Images: a request, its cached repeat, a follow-up and two images in one
   request match one host's answers.
7. Disk cache (`--cache-disk` on both ranks): after both ranks restart,
   `tools/serving/check-continuation.py --restore` restores its greedy cases
   from disk with the same outputs.
8. Killing rank 1 mid-request returns 500 promptly.
9. For speed, report the median of several warm requests.
