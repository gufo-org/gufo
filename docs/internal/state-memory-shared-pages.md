# Shared state pages and the remap barrier: evidence

Measurements behind [shared pages](state-memory.md#shared-pages) and
[remapped addresses](state-memory.md#remapped-addresses). Host: Strix Halo
gfx1151, 128 GB, ROCm 10.1 (HIP 7.16), Linux 7.3.3, `amdgpu` with the MES
scheduler, November 2026. Main is `029a1569`; both sides are Nix release
builds. The probes are HIP programs in
[state-memory-shared-pages/](state-memory-shared-pages/), each with its log.

## Remapped addresses keep their old page

`rules.hip` gives every page handle its own constant, remaps an address, and
writes through it with a kernel. It then reads every handle through a fresh
**witness** mapping, so the report says which page each write reached
([log](state-memory-shared-pages/rules.txt)).

| Sequence, with no pause | Result |
| --- | --- |
| Replace a region page whose old page stays parked | The write reached the old page; the park saw it |
| Map a parked page where a session had its own | The session still read its own page |
| Use a park address for another page | Reads returned the first page |
| Unmap, then map and write | 0 of 50 writes reached the new page |
| Unmap and release the old page, then map and write | 0 of 50; the writes went to memory the driver had freed |

No call made the new page reachable: a second unmap, an access change,
`hipDeviceSynchronize`, mapping and unmapping another address, freeing and
reserving the address, or a 4 GiB kernel on other memory. Only idle time did,
before the unmap or after the remap.

`idle2.hip` remaps eight addresses and a ninth as a canary, 300 to 3,000 times
per setting, each address cycling through five handles so an old page never
looks new ([log](state-memory-shared-pages/idle2.txt)):

| Device idle after the remap | Batches with every address new |
| --- | --- |
| None | 44 of 300 |
| 20 us | 261 of 300 |
| 60 us | 296 of 300 |
| 100 us | 300 of 300 |
| 150 us | 3,000 of 3,000 |
| 150 us, 3,000 other pages touched first | 3,000 of 3,000 |
| 150 us, idle before the unmap instead | 3,000 of 3,000 |

- With another thread running short kernels at 60 us gaps, 400 us of waiting
  left 7 of 300 batches old, and one took 20 ms. With kernels back to back
  the old pages stayed for more than 200 ms.
- The canary is no proof. With 3,000 other pages touched after the remap it
  showed its new page while another address still showed its old one in 142
  of 300 batches: translations leave one by one.
- The copy engine shares the fate
  ([`sdma.hip`](state-memory-shared-pages/sdma.hip),
  [log](state-memory-shared-pages/sdma.txt)): 0 of 20 copies reached the new
  page at once, 28 of 30 after 100 us, 400 of 400 after 200 us, and 1 of 30
  after 1 ms with kernels running on another thread.
- A `hipMemcpy` read can disagree with a kernel read of the same address.
  The region test caught the first failure only once its checks used
  kernels.

The barrier therefore waits for the device, keeps it idle for 400 us (twice
the longest time any remap needed), and holds back the threads that submit
device work on their own. It is no proof against device work it cannot see.

## What needed the barrier

- **Page copies on a rewind.** The first design copied a shared page through
  one scratch address, mapped again for every page. From the second copy on,
  the data went to the previous page. `mem_region_test` now copies one page
  per segment back to back and reads the results with a kernel.
- **Main's own remaps.** A trimmed region that grows again
  (`Region::TrimTo`, then `Ensure`) mapped and cleared pages through
  addresses that had just lost theirs; so did the cache's lendable range
  after a host-pressure release, and the pass scratch on every wide pass.
  The earlier region test passed because its `hipMemcpy` reads went through
  the same stale translation as the clear.

## Flash-Next: rotation on three sessions

One server, `--sessions 3`, UD-Q4_K_XL with MTP, a disk cache seeded with two
conversations. A decoder streams throughout. Conversation A (76K tokens) and
B (34K) take turns with a short request C, so every turn takes over a session
that holds another conversation and restores its own checkpoint. Four runs in
the order candidate, main, main, candidate; all twelve outputs are identical
across the four.

| | Main | Candidate |
| --- | --- | --- |
| TTFT of C taking over a session that holds A or B | 0.281-0.286 s | 0.151-0.190 s |
| Longest decoder gap during that takeover | 291-307 ms | 175-217 ms |
| RAM restore of A, 2.2 GB | 23.8-25.8 ms | 7.4-9.2 ms |
| RAM restore of B, 1.1 GB | 12.4-12.8 ms | 4.1-6.0 ms |
| Disk restore of A or B into a fresh session | 0.42-0.68 s TTFT | 0.35-0.71 s |
| Remap barriers in the run | | 29, 11.7 ms in all, none gave up |

With `--cache-ram-bytes 3221225472` the two builds answer every request the
same way: the same reuse, the same restores from disk, the same refusals.

A cold prefill is unchanged: 76K tokens into a fresh session took 41.19 to
41.29 s against 41.14 to 41.19 s, and 34K tokens into a session whose pages a
checkpoint shared took 20.18 to 20.21 s against 20.15 to 20.17 s.
`gufo bench -p 2048 -n 128 -d 0,32768` gave 1,885 against 1,880 tokens/s of
prefill and 63.6 against 63.6 tokens/s of decode.

## Under memory pressure

The server reads a `/proc/meminfo` whose free memory is lowered by an offset,
so no memory is really taken. With free memory under the state floor, every
new state page comes from the cache: sessions took pages back by evicting
checkpoints that held parked pages (6 evictions in one run, 25 in another).
With available memory then under the 4 GiB reserve, one relief pass freed
what main needed two to four passes for. All 17 requests succeeded, with the
outputs of the run without pressure and of main.

## Defects the first design had

Besides the page copies above. Each cost a rebuild; the numbers are from the
runs that found them.

- **Shared payloads ate the metadata headroom.** They were charged on top of
  the pool's committed blocks. With a 3 GiB cache the ledger filled, disk
  writes failed with `bad_alloc`, and conversations were prefilled again
  (TTFT 50-80 s where main restored in 1-3 s). Shared payloads and committed
  blocks now share the pool's capacity, and `CanFit` counts both.
- **The disk writer read shared pages on its stream.** Each 8 MiB piece
  completed behind a decode round: 2.2 GB took 6 to 13 s to publish against
  2 to 6 s on main, and a deep checkpoint missed the next request.
- **Reading them on the CPU stalled the model.** Mapping each page for the
  CPU from the writer's thread was fast enough, but the model's own passes
  stalled for 100 to 250 ms while it ran. The writer now copies shared pages
  to its staging memory on the copy engine; publication is as early as
  main's or earlier.
- **Short checkpoints were never restored.** A checkpoint under one
  256-row chunk has no chunk to map; the restore asked the ledger for a
  zero-byte reservation, which throws, and fell back to a cold prefill
  without a word. Only the `reuse=` counts showed it: 2,010 cold prefills
  where main had 600.
- **An unmap beside another thread's copies.** A request that took over a
  session unmapped one page per buffer while a disk write was copying: once
  in five runs it stalled for 290 ms. The barrier now holds other threads'
  device work back before the unmaps.

## Not in this change

Qwen 27B's FP16 planes can share pages (segments that start on a page), and
a port restored 2.5 GB from RAM in 9 to 11 ms against 27 to 30 ms, with the
takeover's decoder gap at 0.5 s against 1.2 s. It is left out: a disk
restore into a session whose pages a checkpoint kept must make 2,600 new
pages while the cache is still growing, which took 0.4 to 0.9 s longer than
main's restore. Making them ahead on the pool's thread closed half of that
and raised decoder gaps by 100 to 400 ms.
