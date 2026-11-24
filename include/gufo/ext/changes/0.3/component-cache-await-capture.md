# Component cache await capture

Additive: `cache::ComponentCacheResources` (`src/cache/component_resources.hpp`)
gains an optional `await_capture()` hook. It returns once the backing asked
through `prepare_capture` is free, or will not be. A disk restore whose copies
run beside decoders (`cache::RestoreCopyRunner`, inside a
`cache::BesideDecodeScope`) calls it on the thread that ran the copies, after
them: such a restore can finish before its backing is committed, and the
capture that follows on the serving thread then waited for the rest while no
decoder stepped. Any thread may call it while others capture.

The HIP provider waits on the new
`hip::CommittedBackingPool::WaitWhileGrowingAhead()`
(`src/core/hip/committed_backing.hpp`), which returns when no thread grows
ahead and, unlike `WaitForGrowth`, holds no lock meanwhile.

Migration: none. Providers that leave it empty keep today's behavior.
