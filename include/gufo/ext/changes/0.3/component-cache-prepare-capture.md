# Component cache prepare capture

Additive: `cache::ComponentCacheResources` (`src/cache/component_resources.hpp`)
gains an optional `prepare_capture(bytes)` hook. The component cache calls it
when a disk warm-up starts and before a disk restore, with the payload bytes of
that checkpoint plus those of restores still reading or not captured yet,
because the capture after each restore shares no RAM chunks and needs backing
for its whole checkpoint. The HIP provider grows its committed
backing toward that many free bytes on its own thread, so the capture no longer
commits backing segments on the serving thread.

Migration: none. Providers that leave it empty keep today's behavior.
