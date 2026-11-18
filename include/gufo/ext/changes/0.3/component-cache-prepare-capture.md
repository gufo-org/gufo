# Component cache prepare capture

Additive: `cache::ComponentCacheResources` (`src/cache/component_resources.hpp`)
gains an optional `prepare_capture(bytes)` hook. The component cache calls it
with a disk checkpoint's payload bytes when a disk warm-up starts and before a
disk restore, because the capture after that restore shares no RAM chunks and
needs backing for the whole checkpoint. The HIP provider grows its committed
backing toward that many free bytes on its own thread, so the capture no longer
commits backing segments on the serving thread.

Migration: none. Providers that leave it empty keep today's behavior.
