# RAM cache relief under host memory pressure

Additive (#932):

- `cache::ComponentCacheResources::release_host` (`src/cache/component_resources.hpp`),
  an optional function declared last: given a byte count, a provider gives up
  to about that much committed backing whose payloads are all gone back to
  the host and returns the bytes released. It never evicts. The serving layer
  calls it only while no work runs, because releasing device-mapped pages
  waits for the device. Released backing returns under the provider's usual
  growth rules. Providers without it keep their backing, as before.
- `ext::HostReserveShortfallBytes(sample)` (`gufo/ext/host_budget.hpp`): how
  far MemAvailable is below max(CmaFree, `kHostSnapshotHeadroomBytes`).

Behavior: when that shortfall is positive, the idle text scheduler has the RAM
cache evict checkpoints and give their memory back until MemAvailable is
2 GiB above the reserve. New requests get 503 `insufficient_memory` while the
shortfall exceeds half the reserve.

Migration: none.
