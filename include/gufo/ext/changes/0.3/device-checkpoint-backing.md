# Device checkpoint backing

Resource accounting adds `MemoryDomain`, independent device limits and usage,
and a domain on allocation information. Existing reservations default to host
memory. Metadata and transfer staging remain host allocations. Persistence
pins overlay either domain without charging the physical allocation twice.
Grouped backing and backing searches require a single memory domain.
Snapshots also expose device free backing, so one pool cannot count free
storage in the other domain for capture admission or background growth.

`Adapter` adds optional device row capture and capability methods. Existing
adapters default to unsupported. `ComponentCaptureBuffer` identifies its memory
domain; `ComponentCacheResources` supplies a device-to-host persistence reader.
The provider retains allocation ownership until every transfer settles.
Borrowed rows identify their backing domain and preserve through the adapter's
matching copy method. Their persistence source accepts a separate device
backing reader, retaining its row pin through completion after preservation.
`Payload::Domain()` identifies whether its opaque owner can be read by the host.
Private state and metadata keep their existing host serialization contract.

`TextRunnerResourceClaim` adds an optional device checkpoint capacity, defaulting
to zero. The experimental HIP admission interface exposes the bounded device
cache budget query. Existing extensions keep host-backed behavior after rebuild.
