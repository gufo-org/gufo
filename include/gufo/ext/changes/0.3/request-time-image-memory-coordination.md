# Request-time image memory coordination

`ResourceReservation::Shrink()` releases part of a pending reservation after
its covered temporary work has finished, without growing the claim.
`BackingPoolConfig` and `ComponentCacheHooks` append an optional
`reserve_segment` callback. Its reservation precedes page commitment; conversion
and retained ownership follow the registered pages through asynchronous use and
release. Empty callbacks preserve existing provider behavior and lazy backing.

Host-budget helpers expose one sampling pass and pure policy calculation;
`QueryAdmissionMemoryUsingHost()` preserves the platform device-budget policy
while reusing that host observation. Readings are not an atomic kernel snapshot.
No cache grant or eager backing requirement is added. Rebuild source extensions;
no binary ABI compatibility is promised.
