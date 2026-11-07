# RDNA 3.5 architecture predicates

Publish the RDNA 3.5 device-architecture contract as an experimental forwarding
header (`include/gufo/ext/experimental/arch.hpp`) over `src/core/hip/arch.hpp`:
- `IsRdna35`, `IsStrixPoint`, `IsStrixHalo`, `Rdna35MarketingName`,
  `Rdna35MemoryType`, `IsRdna35PciId`, `Rdna35PciArch`, `CurrentDeviceArch`,
  `DeviceDescription`, `kGfx1150`, `kGfx1151`.

Model implementations need the architecture string to guard kernels, name
target errors and key disk-cache identities, so the header joins the existing
`experimental/` forwarding layer rather than reaching into `src/core/**`.
Additive; source compatible. The `SURFACE` closure gains
`include/gufo/ext/experimental/arch.hpp` and `src/core/hip/arch.hpp`.
