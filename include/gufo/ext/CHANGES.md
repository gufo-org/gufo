# Extension API changes

## 0.1.1

Add excluded extension metadata and artifact-key diagnostics, and generated
selection-mode provenance. Existing registration and provider contracts remain
source compatible. Discovery and exact lock selection are pure CMake; families
also appear in built-extension provenance. Incompatible unselected metadata
can describe a future API series without forcing lock builds to adopt it; only
selected extensions enforce the current major/minor/revision floor. The metadata
schema therefore accepts an `api_version` floor of any major, and it now also
requires the key-kind prefix (`gguf:`, `safetensors:`, `manifest:`, `diffusers:`)
that configure already enforces, so schema validation cannot pass a manifest
that cannot build. Cross-tier key/option/model-ID diagnostics name the first
recorded holder of the colliding value.

## 0.1.0

Initial development API: metadata/registration, core capability kinds, typed
provider options and a borrowed logging Host handle. Compatibility requires
`api_version: ">= 0.1.0"`; the minor series must match and the requested
revision must be no newer than the runtime. An additive revision keeps older
extensions compatible within this series. No stable model/session ABI is
promised by this initial source API.
