# Extension provenance and excluded metadata

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
