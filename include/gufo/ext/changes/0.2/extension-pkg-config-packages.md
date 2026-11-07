# Extension pkg-config packages

Extension CMake contract (`cmake/GufoExtensionDiscovery.cmake`, manifest
`packages`): each selected extension's `packages[].pkg_config` modules are
required at configure time, before extensions build, as global imported
targets `PkgConfig::<module>` with the `<module>_*` variables
`pkg_check_modules` sets. Modules of unselected extensions are not required.

The root build no longer requires sndfile, soxr, libavformat, libavcodec,
libavutil and libswscale for every build, and no longer defines
`PkgConfig::SNDFILE`, `PkgConfig::SOXR` or the `AV_*` variables; only
EmbeddingGemma 2 used them, and it now links `PkgConfig::sndfile` and
`PkgConfig::soxr`. An extension that relied on the old names declares the
modules in its manifest and uses the new names. MiniMax H3 drops its
`pkg_config` entries, since it runs the FFmpeg executable and links no FFmpeg
library.
