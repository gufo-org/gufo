# Public link targets

Public CMake contract: `gufo_ext_core` links the implementation behind the public
GGUF, sampling, JSON constraint, dequantization, SHA-256, image and, in HIP builds,
weight upload services, and carries OpenSSL for `sha256.hpp`. `gufo_ext_hip`
carries `ENGINE_ENABLE_HIP`, `hip::host` and the hipCUB and rocPRIM include paths;
it exists only in HIP builds. `gufo_ext_cache` is now available in every build, with
its HIP backing in HIP builds. None adds a Gufo include path, and direct links to
`gufo_core` and `gufo_cache` stay rejected. Additive; executables that carry their
own copies of these implementations can drop them.
