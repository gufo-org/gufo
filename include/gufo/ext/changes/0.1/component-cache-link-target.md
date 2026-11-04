# Component cache link target

Public CMake contract: `gufo_ext_cache` is the link target for an extension that
implements a continuation adapter. Linking it provides the cache and its HIP
backing, including `hip::CreateHipComponentCacheResources`. It adds no include
paths; the headers remain those of `gufo_ext_runtime`. The target exists only in
HIP builds, so list it under a target that requires `hip`. Additive; extensions
that do not use the component cache are unaffected.
