# Providers that do not use HIP

Let a provider declare that it drives the GPU through its own runtime:
- Optional provider `serving.hip` in `schemas/extension.schema.json`, a
  boolean. CMake rejects any other type. Omitted means `true`.
- `ProviderInfo::uses_hip` (`include/gufo/ext/registry.hpp`), generated from
  that field.

On Linux a process's GPU address space belongs to the first runtime that opens
it, so a provider with its own runtime and HIP cannot share a process. When the
text provider selected for `gufo serve llm` declares `serving.hip: false`, the
server makes no HIP call for that model.

Existing manifests remain valid. Code that builds `ProviderInfo` with
designated initializers keeps compiling; the new member is last.
