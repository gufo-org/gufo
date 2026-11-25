# Safetensors header reader

Additive: `gufo/ext/safetensors.hpp` forwards `src/core/safetensors.hpp`, a
reader for the header of a safetensors file.

- `safetensors::Parse(file)` checks the header of a whole file held in
  memory, such as a mapping. `safetensors::HeaderLength(prefix, file_size)`
  and `safetensors::ParseHeader(header, file_size)` do the same in two steps
  for a caller that reads the header from a file descriptor.
- `safetensors::Header`: the payload offset, the tensors in header order and
  the `__metadata__` entries. `safetensors::Tensor`: a tensor's name,
  `safetensors::Dtype`, shape and its byte range in the payload.
- `safetensors::Dtype`, `safetensors::Name` and `safetensors::Bits`: the
  element types the format names and their sizes.
- `safetensors::Error` with `safetensors::Reason` and the tensor or
  metadata key that fails; `what()` says what is wrong, such as
  `invalid safetensors file: w: unsupported dtype "bf16"`. `std::bad_alloc`
  when memory runs out and `std::logic_error` on a reader bug.

A header is accepted when its length is 1 byte to 32 MiB, it is a JSON
object, `__metadata__` (if present and not null) maps strings to strings, every other
member names a type of the format, a shape of at most 5 dimensions from 1 to
2^31 - 1 and two offsets, all plain integers, and the tensors tile the
payload exactly. The reader does not know which types a loader supports.

Linking: executables get it through `gufo_ext_runtime`, which now links
`gufo_safetensors`; extension modules resolve it from the Gufo executable.

Compatibility: source compatible after rebuilding.
