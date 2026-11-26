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
  22 element types of safetensors 0.8 and their sizes. New types take the
  next values.
- `safetensors::Error` with `safetensors::Reason` and the tensor or
  metadata key that fails; `what()` says what is wrong, such as
  `invalid safetensors header: w: unsupported dtype "bf16"`. `std::bad_alloc`
  when memory runs out and `std::logic_error` on a reader bug.

The reader accepts the safetensors 0.8 format within Gufo's limits, the
strictest of the loaders it replaced. A header is accepted when its length
is 1 byte to 32 MiB, it is a JSON object, `__metadata__` (if present and not
null) maps strings to strings, every other member names a type of the
format, a shape of at most 5 dimensions from 0 to 2^31 - 1 and two offsets,
all plain integers, and the tensors with elements tile the payload exactly.
So unlike the format, it rejects headers above 32 MiB, headers without
tensors, more than 5 dimensions and dimensions above 2^31 - 1. A tensor
without elements has equal offsets between the other tensors or at either
end of the payload, and its dimensions multiplied in order fit 64 bits up to
the first 0, as in safetensors 0.8. The reader does not know which types a
loader supports or whether it can use a tensor without elements.

Linking: executables get it through `gufo_ext_runtime`, which now links
`gufo_safetensors`; extension modules resolve it from the Gufo executable.

Compatibility: source compatible after rebuilding.
