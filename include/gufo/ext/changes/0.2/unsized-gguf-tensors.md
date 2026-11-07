# GGUF tensors core cannot size

GGUF reader (`src/core/gguf_reader.hpp`) and text (`src/ext/text.hpp`): a GGUF
whose tensors use a storage type or shape core cannot size, such as an
extension's own quantization, reaches a text provider that declares support
instead of failing to open.
- `core::GgufOpenOptions::allow_unsized_tensors` (`GgufReader::OpenFile` and
  `OpenMemory`): such tensors open with `GgufTensorInfo::sized == false`, and
  their `size_bytes` spans the bytes the file reserves for them, up to the next
  tensor's offset or the end of the data. Without it, the default, the reader
  rejects the file as before.
- `GgufReader::FirstUnsizedTensor()` and `UnsizedTensorError()`: the first such
  tensor, and the error a default reader reports for the file.
- `TextProvider::accepts_unsized_tensors`: only providers that set it probe a
  file with unsized tensors. `serve`, `prompt`, `chat` and `bench` open text
  models with `allow_unsized_tensors`; when no provider accepts the file they
  report the reader's previous error.

Compatibility: additive. The field defaults to false, so existing providers
never see unsized tensors, and every other reader keeps rejecting them.
