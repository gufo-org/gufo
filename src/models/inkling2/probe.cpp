#include "src/models/inkling2/probe.hpp"

#include <filesystem>

#include "src/models/inkling2/weights.hpp"

namespace gufo::models::inkling2 {

bool ProbeInkling2Gguf(const core::GgufReader& reader, std::string* reason) {
  // Binding is the probe: a file this accepts is a file the loader accepts.
  // It only indexes the mapping, so it stays cheap and GPU-free.
  return Inkling2Weights::Bind(reader, reason).has_value();
}

bool ProbeInkling2Gguf(const core::GgufReader& reader) {
  return ProbeInkling2Gguf(reader, nullptr);
}

bool ProbeInkling2(std::string_view model_root) {
  try {
    const std::filesystem::path path(model_root);
    if (!std::filesystem::is_regular_file(path)) {
      return false;
    }
    std::string error;
    // NVFP4 is unsized to core, so the file is opened with that allowed.
    const auto reader = core::GgufReader::OpenFile(
        path, &error, {.allow_unsized_tensors = true});
    return reader && ProbeInkling2Gguf(*reader);
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::inkling2
