#pragma once

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/text.hpp>
#include <string>
#include <string_view>

namespace gufo::models::inkling2 {

/// Strict, GPU-free: true only if `reader` is an Inkling 2 Mini GGUF in either
/// layout whose hyper-parameters, tensors, shapes and types are all the ones
/// this extension loads. On false, `*reason` (when given) says why.
[[nodiscard]] bool ProbeInkling2Gguf(const core::GgufReader& reader,
                                     std::string* reason);
/// The registry's signature.
[[nodiscard]] bool ProbeInkling2Gguf(const core::GgufReader& reader);
/// Opens `model_root` (a GGUF file) and probes it.
[[nodiscard]] bool ProbeInkling2(std::string_view model_root);
}  // namespace gufo::models::inkling2
