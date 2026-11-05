#ifndef GUFO_FAMILIES_QWEN_CONFIG_QWEN_CONFIG_HPP_
#define GUFO_FAMILIES_QWEN_CONFIG_QWEN_CONFIG_HPP_

#include <optional>
#include <string>

#include "src/core/model_config.hpp"

namespace gufo::core {
class GgufReader;
}  // namespace gufo::core

namespace gufo::models::qwen {

/// Extracts and validates the Qwen architectural configuration a GGUF file
/// records under its architecture prefix (`qwen35.block_count`, `ssm.*`, ...).
///
/// CPU-only by construction: it reads GGUF metadata through the core reader and
/// reaches no HIP, so the chat template's shape check, weight binding, the MTP
/// reference and the `prompt` command resolve shapes in a HIP-less build.
/// `error_msg`, when supplied, receives the rejection reason. Returns no value
/// when the file is not a text-only Qwen configuration.
[[nodiscard]] std::optional<core::ModelConfig> ExtractQwenConfig(
    const core::GgufReader& reader, std::string* error_msg = nullptr);

}  // namespace gufo::models::qwen

#endif  // GUFO_FAMILIES_QWEN_CONFIG_QWEN_CONFIG_HPP_
