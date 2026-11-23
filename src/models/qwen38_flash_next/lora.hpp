#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_LORA_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_LORA_HPP_

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/families/qwen/lora/lora.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {

using qwen::lora::FloatToHalf;
using qwen::lora::kLoraMergeAlgorithm;
using qwen::lora::LoraAdapter;
using qwen::lora::LoraPatch;
using qwen::lora::LoraSelection;
using qwen::lora::QuantizeRowQ8_0;
using qwen::lora::ResolveLoraOptions;

/// Trunk tensors an adapter may patch: the GDN and attention output
/// projections and the shared-expert down projection of every trunk layer.
/// The MTP draft block is not part of the trunk and is never patched.
[[nodiscard]] std::vector<TensorRef> LoraTargets(const ModelWeights& weights);

/// Opens and validates a llama.cpp GGUF LoRA adapter against `targets`
/// (from `LoraTargets`). `architecture` is the trunk's
/// `general.architecture`.
[[nodiscard]] std::unique_ptr<LoraAdapter> OpenLoraAdapter(
    const std::filesystem::path& path, float scale,
    std::string_view architecture, std::span<const TensorRef> targets,
    std::string* error);
/// Validates an already opened adapter; `label` names it in errors.
[[nodiscard]] std::unique_ptr<LoraAdapter> OpenLoraAdapter(
    std::unique_ptr<core::GgufReader> reader, std::string label, float scale,
    std::string_view architecture, std::span<const TensorRef> targets,
    std::string* error);

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_LORA_HPP_
