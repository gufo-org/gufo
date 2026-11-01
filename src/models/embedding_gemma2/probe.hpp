#pragma once

#include <map>
#include <memory>
#include <string>
#include <string_view>

#include "gufo/ext/embeddings.hpp"

namespace gufo::models::embedding_gemma2 {

/// Pure, HIP-free artifact probe verifying config.json model_type and
/// architecture.
[[nodiscard]] bool ProbeEmbeddingGemma2(std::string_view model_root);

/// Resolves prompt mappings from config_sentence_transformers.json.
[[nodiscard]] bool ResolveEmbeddingGemma2Artifacts(
    std::string_view model_root, std::map<std::string, std::string>* prompts,
    std::string* error = nullptr);

#if defined(ENGINE_ENABLE_HIP)
/// Creates the HIP-backed multimodal embedding runner.
[[nodiscard]] std::unique_ptr<ext::EmbeddingRunner> CreateEmbeddingGemma2Runner(
    const std::string& model_root, std::string* error = nullptr);
#endif

}  // namespace gufo::models::embedding_gemma2
