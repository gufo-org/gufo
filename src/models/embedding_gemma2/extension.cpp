#include <exception>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <gufo/ext/registry.hpp>
#include <iterator>
#include <string>
#include <string_view>

#include "config.hpp"
#include "probe.hpp"

namespace gufo::models::embedding_gemma2 {

bool ProbeEmbeddingGemma2(std::string_view model_root) {
  const auto config = LoadModelConfigFromPath(std::string(model_root));
  if (!config.has_value()) {
    return false;
  }
  if (config->model_type != "embedding_gemma2" ||
      config->architecture != "EmbeddingGemma2Model") {
    return false;
  }
  return IsSupportedModelConfig(*config);
}

bool ResolveEmbeddingGemma2Artifacts(
    std::string_view model_root, std::map<std::string, std::string>* prompts,
    std::string* error) {
  if (prompts) {
    prompts->clear();
  }
  const auto path =
      std::filesystem::path(model_root) / "config_sentence_transformers.json";
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return true;
  }
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  try {
    const gufo::json::Value root = gufo::json::parse(text);
    const gufo::json::Value* table =
        root.is_object() ? root.find("prompts") : nullptr;
    if (table == nullptr || table->is_null()) {
      return true;
    }
    if (!table->is_object()) {
      if (error) {
        *error = path.string() + ": 'prompts' must be an object";
      }
      return false;
    }
    if (prompts) {
      for (const auto& [name, value] : table->members()) {
        if (!value.is_string()) {
          if (error) {
            *error = path.string() + ": prompt '" + name + "' must be a string";
          }
          return false;
        }
        prompts->emplace(name, value.str());
      }
    }
  } catch (const std::exception& parse_error) {
    if (error) {
      *error = path.string() + ": " + parse_error.what();
    }
    return false;
  }
  return true;
}

}  // namespace gufo::models::embedding_gemma2

void embedding_gemma2_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::embedding_gemma2;
  registry.AddEmbeddingProvider({
      .id = "embeddinggemma-2",
      .probe = &ProbeEmbeddingGemma2,
      .resolve = &ResolveEmbeddingGemma2Artifacts,
      .create = nullptr,
  });
}
