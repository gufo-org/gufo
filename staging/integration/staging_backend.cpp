#include "staging/integration/staging_backend.hpp"

#include <exception>
#include <limits>
#include <memory>
#include <utility>

#include "staging/models/flint/engine.hpp"
#include "staging/models/flint/serve_runner.hpp"
#include "staging/models/gemma4/engine.hpp"
#include "staging/models/gemma4/serve_runner.hpp"
#include "staging/models/glm5_next/engine.hpp"
#include "staging/models/glm5_next/serve_runner.hpp"
#include "staging/models/ling3_0/engine.hpp"
#include "staging/models/ling3_0/serve_runner.hpp"
#include "staging/models/nemotron_h_moe/engine.hpp"
#include "staging/models/nemotron_h_moe/serve_runner.hpp"
#include "staging/models/qwen36_a3b/engine.hpp"
#include "staging/models/qwen36_a3b/serve_runner.hpp"

namespace gufo::server {

namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

std::optional<StagingLoadedModel> StagingLoadModel(
    const std::string& model_path, const core::GgufReader& reader,
    std::uint32_t max_context, std::size_t session_count,
    const TextSpeculativeConfig& speculative_config,
    const TextDiskCacheConfig& disk_cache_config,
    const std::string& vision_model_path, std::uint32_t attn_window,
    std::uint32_t attn_sink, std::string* error) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture != "qwen35moe" && architecture != "gemma4" &&
      architecture != "bailingmoe3" && architecture != "nemotron_h_moe" &&
      architecture != "glm5-next" && architecture != "flint") {
    return std::nullopt;
  }
  if (max_context == 0) {
    const auto native =
        reader.GetMetadataUint64(std::string(architecture) + ".context_length")
            .value_or(0);
    if (native < 2 || native > std::numeric_limits<std::uint32_t>::max()) {
      SetError(error,
               "GGUF has no valid native context length; specify --context");
      return std::nullopt;
    }
    max_context = static_cast<std::uint32_t>(native);
  }
  std::string load_error;
  try {
    if (architecture == "qwen35moe") {
      auto model = models::qwen36_a3b::LoadServeModel(
          model_path, reader, max_context, attn_window, attn_sink,
          vision_model_path, speculative_config, &load_error);
      if (model == nullptr) {
        SetError(error, load_error);
        return std::nullopt;
      }
      auto runner = models::qwen36_a3b::CreateTextRunner(
          std::move(model), max_context, session_count, speculative_config,
          disk_cache_config, error);
      if (runner == nullptr) {
        return std::nullopt;
      }
      return StagingLoadedModel{.runner = std::move(runner)};
    }
    if (architecture == "gemma4") {
      auto model = models::gemma4::LoadServeModel(
          model_path, reader, max_context, vision_model_path,
          speculative_config, &load_error);
      if (model == nullptr) {
        SetError(error, load_error);
        return std::nullopt;
      }
      // Read the vision flag before the model is moved into the runner; a
      // loaded `gemma4v` mmproj tower is what lets the HTTP layer accept
      // image inputs.
      const bool has_vision = model->HasVision();
      auto runner = models::gemma4::CreateTextRunner(
          std::move(model), max_context, session_count, speculative_config,
          disk_cache_config, error);
      if (runner == nullptr) {
        return std::nullopt;
      }
      return StagingLoadedModel{
          .runner = std::move(runner), .supports_images = has_vision};
    }
    if (architecture == "bailingmoe3") {
      auto model = models::ling3_0::LoadServeModel(
          model_path, reader, max_context, speculative_config, &load_error);
      if (model == nullptr) {
        SetError(error, load_error);
        return std::nullopt;
      }
      auto runner = models::ling3_0::CreateTextRunner(
          std::move(model), max_context, session_count, speculative_config,
          disk_cache_config, error);
      if (runner == nullptr) {
        return std::nullopt;
      }
      return StagingLoadedModel{.runner = std::move(runner)};
    }
    if (architecture == "nemotron_h_moe") {
      auto model = models::nemotron_h_moe::LoadServeModel(
          model_path, reader, max_context, speculative_config, &load_error);
      if (model == nullptr) {
        SetError(error, load_error);
        return std::nullopt;
      }
      auto runner = models::nemotron_h_moe::CreateTextRunner(
          std::move(model), max_context, session_count, speculative_config,
          disk_cache_config, error);
      if (runner == nullptr) {
        return std::nullopt;
      }
      return StagingLoadedModel{.runner = std::move(runner)};
    }
    if (architecture == "glm5-next") {
      auto model = models::glm5_next::LoadServeModel(
          model_path, reader, max_context, vision_model_path,
          speculative_config, &load_error);
      if (model == nullptr) {
        SetError(error, load_error);
        return std::nullopt;
      }
      // Read the vision flag before the model is moved into the runner; a
      // loaded `glm5v` mmproj tower is what lets the HTTP layer accept
      // image inputs.
      const bool has_vision = model->HasVision();
      auto runner = models::glm5_next::CreateTextRunner(
          std::move(model), max_context, session_count, speculative_config,
          disk_cache_config, error);
      if (runner == nullptr) {
        return std::nullopt;
      }
      return StagingLoadedModel{
          .runner = std::move(runner), .supports_images = has_vision};
    }
    auto model = models::flint::LoadServeModel(model_path, reader, max_context,
                                               speculative_config, &load_error);
    if (model == nullptr) {
      SetError(error, load_error);
      return std::nullopt;
    }
    auto runner = models::flint::CreateTextRunner(
        std::move(model), max_context, session_count, speculative_config,
        disk_cache_config, error);
    if (runner == nullptr) {
      return std::nullopt;
    }
    return StagingLoadedModel{.runner = std::move(runner)};
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return std::nullopt;
  }
}

}  // namespace gufo::server
