#include "src/cli/serve/embedding_service.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "gufo/ext/embeddings.hpp"
#include "gufo/ext/registry.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

struct EmbeddingService::Impl {
  explicit Impl(EmbeddingServiceOptions supplied)
      : options(std::move(supplied)) {
    if (options.model_id.empty()) {
      options.model_id = "embeddinggemma-2";
    }

    if (options.runner) {
      // An injected runner needs no checkpoint, so the model is not validated
      // unless the caller asks for it.
      if (options.validate_model && !options.model_root.empty()) {
        const ext::EmbeddingProvider* provider =
            ext::ProbeEmbeddingProvider(options.model_root.string());
        if (!provider && !options.model_id.empty()) {
          provider = ext::FindEmbeddingProvider(options.model_id);
        }
        if (!provider || !provider->probe ||
            !provider->probe(options.model_root.string())) {
          initialization_error = kInvalidModel;
          return;
        }
      }
      ready = true;
      return;
    }

    const ext::EmbeddingProvider* provider = nullptr;
    if (!options.model_root.empty()) {
      provider = ext::ProbeEmbeddingProvider(options.model_root.string());
    }
    if (!provider && !options.model_id.empty()) {
      provider = ext::FindEmbeddingProvider(options.model_id);
    }
#if defined(ENGINE_ENABLE_HIP)
    if (!provider || !provider->create) {
      initialization_error = "embedding provider not available for model: " +
                             options.model_root.string();
      return;
    }
    if (options.validate_model && provider->probe &&
        !provider->probe(options.model_root.string())) {
      initialization_error = kInvalidModel;
      return;
    }
    std::string error;
    runner_ = provider->create(options.model_root.string(), &error);
    if (!runner_) {
      initialization_error = "embedding encoder failed to load: " + error;
      return;
    }
    ready = true;
#else
    initialization_error = "embedding service requires the HIP backend";
#endif
  }

  EmbeddingStatus Run(const std::vector<EmbeddingInput>& inputs,
                      const EmbeddingRequestOptions& request,
                      const std::function<bool()>& is_cancelled,
                      EmbeddingBatch* batch, std::string* error) {
    if (is_cancelled && is_cancelled()) {
      if (error) {
        *error = "embedding request cancelled";
      }
      return EmbeddingStatus::kCancelled;
    }
    if (options.runner) {
      std::lock_guard<std::mutex> lock(runner_mutex);
      return options.runner(inputs, request, batch, error);
    }
    if (runner_) {
      EmbeddingStatus status =
          runner_->EmbedInputs(inputs, is_cancelled, batch, error, request);
      if (runner_->device_lost()) {
        device_lost.store(true);
      }
      return status;
    }
    if (error) {
      *error = initialization_error;
    }
    return EmbeddingStatus::kFailed;
  }

  static constexpr const char* kInvalidModel =
      "embedding service requires a valid model directory";

  EmbeddingServiceOptions options;
  std::unique_ptr<ext::EmbeddingRunner> runner_;
  std::string initialization_error;
  // Serializes the injected runner, which has no device turns of its own.
  std::mutex runner_mutex;
  std::atomic<bool> device_lost{false};
  bool ready{false};
};

bool TruncateKeepingImages(std::vector<std::uint32_t>* tokens,
                           std::size_t max_tokens, std::uint32_t image_token) {
  if (tokens->size() <= max_tokens) {
    return true;
  }
  if (max_tokens < 2) {
    return false;
  }
  // Keep the first tokens and the final <eos>. The last image placeholder and
  // the end-of-image marker after it must both stay.
  const std::size_t keep = max_tokens - 1;
  const auto last_image =
      std::ranges::find(tokens->rbegin(), tokens->rend(), image_token);
  const std::size_t image_end =
      last_image == tokens->rend()
          ? 0
          : static_cast<std::size_t>(tokens->rend() - last_image) + 1;
  if (image_end > keep) {
    return false;
  }
  tokens->erase(tokens->begin() + static_cast<std::ptrdiff_t>(keep),
                tokens->end() - 1);
  return true;
}

bool LoadSentenceTransformerPrompts(const std::filesystem::path& model_root,
                                    std::map<std::string, std::string>* prompts,
                                    std::string* error) {
  prompts->clear();
  const auto path = model_root / "config_sentence_transformers.json";
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return true;
  }
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  try {
    const json::Value root = json::parse(text);
    const json::Value* table =
        root.is_object() ? root.find("prompts") : nullptr;
    if (table == nullptr || table->is_null()) {
      return true;
    }
    if (!table->is_object()) {
      *error = path.string() + ": 'prompts' must be an object";
      return false;
    }
    for (const auto& [name, value] : table->members()) {
      if (!value.is_string()) {
        *error = path.string() + ": prompt '" + name + "' must be a string";
        return false;
      }
      prompts->emplace(name, value.str());
    }
  } catch (const std::exception& parse_error) {
    *error = path.string() + ": " + parse_error.what();
    return false;
  }
  return true;
}

EmbeddingService::EmbeddingService(EmbeddingServiceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {
  const auto& name = impl_->options.default_prompt_name;
  if (impl_->ready && !name.empty() && !impl_->options.prompts.contains(name)) {
    impl_->ready = false;
    impl_->initialization_error =
        "--default-prompt-name '" + name + "' is not a prompt of this model";
  }
}

const std::string* EmbeddingService::prompt(const std::string& name) const {
  const auto found = impl_->options.prompts.find(name);
  return found == impl_->options.prompts.end() ? nullptr : &found->second;
}

const std::string& EmbeddingService::default_prompt_name() const {
  return impl_->options.default_prompt_name;
}

EmbeddingService::~EmbeddingService() = default;

bool EmbeddingService::ready() const noexcept {
  return impl_->ready;
}

bool EmbeddingService::device_lost() const noexcept {
  return impl_->device_lost.load();
}

std::string EmbeddingService::initialization_error() const {
  return impl_->initialization_error;
}

std::string EmbeddingService::model_id() const {
  return impl_->options.model_id;
}

std::string EmbeddingService::backend_name() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->options.runner ? "custom" : "hip";
#else
  return "custom";
#endif
}

EmbeddingStatus EmbeddingService::Embed(
    const std::vector<std::string>& texts,
    const std::function<bool()>& is_cancelled, EmbeddingBatch* batch,
    std::string* error) {
  std::vector<EmbeddingInput> inputs;
  inputs.reserve(texts.size());
  for (const auto& text : texts) {
    inputs.push_back({{{EmbeddingPart::Type::kText, text}}});
  }
  return EmbedInputs(inputs, is_cancelled, batch, error);
}

EmbeddingStatus EmbeddingService::EmbedInputs(
    const std::vector<EmbeddingInput>& inputs,
    const std::function<bool()>& is_cancelled, EmbeddingBatch* batch,
    std::string* error, const EmbeddingRequestOptions& request) {
  if (!ready()) {
    if (error) {
      *error = initialization_error();
    }
    return device_lost() ? EmbeddingStatus::kDeviceLost
                         : EmbeddingStatus::kFailed;
  }
  if (device_lost()) {
    if (error) {
      *error = "embedding device lost";
    }
    return EmbeddingStatus::kDeviceLost;
  }
  if (inputs.empty()) {
    batch->embeddings.clear();
    batch->token_counts.clear();
    return EmbeddingStatus::kOk;
  }
  EmbeddingStatus status =
      impl_->Run(inputs, request, is_cancelled, batch, error);
  if (status == EmbeddingStatus::kDeviceLost) {
    impl_->device_lost.store(true);
  }
  return status;
}

}  // namespace gufo::server
