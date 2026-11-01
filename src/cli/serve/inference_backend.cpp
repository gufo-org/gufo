#include "src/cli/serve/inference_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/text.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_reader.hpp"

namespace gufo::server {

namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool DiskCacheEnabled(const TextDiskCacheConfig& config) {
  return !config.directory.empty() && config.capacity_bytes > 0;
}

}  // namespace

struct InferenceBackend::Impl {
#if defined(ENGINE_ENABLE_HIP)
  struct State {
    std::shared_ptr<TextGenerationScheduler> scheduler;
    std::string model_id;
    std::vector<std::string> extra_model_ids;
    SamplingDefaults sampling_defaults;
    std::uint32_t max_context{0};
    bool supports_images{false};
    ReasoningOptions reasoning_defaults;
    std::string model_path;
    std::string build_info;
  };

  class ScheduledGenerationRequest final : public GenerationRequest {
  public:
    ScheduledGenerationRequest(
        std::shared_ptr<const State> model_state,
        TextGenerationScheduler::Request scheduled_request,
        InitialOutputState initial = InitialOutputState::kContent,
        std::optional<sampling::JsonConstraint::ToolFormat> tool_format = {})
        : state_(std::move(model_state)),
          request_(std::move(scheduled_request)),
          tool_format_(tool_format) {
      if (initial == InitialOutputState::kReasoning)
        reasoning_end_ =
            state_->scheduler->runner().TokenizeControl("</think>");
    }

    Result Wait(const TokenCallback& on_token,
                const ProgressCallback& on_progress,
                const StartCallback& on_start) override {
      auto result = request_.Wait(on_token, on_progress, on_start);
      if (!reasoning_end_.empty()) {
        const auto end =
            std::search(result.tokens.begin(), result.tokens.end(),
                        reasoning_end_.begin(), reasoning_end_.end());
        result.reasoning_tokens =
            static_cast<std::size_t>(end - result.tokens.begin());
      }
      return result;
    }

    void Cancel() noexcept override { request_.Cancel(); }

    std::optional<sampling::JsonConstraint::ToolFormat> ToolFormat()
        const override {
      return tool_format_;
    }

  private:
    std::shared_ptr<const State> state_;
    TextGenerationScheduler::Request request_;
    std::vector<tokenization::TokenId> reasoning_end_;
    const std::optional<sampling::JsonConstraint::ToolFormat> tool_format_;
  };

  [[nodiscard]] std::shared_ptr<const State> Snapshot() const {
    const std::lock_guard<std::mutex> lock(state_mutex);
    return state;
  }

  Result GenerateScheduled(
      std::shared_ptr<const State> current,
      std::vector<TextRunnerToken> prompt_tokens,
      Clock::time_point request_start, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled, const TokenCallback& on_token,
      std::string client_id,
      std::shared_ptr<const TextPromptContext> context = {},
      bool cache_prompt = true, std::size_t cache_prefix_tokens = 0,
      const std::vector<std::string>& stop_sequences = {},
      InitialOutputState initial = InitialOutputState::kContent,
      bool cache_persist = true) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    result.client_id = client_id.empty() ? "anonymous" : client_id;
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      return result;
    }

    auto request = current->scheduler->Submit(
        std::move(prompt_tokens), max_tokens, sampling, is_cancelled,
        static_cast<bool>(on_token),
        TextRequestMetadata{
            .client_id = std::move(client_id),
            .deadline = std::nullopt,
            .request_start = request_start,
            .prompt_context = std::move(context),
            .cache_prompt = cache_prompt,
            .cache_persist = cache_persist,
            .cache_prefix_tokens = cache_prefix_tokens,
            .stop_sequences = stop_sequences,
        });
    ScheduledGenerationRequest generation(std::move(current),
                                          std::move(request), initial);
    return generation.Wait(on_token, {}, {});
  }

  mutable std::mutex state_mutex;
  std::shared_ptr<const State> state;
#endif
};

InferenceBackend::InferenceBackend() : impl_(std::make_unique<Impl>()) {}

InferenceBackend::~InferenceBackend() = default;

#if defined(ENGINE_ENABLE_HIP)
bool InferenceBackend::load(std::shared_ptr<ext::TextModelRunner> runner,
                            std::string* error, std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            const TextDiskCacheConfig& disk_cache_config,
                            TextRunnerRamCacheOptions ram_cache_config) {
  if (runner == nullptr) {
    if (error && error->empty()) {
      SetError(error, "Runner must not be null");
    }
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  try {
    const auto desc = runner->Descriptor();
    auto new_state = std::make_unique<Impl::State>();
    new_state->model_path = desc.model_id;
    new_state->sampling_defaults.max_tokens = 0;
    new_state->sampling_defaults.model = desc.sampling_preset;
    new_state->sampling_defaults.sampling = sampling::SamplingConfig{};
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = desc.capabilities.supports_vision;
    new_state->model_id = desc.model_id;
    new_state->max_context = desc.max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = disk_cache_config.directory,
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    Logger::Info("loader",
                 "event=load_phase phase=sessions " + Logger::MemoryStatus());
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache),
        ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(const std::string& model_path, std::string* error,
                            std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            const TextSpeculativeConfig& speculative_config,
                            const TextDiskCacheConfig& disk_cache_config,
                            const std::string& vision_model_path,
                            TextRunnerRamCacheOptions ram_cache_config,
                            bool enable_yarn) {
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    SetError(error, "Failed to open GGUF: " + load_error);
    return false;
  }
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  const auto* provider = ext::ProbeTextProvider(*reader);
  if (provider == nullptr || provider->create == nullptr) {
    SetError(error, "No text provider found for model architecture");
    return false;
  }
  ext::TextRunnerOptions options{
      .model_path = model_path,
      .reader = reader,
      .max_context = max_context,
      .session_count = session_count,
      .speculative = speculative_config,
      .vision_model_path = vision_model_path,
      .disk_cache = disk_cache_config,
      .ram_cache = ram_cache_config,
      .enable_yarn = enable_yarn,
  };
  auto runner = provider->create(options, error);
  if (runner == nullptr) {
    return false;
  }
  return load(std::move(runner), error, session_count, prefill_policy,
              scheduler_policy, disk_cache_config, ram_cache_config);
}
#else
bool InferenceBackend::load(std::shared_ptr<ext::TextModelRunner>,
                            std::string* error, std::size_t, TextPrefillPolicy,
                            TextSchedulerPolicy, const TextDiskCacheConfig&,
                            TextRunnerRamCacheOptions) {
  SetError(error, "HIP engine is disabled");
  return false;
}

bool InferenceBackend::load(const std::string&, std::string* error,
                            std::uint32_t, std::size_t, TextPrefillPolicy,
                            TextSchedulerPolicy, const TextSpeculativeConfig&,
                            const TextDiskCacheConfig&, const std::string&,
                            TextRunnerRamCacheOptions, bool) {
  SetError(error, "HIP engine is disabled");
  return false;
}
#endif

std::string InferenceBackend::model_id() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_id : "unknown";
#else
  return "unknown";
#endif
}

std::vector<std::string> InferenceBackend::model_ids() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr)
    return {"unknown"};
  std::vector<std::string> ids{state->model_id};
  ids.insert(ids.end(), state->extra_model_ids.begin(),
             state->extra_model_ids.end());
  return ids;
#else
  return {"unknown"};
#endif
}

bool InferenceBackend::ready() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->Snapshot() != nullptr;
#else
  return false;
#endif
}

bool InferenceBackend::device_lost() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->scheduler->device_lost();
#else
  return false;
#endif
}

bool InferenceBackend::supports_images() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->supports_images;
#else
  return false;
#endif
}

std::uint32_t InferenceBackend::max_context() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->max_context : 0;
#else
  return 0;
#endif
}

std::vector<InferenceBackend::SessionState> InferenceBackend::session_states()
    const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->scheduler->SessionStates()
                          : std::vector<SessionState>{};
#else
  return {};
#endif
}

InferenceBackend::SamplingDefaults InferenceBackend::sampling_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->sampling_defaults : SamplingDefaults{};
#else
  return {};
#endif
}

ReasoningOptions InferenceBackend::reasoning_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->reasoning_defaults : ReasoningOptions{};
#else
  return {};
#endif
}

InferenceBackend::InitialOutputState InferenceBackend::initial_output_state(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr
             ? state->scheduler->runner().InitialOutputState(request)
             : InitialOutputState::kAuto;
#else
  (void)request;
  return InitialOutputState::kAuto;
#endif
}

std::string InferenceBackend::model_path() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_path : std::string{};
#else
  return {};
#endif
}

std::string InferenceBackend::build_info() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->build_info : std::string{};
#else
  return {};
#endif
}

void InferenceBackend::set_served_artifact(const std::string& model_path,
                                           const std::string& build_info) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_path = model_path;
  updated->build_info = build_info;
  impl_->state = std::move(updated);
#else
  (void)model_path;
  (void)build_info;
#endif
}

void InferenceBackend::set_model_ids(std::vector<std::string> model_ids) {
#if defined(ENGINE_ENABLE_HIP)
  if (model_ids.empty()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_id = std::move(model_ids.front());
  updated->extra_model_ids.assign(
      std::make_move_iterator(model_ids.begin() + 1),
      std::make_move_iterator(model_ids.end()));
  impl_->state = std::move(updated);
#else
  (void)model_ids;
#endif
}

void InferenceBackend::set_sampling_defaults(
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    sampling::SamplingOverrides supplied) {
#if defined(ENGINE_ENABLE_HIP)
  sampling_config.Validate();
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->sampling_defaults.max_tokens = max_tokens;
  updated->sampling_defaults.sampling = sampling_config;
  updated->sampling_defaults.supplied = supplied;
  impl_->state = std::move(updated);
#else
  (void)max_tokens;
  (void)sampling_config;
  (void)supplied;
#endif
}

void InferenceBackend::set_reasoning_defaults(
    const ReasoningOptions& reasoning) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->reasoning_defaults = reasoning;
  impl_->state = std::move(updated);
#else
  (void)reasoning;
#endif
}

InferenceBackend::Result InferenceBackend::complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool cache_prompt, bool cache_persist) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
  return impl_->GenerateScheduled(
      state, std::move(prompt_tokens), request_start, max_tokens,
      sampling_config, is_cancelled, on_token, std::string(client_id), {},
      cache_prompt, 0, stop_sequences, InitialOutputState::kContent,
      cache_persist);
#else
  (void)cache_prompt;
  (void)cache_persist;
  (void)client_id;
  (void)stop_sequences;
  (void)prompt;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const ChatRequest& request, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto effective_sampling = sampling_config;
  auto constrained = ConstrainChatRequest(request, state->scheduler->runner(),
                                          &effective_sampling);
  const auto& effective_request = constrained ? *constrained : request;
  auto prompt = state->scheduler->runner().PreparePrompt(effective_request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return {};
  }
  const auto initial =
      state->scheduler->runner().InitialOutputState(effective_request);
  if (request.reasoning.budget_tokens &&
      initial == TextGenerationBackend::InitialOutputState::kReasoning)
    effective_sampling.reasoning_budget =
        state->scheduler->runner().BuildReasoningBudget(
            *request.reasoning.budget_tokens, prompt->tokens);
  return impl_->GenerateScheduled(
      state, std::move(prompt->tokens), request_start, max_tokens,
      effective_sampling, is_cancelled, on_token, request.client_id,
      std::move(prompt->context), request.cache_prompt,
      prompt->cache_prefix_tokens, request.stop_sequences, initial,
      request.cache_persist);
#else
  (void)request;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, bool stream_output, bool ignore_eos,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool return_progress, bool cache_prompt, bool cache_persist) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_complete(
        prompt, max_tokens, sampling_config, is_cancelled, stream_output,
        ignore_eos, client_id, stop_sequences, return_progress, cache_prompt,
        cache_persist);
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt_tokens), max_tokens, sampling_config, is_cancelled,
      stream_output,
      TextRequestMetadata{
          .client_id = client_id.empty() ? "anonymous" : std::string(client_id),
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = {},
          .cache_prompt = cache_prompt,
          .cache_persist = cache_persist,
          .cache_prefix_tokens = 0,
          .stop_at_eos = !ignore_eos,
          .stop_sequences = stop_sequences,
          .return_progress = return_progress,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request));
#else
  return TextGenerationBackend::start_complete(
      prompt, max_tokens, sampling_config, is_cancelled, stream_output,
      ignore_eos, client_id, stop_sequences, return_progress, cache_prompt,
      cache_persist);
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_chat(const ChatRequest& request, std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling_config,
                             const CancellationCheck& is_cancelled,
                             bool stream_output) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  auto effective_sampling = sampling_config;
  std::optional<sampling::JsonConstraint::ToolFormat> tool_format;
  auto constrained = ConstrainChatRequest(request, state->scheduler->runner(),
                                          &effective_sampling, &tool_format);
  const auto& effective_request = constrained ? *constrained : request;
  auto prompt = state->scheduler->runner().PreparePrompt(effective_request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  const auto initial =
      state->scheduler->runner().InitialOutputState(effective_request);
  if (request.reasoning.budget_tokens &&
      initial == TextGenerationBackend::InitialOutputState::kReasoning)
    effective_sampling.reasoning_budget =
        state->scheduler->runner().BuildReasoningBudget(
            *request.reasoning.budget_tokens, prompt->tokens);
  const std::string client_id =
      request.client_id.empty() ? "anonymous" : request.client_id;
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt->tokens), max_tokens, effective_sampling, is_cancelled,
      stream_output,
      TextRequestMetadata{
          .client_id = client_id,
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = std::move(prompt->context),
          .cache_prompt = request.cache_prompt,
          .cache_persist = request.cache_persist,
          .cache_prefix_tokens = prompt->cache_prefix_tokens,
          .stop_sequences = request.stop_sequences,
          .return_progress = request.return_progress,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request), initial, tool_format);
#else
  return TextGenerationBackend::start_chat(request, max_tokens, sampling_config,
                                           is_cancelled, stream_output);
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<tokenization::ChatMessage>& messages,
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled) {
  return chat(ChatRequest{messages}, max_tokens, sampling_config, is_cancelled);
}

std::size_t InferenceBackend::count_tokens(std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return 0;
  }
  return state->scheduler->runner().Tokenize(text).size();
#else
  (void)text;
  return 0;
#endif
}

std::optional<std::size_t> InferenceBackend::count_chat_tokens(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return std::nullopt;
  }
  // The same preparation as chat(): constrained tool or format requests add
  // prompt text before the template is rendered.
  auto sampling = sampling::SamplingConfig{};
  auto constrained =
      ConstrainChatRequest(request, state->scheduler->runner(), &sampling);
  const auto& effective_request = constrained ? *constrained : request;
  return state->scheduler->runner().CountPromptTokens(effective_request);
#else
  (void)request;
  return std::nullopt;
#endif
}

std::optional<std::string> InferenceBackend::render_chat_prompt(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return std::nullopt;
  }
  // The same preparation as count_chat_tokens(), so the text is what chat()
  // would tokenize.
  auto sampling = sampling::SamplingConfig{};
  auto constrained =
      ConstrainChatRequest(request, state->scheduler->runner(), &sampling);
  const auto& effective_request = constrained ? *constrained : request;
  return state->scheduler->runner().RenderPrompt(effective_request);
#else
  (void)request;
  return std::nullopt;
#endif
}

}  // namespace gufo::server
