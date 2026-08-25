#include "src/cli/serve/inference_backend.hpp"

#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/generator.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/hip/executor.hpp"
#endif

namespace strix::server {
namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

#if defined(ENGINE_ENABLE_HIP)
void EmitRequestMetrics(const InferenceBackend::Result& result,
                        std::string_view status) {
  static std::mutex output_mutex;
  std::ostringstream line;
  line << std::fixed << std::setprecision(3)
       << "{\"event\":\"http_inference\",\"status\":\"" << status
       << "\",\"prompt_tokens\":" << result.prompt_tokens
       << ",\"completion_tokens\":" << result.completion_tokens
       << ",\"ttft_ms\":" << result.ttft_ms
       << ",\"mean_inter_token_ms\":" << result.mean_inter_token_ms
       << ",\"cancelled\":" << (result.cancelled ? "true" : "false") << "}";
  const std::lock_guard<std::mutex> lock(output_mutex);
  std::clog << line.str() << '\n';
}

class GpuSessionPool {
public:
  class Lease {
  public:
    Lease() = default;
    ~Lease() { Release(); }

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    Lease(Lease&& other) noexcept
        : pool_(std::exchange(other.pool_, nullptr)),
          index_(std::exchange(other.index_, 0)) {}

    Lease& operator=(Lease&& other) noexcept {
      if (this != &other) {
        Release();
        pool_ = std::exchange(other.pool_, nullptr);
        index_ = std::exchange(other.index_, 0);
      }
      return *this;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
      return pool_ != nullptr;
    }

    [[nodiscard]] hip::QwenGpuExecutor& Get() const {
      if (pool_ == nullptr) {
        throw std::logic_error("GPU session lease is empty");
      }
      return *pool_->sessions_.at(index_);
    }

  private:
    friend class GpuSessionPool;

    Lease(GpuSessionPool* pool, std::size_t index)
        : pool_(pool), index_(index) {}

    void Release() noexcept {
      if (pool_ != nullptr) {
        pool_->Release(index_);
        pool_ = nullptr;
      }
    }

    GpuSessionPool* pool_{nullptr};
    std::size_t index_{0};
  };

  GpuSessionPool(std::shared_ptr<const hip::QwenGpuModel> model,
                 std::uint32_t max_context, std::size_t session_count) {
    sessions_.reserve(session_count);
    available_.reserve(session_count);
    for (std::size_t index = 0; index < session_count; ++index) {
      std::string error;
      auto session = hip::QwenGpuExecutor::Create(model, &error, max_context);
      if (session == nullptr) {
        throw std::runtime_error("Failed to create GPU session: " + error);
      }
      sessions_.push_back(std::move(session));
      available_.push_back(index);
    }
  }

  [[nodiscard]] Lease Acquire(
      const InferenceBackend::CancellationCheck& is_cancelled) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (available_.empty()) {
      if (is_cancelled && is_cancelled()) {
        return {};
      }
      condition_.wait_for(lock, std::chrono::milliseconds(10));
    }

    const auto index = available_.back();
    available_.pop_back();
    return Lease(this, index);
  }

private:
  void Release(std::size_t index) noexcept {
    sessions_[index]->Reset();
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      available_.push_back(index);
    }
    condition_.notify_one();
  }

  std::vector<std::unique_ptr<hip::QwenGpuExecutor>> sessions_;
  std::vector<std::size_t> available_;
  std::mutex mutex_;
  std::condition_variable condition_;
};

class DeepSeekSessionPool {
public:
  class Lease {
  public:
    Lease() = default;
    ~Lease() { Release(); }

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    Lease(Lease&& other) noexcept
        : pool_(std::exchange(other.pool_, nullptr)),
          index_(std::exchange(other.index_, 0)) {}

    Lease& operator=(Lease&& other) noexcept {
      if (this != &other) {
        Release();
        pool_ = std::exchange(other.pool_, nullptr);
        index_ = std::exchange(other.index_, 0);
      }
      return *this;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
      return pool_ != nullptr;
    }

    [[nodiscard]] models::deepseek_v4_flash::Session& Get() const {
      if (pool_ == nullptr) {
        throw std::logic_error("DeepSeek session lease is empty");
      }
      return *pool_->sessions_.at(index_);
    }

  private:
    friend class DeepSeekSessionPool;

    Lease(DeepSeekSessionPool* pool, std::size_t index)
        : pool_(pool), index_(index) {}

    void Release() noexcept {
      if (pool_ != nullptr) {
        pool_->Release(index_);
        pool_ = nullptr;
      }
    }

    DeepSeekSessionPool* pool_{nullptr};
    std::size_t index_{0};
  };

  DeepSeekSessionPool(std::shared_ptr<models::deepseek_v4_flash::Model> model,
                      std::uint32_t max_context, std::size_t session_count) {
    sessions_.reserve(session_count);
    available_.reserve(session_count);
    for (std::size_t index = 0; index < session_count; ++index) {
      std::string error;
      auto session = model->CreateSession(max_context, &error);
      if (session == nullptr) {
        throw std::runtime_error("Failed to create DeepSeek session: " + error);
      }
      sessions_.push_back(std::move(session));
      available_.push_back(index);
    }
  }

  [[nodiscard]] Lease Acquire(
      const InferenceBackend::CancellationCheck& is_cancelled) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (available_.empty()) {
      if (is_cancelled && is_cancelled()) {
        return {};
      }
      condition_.wait_for(lock, std::chrono::milliseconds(10));
    }
    const auto index = available_.back();
    available_.pop_back();
    return Lease(this, index);
  }

private:
  void Release(std::size_t index) noexcept {
    sessions_[index]->SetCancellationCheck({});
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      available_.push_back(index);
    }
    condition_.notify_one();
  }

  std::vector<std::unique_ptr<models::deepseek_v4_flash::Session>> sessions_;
  std::vector<std::size_t> available_;
  std::mutex mutex_;
  std::condition_variable condition_;
};

std::string_view ChatRoleName(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
      return "system";
    case tokenization::ChatRole::kDeveloper:
      return "developer";
    case tokenization::ChatRole::kAssistant:
      return "assistant";
    case tokenization::ChatRole::kTool:
      return "tool";
    case tokenization::ChatRole::kUser:
      return "user";
  }
  return "user";
}

#endif

}  // namespace

struct InferenceBackend::Impl {
#if defined(ENGINE_ENABLE_HIP)
  struct State {
    enum class Kind : std::uint8_t {
      kQwen,
      kDeepSeekV4Flash,
    };

    Kind kind = Kind::kQwen;
    std::shared_ptr<const hip::QwenGpuModel> qwen_model;
    std::shared_ptr<GpuSessionPool> qwen_sessions;
    std::shared_ptr<models::deepseek_v4_flash::Model> deepseek_model;
    std::shared_ptr<DeepSeekSessionPool> deepseek_sessions;
    std::string model_id;
  };

  [[nodiscard]] std::shared_ptr<const State> Snapshot() const {
    const std::lock_guard<std::mutex> lock(state_mutex);
    return state;
  }

  Result GenerateQwen(std::shared_ptr<const State> current,
                      std::span<const tokenization::TokenId> prompt_tokens,
                      Clock::time_point request_start, std::size_t max_tokens,
                      float temperature,
                      const CancellationCheck& is_cancelled) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }

    auto lease = current->qwen_sessions->Acquire(is_cancelled);
    if (!lease) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }

    models::GenerationOptions options;
    options.max_new_tokens = max_tokens > 0 ? max_tokens : 1;
    options.temperature = temperature;

    std::optional<Clock::time_point> previous_token;
    std::chrono::duration<double, std::milli> inter_token_total{0};
    std::size_t inter_token_samples = 0;
    try {
      result.tokens = lease.Get().Generate(
          prompt_tokens, options, [&](tokenization::TokenId, std::string_view) {
            const auto now = Clock::now();
            if (!previous_token.has_value()) {
              result.ttft_ms =
                  std::chrono::duration<double, std::milli>(now - request_start)
                      .count();
            } else {
              inter_token_total += now - *previous_token;
              ++inter_token_samples;
            }
            previous_token = now;
            if (is_cancelled && is_cancelled()) {
              result.cancelled = true;
              return false;
            }
            return true;
          });
    } catch (...) {
      EmitRequestMetrics(result, "error");
      throw;
    }

    result.completion_tokens = result.tokens.size();
    result.text = current->qwen_model->GetTokenizer().Decode(result.tokens);
    if (inter_token_samples > 0) {
      result.mean_inter_token_ms =
          inter_token_total.count() / static_cast<double>(inter_token_samples);
    }
    EmitRequestMetrics(result, result.cancelled ? "cancelled" : "ok");
    return result;
  }

  Result GenerateDeepSeek(std::shared_ptr<const State> current,
                          std::span<const int> prompt_tokens,
                          Clock::time_point request_start,
                          std::size_t max_tokens, float temperature,
                          const CancellationCheck& is_cancelled) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }

    auto lease = current->deepseek_sessions->Acquire(is_cancelled);
    if (!lease) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }
    lease.Get().SetCancellationCheck(is_cancelled);

    std::string error;
    if (!lease.Get().Sync(prompt_tokens, &error)) {
      if (is_cancelled && is_cancelled()) {
        result.cancelled = true;
        EmitRequestMetrics(result, "cancelled");
        return result;
      }
      EmitRequestMetrics(result, "error");
      throw std::runtime_error("DeepSeek prefill failed: " + error);
    }

    std::optional<Clock::time_point> previous_token;
    std::chrono::duration<double, std::milli> inter_token_total{0};
    std::size_t inter_token_samples = 0;
    std::random_device random_device;
    std::uint64_t rng_state =
        (static_cast<std::uint64_t>(random_device()) << 32U) ^
        static_cast<std::uint64_t>(random_device());

    for (std::size_t index = 0; index < max_tokens; ++index) {
      if (is_cancelled && is_cancelled()) {
        result.cancelled = true;
        break;
      }
      const int token =
          lease.Get().SelectNext(temperature, &rng_state, 0, 1.0F, 0.05F);
      if (token < 0) {
        EmitRequestMetrics(result, "error");
        throw std::runtime_error("DeepSeek token selection failed");
      }
      if (current->deepseek_model->IsStopToken(token)) {
        break;
      }

      const auto now = Clock::now();
      if (!previous_token.has_value()) {
        result.ttft_ms =
            std::chrono::duration<double, std::milli>(now - request_start)
                .count();
      } else {
        inter_token_total += now - *previous_token;
        ++inter_token_samples;
      }
      previous_token = now;
      result.tokens.push_back(static_cast<tokenization::TokenId>(token));
      result.text += current->deepseek_model->DecodeToken(token);

      if (index + 1 < max_tokens && !lease.Get().Evaluate(token, &error)) {
        if (is_cancelled && is_cancelled()) {
          result.cancelled = true;
          break;
        }
        EmitRequestMetrics(result, "error");
        throw std::runtime_error("DeepSeek decode failed: " + error);
      }
    }

    result.completion_tokens = result.tokens.size();
    if (inter_token_samples > 0) {
      result.mean_inter_token_ms =
          inter_token_total.count() / static_cast<double>(inter_token_samples);
    }
    EmitRequestMetrics(result, result.cancelled ? "cancelled" : "ok");
    return result;
  }

  mutable std::mutex state_mutex;
  std::shared_ptr<const State> state;
#endif
};

InferenceBackend::InferenceBackend() : impl_(std::make_unique<Impl>()) {}

InferenceBackend::~InferenceBackend() = default;

bool InferenceBackend::load(const std::string& model_path, std::string* error,
                            std::uint32_t max_context,
                            std::size_t session_count) {
#if defined(ENGINE_ENABLE_HIP)
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    SetError(error, "Failed to open GGUF: " + load_error);
    return false;
  }
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  if (reader->GetMetadataString("general.architecture") == "deepseek4") {
    if (session_count != 1) {
      SetError(error, "DeepSeek V4 Flash currently supports one HTTP session");
      return false;
    }
    auto model = models::deepseek_v4_flash::Model::Load(
        model_path,
        models::deepseek_v4_flash::ModelOptions{
            .max_context = max_context,
            .prefill_chunk = 2048,
            .power_percent = 100,
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create DeepSeek model: " + load_error);
      return false;
    }
    return load(std::move(model), error, max_context, session_count);
  }
  auto model = hip::QwenGpuModel::CreateFromGguf(reader, &load_error);
  if (model == nullptr) {
    SetError(error, "Failed to create GPU model: " + load_error);
    return false;
  }
  return load(std::move(model), error, max_context, session_count);
#else
  (void)model_path;
  (void)max_context;
  (void)session_count;
  SetError(error, "HTTP inference requires the HIP backend");
  return false;
#endif
}

#if defined(ENGINE_ENABLE_HIP)
bool InferenceBackend::load(std::shared_ptr<const hip::QwenGpuModel> model,
                            std::string* error, std::uint32_t max_context,
                            std::size_t session_count) {
  if (model == nullptr) {
    SetError(error, "Qwen GPU model must not be null");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }

  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->kind = Impl::State::Kind::kQwen;
    new_state->qwen_model = std::move(model);
    new_state->qwen_sessions = std::make_shared<GpuSessionPool>(
        new_state->qwen_model, max_context, session_count);
    new_state->model_id = new_state->qwen_model->GetConfig().model_name;
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

bool InferenceBackend::load(
    std::shared_ptr<models::deepseek_v4_flash::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count) {
  if (model == nullptr) {
    SetError(error, "DeepSeek model must not be null");
    return false;
  }
  if (session_count != 1) {
    SetError(error, "DeepSeek V4 Flash currently supports one HTTP session");
    return false;
  }
  if (max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded DeepSeek model context");
    return false;
  }

  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->kind = Impl::State::Kind::kDeepSeekV4Flash;
    new_state->deepseek_model = std::move(model);
    new_state->deepseek_sessions = std::make_shared<DeepSeekSessionPool>(
        new_state->deepseek_model, max_context, session_count);
    new_state->model_id = new_state->deepseek_model->ModelName();
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
#endif

std::string InferenceBackend::model_id() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_id : "unknown";
#else
  return "unknown";
#endif
}

InferenceBackend::Result InferenceBackend::complete(
    std::string_view prompt, std::size_t max_tokens, float temperature,
    const CancellationCheck& is_cancelled) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  if (state->kind == Impl::State::Kind::kDeepSeekV4Flash) {
    const auto prompt_tokens = state->deepseek_model->Tokenize(prompt);
    return impl_->GenerateDeepSeek(state, prompt_tokens, request_start,
                                   max_tokens, temperature, is_cancelled);
  }
  const auto prompt_tokens = state->qwen_model->GetTokenizer().Encode(prompt);
  return impl_->GenerateQwen(state, prompt_tokens, request_start, max_tokens,
                             temperature, is_cancelled);
#else
  (void)prompt;
  (void)max_tokens;
  (void)temperature;
  (void)is_cancelled;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<tokenization::ChatMessage>& messages,
    std::size_t max_tokens, float temperature,
    const CancellationCheck& is_cancelled) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  if (state->kind == Impl::State::Kind::kDeepSeekV4Flash) {
    std::vector<models::deepseek_v4_flash::ChatMessage> deepseek_messages;
    deepseek_messages.reserve(messages.size());
    for (const auto& message : messages) {
      deepseek_messages.push_back({
          .role = std::string(ChatRoleName(message.role)),
          .content = message.content,
      });
    }
    const auto prompt_tokens =
        state->deepseek_model->EncodeChat(deepseek_messages);
    return impl_->GenerateDeepSeek(state, prompt_tokens, request_start,
                                   max_tokens, temperature, is_cancelled);
  }
  const auto prompt_tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
      state->qwen_model->GetTokenizer(), messages);
  if (!prompt_tokens.has_value() || prompt_tokens->empty()) {
    return {};
  }
  return impl_->GenerateQwen(state, *prompt_tokens, request_start, max_tokens,
                             temperature, is_cancelled);
#else
  (void)messages;
  (void)max_tokens;
  (void)temperature;
  (void)is_cancelled;
  return {};
#endif
}

std::size_t InferenceBackend::count_tokens(std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return 0;
  }
  if (state->kind == Impl::State::Kind::kDeepSeekV4Flash) {
    return state->deepseek_model->Tokenize(text).size();
  }
  return state->qwen_model->GetTokenizer().Encode(text).size();
#else
  (void)text;
  return 0;
#endif
}

}  // namespace strix::server
