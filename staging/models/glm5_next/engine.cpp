#include "staging/models/glm5_next/engine.hpp"

#include <algorithm>
#include <utility>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "staging/models/glm5_next/kernels/rocm/device_model.hpp"
#include "staging/models/glm5_next/kernels/rocm/executor.hpp"
#include "staging/models/glm5_next/weights.hpp"

namespace gufo::models::glm5_next {
namespace {

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.max_context == 0) {
    AssignError(error_msg, "max_context must be positive");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  const Config& c = m->weights_->config;
  if (options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  m->tokenizer_ = GlmTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_, error_msg);
  if (m->device_ == nullptr) {
    return nullptr;
  }
  m->executor_ = rocm::Executor::Create(*m->device_, error_msg);
  if (m->executor_ == nullptr) {
    return nullptr;
  }
  return m;
}

std::shared_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  if (!live_session_.expired()) {
    AssignError(error_msg,
                "the GLM-5 device state is shared by all sessions");
    return nullptr;
  }
  std::unique_ptr<rocm::Session> state =
      executor_->CreateSession(max_context, error_msg);
  if (state == nullptr) {
    return nullptr;
  }
  auto session = std::shared_ptr<Session>(
      new Session(shared_from_this(), std::move(state), max_context));
  live_session_ = session;
  return session;
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  const std::vector<TokenId> ids = tokenizer_->Encode(text);
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<TokenId> ids;
  ids.reserve(tokens.size());
  for (const std::int32_t t : tokens) {
    ids.push_back(static_cast<TokenId>(t));
  }
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("GLM-5.3-Flash"));
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_ == nullptr ? 0 : device_->resident_bytes();
}

std::size_t Model::SessionBytes(std::uint32_t context) const noexcept {
  return executor_ == nullptr ? 0 : executor_->SessionBytes(context);
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> state,
                 std::uint32_t max_context)
    : model_(std::move(model)),
      state_(std::move(state)),
      max_context_(max_context) {}

Session::~Session() = default;

void Session::SyncLogits() {
  logits_.resize(model_->VocabSize());
  if (hipMemcpy(logits_.data(), model_->executor_->logits(),
                logits_.size() * sizeof(float), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    valid_ = false;
  }
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.size() > max_context_) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (prompt.empty()) {
    Reset();
    return true;
  }
  logits_.assign(model_->VocabSize(), 0.0f);
  const auto common = std::mismatch(tokens_.begin(), tokens_.end(),
                                    prompt.begin(), prompt.end());
  if (common.first != tokens_.end()) {
    state_->Reset();
    tokens_.clear();
    model_->executor_->Prefill(*state_, prompt.data(), prompt.size());
    tokens_.assign(prompt.begin(), prompt.end());
    SyncLogits();
    return valid_;
  }
  const std::span<const std::int32_t> tail(
      prompt.begin() +
          static_cast<std::ptrdiff_t>(common.first - tokens_.begin()),
      prompt.end());
  if (!tail.empty()) {
    model_->executor_->Prefill(*state_, tail.data(), tail.size());
    tokens_.insert(tokens_.end(), tail.begin(), tail.end());
    SyncLogits();
  }
  return valid_;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (tokens_.size() + 1 > max_context_) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  model_->executor_->Step(*state_, token);
  tokens_.push_back(token);
  SyncLogits();
  return valid_;
}

std::uint32_t Session::Position() const noexcept {
  return static_cast<std::uint32_t>(std::max<std::int64_t>(
      state_->position() + 1, 0));
}

void Session::Reset() {
  state_->Reset();
  tokens_.clear();
  logits_.clear();
  valid_ = true;
}

}  // namespace gufo::models::glm5_next
