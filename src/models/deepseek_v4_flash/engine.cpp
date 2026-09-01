#include "src/models/deepseek_v4_flash/engine.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gufo::models::deepseek_v4_flash {
namespace {

constexpr std::size_t kErrorCapacity = 512;

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

std::vector<int> TakeTokens(ds4_tokens* tokens) {
  std::vector<int> result;
  if (tokens->v != nullptr && tokens->len > 0) {
    result.assign(tokens->v, tokens->v + tokens->len);
  }
  ds4_tokens_free(tokens);
  return result;
}

}  // namespace

Model::Model(ds4_engine* engine, ModelOptions options)
    : engine_(engine), options_(options) {}

Model::~Model() {
  if (engine_ != nullptr) {
    ds4_engine_close(engine_);
  }
}

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  if (model_path.empty()) {
    AssignError(error_msg, "DeepSeek V4 Flash model path is empty");
    return nullptr;
  }
  if (options.max_context < 2) {
    AssignError(error_msg, "DeepSeek V4 Flash context must be at least 2");
    return nullptr;
  }
  if (options.power_percent < 1 || options.power_percent > 100) {
    AssignError(error_msg, "DeepSeek V4 Flash power must be in [1, 100]");
    return nullptr;
  }
  if (options.max_context >
      static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    AssignError(error_msg, "DeepSeek V4 Flash context exceeds engine limits");
    return nullptr;
  }

  ds4_engine_options engine_options{};
  engine_options.model_path = model_path.c_str();
  engine_options.dspark_model_path = options.dspark_model_path.empty()
                                         ? nullptr
                                         : options.dspark_model_path.c_str();
  engine_options.context_size = static_cast<int>(options.max_context);
  engine_options.prefill_chunk = options.prefill_chunk;
  engine_options.power_percent = options.power_percent;

  ds4_engine* engine = nullptr;
  if (ds4_engine_open(&engine, &engine_options) != 0 || engine == nullptr) {
    AssignError(error_msg, "failed to load DeepSeek V4 Flash GGUF");
    return nullptr;
  }

  return std::shared_ptr<Model>(new Model(engine, options));
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context < 2 || max_context > options_.max_context ||
      max_context >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    AssignError(error_msg,
                "DeepSeek V4 Flash session context is outside model limits");
    return nullptr;
  }

  ds4_session* session = nullptr;
  if (ds4_session_create(&session, engine_, static_cast<int>(max_context)) !=
          0 ||
      session == nullptr) {
    AssignError(error_msg, "failed to create DeepSeek V4 Flash session");
    return nullptr;
  }

  return std::unique_ptr<Session>(new Session(shared_from_this(), session));
}

std::vector<int> Model::Tokenize(std::string_view text) const {
  const std::string owned(text);
  ds4_tokens tokens{};
  ds4_tokenize_text(engine_, owned.c_str(), &tokens);
  return TakeTokens(&tokens);
}

std::vector<int> Model::EncodeChat(std::string_view system_prompt,
                                   std::string_view user_prompt) const {
  const std::string system(system_prompt);
  const std::string prompt(user_prompt);
  ds4_tokens tokens{};
  ds4_encode_chat_prompt(engine_, system.c_str(), prompt.c_str(), &tokens);
  return TakeTokens(&tokens);
}

std::vector<int> Model::EncodeChat(
    std::span<const ChatMessage> messages) const {
  ds4_tokens tokens{};
  ds4_chat_begin(engine_, &tokens);
  for (const auto& message : messages) {
    ds4_chat_append_message(engine_, &tokens, message.role.c_str(),
                            message.content.c_str());
  }
  ds4_chat_append_assistant_prefix(engine_, &tokens);
  return TakeTokens(&tokens);
}

std::string Model::DecodeToken(int token) const {
  std::size_t length = 0;
  char* text = ds4_token_text(engine_, token, &length);
  if (text == nullptr) {
    return {};
  }
  std::string result(text, length);
  std::free(text);
  return result;
}

bool Model::IsStopToken(int token) const {
  return ds4_token_is_stop(engine_, token);
}

int Model::EosToken() const {
  return ds4_token_eos(engine_);
}

int Model::VocabSize() const {
  return ds4_engine_vocab_size(engine_);
}

std::string Model::ModelName() const {
  const char* name = ds4_engine_model_name(engine_);
  return name != nullptr ? std::string(name) : std::string{};
}

std::uint32_t Model::PrefillChunk() const {
  return ds4_engine_prefill_chunk(engine_);
}

std::uint32_t Model::MaxContext() const noexcept {
  return options_.max_context;
}

Session::Session(std::shared_ptr<Model> model, ds4_session* session)
    : model_(std::move(model)), session_(session) {}

Session::~Session() {
  if (session_ != nullptr) {
    ds4_session_free(session_);
  }
}

bool Session::Sync(std::span<const int> prompt, std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "DeepSeek V4 Flash prompt is empty");
    return false;
  }
  if (prompt.size() >= static_cast<std::size_t>(ContextSize())) {
    AssignError(error_msg,
                "DeepSeek V4 Flash prompt does not fit in session context");
    return false;
  }
  if (prompt.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    AssignError(error_msg, "DeepSeek V4 Flash prompt is too large");
    return false;
  }

  ds4_tokens tokens{
      .v = const_cast<int*>(prompt.data()),
      .len = static_cast<int>(prompt.size()),
      .cap = static_cast<int>(prompt.size()),
  };
  std::array<char, kErrorCapacity> error{};
  const int result =
      ds4_session_sync(session_, &tokens, error.data(), error.size());
  if (result != 0) {
    AssignError(error_msg, error[0] != '\0'
                               ? error.data()
                               : "DeepSeek V4 Flash prefill failed");
    return false;
  }
  return true;
}

void Session::Invalidate() noexcept {
  ds4_session_invalidate(session_);
}

int Session::SelectNext(float temperature, std::uint64_t* rng_state, int top_k,
                        float top_p, float min_p) const {
  if (temperature <= 0.0F) {
    return ds4_session_argmax(session_);
  }
  if (rng_state == nullptr) {
    return -1;
  }
  return ds4_session_sample(session_, temperature, top_k, top_p, min_p,
                            rng_state);
}

int Session::SelectNextExcluding(int excluded_token) const {
  return ds4_session_argmax_excluding(session_, excluded_token);
}

bool Session::Evaluate(int token, std::string* error_msg) {
  std::array<char, kErrorCapacity> error{};
  if (ds4_session_eval(session_, token, error.data(), error.size()) != 0) {
    AssignError(error_msg, error[0] != '\0'
                               ? error.data()
                               : "DeepSeek V4 Flash decode failed");
    return false;
  }
  return true;
}

bool Session::HasDspark() const {
  return ds4_engine_has_dspark(model_->engine_);
}

bool Session::DsparkStep(std::vector<int>* emitted, std::string* error_msg) {
  if (emitted == nullptr) {
    AssignError(error_msg, "DSpark step needs an output buffer");
    return false;
  }
  std::array<int, 32> block{};
  int produced = 0;
  std::array<char, kErrorCapacity> error{};
  if (ds4_session_dspark_step(session_, block.data(),
                              static_cast<int>(block.size()), &produced,
                              error.data(), error.size()) != 0) {
    AssignError(error_msg, error[0] != '\0'
                               ? error.data()
                               : "DeepSeek V4 Flash speculative step failed");
    return false;
  }
  emitted->assign(block.begin(), block.begin() + produced);
  return true;
}

Session::DsparkStats Session::DsparkStatistics() const {
  DsparkStats stats;
  ds4_session_dspark_stats(session_, &stats.verifier_rows,
                           &stats.verifier_accepted, &stats.support_drafted,
                           &stats.support_accepted, &stats.positional_accepted,
                           &stats.anchors, &stats.full_blocks, &stats.steps,
                           &stats.skipped, &stats.context_tokens);
  return stats;
}

bool Session::DsparkDraftSelfTest(int cycles, std::string* error_msg) {
  std::array<char, kErrorCapacity> error{};
  if (ds4_session_dspark_draft_selftest(session_, cycles, error.data(),
                                        error.size()) != 0) {
    AssignError(error_msg,
                error[0] != '\0'
                    ? error.data()
                    : "DeepSeek V4 Flash DSpark draft self-test failed");
    return false;
  }
  return true;
}

bool Session::DsparkSelfTest(int rows, std::string* error_msg) {
  std::array<char, kErrorCapacity> error{};
  if (ds4_session_dspark_selftest(session_, rows, error.data(), error.size()) !=
      0) {
    AssignError(error_msg,
                error[0] != '\0'
                    ? error.data()
                    : "DeepSeek V4 Flash DSpark verifier self-test failed");
    return false;
  }
  return true;
}

std::vector<float> Session::CopyLogits(std::string* error_msg) const {
  const int vocab_size = model_->VocabSize();
  if (vocab_size <= 0) {
    AssignError(error_msg, "DeepSeek V4 Flash vocabulary size is invalid");
    return {};
  }
  std::vector<float> logits(static_cast<std::size_t>(vocab_size));
  if (ds4_session_copy_logits(session_, logits.data(), vocab_size) !=
      vocab_size) {
    AssignError(error_msg, "failed to copy DeepSeek V4 Flash logits");
    return {};
  }
  return logits;
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  ds4_session_snapshot snapshot{};
  std::array<char, kErrorCapacity> error{};
  if (ds4_session_save_snapshot(session_, &snapshot, error.data(),
                                error.size()) != 0) {
    AssignError(error_msg, error[0] != '\0'
                               ? error.data()
                               : "failed to snapshot DeepSeek session");
    return nullptr;
  }
  return std::unique_ptr<SessionSnapshot>(new SessionSnapshot(snapshot));
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  std::array<char, kErrorCapacity> error{};
  if (ds4_session_load_snapshot(session_, &snapshot.snapshot_, error.data(),
                                error.size()) != 0) {
    AssignError(error_msg, error[0] != '\0'
                               ? error.data()
                               : "failed to restore DeepSeek session");
    return false;
  }
  return true;
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  if (payload.empty()) {
    AssignError(error_msg, "DeepSeek session snapshot payload is empty");
    return false;
  }
  ds4_session_snapshot snapshot{
      .ptr = const_cast<std::uint8_t*>(payload.data()),
      .len = payload.size(),
      .cap = payload.size(),
  };
  std::array<char, kErrorCapacity> error{};
  if (ds4_session_load_snapshot(session_, &snapshot, error.data(),
                                error.size()) != 0) {
    AssignError(error_msg, error[0] != '\0'
                               ? error.data()
                               : "failed to restore DeepSeek session");
    return false;
  }
  return true;
}

void Session::SetCancellationCheck(CancellationCheck is_cancelled) {
  is_cancelled_ = std::move(is_cancelled);
  ds4_session_cancel_fn callback = nullptr;
  if (is_cancelled_) {
    callback = +[](void* user_data) -> bool {
      const auto* session = static_cast<const Session*>(user_data);
      return session->is_cancelled_ && session->is_cancelled_();
    };
  }
  ds4_session_set_cancel(session_, callback, this);
}

int Session::Position() const {
  return ds4_session_pos(session_);
}

int Session::ContextSize() const {
  return ds4_session_ctx(session_);
}

std::uint64_t Session::PayloadBytes() const {
  return ds4_session_payload_bytes(session_);
}

SessionSnapshot::SessionSnapshot(ds4_session_snapshot snapshot)
    : snapshot_(snapshot) {}

SessionSnapshot::~SessionSnapshot() {
  if (snapshot_.ptr != nullptr) {
    ds4_session_snapshot_free(&snapshot_);
  }
}

std::uint64_t SessionSnapshot::SizeBytes() const noexcept {
  return snapshot_.len;
}

bool SessionSnapshot::CopyTo(
    std::span<std::uint8_t> destination) const noexcept {
  if (snapshot_.ptr == nullptr || snapshot_.len != destination.size()) {
    return false;
  }
  std::copy_n(snapshot_.ptr, destination.size(), destination.data());
  return true;
}

}  // namespace gufo::models::deepseek_v4_flash
