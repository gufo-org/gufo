#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_ENGINE_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/deepseek_v4_flash/chat_template.hpp"
#include "src/models/deepseek_v4_flash/runtime/model.h"

namespace gufo::models::deepseek_v4_flash {

struct ModelOptions {
  std::uint32_t max_context = 4096;
  /// Optional DSpark support model. Empty leaves speculative decoding off.
  std::string dspark_model_path;
};

class Session;
class SessionSnapshot;

struct SessionBatchItem {
  Session* session = nullptr;
  int token = 0;
};

struct SessionDsparkBatchItem {
  Session* session = nullptr;
  std::size_t max_tokens = 32;
  std::uint32_t max_draft_tokens = 5;
  std::vector<int>* emitted = nullptr;
};

/// GPU operations, including session creation/destruction, share scratch
/// storage and must be serialized. Use the batch APIs to advance concurrent
/// requests; the server scheduler owns and serializes their dispatch.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);
  [[nodiscard]] bool EvaluateBatch(std::span<const SessionBatchItem> items,
                                   std::string* error_msg = nullptr) const;
  [[nodiscard]] bool DsparkStepBatch(
      std::span<const SessionDsparkBatchItem> items,
      std::string* error_msg = nullptr) const;
  [[nodiscard]] std::vector<int> Tokenize(std::string_view text) const;
  [[nodiscard]] std::vector<int> EncodeChat(std::string_view system_prompt,
                                            std::string_view user_prompt) const;
  [[nodiscard]] std::vector<int> EncodeChat(
      std::span<const ChatMessage> messages,
      const ChatTemplateOptions& options = {}) const;
  [[nodiscard]] std::vector<int> EncodeChat(
      std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
      const ChatTemplateOptions& options = {}) const;
  [[nodiscard]] std::string DecodeToken(int token) const;
  [[nodiscard]] bool IsStopToken(int token) const;
  [[nodiscard]] int EosToken() const;
  [[nodiscard]] int VocabSize() const;
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept;
  [[nodiscard]] bool HasDspark() const;

private:
  Model(ds4_engine* engine, ModelOptions options);

  ds4_engine* engine_ = nullptr;
  ModelOptions options_;

  friend class Session;
};

class Session final {
public:
  using CancellationCheck = std::function<bool()>;

  ~Session();

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) = delete;
  Session& operator=(Session&&) = delete;

  [[nodiscard]] bool Sync(std::span<const int> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] int SelectNext(float temperature, std::uint64_t* rng_state,
                               int top_k = 0, float top_p = 1.0F,
                               float min_p = 0.0F) const;
  [[nodiscard]] bool Evaluate(int token, std::string* error_msg = nullptr);
  /// True when this session has a DSpark drafter attached.
  [[nodiscard]] bool HasDspark() const;
  /// Reset request statistics and draft policy, retaining a restored prefix.
  void BeginRequest();
  /// Discards DSpark-only request state before exact multi-session execution.
  void PrepareBatchExecution();
  /// Runs one greedy speculative cycle and appends the emitted tokens.
  [[nodiscard]] bool DsparkStep(std::vector<int>* emitted,
                                std::string* error_msg = nullptr);
  /// Runs one greedy speculative cycle without committing more than max_tokens.
  [[nodiscard]] bool DsparkStep(std::size_t max_tokens,
                                std::vector<int>* emitted,
                                std::string* error_msg = nullptr);
  [[nodiscard]] bool DsparkStep(std::size_t max_tokens,
                                std::uint32_t max_draft_tokens,
                                std::vector<int>* emitted,
                                std::string* error_msg = nullptr);
  struct DsparkStats {
    std::uint64_t verifier_rows{0};
    std::uint64_t verifier_accepted{0};
    std::uint64_t support_drafted{0};
    std::uint64_t support_accepted{0};
    std::uint64_t positional_accepted{0};
    std::uint64_t anchors{0};
    std::uint64_t full_blocks{0};
    std::uint64_t steps{0};
    std::uint64_t skipped{0};
    std::uint32_t context_tokens{0};
    bool operator==(const DsparkStats&) const = default;
  };
  [[nodiscard]] DsparkStats DsparkStatistics() const;
  [[nodiscard]] std::vector<float> CopyLogits(
      std::string* error_msg = nullptr) const;
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);
  void SetCancellationCheck(CancellationCheck is_cancelled);
  void Invalidate() noexcept;

  [[nodiscard]] int Position() const;
  [[nodiscard]] int ContextSize() const;
  [[nodiscard]] std::uint32_t PrefillCapacity() const;
  [[nodiscard]] std::uint64_t PayloadBytes() const;

private:
  Session(std::shared_ptr<Model> model, ds4_session* session);

  std::shared_ptr<Model> model_;
  ds4_session* session_ = nullptr;
  CancellationCheck is_cancelled_;

  friend class Model;
};

class SessionSnapshot final {
public:
  ~SessionSnapshot();

  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;
  SessionSnapshot(SessionSnapshot&&) = delete;
  SessionSnapshot& operator=(SessionSnapshot&&) = delete;

  [[nodiscard]] std::uint64_t SizeBytes() const noexcept;
  [[nodiscard]] bool CopyTo(std::span<std::uint8_t> destination) const noexcept;

private:
  explicit SessionSnapshot(ds4_session_snapshot snapshot);

  ds4_session_snapshot snapshot_{};

  friend class Session;
};

}  // namespace gufo::models::deepseek_v4_flash

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_ENGINE_HPP_
