#ifndef GUFO_MODELS_NEMOTRON35_BOLT_ENGINE_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/nemotron35_bolt/config.hpp"
#include "src/models/nemotron35_bolt/tokenizer.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::nemotron35_bolt {

struct ModelWeights;
class Drafter;
class DraftContext;
namespace rocm {
class DeviceModel;
}

struct ModelOptions {
  /// Longest session the model accepts; each session allocates its own KV
  /// cache up to its requested context.
  std::uint32_t max_context = 32768;
};

class Session;
struct BatchState;

/// Host copy of a session at one position: K and V rows up to the position for
/// each attention block, and the conv and SSM states of each Mamba block.
struct SessionSnapshot {
  std::uint32_t position{0};
  std::vector<std::vector<std::uint16_t>> kv;  ///< Per attention block: K|V.
  std::vector<std::vector<float>> recurrent;   ///< Per Mamba block: conv|SSM.
  std::vector<std::uint8_t> draft;  ///< Drafter context, when it has one.

  [[nodiscard]] std::size_t Bytes() const noexcept;
};

/// One resident model on the GPU; any number of sessions with independent
/// KV caches and Mamba states.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  [[nodiscard]] const Tokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Device bytes one session of `context` tokens allocates.
  [[nodiscard]] std::size_t SessionBytes(std::uint32_t context) const noexcept;
  [[nodiscard]] const std::string& ModelName() const noexcept { return name_; }

  /// Loads NVIDIA's LiLiCorr DFlash drafter for this trunk. Sessions created
  /// afterwards keep its context and can draft.
  [[nodiscard]] bool LoadDrafter(const std::string& path,
                                 std::string* error_msg = nullptr);
  [[nodiscard]] const Drafter* drafter() const noexcept {
    return drafter_.get();
  }

private:
  friend class Session;
  Model() = default;

  ModelOptions options_;
  /// Scratch, streams and captured graphs of Session::DecodeBatch, created
  /// with the first batch; one batch runs at a time.
  mutable std::mutex batch_mutex_;
  mutable std::unique_ptr<BatchState> batch_;
  std::string name_;
  std::unique_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<Drafter> drafter_;
};

/// One sequence: its position, KV cache, Mamba conv/SSM states and scratch.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Appends `tokens` at the current position, in chunks. `logits` receives
  /// the next-token distribution after the last token when it holds one
  /// vocabulary row, or after every token when it holds one row per token;
  /// an empty span skips the output head.
  [[nodiscard]] bool Append(std::span<const std::int32_t> tokens,
                            std::span<float> logits,
                            std::string* error_msg = nullptr);
  /// Append returning one row, which the device also summarizes for
  /// sampling::SamplerState::Sample as `filter` requires: head->candidates
  /// stay empty when the row needs more than a short head.
  [[nodiscard]] bool Append(std::span<const std::int32_t> tokens,
                            std::span<float> logits,
                            const sampling::LogitHeadFilter& filter,
                            sampling::LogitHead* head,
                            std::string* error_msg = nullptr);
  /// Runs 1 to 8 `tokens` at Position() and returns one logits row per token
  /// without advancing: the Mamba states stay put and their inputs are kept
  /// until Commit keeps a prefix. Another Append or Verify discards them.
  [[nodiscard]] bool Verify(std::span<const std::int32_t> tokens,
                            std::span<float> logits,
                            std::string* error_msg = nullptr);
  /// Verify with the head of each row, as Append does: heads[i] for row i.
  [[nodiscard]] bool Verify(std::span<const std::int32_t> tokens,
                            std::span<float> logits,
                            const sampling::LogitHeadFilter& filter,
                            std::span<sampling::LogitHead> heads,
                            std::string* error_msg = nullptr);
  /// Keeps the first `n` tokens of the last Verify and advances past them;
  /// zero discards the verification.
  [[nodiscard]] bool Commit(std::uint32_t n, std::string* error_msg = nullptr);

  /// Appends one token to each of 2 to 8 sessions of one model in a single
  /// pass that reads the weights once: tokens[i] at sessions[i]'s position,
  /// its next-token row into logits[i] and, when filters[i] is not null, its
  /// summary into heads[i]. Each session's recurrent and attention state
  /// advances with the arithmetic of a one-token Append; the matrix products
  /// take the rows together.
  [[nodiscard]] static bool DecodeBatch(
      std::span<Session* const> sessions, std::span<const std::int32_t> tokens,
      std::span<const std::span<float>> logits,
      std::span<const sampling::LogitHeadFilter* const> filters,
      std::span<sampling::LogitHead> heads, std::string* error_msg = nullptr);

  /// Returns to position zero with empty caches and zero states.
  [[nodiscard]] bool Reset(std::string* error_msg = nullptr);
  /// Marks the session empty without device work; the next Append resets it.
  void Invalidate() noexcept;

  /// Bytes Save writes at the current position.
  [[nodiscard]] std::size_t SnapshotBytes() const noexcept;
  [[nodiscard]] bool Save(SessionSnapshot* out,
                          std::string* error_msg = nullptr) const;
  /// Replaces this session's state; the snapshot may come from any session of
  /// the same model with a context no larger than this one.
  [[nodiscard]] bool Restore(const SessionSnapshot& snapshot,
                             std::string* error_msg = nullptr);

  /// The session's drafter state; null when the model had no drafter when
  /// the session was created.
  [[nodiscard]] DraftContext* draft() noexcept;

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return max_context_;
  }

  /// Device buffers of the session; defined in engine.cpp.
  struct State;

private:
  friend class Model;
  Session() = default;
  /// One token through the trunk and head into the device logits row.
  bool RunDecodeGraph(std::string* error_msg);
  bool AppendRows(std::span<const std::int32_t> tokens, std::span<float> logits,
                  const sampling::LogitHeadFilter* filter,
                  std::span<sampling::LogitHead> heads, std::string* error_msg);
  bool VerifyRows(std::span<const std::int32_t> tokens, std::span<float> logits,
                  const sampling::LogitHeadFilter* filter,
                  std::span<sampling::LogitHead> heads, std::string* error_msg);

  std::shared_ptr<const Model> model_;
  std::unique_ptr<State> state_;
  std::uint32_t position_{0};
  std::uint32_t max_context_{0};
  std::uint32_t verified_{0};  ///< Tokens of a Verify awaiting Commit.
  bool needs_reset_{false};
};

}  // namespace gufo::models::nemotron35_bolt

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_ENGINE_HPP_
