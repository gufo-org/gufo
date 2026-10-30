#ifndef GUFO_MODELS_LFM3_ENGINE_HPP_
#define GUFO_MODELS_LFM3_ENGINE_HPP_
#include <memory>

#include "src/models/lfm3/tokenizer.hpp"
#include "src/models/lfm3/weights.hpp"
namespace gufo::models::lfm3 {
class Session;
class Model : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  static std::shared_ptr<Model> Load(const std::string& path,
                                     std::string* error = nullptr);
  std::unique_ptr<Session> CreateSession(std::uint32_t context,
                                         std::string* error = nullptr);
  const Config& config() const;
  const Tokenizer& tokenizer() const;

private:
  Model();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class Session;
};
class Session {
public:
  static constexpr std::size_t kPrefillChunk = 512;
  static constexpr std::size_t kMaxBatch = 8;
  ~Session();
  // d1 complete prompt: full attention is bidirectional, while
  // convolution and sliding attention remain causal. Starts from empty state.
  // Embeddings, when supplied, replace every input row (host floats).
  bool ForwardDecision(std::span<const std::uint32_t> tokens,
                       std::span<const float> embeddings,
                       std::span<float> logits, std::string* error = nullptr);
  // Packed d1: -1 marks state rows; positive IDs mark question groups.
  // Full-attention state rows see all rows; question rows see the state and
  // their own group. Empty groups means one bidirectional prompt. Projects
  // only positions through vocabulary (empty vocabulary means every token).
  // logits is [positions.size()][vocabulary.size() or config.vocab].
  bool ForwardDecision(std::span<const std::uint32_t> tokens,
                       std::span<const float> embeddings,
                       std::span<const std::int32_t> groups,
                       std::span<const std::uint32_t> positions,
                       std::span<const std::uint32_t> vocabulary,
                       std::span<float> logits, std::string* error = nullptr);
  // Advances text tokens and optionally returns the final logits.
  // State is session-owned; device weights are shared by sessions.
  bool Forward(std::span<const std::uint32_t> tokens,
               std::span<float> logits = {}, std::string* error = nullptr);
  // One token per independent session, sharing dense projections across rows.
  static bool ForwardBatch(std::span<Session* const> sessions,
                           std::span<const std::uint32_t> tokens,
                           std::span<std::span<float>> logits,
                           std::string* error = nullptr);
  // One row per independent session, sharing dense projections across rows:
  // rows[i], when set, is session i's input embedding in device memory
  // (ready before the call), otherwise tokens[i] is. Logits are computed
  // only when requested; DeviceHidden of every session is updated.
  static bool ForwardBatchInputs(std::span<Session* const> sessions,
                                 std::span<const std::uint32_t> tokens,
                                 std::span<const float* const> rows,
                                 bool logits, std::string* error = nullptr);
  // Advances rows of input embeddings (hidden floats per row) instead of
  // token IDs. Rows live in host memory, or in device memory on this
  // session's stream when on_device is set.
  bool ForwardEmbeddings(std::span<const float> rows, bool on_device = false,
                         std::span<float> logits = {},
                         std::string* error = nullptr);
  // Final-normed hidden state of the last advanced row (hidden floats).
  bool ReadHidden(std::span<float> hidden, std::string* error = nullptr) const;
  // Device copy of the same hidden state, valid until the next forward.
  const float* DeviceHidden() const;
  void* DeviceStream() const;
  bool Reset(std::string* error = nullptr);
  std::uint32_t Position() const;
  // Payloads are compact, immutable host bytes. Persistent callers must bind
  // them to an artifact identity as well as this versioned state layout.
  std::size_t SnapshotBytes() const;
  std::vector<std::uint8_t> SaveSnapshot() const;
  bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                       std::string* error = nullptr);
  bool ReadLogits(std::span<float> logits, std::string* error = nullptr) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Session(std::shared_ptr<Model> model, std::uint32_t context);
  friend class Model;
};
}  // namespace gufo::models::lfm3
#endif
