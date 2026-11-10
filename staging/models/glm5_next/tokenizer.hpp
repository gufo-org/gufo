#ifndef GUFO_MODELS_GLM5_NEXT_TOKENIZER_HPP_
#define GUFO_MODELS_GLM5_NEXT_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::glm5_next {

using TokenId = std::uint32_t;

constexpr TokenId kInvalidTokenId = 0xFFFFFFFFU;

/// GGUF `tokenizer.ggml.token_type` classification. GLM-5 registers both the
/// CONTROL (154820-154840, 154852-154855) and USER_DEFINED (154841-154851)
/// entries as parse specials, matching llama.cpp's special set; UNUSED pad
/// entries encode as ordinary text.
enum class TokenType : std::int32_t {
  kNormal = 1,
  kUnknown = 2,
  kControl = 3,
  kUserDefined = 4,
  kUnused = 5,
  kByte = 6,
};

/// Configuration options for the encoding pass. The GLM-5 artifact carries
/// neither `add_bos_token` nor `add_eos_token`; callers keep the defaults and
/// frame the prompt through the chat template instead.
struct TokenizerOptions {
  bool add_bos{false};
  bool add_eos{false};
  bool parse_special_tokens{true};
};

/// Deterministic GLM-5 gpt2-BPE tokenizer: Unicode split pieces following the
/// llama.cpp `glm4` (`LLAMA_VOCAB_PRE_TYPE_CHATGLM4`) pre-tokenizer rules, then
/// byte-level merge application.
class GlmTokenizer {
public:
  ~GlmTokenizer() = default;

  GlmTokenizer(const GlmTokenizer&) = delete;
  GlmTokenizer& operator=(const GlmTokenizer&) = delete;
  GlmTokenizer(GlmTokenizer&& other) noexcept = default;
  GlmTokenizer& operator=(GlmTokenizer&& other) noexcept = default;

  /// Creates a tokenizer from GGUF metadata. Requires
  /// `tokenizer.ggml.model == "gpt2"` and `tokenizer.ggml.pre == "glm4"`.
  /// Special tokens are every CONTROL- or USER_DEFINED-typed vocabulary
  /// entry; stop tokens are `tokenizer.ggml.eog_token_ids` plus the eos id.
  [[nodiscard]] static std::unique_ptr<GlmTokenizer> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Creates a tokenizer directly from vocabulary lists (tests and fixtures).
  /// `special_ids` and `eog_ids` carry the special classifications a GGUF
  /// would express through token_type and eog metadata.
  [[nodiscard]] static std::unique_ptr<GlmTokenizer> CreateFromVocabulary(
      std::span<const std::string> tokens, std::span<const std::string> merges,
      std::span<const TokenId> special_ids, TokenId bos_token,
      TokenId eos_token, TokenId pad_token,
      std::span<const TokenId> eog_ids = {}, std::string* error_msg = nullptr);

  /// Encodes a UTF-8 text string into token IDs.
  [[nodiscard]] std::vector<TokenId> Encode(
      std::string_view text, const TokenizerOptions& options = {}) const;

  /// Decodes a sequence of token IDs into a UTF-8 string.
  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens) const;

  /// Decodes a single token ID into its string representation.
  [[nodiscard]] std::string_view DecodeToken(TokenId token_id) const noexcept;
  /// Decodes a single token by value.
  [[nodiscard]] std::string DecodeTokenCopy(TokenId token_id) const;

  [[nodiscard]] std::size_t GetVocabSize() const noexcept {
    return id_to_token_.size();
  }

  [[nodiscard]] TokenId GetBosTokenId() const noexcept { return bos_token_id_; }
  [[nodiscard]] TokenId GetEosTokenId() const noexcept { return eos_token_id_; }
  [[nodiscard]] TokenId GetPadTokenId() const noexcept { return pad_token_id_; }

  /// End-of-generation set: eog metadata plus eos.
  [[nodiscard]] bool IsStopToken(TokenId token) const noexcept {
    return token != kInvalidTokenId && stop_token_ids_.contains(token);
  }

  [[nodiscard]] std::optional<TokenId> FindSpecialToken(
      std::string_view token_str) const noexcept;
  [[nodiscard]] bool IsSpecialToken(TokenId id) const noexcept;

  /// Special tokens in one contiguous list, scanned per Encode call.
  [[nodiscard]] const std::vector<std::pair<std::string, TokenId>>&
  SpecialTokens() const noexcept {
    return special_token_list_;
  }

private:
  struct PairHash {
    std::size_t operator()(
        const std::pair<TokenId, TokenId>& p) const noexcept {
      return (static_cast<std::size_t>(p.first) << 32) ^
             static_cast<std::size_t>(p.second);
    }
  };

  /// Merge table entry: the rank that orders merges, and the token the pair
  /// merges into (kInvalidTokenId when the vocabulary has no such token).
  struct Merge {
    std::uint32_t rank;
    TokenId token;
  };

  GlmTokenizer() = default;

  void InitializeByteTokens();
  std::vector<TokenId> BpeMergeChunk(std::string_view chunk) const;
  std::vector<TokenId> BpeEncodeText(std::string_view text) const;

  std::vector<std::string> id_to_token_;
  std::vector<std::string> id_to_decoded_token_;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::unordered_map<std::pair<TokenId, TokenId>, Merge, PairHash> merge_ranks_;
  std::unordered_map<std::string, TokenId> special_token_to_id_;
  std::vector<std::pair<std::string, TokenId>> special_token_list_;
  std::unordered_set<TokenId> is_special_token_;

  std::array<TokenId, 256> byte_tokens_{};
  TokenId bos_token_id_{kInvalidTokenId};
  TokenId eos_token_id_{kInvalidTokenId};
  TokenId pad_token_id_{kInvalidTokenId};
  std::unordered_set<TokenId> stop_token_ids_;
};

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_TOKENIZER_HPP_
