#ifndef GUFO_MODELS_GEMMA4_TOKENIZER_HPP_
#define GUFO_MODELS_GEMMA4_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::gemma4 {

using TokenId = std::int32_t;

/// GGUF `tokenizer.ggml.token_type` values.
enum class TokenType : std::uint8_t {
  kUndefined = 0,
  kNormal = 1,
  kUnknown = 2,
  kControl = 3,
  kUserDefined = 4,
  kUnused = 5,
  kByte = 6,
};

/// The `gemma4` tokenizer: SentencePiece-style BPE over raw UTF-8.
///
/// Semantics follow llama.cpp's `gemma4` vocabulary (391fac16):
/// special tokens are partitioned out first (longest text first; control
/// tokens only when parsing special text), spaces become U+2581, text splits
/// into newline and non-newline runs, a newline run that is itself a token is
/// kept whole, BPE merges by rank (ties by leftmost position), and symbols
/// outside the vocabulary fall back to `<0xXX>` byte tokens. Invalid UTF-8
/// bytes become U+FFFD before merging. End-of-generation tokens are treated
/// as control tokens.
class Tokenizer {
public:
  struct Vocabulary {
    std::vector<std::string> tokens;
    std::vector<TokenType> types;
    std::vector<std::string> merges;  ///< "left right", rank = index.
    TokenId bos{-1};
    TokenId eos{-1};
    TokenId pad{-1};
    TokenId unknown{-1};
    bool add_bos{false};
  };

  [[nodiscard]] static std::unique_ptr<Tokenizer> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
  [[nodiscard]] static std::unique_ptr<Tokenizer> Create(
      Vocabulary vocabulary, std::string* error_msg = nullptr);

  /// `parse_special` also matches control tokens in the text; user-defined
  /// tokens are always matched. `add_bos` follows the vocabulary flag only
  /// when requested: chat templates emit `<bos>` themselves.
  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text, bool add_bos,
                                            bool parse_special) const;

  /// Text of one token. Control and unknown tokens render only with
  /// `special`; user-defined tokens always render; byte tokens yield one raw
  /// byte, so a multi-byte character may span several calls.
  [[nodiscard]] std::string TokenText(TokenId token, bool special) const;
  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens,
                                   bool special) const;

  [[nodiscard]] std::size_t VocabSize() const noexcept {
    return tokens_.size();
  }
  [[nodiscard]] TokenId BosToken() const noexcept { return bos_; }
  [[nodiscard]] TokenId EosToken() const noexcept { return eos_; }
  [[nodiscard]] bool IsEndOfGeneration(TokenId token) const noexcept;
  [[nodiscard]] std::span<const TokenId> EndOfGenerationTokens()
      const noexcept {
    return eog_;
  }
  [[nodiscard]] std::optional<TokenId> FindToken(
      std::string_view text) const noexcept;
  [[nodiscard]] TokenType Type(TokenId token) const noexcept;

private:
  Tokenizer() = default;

  void EncodeText(std::string_view text, std::vector<TokenId>* out) const;
  void EncodeWord(std::string_view word, std::vector<TokenId>* out) const;
  /// A pair's merge: its rank (-1 when the pair never merges) and the token
  /// of the concatenated text.
  struct Merge {
    std::int64_t rank;
    TokenId merged;
  };
  [[nodiscard]] Merge FindMerge(TokenId left, TokenId right) const noexcept;

  std::vector<std::string> tokens_;
  std::vector<TokenType> types_;
  std::unordered_map<std::string_view, TokenId> token_to_id_;
  /// Open-addressing merge table keyed by the (left, right) id pair; the
  /// lowest rank wins for a repeated pair.
  struct MergeSlot {
    std::uint64_t key;
    std::uint32_t rank;
    TokenId merged;
  };
  std::vector<MergeSlot> merges_;
  std::uint32_t merge_shift_{64};
  /// Token of each single byte that is itself a vocabulary entry, else -1.
  std::array<TokenId, 256> char_tokens_{};

  /// Recently encoded long pre-split words (whole lines), least recently
  /// used first: a conversation resends every earlier line with each turn.
  struct CachedWord {
    std::string text;
    std::vector<TokenId> ids;
  };
  mutable std::mutex word_cache_mutex_;
  mutable std::list<CachedWord> word_cache_;
  mutable std::unordered_map<std::string_view, std::list<CachedWord>::iterator>
      word_index_;
  mutable std::size_t word_cache_bytes_{0};
  std::vector<TokenId> special_;  ///< Partition order: longest text first.
  std::vector<TokenId> eog_;
  std::array<TokenId, 256> byte_tokens_{};
  TokenId bos_{-1};
  TokenId eos_{-1};
  bool add_bos_{false};
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_TOKENIZER_HPP_
