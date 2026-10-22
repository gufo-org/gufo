#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_TOKENIZER_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_TOKENIZER_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::models::embedding_gemma2 {

/// Gemma SentencePiece-style BPE tokenizer for EmbeddingGemma 2. It reproduces
/// the Hugging Face `tokenizers` output for `<model_dir>/tokenizer.json`.
/// Special tokens in the text match first, longest match wins. In the
/// remaining text, every space becomes U+2581 and the whole string merges as
/// one BPE word. Bytes of characters missing from the vocabulary fall back to
/// `<0xXX>` tokens. When requested, `<bos>` and `<eos>` wrap the result.
///
/// Only the configuration shipped with the checkpoint is accepted. Any other
/// normalizer, pre-tokenizer, model option, or post-processor is an error.
class Tokenizer {
public:
  /// Reads and validates `<model_dir>/tokenizer.json`. On failure it returns
  /// nullopt and, when `error` is not null, describes the problem.
  [[nodiscard]] static std::optional<Tokenizer> Load(
      const std::string& model_dir, std::string* error = nullptr);

  Tokenizer(Tokenizer&&) noexcept = default;
  Tokenizer& operator=(Tokenizer&&) noexcept = default;
  // The vocabulary index points into `tokens_`, so copies are not allowed.
  // Moves are safe because moving the vector keeps each string's buffer, so
  // the `string_view` keys in `ids_` stay valid.
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  ~Tokenizer() = default;

  /// Encodes UTF-8 `text`. `add_special_tokens` wraps the ids with the
  /// post-processor template (`<bos>` ... `<eos>`). Special tokens that
  /// appear in the text itself are always recognized. Invalid UTF-8 never
  /// fails, and the offending bytes fall back to byte tokens.
  [[nodiscard]] std::vector<std::uint32_t> Encode(
      std::string_view text, bool add_special_tokens = true) const;

  [[nodiscard]] std::size_t vocab_size() const { return tokens_.size(); }
  /// Longest vocabulary entry in bytes of normalized text. No token covers
  /// more, so a text needs at least `size / max_token_bytes()` tokens.
  [[nodiscard]] std::size_t max_token_bytes() const { return max_token_bytes_; }
  [[nodiscard]] std::optional<std::uint32_t> TokenId(
      std::string_view token) const;

private:
  Tokenizer() = default;

  struct Merge {
    std::uint32_t rank{0};
    std::uint32_t new_id{0};
  };

  // U+2581 LOWER ONE EIGHTH BLOCK, the replacement for every ASCII space.
  static constexpr std::string_view kSpaceSymbol = "\xE2\x96\x81";

  struct AddedToken {
    std::string content;
    std::uint32_t id{0};
  };

  [[nodiscard]] static std::uint64_t PairKey(std::uint32_t left,
                                             std::uint32_t right) {
    return (static_cast<std::uint64_t>(left) << 32U) | right;
  }
  [[nodiscard]] const Merge* FindMerge(std::uint32_t left,
                                       std::uint32_t right) const;
  void EncodeSegment(std::string_view text,
                     std::vector<std::uint32_t>* out) const;
  // Length of the added token starting at text[pos], or 0.
  [[nodiscard]] std::size_t MatchAddedToken(std::string_view text,
                                            std::size_t pos,
                                            std::uint32_t* id) const;

  friend struct TokenizerLoader;

  std::vector<std::string> tokens_;
  std::unordered_map<std::string_view, std::uint32_t> ids_;
  std::unordered_map<std::uint64_t, Merge> merges_;
  std::vector<AddedToken> added_;  // longest first
  bool added_first_byte_[256]{};
  std::uint32_t byte_tokens_[256]{};
  std::size_t max_token_bytes_{0};
  std::vector<std::uint32_t> prefix_ids_;
  std::vector<std::uint32_t> suffix_ids_;
};

}  // namespace gufo::models::embedding_gemma2

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_TOKENIZER_HPP_
