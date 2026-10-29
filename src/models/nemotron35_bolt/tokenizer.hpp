#ifndef GUFO_MODELS_NEMOTRON35_BOLT_TOKENIZER_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::nemotron35_bolt {

using TokenId = std::uint32_t;
inline constexpr TokenId kInvalidToken = 0xFFFFFFFFU;

/// Splits text the way the artifact's `pixtral` pre-tokenizer regex does,
/// evaluated with Unicode general categories as in Hugging Face tokenizers:
///   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
///   |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
///   |\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// llama.cpp collapses non-ASCII letters to one class and so keeps a
/// lowercase-to-uppercase change across non-ASCII letters ("aÄb") in one piece
/// that this split separates ("a", "Äb"); on the golden corpus BPE still
/// yields the same token IDs. Returns byte ranges of `text`.
[[nodiscard]] std::vector<std::string_view> PreTokenize(std::string_view text);

/// Byte-level BPE tokenizer of the pinned artifact: added tokens are matched
/// before the split, whole pieces found in the vocabulary skip merging
/// (`ignore_merges`), and no BOS is prepended.
class Tokenizer {
public:
  [[nodiscard]] static std::unique_ptr<Tokenizer> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Control tokens (special added tokens such as <|im_start|>) are matched
  /// only with `parse_control`; the other added tokens (<think>, <tool_call>,
  /// ...) are always matched, as in Hugging Face and llama.cpp.
  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text,
                                            bool parse_control) const;
  /// Concatenated bytes of `tokens`; a token may end inside a UTF-8 sequence.
  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens) const;
  /// Bytes of one token; added tokens decode to their literal text.
  [[nodiscard]] std::string_view TokenBytes(TokenId id) const noexcept;

  [[nodiscard]] std::optional<TokenId> AddedToken(
      std::string_view text) const noexcept;
  [[nodiscard]] bool IsControl(TokenId id) const noexcept;
  /// Whether `text` contains the spelling of a control token.
  [[nodiscard]] bool SpellsControl(std::string_view text) const;
  /// <|im_end|> and </s>, the end-of-generation tokens of the
  /// generation config.
  [[nodiscard]] bool IsStop(TokenId id) const noexcept;
  [[nodiscard]] TokenId Eos() const noexcept { return eos_; }
  [[nodiscard]] std::size_t VocabSize() const noexcept { return bytes_.size(); }

private:
  struct Merge {
    std::uint32_t rank;
    TokenId token;
  };
  struct Added {
    TokenId id;
    bool control;
  };

  Tokenizer() = default;
  void EncodePiece(std::string_view piece, std::vector<TokenId>& out) const;
  void EncodeText(std::string_view text, std::vector<TokenId>& out) const;

  std::vector<std::string> bytes_;  ///< Decoded bytes per token id.
  std::unordered_map<std::string, TokenId> by_bytes_;  ///< Normal tokens.
  std::unordered_map<std::uint64_t, Merge> merges_;
  std::array<TokenId, 256> byte_token_{};
  std::unordered_map<std::string, Added> added_;
  std::vector<std::size_t> added_lengths_;  ///< Distinct, longest first.
  std::array<bool, 256> added_first_byte_{};
  std::vector<std::uint8_t> control_;  ///< Per id: control added token.
  TokenId eos_{kInvalidToken};
  TokenId end_of_sequence_{kInvalidToken};
};

}  // namespace gufo::models::nemotron35_bolt

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_TOKENIZER_HPP_
