// Inkling 2 byte-level BPE tokenizer (tiktoken-compatible; HF "ignore_merges").
//
// Built from the GGUF tokenizer metadata (tokens in GPT-2 byte-unicode
// spelling, merges, token types). Pre-tokenization is the hand-written scanner
// in pretokenizer.hpp. Encode matches
// tokenizers.Tokenizer.encode(add_special_tokens =False) of the official
// tokenizer.json for text with special-token parsing, and tiktoken's
// encode_ordinary for parse_special=false.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/models/inkling2/text/gguf_meta.hpp"

namespace inkling2::text {

using TokenId = std::uint32_t;
inline constexpr TokenId kInvalidToken = 0xFFFFFFFFu;

class Tokenizer {
public:
  /// Builds a tokenizer. Returns null and sets *error when the data is
  /// unusable.
  [[nodiscard]] static std::unique_ptr<Tokenizer> Create(
      const TokenizerData& data, std::string* error = nullptr);
  [[nodiscard]] static std::unique_ptr<Tokenizer> CreateFromGguf(
      const std::string& path, std::string* error = nullptr);
  /// The tokenizer of an opened GGUF.
  [[nodiscard]] static std::unique_ptr<Tokenizer> CreateFromGguf(
      const gufo::core::GgufReader& reader, std::string* error);

  /// Text to ids. With `parse_special`, spellings of control tokens in `text`
  /// (leftmost, longest) become those tokens; otherwise they are plain text.
  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text,
                                            bool parse_special = true) const;
  void Encode(std::string_view text, bool parse_special,
              std::vector<TokenId>* out) const;
  /// BPE of text with no special-token parsing.
  void EncodeOrdinary(std::string_view text, std::vector<TokenId>* out) const;

  /// Concatenated bytes of the tokens (may end inside a UTF-8 sequence).
  [[nodiscard]] std::string Decode(std::span<const TokenId> ids) const;
  void DecodeTo(std::span<const TokenId> ids, std::string* out) const;
  /// Bytes of one token (control tokens: their literal text).
  [[nodiscard]] std::string_view TokenBytes(TokenId id) const;

  [[nodiscard]] std::size_t VocabSize() const { return bytes_.size(); }
  [[nodiscard]] bool IsSpecial(TokenId id) const {
    return id < is_special_.size() && is_special_[id];
  }
  [[nodiscard]] std::optional<TokenId> FindSpecial(std::string_view text) const;
  [[nodiscard]] std::optional<TokenId> FindToken(std::string_view bytes) const;
  [[nodiscard]] TokenId eos_id() const { return eos_; }
  [[nodiscard]] TokenId pad_id() const { return pad_; }
  [[nodiscard]] bool IsStop(TokenId id) const { return id == eos_; }
  /// All control-token spellings with their ids (longest first).
  [[nodiscard]] const std::vector<std::pair<std::string, TokenId>>& Specials()
      const {
    return specials_;
  }

private:
  Tokenizer() = default;
  void BpePiece(std::string_view piece, std::vector<TokenId>* out) const;

  std::vector<std::string> bytes_;  // id -> raw bytes
  std::vector<std::uint8_t> is_special_;
  std::unordered_map<std::string, TokenId>
      token_to_id_;  // normal tokens, raw bytes
  struct PairHash {
    std::size_t operator()(std::uint64_t k) const noexcept {
      return static_cast<std::size_t>(k * 0x9E3779B97F4A7C15ull >> 20);
    }
  };
  struct Merge {
    std::uint32_t rank;
    TokenId result;
  };
  std::unordered_map<std::uint64_t, Merge, PairHash> merges_;
  std::vector<std::pair<std::string, TokenId>> specials_;
  bool special_first_byte_[256]{};
  TokenId byte_token_[256]{};
  TokenId eos_{kInvalidToken};
  TokenId pad_{kInvalidToken};
};

/// Incremental detokenizer: holds back an incomplete trailing UTF-8 sequence so
/// that streamed text never contains a split character.
class StreamDecoder {
public:
  explicit StreamDecoder(const Tokenizer& tok) : tok_(&tok) {}
  /// Appends one token; returns the newly complete text (possibly empty).
  std::string Push(TokenId id);
  /// Returns held-back bytes (an incomplete sequence is returned as is).
  std::string Flush();
  void Reset() { pending_.clear(); }

private:
  const Tokenizer* tok_;
  std::string pending_;
};

/// Number of bytes at the end of `s` that form an incomplete (but so far valid)
/// UTF-8 sequence prefix; 0 when `s` ends on a character boundary or on bytes
/// that can never complete.
[[nodiscard]] std::size_t IncompleteUtf8Tail(std::string_view s);

}  // namespace inkling2::text
