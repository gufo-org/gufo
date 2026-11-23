#ifndef GUFO_MODELS_YUE2_TOKENIZER_HPP_
#define GUFO_MODELS_YUE2_TOKENIZER_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::models::yue2 {

/// The checkpoint's frozen text and ABC tokenizer: tiktoken byte-pair ranks
/// (`qwen.tiktoken`), the Qwen2 split pattern and NFC normalization. Text
/// never produces special tokens (upstream `encode_ordinary`).
class TextTokenizer {
public:
  static constexpr std::uint32_t kOrdinaryTokens = 151643;

  [[nodiscard]] static std::unique_ptr<TextTokenizer> Load(
      const std::filesystem::path& ranks_file, std::string* error);

  [[nodiscard]] std::vector<std::uint32_t> Encode(std::string_view text) const;
  /// Ids outside the ordinary range are skipped; invalid UTF-8 becomes
  /// U+FFFD, as upstream `decode(errors="replace")`.
  [[nodiscard]] std::string Decode(std::span<const std::uint32_t> ids) const;

private:
  TextTokenizer() = default;
  void MergePiece(std::string_view piece,
                  std::vector<std::uint32_t>* output) const;

  std::vector<std::string> tokens_;
  std::unordered_map<std::string, std::uint32_t> ranks_;
};

/// Splits NFC text with the Qwen2 pattern
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
///  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`. Exposed for tests.
[[nodiscard]] std::vector<std::string_view> SplitQwen2(std::string_view text);

}  // namespace gufo::models::yue2

#endif  // GUFO_MODELS_YUE2_TOKENIZER_HPP_
