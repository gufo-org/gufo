#ifndef GUFO_MODELS_QWEN3_ASR_TOKENIZER_HPP_
#define GUFO_MODELS_QWEN3_ASR_TOKENIZER_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gufo::tokenization {
class QwenTokenizer;
}

namespace gufo::models::qwen3_asr {

class Tokenizer {
public:
  Tokenizer();
  ~Tokenizer();
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  [[nodiscard]] static bool Load(const std::filesystem::path& model_root,
                                 Tokenizer* output,
                                 std::string* error = nullptr);

  [[nodiscard]] std::vector<std::uint32_t> Encode(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::uint32_t> ids,
                                   bool skip_special_tokens) const;

private:
  void EnsureEncodingTokenizer() const;

  mutable std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::filesystem::path model_root_;
  std::vector<std::string> tokens_;
  std::unordered_map<std::string, std::uint32_t> added_tokens_;
  std::unordered_set<std::uint32_t> special_ids_;
  mutable bool merges_loaded_{false};
};

}  // namespace gufo::models::qwen3_asr

#endif  // GUFO_MODELS_QWEN3_ASR_TOKENIZER_HPP_
