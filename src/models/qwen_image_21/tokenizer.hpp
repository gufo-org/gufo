#ifndef GUFO_MODELS_QWEN_IMAGE_21_TOKENIZER_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_TOKENIZER_HPP_

#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::qwen_image_21 {

class Tokenizer {
public:
  explicit Tokenizer(const std::filesystem::path& root);
  std::vector<std::uint32_t> Encode(std::string_view input) const;

private:
  std::unique_ptr<tokenization::QwenTokenizer> bpe_;
  std::unordered_map<std::string, std::uint32_t> specials_;
  std::string pattern_;
};

}  // namespace gufo::models::qwen_image_21
#endif
