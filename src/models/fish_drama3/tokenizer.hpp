#ifndef GUFO_MODELS_FISH_DRAMA3_TOKENIZER_HPP_
#define GUFO_MODELS_FISH_DRAMA3_TOKENIZER_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::fish_drama3 {

class Drama3Tokenizer {
public:
  explicit Drama3Tokenizer(const std::filesystem::path& model_dir);

  [[nodiscard]] std::vector<std::int32_t> Encode(std::string_view input) const;

private:
  std::string pattern_;
  std::unordered_map<std::string, std::uint32_t> specials_;
  std::unique_ptr<gufo::tokenization::QwenTokenizer> bpe_;
};

}  // namespace gufo::models::fish_drama3

#endif  // GUFO_MODELS_FISH_DRAMA3_TOKENIZER_HPP_
