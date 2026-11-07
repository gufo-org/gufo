#ifndef GUFO_MODELS_MINIMAX_H3_TOKENIZER_HPP_
#define GUFO_MODELS_MINIMAX_H3_TOKENIZER_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::minimax_h3 {

class Tokenizer {
public:
  static constexpr std::uint32_t kPadTokenId = 151643;
  static constexpr std::uint32_t kEmbeddingVocabSize = 151936;
  static constexpr std::uint32_t kBaseVocabularySize = 151643;
  static constexpr std::uint32_t kTokenizerVocabularySize = 151669;

  struct LoadOptions {
    bool require_pinned_h3_contract{true};
  };

  [[nodiscard]] static bool Load(const std::filesystem::path& path,
                                 Tokenizer* output,
                                 std::string* error = nullptr);
  [[nodiscard]] static bool Load(const std::filesystem::path& path,
                                 const LoadOptions& options, Tokenizer* output,
                                 std::string* error = nullptr);

  [[nodiscard]] bool Encode(std::string_view text, bool pad_empty,
                            std::vector<std::uint32_t>* ids,
                            std::string* error = nullptr) const;
  [[nodiscard]] bool EncodePrompt(std::string_view prompt,
                                  std::vector<std::uint32_t>* ids,
                                  std::string* error = nullptr) const {
    return Encode(prompt, true, ids, error);
  }
  [[nodiscard]] bool Decode(const std::vector<std::uint32_t>& ids,
                            std::string* text,
                            std::string* error = nullptr) const;

  /// Id of an added (special) token by its exact spelling, e.g. a shared
  /// `tokenization::k*` control-token constant. False when absent.
  [[nodiscard]] bool AddedTokenId(std::string_view content,
                                  std::uint32_t* id) const;

  [[nodiscard]] std::size_t vocabulary_size() const noexcept {
    return inverse_vocab_.size();
  }

private:
  struct AddedToken {
    std::string content;
    std::uint32_t id{0};
  };

  [[nodiscard]] bool EncodePlain(std::string_view text,
                                 std::vector<std::uint32_t>* ids,
                                 std::string* error) const;
  [[nodiscard]] bool ApplyBpe(std::string_view piece,
                              std::vector<std::uint32_t>* ids,
                              std::string* error) const;
  static void ValidatePinnedContract(const Tokenizer& tokenizer);

  std::unordered_map<std::string, std::uint32_t> vocab_;
  std::vector<std::string> inverse_vocab_;
  std::vector<bool> inverse_vocab_present_;
  std::unordered_map<std::string, std::uint32_t> merge_ranks_;
  std::unordered_map<std::string, std::uint32_t> added_tokens_;
  std::vector<std::string> inverse_added_;
  std::vector<bool> inverse_added_present_;
  std::vector<AddedToken> added_alternatives_;
  std::array<std::string, 256> byte_encoder_;
  std::array<std::int16_t, 324> byte_decoder_{};
};

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_TOKENIZER_HPP_
