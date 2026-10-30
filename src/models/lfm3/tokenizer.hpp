#ifndef GUFO_MODELS_LFM3_TOKENIZER_HPP_
#define GUFO_MODELS_LFM3_TOKENIZER_HPP_
#include "src/models/qwen/tokenizer.hpp"
namespace gufo::models::lfm3 {
class Tokenizer {
public:
  static std::unique_ptr<Tokenizer> Load(const core::GgufReader& reader,
                                         std::string* error = nullptr);
  std::vector<std::uint32_t> Encode(std::string_view text,
                                    bool special = true) const;
  // Literal spans suppress control-token matching without introducing BPE
  // boundaries between client text and adjacent template whitespace.
  std::vector<std::uint32_t> EncodeFramed(
      std::string_view text,
      std::span<const std::pair<std::size_t, std::size_t>> literals) const;
  std::string Decode(std::span<const std::uint32_t> tokens) const {
    std::string text;
    for (auto token : tokens)
      text += Piece(token);
    return text;
  }
  std::string_view Piece(std::uint32_t token) const {
    return token < special_pieces_.size() && !special_pieces_[token].empty()
               ? std::string_view(special_pieces_[token])
               : bpe_->DecodeToken(token);
  }
  bool IsStop(std::uint32_t token) const { return bpe_->IsStopToken(token); }
  // GGUF CONTROL tokens (special in the Hugging Face tokenizer), which
  // decoding with skip_special_tokens drops.
  bool IsControl(std::uint32_t token) const {
    return token < control_.size() && control_[token];
  }
  std::uint32_t Bos() const { return bpe_->GetBosTokenId(); }
  static std::string Render(
      std::span<const std::pair<std::string, std::string>> messages,
      bool thinking = true);

private:
  std::vector<std::uint32_t> EncodeImpl(
      std::string_view text, bool special,
      std::span<const std::pair<std::size_t, std::size_t>> literals) const;
  std::unique_ptr<tokenization::QwenTokenizer> bpe_;
  std::vector<std::pair<std::string, std::uint32_t>> special_tokens_;
  std::vector<std::string> special_pieces_;
  std::vector<bool> control_;
};
}  // namespace gufo::models::lfm3
#endif
