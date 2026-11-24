// An external extension's view of gufo::qwen_family: only public headers, and
// only gufo_ext_runtime and gufo::qwen_family on the link line. A family
// header outside SURFACE or an implementation the target drops fails here.
#include <cstdlib>
#include <gufo/ext/families/qwen/control_tokens/control_tokens.hpp>
#include <gufo/ext/families/qwen/text/tokenizer.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// Linking the family must not publish the repository root.
#if __has_include("src/families/qwen/config/qwen_config.cpp")
#error "the repository root is reachable through gufo::qwen_family"
#endif

int main() {
  using gufo::tokenization::QwenTokenizer;
  using gufo::tokenization::TokenId;
  const std::vector<std::string> vocab = {
      "a", std::string(gufo::tokenization::kImEnd)};
  const std::unordered_map<std::string, TokenId> specials = {
      {std::string(gufo::tokenization::kImEnd), 1}};
  std::string error;
  auto tokenizer =
      QwenTokenizer::CreateFromVocabulary(vocab, {}, specials, &error);
  if (tokenizer == nullptr) {
    std::cerr << "tokenizer fixture failed: " << error << "\n";
    return EXIT_FAILURE;
  }
  const std::vector<TokenId> ids =
      tokenizer->Encode(gufo::tokenization::kImEnd);
  if (ids != std::vector<TokenId>{1} ||
      tokenizer->Decode(ids) != gufo::tokenization::kImEnd) {
    std::cerr
        << "control token did not round-trip through the public tokenizer\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
