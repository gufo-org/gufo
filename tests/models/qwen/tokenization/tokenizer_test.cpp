#include "src/models/qwen/tokenizer.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

// Simple in-memory GGUF builder for tokenizer tests
class GgufTokenizerBuilder {
public:
  GgufTokenizerBuilder() {
    const char magic[4] = {'G', 'G', 'U', 'F'};
    AppendBytes(magic, 4);
    AppendPod(static_cast<std::uint32_t>(3));
    tensor_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
    metadata_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
  }

  void AddMetadataString(std::string_view key, std::string_view val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendString(val);
    metadata_count_++;
  }

  void AddMetadataUint32(std::string_view key, std::uint32_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataStringArray(std::string_view key,
                              const std::vector<std::string>& arr) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendPod(static_cast<std::uint64_t>(arr.size()));
    for (const auto& s : arr) {
      AppendString(s);
    }
    metadata_count_++;
  }

  std::vector<std::uint8_t> Build() {
    std::memcpy(buffer_.data() + metadata_count_pos_, &metadata_count_,
                sizeof(metadata_count_));
    // Align to 32
    std::size_t rem = buffer_.size() % 32;
    if (rem != 0) {
      buffer_.resize(buffer_.size() + (32 - rem), 0);
    }
    return buffer_;
  }

private:
  void AppendBytes(const void* data, std::size_t len) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), p, p + len);
  }

  template<typename T>
  void AppendPod(T val) {
    AppendBytes(&val, sizeof(T));
  }

  void AppendString(std::string_view s) {
    std::uint64_t len = s.size();
    AppendPod(len);
    AppendBytes(s.data(), len);
  }

  std::vector<std::uint8_t> buffer_;
  std::size_t tensor_count_pos_{0};
  std::size_t metadata_count_pos_{0};
  std::uint64_t metadata_count_{0};
};

void TestDirectVocabularyTokenizer() {
  std::vector<std::string> tokens = {
      "<|endoftext|>",  // 0
      "<|im_start|>",   // 1
      "<|im_end|>",     // 2
      "H",
      "e",
      "l",
      "o",
      "W",
      "r",
      "d",
      " ",  // 3..10
      "He",
      "ll",
      "llo",
      "Hello",
      "World"  // 11..15
  };

  std::vector<std::string> merges = {
      "H e",     // -> He (11)
      "l l",     // -> ll (12)
      "ll o",    // -> llo (13)
      "He llo",  // -> Hello (14)
      "W o",    "r d", "Wo r", "Wor d", "W orld",
  };

  std::unordered_map<std::string, gufo::tokenization::TokenId> specials = {
      {"<|endoftext|>", 0},
      {"<|im_start|>", 1},
      {"<|im_end|>", 2},
  };

  std::string err;
  auto tok = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      tokens, merges, specials, &err);
  Expect(tok != nullptr, "CreateFromVocabulary succeeds: " + err);
  Expect(tok->GetVocabSize() == tokens.size(), "Vocab size matches");

  // Test special token recognition
  Expect(tok->IsSpecialToken(0), "Token 0 is special");
  Expect(tok->IsSpecialToken(1), "Token 1 is special");
  Expect(tok->IsSpecialToken(2), "Token 2 is special");
  Expect(!tok->IsSpecialToken(3), "Token 3 is not special");

  // Encode with special tokens
  gufo::tokenization::TokenizerOptions opts;
  opts.parse_special_tokens = true;

  auto encoded = tok->Encode("<|im_start|>Hello World<|im_end|>", opts);
  Expect(!encoded.empty(), "Encoded not empty");
  Expect(encoded.front() == 1, "First token is <|im_start|>");
  Expect(encoded.back() == 2, "Last token is <|im_end|>");

  // Decode back
  auto decoded = tok->Decode(encoded);
  Expect(decoded == "<|im_start|>Hello World<|im_end|>",
         "Decoded text matches original");
}

void TestGgufTokenizerLoading() {
  GgufTokenizerBuilder builder;
  builder.AddMetadataString("tokenizer.ggml.model", "gpt2");
  builder.AddMetadataString("tokenizer.ggml.pre", "qwen35");
  builder.AddMetadataUint32("tokenizer.ggml.eos_token_id", 151645U);
  builder.AddMetadataUint32("tokenizer.ggml.padding_token_id", 151643U);

  std::vector<std::string> vocab;
  vocab.reserve(256 + 10);
  for (int i = 0; i < 256; ++i) {
    vocab.emplace_back(1, static_cast<char>(i));
  }
  vocab.emplace_back("th");
  vocab.emplace_back("the");
  vocab.emplace_back("<|im_start|>");
  vocab.emplace_back("<|im_end|>");
  const auto tool_call_start =
      static_cast<gufo::tokenization::TokenId>(vocab.size());
  vocab.emplace_back("<tool_call>");
  const auto tool_call_end =
      static_cast<gufo::tokenization::TokenId>(vocab.size());
  vocab.emplace_back("</tool_call>");
  const auto tool_response_start =
      static_cast<gufo::tokenization::TokenId>(vocab.size());
  vocab.emplace_back("<tool_response>");
  const auto tool_response_end =
      static_cast<gufo::tokenization::TokenId>(vocab.size());
  vocab.emplace_back("</tool_response>");

  builder.AddMetadataStringArray("tokenizer.ggml.tokens", vocab);
  builder.AddMetadataStringArray("tokenizer.ggml.merges", {"t h", "th e"});

  auto binary = builder.Build();

  std::string err;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "Reader open succeeds: " + err);

  auto tok = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
  Expect(tok != nullptr, "Tokenizer created from GGUF: " + err);
  Expect(tok->GetEosTokenId() == 151645U, "EOS token ID matches");
  Expect(tok->GetPadTokenId() == 151643U, "PAD token ID matches");

  // Encode "the"
  auto encoded = tok->Encode("the");
  Expect(!encoded.empty(), "Encoded 'the' is not empty");
  auto decoded = tok->Decode(encoded);
  Expect(decoded == "the", "Roundtrip decoding of 'the' matches");

  encoded = tok->Encode("<tool_call>x</tool_call>");
  Expect(encoded.size() == 3 && encoded.front() == tool_call_start &&
             encoded.back() == tool_call_end,
         "Qwen tool-call markers are parsed as special tokens");
  encoded = tok->Encode("<tool_response>x</tool_response>");
  Expect(encoded.size() == 3 && encoded.front() == tool_response_start &&
             encoded.back() == tool_response_end,
         "Qwen tool-response markers are parsed as special tokens");
}

void TestStopTokensFollowVocabulary() {
  const std::vector<std::string> vocab{"text", "<|endoftext|>", "<|im_start|>",
                                       "<|im_end|>", "<|image_pad|>"};
  const std::unordered_map<std::string, gufo::tokenization::TokenId> specials{
      {"<|endoftext|>", 1},
      {"<|im_start|>", 2},
      {"<|im_end|>", 3},
      {"<|image_pad|>", 4}};
  auto direct = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocab, {}, specials);
  Expect(direct && direct->IsStopToken(1) && direct->IsStopToken(3),
         "direct vocabulary derives both generation terminators");
  Expect(!direct->IsStopToken(0) && !direct->IsStopToken(4) &&
             !direct->IsStopToken(151643) && !direct->IsStopToken(248044) &&
             !direct->IsStopToken(248046) &&
             !direct->IsStopToken(gufo::tokenization::kInvalidTokenId),
         "ordinary, image and foreign-vocabulary IDs do not end generation");

  GgufTokenizerBuilder builder;
  builder.AddMetadataStringArray("tokenizer.ggml.tokens", vocab);
  builder.AddMetadataUint32("tokenizer.ggml.eos_token_id", 3);
  builder.AddMetadataUint32("tokenizer.ggml.padding_token_id", 4);
  auto bytes = builder.Build();
  auto reader = gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size());
  Expect(reader != nullptr, "stop-token GGUF opens");
  auto gguf = gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader);
  Expect(gguf && gguf->IsStopToken(1) && gguf->IsStopToken(3) &&
             !gguf->IsStopToken(gguf->GetPadTokenId()),
         "GGUF padding is not a generation terminator");
}

void TestUnicodeContractionBoundary() {
  GgufTokenizerBuilder builder;
  builder.AddMetadataString("tokenizer.ggml.model", "gpt2");
  builder.AddMetadataString("tokenizer.ggml.pre", "qwen35");
  // Byte-level spellings for U+017F, with a merge across the boundary that
  // the official case-insensitive 's contraction must prevent.
  builder.AddMetadataStringArray("tokenizer.ggml.tokens",
                                 {"'", "Å", "¿", "a", "Å¿", "Å¿a"});
  builder.AddMetadataStringArray("tokenizer.ggml.merges", {"Å ¿", "Å¿ a"});
  auto binary = builder.Build();
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &error);
  Expect(reader != nullptr, error);
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  Expect(tokenizer != nullptr, error);
  Expect(tokenizer->Encode("'ſa") ==
             std::vector<gufo::tokenization::TokenId>({0, 4, 3}),
         "Unicode contraction prevents a merge into the following word");
}

void TestEmptyAndSpecialEdgeCases() {
  std::vector<std::string> tokens = {"<|endoftext|>", "a", "b", "c"};
  std::vector<std::string> merges = {};
  std::string err;
  auto tok = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      tokens, merges, {{"<|endoftext|>", 0}}, &err);
  Expect(tok != nullptr, "Tokenizer initialized");

  gufo::tokenization::TokenizerOptions opts;
  opts.add_bos = false;
  opts.add_eos = false;
  auto empty_encoded = tok->Encode("", opts);
  Expect(empty_encoded.empty(), "Empty input encodes to empty vector");

  auto empty_decoded = tok->Decode({});
  Expect(empty_decoded.empty(), "Empty vector decodes to empty string");
}

void TestCorpusConformance() {
  std::vector<std::string> vocab;
  vocab.reserve(256 + 20);
  for (int i = 0; i < 256; ++i) {
    vocab.emplace_back(1, static_cast<char>(i));
  }
  vocab.emplace_back("<|im_start|>");
  vocab.emplace_back("<|im_end|>");
  vocab.emplace_back("<|endoftext|>");
  vocab.emplace_back("<think>");
  vocab.emplace_back("</think>");

  std::unordered_map<std::string, gufo::tokenization::TokenId> specials = {
      {"<|im_start|>", 256}, {"<|im_end|>", 257}, {"<|endoftext|>", 258},
      {"<think>", 259},      {"</think>", 260},
  };

  std::string err;
  auto tok = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocab, {}, specials, &err);
  Expect(tok != nullptr, "Tokenizer initialized for corpus conformance");

  const std::vector<std::string> test_strings = {
      "",
      "Hello, world! Welcome to Strix Halo.",
      "  Leading spaces\n\nDouble newline\tTab\r\nCRLF",
      "你好世界！ Привет мир! Γειά σου κόσμε! مرحبا بالعالم! 🚀✨🔥",
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
      "<think>\nLet's analyze the hardware "
      "architecture.\n</think><|endoftext|>",
      "template <typename T>\nconstexpr T Square(T x) noexcept {\n  return x "
      "* "
      "x;\n}\n",
      "3.1415926535 0xDEADBEEF 128*1024=131072",
  };

  gufo::tokenization::TokenizerOptions opts;
  opts.parse_special_tokens = true;

  for (const auto& text : test_strings) {
    auto encoded = tok->Encode(text, opts);
    auto decoded = tok->Decode(encoded);
    Expect(decoded == text, "Corpus roundtrip matches exactly: " + text);
  }
}

}  // namespace

int main() {
  std::cout << "Running QwenTokenizer unit tests...\n";
  TestDirectVocabularyTokenizer();
  TestGgufTokenizerLoading();
  TestStopTokensFollowVocabulary();
  TestUnicodeContractionBoundary();
  TestEmptyAndSpecialEdgeCases();
  TestCorpusConformance();
  std::cout << "All QwenTokenizer tests passed successfully!\n";
  return 0;
}
