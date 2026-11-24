#include "src/families/qwen/text/tokenizer.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/control_tokens/control_tokens.hpp"

using gufo::tokenization::kEndOfText;
using gufo::tokenization::kImagePad;
using gufo::tokenization::kImEnd;
using gufo::tokenization::kImStart;

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
      std::string(kEndOfText),  // 0
      std::string(kImStart),    // 1
      std::string(kImEnd),      // 2
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
      {std::string(kEndOfText), 0},
      {std::string(kImStart), 1},
      {std::string(kImEnd), 2},
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
  vocab.emplace_back(kImStart);
  vocab.emplace_back(kImEnd);
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

  // Independent short encodes provide exact token IDs at known regex
  // boundaries. Exercise repetition, memo saturation and unbounded pieces.
  std::string paragraph;
  std::vector<gufo::tokenization::TokenId> expected;
  for (std::size_t i = 0; i < 768; ++i) {
    const auto line =
        (i % 3 == 0 ? std::string("the the e\u0301")
         : i % 3 == 1
             ? std::string(i % 150 + 2, 't')
             : "the " + std::string{static_cast<char>('a' + i % 26),
                                    static_cast<char>('a' + i / 26 % 26)}) +
        "\n";
    paragraph += line;
    const auto part = tok->Encode(line);
    expected.insert(expected.end(), part.begin(), part.end());
  }
  Expect(tok->Encode(paragraph) == expected,
         "long normalized spans preserve exact per-piece BPE tokens");
  Expect(tok->Encode(paragraph) == expected,
         "BPE memo does not retain request state");
}

void TestStopTokensFollowVocabulary() {
  const std::vector<std::string> vocab{
      "text", std::string(kEndOfText), std::string(kImStart),
      std::string(kImEnd), std::string(kImagePad)};
  const std::unordered_map<std::string, gufo::tokenization::TokenId> specials{
      {std::string(kEndOfText), 1},
      {std::string(kImStart), 2},
      {std::string(kImEnd), 3},
      {std::string(kImagePad), 4}};
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

// The same small byte-level tokenizer as Hugging Face tokenizer.json and as
// its GGUF conversion: "Ġ" spells a space, "Í¡" the bytes CD A1 of U+0361
// (a combining mark). Ids 18 and 19 are unused and padded to vocab_size 20.
// The pipeline is the one the Qwen3.5 and Qwen3.8 tokenizer.json files use.
constexpr std::string_view kQwen35TokenizerJson = R"json({"version":"1.0",
  "added_tokens":[
    {"id":11,"content":"<|endoftext|>","single_word":false,"lstrip":false,
     "rstrip":false,"normalized":false,"special":true},
    {"id":12,"content":"<|im_start|>","special":true},
    {"id":13,"content":"<|im_end|>","special":true},
    {"id":14,"content":"<think>","special":false},
    {"id":15,"content":"</think>","special":false}],
  "normalizer":{"type":"NFC"},
  "pre_tokenizer":{"type":"Sequence","pretokenizers":[
    {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
     "behavior":"Isolated","invert":false},
    {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,
     "use_regex":false}]},
  "post_processor":{"type":"ByteLevel","add_prefix_space":false,
    "trim_offsets":false,"use_regex":false},
  "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,
    "use_regex":false},
  "model":{"type":"BPE","dropout":null,"unk_token":null,
    "continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,
    "byte_fallback":false,"ignore_merges":false,
    "vocab":{"a":0,"b":1,"h":2,"i":3,"Ġ":4,"Í":5,"¡":6,"hi":7,"Ġhi":8,
      "aÍ":9,"aÍ¡":10,"1":16,"11":17},
    "merges":["h i",["Ġ","hi"],"a Í","aÍ ¡","1 1"]}})json";
constexpr std::size_t kTokenizerJsonVocabSize = 20;

// The fixture with the first `from` replaced by `to`.
std::string TokenizerJsonWith(std::string_view from, std::string_view to) {
  std::string json(kQwen35TokenizerJson);
  const auto at = json.find(from);
  Expect(at != std::string::npos, "fixture holds " + std::string(from));
  return json.replace(at, from.size(), to);
}

void TestTokenizerJsonMatchesGguf() {
  using gufo::tokenization::QwenTokenizer;
  using gufo::tokenization::TokenId;
  std::string err;
  auto from_json = QwenTokenizer::CreateFromTokenizerJson(
      kQwen35TokenizerJson, kTokenizerJsonVocabSize, &err);
  Expect(from_json != nullptr, "tokenizer.json loads: " + err);

  GgufTokenizerBuilder builder;
  builder.AddMetadataString("tokenizer.ggml.model", "gpt2");
  builder.AddMetadataString("tokenizer.ggml.pre", "qwen35");
  builder.AddMetadataUint32("tokenizer.ggml.eos_token_id", 13U);
  builder.AddMetadataUint32("tokenizer.ggml.bos_token_id", 11U);
  builder.AddMetadataUint32("tokenizer.ggml.padding_token_id", 11U);
  const std::vector<std::string> gguf_tokens{"a",
                                             "b",
                                             "h",
                                             "i",
                                             "Ġ",
                                             "Í",
                                             "¡",
                                             "hi",
                                             "Ġhi",
                                             "aÍ",
                                             "aÍ¡",
                                             "<|endoftext|>",
                                             "<|im_start|>",
                                             "<|im_end|>",
                                             "<think>",
                                             "</think>",
                                             "1",
                                             "11",
                                             "[PAD18]",
                                             "[PAD19]"};
  builder.AddMetadataStringArray("tokenizer.ggml.tokens", gguf_tokens);
  builder.AddMetadataStringArray("tokenizer.ggml.merges",
                                 {"h i", "Ġ hi", "a Í", "aÍ ¡", "1 1"});
  auto binary = builder.Build();
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &err);
  Expect(reader != nullptr, "tokenizer GGUF opens: " + err);
  auto from_gguf = QwenTokenizer::CreateFromGguf(*reader, &err);
  Expect(from_gguf != nullptr, "tokenizer GGUF loads: " + err);

  Expect(from_json->GetVocabSize() == kTokenizerJsonVocabSize &&
             from_gguf->GetVocabSize() == kTokenizerJsonVocabSize,
         "tokenizer.json pads to the model vocabulary size");
  Expect(from_json->GetEosTokenId() == 13U &&
             from_json->GetBosTokenId() == 11U &&
             from_json->GetPadTokenId() == 11U,
         "tokenizer.json EOS, BOS and padding match the GGUF metadata");
  Expect(from_json->GetEosTokenId() == from_gguf->GetEosTokenId() &&
             from_json->GetBosTokenId() == from_gguf->GetBosTokenId() &&
             from_json->GetPadTokenId() == from_gguf->GetPadTokenId(),
         "EOS, BOS and padding are identical");
  for (TokenId id = 0; id < kTokenizerJsonVocabSize; ++id) {
    const std::vector<TokenId> one{id};
    Expect(from_json->IsSpecialToken(id) == from_gguf->IsSpecialToken(id) &&
               from_json->IsStopToken(id) == from_gguf->IsStopToken(id) &&
               from_json->Decode(one) == from_gguf->Decode(one),
           "token " + std::to_string(id) + " matches");
  }
  Expect(from_json->IsSpecialToken(14) && !from_json->IsSpecialToken(10),
         "control tokens are special and BPE pieces are not");

  gufo::tokenization::TokenizerOptions opts;
  opts.parse_special_tokens = true;
  for (const std::string_view text :
       {"hi hi", "11 1", "a\u0361b",
        "<|im_start|>hi<think>a\u0361</think><|im_end|>",
        "ba\u0361 hi<|endoftext|>"}) {
    const auto encoded = from_json->Encode(text, opts);
    Expect(encoded == from_gguf->Encode(text, opts),
           "tokenizer.json and GGUF encode alike: " + std::string(text));
    Expect(from_json->Decode(encoded) == text,
           "tokenizer.json round-trips: " + std::string(text));
  }
  Expect(from_json->Encode("hi hi") == std::vector<TokenId>({7, 8}),
         "array and string merges both apply");
  Expect(from_json->Encode("a\u0361") == std::vector<TokenId>({10}),
         "the Qwen3.5 split keeps a combining mark with its letter");

  Expect(from_json->Encode("11") == std::vector<TokenId>({16, 16}),
         "the Qwen3.5 split keeps digits apart");

  // Without `vocab_size` the table ends at the last entry.
  auto unpadded = QwenTokenizer::CreateFromTokenizerJson(
      std::string(kQwen35TokenizerJson), 0, &err);
  Expect(unpadded != nullptr && unpadded->GetVocabSize() == 18,
         "tokenizer.json without vocab_size loads unpadded: " + err);
  // An added token may repeat its vocabulary entry.
  Expect(QwenTokenizer::CreateFromTokenizerJson(
             TokenizerJsonWith(R"("id":15,)",
                               R"("id":13,"content":"<|im_end|>"},{"id":15,)"),
             kTokenizerJsonVocabSize, &err) != nullptr,
         "a repeated identical added token loads: " + err);
}

// Every file this tokenizer cannot encode exactly as the reference fails.
void TestTokenizerJsonRejectsUnsupported() {
  using gufo::tokenization::QwenTokenizer;
  const auto rejected = [](const std::string& json, std::string_view what,
                           std::size_t vocab_size = kTokenizerJsonVocabSize) {
    std::string err;
    Expect(QwenTokenizer::CreateFromTokenizerJson(json, vocab_size, &err) ==
                   nullptr &&
               !err.empty(),
           "tokenizer.json is rejected: " + std::string(what));
  };
  rejected("{", "malformed JSON");
  rejected(TokenizerJsonWith(R"("type":"BPE")", R"("type":"Unigram")"),
           "non-BPE model");
  // Pipeline: the Qwen3 split (no \p{M}), another normalizer or stage.
  rejected(TokenizerJsonWith(R"([\\p{L}\\p{M}]+)", R"(\\p{L}+)"),
           "Qwen3 split pattern");
  rejected(TokenizerJsonWith(R"("normalizer":{"type":"NFC"})",
                             R"("normalizer":null)"),
           "no normalizer");
  rejected(TokenizerJsonWith(R"({"type":"NFC"})", R"({"type":"NFKC"})"),
           "NFKC normalizer");
  rejected(TokenizerJsonWith(R"("pre_tokenizer":)", R"("unused":)"),
           "no pre-tokenizer");
  rejected(TokenizerJsonWith(R"("Isolated")", R"("Removed")"),
           "split behavior");
  rejected(TokenizerJsonWith(R"("add_prefix_space":false)",
                             R"("add_prefix_space":true)"),
           "byte level with a prefix space");
  rejected(TokenizerJsonWith(R"("decoder":{"type":"ByteLevel")",
                             R"("decoder":{"type":"Metaspace")"),
           "decoder");
  // BPE options other than the defaults.
  rejected(TokenizerJsonWith(R"("dropout":null)", R"("dropout":0.1)"),
           "dropout");
  rejected(TokenizerJsonWith(R"("unk_token":null)", R"("unk_token":"a")"),
           "unknown token");
  rejected(TokenizerJsonWith(R"("continuing_subword_prefix":"")",
                             R"("continuing_subword_prefix":"##")"),
           "continuing subword prefix");
  rejected(TokenizerJsonWith(R"("end_of_word_suffix":"")",
                             R"("end_of_word_suffix":"</w>")"),
           "end of word suffix");
  rejected(
      TokenizerJsonWith(R"("byte_fallback":false)", R"("byte_fallback":true)"),
      "byte fallback");
  rejected(
      TokenizerJsonWith(R"("ignore_merges":false)", R"("ignore_merges":true)"),
      "ignore merges");
  // Ids: duplicates, conflicts with added tokens, the vocabulary size.
  rejected(TokenizerJsonWith(R"("b":1,)", R"("b":0,)"),
           "two entries with one id");
  rejected(TokenizerJsonWith(R"("id":14,)", R"("id":7,)"),
           "an added token over a vocabulary entry");
  rejected(TokenizerJsonWith(R"("b":1,)", R"("b":1,"a2":"x",)"),
           "a non-numeric id");
  rejected(std::string(kQwen35TokenizerJson), "an id at vocab_size", 17);
  rejected(TokenizerJsonWith(R"("b":1,)", R"("b":1,"[PAD18]":19,)"),
           "an entry spelled as padding");
  // Added tokens: only whole control tokens without stripping.
  rejected(TokenizerJsonWith(R"("added_tokens":[)",
                             R"("added_tokens":{},"unused":[)"),
           "non-array added_tokens");
  rejected(
      TokenizerJsonWith(R"("content":"<think>")", R"("content":"thought")"),
      "an added token that is not a control token");
  rejected(TokenizerJsonWith(R"("content":"<think>",)",
                             R"("content":"<think>","lstrip":true,)"),
           "a stripping added token");
  // Merges: one space between two entries that merge into an entry.
  rejected(TokenizerJsonWith(R"("h i")", "7"), "a numeric merge");
  rejected(TokenizerJsonWith(R"(["Ġ","hi"])", R"(["Ġ","hi","a"])"),
           "a three-part merge");
  rejected(TokenizerJsonWith(R"("h i")", R"("hi")"), "a merge without a space");
  rejected(TokenizerJsonWith(R"(["Ġ","hi"])", R"(["Ġ"," hi"])"),
           "a merge part holding a space");
  rejected(TokenizerJsonWith(R"("h i")", R"("h z")"),
           "a merge of an unknown token");
  rejected(TokenizerJsonWith(R"("a Í")", R"("a b")"),
           "a merge into an unknown token");
}

void TestEmptyAndSpecialEdgeCases() {
  std::vector<std::string> tokens = {std::string(kEndOfText), "a", "b", "c"};
  std::vector<std::string> merges = {};
  std::string err;
  auto tok = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      tokens, merges, {{std::string(kEndOfText), 0}}, &err);
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
  vocab.emplace_back(kImStart);
  vocab.emplace_back(kImEnd);
  vocab.emplace_back(kEndOfText);
  vocab.emplace_back("<think>");
  vocab.emplace_back("</think>");

  std::unordered_map<std::string, gufo::tokenization::TokenId> specials = {
      {std::string(kImStart), 256},
      {std::string(kImEnd), 257},
      {std::string(kEndOfText), 258},
      {"<think>", 259},
      {"</think>", 260},
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
  TestTokenizerJsonMatchesGguf();
  TestTokenizerJsonRejectsUnsupported();
  TestEmptyAndSpecialEdgeCases();
  TestCorpusConformance();
  std::cout << "All QwenTokenizer tests passed successfully!\n";
  return 0;
}
