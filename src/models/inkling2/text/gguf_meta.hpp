// The tokenizer metadata of a GGUF (tokenizer.ggml.* and
// tokenizer.chat_template), read through core::GgufReader.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace inkling2::text {

struct TokenizerData {
  std::string model;  // tokenizer.ggml.model ("gpt2")
  std::string pre;    // tokenizer.ggml.pre ("inkling2")
  std::vector<std::string>
      tokens;  // GPT-2 byte-unicode spelling; specials literal
  std::vector<std::int32_t>
      token_types;                  // 1 normal, 3 control, 4 user defined, ...
  std::vector<std::string> merges;  // "left right"
  std::optional<std::uint32_t> eos_id;
  std::optional<std::uint32_t> eot_id;
  std::optional<std::uint32_t> bos_id;
  std::optional<std::uint32_t> pad_id;
  bool add_bos{false};
  bool add_eos{false};
  std::string chat_template;
};

/// Reads the tokenizer metadata of an opened GGUF. False and *error on
/// failure.
[[nodiscard]] bool ReadTokenizerData(const gufo::core::GgufReader& reader,
                                     TokenizerData* out, std::string* error);

/// Opens the GGUF at `path` and reads its tokenizer metadata. False and
/// *error on failure.
[[nodiscard]] bool ReadTokenizerDataFromGguf(const std::string& path,
                                             TokenizerData* out,
                                             std::string* error);

}  // namespace inkling2::text
