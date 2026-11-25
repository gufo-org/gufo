#include "src/models/inkling2/text/gguf_meta.hpp"

#include <gufo/ext/gguf_reader.hpp>
#include <limits>
#include <string_view>
#include <variant>

namespace inkling2::text {

namespace {

// An integer id stored as any GGUF integer type; false when the key is
// present but not an id.
bool ReadId(const gufo::core::GgufReader& reader, std::string_view key,
            std::optional<std::uint32_t>* out) {
  if (reader.FindMetadata(key) == nullptr)
    return true;
  *out = reader.GetMetadataUint32(key);
  return out->has_value();
}

}  // namespace

bool ReadTokenizerData(const gufo::core::GgufReader& reader, TokenizerData* out,
                       std::string* error) {
  const auto fail = [&](std::string_view m) {
    if (error != nullptr)
      *error = std::string(m);
    return false;
  };
  const auto str = [&](std::string_view key, std::string* v) {
    const auto* meta = reader.FindMetadata(key);
    if (meta == nullptr)
      return true;
    const auto* s = std::get_if<std::string_view>(&meta->value);
    if (s == nullptr)
      return false;
    v->assign(*s);
    return true;
  };
  const auto strings = [&](std::string_view key, std::vector<std::string>* v) {
    const auto* meta = reader.FindMetadata(key);
    if (meta == nullptr)
      return true;
    const auto* s = std::get_if<std::vector<std::string_view>>(&meta->value);
    if (s == nullptr)
      return false;
    v->assign(s->begin(), s->end());
    return true;
  };
  if (!str("tokenizer.ggml.model", &out->model))
    return fail("bad metadata value for tokenizer.ggml.model");
  if (!str("tokenizer.ggml.pre", &out->pre))
    return fail("bad metadata value for tokenizer.ggml.pre");
  if (!str("tokenizer.chat_template", &out->chat_template))
    return fail("bad metadata value for tokenizer.chat_template");
  if (!strings("tokenizer.ggml.tokens", &out->tokens))
    return fail("bad metadata value for tokenizer.ggml.tokens");
  if (!strings("tokenizer.ggml.merges", &out->merges))
    return fail("bad metadata value for tokenizer.ggml.merges");
  if (const auto* meta = reader.FindMetadata("tokenizer.ggml.token_type")) {
    const auto* v = std::get_if<std::vector<std::int64_t>>(&meta->value);
    if (meta->type != gufo::core::GgufValueType::kArray || v == nullptr)
      return fail("bad metadata value for tokenizer.ggml.token_type");
    out->token_types.clear();
    out->token_types.reserve(v->size());
    for (const auto t : *v) {
      if (t < std::numeric_limits<std::int32_t>::min() ||
          t > std::numeric_limits<std::int32_t>::max())
        return fail("bad metadata value for tokenizer.ggml.token_type");
      out->token_types.push_back(static_cast<std::int32_t>(t));
    }
  }
  if (!ReadId(reader, "tokenizer.ggml.eos_token_id", &out->eos_id))
    return fail("bad metadata value for tokenizer.ggml.eos_token_id");
  if (!ReadId(reader, "tokenizer.ggml.eot_token_id", &out->eot_id))
    return fail("bad metadata value for tokenizer.ggml.eot_token_id");
  if (!ReadId(reader, "tokenizer.ggml.bos_token_id", &out->bos_id))
    return fail("bad metadata value for tokenizer.ggml.bos_token_id");
  if (!ReadId(reader, "tokenizer.ggml.padding_token_id", &out->pad_id))
    return fail("bad metadata value for tokenizer.ggml.padding_token_id");
  out->add_bos =
      reader.GetMetadataBool("tokenizer.ggml.add_bos_token").value_or(false);
  out->add_eos =
      reader.GetMetadataBool("tokenizer.ggml.add_eos_token").value_or(false);
  if (out->tokens.empty())
    return fail("no tokenizer.ggml.tokens");
  return true;
}

bool ReadTokenizerDataFromGguf(const std::string& path, TokenizerData* out,
                               std::string* error) {
  std::string why;
  // Unsized tensors (NVFP4 experts) do not matter for the metadata.
  const auto reader = gufo::core::GgufReader::OpenFile(
      path, &why, {.allow_unsized_tensors = true});
  if (reader == nullptr) {
    if (error != nullptr)
      *error = why + ": " + path;
    return false;
  }
  if (!ReadTokenizerData(*reader, out, error)) {
    if (error != nullptr)
      *error += ": " + path;
    return false;
  }
  return true;
}

}  // namespace inkling2::text
