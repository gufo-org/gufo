#include "src/models/qwen/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace strix::tokenization {

namespace {

std::string FormatHexByteToken(std::uint8_t byte_val) {
  char buf[8];
  static_cast<void>(std::snprintf(buf, sizeof(buf), "<0x%02X>", byte_val));
  return {buf};
}

constexpr int HexCharToInt(char c) noexcept {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  return -1;
}

std::string ByteToGpt2Utf8(std::uint8_t b) {
  static const auto b2u_map = []() {
    std::array<std::string, 256> table;
    for (int i = '!'; i <= '~'; ++i) {
      table[i] = std::string(1, static_cast<char>(i));
    }
    for (int i = 161; i <= 172; ++i) {
      char buf[3] = {static_cast<char>(0xC0 | (i >> 6)),
                     static_cast<char>(0x80 | (i & 0x3F)), 0};
      table[i] = buf;
    }
    for (int i = 174; i <= 255; ++i) {
      char buf[3] = {static_cast<char>(0xC0 | (i >> 6)),
                     static_cast<char>(0x80 | (i & 0x3F)), 0};
      table[i] = buf;
    }
    int n = 0;
    for (int i = 0; i < 256; ++i) {
      if ((i < '!' || i > '~') && (i < 161 || i > 172) &&
          (i < 174 || i > 255)) {
        const int cp = 256 + n;
        char buf[3] = {static_cast<char>(0xC0 | (cp >> 6)),
                       static_cast<char>(0x80 | (cp & 0x3F)), 0};
        table[i] = buf;
        ++n;
      }
    }
    return table;
  }();
  return b2u_map[b];
}

std::string UnescapeGpt2Bytes(std::string_view text) {
  static const auto u2b_map = []() {
    std::unordered_map<char32_t, std::uint8_t> m;
    for (int b = '!'; b <= '~'; ++b) {
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    }
    for (int b = 161; b <= 172; ++b) {
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    }
    for (int b = 174; b <= 255; ++b) {
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if ((b < '!' || b > '~') && (b < 161 || b > 172) &&
          (b < 174 || b > 255)) {
        m[static_cast<char32_t>(256 + n)] = static_cast<std::uint8_t>(b);
        ++n;
      }
    }
    return m;
  }();

  std::string result;
  result.reserve(text.size());

  std::size_t i = 0;
  while (i < text.size()) {
    const auto b0 = static_cast<unsigned char>(text[i]);
    char32_t cp = b0;
    std::size_t len = 1;

    if ((b0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
      cp =
          ((b0 & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
      len = 2;
    } else if ((b0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
      cp = ((b0 & 0x0F) << 12) |
           ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
           (static_cast<unsigned char>(text[i + 2]) & 0x3F);
      len = 3;
    } else if ((b0 & 0xF8) == 0xF0 && i + 3 < text.size()) {
      cp = ((b0 & 0x07) << 18) |
           ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
           ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
           (static_cast<unsigned char>(text[i + 3]) & 0x3F);
      len = 4;
    }

    auto it = u2b_map.find(cp);
    if (it != u2b_map.end()) {
      result.push_back(static_cast<char>(it->second));
    } else {
      result.append(text.substr(i, len));
    }
    i += len;
  }

  return result;
}

}  // namespace

std::unique_ptr<QwenTokenizer> QwenTokenizer::CreateFromBinaryFile(
    const std::string& path, std::string* error_msg) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    if (error_msg != nullptr) {
      *error_msg = "Could not open binary vocab file: " + path;
    }
    return nullptr;
  }

  std::uint32_t num_tokens = 0;
  f.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
  if (!f || num_tokens == 0) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid binary vocab format";
    }
    return nullptr;
  }

  auto tokenizer = std::unique_ptr<QwenTokenizer>(new QwenTokenizer());
  tokenizer->id_to_token_.reserve(num_tokens);
  tokenizer->token_to_id_.reserve(num_tokens);

  for (std::uint32_t i = 0; i < num_tokens; ++i) {
    std::uint16_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string str(len, '\0');
    f.read(str.data(), len);
    tokenizer->id_to_token_.push_back(str);
    tokenizer->token_to_id_[str] = i;
  }

  std::uint32_t num_merges = 0;
  f.read(reinterpret_cast<char*>(&num_merges), sizeof(num_merges));
  tokenizer->merge_ranks_.reserve(num_merges);
  for (std::uint32_t rank = 0; rank < num_merges; ++rank) {
    std::uint16_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string merge(len, '\0');
    f.read(merge.data(), len);

    const auto space_pos = merge.find(' ');
    if (space_pos == std::string::npos) {
      continue;
    }
    const std::string part1 = merge.substr(0, space_pos);
    const std::string part2 = merge.substr(space_pos + 1);

    auto it1 = tokenizer->token_to_id_.find(part1);
    auto it2 = tokenizer->token_to_id_.find(part2);
    if (it1 != tokenizer->token_to_id_.end() &&
        it2 != tokenizer->token_to_id_.end()) {
      tokenizer->merge_ranks_[{it1->second, it2->second}] = rank;
    }
  }

  std::uint32_t num_specials = 0;
  f.read(reinterpret_cast<char*>(&num_specials), sizeof(num_specials));
  for (std::uint32_t i = 0; i < num_specials; ++i) {
    std::uint32_t id = 0;
    std::uint16_t len = 0;
    f.read(reinterpret_cast<char*>(&id), sizeof(id));
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string name(len, '\0');
    f.read(name.data(), len);

    tokenizer->special_token_to_id_[name] = id;
    tokenizer->is_special_token_[id] = true;
    if (id < tokenizer->id_to_token_.size()) {
      tokenizer->id_to_token_[id] = name;
      tokenizer->token_to_id_[name] = id;
    }
  }

  tokenizer->InitializeByteTokens();
  return tokenizer;
}

std::unique_ptr<QwenTokenizer> QwenTokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto token_views =
      reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  if (token_views.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF metadata missing 'tokenizer.ggml.tokens'";
    }
    return nullptr;
  }

  std::vector<std::string> tokens;
  tokens.reserve(token_views.size());
  for (const auto& tv : token_views) {
    tokens.emplace_back(tv);
  }

  const auto merge_views =
      reader.GetMetadataStringArray("tokenizer.ggml.merges");
  std::vector<std::string> merges;
  merges.reserve(merge_views.size());
  for (const auto& mv : merge_views) {
    merges.emplace_back(mv);
  }

  std::unordered_map<std::string, TokenId> special_tokens;
  // Dynamic lookup of special tokens from GGUF vocabulary table
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto& t = tokens[i];
    if (t == "<|endoftext|>" || t == "<|im_start|>" || t == "<|im_end|>" ||
        t == "<think>" || t == "</think>" || t == "<|object_ref_start|>" ||
        t == "<|object_ref_end|>" || t == "<|vision_start|>" ||
        t == "<|vision_end|>" || t == "<|image_pad|>" || t == "<|video_pad|>" ||
        t == "<|quad_start|>" || t == "<|quad_end|>" ||
        (t.size() >= 4 && t.starts_with("<|") && t.ends_with("|>"))) {
      special_tokens[t] = static_cast<TokenId>(i);
    }
  }

  auto tokenizer =
      CreateFromVocabulary(tokens, merges, special_tokens, error_msg);
  if (tokenizer == nullptr) {
    return nullptr;
  }

  if (auto eos = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")) {
    tokenizer->eos_token_id_ = *eos;
  } else if (auto it = special_tokens.find("<|im_end|>");
             it != special_tokens.end()) {
    tokenizer->eos_token_id_ = it->second;
  } else if (auto it = special_tokens.find("<|endoftext|>");
             it != special_tokens.end()) {
    tokenizer->eos_token_id_ = it->second;
  }

  if (auto bos = reader.GetMetadataUint32("tokenizer.ggml.bos_token_id")) {
    tokenizer->bos_token_id_ = *bos;
  } else if (auto it = special_tokens.find("<|im_start|>");
             it != special_tokens.end()) {
    tokenizer->bos_token_id_ = it->second;
  }

  if (auto pad = reader.GetMetadataUint32("tokenizer.ggml.padding_token_id")) {
    tokenizer->pad_token_id_ = *pad;
  } else if (auto it = special_tokens.find("<|endoftext|>");
             it != special_tokens.end()) {
    tokenizer->pad_token_id_ = it->second;
  }

  return tokenizer;
}

std::unique_ptr<QwenTokenizer> QwenTokenizer::CreateFromVocabulary(
    std::span<const std::string> tokens, std::span<const std::string> merges,
    const std::unordered_map<std::string, TokenId>& special_tokens,
    std::string* error_msg) {
  if (tokens.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Vocabulary cannot be empty";
    }
    return nullptr;
  }

  auto tokenizer = std::unique_ptr<QwenTokenizer>(new QwenTokenizer());
  tokenizer->id_to_token_.reserve(tokens.size());
  tokenizer->token_to_id_.reserve(tokens.size());

  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokenizer->id_to_token_.push_back(tokens[i]);
    tokenizer->token_to_id_[tokens[i]] = static_cast<TokenId>(i);
  }

  // Parse BPE merges
  tokenizer->merge_ranks_.reserve(merges.size());
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const auto& merge = merges[rank];
    const auto space_pos = merge.find(' ');
    if (space_pos == std::string::npos) {
      continue;
    }
    const std::string part1 = merge.substr(0, space_pos);
    const std::string part2 = merge.substr(space_pos + 1);

    auto it1 = tokenizer->token_to_id_.find(part1);
    auto it2 = tokenizer->token_to_id_.find(part2);
    if (it1 != tokenizer->token_to_id_.end() &&
        it2 != tokenizer->token_to_id_.end()) {
      tokenizer->merge_ranks_[{it1->second, it2->second}] = rank;
    }
  }

  // Register special tokens
  for (const auto& [name, id] : special_tokens) {
    tokenizer->special_token_to_id_[name] = id;
    tokenizer->is_special_token_[id] = true;
    if (id < tokenizer->id_to_token_.size()) {
      tokenizer->id_to_token_[id] = name;
      tokenizer->token_to_id_[name] = id;
    }
  }

  tokenizer->InitializeByteTokens();
  return tokenizer;
}

void QwenTokenizer::InitializeByteTokens() {
  id_to_decoded_token_.resize(id_to_token_.size());
  for (std::size_t i = 0; i < id_to_token_.size(); ++i) {
    if (is_special_token_.contains(static_cast<TokenId>(i))) {
      id_to_decoded_token_[i] = id_to_token_[i];
    } else {
      const auto& tok = id_to_token_[i];
      if (tok.size() == 6 && tok.starts_with("<0x") && tok.ends_with('>')) {
        const int h1 = HexCharToInt(tok[3]);
        const int h2 = HexCharToInt(tok[4]);
        if (h1 >= 0 && h2 >= 0) {
          const auto byte_val = static_cast<std::uint8_t>((h1 << 4) | h2);
          id_to_decoded_token_[i] = std::string(1, static_cast<char>(byte_val));
          continue;
        }
      }
      id_to_decoded_token_[i] = UnescapeGpt2Bytes(tok);
    }
  }

  for (std::size_t b = 0; b < 256; ++b) {
    const auto byte_val = static_cast<std::uint8_t>(b);
    const std::string gpt2_utf8 = ByteToGpt2Utf8(byte_val);
    auto it = token_to_id_.find(gpt2_utf8);
    if (it != token_to_id_.end()) {
      byte_tokens_[b] = it->second;
    } else {
      const std::string direct_char(1, static_cast<char>(byte_val));
      auto direct_it = token_to_id_.find(direct_char);
      if (direct_it != token_to_id_.end()) {
        byte_tokens_[b] = direct_it->second;
      } else {
        const std::string hex_token = FormatHexByteToken(byte_val);
        auto hex_it = token_to_id_.find(hex_token);
        if (hex_it != token_to_id_.end()) {
          byte_tokens_[b] = hex_it->second;
        } else {
          byte_tokens_[b] = kInvalidTokenId;
        }
      }
    }
  }
}

std::vector<TokenId> QwenTokenizer::BpeMergeChunk(
    std::string_view chunk) const {
  if (chunk.empty()) {
    return {};
  }

  // Initial tokenization at byte level
  std::vector<TokenId> word_tokens;
  word_tokens.reserve(chunk.size());

  for (const unsigned char c : chunk) {
    const TokenId tid = byte_tokens_[c];
    if (tid != kInvalidTokenId) {
      word_tokens.push_back(tid);
    }
  }

  if (word_tokens.size() <= 1) {
    return word_tokens;
  }

  // Iteratively merge the highest-ranked adjacent pairs
  while (word_tokens.size() >= 2) {
    std::optional<std::uint32_t> best_rank;
    std::size_t best_idx = 0;

    for (std::size_t i = 0; i < word_tokens.size() - 1; ++i) {
      auto it = merge_ranks_.find({word_tokens[i], word_tokens[i + 1]});
      if (it != merge_ranks_.end()) {
        if (!best_rank.has_value() || it->second < *best_rank) {
          best_rank = it->second;
          best_idx = i;
        }
      }
    }

    if (!best_rank.has_value()) {
      break;
    }

    const std::string merged_str = id_to_token_[word_tokens[best_idx]] +
                                   id_to_token_[word_tokens[best_idx + 1]];
    auto merged_it = token_to_id_.find(merged_str);
    if (merged_it == token_to_id_.end()) {
      break;
    }

    word_tokens[best_idx] = merged_it->second;
    word_tokens.erase(word_tokens.begin() +
                      static_cast<std::ptrdiff_t>(best_idx + 1));
  }

  return word_tokens;
}

std::vector<TokenId> QwenTokenizer::Encode(
    std::string_view text, const TokenizerOptions& options) const {
  std::vector<TokenId> tokens;
  if (text.empty()) {
    if (options.add_bos && bos_token_id_ != kInvalidTokenId) {
      tokens.push_back(bos_token_id_);
    }
    if (options.add_eos && eos_token_id_ != kInvalidTokenId) {
      tokens.push_back(eos_token_id_);
    }
    return tokens;
  }

  if (options.add_bos && bos_token_id_ != kInvalidTokenId) {
    tokens.push_back(bos_token_id_);
  }

  if (!options.parse_special_tokens || special_token_to_id_.empty()) {
    const auto chunk_tokens = BpeMergeChunk(text);
    tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
  } else {
    // Scan text for special token delimiters
    std::size_t pos = 0;
    while (pos < text.size()) {
      std::size_t next_special_pos = std::string_view::npos;
      std::string_view matched_special;
      TokenId matched_id = kInvalidTokenId;

      for (const auto& [special_str, id] : special_token_to_id_) {
        const auto found = text.find(special_str, pos);
        if (found != std::string_view::npos) {
          if (next_special_pos == std::string_view::npos ||
              found < next_special_pos) {
            next_special_pos = found;
            matched_special = special_str;
            matched_id = id;
          }
        }
      }

      if (next_special_pos == std::string_view::npos) {
        // No more special tokens; encode remainder
        const auto chunk = text.substr(pos);
        const auto chunk_tokens = BpeMergeChunk(chunk);
        tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
        break;
      }

      if (next_special_pos > pos) {
        const auto chunk = text.substr(pos, next_special_pos - pos);
        const auto chunk_tokens = BpeMergeChunk(chunk);
        tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
      }

      tokens.push_back(matched_id);
      pos = next_special_pos + matched_special.size();
    }
  }

  if (options.add_eos && eos_token_id_ != kInvalidTokenId) {
    tokens.push_back(eos_token_id_);
  }

  return tokens;
}

std::string QwenTokenizer::Decode(std::span<const TokenId> tokens) const {
  std::string result;
  for (const TokenId id : tokens) {
    result.append(DecodeToken(id));
  }
  return result;
}

std::string_view QwenTokenizer::DecodeToken(TokenId token_id) const noexcept {
  if (token_id < id_to_decoded_token_.size()) {
    return id_to_decoded_token_[token_id];
  }
  if (token_id < id_to_token_.size()) {
    return id_to_token_[token_id];
  }
  return "";
}

std::optional<TokenId> QwenTokenizer::FindSpecialToken(
    std::string_view token_str) const noexcept {
  auto it = special_token_to_id_.find(std::string(token_str));
  if (it != special_token_to_id_.end()) {
    return it->second;
  }
  return std::nullopt;
}

bool QwenTokenizer::IsSpecialToken(TokenId id) const noexcept {
  return is_special_token_.contains(id);
}

}  // namespace strix::tokenization
