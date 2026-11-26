// Frozen copies of the six safetensors header parsers that the Rust reader
// (src/core/safetensors) replaced, as of 0.35.0, for the differential test.
// Each reads a whole file held in memory instead of a mapping or file
// descriptor and returns the tensors it accepted or its error. Only the
// header checks are kept: what each loader does with a tensor afterwards
// (LoRA key rules, tensor stores, index files) is unchanged by the port.
#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::safetensors::reference {

struct Tensor {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t begin{0};
  std::uint64_t end{0};
};

struct Result {
  bool ok{false};
  std::string error;
  std::vector<Tensor> tensors;
  /// LoRA only: `lora_adapter_metadata` as the loader read it.
  std::optional<std::string> adapter_metadata;
};

inline Result Failed(std::string error) {
  Result result;
  result.error = std::move(error);
  return result;
}

inline std::uint64_t LengthPrefix(std::span<const std::byte> file) {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index)
    value |=
        static_cast<std::uint64_t>(std::to_integer<unsigned char>(file[index]))
        << (index * 8U);
  return value;
}

inline std::string_view Text(std::span<const std::byte> file,
                             std::uint64_t header_size) {
  return {reinterpret_cast<const char*>(file.data()) + 8,
          static_cast<std::size_t>(header_size)};
}

// src/core/lora/adapter.cpp, Adapter::Adapter.
namespace lora {

inline std::optional<std::size_t> Integer(const json::Value& value) {
  const double number = value.as_double(-1);
  if (!value.is_number() || !std::isfinite(number) || number < 0 ||
      number > (1ULL << 40) ||
      number != static_cast<double>(static_cast<std::uint64_t>(number)))
    return std::nullopt;
  return static_cast<std::size_t>(number);
}

inline Result Parse(std::span<const std::byte> file) {
  const auto size = file.size();
  if (size < 8)
    return Failed("invalid safetensors file");
  const std::uint64_t header_size = LengthPrefix(file);
  if (!header_size || header_size > (32U << 20U) || header_size > size - 8)
    return Failed("invalid safetensors header");
  json::Value header;
  try {
    header = json::parse(Text(file, header_size));
  } catch (const std::exception&) {
    return Failed("invalid safetensors header");
  }
  if (!header.is_object())
    return Failed("invalid safetensors header");
  const auto payload_size = size - 8 - header_size;
  Result result;
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  const json::Value* metadata = nullptr;
  for (const auto& [key, entry] : header.members()) {
    if (key == "__metadata__") {
      metadata = &entry;
      continue;
    }
    const auto dtype = entry.member_str("dtype");
    std::size_t bytes = 2;
    if (dtype == "F32")
      bytes = 4;
    else if (dtype != "BF16" && dtype != "F16")
      return Failed(key + ": unsupported adapter dtype " + dtype);
    const auto* shape = entry.find("shape");
    const auto* offsets = entry.find("data_offsets");
    if (!shape || !shape->is_array() || shape->size() > 5 || !offsets ||
        !offsets->is_array() || offsets->size() != 2)
      return Failed(key + ": invalid tensor layout");
    Tensor tensor{.name = key, .dtype = dtype, .shape = {}};
    for (const auto& dim : shape->items()) {
      const auto value = Integer(dim).value_or(0);
      if (!value || value > std::numeric_limits<int>::max() ||
          bytes > payload_size / value)
        return Failed(key + ": invalid tensor shape");
      tensor.shape.push_back(value);
      bytes *= value;
    }
    const auto begin = Integer(offsets->items()[0]);
    const auto end = Integer(offsets->items()[1]);
    if (!begin || !end || *begin > *end || *end > payload_size ||
        *end - *begin != bytes)
      return Failed(key + ": invalid tensor extent");
    tensor.begin = *begin;
    tensor.end = *end;
    ranges.emplace_back(*begin, *end);
    result.tensors.push_back(std::move(tensor));
  }
  std::sort(ranges.begin(), ranges.end());
  std::size_t cursor = 0;
  for (auto [begin, end] : ranges) {
    if (begin != cursor)
      return Failed("noncontiguous safetensors payload");
    cursor = end;
  }
  if (cursor != payload_size)
    return Failed("unexpected safetensors data");
  if (metadata && metadata->is_object())
    if (const auto* text = metadata->find("lora_adapter_metadata"))
      result.adapter_metadata = text->str();
  // Every tensor becomes part of a module, or the loader rejects it.
  if (result.tensors.empty())
    return Failed("no adapter tensors");
  result.ok = true;
  return result;
}

}  // namespace lora

// The element counting of the EmbeddingGemma 2, Qwen3-ASR and Qwen3-TTS
// Tensor::NumElements.
inline std::optional<std::size_t> NumElements(
    const std::vector<std::uint64_t>& shape) {
  std::size_t elements = 1;
  for (const std::uint64_t dimension : shape) {
    if (dimension > std::numeric_limits<std::size_t>::max() ||
        (dimension != 0 &&
         elements > std::numeric_limits<std::size_t>::max() /
                        static_cast<std::size_t>(dimension))) {
      return std::nullopt;
    }
    elements *= static_cast<std::size_t>(dimension);
  }
  return elements;
}

// src/models/embedding_gemma2/loader.cpp, LoadSafetensors, with both
// towers loaded.
namespace embedding_gemma2 {

inline std::size_t DTypeSize(std::string_view name) {
  if (name == "F32")
    return 4;
  if (name == "BF16" || name == "F16")
    return 2;
  return 0;
}

inline Result Parse(std::span<const std::byte> file) {
  if (file.size() < 8U)
    return Failed("cannot map");
  const std::uint64_t header_size = LengthPrefix(file);
  if (header_size == 0 || header_size > file.size() - 8U)
    return Failed("invalid safetensors header");
  json::Value header;
  try {
    header = json::parse(Text(file, header_size));
  } catch (const std::exception&) {
    return Failed("unparsable safetensors header");
  }
  if (!header.is_object())
    return Failed("invalid safetensors header");
  const std::uint64_t payload_offset = 8U + header_size;
  Result result;
  for (const auto& [name, specification] : header.members()) {
    if (name == "__metadata__" || !(name.starts_with("language_model.") ||
                                    name.starts_with("vision_tower.") ||
                                    name.starts_with("embed_vision.") ||
                                    name.starts_with("audio_tower.") ||
                                    name.starts_with("embed_audio."))) {
      continue;
    }
    if (!specification.is_object())
      return Failed("invalid tensor entry " + name);
    Tensor tensor{
        .name = name, .dtype = specification.member_str("dtype"), .shape = {}};
    if (const json::Value* shape = specification.find("shape");
        shape != nullptr && shape->is_array()) {
      for (const auto& dimension : shape->items())
        tensor.shape.push_back(dimension.as_size());
    }
    const json::Value* offsets = specification.find("data_offsets");
    if (offsets == nullptr || !offsets->is_array() || offsets->size() != 2)
      return Failed("missing data_offsets for " + name);
    const std::uint64_t begin = offsets->items()[0].as_size();
    const std::uint64_t end = offsets->items()[1].as_size();
    const auto elements = NumElements(tensor.shape);
    const std::size_t element_size = DTypeSize(tensor.dtype);
    if (element_size == 0)
      return Failed("unsupported dtype for " + name);
    if (!elements.has_value() || begin >= end ||
        end > file.size() - payload_offset ||
        *elements > std::numeric_limits<std::uint64_t>::max() / element_size ||
        end - begin != static_cast<std::uint64_t>(*elements) * element_size)
      return Failed("inconsistent tensor extent for " + name);
    tensor.begin = begin;
    tensor.end = end;
    result.tensors.push_back(std::move(tensor));
  }
  result.ok = true;
  return result;
}

}  // namespace embedding_gemma2

// src/models/qwen3_asr/loader.cpp and src/models/qwen3_tts/loader.cpp,
// which differ only in the type names: Qwen3-TTS also takes lower case.
namespace qwen3 {

inline std::string_view CanonicalType(std::string_view name, bool lower_case) {
  for (std::string_view type : {"F32", "BF16", "F16", "I64", "I32", "U8"}) {
    if (name == type)
      return type;
    if (lower_case && name.size() == type.size() &&
        std::equal(name.begin(), name.end(), type.begin(), [](char a, char b) {
          return a == static_cast<char>(b - 'A' + 'a') || (b < 'A' && a == b);
        }))
      return type;
  }
  return {};
}

inline std::size_t DTypeSize(std::string_view type) {
  if (type == "F32" || type == "I32")
    return 4;
  if (type == "BF16" || type == "F16")
    return 2;
  if (type == "I64")
    return 8;
  if (type == "U8")
    return 1;
  return 0;
}

inline Result Parse(std::span<const std::byte> file, bool lower_case) {
  if (file.size() < 8U)
    return Failed("cannot open");
  const std::uint64_t header_size = LengthPrefix(file);
  if (header_size == 0 || header_size > file.size() - 8U)
    return Failed("invalid header length");
  json::Value header;
  try {
    header = json::parse(Text(file, header_size));
  } catch (const std::exception&) {
    return Failed("unparsable header");
  }
  if (!header.is_object())
    return Failed("header is not an object");
  const std::uint64_t payload_offset = 8U + header_size;
  Result result;
  for (const auto& [name, specification] : header.members()) {
    if (name == "__metadata__" || !specification.is_object())
      continue;
    // Tensors keep the canonical spelling, which the stores record.
    Tensor tensor{.name = name,
                  .dtype = std::string(CanonicalType(
                      specification.member_str("dtype"), lower_case)),
                  .shape = {}};
    if (const json::Value* shape = specification.find("shape");
        shape != nullptr && shape->is_array()) {
      for (const auto& dimension : shape->items())
        tensor.shape.push_back(dimension.as_size());
    }
    const json::Value* offsets = specification.find("data_offsets");
    if (offsets == nullptr || !offsets->is_array() || offsets->size() != 2)
      continue;
    const std::uint64_t begin = offsets->items()[0].as_size();
    const std::uint64_t end = offsets->items()[1].as_size();
    const auto elements = NumElements(tensor.shape);
    const std::size_t element_size = DTypeSize(tensor.dtype);
    if (!elements.has_value() || element_size == 0 || begin >= end ||
        end > file.size() - payload_offset ||
        *elements > std::numeric_limits<std::uint64_t>::max() / element_size ||
        end - begin != static_cast<std::uint64_t>(*elements) * element_size)
      continue;
    tensor.begin = begin;
    tensor.end = end;
    result.tensors.push_back(std::move(tensor));
  }
  if (result.tensors.empty())
    return Failed("no usable tensor");
  result.ok = true;
  return result;
}

}  // namespace qwen3

// src/models/qwen_image_21/weights.cpp, Weights::Load.
namespace qwen_image {

inline std::size_t Integer(const json::Value& value) {
  const double number = value.as_double(-1);
  if (!value.is_number() || !std::isfinite(number) || number < 0 ||
      number > (1ULL << 40) ||
      number != static_cast<double>(static_cast<std::uint64_t>(number)))
    throw std::runtime_error("invalid Qwen-Image integer metadata");
  return static_cast<std::size_t>(number);
}

inline Result Load(std::span<const std::byte> file) {
  const auto size = file.size();
  if (size < 8)
    return Failed("invalid safetensors file");
  const std::uint64_t header_size = LengthPrefix(file);
  if (!header_size || header_size > (32U << 20U) || header_size > size - 8)
    return Failed("invalid safetensors header");
  const auto header = json::parse(Text(file, header_size));
  if (!header.is_object())
    return Failed("invalid safetensors header");
  const auto payload_size = size - 8 - header_size;
  Result result;
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  for (const auto& [name, entry] : header.members()) {
    if (name == "__metadata__")
      continue;
    const auto dtype = entry.member_str("dtype");
    if (dtype != "BF16" && dtype != "F32")
      return Failed("unsupported Qwen-Image weight dtype " + dtype);
    const auto* shape = entry.find("shape");
    const auto* offsets = entry.find("data_offsets");
    if (!shape || !shape->is_array() || shape->size() > 5 || !offsets ||
        !offsets->is_array() || offsets->size() != 2)
      return Failed("invalid tensor layout " + name);
    Tensor tensor{.name = name, .dtype = dtype, .shape = {}};
    std::size_t bytes = dtype == "BF16" ? 2 : 4;
    for (const auto& dim : shape->items()) {
      const auto value = Integer(dim);
      if (!value || value > std::numeric_limits<int>::max() ||
          bytes > payload_size / value)
        return Failed("invalid tensor shape " + name);
      tensor.shape.push_back(value);
      bytes *= value;
    }
    const auto begin = Integer(offsets->items()[0]);
    const auto end = Integer(offsets->items()[1]);
    if (begin > end || end > payload_size || end - begin != bytes)
      return Failed("invalid tensor extent " + name);
    tensor.begin = begin;
    tensor.end = end;
    ranges.emplace_back(begin, end);
    result.tensors.push_back(std::move(tensor));
  }
  std::sort(ranges.begin(), ranges.end());
  std::size_t cursor = 0;
  for (auto [begin, end] : ranges) {
    if (begin != cursor)
      return Failed("noncontiguous safetensors payload");
    cursor = end;
  }
  if (cursor != payload_size)
    return Failed("unexpected safetensors data");
  result.ok = true;
  return result;
}

// The loader let the JSON and integer errors propagate.
inline Result Parse(std::span<const std::byte> file) {
  try {
    return Load(file);
  } catch (const std::runtime_error& error) {
    return Failed(error.what());
  }
}

}  // namespace qwen_image

// src/models/minimax_h3/runtime.cpp, ParseSafetensorsHeader. MiniMax H3
// parses with its own JSON parser (src/models/minimax_h3/json.cpp), which a
// core test may not link; its number rules are rebuilt here on the core
// parser from the tokens: integers are exact and fail the whole parse when
// out of range, and AsUint64 takes only integers that are not negative.
namespace minimax {

struct Rejected {
  std::string message;
};

inline std::string Token(const json::Value& number) {
  return number.dump(json::Value::Numbers::kAsParsed);
}

inline bool IsInteger(std::string_view token) {
  return token.find_first_of(".eE") == std::string_view::npos;
}

// MiniMax's parser rejects an integer outside int64 (negative) or uint64.
inline void CheckIntegers(const json::Value& value) {
  if (value.is_number()) {
    const auto token = Token(value);
    if (!IsInteger(token))
      return;
    const auto* first = token.data();
    const auto* last = token.data() + token.size();
    bool fits;
    if (token.starts_with('-')) {
      std::int64_t parsed = 0;
      const auto result = std::from_chars(first, last, parsed);
      fits = result.ec == std::errc{} && result.ptr == last;
    } else {
      std::uint64_t parsed = 0;
      const auto result = std::from_chars(first, last, parsed);
      fits = result.ec == std::errc{} && result.ptr == last;
    }
    if (!fits)
      throw Rejected{"integer is out of range"};
  } else if (value.is_array()) {
    for (const auto& item : value.items())
      CheckIntegers(item);
  } else if (value.is_object()) {
    for (const auto& [key, member] : value.members())
      CheckIntegers(member);
  }
}

inline std::uint64_t AsUint64(const json::Value& value) {
  if (!value.is_number())
    throw Rejected{"expected an unsigned integer"};
  const auto token = Token(value);
  if (!IsInteger(token))
    throw Rejected{"expected an unsigned integer"};
  if (token.starts_with('-')) {
    // An int64 that is not negative: only -0.
    if (token != "-0")
      throw Rejected{"expected an unsigned integer"};
    return 0;
  }
  std::uint64_t parsed = 0;
  std::from_chars(token.data(), token.data() + token.size(), parsed);
  return parsed;
}

inline const json::Value& RequireField(const json::Value& object,
                                       std::string_view key) {
  const json::Value* value =
      object.is_object() ? object.find(std::string(key)) : nullptr;
  if (value == nullptr)
    throw Rejected{"missing required field " + std::string(key)};
  return *value;
}

inline std::uint64_t CheckedMultiply(std::uint64_t left, std::uint64_t right) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    throw Rejected{"overflows uint64"};
  return left * right;
}

inline std::uint64_t DTypeSize(std::string_view value) {
  if (value == "BF16")
    return 2;
  if (value == "F32" || value == "I32")
    return 4;
  throw Rejected{"unsupported H3 tensor dtype " + std::string(value)};
}

inline Result Load(std::span<const std::byte> file) {
  const std::uint64_t file_size = file.size();
  if (file_size < 8)
    return Failed("invalid safetensors file");
  const std::uint64_t header_bytes = LengthPrefix(file);
  if (header_bytes == 0 || header_bytes > (64U << 20U) ||
      header_bytes > file_size - 8U)
    return Failed("invalid safetensors header length");
  json::Value root;
  try {
    root = json::parse(Text(file, header_bytes));
  } catch (const std::exception& error) {
    return Failed(error.what());
  }
  CheckIntegers(root);
  if (!root.is_object())
    throw Rejected{"expected an object"};
  const std::uint64_t payload_offset = 8U + header_bytes;
  Result result;
  std::map<std::string, Tensor, std::less<>> tensors;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> intervals;
  for (const auto& [name, value] : root.members()) {
    if (name == "__metadata__")
      continue;
    const auto& dtype_value = RequireField(value, "dtype");
    if (!dtype_value.is_string())
      throw Rejected{"expected a string"};
    const std::string dtype = dtype_value.str();
    const std::uint64_t element_bytes = DTypeSize(dtype);
    Tensor tensor{.name = name, .dtype = dtype, .shape = {}};
    std::uint64_t elements = 1;
    const auto& shape = RequireField(value, "shape");
    if (!shape.is_array())
      throw Rejected{"expected an array"};
    for (const auto& dimension : shape.items()) {
      const std::uint64_t parsed_dimension = AsUint64(dimension);
      elements = CheckedMultiply(elements, parsed_dimension);
      tensor.shape.push_back(parsed_dimension);
    }
    const auto& offsets = RequireField(value, "data_offsets");
    if (!offsets.is_array())
      throw Rejected{"expected an array"};
    if (offsets.size() != 2)
      return Failed("tensor " + name + " has invalid data_offsets");
    const std::uint64_t begin = AsUint64(offsets.items()[0]);
    const std::uint64_t end = AsUint64(offsets.items()[1]);
    const std::uint64_t expected_bytes =
        CheckedMultiply(elements, element_bytes);
    if (begin >= end || end - begin != expected_bytes ||
        end > file_size - payload_offset)
      return Failed("tensor " + name + " has invalid payload bounds");
    tensor.begin = begin;
    tensor.end = end;
    tensors.emplace(name, std::move(tensor));
    intervals.emplace_back(begin, end);
  }
  std::sort(intervals.begin(), intervals.end());
  for (std::size_t index = 1; index < intervals.size(); ++index) {
    if (intervals[index].first < intervals[index - 1].second)
      return Failed("overlapping safetensors payloads");
  }
  // The loader keeps its tensors in a map, ordered by name.
  for (auto& [name, tensor] : tensors)
    result.tensors.push_back(std::move(tensor));
  result.ok = true;
  return result;
}

inline Result Parse(std::span<const std::byte> file) {
  try {
    return Load(file);
  } catch (const Rejected& error) {
    return Failed(error.message);
  }
}

}  // namespace minimax

}  // namespace gufo::safetensors::reference
