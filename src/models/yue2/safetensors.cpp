#include "src/models/yue2/safetensors.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <gufo/ext/json.hpp>
#include <stdexcept>

namespace gufo::models::yue2 {

std::unique_ptr<SafetensorsFile> SafetensorsFile::Open(
    const std::filesystem::path& path, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = path.string() + ": " + message;
    return nullptr;
  };
  auto result = std::unique_ptr<SafetensorsFile>(new SafetensorsFile());
  std::string open_error;
  result->file_ = platform::ReadOnlyMappedFile::Open(path, &open_error);
  if (!result->file_.valid())
    return fail(open_error.empty() ? "cannot map file" : open_error);
  const std::uint8_t* data = result->file_.data();
  const std::size_t size = result->file_.size();
  if (size < 8)
    return fail("not a safetensors file");
  std::uint64_t header = 0;
  std::memcpy(&header, data, sizeof(header));
  if (header == 0 || header > size - 8 || header > (100ULL << 20))
    return fail("invalid safetensors header length");
  result->payload_offset_ = 8 + static_cast<std::size_t>(header);
  const std::size_t payload = size - result->payload_offset_;
  try {
    const json::Value root =
        json::parse(std::string_view(reinterpret_cast<const char*>(data + 8),
                                     static_cast<std::size_t>(header)));
    if (!root.is_object())
      return fail("safetensors header is not an object");
    for (const auto& [name, entry] : root.members()) {
      if (name == "__metadata__")
        continue;
      const json::Value* offsets = entry.find("data_offsets");
      const json::Value* shape = entry.find("shape");
      if (offsets == nullptr || !offsets->is_array() || offsets->size() != 2 ||
          shape == nullptr || !shape->is_array())
        return fail("malformed tensor entry " + name);
      TensorInfo info;
      info.dtype = entry.member_str("dtype");
      const std::uint64_t begin = offsets->items()[0].as_size(0);
      const std::uint64_t end = offsets->items()[1].as_size(0);
      if (end < begin || end > payload)
        return fail("tensor " + name + " is outside the payload");
      info.offset = begin;
      info.bytes = end - begin;
      for (const auto& dimension : shape->items())
        info.shape.push_back(dimension.as_size(0));
      result->tensors_.emplace(name, std::move(info));
    }
  } catch (const std::exception& exception) {
    return fail(exception.what());
  }
  return result;
}

const TensorInfo* SafetensorsFile::Find(std::string_view name) const {
  const auto found = tensors_.find(name);
  return found == tensors_.end() ? nullptr : &found->second;
}

const TensorInfo& SafetensorsFile::Require(
    std::string_view name, std::string_view dtype,
    std::span<const std::uint64_t> shape) const {
  const TensorInfo* tensor = Find(name);
  if (tensor == nullptr)
    throw std::runtime_error("missing tensor " + std::string(name));
  if (tensor->dtype != dtype || !std::ranges::equal(tensor->shape, shape))
    throw std::runtime_error("unexpected dtype or shape for tensor " +
                             std::string(name));
  return *tensor;
}

std::vector<float> SafetensorsFile::ReadFloat(const TensorInfo& tensor) const {
  const std::uint8_t* source = payload() + tensor.offset;
  std::vector<float> values;
  if (tensor.dtype == "F32") {
    values.resize(tensor.bytes / sizeof(float));
    std::memcpy(values.data(), source, values.size() * sizeof(float));
  } else if (tensor.dtype == "BF16") {
    values.resize(tensor.bytes / 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, source + 2 * i, 2);
      values[i] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }
  } else {
    throw std::runtime_error("unsupported tensor dtype " + tensor.dtype);
  }
  return values;
}

}  // namespace gufo::models::yue2
