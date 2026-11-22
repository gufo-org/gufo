// In-memory GGUF fixtures for the Qwen LoRA tests: tensors carry raw bytes in
// any storage type, so adapters can hold F32, F16 or BF16 factors.
#ifndef GUFO_TESTS_FAMILIES_QWEN_LORA_FIXTURE_HPP_
#define GUFO_TESTS_FAMILIES_QWEN_LORA_FIXTURE_HPP_

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/families/qwen/lora/lora.hpp"
#include "tests/fixtures/gguf_builder.hpp"

namespace lora_fixture {

using gufo::core::GgmlType;

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  GgmlType type{GgmlType::kF32};
  std::vector<std::uint8_t> bytes;
  /// Placed at the previous tensor's offset, sharing its bytes.
  bool shares_previous{false};
};

template<typename T>
std::vector<std::uint8_t> Bytes(const std::vector<T>& values) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

/// A GGUF file with `tensors` after the metadata `builder` already holds.
inline std::vector<std::uint8_t> Build(gufo::test_fixture::GgufBuilder builder,
                                       const std::vector<Tensor>& tensors) {
  std::uint64_t offset = 0;
  std::vector<std::uint64_t> offsets;
  for (const auto& t : tensors) {
    if (t.shares_previous) {
      builder.AddTensor(t.name, t.dims, t.type, offsets.back());
      offsets.push_back(offsets.back());
      continue;
    }
    builder.AddTensor(t.name, t.dims, t.type, offset);
    offsets.push_back(offset);
    offset += (t.bytes.size() + 31) / 32 * 32;
  }
  auto bytes = builder.Build(offset);
  const std::size_t start = bytes.size() - offset;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].shares_previous) {
      continue;
    }
    std::memcpy(bytes.data() + start + offsets[i], tensors[i].bytes.data(),
                tensors[i].bytes.size());
  }
  return bytes;
}

/// A llama.cpp LoRA adapter for `architecture` with `tensors`.
inline std::vector<std::uint8_t> Adapter(const std::string& architecture,
                                         float alpha,
                                         const std::vector<Tensor>& tensors) {
  gufo::test_fixture::GgufBuilder builder;
  builder.AddMetadataString("general.architecture", architecture);
  builder.AddMetadataString("general.type", "adapter");
  builder.AddMetadataString("adapter.type", "lora");
  builder.AddMetadataFloat32("adapter.lora.alpha", alpha);
  return Build(std::move(builder), tensors);
}

inline std::unique_ptr<gufo::core::GgufReader> Read(
    const std::vector<std::uint8_t>& bytes) {
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  if (!reader) {
    throw std::runtime_error("fixture does not parse: " + error);
  }
  return reader;
}

}  // namespace lora_fixture

#endif  // GUFO_TESTS_FAMILIES_QWEN_LORA_FIXTURE_HPP_
