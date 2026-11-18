#ifndef GUFO_TESTS_CORE_GGUF_HEADER_REFERENCE_HPP_
#define GUFO_TESTS_CORE_GGUF_HEADER_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::gguf_header_reference {

// What 0.35.0 made of a GGUF header: the version, alignment, metadata and
// tensors, pointing into the parsed bytes, or the message it failed with.
struct Result {
  bool opened = false;
  std::string error;
  std::uint32_t version = 0;
  std::uint64_t alignment = 32;
  std::unordered_map<std::string_view, core::GgufMetadataValue> metadata;
  std::vector<core::GgufTensorInfo> tensors;
};

// GgufReader::OpenMemory, and the parse of a mapped file in OpenFile.
Result OpenMemory(const void* data, std::size_t size,
                  core::GgufOpenOptions options);
Result Parse(const std::uint8_t* data, std::size_t size,
             core::GgufOpenOptions options);

}  // namespace gufo::gguf_header_reference

#endif  // GUFO_TESTS_CORE_GGUF_HEADER_REFERENCE_HPP_
