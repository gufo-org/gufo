#ifndef GUFO_MODELS_YUE2_SAFETENSORS_HPP_
#define GUFO_MODELS_YUE2_SAFETENSORS_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gufo/ext/platform/mapped_file.hpp>

namespace gufo::models::yue2 {

struct TensorInfo {
  std::string dtype;
  std::vector<std::uint64_t> shape;
  /// Byte offset inside the payload, which follows the header.
  std::uint64_t offset{0};
  std::uint64_t bytes{0};
};

/// A memory-mapped single-file safetensors checkpoint.
class SafetensorsFile {
public:
  [[nodiscard]] static std::unique_ptr<SafetensorsFile> Open(
      const std::filesystem::path& path, std::string* error);

  [[nodiscard]] const TensorInfo* Find(std::string_view name) const;
  /// The named tensor with this dtype and shape, or an exception.
  [[nodiscard]] const TensorInfo& Require(
      std::string_view name, std::string_view dtype,
      std::span<const std::uint64_t> shape) const;
  [[nodiscard]] const std::map<std::string, TensorInfo, std::less<>>& tensors()
      const {
    return tensors_;
  }
  [[nodiscard]] const std::uint8_t* payload() const {
    return file_.data() + payload_offset_;
  }
  [[nodiscard]] std::size_t payload_size() const {
    return file_.size() - payload_offset_;
  }
  [[nodiscard]] std::size_t payload_offset() const { return payload_offset_; }
  [[nodiscard]] const platform::ReadOnlyMappedFile& file() const {
    return file_;
  }
  /// FP32 copy of an F32 or BF16 tensor.
  [[nodiscard]] std::vector<float> ReadFloat(const TensorInfo& tensor) const;

private:
  platform::ReadOnlyMappedFile file_;
  std::size_t payload_offset_{0};
  std::map<std::string, TensorInfo, std::less<>> tensors_;
};

}  // namespace gufo::models::yue2

#endif  // GUFO_MODELS_YUE2_SAFETENSORS_HPP_
