#ifndef GUFO_CORE_HIP_WEIGHT_UPLOAD_HPP_
#define GUFO_CORE_HIP_WEIGHT_UPLOAD_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::hip {

/// Bounded disk-to-device pipeline. Copy queues a tensor without waiting for
/// its upload; destinations must remain alive until Finish or destruction.
/// The mapped GGUF payloads are never touched.
class WeightUpload {
public:
  static std::unique_ptr<WeightUpload> Create(
      std::span<const core::GgufMappedRegion> shards, std::string* error);
  ~WeightUpload();
  WeightUpload(const WeightUpload&) = delete;
  WeightUpload& operator=(const WeightUpload&) = delete;

  bool Copy(std::uint32_t shard, std::uint64_t offset, std::size_t size,
            void* device, std::string* error);
  /// Resolve a pointer within the supplied mappings without reading its pages.
  bool CopyMapped(const void* source, std::size_t size, void* device,
                  std::string* error);
  /// Waits for every queued copy, including on failure.
  bool Finish(std::string* error);

private:
  struct State;
  WeightUpload();
  std::unique_ptr<State> state_;
};

}  // namespace gufo::hip

#endif
