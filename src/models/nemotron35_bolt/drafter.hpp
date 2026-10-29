#ifndef GUFO_MODELS_NEMOTRON35_BOLT_DRAFTER_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_DRAFTER_HPP_

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/nemotron35_bolt/config.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::nemotron35_bolt {

struct DrafterWeights;
namespace rocm {
class DeviceModel;
struct DeviceDrafter;
}  // namespace rocm

/// Resident LiLiCorr DFlash drafter weights. It drafts with the trunk's
/// embedding and output head, so it is only usable next to that trunk.
class Drafter final {
public:
  ~Drafter();
  Drafter(const Drafter&) = delete;
  Drafter& operator=(const Drafter&) = delete;

  [[nodiscard]] static std::unique_ptr<Drafter> Load(
      const std::string& path, const Config& target,
      std::string* error_msg = nullptr);

  [[nodiscard]] const DrafterConfig& config() const noexcept;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] const rocm::DeviceDrafter& device() const noexcept {
    return *device_;
  }

private:
  Drafter() = default;
  std::unique_ptr<core::GgufReader> reader_;
  std::unique_ptr<DrafterWeights> weights_;
  std::unique_ptr<rocm::DeviceDrafter> device_;
};

/// The LiLiCorr lattice of one drafted block: per drafted slot its top
/// candidates in descending drafter logit, and the pairwise scores.
struct DraftLattice {
  std::uint32_t slots{0};
  std::uint32_t top_k{0};
  std::vector<std::int32_t> candidates;  ///< [slots][top_k]
  /// top_k scores of the first slot's candidates against the anchor, then
  /// [slots - 1][top_k previous][top_k next] transition scores.
  std::vector<float> scores;

  /// llama.cpp's greedy walk: the best first candidate, then at each slot
  /// the best successor of the previous choice; at most `max_tokens`.
  [[nodiscard]] std::vector<std::int32_t> Walk(std::uint32_t max_tokens) const;
};

/// Per-sequence drafter state on the device: the drafter layers' K/V rings
/// over the trunk's context, the projected context rows the LiLiCorr anchor
/// reads, and scratch. Work runs on the owner's stream.
class DraftContext final {
public:
  ~DraftContext();
  DraftContext(const DraftContext&) = delete;
  DraftContext& operator=(const DraftContext&) = delete;

  /// `max_rows` bounds one Inject call.
  [[nodiscard]] static std::unique_ptr<DraftContext> Create(
      const Drafter& drafter, const rocm::DeviceModel& trunk,
      std::uint32_t max_rows, hipStream_t stream, hipblasHandle_t blas,
      std::string* error_msg = nullptr);

  /// Enqueues the projection of `n` rows of trunk features (device memory,
  /// FeatureSize floats each) at positions *first.. into the rings and the
  /// anchor rows. `first` is device memory so a captured graph can replay.
  [[nodiscard]] bool Inject(const float* features, const int* first, int n,
                            std::string* error_msg = nullptr);
  /// Host features, for validation: uploads and injects them at `first`.
  [[nodiscard]] bool InjectHost(std::span<const float> features,
                                std::uint32_t first,
                                std::string* error_msg = nullptr);

  /// Drafts the block that starts with `anchor` at position `pos0` (pos0 >= 1;
  /// the rings must hold the features before pos0) and waits for it.
  /// `logits`, when not empty, receives the drafted slots' logits.
  [[nodiscard]] bool Draft(std::int32_t anchor, std::uint32_t pos0,
                           DraftLattice* out, std::span<float> logits = {},
                           std::string* error_msg = nullptr);

  /// Ring and anchor contents, for session snapshots.
  [[nodiscard]] std::size_t StateBytes() const noexcept;
  [[nodiscard]] bool SaveState(std::span<std::uint8_t> out) const;
  [[nodiscard]] bool RestoreState(std::span<const std::uint8_t> in);

  struct Buffers;

private:
  DraftContext() = default;
  bool Fail(std::string* error_msg, std::string message) const;

  const Drafter* drafter_{nullptr};
  const rocm::DeviceModel* trunk_{nullptr};
  hipStream_t stream_{nullptr};
  hipblasHandle_t blas_{nullptr};
  std::uint32_t max_rows_{0};
  std::unique_ptr<Buffers> b_;
};

}  // namespace gufo::models::nemotron35_bolt

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_DRAFTER_HPP_
