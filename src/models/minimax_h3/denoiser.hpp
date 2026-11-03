#ifndef GUFO_MODELS_MINIMAX_H3_DENOISER_HPP_
#define GUFO_MODELS_MINIMAX_H3_DENOISER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sampling.hpp"

namespace gufo::minimax_h3 {

inline constexpr std::size_t kH3TextConditionWidth = 5120;
inline constexpr std::size_t kH3VideoLatentChannels = 24;
inline constexpr std::size_t kH3VideoPatchWidth = 96;
inline constexpr std::size_t kH3AudioLatentChannels = 32;
inline constexpr std::size_t kH3AudioTracks = 2;

struct DenoiserOptions {
  GenerationGeometry geometry;
  int evaluations{0};
  std::size_t text_rows{0};
  int active_blocks{50};
  int reuse_interval{1};
  bool row_parallel_attention{true};
  // FL2VA keyframe anchors ride as extra leading video rows; zero keeps the
  // text-only packed sequence [text | audio | video].
  std::size_t condition_video_rows{0};
};

struct DenoiserConditioning {
  std::span<const std::uint16_t> layer50;
  std::span<const std::uint8_t> tags;
  // Patchified anchor rows in packed order (first, then last),
  // condition_video_rows * kH3VideoPatchWidth F32, frozen for the loop.
  std::span<const float> condition_rows;
  // fp64 rotary time per anchor, same packed order; empty for text-only.
  std::span<const double> anchor_temporal;
};

struct DenoiserVelocity {
  std::vector<float> video;
  std::vector<float> audio;
};

[[nodiscard]] std::optional<double> DenoiserGateScore(
    std::span<const std::uint16_t> modulation, std::uint32_t time_rows,
    std::string* error = nullptr);
[[nodiscard]] std::optional<std::vector<std::size_t>>
SelectGateRankedDenoiserBlocks(std::span<const double> gate_scores,
                               int active_blocks, std::string* error = nullptr);

struct DenoiserTelemetry {
  double setup_ms{0.0};
  double text_refiner_ms{0.0};
  double adaln_precompute_ms{0.0};
  double core_load_ms{0.0};
  std::vector<double> forward_ms;
  double sampler_ms{0.0};
  std::uint64_t core_weight_bytes{0};
  std::uint64_t persistent_bytes{0};
  std::uint64_t scratch_bytes{0};
  std::uint64_t peak_live_bytes{0};
  std::uint64_t cumulative_allocation_bytes{0};
  std::uint64_t core_gemm_plan_validations{0};
  std::uint64_t minor_page_faults{0};
  std::uint64_t major_page_faults{0};
  std::uint64_t swap_bytes{0};
  std::uint64_t dispatches{0};
  std::uintptr_t scratch_address{0};
  bool row_parallel_attention{true};
  std::string first_non_finite;
};

using DenoiserProgress = void (*)(int completed_steps, int total_steps,
                                  void* opaque);

class DenoiserSession {
public:
  ~DenoiserSession();

  DenoiserSession(const DenoiserSession&) = delete;
  DenoiserSession& operator=(const DenoiserSession&) = delete;
  DenoiserSession(DenoiserSession&&) noexcept;
  DenoiserSession& operator=(DenoiserSession&&) noexcept;

  [[nodiscard]] static std::unique_ptr<DenoiserSession> Create(
      const ModelInventory& inventory, const DenoiserOptions& options,
      const DenoiserConditioning& conditioning,
      const CancellationToken* cancellation, DenoiserTelemetry* telemetry,
      std::string* error = nullptr);

  [[nodiscard]] bool Forward(int step, std::span<const float> video_latent,
                             std::span<const float> audio_latent,
                             const CancellationToken* cancellation,
                             DenoiserVelocity* velocity,
                             DenoiserTelemetry* telemetry,
                             std::string* error = nullptr);

  [[nodiscard]] bool Denoise(std::span<float> video_latent,
                             std::span<float> audio_latent,
                             const CancellationToken* cancellation,
                             DenoiserTelemetry* telemetry,
                             std::string* error = nullptr);

  [[nodiscard]] bool DenoiseWithProgress(std::span<float> video_latent,
                                         std::span<float> audio_latent,
                                         const CancellationToken* cancellation,
                                         DenoiserProgress progress,
                                         void* progress_opaque,
                                         DenoiserTelemetry* telemetry,
                                         std::string* error = nullptr);

  [[nodiscard]] const GenerationGeometry& geometry() const noexcept;
  [[nodiscard]] const PackedLayout& layout() const noexcept;
  [[nodiscard]] const SigmaSchedule& schedule() const noexcept;
  [[nodiscard]] std::span<const std::uint16_t> block0_modulation()
      const noexcept;
  [[nodiscard]] std::span<const std::uint16_t> refined_text() const noexcept;

private:
  struct Impl;
  explicit DenoiserSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_DENOISER_HPP_
