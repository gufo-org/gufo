#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_VIDEO_PROCESSOR_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_VIDEO_PROCESSOR_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "src/models/embedding_gemma2/image_processor.hpp"

namespace gufo::models::embedding_gemma2 {
/// The pinned video processor's per-frame `max_soft_tokens`; frames accept
/// the same budgets as images (`kImageSoftTokenBudgets`).
inline constexpr std::uint32_t kDefaultVideoSoftTokens = 140;
struct VideoOptions {
  double fps{1};
  std::size_t max_frames{32};
  bool timestamps{false};
  bool truncate_overflow{false};
  std::uint32_t max_soft_tokens{kDefaultVideoSoftTokens};
};
struct VideoFrame {
  std::size_t index{0};
  double timestamp{0};
  ImagePatches patches;
};
/// Exact FPS/floor selection followed by uniform linspace/floor overflow.
[[nodiscard]] std::vector<std::size_t> SampleVideoFrames(
    std::size_t frames, double fps, const VideoOptions& options = {});
[[nodiscard]] bool ValidateVideoProcessor(const std::string& model_dir,
                                          std::string* error = nullptr);
/// In-memory FFmpeg decoder; never opens nested URLs or files. Selects the
/// first video stream and ignores audio, matching the visual reference path.
/// Bounds: 20 MiB, 300 seconds, 4096 decoded frames, 4M pixels/frame,
/// 512M decoded pixels over both passes, and 15 seconds including conversion.
/// Each frame is resized to at most `options.max_soft_tokens`; the selected
/// frames, repeats included, may hold at most `soft_token_budget` soft tokens.
[[nodiscard]] std::vector<VideoFrame> ProcessVideo(
    std::span<const std::uint8_t> bytes, const VideoOptions& options = {},
    const std::function<bool()>& is_cancelled = {},
    std::size_t frame_budget = 32, std::size_t soft_token_budget = SIZE_MAX);
/// Reference per-frame image boundaries, video placeholders and optional mm:ss.
[[nodiscard]] std::string FrameVideo(const std::vector<VideoFrame>& frames,
                                     const VideoOptions& options);
}  // namespace gufo::models::embedding_gemma2
#endif
