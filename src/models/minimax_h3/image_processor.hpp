#ifndef GUFO_MODELS_MINIMAX_H3_IMAGE_PROCESSOR_HPP_
#define GUFO_MODELS_MINIMAX_H3_IMAGE_PROCESSOR_HPP_

// Keyframe image preprocessing for MiniMax H3 FL2VA conditioning (upstream
// issue #177). Two responsibilities:
//
//   1. Canvas fitting (diffusers modular_pipelines/minimax_h3
//      before_encoder.py:MiniMaxH3ResizeStep, reproduced verbatim in
//      local://h3-fl2va-reference.md Part 3): the packed-index-0 keyframe is
//      stretched to the exact canvas with a PIL LANCZOS resize (no aspect
//      preservation, exact-size inputs pass through untouched); index >= 1
//      keyframes are aspect-cover scaled and center cropped with explicit
//      Python round() (banker's rounding) arithmetic.
//
//   2. Patchification (FL2VA/processor/preprocessor_config.json,
//      Qwen2VLImageProcessorFast semantics): canvas RGB8 -> KeyframeVision
//      rows for the vision tower.
//
// The Lanczos resampler is Pillow-parity 22-bit fixed-point with per-pass
// uint8 rounding, copied into this model-private file from
// src/models/qwen_image_21/image.cpp (ResizeImage / LanczosFilters, lines
// 45-73 and 179-237, attribution at the implementation). Inputs are
// core::Image RGB8 already in display orientation (core::DecodeImage
// applies EXIF), so no alpha or premultiplication handling is required;
// ffmpeg's lanczos is NOT pixel-identical to Pillow and is unusable here
// (#177 gates pixel parity).

#include <cstdint>
#include <gufo/ext/image.hpp>
#include <string>
#include <vector>

#include "src/models/minimax_h3/vision_tower.hpp"

namespace gufo::minimax_h3::image_processor {

enum class KeyframeFit {
  kStretch,   // Packed index 0: exact (canvas_width, canvas_height) resize.
  kCoverCrop  // Packed index >= 1: cover scale, center crop.
};

// Resamples `image` onto the canvas. `rgb` receives canvas_width *
// canvas_height * 3 tightly packed RGB8 bytes. Exact-size inputs copy
// through without resampling. The canvas must be a positive multiple of
// 32 so the vision grid stays even (the tower merges 2x2 patches).
[[nodiscard]] bool PrepareKeyframe(const core::Image& image, int canvas_width,
                                   int canvas_height, KeyframeFit fit,
                                   std::vector<std::uint8_t>* rgb,
                                   std::string* error);

// The processor's `size.shortest_edge` (smart_resize min_pixels). Below it the
// reference upscales the keyframe before patching, which BuildKeyframeVision
// does not reproduce, so smaller canvases are rejected.
inline constexpr std::int64_t kVisionMinimumPixels = 65536;

// Builds vision-tower rows from canvas RGB8 produced by PrepareKeyframe:
// width/height must be multiples of 32, `rgb` covers width * height * 3
// bytes. The single frame is duplicated across both temporal taps and each
// row holds ((c * 2 + t) * 16 + y) * 16 + x values of
// (uint8 / 255 - 0.5) / 0.5 in float32, rows in merge-block order.
[[nodiscard]] bool BuildKeyframeVision(const std::vector<std::uint8_t>& rgb,
                                       int width, int height,
                                       KeyframeVision* output,
                                       std::string* error);

}  // namespace gufo::minimax_h3::image_processor

#endif  // GUFO_MODELS_MINIMAX_H3_IMAGE_PROCESSOR_HPP_
