#ifndef GUFO_MODELS_MOSS_IMAGE_MODEL_HPP_
#define GUFO_MODELS_MOSS_IMAGE_MODEL_HPP_

#include <filesystem>
#include <memory>
#include <vector>

#include "src/models/qwen_image_21/model.hpp"

namespace gufo::models::moss_image {

// Requests, results and observers share the Qwen-Image serving contract.
using qwen_image_21::CancellationCheck;
using qwen_image_21::Image;
using qwen_image_21::Observer;
using qwen_image_21::Request;
using qwen_image_21::Result;

/// True when `root` is a Diffusers directory of MOSSImagePipeline.
bool IsMossImageDirectory(const std::filesystem::path& root);

/// Text-to-image only: reference images, sizes off the 16-pixel grid and
/// step counts outside [1, 100] are rejected.
void ValidateRequest(const Request& request);

/// The MOSSImageScheduler sigmas for `steps`, snapped to the training grid
/// and ending with the terminal zero.
std::vector<float> FlowSigmas(int steps);

struct Options {
  /// MOSS's Qwen3.5 text encoder converted to a GGUF (qwen35, BF16 or Q8_0).
  std::filesystem::path text_encoder;
  float guidance_scale{3.0F};
  /// Zero-based DiT block skipped in the negative branch (-1 disables).
  int uncond_skip_layer{22};
  /// Development only: raw FP32 [128, height / 16, width / 16] noise that
  /// replaces the seeded draw, to replay a reference trajectory.
  std::filesystem::path initial_noise;
};

class Model {
public:
  Model(const std::filesystem::path& root, Options options);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// One execution at a time. Each request draws its own noise from
  /// `request.seed`; the reference pipeline's default is 50 steps.
  Result Generate(const Request& request,
                  const CancellationCheck& cancelled = {},
                  const Observer& observer = {});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::moss_image
#endif
