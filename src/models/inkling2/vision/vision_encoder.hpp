// Inkling 2 Mini vision encoder (mmproj) on HIP.
//
//   VisionEncoder enc;
//   enc.Load("mmproj-Inkling-2-Mini-Q8_0.gguf", &err);
//   VisionEncoder::Result r;
//   enc.Encode(rgb, w, h, stream, &r, &err);   // r.embd: device float
//   [r.n_tokens][r.dim] (dim = 2304)
//
// The embeddings are the rows the text model consumes in place of the image
// placeholder tokens: not normed, one per token, consecutive positions. The
// algorithm follows llama.cpp tools/mtmd/models/inkling2v.cpp (e412c6f).
#pragma once

#include <hip/hip_runtime.h>

#include <memory>
#include <string>

#include "vision_image.hpp"

namespace gufo::models::inkling2::vision {

class VisionEncoder {
public:
  struct Result {
    float* embd = nullptr;  // device, [n_tokens][dim] row-major; owned by the
                            // encoder, valid until the next Encode() on this
                            // encoder (copy or consume on `stream`)
    int n_tokens = 0;       // tok_w * tok_h
    int tok_w = 0, tok_h = 0;  // token grid (raster order, row-major)
    int dim = 0;               // 2304
    ImagePlan plan;
  };
  struct Options {
    /// Test only (needs Load(..., exact_reference = true)): round-trip the
    /// activations of every Q8_0 GEMM through Q8_0 (per-32 amax/127, f16 scale)
    /// like ggml's mat-mul does, to reproduce the oracle's own rounding.
    bool emulate_q8_acts = false;
  };

  VisionEncoder();
  ~VisionEncoder();
  VisionEncoder(const VisionEncoder&) = delete;
  VisionEncoder& operator=(const VisionEncoder&) = delete;

  /// Loads and uploads the mmproj GGUF (clip.vision.*, projector_type
  /// inkling2v). Call once. Default (fast): linear weights are dequantised to
  /// f16 on the device (~650 MB), GEMMs and attention use f16 WMMA with f32
  /// accumulation. `exact_reference = true` keeps the file's weight types and
  /// runs FP32 FMA GEMMs and FP32 attention (about as fast at tiny sizes, 10x
  /// slower at large ones; for tests).
  bool Load(const std::string& mmproj_path, std::string* err,
            bool exact_reference = false);
  int embedding_dim() const;    // text-model width the projector outputs (2304)
  size_t device_bytes() const;  // weights + current workspace
  /// Frees the workspace an Encode grew (up to ~0.5 GiB for the largest
  /// images); the next Encode allocates it again.
  void ReleaseWorkspace();

  /// Encodes one RGB8 image (w*h*3, interleaved). Over-limit images are
  /// downscaled first. All work is enqueued on `stream`; the call itself
  /// synchronizes only for the small host
  /// -> device image upload (it uses a pinned-free hipMemcpyAsync + stream
  /// order).
  bool Encode(const uint8_t* rgb, int w, int h, hipStream_t stream, Result* out,
              std::string* err);

  /// Number of image tokens an image of w x h will produce (no GPU work).
  static int CountTokens(int w, int h) { return PlanImage(w, h).n_tokens; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::inkling2::vision
