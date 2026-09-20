#ifndef GUFO_MODELS_QWEN3_ASR_HIP_AUDIO_ENCODER_RUNTIME_HPP_
#define GUFO_MODELS_QWEN3_ASR_HIP_AUDIO_ENCODER_RUNTIME_HPP_

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::qwen3_asr::hip {

struct AudioEncoderOutput {
  // Float32 view of BF16 activations, row-major [tokens, 1024].
  std::vector<float> values;
  std::size_t tokens{0};
};

struct AudioEncoderTrace {
  AudioEncoderOutput frontend;
  AudioEncoderOutput layer0;
  // Row-major [tokens, 2048].
  AudioEncoderOutput final;
};

struct AudioEncoderDeviceOutput {
  // Device-resident float32 [tokens, 2048]. The pointer remains valid until
  // the next call on this runtime.
  const float* values{nullptr};
  std::size_t tokens{0};
};

class AudioEncoderHipRuntime {
public:
  ~AudioEncoderHipRuntime();

  AudioEncoderHipRuntime(const AudioEncoderHipRuntime&) = delete;
  AudioEncoderHipRuntime& operator=(const AudioEncoderHipRuntime&) = delete;
  AudioEncoderHipRuntime(AudioEncoderHipRuntime&&) = delete;
  AudioEncoderHipRuntime& operator=(AudioEncoderHipRuntime&&) = delete;

  [[nodiscard]] static std::unique_ptr<AudioEncoderHipRuntime> Create(
      const std::string& model_root, std::string* error = nullptr);

  /// Runs chunking, the three convolutional downsamplers, conv_out, and
  /// sinusoidal position addition. This is the first captured parity boundary.
  [[nodiscard]] bool EncodeFrontend(std::span<const float> log_mel,
                                    std::size_t frames,
                                    AudioEncoderOutput* output,
                                    std::string* error = nullptr);

  /// Runs the complete 24-layer audio transformer and final 2048-wide
  /// projection, retaining the two official intermediate parity boundaries.
  [[nodiscard]] bool Encode(std::span<const float> log_mel, std::size_t frames,
                            AudioEncoderTrace* output,
                            std::string* error = nullptr);

  /// Production path. Runs the same encoder without intermediate host captures
  /// and retains the final embeddings on the GPU.
  [[nodiscard]] bool EncodeDevice(
      std::span<const float> log_mel, std::size_t frames,
      AudioEncoderDeviceOutput* output, std::string* error = nullptr,
      const std::function<bool()>& is_cancelled = {});

private:
  struct Impl;
  explicit AudioEncoderHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_asr::hip

#endif  // GUFO_MODELS_QWEN3_ASR_HIP_AUDIO_ENCODER_RUNTIME_HPP_
