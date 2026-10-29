#ifndef GUFO_MODELS_LFM3_AUDIO_MODEL_HPP_
#define GUFO_MODELS_LFM3_AUDIO_MODEL_HPP_
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/models/lfm3/engine.hpp"
#include "src/models/lfm3_audio/weights.hpp"
namespace gufo::models::lfm3_audio {
// Special tokens of the LFM3-Audio vocabulary.
struct Tokens {
  std::uint32_t audio{}, audio_out{}, audio_start{}, text_end{}, mixed_start{},
      im_end{}, im_start{}, bos{};
};
class Model;
// Device state of one stream of speech output: the depth decoder sampling
// codebook frames and the LiquidCodec decoder turning them into 24 kHz audio.
// Weights are shared; state is per voice.
class Voice {
public:
  ~Voice();
  // Codes of one frame from a backbone hidden state already on the device.
  // Codebooks are revealed in a random order over three passes; each pass
  // scores the codebooks still masked. temperature 0 picks the best code.
  struct Sampling {
    float temperature{1.0F};
    int top_k{128};
    float repetition_penalty{1.1F};
    std::size_t repetition_window{4096};
  };
  // The whole frame runs on the device as one graph; embedding, when set,
  // receives the frame's backbone input embedding (all end codes for an end
  // frame). ordered reveals codebooks in index order, for reproducible tests.
  bool SampleFrame(const float* device_hidden, const Sampling& sampling,
                   std::mt19937_64& rng, std::span<std::int32_t> codes,
                   float* embedding = nullptr, bool ordered = false,
                   std::string* error = nullptr);
  // Depth logits of all codebooks for a hidden state, codes and mask
  // (masked codebooks use the mask embedding), for tests: 12 x 2017.
  bool DepthLogits(std::span<const float> hidden,
                   std::span<const std::int32_t> codes,
                   std::span<const std::int32_t> masked,
                   std::span<float> logits, std::string* error = nullptr);
  // Backbone input embedding of a frame (sum of its codebook embeddings),
  // written to device memory on the voice stream and synchronized.
  bool FrameEmbedding(std::span<const std::int32_t> codes, float* device_out,
                      std::string* error = nullptr);
  // Decodes frames to audio. Until finalize, the trailing samples that later
  // frames still overlap are held back; chunks concatenate to the same audio
  // as one call over all frames.
  bool Decode(std::span<const std::int32_t> frames, bool finalize,
              std::vector<float>* samples, std::string* error = nullptr);
  // Forgets codec state and repetition history. Decode runs on its own
  // stream, so it may overlap a backbone step of another thread.
  void Reset();

private:
  friend class Model;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit Voice(std::shared_ptr<Model> model);
};
class Model : public std::enable_shared_from_this<Model> {
public:
  static constexpr std::uint32_t kSampleRate = 16000, kOutputRate = 24000,
                                 kFrameSamples = 1920;
  ~Model();
  static std::shared_ptr<Model> Load(const std::string& backbone,
                                     const std::string& mmproj,
                                     std::string* error = nullptr);
  lfm3::Model& backbone() const;
  const lfm3::Tokenizer& tokenizer() const;
  const AudioWeights& weights() const;
  const Tokens& tokens() const;
  // Log-mel features of 16 kHz mono samples, frame-major [frames][128], as
  // the official extractor: right padding to whole hops, periodic Hann STFT
  // (400/160), Slaney mel, log10 floored at -12, then (x + 4) / 4.
  bool Features(std::span<const float> samples, std::vector<float>* mel,
                std::uint32_t* frames, std::string* error = nullptr);
  // Projected backbone embeddings of one 16 kHz clip, one row of backbone
  // width per eight mel frames.
  bool Encode(std::span<const float> samples, std::vector<float>* embeddings,
              std::string* error = nullptr);
  std::unique_ptr<Voice> CreateVoice(std::string* error = nullptr);

private:
  friend class Voice;
  Model();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
std::uint32_t MelFrames(std::size_t samples);
std::uint32_t EncoderTokens(std::uint32_t frames);
// Mono float samples of a PCM16/PCM24/PCM32/float32 WAV.
bool DecodeWav(std::span<const std::byte> wav, std::vector<float>* samples,
               std::uint32_t* rate, std::string* error = nullptr);
std::vector<std::byte> EncodeWav(std::span<const float> samples,
                                 std::uint32_t rate);
// Band-limited (windowed-sinc) resampling of mono samples.
std::vector<float> Resample(std::span<const float> samples, std::uint32_t from,
                            std::uint32_t to);
}  // namespace gufo::models::lfm3_audio
#endif
