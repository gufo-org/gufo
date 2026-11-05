#include "audio_decoder.hpp"

#include <sndfile.h>
#include <soxr.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "audio_processor.hpp"

namespace gufo::models::embedding_gemma2 {
namespace {
struct Reader {
  std::span<const std::uint8_t> bytes;
  sf_count_t position{0};
  const std::function<bool()>& is_cancelled;
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::now() + std::chrono::seconds(15)};
  bool stopped{false};
  bool Stop() {
    stopped = stopped || (is_cancelled && is_cancelled()) ||
              std::chrono::steady_clock::now() > deadline;
    return stopped;
  }
};
sf_count_t Length(void* context) {
  return static_cast<sf_count_t>(static_cast<Reader*>(context)->bytes.size());
}
sf_count_t Seek(sf_count_t offset, int whence, void* context) {
  auto& r = *static_cast<Reader*>(context);
  const sf_count_t size = static_cast<sf_count_t>(r.bytes.size());
  const sf_count_t base = whence == SEEK_SET   ? 0
                          : whence == SEEK_CUR ? r.position
                          : whence == SEEK_END ? size
                                               : -1;
  if (r.Stop() || base < 0 || offset < -base || offset > size - base)
    return -1;
  r.position = base + offset;
  return r.position;
}
sf_count_t Read(void* target, sf_count_t count, void* context) {
  auto& r = *static_cast<Reader*>(context);
  if (r.Stop() || count < 0)
    return 0;
  count = std::min(count, static_cast<sf_count_t>(r.bytes.size()) - r.position);
  std::memcpy(target, r.bytes.data() + r.position,
              static_cast<std::size_t>(count));
  r.position += count;
  return count;
}
sf_count_t Tell(void* context) {
  return static_cast<Reader*>(context)->position;
}
}  // namespace

std::vector<float> DecodeAudio(std::span<const std::uint8_t> bytes,
                               const std::function<bool()>& is_cancelled) {
  if (bytes.empty() || bytes.size() > 20 * 1024 * 1024)
    throw std::invalid_argument("embedding audio is empty or exceeds 20 MiB");
  Reader reader{bytes, 0, is_cancelled};
  SF_VIRTUAL_IO io{Length, Seek, Read, nullptr, Tell};
  SF_INFO info{};
  std::unique_ptr<SNDFILE, decltype(&sf_close)> file(
      sf_open_virtual(&io, SFM_READ, &info, &reader), sf_close);
  if (!file || reader.stopped)
    throw std::invalid_argument("embedding audio could not be decoded");
  const int type = info.format & SF_FORMAT_TYPEMASK;
  if (type != SF_FORMAT_WAV && type != SF_FORMAT_WAVEX &&
      type != SF_FORMAT_FLAC && type != SF_FORMAT_OGG && type != SF_FORMAT_MPEG)
    throw std::invalid_argument(
        "embedding audio supports WAV, FLAC, Ogg and MP3");
  // Without a Xing/Info header, libsndfile estimates an MP3's length from its
  // size and bitrate, and it does not say whether a length is exact. Like
  // soundfile under librosa, keep the samples that actually decode for every
  // MP3. The 30-second bound then applies while decoding.
  const bool estimated = type == SF_FORMAT_MPEG;
  const auto max_frames = static_cast<sf_count_t>(info.samplerate) * 30;
  if (info.channels < 1 || info.channels > 8 || info.samplerate < 8000 ||
      info.samplerate > 192000 || info.frames < 1 ||
      (!estimated && info.frames > max_frames))
    throw std::invalid_argument(
        "embedding audio exceeds 30 seconds, 8 channels or 8–192 kHz bounds");
  std::vector<float> mono;
  mono.reserve(static_cast<std::size_t>(std::min(info.frames, max_frames)));
  std::array<float, 4096 * 8> block{};
  for (;;) {
    if (reader.Stop())
      throw std::runtime_error(
          "embedding audio decoding cancelled or timed out");
    const auto frames = sf_readf_float(file.get(), block.data(), 4096);
    if (frames <= 0)
      break;
    if (mono.size() + frames > static_cast<std::size_t>(info.samplerate) * 30)
      throw std::invalid_argument("embedding audio exceeds 30 seconds");
    for (sf_count_t t = 0; t < frames; ++t) {
      float sum = 0;
      for (int c = 0; c < info.channels; ++c) {
        const float value = block[t * info.channels + c];
        if (!std::isfinite(value))
          throw std::invalid_argument("embedding audio has nonfinite samples");
        sum += value;
      }
      mono.push_back(sum / info.channels);
    }
  }
  if (reader.stopped || sf_error(file.get()) || mono.empty() ||
      (!estimated && mono.size() != static_cast<std::size_t>(info.frames)))
    throw std::invalid_argument(
        "embedding audio is truncated or decoding failed");
  if (info.samplerate == static_cast<int>(kAudioSampleRate))
    return mono;
  const std::size_t length =
      (mono.size() * kAudioSampleRate + info.samplerate - 1) / info.samplerate;
  std::vector<float> output(length + 1, 0);
  const auto io_spec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
  const auto quality = soxr_quality_spec(SOXR_HQ, 0);
  std::size_t consumed = 0, produced = 0;
  const auto failure = soxr_oneshot(
      info.samplerate, kAudioSampleRate, 1, mono.data(), mono.size(), &consumed,
      output.data(), output.size(), &produced, &io_spec, &quality, nullptr);
  if (failure || consumed != mono.size() || reader.Stop())
    throw std::runtime_error("embedding audio resampling failed or cancelled");
  output.resize(length);  // librosa's fix_length uses ceil, padding with zero.
  return output;
}
}  // namespace gufo::models::embedding_gemma2
