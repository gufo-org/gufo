#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "src/models/lfm3_audio/model.hpp"
namespace gufo::models::lfm3_audio {
namespace {
std::uint32_t Read32(const std::byte* p) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i)
    v |= std::uint32_t(std::to_integer<std::uint8_t>(p[i])) << (8 * i);
  return v;
}
std::uint16_t Read16(const std::byte* p) {
  return std::uint16_t(std::to_integer<std::uint8_t>(p[0]) |
                       (std::to_integer<std::uint8_t>(p[1]) << 8));
}
void Write32(std::vector<std::byte>& out, std::uint32_t v) {
  for (int i = 0; i < 4; ++i)
    out.push_back(std::byte((v >> (8 * i)) & 255));
}
void Write16(std::vector<std::byte>& out, std::uint16_t v) {
  out.push_back(std::byte(v & 255));
  out.push_back(std::byte(v >> 8));
}
}  // namespace
bool DecodeWav(std::span<const std::byte> wav, std::vector<float>* samples,
               std::uint32_t* rate, std::string* error) {
  try {
    if (wav.size() < 12 || std::memcmp(wav.data(), "RIFF", 4) ||
        std::memcmp(wav.data() + 8, "WAVE", 4))
      throw std::invalid_argument("audio must be a RIFF/WAVE file");
    std::uint16_t format = 0, channels = 0, bits = 0;
    std::uint32_t sample_rate = 0;
    std::span<const std::byte> data;
    for (std::size_t at = 12; at + 8 <= wav.size();) {
      const auto size = Read32(wav.data() + at + 4);
      const auto body = at + 8;
      if (size > wav.size() - body)
        throw std::invalid_argument("truncated WAV chunk");
      if (!std::memcmp(wav.data() + at, "fmt ", 4) && size >= 16) {
        format = Read16(wav.data() + body);
        channels = Read16(wav.data() + body + 2);
        sample_rate = Read32(wav.data() + body + 4);
        bits = Read16(wav.data() + body + 14);
        if (format == 0xFFFE && size >= 26)
          format = Read16(wav.data() + body + 24);  // extensible subformat
      } else if (!std::memcmp(wav.data() + at, "data", 4))
        data = wav.subspan(body, size);
      at = body + size + (size & 1);
    }
    const bool pcm = format == 1 && (bits == 16 || bits == 24 || bits == 32);
    const bool ieee = format == 3 && bits == 32;
    if (!channels || !sample_rate || (!pcm && !ieee) || data.empty())
      throw std::invalid_argument(
          "WAV must be PCM16/24/32 or float32 with audio data");
    const std::size_t width = bits / 8, frame = width * channels;
    const std::size_t count = data.size() / frame;
    samples->assign(count, 0.0F);
    for (std::size_t i = 0; i < count; ++i) {
      float sum = 0;
      for (std::size_t c = 0; c < channels; ++c) {
        const auto* p = data.data() + i * frame + c * width;
        float v;
        if (ieee) {
          const auto bits_value = Read32(p);
          std::memcpy(&v, &bits_value, 4);
        } else if (width == 2)
          v = std::int16_t(Read16(p)) / 32768.0F;
        else if (width == 3) {
          std::int32_t x =
              std::int32_t(Read16(p)) |
              (std::int32_t(std::to_integer<std::int8_t>(p[2])) << 16);
          v = x / 8388608.0F;
        } else
          v = float(std::int32_t(Read32(p)) / 2147483648.0);
        sum += v;
      }
      (*samples)[i] = sum / channels;
    }
    *rate = sample_rate;
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
std::vector<std::byte> EncodeWav(std::span<const float> samples,
                                 std::uint32_t rate) {
  std::vector<std::byte> out;
  out.reserve(44 + samples.size() * 2);
  const auto data_bytes = std::uint32_t(samples.size() * 2);
  for (char c : std::string_view("RIFF"))
    out.push_back(std::byte(c));
  Write32(out, 36 + data_bytes);
  for (char c : std::string_view("WAVEfmt "))
    out.push_back(std::byte(c));
  Write32(out, 16);
  Write16(out, 1);
  Write16(out, 1);
  Write32(out, rate);
  Write32(out, rate * 2);
  Write16(out, 2);
  Write16(out, 16);
  for (char c : std::string_view("data"))
    out.push_back(std::byte(c));
  Write32(out, data_bytes);
  for (float v : samples) {
    const float clipped = std::clamp(v, -1.0F, 1.0F);
    Write16(out, std::uint16_t(std::int16_t(std::lround(clipped * 32767.0F))));
  }
  return out;
}
Resampler::Resampler(std::uint32_t from, std::uint32_t to)
    : ratio_(double(to) / from),
      cutoff_(std::min(1.0, ratio_) * 0.95),
      half_(24 / cutoff_),
      same_(from == to) {}
std::vector<float> Resampler::Push(std::span<const float> x, bool finish) {
  if (same_)
    return {x.begin(), x.end()};
  // Hann-windowed sinc with the cutoff below the lower Nyquist rate. Output
  // i sits at input time i / ratio and waits until its window has arrived.
  history_.insert(history_.end(), x.begin(), x.end());
  received_ += x.size();
  const auto total = std::int64_t(received_);
  const auto end = finish ? std::uint64_t(std::ceil(received_ * ratio_))
                          : std::numeric_limits<std::uint64_t>::max();
  std::vector<float> y;
  for (; next_ < end; ++next_) {
    const double t = next_ / ratio_;
    const auto first = std::int64_t(std::ceil(t - half_));
    const auto last = std::int64_t(std::floor(t + half_));
    if (!finish && last >= total)
      break;
    double sum = 0;
    for (auto j = std::max<std::int64_t>(first, 0);
         j <= std::min<std::int64_t>(last, total - 1); ++j) {
      const double d = t - double(j);
      const double a = std::numbers::pi * cutoff_ * d;
      const double sinc = std::abs(a) < 1e-9 ? 1.0 : std::sin(a) / a;
      const double window = 0.5 + 0.5 * std::cos(std::numbers::pi * d / half_);
      sum += history_[std::size_t(j - base_)] * cutoff_ * sinc * window;
    }
    y.push_back(float(sum));
  }
  // Drop input the next output's window no longer reaches.
  const auto keep = std::clamp<std::int64_t>(
      std::int64_t(std::ceil(next_ / ratio_ - half_)), base_, total);
  history_.erase(history_.begin(), history_.begin() + (keep - base_));
  base_ = keep;
  return y;
}
std::vector<float> Resample(std::span<const float> x, std::uint32_t from,
                            std::uint32_t to) {
  if (x.empty())
    return {};
  return Resampler(from, to).Push(x, true);
}
}  // namespace gufo::models::lfm3_audio
