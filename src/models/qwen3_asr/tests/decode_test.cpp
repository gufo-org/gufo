#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_asr/audio.hpp"

namespace qwen3_asr = gufo::models::qwen3_asr;

namespace {

using Bytes = std::vector<std::byte>;

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL qwen3_asr_decode_test: " << message << '\n';
    std::exit(1);
  }
}

void Append(Bytes& bytes, std::string_view text) {
  for (const char value : text) {
    bytes.push_back(static_cast<std::byte>(value));
  }
}

void AppendLe(Bytes& bytes, std::uint32_t value, std::size_t width) {
  for (std::size_t index = 0; index < width; ++index) {
    bytes.push_back(static_cast<std::byte>((value >> (8U * index)) & 0xFFU));
  }
}

// A RIFF WAV with one 16-byte fmt chunk and one data chunk.
Bytes Wav(std::uint16_t format, std::uint16_t channels, std::uint32_t rate,
          std::uint16_t bits, const Bytes& data) {
  const std::uint32_t block = channels * bits / 8U;
  Bytes bytes;
  Append(bytes, "RIFF");
  AppendLe(bytes, static_cast<std::uint32_t>(4U + 24U + 8U + data.size()), 4);
  Append(bytes, "WAVEfmt ");
  AppendLe(bytes, 16, 4);
  AppendLe(bytes, format, 2);
  AppendLe(bytes, channels, 2);
  AppendLe(bytes, rate, 4);
  AppendLe(bytes, rate * block, 4);
  AppendLe(bytes, block, 2);
  AppendLe(bytes, bits, 2);
  Append(bytes, "data");
  AppendLe(bytes, static_cast<std::uint32_t>(data.size()), 4);
  bytes.insert(bytes.end(), data.begin(), data.end());
  return bytes;
}

Bytes ReadFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "cannot open " + path.string());
  const std::vector<char> chars{std::istreambuf_iterator<char>(input),
                                std::istreambuf_iterator<char>()};
  Bytes bytes(chars.size());
  std::memcpy(bytes.data(), chars.data(), chars.size());
  return bytes;
}

std::string Rejection(const Bytes& bytes) {
  qwen3_asr::AudioBuffer audio;
  std::string error;
  Check(!qwen3_asr::DecodeWav(bytes, &audio, &error), "input was accepted");
  return error;
}

// PCM and float WAV keep the strict reader: exact integer scaling, floats
// clamped to [-1, 1], channels kept for ResampleMono16k to mix.
void TestWavUnchanged() {
  Bytes pcm;
  for (const std::int16_t value : {-32768, -1, 0, 1, 12345, 32767}) {
    for (int channel = 0; channel < 2; ++channel) {
      AppendLe(pcm, static_cast<std::uint16_t>(value), 2);
    }
  }
  qwen3_asr::AudioBuffer audio;
  std::string error;
  Check(qwen3_asr::DecodeWav(Wav(1, 2, 16000, 16, pcm), &audio, &error), error);
  Check(audio.sample_rate == 16000U && audio.channels == 2U &&
            audio.samples.size() == 12U,
        "PCM16 stereo layout");
  Check(audio.samples[0] == -1.0F && audio.samples[8] == 12345.0F / 32768.0F &&
            audio.samples[10] == 32767.0F / 32768.0F,
        "PCM16 scaling");

  Bytes floats;
  for (const float value : {1.5F, -2.0F, 0.25F}) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    AppendLe(floats, bits, 4);
  }
  Check(qwen3_asr::DecodeWav(Wav(3, 1, 48000, 32, floats), &audio, &error),
        error);
  Check(audio.samples == std::vector<float>({1.0F, -1.0F, 0.25F}),
        "float32 WAV is clamped as before");

  // A PCM WAV the strict reader rejects stays rejected with its reason.
  Bytes cut = Wav(1, 1, 16000, 16, Bytes(64));
  cut.resize(cut.size() - 10U);
  Check(Rejection(cut) == "Qwen3-ASR audio is truncated", "truncated WAV");
  Check(Rejection(Wav(1, 9, 16000, 16, Bytes(18))) ==
            "Qwen3-ASR audio must have 1 to 8 channels and at most 192 kHz",
        "too many channels");
  Check(Rejection(Wav(1, 1, 8000, 16, Bytes(8000U * 2U * (30U * 60U + 1U)))) ==
            "Qwen3-ASR audio must be at most 30 minutes",
        "too long");
}

// Sample widths the strict reader refuses go to the shared decoder, which
// mixes them to mono.
void TestOtherWav() {
  Bytes pcm8;
  for (const std::uint8_t value : {0U, 128U, 255U, 64U}) {
    pcm8.push_back(static_cast<std::byte>(value));
  }
  qwen3_asr::AudioBuffer audio;
  std::string error;
  Check(qwen3_asr::DecodeWav(Wav(1, 2, 22050, 8, pcm8), &audio, &error), error);
  Check(audio.sample_rate == 22050U && audio.channels == 1U &&
            audio.samples.size() == 2U,
        "PCM8 stereo decodes to mono");
  Check(audio.samples[0] == -0.5F, "PCM8 downmix");
}

void TestEncoded(const std::filesystem::path& speech_root,
                 const std::filesystem::path& decoder_root) {
  qwen3_asr::AudioBuffer wav;
  std::string error;
  Check(qwen3_asr::DecodeWav(ReadFile(speech_root / "clear-english-voice.wav"),
                             &wav, &error),
        error);
  const std::vector<float> reference = qwen3_asr::ResampleMono16k(wav);

  qwen3_asr::AudioBuffer mp3;
  Check(qwen3_asr::DecodeWav(ReadFile(speech_root / "clear-english-voice.mp3"),
                             &mp3, &error),
        error);
  Check(mp3.sample_rate == 16000U && mp3.channels == 1U, "MP3 layout");
  const std::vector<float> decoded = qwen3_asr::ResampleMono16k(mp3);
  // The encoder pads the stream by under a frame; the clip is 10 s.
  Check(decoded.size() >= reference.size() &&
            decoded.size() < reference.size() + 2304U,
        "MP3 length");
  // MP3 delays the signal by a fixed number of samples: the best lag must
  // line the clip up with the WAV almost perfectly.
  double best = 0.0;
  for (std::size_t lag = 0; lag < 2304U; ++lag) {
    double dot = 0.0;
    double left = 0.0;
    double right = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
      const double a = reference[index];
      const double b = index + lag < decoded.size() ? decoded[index + lag] : 0;
      dot += a * b;
      left += a * a;
      right += b * b;
    }
    best = std::max(best, dot / std::sqrt(left * right));
  }
  Check(best > 0.99, "MP3 waveform matches the WAV: " + std::to_string(best));

  qwen3_asr::AudioBuffer vorbis;
  Check(qwen3_asr::DecodeWav(ReadFile(decoder_root / "vorbis-first.ogg"),
                             &vorbis, &error),
        error);
  Check(vorbis.sample_rate == 48000U && vorbis.channels == 1U &&
            vorbis.samples.size() > 48000U * 2U,
        "Ogg Vorbis layout");

  Bytes text;
  Append(text, "not audio at all, just some bytes");
  Check(Rejection(text) ==
            "Qwen3-ASR input must be WAV, FLAC, MP3, Ogg Vorbis or Ogg Opus",
        "unknown input");
}

}  // namespace

int main() {
  TestWavUnchanged();
  TestOtherWav();
  TestEncoded(GUFO_QWEN3_ASR_SPEECH_FIXTURES, GUFO_AUDIO_DECODE_FIXTURES);
  std::cout << "PASS qwen3_asr_decode_test\n";
  return 0;
}
