// Tests core::DecodeAudio, DownmixMono and Resample on generated audio:
// libsndfile's float values for each WAV encoding, the strict PCM WAV path,
// the channel mean, also while decoding, the resampler's response and output
// length, and the status of each rejection. The Ogg Opus
// and MP3 fixtures were made with FFmpeg 8 (libopus, LAME 3.101) and -fflags
// +bitexact.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "src/core/audio.hpp"

namespace {

namespace core = gufo::core;
using Bytes = std::vector<std::uint8_t>;
using Status = core::AudioStatus;

// EmbeddingGemma 2's limits.
constexpr core::AudioLimits kAll{
    .formats = core::AudioFormats::kAll,
    .max_channels = 8,
    .min_sample_rate = 8000,
    .max_sample_rate = 192000,
    .max_seconds = 30,
};

void Check(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Bytes Hex(std::string_view hex) {
  Bytes bytes;
  const auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
  for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
    bytes.push_back(
        static_cast<std::uint8_t>(nibble(hex[i]) * 16 + nibble(hex[i + 1])));
  }
  return bytes;
}

// 20 ms of a 440 Hz sine at 48 kHz in Ogg Opus.
const Bytes kOpus =
    Hex("4f67675300020000000000000000b1fbc83900000000a879f1d601134f707573"
        "486561640101380180bb00000000004f67675300000000000000000000b1fbc8"
        "390100000065cb9a1601344f707573546167730c0000004c61766636322e332e"
        "3130300100000014000000656e636f6465723d4c617663206c69626f7075734f"
        "6767530004f804000000000000b1fbc8390200000095000f2e020d0c08836d82"
        "d01cfdea49fe013fc008066ed53019317066b99418");

// 1.66 s of a 440 Hz sine at 8 kHz, mono MP3 at 8 kbit/s without a
// Xing/Info header.
const Bytes kMp3 =
    Hex("ffe318c4000ca892a8794800000f6dbfe1dddddeffd1112bf97777f9492c1b80"
        "b80b80b8373e5031c3fca1cfdfd1efe8f7f070319007df5810e6807f94777f42"
        "81090a445881133fffe318c4080df0eaa0c1943800e4c90126873bfc4061c903"
        "6c1803814b8073c60a24a606b128d30841c83d1e1b0dbffc746a35381a12ff84"
        "81a129d06affffffffffffffffffffffffe318c40b0f10a63801de0000f86846"
        "000601600c5fa06808181a838184983e184e8bd189ade919000b2187e8351825"
        "8260080ec2a00e60200025b16497ead3fffef812a5502029ffe318c4090dc8a6"
        "2c0056bc60891467d01b272797d18800db9c77e879a7e0a9986a82f182500b94"
        "00f20441a0008054beb1ffffffdfe8fffffff4d5ffeffbc0b06880c2d120c07c"
        "ffe318c40c0e80a6300007b048034c158110c3cc4a4e252474d32c3acc340120"
        "c15408cc098068d0c008d13169e3ffffb3ffe9fffffffedfffabe8fffff93b7a"
        "9740e0130213029bffe318c40d0a58a6340006fc440321c123d06101d1060723"
        "00480ea1602e00807860019100226cb51beafffff8057917d40a00c78c3445ce"
        "8b830d80e737b173934480af30b50220ffe318c41e0a50a6340006bc44807d0c"
        "023411a3b294b22c2affe7fc30cad401c74e4300d019303a05630b515f3698aa"
        "333ad115309c054303d027301c01633180c34c061f8fffffffe318c42f0c50a6"
        "340007b048fffaeafffff8b34d41603029840b9859b18b4198258ed198a70d18"
        "d78b918190378140a04600e99e4400299ccd6fffffffffaafffff82585264a09"
        "ffe318c4380bf0a6340006fc448c38033804e38c30c6055372131733f8039070"
        "a70d03a01808c2e0029d8c259f6affeffc619da962de21002008139813037184"
        "58d29aa564c99788ffe318c4430a30a6380006bc44a5983d0331817816980c00"
        "b19880e3a56313b3fffff8d3394562cc9882263d41a4ca60de2f468956c464ba"
        "24c60ac0a660220309eacd924558996dffe318c4550b38a6340007b048b5f81e"
        "a3894a8dc0e00e1a041080ab30e6038382f25e349002e30c5007304a004301d0"
        "0b0280388800165b08d2fffff963aeb713584218653988d4ffe318c4630a20a6"
        "380006bc44608e32a664185e637227e60800c0602e04e601401c060001200352"
        "0dbdeafffff7d94c41200c30732c6cd2b4399dcc32c6a8dcc3308d004530c290"
        "ffe318c4750ae8a6340001fc40174c0d406048015608b92914b0b6fffffffbaa"
        "fffff80d7c265ada42e033439a9cc370320df5d990d178278c2f40b4c120050c"
        "07c044c00c024b50ffe318c4840ae0a6380006bc44a56c832affeffc81b3a669"
        "6f8020202003720086300b1e7314ce7f30b117f3010074300503130060100710"
        "696940d4ef7fffffffaefffff7492341ffe318c4930bd0a6300006bc44a18c30"
        "b33874dab03d5a4c4545e0e61ad60d54c488c398134c14c028480b0600082c00"
        "68264e5b7fffffffeafffffffedffffe8aff5fffc25a6817ffe318c49e0ae0a6"
        "340006bc449d8020b981000f182f83498830cb1c9fe1c1a950a11872833982f0"
        "1918188129be6662177d392d7fffffeafaffffeffffb7ffeaaffe7fc85b64710"
        "ffe318c4ad0c30a6340007b0485031838a1821b984c318130f298faf3498850b"
        "e980983910016830028bf05000c9e2d0af7ffffffffffffffffffffa6afffffe"
        "29300900618c80c0ffe318c4b70e60a62c0006bc44200c0304206c00418018a0"
        "1e006490f901e70760071f4b08186d08200c1080c0d00200c0081e88826191d7"
        "050408000217fffffff6430b7e20efffffe318c4b80ea0a62c0057b064fffef1"
        "bf5d0de2b1fc40d5cd97e66c0cd720535d070ba866e881077fd3b607301e00a3"
        "8c1873046ffdbf97c941ee665c2fa65cffc2e098902e2bffffe318c4b80d90a6"
        "340006fc44e2c2e2a2c2ecfffc585d82cb637ffff2ac0915b2e5df5ae1d1944d"
        "2efad602509511d1f5d6ba98050011353191f7b262627b09244926c56f656fb2"
        "ffe318c4bc0d78a63c01560000b5d631160a6fc5c457f9b1554c414d45332e31"
        "3031202862657461203329555555555555555555555555555555555555555555"
        "5555555555555555ffe318c4c117813a94f19b68005555555555555555555555"
        "5555555555555555555555555555555555555555555555555555555555555555"
        "55555555555555555555555555555555ffe318c49e0cd0c66801cc1801555555"
        "5555555555555555555555555555555555555555555555555555555555555555"
        "555555555555555555555555555555555555555555555555");

void Put(Bytes& bytes, std::uint64_t value, int size) {
  for (int i = 0; i < size; ++i) {
    bytes.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  }
}

// A RIFF WAV with one fmt chunk (with cbSize for non-PCM formats) and one
// data chunk that declares data_size bytes, by default those of data.
Bytes Wav(std::uint16_t format, std::uint16_t channels, std::uint32_t rate,
          std::uint16_t bits, const Bytes& data,
          std::uint32_t data_size = std::numeric_limits<std::uint32_t>::max()) {
  if (data_size == std::numeric_limits<std::uint32_t>::max()) {
    data_size = static_cast<std::uint32_t>(data.size());
  }
  const bool extended = format != 1;
  const std::uint16_t align = static_cast<std::uint16_t>(channels * bits / 8);
  Bytes wav = {'R', 'I', 'F', 'F'};
  Put(wav, 4 + 8 + (extended ? 18 : 16) + 8 + data.size(), 4);
  wav.insert(wav.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  Put(wav, extended ? 18 : 16, 4);
  Put(wav, format, 2);
  Put(wav, channels, 2);
  Put(wav, rate, 4);
  Put(wav, static_cast<std::uint64_t>(rate) * align, 4);
  Put(wav, align, 2);
  Put(wav, bits, 2);
  if (extended) {
    Put(wav, 0, 2);
  }
  wav.insert(wav.end(), {'d', 'a', 't', 'a'});
  Put(wav, data_size, 4);
  wav.insert(wav.end(), data.begin(), data.end());
  return wav;
}

constexpr std::uint16_t kPcm = 1;
constexpr std::uint16_t kFloat = 3;
constexpr std::uint16_t kALaw = 6;
constexpr std::uint16_t kMuLaw = 7;
constexpr std::uint16_t kGsm = 0x31;

// A WAVE_FORMAT_EXTENSIBLE WAV whose samples have `valid` of `bits` bits.
Bytes WavExtensible(std::uint16_t subformat, std::uint16_t channels,
                    std::uint16_t bits, std::uint16_t valid,
                    const Bytes& data) {
  const std::uint32_t rate = 16000;
  const std::uint16_t align = static_cast<std::uint16_t>(channels * bits / 8);
  Bytes wav = {'R', 'I', 'F', 'F'};
  Put(wav, 4 + 8 + 40 + 8 + data.size(), 4);
  wav.insert(wav.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  Put(wav, 40, 4);
  Put(wav, 0xfffe, 2);
  Put(wav, channels, 2);
  Put(wav, rate, 4);
  Put(wav, static_cast<std::uint64_t>(rate) * align, 4);
  Put(wav, align, 2);
  Put(wav, bits, 2);
  Put(wav, 22, 2);
  Put(wav, valid, 2);
  Put(wav, 0, 4);  // channel mask
  Put(wav, subformat, 2);
  wav.insert(wav.end(), {0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00,
                         0xaa, 0x00, 0x38, 0x9b, 0x71});
  wav.insert(wav.end(), {'d', 'a', 't', 'a'});
  Put(wav, data.size(), 4);
  wav.insert(wav.end(), data.begin(), data.end());
  return wav;
}

// G.711 to 16-bit linear PCM, as libsndfile's tables hold it.
std::int16_t ALaw(std::uint8_t byte) {
  const int value = byte ^ 0x55;
  const int exponent = (value >> 4) & 7;
  const int mantissa = value & 0x0f;
  const int magnitude = exponent == 0
                            ? mantissa * 16 + 8
                            : (mantissa * 16 + 0x108) << (exponent - 1);
  return static_cast<std::int16_t>((value & 0x80) != 0 ? magnitude
                                                       : -magnitude);
}

std::int16_t MuLaw(std::uint8_t byte) {
  const int value = ~byte & 0xff;
  const int exponent = (value >> 4) & 7;
  const int mantissa = value & 0x0f;
  const int magnitude = ((mantissa * 8 + 0x84) << exponent) - 0x84;
  return static_cast<std::int16_t>((value & 0x80) != 0 ? -magnitude
                                                       : magnitude);
}

Bytes Float32(const std::vector<float>& samples) {
  Bytes data;
  for (const float sample : samples) {
    Put(data, std::bit_cast<std::uint32_t>(sample), 4);
  }
  return data;
}

core::DecodedAudio Decode(const Bytes& bytes,
                          const core::AudioLimits& limits = kAll,
                          const std::function<bool()>& should_stop = {}) {
  return core::DecodeAudio(std::as_bytes(std::span(bytes)), limits,
                           should_stop);
}

void ExpectStatus(const std::string& name, const std::function<void()>& call,
                  Status status) {
  try {
    call();
  } catch (const core::AudioError& error) {
    Check(error.status() == status,
          name + ": status " +
              std::to_string(static_cast<int>(error.status())) + " (" +
              error.what() + "), expected " +
              std::to_string(static_cast<int>(status)));
    Check(!std::string_view(error.what()).empty(), name + ": empty message");
    return;
  }
  throw std::runtime_error(name + ": no AudioError");
}

template<typename Error>
void ExpectError(const std::string& name, const std::function<void()>& call) {
  try {
    call();
  } catch (const Error&) {
    return;
  }
  throw std::runtime_error(name + ": no exception of the expected type");
}

void ExpectEqual(const std::string& name, const std::vector<float>& actual,
                 const std::vector<float>& expected) {
  Check(actual.size() == expected.size(),
        name + ": length " + std::to_string(actual.size()) + ", expected " +
            std::to_string(expected.size()));
  for (std::size_t i = 0; i < expected.size(); ++i) {
    Check(std::bit_cast<std::uint32_t>(actual[i]) ==
              std::bit_cast<std::uint32_t>(expected[i]),
          name + ": sample " + std::to_string(i) + " is " +
              std::to_string(actual[i]) + ", expected " +
              std::to_string(expected[i]));
  }
}

void ExpectSamples(const std::string& name, const Bytes& wav,
                   const std::vector<float>& expected,
                   const core::AudioLimits& limits = kAll) {
  ExpectEqual(name, Decode(wav, limits).samples, expected);
}

// libsndfile's float view of each WAV encoding: integers over a power of two,
// the G.711 tables for A-law and mu-law.
void TestEncodings() {
  Bytes pcm16;
  for (const std::int16_t v : {0, 1, -1, 32767, -32768, 12345}) {
    Put(pcm16, static_cast<std::uint16_t>(v), 2);
  }
  const std::vector<float> pcm16_expected = {0.0f,          1 / 32768.0f,
                                             -1 / 32768.0f, 32767 / 32768.0f,
                                             -1.0f,         12345 / 32768.0f};
  ExpectSamples("PCM16", Wav(kPcm, 1, 16000, 16, pcm16), pcm16_expected);
  ExpectSamples("PCM8", Wav(kPcm, 1, 16000, 8, {128, 0, 255, 129}),
                {0.0f, -1.0f, 127 / 128.0f, 1 / 128.0f});
  Bytes pcm24;
  for (const std::int32_t v : {1, 8388607, -8388608, -2}) {
    Put(pcm24, static_cast<std::uint32_t>(v), 3);
  }
  const std::vector<float> pcm24_expected = {
      1 / 8388608.0f, 8388607 / 8388608.0f, -1.0f, -2 / 8388608.0f};
  ExpectSamples("PCM24", Wav(kPcm, 1, 16000, 24, pcm24), pcm24_expected);
  Bytes pcm32;
  const std::int32_t pcm32_values[] = {
      1, std::numeric_limits<std::int32_t>::max(),
      std::numeric_limits<std::int32_t>::min()};
  std::vector<float> pcm32_expected;
  for (const std::int32_t v : pcm32_values) {
    Put(pcm32, static_cast<std::uint32_t>(v), 4);
    pcm32_expected.push_back(static_cast<float>(v) / 2147483648.0f);
  }
  ExpectSamples("PCM32", Wav(kPcm, 1, 16000, 32, pcm32), pcm32_expected);
  const Bytes floats = Float32({0.5f, -0.25f, 1.5f});
  ExpectSamples("float32", Wav(kFloat, 1, 16000, 32, floats),
                {0.5f, -0.25f, 1.5f});
  Bytes float64;
  for (const double v : {0.1, -0.7}) {
    Put(float64, std::bit_cast<std::uint64_t>(v), 8);
  }
  ExpectSamples("float64", Wav(kFloat, 1, 16000, 64, float64),
                {static_cast<float>(0.1), static_cast<float>(-0.7)});
  ExpectSamples("mu-law", Wav(kMuLaw, 1, 16000, 8, {0x00, 0x80, 0x7f, 0xff}),
                {-32124 / 32768.0f, 32124 / 32768.0f, 0.0f, 0.0f});
  ExpectSamples(
      "A-law", Wav(kALaw, 1, 16000, 8, {0x55, 0xd5, 0x2a, 0xaa}),
      {-8 / 32768.0f, 8 / 32768.0f, -32256 / 32768.0f, 32256 / 32768.0f});

  // Every 16-bit, A-law and mu-law value, over several packets.
  Bytes every_pcm16;
  std::vector<float> every_pcm16_expected;
  for (int v = -32768; v <= 32767; ++v) {
    Put(every_pcm16, static_cast<std::uint16_t>(v), 2);
    every_pcm16_expected.push_back(static_cast<float>(v) / 32768.0f);
  }
  ExpectSamples("every PCM16 value", Wav(kPcm, 1, 16000, 16, every_pcm16),
                every_pcm16_expected);
  Bytes every_byte;
  std::vector<float> alaw_expected;
  std::vector<float> mulaw_expected;
  for (int byte = 0; byte < 256; ++byte) {
    every_byte.push_back(static_cast<std::uint8_t>(byte));
    alaw_expected.push_back(ALaw(static_cast<std::uint8_t>(byte)) / 32768.0f);
    mulaw_expected.push_back(MuLaw(static_cast<std::uint8_t>(byte)) / 32768.0f);
  }
  ExpectSamples("every A-law value", Wav(kALaw, 1, 16000, 8, every_byte),
                alaw_expected);
  ExpectSamples("every mu-law value", Wav(kMuLaw, 1, 16000, 8, every_byte),
                mulaw_expected);

  // WAVE_FORMAT_EXTENSIBLE scales by the written width, whatever the valid
  // bits say, and leaves floats unclamped.
  ExpectSamples("extensible PCM24, 20 valid bits",
                WavExtensible(kPcm, 1, 24, 20, pcm24), pcm24_expected);
  ExpectSamples("extensible PCM16, 12 valid bits",
                WavExtensible(kPcm, 1, 16, 12, pcm16), pcm16_expected);
  ExpectSamples("extensible float32", WavExtensible(kFloat, 1, 32, 32, floats),
                {0.5f, -0.25f, 1.5f});

  // The strict path decodes the same integers, clamps floats to [-1, 1] and
  // takes PCM24 only when max_pcm_bits allows it.
  const core::AudioLimits pcm_wav{};
  ExpectSamples("strict PCM16", Wav(kPcm, 1, 16000, 16, pcm16), pcm16_expected,
                pcm_wav);
  ExpectSamples("strict PCM24", Wav(kPcm, 1, 16000, 24, pcm24), pcm24_expected,
                {.max_pcm_bits = 24});
  ExpectSamples("strict float32", Wav(kFloat, 1, 16000, 32, floats),
                {0.5f, -0.25f, 1.0f}, pcm_wav);
}

// DecodeAudio keeps the channels interleaved; DownmixMono takes their float
// sum in order and divides by their count.
void TestChannels() {
  Bytes stereo;
  for (const std::int16_t v : {1000, -200, -3, 7}) {
    Put(stereo, static_cast<std::uint16_t>(v), 2);
  }
  for (const auto& limits : {kAll, core::AudioLimits{}}) {
    const auto audio = Decode(Wav(kPcm, 2, 16000, 16, stereo), limits);
    Check(audio.sample_rate == 16000 && audio.channels == 2, "stereo info");
    ExpectEqual(
        "stereo", audio.samples,
        {1000 / 32768.0f, -200 / 32768.0f, -3 / 32768.0f, 7 / 32768.0f});
    ExpectEqual("stereo mean", core::DownmixMono(audio.samples, 2),
                {(1000 / 32768.0f + -200 / 32768.0f) / 2,
                 (-3 / 32768.0f + 7 / 32768.0f) / 2});
  }
  ExpectEqual("three channels",
              core::DownmixMono(std::vector<float>{0.1f, 0.2f, 0.4f}, 3),
              {(0.1f + 0.2f + 0.4f) / 3});
  ExpectError<std::invalid_argument>("no channels", [] {
    (void)core::DownmixMono(std::vector<float>{0.1f}, 0);
  });
  ExpectError<std::invalid_argument>("partial frame", [] {
    (void)core::DownmixMono(std::vector<float>{0.1f, 0.2f, 0.3f}, 2);
  });
}

// downmix_mono gives DownmixMono's samples, block by block while decoding,
// on both paths.
void TestDownmixWhileDecoding() {
  Bytes data;
  for (int n = 0; n < 3000 * 3; ++n) {
    Put(data, static_cast<std::uint16_t>(n * 37), 2);
  }
  // 3,000 frames span packets of symphonia's 1,152 frames; the strict path
  // reads blocks of 16,384 frames.
  const Bytes three = Wav(kPcm, 3, 16000, 16, data);
  Bytes long_data;
  for (int n = 0; n < 40000 * 2; ++n) {
    Put(long_data, static_cast<std::uint16_t>(n * 101), 2);
  }
  const Bytes long_stereo = Wav(kPcm, 2, 16000, 16, long_data);
  const core::AudioLimits pcm_wav{};
  for (const auto& [name, bytes, limits] :
       {std::tuple{"three channels", three, kAll},
        std::tuple{"MP3", kMp3, kAll},
        std::tuple{"strict three channels", three, pcm_wav},
        std::tuple{"strict long stereo", long_stereo, pcm_wav}}) {
    const auto whole = Decode(bytes, limits);
    auto mono_limits = limits;
    mono_limits.downmix_mono = true;
    const auto mono = Decode(bytes, mono_limits);
    Check(mono.sample_rate == whole.sample_rate && mono.channels == 1,
          std::string(name) + ": mono info");
    ExpectEqual(std::string(name) + ": mono", mono.samples,
                core::DownmixMono(whole.samples, whole.channels));
  }
}

std::vector<float> Sine(std::uint32_t rate, double frequency, double seconds) {
  std::vector<float> samples(static_cast<std::size_t>(rate * seconds));
  for (std::size_t n = 0; n < samples.size(); ++n) {
    samples[n] = static_cast<float>(std::sin(2 * std::numbers::pi * frequency *
                                             static_cast<double>(n) / rate));
  }
  return samples;
}

// Sines through the resampler: flat and in phase up to 0.85 of the lower
// Nyquist frequency, gone above it. The first and last eighth see the zeros
// around the input and are left out.
void TestResampler() {
  for (const std::uint32_t rate : {8000u, 11025u, 22050u, 24000u, 32000u,
                                   44100u, 48000u, 96000u, 192000u}) {
    for (const std::uint32_t out_rate : {16000u, 48000u}) {
      if (rate == out_rate) {
        continue;
      }
      const double edge = std::min(rate, out_rate) / 2.0;
      for (const double frequency : {1000.0, 0.85 * edge}) {
        const auto input = Sine(rate, frequency, 0.25);
        const auto output = core::Resample(input, rate, out_rate);
        const std::string name = std::to_string(rate) + " Hz to " +
                                 std::to_string(out_rate) + " Hz, " +
                                 std::to_string(frequency) + " Hz";
        // soxr_oneshot's length: rounded half up.
        const auto expected_length = static_cast<std::size_t>(
            static_cast<double>(input.size()) /
                (static_cast<double>(rate) / out_rate) +
            0.5);
        Check(output.size() == expected_length, name + ": length");
        double worst = 0;
        const std::size_t margin = output.size() / 8;
        for (std::size_t n = margin; n < output.size() - margin; ++n) {
          const double ideal = std::sin(2 * std::numbers::pi * frequency *
                                        static_cast<double>(n) / out_rate);
          worst = std::max(worst, std::abs(output[n] - ideal));
        }
        Check(worst < 1e-4, name + ": passband error " + std::to_string(worst));
      }
      if (rate <= out_rate) {
        continue;
      }
      for (const double frequency : {1.02 * out_rate / 2, 0.98 * rate / 2}) {
        const auto output =
            core::Resample(Sine(rate, frequency, 0.25), rate, out_rate);
        const std::size_t margin = output.size() / 8;
        double energy = 0;
        for (std::size_t n = margin; n < output.size() - margin; ++n) {
          energy += static_cast<double>(output[n]) * output[n];
        }
        const double rms = std::sqrt(energy / (output.size() - 2 * margin));
        Check(rms < 1e-5, std::to_string(rate) + " Hz to " +
                              std::to_string(out_rate) + " Hz, " +
                              std::to_string(frequency) + " Hz: stopband rms " +
                              std::to_string(rms));
      }
    }
  }
  const std::vector<float> few = {0.25f, -0.5f, 0.75f};
  ExpectEqual("equal rates copy", core::Resample(few, 16000, 16000), few);
  Check(core::Resample({}, 44100, 16000).empty(), "empty input");
  ExpectError<std::invalid_argument>(
      "rate 0", [&] { (void)core::Resample(few, 0, 16000); });
  ExpectError<std::invalid_argument>(
      "ratio above 64", [&] { (void)core::Resample(few, 16000 * 65, 16000); });
}

void TestMp3() {
  const auto audio = Decode(kMp3);
  Check(audio.sample_rate == 8000 && audio.channels == 1 &&
            audio.samples.size() > 12000,
        "MP3 decodes every frame");
  Check(core::Resample(audio.samples, 8000, 16000).size() ==
            2 * audio.samples.size(),
        "MP3 doubles to 16 kHz");
  ExpectStatus(
      "MP3 past max_seconds",
      [] {
        auto limits = kAll;
        limits.max_seconds = 1;
        (void)Decode(kMp3, limits);
      },
      Status::kTooLong);
}

// Ogg Opus decodes at the rate its header asks for (48 kHz here), as long as
// the last granule position less the pre-skip: 1272 - 312 samples.
void TestOpus() {
  const auto audio = Decode(kOpus);
  Check(audio.sample_rate == 48000 && audio.channels == 1 &&
            audio.samples.size() == 960,
        "Opus: 20 ms at 48 kHz, " + std::to_string(audio.samples.size()) +
            " samples");
  double energy = 0;
  for (const float sample : audio.samples) {
    energy += static_cast<double>(sample) * sample;
  }
  Check(energy > 1.0, "Opus: energy " + std::to_string(energy));
  ExpectStatus(
      "Opus on the strict path",
      [] { (void)Decode(kOpus, core::AudioLimits{}); }, Status::kUnrecognized);
}

// libsndfile returns only the first physical stream of a chained Ogg file.
// Symphonia signals the next stream with ResetRequired, rather than EOF.
void TestChainedVorbis() {
  const auto fixture = [](const char* name) {
    std::ifstream input(std::string(GUFO_AUDIO_DECODE_FIXTURES) + "/" + name,
                        std::ios::binary);
    Check(input.good(), std::string("missing audio fixture: ") + name);
    return Bytes(std::istreambuf_iterator<char>(input), {});
  };
  const Bytes first = fixture("vorbis-first.ogg");
  const Bytes second = fixture("vorbis-second.ogg");
  const auto expected = Decode(first);
  Check(expected.sample_rate == 48000 && expected.channels == 1 &&
            expected.samples.size() == 105600,
        "first Vorbis stream: 2.2 seconds at 48 kHz");
  Check(Decode(second).samples.size() == 9600,
        "second Vorbis stream: 0.2 seconds at 48 kHz");
  Bytes chained = first;
  chained.insert(chained.end(), second.begin(), second.end());
  const auto actual = Decode(chained);
  Check(actual.sample_rate == expected.sample_rate &&
            actual.channels == expected.channels,
        "chained Vorbis: first stream info");
  ExpectEqual("chained Vorbis: first stream samples", actual.samples,
              expected.samples);

  // Damage the middle audio page while retaining its final EOS page and
  // declared frame count. A subsequent valid stream must not hide the loss.
  chained[first.size() * 3 / 4] ^= 1;
  ExpectStatus(
      "damaged first Vorbis stream", [&] { (void)Decode(chained); },
      Status::kTruncated);
}

void TestRejections() {
  const Bytes pcm16(320, 0);
  ExpectStatus("empty", [] { (void)Decode({}); }, Status::kUnrecognized);
  ExpectStatus(
      "text", [] { (void)Decode(Bytes(400, 'x')); }, Status::kUnrecognized);
  // symphonia's WAV reader rejects a format tag it has no codec for.
  ExpectStatus(
      "GSM 6.10", [] { (void)Decode(Wav(kGsm, 1, 8000, 0, Bytes(65 * 4, 0))); },
      Status::kUnrecognized);
  ExpectStatus(
      "nine channels",
      [&] { (void)Decode(Wav(kPcm, 9, 16000, 16, Bytes(9 * 2 * 10, 0))); },
      Status::kOutOfBounds);
  ExpectStatus(
      "7999 Hz", [&] { (void)Decode(Wav(kPcm, 1, 7999, 16, pcm16)); },
      Status::kOutOfBounds);
  ExpectStatus(
      "192001 Hz", [&] { (void)Decode(Wav(kPcm, 1, 192001, 16, pcm16)); },
      Status::kOutOfBounds);
  ExpectStatus(
      "31 seconds",
      [] { (void)Decode(Wav(kPcm, 1, 8000, 16, Bytes(8000 * 31 * 2, 0))); },
      Status::kTooLong);
  ExpectStatus(
      "NaN",
      [] {
        (void)Decode(
            Wav(kFloat, 1, 16000, 32,
                Float32({0.0f, std::numeric_limits<float>::quiet_NaN()})));
      },
      Status::kNonFinite);
  ExpectStatus(
      "cut off",
      [&] {
        auto wav = Wav(kPcm, 1, 16000, 16, pcm16);
        wav.resize(wav.size() - 100);
        (void)Decode(wav);
      },
      Status::kTruncated);
  ExpectStatus(
      "data chunk past the RIFF chunk",
      [&] { (void)Decode(Wav(kPcm, 1, 16000, 16, pcm16, 2 * pcm16.size())); },
      Status::kUnrecognized);
  ExpectStatus(
      "stopped before decoding",
      [&] {
        (void)Decode(Wav(kPcm, 1, 16000, 16, pcm16), kAll, [] { return true; });
      },
      Status::kStopped);
  int calls = 0;
  ExpectStatus(
      "stopped while decoding",
      [&] { (void)Decode(kMp3, kAll, [&] { return ++calls > 3; }); },
      Status::kStopped);
  ExpectError<std::domain_error>("should_stop throws", [&] {
    (void)Decode(Wav(kPcm, 1, 16000, 16, pcm16), kAll,
                 []() -> bool { throw std::domain_error("from should_stop"); });
  });
}

// The strict path's own rejections, with the statuses serve, Qwen3-ASR and
// Qwen3-TTS map to their messages.
void TestPcmWavRejections() {
  const Bytes pcm16(320, 0);
  const core::AudioLimits limits{};
  ExpectStatus(
      "strict: not RIFF",
      [&] {
        auto wav = Wav(kPcm, 1, 16000, 16, pcm16);
        wav[0] = 'f';
        (void)Decode(wav, limits);
      },
      Status::kUnrecognized);
  ExpectStatus(
      "strict: mu-law",
      [&] { (void)Decode(Wav(kMuLaw, 1, 16000, 8, pcm16), limits); },
      Status::kUnsupportedCodec);
  ExpectStatus(
      "strict: PCM24 above max_pcm_bits",
      [&] { (void)Decode(Wav(kPcm, 1, 16000, 24, Bytes(300, 0)), limits); },
      Status::kUnsupportedLayout);
  ExpectStatus(
      "strict: nine channels",
      [&] {
        (void)Decode(Wav(kPcm, 9, 16000, 16, Bytes(9 * 2 * 10, 0)), limits);
      },
      Status::kOutOfBounds);
  ExpectStatus(
      "strict: below min_sample_rate",
      [&] {
        (void)Decode(Wav(kPcm, 1, 7999, 16, pcm16), {.min_sample_rate = 8000});
      },
      Status::kOutOfBounds);
  ExpectStatus(
      "strict: no data",
      [&] { (void)Decode(Wav(kPcm, 1, 16000, 16, {}), limits); },
      Status::kNoAudio);
  ExpectStatus(
      "strict: short fmt chunk",
      [&] {
        auto wav = Wav(kPcm, 1, 16000, 16, pcm16);
        wav[16] = 14;  // fmt chunk size
        (void)Decode(wav, limits);
      },
      Status::kMalformedHeader);
  ExpectStatus(
      "strict: 31 seconds",
      [] {
        (void)Decode(Wav(kPcm, 1, 8000, 16, Bytes(8000 * 31 * 2, 0)),
                     core::AudioLimits{});
      },
      Status::kTooLong);
  // Blocks of 16,384 frames: a stop on the second poll ends a longer WAV.
  int calls = 0;
  ExpectStatus(
      "strict: stopped while decoding",
      [&] {
        (void)Decode(Wav(kPcm, 1, 16000, 16, Bytes(20000 * 2, 0)), limits,
                     [&] { return ++calls > 2; });
      },
      Status::kStopped);
}

}  // namespace

int main() {
  try {
    TestEncodings();
    TestChannels();
    TestDownmixWhileDecoding();
    TestResampler();
    TestMp3();
    TestOpus();
    TestChainedVorbis();
    TestRejections();
    TestPcmWavRejections();
    std::cout << "audio decoding passes\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
