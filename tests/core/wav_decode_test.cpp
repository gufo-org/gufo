// Compares core::DecodeAudio's strict WAV path, under each caller's limits
// and status messages, with frozen copies of the three C++ decoders it
// replaced (tests/core/wav_reference.cpp): the same samples bit for bit, or
// the same message. The callers report only the message.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/audio.hpp"

namespace gufo::core::wav_reference {
namespace serve {
DecodedAudio DecodeWavBytes(std::span<const std::byte> bytes);
}
namespace asr {
DecodedAudio DecodeWavBytes(std::span<const std::byte> bytes);
}
namespace tts {
DecodedAudio DecodeWavBytes(std::span<const std::byte> bytes);
}
}  // namespace gufo::core::wav_reference

namespace {

namespace core = gufo::core;
using Bytes = std::vector<std::uint8_t>;

// The messages each caller reports.
struct Messages {
  std::string_view not_riff;
  std::string_view truncated;
  std::string_view invalid_fmt;
  std::string_view unsupported_format;
  std::string_view unsupported_layout;
  std::string_view too_long;
  std::string_view non_finite;
};

// The status-to-message switch of every caller.
std::string_view Message(const Messages& messages, core::AudioStatus status) {
  switch (status) {
    case core::AudioStatus::kUnrecognized:
      return messages.not_riff;
    case core::AudioStatus::kMalformedHeader:
      return messages.invalid_fmt;
    case core::AudioStatus::kUnsupportedCodec:
    case core::AudioStatus::kOutOfBounds:
    case core::AudioStatus::kNoAudio:
      return messages.unsupported_format;
    case core::AudioStatus::kUnsupportedLayout:
      return messages.unsupported_layout;
    case core::AudioStatus::kTooLong:
      return messages.too_long;
    case core::AudioStatus::kTruncated:
      return messages.truncated;
    case core::AudioStatus::kNonFinite:
      return messages.non_finite;
    case core::AudioStatus::kStopped:
      break;
  }
  return "unexpected status";
}

// The limits and messages of each caller, as in src/cli/serve/audio_buffer.cpp,
// src/models/qwen3_asr/audio.cpp and src/models/qwen3_tts/audio.cpp.
struct Caller {
  std::string_view name;
  core::AudioLimits limits;
  Messages messages;
  core::DecodedAudio (*reference)(std::span<const std::byte>);
};

const Caller kCallers[] = {
    {"serve",
     core::AudioLimits{},
     {
         .not_riff = "reference audio must be a RIFF WAV",
         .truncated = "reference WAV is truncated",
         .invalid_fmt = "reference WAV fmt is invalid",
         .unsupported_format = "reference WAV format is unsupported",
         .unsupported_layout = "supports PCM16 or float32 interleaved WAV",
         .too_long = "reference WAV must be at most 30 seconds",
         .non_finite = "reference WAV contains non-finite samples",
     },
     &core::wav_reference::serve::DecodeWavBytes},
    {"qwen3_asr",
     {
         .max_channels = 8,
         .max_sample_rate = 192000,
         .max_seconds = 30 * 60,
         .max_pcm_bits = 32,
     },
     {
         .not_riff = "Qwen3-ASR input must be a RIFF WAV",
         .truncated = "Qwen3-ASR WAV is truncated",
         .invalid_fmt = "Qwen3-ASR WAV fmt chunk is invalid",
         .unsupported_format = "Qwen3-ASR WAV format is unsupported",
         .unsupported_layout =
             "Qwen3-ASR supports PCM16/24/32 or float32 interleaved WAV",
         .too_long = "Qwen3-ASR WAV must be at most 30 minutes",
         .non_finite = "Qwen3-ASR WAV contains non-finite samples",
     },
     &core::wav_reference::asr::DecodeWavBytes},
    {"qwen3_tts",
     core::AudioLimits{},
     {
         .not_riff = "Qwen3-TTS reference audio must be a RIFF WAV",
         .truncated = "Qwen3-TTS reference WAV is truncated",
         .invalid_fmt = "Qwen3-TTS reference WAV fmt is invalid",
         .unsupported_format = "Qwen3-TTS reference WAV format is unsupported",
         .unsupported_layout =
             "Qwen3-TTS supports PCM16 or float32 interleaved WAV",
         .too_long = "Qwen3-TTS reference WAV must be at most 30 seconds",
         .non_finite = "Qwen3-TTS reference WAV contains non-finite samples",
     },
     &core::wav_reference::tts::DecodeWavBytes},
};

struct Outcome {
  enum class Kind { kOk, kRejected, kOther };
  Kind kind{Kind::kOther};
  std::string message;
  core::DecodedAudio audio;
};

// A frozen decoder rejects with std::invalid_argument or std::length_error.
Outcome Reference(const Caller& caller, std::span<const std::byte> bytes) {
  Outcome outcome;
  try {
    outcome.audio = caller.reference(bytes);
    outcome.kind = Outcome::Kind::kOk;
  } catch (const std::length_error& error) {
    outcome.kind = Outcome::Kind::kRejected;
    outcome.message = error.what();
  } catch (const std::invalid_argument& error) {
    outcome.kind = Outcome::Kind::kRejected;
    outcome.message = error.what();
  } catch (const std::exception& error) {
    outcome.message = error.what();
  }
  return outcome;
}

Outcome Decode(const Caller& caller, std::span<const std::byte> bytes) {
  Outcome outcome;
  try {
    outcome.audio = core::DecodeAudio(bytes, caller.limits);
    outcome.kind = Outcome::Kind::kOk;
  } catch (const core::AudioError& error) {
    outcome.kind = Outcome::Kind::kRejected;
    outcome.message = Message(caller.messages, error.status());
  } catch (const std::exception& error) {
    outcome.message = error.what();
  }
  return outcome;
}

bool SameSamples(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
}

std::string Describe(const Outcome& outcome) {
  switch (outcome.kind) {
    case Outcome::Kind::kOk:
      return "ok " + std::to_string(outcome.audio.sample_rate) + " Hz, " +
             std::to_string(outcome.audio.channels) + " channels, " +
             std::to_string(outcome.audio.samples.size()) + " samples";
    case Outcome::Kind::kRejected:
      return "rejected: " + outcome.message;
    case Outcome::Kind::kOther:
      break;
  }
  return "other exception: " + outcome.message;
}

bool Same(const Outcome& a, const Outcome& b) {
  if (a.kind != b.kind || a.message != b.message) {
    return false;
  }
  return a.kind != Outcome::Kind::kOk ||
         (a.audio.sample_rate == b.audio.sample_rate &&
          a.audio.channels == b.audio.channels &&
          SameSamples(a.audio.samples, b.audio.samples));
}

std::string Difference(const Outcome& expected, const Outcome& actual) {
  std::string text =
      "expected " + Describe(expected) + ", got " + Describe(actual);
  if (expected.kind == Outcome::Kind::kOk &&
      actual.kind == Outcome::Kind::kOk &&
      expected.audio.samples.size() == actual.audio.samples.size()) {
    for (std::size_t index = 0; index < expected.audio.samples.size();
         ++index) {
      const float want = expected.audio.samples[index];
      const float got = actual.audio.samples[index];
      if (std::bit_cast<std::uint32_t>(want) !=
          std::bit_cast<std::uint32_t>(got)) {
        text += "; sample " + std::to_string(index) + " is " +
                std::to_string(got) + " instead of " + std::to_string(want);
        break;
      }
    }
  }
  return text;
}

// ---- WAV builders ---------------------------------------------------------

void Append16(Bytes& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void Append32(Bytes& out, std::uint32_t value) {
  Append16(out, value & 0xFFFFU);
  Append16(out, value >> 16U);
}

Bytes Chunk(std::string_view id, const Bytes& body, bool pad = true) {
  Bytes out(id.begin(), id.end());
  Append32(out, static_cast<std::uint32_t>(body.size()));
  out.insert(out.end(), body.begin(), body.end());
  if (pad && body.size() % 2 == 1) {
    out.push_back(0);
  }
  return out;
}

struct Fmt {
  std::uint32_t format{1};
  std::uint32_t channels{1};
  std::uint32_t sample_rate{16000};
  std::uint32_t bits{16};
  // Zero derives block_align from channels and bits.
  std::uint32_t block_align{0};
  Bytes extension{};
};

Bytes FmtChunk(const Fmt& fmt) {
  const std::uint32_t block_align =
      fmt.block_align != 0 ? fmt.block_align : fmt.channels * (fmt.bits / 8);
  Bytes body;
  Append16(body, fmt.format);
  Append16(body, fmt.channels);
  Append32(body, fmt.sample_rate);
  Append32(body, fmt.sample_rate * block_align);
  Append16(body, block_align);
  Append16(body, fmt.bits);
  body.insert(body.end(), fmt.extension.begin(), fmt.extension.end());
  return Chunk("fmt ", body);
}

Bytes Riff(const std::vector<Bytes>& chunks) {
  Bytes out{'R', 'I', 'F', 'F'};
  std::size_t size = 4;
  for (const Bytes& chunk : chunks) {
    size += chunk.size();
  }
  Append32(out, static_cast<std::uint32_t>(size));
  out.insert(out.end(), {'W', 'A', 'V', 'E'});
  for (const Bytes& chunk : chunks) {
    out.insert(out.end(), chunk.begin(), chunk.end());
  }
  return out;
}

Bytes Wav(const Fmt& fmt, const Bytes& data) {
  return Riff({FmtChunk(fmt), Chunk("data", data)});
}

Bytes RandomBytes(std::mt19937& rng, std::size_t size) {
  std::uniform_int_distribution<int> byte(0, 255);
  Bytes out(size);
  for (auto& value : out) {
    value = static_cast<std::uint8_t>(byte(rng));
  }
  return out;
}

Bytes FloatData(std::initializer_list<float> values) {
  Bytes out;
  for (const float value : values) {
    Append32(out, std::bit_cast<std::uint32_t>(value));
  }
  return out;
}

// ---- Cases ----------------------------------------------------------------

struct Case {
  std::string name;
  Bytes bytes;
};

void AddFormatGrid(std::vector<Case>& cases) {
  std::mt19937 rng(1151);
  for (const std::uint32_t format : {0U, 1U, 2U, 3U, 0xFFFEU}) {
    for (const std::uint32_t bits : {0U, 8U, 12U, 16U, 24U, 32U, 64U}) {
      for (const std::uint32_t channels : {0U, 1U, 2U, 8U, 9U}) {
        for (const std::uint32_t rate :
             {0U, 1U, 8000U, 44100U, 192000U, 192001U}) {
          const std::size_t frame = std::max(1U, channels * bits / 8);
          cases.push_back(
              {"grid-" + std::to_string(format) + "-" + std::to_string(bits) +
                   "-" + std::to_string(channels) + "-" + std::to_string(rate),
               Wav({.format = format,
                    .channels = channels,
                    .sample_rate = rate,
                    .bits = bits},
                   RandomBytes(rng, 5 * frame))});
        }
      }
    }
  }
}

void AddSampleCases(std::vector<Case>& cases) {
  Bytes pcm16;
  for (const std::uint32_t value :
       {0x8000U, 0x8001U, 0xFFFFU, 0U, 1U, 0x7FFFU}) {
    Append16(pcm16, value);
  }
  cases.push_back({"pcm16-extremes", Wav({.channels = 2}, pcm16)});
  cases.push_back(
      {"pcm24-extremes",
       Wav({.channels = 2, .bits = 24}, {0x00, 0x00, 0x80, 0xFF, 0xFF, 0x7F,
                                         0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00})});
  Bytes pcm32;
  for (const std::uint32_t value :
       {0x80000000U, 0x7FFFFFFFU, 0xFFFFFFFFU, 1U}) {
    Append32(pcm32, value);
  }
  cases.push_back({"pcm32-extremes", Wav({.channels = 4, .bits = 32}, pcm32)});
  const float max = std::numeric_limits<float>::max();
  const float denormal = std::numeric_limits<float>::denorm_min();
  cases.push_back(
      {"float-values", Wav({.format = 3, .channels = 2, .bits = 32},
                           FloatData({0.5F, -0.0F, 1.5F, -2.0F, max, -max,
                                      denormal, -denormal}))});
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const float bad : {nan, inf, -inf}) {
    cases.push_back(
        {"float-non-finite-" + std::to_string(bad),
         Wav({.format = 3, .bits = 32}, FloatData({0.25F, bad, 0.5F, 0.75F}))});
  }
}

void AddStructureCases(std::vector<Case>& cases) {
  const Bytes samples{0x00, 0x40, 0x00, 0xC0};
  const Bytes fmt = FmtChunk({});
  const Bytes data = Chunk("data", samples);
  const Bytes odd = Chunk("LIST", {'a', 'b', 'c'});
  Bytes unpadded = Riff({fmt, data, Chunk("LIST", {'a', 'b', 'c'}, false)});
  cases.push_back({"data-before-fmt", Riff({data, odd, fmt})});
  cases.push_back({"unpadded-last-chunk", unpadded});
  cases.push_back({"two-fmt-chunks",
                   Riff({FmtChunk({.format = 3, .bits = 32}), data, fmt})});
  cases.push_back(
      {"two-data-chunks", Riff({fmt, data, Chunk("data", {1, 2})})});
  cases.push_back({"no-data", Riff({fmt, odd, odd})});
  cases.push_back({"empty-data", Riff({fmt, Chunk("data", {}), odd})});
  cases.push_back({"no-fmt", Riff({data, odd, odd})});
  cases.push_back(
      {"fmt-15-bytes", Riff({Chunk("fmt ", Bytes(15, 1)), data, odd})});
  cases.push_back(
      {"fmt-18-bytes", Riff({FmtChunk({.extension = {0, 0}}), data})});
  cases.push_back(
      {"fmt-extensible",
       Riff(
           {FmtChunk({.format = 0xFFFE, .bits = 16, .extension = Bytes(24, 0)}),
            data})});
  cases.push_back({"block-align-mismatch", Wav({.block_align = 4}, samples)});
  cases.push_back({"block-align-zero", [&] {
                     Bytes bytes = Wav({}, samples);
                     bytes[32] = 0;  // block_align
                     bytes[33] = 0;
                     return bytes;
                   }()});
  cases.push_back(
      {"misaligned-data", Wav({.channels = 2}, {0, 0, 0, 0, 0, 0})});
  cases.push_back({"riff-size-ignored", [&] {
                     Bytes bytes = Riff({fmt, data, odd});
                     bytes[4] = 0xFF;
                     return bytes;
                   }()});
  cases.push_back({"rifx", [&] {
                     Bytes bytes = Riff({fmt, data, odd});
                     bytes[3] = 'X';
                     return bytes;
                   }()});
  cases.push_back({"not-wave", [&] {
                     Bytes bytes = Riff({fmt, data, odd});
                     bytes[8] = 'A';
                     return bytes;
                   }()});
  cases.push_back({"too-short", [&] {
                     Bytes bytes = Riff({fmt, data});
                     bytes.resize(43);
                     return bytes;
                   }()});
  cases.push_back({"empty", {}});
  for (std::size_t cut = 1; cut <= 8; ++cut) {
    Bytes bytes = Riff({fmt, odd, data});
    bytes.resize(bytes.size() - cut);
    cases.push_back({"truncated-" + std::to_string(cut), bytes});
  }
  cases.push_back({"trailing-bytes", [&] {
                     Bytes bytes = Riff({fmt, data});
                     bytes.insert(bytes.end(), {'j', 'u', 'n', 'k', 0});
                     return bytes;
                   }()});
  cases.push_back({"huge-chunk", [&] {
                     Bytes bytes = Riff({fmt, odd, data});
                     Bytes huge{'J', 'U', 'N', 'K', 0xFF, 0xFF, 0xFF, 0xFF};
                     bytes.insert(bytes.begin() + 12, huge.begin(), huge.end());
                     return bytes;
                   }()});
}

void AddLengthCases(std::vector<Case>& cases) {
  // At one sample per second, the 30-second and 30-minute limits are short.
  for (const std::uint32_t frames : {29U, 30U, 31U, 1799U, 1800U, 1801U}) {
    for (const std::uint32_t channels : {1U, 8U}) {
      cases.push_back(
          {"length-" + std::to_string(frames) + "x" + std::to_string(channels),
           Wav({.channels = channels, .sample_rate = 1},
               Bytes(static_cast<std::size_t>(frames) * channels * 2, 0x11))});
    }
  }
}

void AddMutations(std::vector<Case>& cases) {
  std::mt19937 rng(20261108);
  const std::vector<Bytes> seeds = {
      Riff({FmtChunk({.channels = 2}), Chunk("LIST", {'a', 'b', 'c'}),
            Chunk("data", RandomBytes(rng, 64))}),
      Wav({.channels = 3, .sample_rate = 48000, .bits = 24},
          RandomBytes(rng, 90)),
      Wav({.channels = 2, .sample_rate = 44100, .bits = 32},
          RandomBytes(rng, 64)),
      Wav({.format = 3, .channels = 1, .bits = 32},
          FloatData({0.1F, -0.2F, 0.3F, 2.0F, -0.5F, 0.0F})),
  };
  std::uniform_int_distribution<int> byte(0, 255);
  for (std::size_t seed = 0; seed < seeds.size(); ++seed) {
    for (int mutation = 0; mutation < 2000; ++mutation) {
      Bytes bytes = seeds[seed];
      const int edits = 1 + mutation % 3;
      for (int edit = 0; edit < edits; ++edit) {
        std::uniform_int_distribution<std::size_t> position(0,
                                                            bytes.size() - 1);
        switch ((mutation + edit) % 4) {
          case 0:
          case 1:
            bytes[position(rng)] = static_cast<std::uint8_t>(byte(rng));
            break;
          case 2:
            bytes.resize(position(rng) + 1);
            break;
          default:
            // Rewrite a byte of a chunk header, where the walk is decided.
            bytes[std::min<std::size_t>(bytes.size() - 1,
                                        12 + position(rng) % 32)] =
                static_cast<std::uint8_t>(byte(rng));
            break;
        }
      }
      cases.push_back(
          {"mutation-" + std::to_string(seed) + "-" + std::to_string(mutation),
           bytes});
    }
  }
}

std::vector<Case> BuildCases() {
  std::vector<Case> cases;
  AddFormatGrid(cases);
  AddSampleCases(cases);
  AddStructureCases(cases);
  AddLengthCases(cases);
  AddMutations(cases);
  return cases;
}

std::span<const std::byte> AsBytes(const Bytes& bytes) {
  return std::as_bytes(std::span(bytes));
}

int RunCases() {
  const std::vector<Case> cases = BuildCases();
  std::size_t failures = 0;
  std::size_t accepted = 0;
  for (const Case& test : cases) {
    for (const Caller& caller : kCallers) {
      const Outcome expected = Reference(caller, AsBytes(test.bytes));
      const Outcome actual = Decode(caller, AsBytes(test.bytes));
      accepted += expected.kind == Outcome::Kind::kOk ? 1 : 0;
      if (!Same(expected, actual)) {
        if (++failures <= 20) {
          std::cerr << test.name << " (" << caller.name
                    << "): " << Difference(expected, actual) << "\n";
        }
      }
    }
  }
  std::cout << cases.size() << " cases x " << std::size(kCallers)
            << " callers, " << accepted << " decoded, " << failures
            << " mismatches\n";
  // Guard against a generator that only produces rejections.
  if (accepted < 500) {
    std::cerr << "too few accepted cases: " << accepted << "\n";
    return EXIT_FAILURE;
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

// ---- Timing ---------------------------------------------------------------

double MedianMilliseconds(const std::function<void()>& run, int repeat) {
  std::vector<double> times;
  for (int index = 0; index < repeat; ++index) {
    const auto start = std::chrono::steady_clock::now();
    run();
    times.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count());
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

Bytes RandomFloats(std::mt19937& rng, std::size_t count) {
  std::uniform_real_distribution<float> value(-1.0F, 1.0F);
  Bytes out;
  out.reserve(count * 4);
  for (std::size_t index = 0; index < count; ++index) {
    Append32(out, std::bit_cast<std::uint32_t>(value(rng)));
  }
  return out;
}

int RunTiming(int repeat) {
  std::mt19937 rng(7);
  struct Input {
    std::string_view name;
    const Caller& caller;
    Bytes bytes;
  };
  const Input inputs[] = {
      {"serve 30 s 48 kHz stereo PCM16", kCallers[0],
       Wav({.channels = 2, .sample_rate = 48000},
           RandomBytes(rng, 30ULL * 48000 * 4))},
      {"qwen3_asr 30 min 16 kHz mono PCM16", kCallers[1],
       Wav({.sample_rate = 16000}, RandomBytes(rng, 1800ULL * 16000 * 2))},
      {"qwen3_asr 10 min 48 kHz stereo PCM24", kCallers[1],
       Wav({.channels = 2, .sample_rate = 48000, .bits = 24},
           RandomBytes(rng, 600ULL * 48000 * 6))},
      {"qwen3_tts 30 s 24 kHz mono float32", kCallers[2],
       Wav({.format = 3, .sample_rate = 24000, .bits = 32},
           RandomFloats(rng, 30ULL * 24000))},
  };
  for (const Input& input : inputs) {
    const auto span = AsBytes(input.bytes);
    const double reference =
        MedianMilliseconds([&] { (void)input.caller.reference(span); }, repeat);
    const double rust = MedianMilliseconds(
        [&] { (void)core::DecodeAudio(span, input.caller.limits); }, repeat);
    std::cout << input.name << ": C++ " << reference << " ms, Rust " << rust
              << " ms (" << rust / reference << "x)\n";
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  if (!args.empty() && args[0] == "--time") {
    int repeat = 21;
    if (args.size() == 3 && args[1] == "--repeat") {
      repeat = std::max(1, std::atoi(argv[3]));
    }
    return RunTiming(repeat);
  }
  if (!args.empty()) {
    std::cerr << "usage: wav_decode_test [--time [--repeat N]]\n";
    return EXIT_FAILURE;
  }
  return RunCases();
}
