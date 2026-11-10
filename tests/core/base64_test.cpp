// Compares base64 through each caller with frozen copies of the base64 code
// in 0.33.0 (tests/core/base64_reference.cpp): media data URLs
// (core::ReadImageUrl), TTS reference audio (server::DecodeBase64Wav), audio
// streams and the WebSocket key (server::DecodeAudioBase64), and the
// encoders. The cases cover every quad over a set of digits, padding and
// invalid bytes in each position, round trips of every length, decoded-size
// limits and random mutations. The decoded bytes, or the result and message,
// must match.
//
// base64_test --time [--repeat N] prints the median time of both
// implementations for a 20 MiB payload; CTest does not run this mode.
#include "src/core/base64.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/audio_buffer.hpp"
#include "src/cli/serve/audio_stream.hpp"
#include "src/core/image.hpp"

namespace gufo::base64_reference {
std::string EncodeEmbeddings(std::span<const std::uint8_t> bytes);
std::string EncodeImageApi(std::span<const std::uint8_t> bytes);
std::vector<std::uint8_t> DecodeImageDataUrl(std::string_view encoded,
                                             std::size_t remaining_bytes);
std::vector<std::byte> DecodeReferenceAudio(std::string_view encoded);
std::string EncodeAudioBase64(std::string_view bytes);
bool DecodeAudioBase64(std::string_view text, std::size_t limit,
                       std::string* bytes);
}  // namespace gufo::base64_reference

namespace {

namespace core = gufo::core;
namespace ref = gufo::base64_reference;
namespace server = gufo::server;

// Decoded bytes, or the result and message of a rejection.
struct Outcome {
  bool ok = false;
  std::string value;
  bool operator==(const Outcome&) const = default;
};

template<typename Bytes>
std::string AsString(const Bytes& bytes) {
  return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

Outcome Rejected(const std::exception& error) {
  return {false, error.what()};
}

Outcome NewImage(std::string_view text) {
  try {
    core::ImageReadBudget budget;
    return {true, AsString(core::ReadImageUrl(
                      "data:image/png;base64," + std::string(text), budget))};
  } catch (const std::invalid_argument& error) {
    return Rejected(error);
  }
}

Outcome OldImage(std::string_view text) {
  try {
    return {true, AsString(ref::DecodeImageDataUrl(
                      text, core::kMaxEncodedImageBytes))};
  } catch (const std::invalid_argument& error) {
    return Rejected(error);
  }
}

// The decoded WAV's format and samples, or the message.
Outcome Wav(bool ok, const server::AudioBuffer& audio,
            const std::string& error) {
  if (!ok) {
    return {false, error};
  }
  return {true, std::to_string(audio.sample_rate) + "/" +
                    std::to_string(audio.channels) + "/" +
                    AsString(std::as_bytes(std::span(audio.samples)))};
}

Outcome NewReferenceAudio(std::string_view text) {
  server::AudioBuffer audio;
  std::string error;
  const bool ok = server::DecodeBase64Wav(text, &audio, &error);
  return Wav(ok, audio, error);
}

Outcome OldReferenceAudio(std::string_view text) {
  std::vector<std::byte> bytes;
  try {
    bytes = ref::DecodeReferenceAudio(text);
  } catch (const std::invalid_argument& error) {
    return Rejected(error);
  }
  server::AudioBuffer audio;
  std::string error;
  const bool ok = server::DecodeWav(bytes, &audio, &error);
  return Wav(ok, audio, error);
}

Outcome Stream(bool ok, const std::string& bytes) {
  return {ok, ok ? bytes : std::string()};
}

struct Totals {
  std::size_t cases = 0;
  std::size_t decoded = 0;
  std::size_t mismatches = 0;
};

void Report(Totals& totals, const char* caller, std::string_view text,
            const Outcome& expected, const Outcome& actual) {
  ++totals.cases;
  totals.decoded += expected.ok ? 1 : 0;
  if (expected == actual) {
    return;
  }
  if (++totals.mismatches <= 10) {
    std::fprintf(
        stderr,
        "%s: \"%.40s\" (%zu chars): expected %s \"%.60s\", got %s \"%.60s\"\n",
        caller, std::string(text).c_str(), text.size(),
        expected.ok ? "ok" : "error", expected.value.c_str(),
        actual.ok ? "ok" : "error", actual.value.c_str());
  }
}

// Every caller's decoder on `text`; the stream decoder with limits around
// the decoded length and the WebSocket key's 16 bytes.
void Decode(Totals& totals, std::string_view text) {
  Report(totals, "image", text, OldImage(text), NewImage(text));
  Report(totals, "reference audio", text, OldReferenceAudio(text),
         NewReferenceAudio(text));
  const std::size_t full = text.size() / 4 * 3;
  for (const std::size_t limit : {full, full > 0 ? full - 1 : 0,
                                  full > 1 ? full - 2 : 0, std::size_t{16}}) {
    std::string old_bytes = "stale";
    std::string new_bytes = "stale";
    const bool old_ok = ref::DecodeAudioBase64(text, limit, &old_bytes);
    const bool new_ok = server::DecodeAudioBase64(text, limit, &new_bytes);
    Report(totals, "audio stream", text, Stream(old_ok, old_bytes),
           Stream(new_ok, new_bytes));
  }
}

void Encode(Totals& totals, const std::vector<std::uint8_t>& bytes) {
  const std::string expected = ref::EncodeEmbeddings(bytes);
  const std::string raw = AsString(bytes);
  const auto check = [&](const char* name, const std::string& actual) {
    Report(totals, name, raw, {true, expected}, {true, actual});
  };
  check("image API encoder (0.33.0)", ref::EncodeImageApi(bytes));
  check("audio encoder (0.33.0)", ref::EncodeAudioBase64(raw));
  check("core::EncodeBase64", core::EncodeBase64(bytes));
  check("server::EncodeAudioBase64", server::EncodeAudioBase64(raw));
}

// A small mono PCM16 WAV, so that reference audio reaches the WAV decoder.
std::string SmallWav() {
  std::string wav = "RIFF";
  const auto put = [&](std::uint32_t value, int size) {
    for (int i = 0; i < size; ++i) {
      wav.push_back(static_cast<char>(value >> (8 * i)));
    }
  };
  put(36 + 8, 4);
  wav += "WAVEfmt ";
  put(16, 4);
  put(1, 2);
  put(1, 2);
  put(16000, 4);
  put(32000, 4);
  put(2, 2);
  put(16, 2);
  wav += "data";
  put(8, 4);
  for (const std::uint16_t sample : {0, 1000, 0x8000, 0x7fff}) {
    put(sample, 2);
  }
  return wav;
}

int RunComparison() {
  Totals totals;
  // Every quad over digits with zero and nonzero low bits, padding, the
  // URL-safe alphabet and other bytes: alone, after a quad and before one.
  constexpr char kBytes[] = {'A', 'Q', 'w', '+',  '/',
                             '=', '-', ' ', '\0', '\xff'};
  constexpr std::size_t kCount = sizeof(kBytes);
  for (std::size_t code = 0; code < kCount * kCount * kCount * kCount; ++code) {
    std::string quad(4, ' ');
    for (std::size_t i = 0, rest = code; i < 4; ++i, rest /= kCount) {
      quad[i] = kBytes[rest % kCount];
    }
    Decode(totals, quad);
    Decode(totals, "QUJD" + quad);
    Decode(totals, quad + "QUJD");
  }

  // Round trips of every length up to 300 bytes, each with mutations: a
  // replaced, inserted or removed character, or nonzero bits under the
  // padding. Reference audio also gets its data URL prefix.
  std::mt19937_64 random(864);
  const std::string wav = SmallWav();
  for (std::size_t length = 0; length <= 300; ++length) {
    std::vector<std::uint8_t> bytes(length);
    for (auto& byte : bytes) {
      byte = static_cast<std::uint8_t>(random());
    }
    Encode(totals, bytes);
    const std::string text = ref::EncodeEmbeddings(bytes);
    Decode(totals, text);
    Decode(totals, "data:audio/wav;base64," + text);
    for (int mutation = 0; mutation < 12 && !text.empty(); ++mutation) {
      std::string mutated = text;
      const std::size_t at = random() % mutated.size();
      const char byte = mutation % 2 == 0 ? kBytes[random() % kCount]
                                          : static_cast<char>(random());
      switch (mutation % 4) {
        case 0:
        case 1:
          mutated[at] = byte;
          break;
        case 2:
          mutated.insert(mutated.begin() + static_cast<std::ptrdiff_t>(at),
                         byte);
          break;
        default:
          mutated.erase(at, 1);
          break;
      }
      Decode(totals, mutated);
    }
    if (text.ends_with('=')) {
      std::string noncanonical = text;
      const std::size_t last = noncanonical.find('=') - 1;
      noncanonical[last] = noncanonical[last] == '/' ? '+' : '/';
      Decode(totals, noncanonical);
    }
  }
  const std::string wav_text = server::EncodeAudioBase64(wav);
  Decode(totals, wav_text);
  Decode(totals, "data:audio/wav;base64," + wav_text);

  // A megabyte through the table path, then an error deep inside it and in
  // the last quad.
  std::vector<std::uint8_t> large(1 << 20);
  for (auto& byte : large) {
    byte = static_cast<std::uint8_t>(random());
  }
  Encode(totals, large);
  const std::string large_text = ref::EncodeEmbeddings(large);
  Decode(totals, large_text);
  for (const std::size_t at : {large_text.size() / 2, large_text.size() - 1}) {
    for (const char byte : {'!', '='}) {
      std::string mutated = large_text;
      mutated[at] = byte;
      Decode(totals, mutated);
    }
  }

  std::printf("base64: %zu cases, %zu decoded, %zu mismatches\n", totals.cases,
              totals.decoded, totals.mismatches);
  return totals.mismatches == 0 ? 0 : 1;
}

double MedianMs(int repeat, const std::function<void()>& run) {
  std::vector<double> times;
  for (int i = 0; i < repeat; ++i) {
    const auto start = std::chrono::steady_clock::now();
    run();
    times.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count());
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

int RunTiming(int repeat) {
  std::vector<std::uint8_t> bytes(core::kMaxEncodedImageBytes);
  std::mt19937_64 random(1);
  for (auto& byte : bytes) {
    byte = static_cast<std::uint8_t>(random());
  }
  const std::string raw = AsString(bytes);
  const std::string text = ref::EncodeEmbeddings(bytes);
  const std::string url = "data:image/png;base64," + text;
  std::string out;
  std::printf("20 MiB payload, %zu characters; median of %d (0.33.0 / now)\n",
              text.size(), repeat);
  std::printf("encode:            %8.2f / %8.2f ms\n",
              MedianMs(repeat, [&] { (void)ref::EncodeEmbeddings(bytes); }),
              MedianMs(repeat, [&] { (void)core::EncodeBase64(bytes); }));
  std::printf("image data URL:    %8.2f / %8.2f ms\n",
              MedianMs(repeat,
                       [&] {
                         (void)ref::DecodeImageDataUrl(
                             text, core::kMaxEncodedImageBytes);
                       }),
              MedianMs(repeat, [&] {
                core::ImageReadBudget budget;
                (void)core::ReadImageUrl(url, budget);
              }));
  std::printf(
      "audio stream:      %8.2f / %8.2f ms\n",
      MedianMs(repeat,
               [&] { (void)ref::DecodeAudioBase64(text, bytes.size(), &out); }),
      MedianMs(repeat, [&] {
        (void)server::DecodeAudioBase64(text, bytes.size(), &out);
      }));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::strcmp(argv[1], "--time") == 0) {
      int repeat = 9;
      if (argc == 4 && std::strcmp(argv[2], "--repeat") == 0) {
        repeat = std::max(1, std::atoi(argv[3]));
      }
      return RunTiming(repeat);
    }
    return RunComparison();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
