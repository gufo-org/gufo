// WAV's strict C++ caller: sample scaling, limits and exception propagation.
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/audio.hpp"
#include "src/core/rust/callback.hpp"

namespace core = gufo::core;
void Require(bool value) {
  if (!value)
    throw std::runtime_error("WAV expected-result check failed");
}
int main() {
  try {
    // Mono PCM16: -32768, 0, 32767 at 16 kHz.
    const std::vector<std::uint8_t> wav{
        'R', 'I', 'F', 'F',  42, 0, 0,    0, 'W', 'A',  'V', 'E',  'f',
        'm', 't', ' ', 16,   0,  0, 0,    1, 0,   1,    0,   0x80, 0x3e,
        0,   0,   0,   0x7d, 0,  0, 2,    0, 16,  0,    'd', 'a',  't',
        'a', 6,   0,   0,    0,  0, 0x80, 0, 0,   0xff, 0x7f};
    const auto bytes = std::as_bytes(std::span(wav));
    const auto audio = core::DecodeAudio(bytes, {});
    Require(audio.sample_rate == 16000 && audio.channels == 1);
    Require(audio.samples ==
            std::vector<float>({-1.0F, 0.0F, 32767.0F / 32768.0F}));
    const auto rejects = [&](std::span<const std::byte> input,
                             const core::AudioLimits& limits,
                             core::AudioStatus status) {
      try {
        (void)core::DecodeAudio(input, limits);
      } catch (const core::AudioError& error) {
        Require(error.status() == status);
        return;
      }
      throw std::runtime_error("WAV unexpectedly accepted");
    };
    rejects(bytes.first(3), {}, core::AudioStatus::kUnrecognized);
    rejects(bytes.first(bytes.size() - 1), {}, core::AudioStatus::kTruncated);
    rejects(bytes, {.max_sample_rate = 8000}, core::AudioStatus::kOutOfBounds);
    rejects(bytes, {.max_seconds = 0}, core::AudioStatus::kTooLong);
    bool propagated = false;
    try {
      (void)core::DecodeAudio(bytes, {}, []() -> bool {
        throw std::runtime_error("stop callback");
      });
    } catch (const std::runtime_error& error) {
      propagated = std::string(error.what()) == "stop callback";
    }
    Require(propagated);
    core::ffi::CallbackFailure failure;
    Require(failure.Guard(
                7, []() -> int { throw std::runtime_error("first"); }) == 7);
    Require(failure.Guard(9, [] { return 1; }) == 9);
    try {
      failure.Rethrow();
      Require(false);
    } catch (const std::runtime_error& error) {
      Require(std::string(error.what()) == "first");
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
