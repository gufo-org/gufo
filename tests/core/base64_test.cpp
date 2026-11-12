// Expected results through the C ABI and the production callers.
#include "src/core/base64.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/audio_buffer.hpp"
#include "src/cli/serve/audio_stream.hpp"
#include "src/core/image.hpp"

namespace core = gufo::core;
namespace server = gufo::server;

void Require(bool value) {
  if (!value)
    throw std::runtime_error("base64 expected-result check failed");
}

int main() {
  try {
    for (const auto& [plain, encoded] :
         std::array<std::pair<std::string, std::string>, 7>{
             {{"", ""},
              {"f", "Zg=="},
              {"fo", "Zm8="},
              {"foo", "Zm9v"},
              {"foob", "Zm9vYg=="},
              {"fooba", "Zm9vYmE="},
              {"foobar", "Zm9vYmFy"}}}) {
      Require(core::EncodeBase64(plain) == encoded);
      Require(server::EncodeAudioBase64(plain) == encoded);
      std::string decoded;
      Require(server::DecodeAudioBase64(encoded, plain.size(), &decoded) &&
              decoded == plain);
    }
    const auto check = [](std::string_view text, core::Base64Status expected,
                          core::Base64Options options = {}) {
      std::string bytes = "old";
      Require(core::DecodeBase64(text, options, bytes) == expected);
      if (expected != core::Base64Status::kOk)
        Require(bytes.empty());
    };
    check("a", core::Base64Status::kLength);
    check("=AAA", core::Base64Status::kPadding);
    check("AA=A", core::Base64Status::kAfterPadding);
    check("AA?A", core::Base64Status::kCharacter);
    check("Zh==", core::Base64Status::kNonCanonical);
    check("Zh==", core::Base64Status::kOk, {.canonical = false});
    check("Zm9v", core::Base64Status::kTooLong, {.max_decoded = 2});
    for (std::size_t size = 0; size <= 300; ++size) {
      std::string plain(size, '\0');
      for (std::size_t i = 0; i < size; ++i)
        plain[i] = static_cast<char>(i * 37);
      std::string decoded;
      const auto encoded = core::EncodeBase64(plain);
      Require(server::DecodeAudioBase64(encoded, size, &decoded) &&
              decoded == plain);
      if (size)
        Require(!server::DecodeAudioBase64(encoded, size - 1, &decoded));
    }
    const auto image = core::ReadImageUrl("data:image/png;base64,Zm9v");
    Require(image == std::vector<std::uint8_t>({'f', 'o', 'o'}));
    server::AudioBuffer audio;
    std::string error;
    Require(!server::DecodeBase64Wav("?", &audio, &error) && !error.empty());
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
