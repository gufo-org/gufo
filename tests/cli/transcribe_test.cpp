#include "src/cli/transcribe/transcribe.hpp"

#include <array>
#include <gufo/ext/asr.hpp>
#include <gufo/ext/registry.hpp>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

bool created = false;
bool transcribed = false;

class FakeAsr final : public gufo::ext::AsrRunner {
public:
  bool Transcribe(const gufo::ext::TranscriptionRequest& request,
                  const gufo::ext::CancellationCheck&,
                  gufo::ext::TranscriptionResult* result,
                  std::string*) override {
    transcribed = !request.wav.empty();
    result->text = "registered ASR transcript";
    return transcribed;
  }
};

}  // namespace

// Stand in for generated extension registration. The CLI must invoke it before
// probing: this provider deliberately has no built-in Qwen provider ID.
void gufo::ext::RegisterBuiltExtensions(gufo::ext::Registry& registry) {
  registry.AddAsrProvider({
      .id = "test-asr",
      .probe = +[](std::string_view root) { return root == "test-model"; },
      .create = +[](const std::string& root, std::size_t capacity,
                    std::string*) -> std::unique_ptr<gufo::ext::AsrRunner> {
        created = root == "test-model" && capacity == 1024;
        return std::make_unique<FakeAsr>();
      },
  });
}

int main(int argc, char** argv) {
  if (argc != 2) {
    return 2;
  }
  const std::array<const char*, 4> args = {"--model", "test-model", "--audio",
                                           argv[1]};
  std::ostringstream output;
  auto* previous = std::cout.rdbuf(output.rdbuf());
  const int status = gufo::cli::RunTranscribe(args);
  std::cout.rdbuf(previous);
  if (status != 0 || !created || !transcribed ||
      output.str() != "registered ASR transcript\n") {
    std::cerr
        << "FAIL: transcribe must discover and run its built ASR provider\n";
    return 1;
  }
  return 0;
}
