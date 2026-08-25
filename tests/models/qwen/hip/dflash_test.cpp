#include "src/models/qwen/hip/dflash.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/dflash_reference.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

constexpr int kSkipped = 77;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    if (argc < 3) {
      std::cout << "qwen_dflash_gpu_test: skipped "
                   "(pass base and DFlash GGUF paths)\n";
      return kSkipped;
    }

    std::string error;
    auto base_owner = strix::core::GgufReader::OpenFile(argv[1], &error);
    Expect(base_owner != nullptr, error);
    auto dflash_owner = strix::core::GgufReader::OpenFile(argv[2], &error);
    Expect(dflash_owner != nullptr, error);

    std::shared_ptr<const strix::core::GgufReader> base_reader(
        std::move(base_owner));
    std::shared_ptr<const strix::core::GgufReader> dflash_reader(
        std::move(dflash_owner));

    auto target_model =
        strix::hip::QwenGpuModel::CreateFromGguf(base_reader, &error);
    Expect(target_model != nullptr, error);

    auto dflash_model = strix::hip::QwenDFlashGpuModel::Create(
        dflash_reader, target_model, &error);
    Expect(dflash_model != nullptr, error);

    strix::hip::QwenDFlashGpuDraftConfig config{
        .max_context = 512,
        .max_draft_tokens = 8,
    };
    auto backend = strix::hip::QwenDFlashGpuDraftBackend::Create(
        dflash_model, config, &error);
    Expect(backend != nullptr, error);

    Expect(backend->RequiresTargetHiddenStates(), "RequiresTargetHiddenStates");
    Expect(backend->Name() == "QwenDFlashGpuDraftBackend", "Name matches");

    std::cout << "qwen_dflash_gpu_test: ALL TESTS PASSED\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "DFlash GPU test exception: " << ex.what() << '\n';
    return 1;
  }
}
