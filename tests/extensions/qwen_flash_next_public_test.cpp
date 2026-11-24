// An external extension's view of the experimental Flash-Next headers: the
// public forwards under include/gufo/ext/families/qwen/models/flash_next/
// compile on their own (CMake adds one translation unit per header) and
// reach the default stages and weight source.
// gufo_qwen38_flash_next_hip still publishes the repository root (no
// narrowed export target yet), so this checks the headers, not isolation.
#include <cstdlib>
#include <gufo/ext/families/qwen/models/flash_next/config.hpp>
#include <gufo/ext/families/qwen/models/flash_next/model.hpp>
#include <gufo/ext/families/qwen/models/flash_next/ngram.hpp>
#include <gufo/ext/families/qwen/models/flash_next/primitives.hpp>
#include <gufo/ext/families/qwen/models/flash_next/runner.hpp>
#include <gufo/ext/families/qwen/models/flash_next/stages.hpp>
#include <gufo/ext/families/qwen/models/flash_next/weight_source.hpp>
#include <gufo/ext/families/qwen/models/flash_next/weights.hpp>
#include <iostream>
#include <memory>

namespace fn = gufo::ext::qwen::flash_next;

int main() {
  const auto source = fn::CreateGgufWeightSource();
  fn::Stages stages;
  if (source == nullptr || source->Probe("/nonexistent/flash-next.gguf") ||
      stages.Hooks() != fn::kNoStageHooks ||
      source->PreferredPrefillChunk() != 4096) {
    std::cerr << "the public Flash-Next defaults changed\n";
    return EXIT_FAILURE;
  }
  gufo::models::qwen38_flash_next::Config config;
  std::string error;
  if (fn::Config::Validated(config, true, &error) || error.empty()) {
    std::cerr << "Config::Validated accepted an empty configuration\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
