#ifndef GUFO_MODELS_GLM5_NEXT_CLI_RUNNER_HPP_
#define GUFO_MODELS_GLM5_NEXT_CLI_RUNNER_HPP_

#include <chrono>
#include <memory>

namespace gufo::core {
class GgufReader;
}

namespace gufo::ext {
struct TextBenchOptions;
}

namespace gufo::models::glm5_next {

#if defined(ENGINE_ENABLE_HIP)

bool IsGlm5Next(const core::GgufReader& reader);

int RunGlm5NextBenchmark(const ext::TextBenchOptions& options,
                         const std::shared_ptr<const core::GgufReader>& reader,
                         std::chrono::steady_clock::time_point model_load_start);

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_CLI_RUNNER_HPP_
