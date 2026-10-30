#ifndef GUFO_SERVER_LFM3_RUNNER_HPP_
#define GUFO_SERVER_LFM3_RUNNER_HPP_
#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/lfm3/engine.hpp"
#include "src/models/lfm3_vl/vision.hpp"
namespace gufo::server {
std::shared_ptr<TextModelRunner> CreateLfm3Runner(
    std::shared_ptr<models::lfm3::Model> model, std::uint32_t context,
    std::string artifact_fingerprint = {},
    std::shared_ptr<models::lfm3_vl::Vision> vision = {});
}  // namespace gufo::server
#endif
