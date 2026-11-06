#include "src/cli/serve/decision_service.hpp"

#include <mutex>
#include <utility>

namespace gufo::server {

std::shared_ptr<DecisionService> CreateDecisionService(
    const ext::DecisionRunnerOptions& options, std::string served_model_name,
    std::string* error) {
  const auto* provider = ext::ProbeDecisionProvider(options.model_root);
  if (provider == nullptr) {
    *error = "no decision provider claims " + options.model_root;
    return nullptr;
  }
  if (provider->resolve && !provider->resolve(options.model_root, error))
    return nullptr;
  if (provider->create == nullptr) {
    *error =
        "decision provider " + provider->id + " is not available in this build";
    return nullptr;
  }
  std::shared_ptr<ext::DecisionRunner> runner =
      provider->create(options, error);
  if (!runner)
    return nullptr;
  auto service = std::make_shared<DecisionService>();
  service->model_id = !served_model_name.empty() ? std::move(served_model_name)
                      : !runner->ModelId().empty() ? runner->ModelId()
                                                   : provider->id;
  service->multimodal = runner->Multimodal();
  auto mutex = std::make_shared<std::mutex>();
  service->decide = [runner, mutex](const json::Value& request) {
    const std::lock_guard lock(*mutex);
    return runner->Decide(request);
  };
  return service;
}

}  // namespace gufo::server
