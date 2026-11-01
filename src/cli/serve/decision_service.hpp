#ifndef GUFO_SERVER_DECISION_SERVICE_HPP_
#define GUFO_SERVER_DECISION_SERVICE_HPP_
#include <functional>
#include <memory>
#include <string>

#include "gufo/ext/decision.hpp"
#include "src/core/json.hpp"
namespace gufo::server {
struct DecisionService {
  std::string model_id;
  bool multimodal{};
  /// Throws std::invalid_argument for rejected requests.
  std::function<json::Value(const json::Value&)> decide;
};
/// Resolves the decision provider for options.model_root and loads it.
/// Requests are serialized over the one loaded model. Null with an error
/// when no provider claims the model or loading fails.
std::shared_ptr<DecisionService> CreateDecisionService(
    const ext::DecisionRunnerOptions& options, std::string served_model_name,
    std::string* error);
}  // namespace gufo::server
#endif
