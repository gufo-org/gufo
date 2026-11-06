#ifndef GUFO_SERVER_DECISION_SERVICE_HPP_
#define GUFO_SERVER_DECISION_SERVICE_HPP_
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/decision.hpp"
#include "src/core/json.hpp"
namespace gufo::server {
struct DecisionService {
  std::string model_id;
  /// Additional IDs accepted from requests; `model_id` stays primary in
  /// listings and responses (#957).
  std::vector<std::string> model_aliases;
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
/// True for the primary ID or a configured alias.
inline bool ServesModel(const DecisionService& service, std::string_view name) {
  return name == service.model_id ||
         std::find(service.model_aliases.begin(), service.model_aliases.end(),
                   name) != service.model_aliases.end();
}
}  // namespace gufo::server
#endif
