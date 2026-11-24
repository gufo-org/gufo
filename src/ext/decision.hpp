#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

#include "gufo/ext/options.hpp"
#include "src/core/json.hpp"

namespace gufo::ext {

/// Model artifacts and capacity for a decision runner. projector_path is an
/// optional media sidecar (for example an mmproj GGUF).
struct DecisionRunnerOptions {
  std::string model_root;
  std::string projector_path;
  /// Longest complete decision prompt.
  std::size_t maximum_context_tokens{8192};
  /// The explicit command-line values of the options the provider declares
  /// for this command (see TextRunnerOptions::provider_options).
  ParsedOptions provider_options{};
};

/// Abstract capability runner for decision models: a request of typed
/// questions about text and media, answered with calibrated typed values.
/// The JSON shapes are the model's own (for d1, the /v1/systemone body).
class DecisionRunner {
public:
  virtual ~DecisionRunner() = default;
  /// Throws std::invalid_argument for a request the model rejects; other
  /// exceptions are failures. Calls are serialized by the host.
  [[nodiscard]] virtual json::Value Decide(const json::Value& request) = 0;
  /// Accepts image and audio inputs besides text.
  [[nodiscard]] virtual bool Multimodal() const { return false; }
  /// Public model ID when the service names none.
  [[nodiscard]] virtual std::string ModelId() const { return {}; }
};

using DecisionProbe = bool (*)(std::string_view model_root);
using DecisionArtifactResolver = bool (*)(std::string_view model_root,
                                          std::string* error);
using DecisionRunnerFactory = std::unique_ptr<DecisionRunner> (*)(
    const DecisionRunnerOptions& options, std::string* error);

struct DecisionProvider {
  std::string id;
  DecisionProbe probe = nullptr;
  DecisionArtifactResolver resolve = nullptr;
  DecisionRunnerFactory create = nullptr;
};

void RegisterDecisionProvider(DecisionProvider provider);
const DecisionProvider* FindDecisionProvider(std::string_view id);
const DecisionProvider* ProbeDecisionProvider(std::string_view model_root);

}  // namespace gufo::ext
