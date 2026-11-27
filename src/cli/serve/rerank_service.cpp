#include "src/cli/serve/rerank_service.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <utility>

#include "gufo/ext/registry.hpp"
#include "src/cli/serve/rerank_metrics.hpp"

namespace gufo::server {

struct RerankService::Impl {
  explicit Impl(RerankServiceOptions supplied) : options(std::move(supplied)) {
    if (options.runner) {
      // An injected runner needs no checkpoint, so the model is not validated
      // unless the caller asks for it.
      if (options.validate_model && !options.model_root.empty()) {
        const ext::RerankProvider* provider =
            ext::ProbeRerankProvider(options.model_root.string());
        if (!provider || !provider->probe ||
            !provider->probe(options.model_root.string())) {
          initialization_error = kInvalidModel;
          return;
        }
      }
      NameModel();
      ready = true;
      return;
    }

    const ext::RerankProvider* provider = nullptr;
    if (!options.model_root.empty()) {
      provider = ext::ProbeRerankProvider(options.model_root.string());
    }
    if (!provider || !provider->create_with_options) {
      initialization_error = "rerank provider not available for model: " +
                             options.model_root.string();
      return;
    }
    if (options.validate_model && provider->probe &&
        !provider->probe(options.model_root.string())) {
      initialization_error = kInvalidModel;
      return;
    }
    if (provider->resolve) {
      std::string error;
      if (!provider->resolve(options.model_root.string(), &error)) {
        initialization_error =
            "rerank model artifacts are incomplete: " + error;
        return;
      }
    }
    std::string error;
    runner_ = provider->create_with_options(
        {.model_root = options.model_root.string(),
         .provider_options = options.provider_options},
        &error);
    if (!runner_) {
      initialization_error = "rerank model failed to load: " + error;
      return;
    }
    NameModel();
    ready = true;
  }

  // The name clients see: the one given, else the runner's, else the model
  // directory's.
  void NameModel() {
    if (!options.model_id.empty()) {
      return;
    }
    if (runner_) {
      options.model_id = runner_->ModelId();
    }
    if (options.model_id.empty()) {
      options.model_id = options.model_root.filename().string();
    }
    if (options.model_id.empty()) {
      options.model_id = "rerank";
    }
  }

  RerankStatus Run(const RerankInput& query,
                   std::span<const RerankInput> documents,
                   const RerankRequestOptions& request,
                   const std::function<bool()>& is_cancelled,
                   RerankScores* scores, std::string* error) {
    if (options.runner) {
      std::lock_guard<std::mutex> lock(runner_mutex);
      return options.runner(query, documents, request, is_cancelled, scores,
                            error);
    }
    const RerankStatus status =
        runner_->Rerank(query, documents, is_cancelled, scores, error, request);
    if (runner_->device_lost()) {
      MarkDeviceLost();
    }
    return status;
  }

  void MarkDeviceLost() {
    if (!device_lost.exchange(true)) {
      metrics.device_lost.Add(1);
    }
  }

  static constexpr const char* kInvalidModel =
      "rerank service requires a valid model directory";

  RerankServiceOptions options;
  std::unique_ptr<ext::RerankRunner> runner_;
  std::string initialization_error;
  // Serializes the injected runner, which has no device turns of its own.
  std::mutex runner_mutex;
  std::atomic<bool> device_lost{false};
  RerankMetrics metrics;
  bool ready{false};
};

namespace {

const char* ScoresProblem(std::size_t documents, const RerankScores& scores) {
  if (scores.scores.size() != documents ||
      scores.token_counts.size() != documents) {
    return "rerank runtime returned an inconsistent result";
  }
  for (const float score : scores.scores) {
    if (!std::isfinite(score)) {
      return "rerank runtime returned a non-finite score";
    }
  }
  return nullptr;
}

}  // namespace

RerankService::RerankService(RerankServiceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

RerankService::~RerankService() = default;

bool RerankService::ready() const noexcept {
  return impl_->ready;
}

bool RerankService::device_lost() const noexcept {
  return impl_->device_lost.load();
}

std::string RerankService::initialization_error() const {
  return impl_->initialization_error;
}

std::string RerankService::model_id() const {
  return impl_->options.model_id;
}

bool RerankService::serves_model(std::string_view name) const {
  return name == model_id() ||
         std::find(impl_->options.model_aliases.begin(),
                   impl_->options.model_aliases.end(),
                   std::string(name)) != impl_->options.model_aliases.end();
}

const std::vector<std::string>& RerankService::model_aliases() const {
  return impl_->options.model_aliases;
}

std::string RerankService::backend_name() const {
  return impl_->options.runner ? "custom" : "extension";
}

std::string RerankService::RenderMetrics() const {
  return impl_->metrics.registry.Render();
}

RerankStatus RerankService::Rerank(const RerankInput& query,
                                   std::span<const RerankInput> documents,
                                   const std::function<bool()>& is_cancelled,
                                   RerankScores* scores, std::string* error,
                                   const RerankRequestOptions& request) {
  RerankRequestScope scope(impl_->metrics);
  const RerankStatus status = RerankAdmitted(query, documents, is_cancelled,
                                             scores, error, request, scope);
  scope.Finish(status, *scores);
  return status;
}

RerankStatus RerankService::RerankAdmitted(
    const RerankInput& query, std::span<const RerankInput> documents,
    const std::function<bool()>& is_cancelled, RerankScores* scores,
    std::string* error, const RerankRequestOptions& request,
    RerankRequestScope& scope) {
  *scores = {};
  if (!ready()) {
    if (error) {
      *error = initialization_error();
    }
    return device_lost() ? RerankStatus::kDeviceLost : RerankStatus::kFailed;
  }
  if (device_lost()) {
    if (error) {
      *error = "rerank device lost";
    }
    return RerankStatus::kDeviceLost;
  }
  if (documents.empty()) {
    return RerankStatus::kOk;
  }
  scope.Accepted(documents.size());
  if (is_cancelled && is_cancelled()) {
    if (error) {
      *error = "rerank request cancelled";
    }
    return RerankStatus::kCancelled;
  }
  const RerankStatus status =
      impl_->Run(query, documents, request, is_cancelled, scores, error);
  if (status == RerankStatus::kDeviceLost) {
    impl_->MarkDeviceLost();
  }
  if (status != RerankStatus::kOk) {
    return status;
  }
  if (is_cancelled && is_cancelled()) {
    if (error) {
      *error = "rerank request cancelled";
    }
    return RerankStatus::kCancelled;
  }
  if (const char* problem = ScoresProblem(documents.size(), *scores)) {
    if (error) {
      *error = problem;
    }
    return RerankStatus::kFailed;
  }
  return status;
}

}  // namespace gufo::server
