// TP=2 probe: --tp-rank 0|1 --tp-world-size 2
//              --tp-bootstrap-host HOST --tp-bootstrap-port PORT
//              --tp-operation-id N (same on both ranks)
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#ifdef GUFO_ENABLE_TP2_RDMA
#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"
#endif

namespace q = gufo::models::qwen38_flash_next;

namespace {

#ifdef GUFO_ENABLE_TP2_RDMA
class OperationGuard {
public:
  OperationGuard(std::shared_ptr<q::rocm::Communicator> communicator,
                 std::uint64_t scope_id)
      : communicator_(std::move(communicator)), scope_id_(scope_id) {}

  ~OperationGuard() {
    if (!active_) {
      return;
    }
    try {
      std::string error;
      (void)communicator_->EndOperation(scope_id_, &error);
    } catch (...) {
    }
  }

  [[nodiscard]] bool Begin(std::string* error) {
    if (!communicator_->BeginOperation(scope_id_, error)) {
      return false;
    }
    active_ = true;
    return true;
  }

private:
  std::shared_ptr<q::rocm::Communicator> communicator_;
  const std::uint64_t scope_id_;
  bool active_{false};
};
#endif

void Fail(const std::string& message) {
  std::fprintf(stderr, "%s\n", message.c_str());
}

bool ParseUint(std::string_view text, std::uint32_t* value) {
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return error == std::errc{} && end == text.data() + text.size();
}

bool ParseUint64(std::string_view text, std::uint64_t* value) {
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return error == std::errc{} && end == text.data() + text.size();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string mtp_path;
  std::string prompt = "The capital of France is";
  std::uint32_t context = 4096;
  std::uint32_t tokens = 16;
  std::uint32_t rank = 0;
  std::uint32_t world_size = 1;
  std::uint32_t device = 0;
  std::uint32_t gid = 0;
  std::uint64_t operation_id = 1;
#ifndef GUFO_ENABLE_TP2_RDMA
  (void)operation_id;
#endif
  std::uint16_t port = 18515;
  float temperature = 0.0F;
  std::int64_t seed = 7;
  std::string bootstrap_host;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--temperature") {
      const std::string value = next();
      char* end = nullptr;
      temperature = std::strtof(value.c_str(), &end);
      if (end == value.c_str() || *end != '\0' || !std::isfinite(temperature) ||
          temperature < 0.0F) {
        Fail("--temperature requires a nonnegative number");
        return 2;
      }
    } else if (arg == "--seed") {
      std::uint32_t parsed = 0;
      if (!ParseUint(next(), &parsed)) {
        Fail("--seed requires an unsigned integer");
        return 2;
      }
      seed = static_cast<std::int64_t>(parsed);
    } else if (arg == "--model") {
      model_path = next();
    } else if (arg == "--mtp-model") {
      mtp_path = next();
    } else if (arg == "--prompt") {
      prompt = next();
    } else if (arg == "--context") {
      if (!ParseUint(next(), &context)) {
        Fail("--context requires an unsigned integer");
        return 2;
      }
    } else if (arg == "--tokens") {
      if (!ParseUint(next(), &tokens)) {
        Fail("--tokens requires an unsigned integer");
        return 2;
      }
    } else if (arg == "--tp-rank") {
      if (!ParseUint(next(), &rank)) {
        Fail("--tp-rank requires an unsigned integer");
        return 2;
      }
    } else if (arg == "--tp-world-size") {
      if (!ParseUint(next(), &world_size)) {
        Fail("--tp-world-size requires an unsigned integer");
        return 2;
      }
    } else if (arg == "--tp-device") {
      if (!ParseUint(next(), &device)) {
        Fail("--tp-device requires an unsigned integer");
        return 2;
      }
    } else if (arg == "--tp-gid-index") {
      if (!ParseUint(next(), &gid)) {
        Fail("--tp-gid-index requires an unsigned integer");
        return 2;
      }
    } else if (arg == "--tp-bootstrap-port") {
      std::uint32_t parsed = 0;
      if (!ParseUint(next(), &parsed) || parsed == 0 || parsed > 65535) {
        Fail("--tp-bootstrap-port is out of range");
        return 2;
      }
      port = static_cast<std::uint16_t>(parsed);
    } else if (arg == "--tp-operation-id") {
      if (!ParseUint64(next(), &operation_id)) {
        Fail("--tp-operation-id requires an unsigned integer");
        return 2;
      }
    } else if (arg == "--tp-bootstrap-host") {
      bootstrap_host = next();
    } else {
      Fail("unknown argument: " + arg);
      return 2;
    }
  }
  if (model_path.empty() || context == 0 || tokens == 0 ||
      (world_size != 1 && world_size != 2) || rank >= world_size ||
      (world_size == 1 && rank != 0) ||
      device > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      gid > std::numeric_limits<std::uint8_t>::max() ||
      (world_size == 2 && rank == 1 && bootstrap_host.empty())) {
    Fail("invalid TP2 probe arguments");
    return 2;
  }

  std::shared_ptr<q::rocm::Communicator> communicator;
#ifdef GUFO_ENABLE_TP2_RDMA
  std::unique_ptr<OperationGuard> operation;
  if (world_size == 2) {
    std::string error;
    const q::rocm::IbrverbsConfig config{
        .rank = rank,
        .world_size = world_size,
        .bootstrap_host = bootstrap_host,
        .bootstrap_port = port,
        .device_index = device,
        .gid_index = gid,
    };
    communicator = q::rocm::CreateIbrverbsCommunicator(config, &error);
    if (!communicator) {
      Fail("RDMA communicator failed: " + error);
      return 1;
    }
    operation = std::make_unique<OperationGuard>(communicator, operation_id);
    if (!operation->Begin(&error)) {
      Fail("TP operation scope bind failed: " + error);
      return 1;
    }
  }
#else
  if (world_size == 2) {
    Fail("this probe was built without TP2 RDMA support");
    return 2;
  }
#endif

  std::string error;
  q::ModelOptions options{
      .max_context = context,
      .mtp_model_path = mtp_path,
      .max_draft_tokens = 7,
      .tp_rank = rank,
      .tp_world_size = world_size,
      .hip_device = static_cast<int>(device),
      .communicator = communicator,
  };
  auto model = q::Model::Load(model_path, options, &error);
  if (!model) {
    Fail("model load failed: " + error);
    return 1;
  }
  const auto prompt_tokens = model->Tokenize(prompt);
  if (prompt_tokens.empty() || prompt_tokens.size() > context ||
      tokens > context - prompt_tokens.size()) {
    Fail("prompt is outside the requested context");
    return 2;
  }
  auto session = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      context, &error);
  if (!session || !session->Sync(prompt_tokens, &error)) {
    Fail("prompt sync failed: " + error);
    return 1;
  }

  const gufo::sampling::SamplingConfig sampling{
      .temperature = temperature,
      .seed = seed,
  };
  gufo::sampling::SamplerState sampler(sampling);
  std::vector<std::int32_t> output;
  while (output.size() < tokens) {
    q::Session::DecodeResult step;
    if (!session->DecodeStep(tokens - output.size(), sampler, &step, &error,
                             false)) {
      Fail("decode failed: " + error);
      return 1;
    }
    output.insert(output.end(), step.tokens.begin(), step.tokens.end());
  }
  if (output.size() > tokens) {
    output.resize(tokens);
  }
  std::printf("rank=%u tokens=%zu text=%s\n", rank, output.size(),
              model->Decode(std::span<const std::int32_t>(output)).c_str());
  return 0;
}
