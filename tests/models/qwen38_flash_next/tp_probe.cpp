// TP=2 probe: --tp-rank 0|1 --tp-world-size 2
//              --tp-bootstrap-host HOST --tp-bootstrap-port PORT
//              --tp-operation-id N (same on both ranks)
// Prefill steps: --split N prefills the prompt once in one step and once as
// N tokens then the rest, and compares each MoE input and output of the last
// token and the final logits: a prefill must not depend on its steps.
// --decode-tail K instead feeds the last K tokens one at a time, as decoding
// does: a token must compute the same whether prefilled or decoded.

#include <hip/hip_runtime.h>

#include <algorithm>
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
  std::uint32_t split = 0;
  std::uint32_t decode_tail = 0;

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
    } else if (arg == "--decode-tail") {
      if (!ParseUint(next(), &decode_tail)) {
        Fail("--decode-tail requires an unsigned integer");
        return 2;
      }
    } else if (arg == "--split") {
      if (!ParseUint(next(), &split)) {
        Fail("--split requires an unsigned integer");
        return 2;
      }
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
  // `--split`: a hash of the last token's row at every MoE input and output.
  struct RowHashes {
    bool armed{false};
    std::size_t hidden{0};
    std::vector<std::uint64_t> hashes;
  };
  const auto rows = std::make_shared<RowHashes>();
  if (split != 0 || decode_tail != 0) {
    options.moe_observer = [rows](const float* data, std::size_t bytes,
                                  ihipStream_t* stream) {
      const std::size_t row = rows->hidden * sizeof(float);
      hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
      // A graph being captured cannot be read; its logits still compare.
      if (!rows->armed || row == 0 || bytes < row ||
          hipStreamIsCapturing(stream, &capture) != hipSuccess ||
          capture != hipStreamCaptureStatusNone) {
        return;
      }
      std::vector<std::uint8_t> host(row);
      if (hipStreamSynchronize(stream) != hipSuccess ||
          hipMemcpy(host.data(),
                    reinterpret_cast<const char*>(data) + bytes - row, row,
                    hipMemcpyDeviceToHost) != hipSuccess) {
        rows->hashes.push_back(0);
        return;
      }
      std::uint64_t hash = 1469598103934665603ULL;
      for (const auto byte : host) {
        hash = (hash ^ byte) * 1099511628211ULL;
      }
      rows->hashes.push_back(hash);
    };
  }
  auto model = q::Model::Load(model_path, options, &error);
  if (!model) {
    Fail("model load failed: " + error);
    return 1;
  }
  if (split != 0 || decode_tail != 0) {
    rows->hidden = model->config().hidden_size;
    const auto prompt_tokens = model->Tokenize(prompt);
    if (split == 0) {
      split =
          decode_tail < prompt_tokens.size()
              ? static_cast<std::uint32_t>(prompt_tokens.size()) - decode_tail
              : 0;
    }
    if (split == 0 || split >= prompt_tokens.size() ||
        prompt_tokens.size() > context) {
      Fail("--split must fall inside the prompt, which must fit the context");
      return 2;
    }
    const auto mode = model->HasMtp()
                          ? gufo::core::SessionMode::kSpeculative
                          : gufo::core::SessionMode::kAutoregressive;
    const auto run = [&](bool in_two)
        -> std::optional<
            std::pair<std::vector<std::uint64_t>, std::vector<float>>> {
      auto session = model->CreateSession(mode, context, &error);
      if (!session) {
        return std::nullopt;
      }
      const std::span<const std::int32_t> all(prompt_tokens);
      if (in_two && !session->Sync(all.first(split), &error)) {
        return std::nullopt;
      }
      rows->hashes.clear();
      rows->armed = true;
      bool ok = true;
      if (in_two && decode_tail != 0) {
        // Decoding: one token per forward; only the last one's observations
        // are kept, to compare with the last row of the prefill.
        for (std::size_t t = split; ok && t < all.size(); ++t) {
          rows->hashes.clear();
          ok = session->Evaluate(all[t], &error);
        }
      } else {
        ok = session->Sync(all, &error);
      }
      rows->armed = false;
      if (!ok) {
        return std::nullopt;
      }
      const auto logits = session->Logits();
      return std::make_pair(rows->hashes,
                            std::vector<float>(logits.begin(), logits.end()));
    };
    const auto whole = run(false);
    const auto parts = run(true);
    if (!whole || !parts) {
      Fail("split prefill failed: " + error);
      return 1;
    }
    std::size_t first = whole->first.size();
    for (std::size_t i = 0;
         i < std::min(whole->first.size(), parts->first.size()); ++i) {
      if (whole->first[i] != parts->first[i]) {
        first = i;
        break;
      }
    }
    float max_diff = 0.0F;
    std::size_t differing = 0;
    for (std::size_t i = 0; i < whole->second.size(); ++i) {
      const float diff = std::fabs(whole->second[i] - parts->second[i]);
      differing += diff != 0.0F ? 1 : 0;
      max_diff = std::max(max_diff, diff);
    }
    std::printf(
        "rank=%u prompt=%zu split=%u observations=%zu/%zu first_difference=%zu "
        "logits_differing=%zu max_abs_diff=%.9g\n",
        rank, prompt_tokens.size(), split, whole->first.size(),
        parts->first.size(), first, differing, max_diff);
    return 0;
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
