// Dumps trunk prefill logits and optionally compares them with the F32 oracle.
// Use the production CLI for generation and the session test for replay.
//
// gpu_probe --model FIRST_SHARD.gguf --prompt TEXT [--reference]
//           [--batch T] [--context N] [--dump logits.bin]
// TP=2 probe: --tp-rank 0|1 --tp-world-size 2
//              --tp-bootstrap-host HOST --tp-bootstrap-port PORT
//              --tp-operation-id N (same on both ranks)
// Independent predictor oracle: --mtp-model MTP.gguf --mtp-audit
// MTP cost calibration: --mtp-model MTP.gguf --cost-audit C (0 = all)
//                      [--depth N] (default: 0, 4096, 32768)
// External reference: --kld-base FILE, a llama.cpp --kl-divergence-base file
//                     (every TP rank needs the file's tokens)
//                     [--kld-save OUT] writes Gufo's log-probs in its format
// Matched-token decode: --prompt-file P --decode-file C --dump rows.bin
//                       [--decode-ids ids.bin] (prefill P, then feed C's
//                       tokens one at a time; row i predicts the token
//                       after ids[i], so score row i against ids[i + 1])
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/distributed/tp_partition.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/communicator.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#ifdef GUFO_ENABLE_TP2_RDMA
#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"
#endif
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next;

void AuditMtp(q::rocm::Executor& executor, const q::rocm::DeviceModel& device,
              const q::ModelWeights& weights, const q::MtpWeights& mtp,
              const gufo::tokenization::QwenTokenizer& tokenizer,
              const std::filesystem::path& path);
void AuditMtpCosts(q::rocm::Executor& executor,
                   const gufo::tokenization::QwenTokenizer& tokenizer,
                   std::span<const std::uint32_t> depths,
                   std::uint32_t concurrency);

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

struct Stats {
  std::uint32_t argmax_a;
  std::uint32_t argmax_b;
  double kl;
  double max_abs;
};

Stats Compare(const float* a, const float* b, std::size_t n) {
  std::vector<double> pa(n);
  std::vector<double> pb(n);
  const float ma = *std::max_element(a, a + n);
  const float mb = *std::max_element(b, b + n);
  double sa = 0.0;
  double sb = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pa[i] = std::exp(static_cast<double>(a[i] - ma));
    pb[i] = std::exp(static_cast<double>(b[i] - mb));
    sa += pa[i];
    sb += pb[i];
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(a[i] - b[i])));
  }
  double kl = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pa[i] /= sa;
    pb[i] /= sb;
    kl += pa[i] * (std::log(pa[i] + 1e-30) - std::log(pb[i] + 1e-30));
  }
  return {static_cast<std::uint32_t>(std::max_element(a, a + n) - a),
          static_cast<std::uint32_t>(std::max_element(b, b + n) - b), kl,
          max_abs};
}

/// Compares next-token distributions with a llama.cpp `--kl-divergence-base`
/// file: "_logits_", n_ctx, n_vocab and n_chunk as int32, the chunks'
/// n_ctx * n_chunk token ids, then for every chunk the log-probs of positions
/// n_ctx / 2 .. n_ctx - 2, each as a float scale, a float minimum log-prob and
/// n_vocab (padded to even) 16-bit codes: log-prob = minimum + scale * code,
/// code 0 at or below the minimum. Runs every chunk from an empty session in
/// prefill steps of `step` tokens and reports both perplexities, the mean
/// KL(file || gufo) and how often the top tokens agree; with `save`, also
/// writes Gufo's log-probs in the same format and quantization. Every rank
/// runs the same forwards; only a reporting rank reads the log-probs.
int EvaluateKld(q::rocm::Executor& executor, std::uint32_t vocab,
                const std::string& path, const std::string& save,
                std::uint32_t step, bool report) {
  std::ifstream in(path, std::ios::binary);
  std::array<char, 8> magic{};
  std::array<std::int32_t, 3> header{};
  if (!in.read(magic.data(), magic.size()) ||
      std::string_view(magic.data(), magic.size()) != "_logits_" ||
      !in.read(reinterpret_cast<char*>(header.data()),
               header.size() * sizeof(std::int32_t))) {
    std::fprintf(stderr, "%s is not a llama.cpp logits file\n", path.c_str());
    return 1;
  }
  const auto [n_ctx, n_vocab, n_chunk] = header;
  if (n_ctx < 4 || n_chunk < 1 || n_vocab < 1 ||
      static_cast<std::uint32_t>(n_vocab) != vocab) {
    std::fprintf(stderr, "logits file geometry does not match the model\n");
    return 1;
  }
  std::vector<std::int32_t> tokens(static_cast<std::size_t>(n_ctx) * n_chunk);
  if (!in.read(
          reinterpret_cast<char*>(tokens.data()),
          static_cast<std::streamsize>(tokens.size() * sizeof(std::int32_t)))) {
    std::fprintf(stderr, "logits file has no tokens\n");
    return 1;
  }
  const auto ctx = static_cast<std::uint32_t>(n_ctx);
  const std::uint32_t first = ctx / 2;
  const std::size_t codes = 2 * ((static_cast<std::size_t>(vocab) + 1) / 2) + 4;
  std::vector<std::uint16_t> base(codes);
  std::vector<std::uint16_t> mine(codes, 0);
  std::ofstream out;
  if (report && !save.empty()) {
    out.open(save, std::ios::binary);
    out.write(magic.data(), magic.size());
    out.write(reinterpret_cast<const char*>(header.data()),
              header.size() * sizeof(std::int32_t));
    out.write(
        reinterpret_cast<const char*>(tokens.data()),
        static_cast<std::streamsize>(tokens.size() * sizeof(std::int32_t)));
  }
  std::vector<float> logits(static_cast<std::size_t>(step) * vocab);
  double nll = 0.0;
  double nll_base = 0.0;
  double kl = 0.0;
  std::size_t same_top = 0;
  std::size_t count = 0;
  std::string error;
  for (std::int32_t chunk = 0; chunk < n_chunk; ++chunk) {
    auto session = executor.CreateSession(
        gufo::core::SessionMode::kAutoregressive, ctx, &error);
    if (!session) {
      std::fprintf(stderr, "session failed: %s\n", error.c_str());
      return 1;
    }
    const std::int32_t* seq =
        tokens.data() + static_cast<std::size_t>(chunk) * ctx;
    for (std::uint32_t off = 0; off < ctx; off += step) {
      const std::uint32_t n = std::min(step, ctx - off);
      const bool scored = off + n > first && off + 1 < ctx;
      if (!executor.Forward(*session,
                            std::span<const std::int32_t>(seq + off, n),
                            scored ? n : 0, scored ? logits.data() : nullptr,
                            q::rocm::Executor::ForwardMode::kPrefill, &error)) {
        std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
        return 1;
      }
      if (!scored || !report) {
        continue;
      }
      for (std::uint32_t r = 0; r < n; ++r) {
        const std::uint32_t pos = off + r;
        if (pos < first || pos + 1 >= ctx) {
          continue;
        }
        if (!in.read(
                reinterpret_cast<char*>(base.data()),
                static_cast<std::streamsize>(codes * sizeof(std::uint16_t)))) {
          std::fprintf(stderr, "logits file ends early\n");
          return 1;
        }
        const float* row = logits.data() + static_cast<std::size_t>(r) * vocab;
        const float* top = std::max_element(row, row + vocab);
        double sum = 0.0;
        for (std::uint32_t i = 0; i < vocab; ++i) {
          sum += std::exp(static_cast<double>(row[i] - *top));
        }
        const double lse = *top + std::log(sum);
        const auto next = static_cast<std::uint32_t>(seq[pos + 1]);
        nll += lse - row[next];
        float scale = 0.0F;
        float minimum = 0.0F;
        std::memcpy(&scale, base.data(), sizeof(float));
        std::memcpy(&minimum, base.data() + 2, sizeof(float));
        const std::uint16_t* code = base.data() + 4;
        nll_base -= static_cast<double>(scale) * code[next] + minimum;
        // Like llama.cpp, skip tokens at the file's floor (code 0): their
        // stored log-prob is only an upper bound.
        double divergence = 0.0;
        std::uint32_t base_top = 0;
        for (std::uint32_t i = 0; i < vocab; ++i) {
          if (code[i] == 0) {
            continue;
          }
          const double lb = static_cast<double>(scale) * code[i] + minimum;
          divergence += std::exp(lb) * (lb - (row[i] - lse));
          if (code[i] > code[base_top]) {
            base_top = i;
          }
        }
        kl += divergence;
        same_top += base_top == static_cast<std::uint32_t>(top - row);
        ++count;
        if (out.is_open()) {
          // llama.cpp's quantization: 16 nats below the top logit in 16 bits.
          const float lowest =
              std::max(*std::min_element(row, row + vocab), *top - 16.0F);
          const float log_sum = static_cast<float>(std::log(sum));
          const float mine_scale = (*top - lowest) / 65535.0F;
          const float mine_min = lowest - *top - log_sum;
          std::memcpy(mine.data(), &mine_scale, sizeof(float));
          std::memcpy(mine.data() + 2, &mine_min, sizeof(float));
          for (std::uint32_t i = 0; i < vocab; ++i) {
            mine[4 + i] = mine_scale > 0.0F && row[i] > lowest
                              ? static_cast<std::uint16_t>(
                                    std::lround((row[i] - lowest) / mine_scale))
                              : 0;
          }
          out.write(
              reinterpret_cast<const char*>(mine.data()),
              static_cast<std::streamsize>(codes * sizeof(std::uint16_t)));
        }
      }
    }
  }
  if (report) {
    const double positions = static_cast<double>(count);
    std::printf(
        "kld chunks=%d positions=%zu ppl=%.4f ppl_base=%.4f mean_kld=%.6f "
        "top1_agree=%.2f%%\n",
        n_chunk, count, std::exp(nll / positions),
        std::exp(nll_base / positions), kl / positions,
        100.0 * static_cast<double>(same_top) / positions);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt = "The capital of France is";
  std::string dump_path;
  std::string mtp_path;
  std::string kld_base;
  std::string kld_save;
  std::string decode_text;
  std::string decode_ids_path;
  bool decode = false;
  bool mtp_audit = false;
  bool reference = false;
  bool cost_audit = false;
  std::uint32_t cost_concurrency = 0;
  std::optional<std::uint32_t> cost_depth;
  std::uint32_t batch = 512;
  std::uint32_t context = 4096;
  std::uint32_t tp_rank = 0;
  std::uint32_t tp_world_size = 1;
  std::string tp_bootstrap_host;
  std::uint16_t tp_bootstrap_port = 18515;
  std::uint32_t tp_device = 0;
  std::uint32_t tp_gid = 0;
  std::uint64_t tp_operation_id = 1;
#ifndef GUFO_ENABLE_TP2_RDMA
  (void)tp_operation_id;
#endif
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--mtp-model") {
      mtp_path = next();
    } else if (arg == "--mtp-audit") {
      mtp_audit = true;
    } else if (arg == "--cost-audit") {
      cost_audit = true;
      const auto value = next();
      const auto [end, ec] = std::from_chars(
          value.data(), value.data() + value.size(), cost_concurrency);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          (cost_concurrency != 0 && cost_concurrency != 1 &&
           cost_concurrency != 2 && cost_concurrency != 4 &&
           cost_concurrency != 6 && cost_concurrency != 8)) {
        std::fprintf(stderr, "--cost-audit requires C=0/1/2/4/6/8 (0 = all)\n");
        return 2;
      }
    } else if (arg == "--depth") {
      const auto value = next();
      std::uint32_t depth = 0;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), depth);
      if (ec != std::errc{} || end != value.data() + value.size()) {
        std::fprintf(stderr, "--depth requires a nonnegative integer\n");
        return 2;
      }
      cost_depth = depth;
    } else if (arg == "--tp-rank" || arg == "--tp-world-size" ||
               arg == "--tp-bootstrap-port" || arg == "--tp-device" ||
               arg == "--tp-gid-index") {
      const auto value = next();
      std::uint32_t parsed = 0;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), parsed);
      if (ec != std::errc{} || end != value.data() + value.size()) {
        std::fprintf(stderr, "%s requires an unsigned integer\n", arg.c_str());
        return 2;
      }
      if (arg == "--tp-rank") {
        tp_rank = parsed;
      } else if (arg == "--tp-world-size") {
        tp_world_size = parsed;
      } else if (arg == "--tp-bootstrap-port") {
        if (parsed == 0 || parsed > std::numeric_limits<std::uint16_t>::max()) {
          std::fprintf(stderr, "--tp-bootstrap-port is out of range\n");
          return 2;
        }
        tp_bootstrap_port = static_cast<std::uint16_t>(parsed);
      } else if (arg == "--tp-device") {
        if (parsed >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
          std::fprintf(stderr, "--tp-device is out of range\n");
          return 2;
        }
        tp_device = parsed;
      } else {
        tp_gid = parsed;
      }
    } else if (arg == "--tp-operation-id") {
      const auto value = next();
      const auto [end, ec] = std::from_chars(
          value.data(), value.data() + value.size(), tp_operation_id);
      if (ec != std::errc{} || end != value.data() + value.size()) {
        std::fprintf(stderr,
                     "--tp-operation-id requires an unsigned integer\n");
        return 2;
      }
    } else if (arg == "--tp-bootstrap-host") {
      tp_bootstrap_host = next();
    } else if (arg == "--prompt") {
      prompt = next();
    } else if (arg == "--prompt-file" || arg == "--decode-file") {
      const auto path = next();
      std::ifstream in(path, std::ios::binary);
      if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return 2;
      }
      std::string text((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
      if (in.bad()) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        return 2;
      }
      (arg == "--prompt-file" ? prompt : decode_text) = std::move(text);
      decode |= arg == "--decode-file";
    } else if (arg == "--decode-ids") {
      decode_ids_path = next();
    } else if (arg == "--reference") {
      reference = true;
    } else if (arg == "--batch" || arg == "--context") {
      const auto value = next();
      auto& count = arg == "--batch" ? batch : context;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), count);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          count == 0) {
        std::fprintf(stderr, "%s requires a positive integer\n", arg.c_str());
        return 2;
      }
    } else if (arg == "--dump") {
      dump_path = next();
    } else if (arg == "--kld-base") {
      kld_base = next();
    } else if (arg == "--kld-save") {
      kld_save = next();
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "--model is required\n");
    return 2;
  }
  if ((decode &&
       (decode_text.empty() || reference || mtp_audit || cost_audit)) ||
      (!decode && !decode_ids_path.empty())) {
    std::fprintf(stderr,
                 "--decode-file requires a nonempty continuation and cannot "
                 "be combined with other probes; --decode-ids requires it\n");
    return 2;
  }
  if ((mtp_audit || cost_audit) && mtp_path.empty()) {
    std::fprintf(stderr, "MTP audits require --mtp-model\n");
    return 2;
  }
  if (cost_audit && (mtp_audit || reference || !dump_path.empty())) {
    std::fprintf(stderr, "--cost-audit cannot be combined with other probes\n");
    return 2;
  }
  if (cost_depth && !cost_audit) {
    std::fprintf(stderr, "--depth requires --cost-audit\n");
    return 2;
  }
  if (tp_world_size != 1 && tp_world_size != 2) {
    std::fprintf(stderr, "--tp-world-size must be 1 or 2\n");
    return 2;
  }
  if (tp_world_size == 1 && tp_rank != 0) {
    std::fprintf(stderr, "single-rank probe requires --tp-rank 0\n");
    return 2;
  }
  if (tp_rank >= tp_world_size) {
    std::fprintf(stderr, "--tp-rank must be less than --tp-world-size\n");
    return 2;
  }
  if (tp_world_size == 2 && tp_rank == 1 && tp_bootstrap_host.empty()) {
    std::fprintf(stderr, "rank one requires --tp-bootstrap-host\n");
    return 2;
  }
  if (tp_device > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      hipSetDevice(static_cast<int>(tp_device)) != hipSuccess) {
    std::fprintf(stderr, "HIP device selection failed\n");
    return 1;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto weights = q::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }
  const auto& c = weights->config;
  if (cost_depth &&
      (c.context_length < 96 || *cost_depth > c.context_length - 96)) {
    std::fprintf(stderr, "--depth leaves no room for the cost-audit suffix\n");
    return 2;
  }
  std::optional<q::distributed::TpPartition> partition;
  std::shared_ptr<q::rocm::Communicator> communicator;
#ifdef GUFO_ENABLE_TP2_RDMA
  std::unique_ptr<OperationGuard> operation;
#endif
  if (tp_world_size == 2) {
    partition = q::distributed::TpPartition::Create(c.expert_ff, tp_rank,
                                                    tp_world_size, &error);
    if (!partition) {
      std::fprintf(stderr, "partition failed: %s\n", error.c_str());
      return 1;
    }
#ifdef GUFO_ENABLE_TP2_RDMA
    const q::rocm::IbrverbsConfig config{
        .rank = tp_rank,
        .world_size = tp_world_size,
        .bootstrap_host = tp_bootstrap_host,
        .bootstrap_port = tp_bootstrap_port,
        .device_index = tp_device,
        .gid_index = tp_gid,
    };
    communicator = q::rocm::CreateIbrverbsCommunicator(config, &error);
    if (!communicator) {
      std::fprintf(stderr, "RDMA communicator failed: %s\n", error.c_str());
      return 1;
    }
    operation = std::make_unique<OperationGuard>(communicator, tp_operation_id);
    if (!operation->Begin(&error)) {
      std::fprintf(stderr, "TP operation scope bind failed: %s\n",
                   error.c_str());
      return 1;
    }
#else
    std::fprintf(stderr, "this probe was built without TP2 RDMA support\n");
    return 2;
#endif
  }
  std::unique_ptr<q::NgramTable> ngram;
  if (c.ple_layer >= 0) {
    const auto& t = weights->ple_table;
    ngram = q::NgramTable::Open(
        reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, c.ple_head_dim, t.type, &error);
    if (!ngram) {
      std::fprintf(stderr, "n-gram table failed: %s\n", error.c_str());
      return 1;
    }
  }

  std::shared_ptr<gufo::core::GgufReader> mtp_reader;
  std::optional<q::MtpWeights> mtp_weights;
  if (!mtp_path.empty()) {
    mtp_reader = gufo::core::GgufReader::OpenFile(mtp_path, &error);
    if (mtp_reader)
      mtp_weights = q::MtpWeights::Bind(*mtp_reader, c, &error);
    if (!mtp_weights) {
      std::fprintf(stderr, "MTP bind failed: %s\n", error.c_str());
      return 1;
    }
  }
  if (partition) {
    const auto routed_plan = q::distributed::PlanRoutedBytes(
        *weights, *partition, mtp_weights ? &*mtp_weights : nullptr, &error);
    if (!routed_plan) {
      std::fprintf(stderr, "routed weight plan failed: %s\n", error.c_str());
      return 1;
    }
    std::printf("TP2 routed_bytes full=%zu local=%zu\\n",
                routed_plan->full_encoded_bytes,
                routed_plan->local_encoded_bytes);
  }
  const auto* partition_ptr = partition ? &*partition : nullptr;
  auto device = q::rocm::DeviceModel::Upload(
      *weights, *reader, mtp_weights ? &*mtp_weights : nullptr,
      mtp_reader.get(), &error, partition_ptr);
  if (!device) {
    std::fprintf(stderr, "upload failed: %s\n", error.c_str());
    return 1;
  }
  if (tp_world_size == 2) {
    std::printf("TP2 rank=%u world=%u expert_share=%u resident_bytes=%zu\n",
                tp_rank, tp_world_size, partition->ff_count,
                device->resident_bytes());
  }
  q::rocm::Executor::Options options;
  options.device_index = static_cast<int>(tp_device);
  options.max_batch = batch;
  options.max_logit_rows = std::min<std::uint32_t>(batch, 64);
  if (mtp_audit || cost_audit)
    options.max_speculative = 8;
  if (communicator) {
    options.all_reduce = q::rocm::Executor::TwoRankAllReduce(
        communicator, device->config().hidden_size);
    options.reduce_status =
        q::rocm::Executor::TwoRankReduceStatus(communicator);
    options.split_reduce = q::rocm::Executor::TwoRankSplitReduce(
        communicator, device->config().hidden_size);
  }

  if (cost_audit) {
    options.max_batch = 2048;
    options.max_logit_rows = 8;
  }
  auto executor =
      q::rocm::Executor::Create(*device, ngram.get(), options, &error);
  if (!executor) {
    std::fprintf(stderr, "executor failed: %s\n", error.c_str());
    return 1;
  }
  if (cost_audit) {
    try {
      const std::array<std::uint32_t, 3> depths{0, 4096, 32768};
      std::span<const std::uint32_t> selected_depths = depths;
      if (cost_depth)
        selected_depths = std::span(&*cost_depth, 1);
      AuditMtpCosts(*executor, *tokenizer, selected_depths, cost_concurrency);
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "cost audit failed: %s\n", e.what());
      return 1;
    }
  }
  if (mtp_audit) {
    try {
      AuditMtp(*executor, *device, *weights, *mtp_weights, *tokenizer,
               model_path);
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "MTP oracle failed: %s\n", e.what());
      return 1;
    }
  }
  if (!kld_base.empty()) {
    return EvaluateKld(*executor, c.vocab_size, kld_base, kld_save,
                       options.max_logit_rows, tp_rank == 0);
  }
  auto session = executor->CreateSession(
      gufo::core::SessionMode::kAutoregressive, context, &error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::int32_t> tokens;
  for (auto id : tokenizer->Encode(prompt)) {
    tokens.push_back(static_cast<std::int32_t>(id));
  }
  if (tokens.empty() || tokens.size() > context) {
    std::fprintf(stderr, "prompt must contain 1..%u tokens\n", context);
    return 2;
  }
  std::vector<std::int32_t> continuation;
  if (decode) {
    const auto ids = tokenizer->Encode(decode_text);
    continuation.assign(ids.begin(), ids.end());
    if (continuation.empty() || continuation.size() > context - tokens.size()) {
      std::fprintf(stderr, "complete continuation must fit in the context\n");
      return 2;
    }
  }
  std::printf("prompt tokens (%zu)\n", tokens.size());
  std::ofstream dump;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary);
    if (!dump) {
      std::fprintf(stderr, "cannot open logit dump %s\n", dump_path.c_str());
      return 1;
    }
  }

  std::vector<float> gpu_logits;
  std::size_t logit_rows = 0;
  // Chunks as the engine feeds them: under TP two batches run as a pair,
  // whose logits come from the second batch.
  const std::size_t step = executor->PrefillChunk();
  for (std::size_t off = 0; off < tokens.size(); off += step) {
    const std::size_t n = std::min<std::size_t>(step, tokens.size() - off);
    const auto rows = static_cast<std::uint32_t>(std::min<std::size_t>(
        n - executor->PairLead(static_cast<std::uint32_t>(n)),
        options.max_logit_rows));
    std::vector<float> out(static_cast<std::size_t>(rows) * c.vocab_size);
    if (!executor->Forward(
            *session, std::span<const std::int32_t>(tokens.data() + off, n),
            rows, out.data(), q::rocm::Executor::ForwardMode::kPrefill,
            &error)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const bool last = off + n == tokens.size();
    if (last) {
      gpu_logits = std::move(out);
      logit_rows = rows;
    }
  }
  if (decode) {
    // Matched-token decode: feed a fixed continuation one token at a time
    // through the decode path and dump each step's logits.
    std::vector<float> row(c.vocab_size);
    std::size_t steps = 0;
    for (const auto t : continuation) {
      if (!executor->Forward(*session, std::span<const std::int32_t>(&t, 1), 1,
                             row.data(),
                             q::rocm::Executor::ForwardMode::kDecode, &error)) {
        std::fprintf(stderr, "decode failed: %s\n", error.c_str());
        return 1;
      }
      if (dump.is_open())
        dump.write(reinterpret_cast<const char*>(row.data()),
                   static_cast<std::streamsize>(row.size() * sizeof(float)));
      ++steps;
    }
    if (dump.is_open()) {
      dump.close();
      if (!dump) {
        std::fprintf(stderr, "cannot write logit dump %s\n", dump_path.c_str());
        return 1;
      }
    }
    if (!decode_ids_path.empty()) {
      std::ofstream ids(decode_ids_path, std::ios::binary);
      ids.write(reinterpret_cast<const char*>(continuation.data()),
                static_cast<std::streamsize>(continuation.size() *
                                             sizeof(continuation[0])));
      ids.close();
      if (!ids) {
        std::fprintf(stderr, "cannot write token IDs %s\n",
                     decode_ids_path.c_str());
        return 1;
      }
    }
    std::printf("decode steps %zu after %zu prompt tokens\n", steps,
                tokens.size());
    return 0;
  }
  if (dump.is_open()) {
    dump.write(reinterpret_cast<const char*>(gpu_logits.data()),
               static_cast<std::streamsize>(gpu_logits.size() * sizeof(float)));
    dump.flush();
    if (!dump) {
      std::fprintf(stderr, "cannot write logit dump %s\n", dump_path.c_str());
      return 1;
    }
  }

  if (reference) {
    q::ReferenceModel ref(*weights, ngram.get(),
                          static_cast<std::uint32_t>(tokens.size()));
    std::vector<float> ref_logits(c.vocab_size);
    const std::size_t first = tokens.size() - logit_rows;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (!ref.Step(tokens[i], ref_logits, {}, &error)) {
        std::fprintf(stderr, "reference failed: %s\n", error.c_str());
        return 1;
      }
      if (i < first) {
        continue;
      }
      const Stats st =
          Compare(ref_logits.data(),
                  gpu_logits.data() + (i - first) * c.vocab_size, c.vocab_size);
      std::printf("pos %3zu ref %6u gpu %6u %s  KL %.5f  max|d| %.3f\n", i,
                  st.argmax_a, st.argmax_b,
                  st.argmax_a == st.argmax_b ? "OK  " : "DIFF", st.kl,
                  st.max_abs);
    }
  }

  return 0;
}
