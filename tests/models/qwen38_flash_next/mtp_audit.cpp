// Independent scalar MTP oracle and model-level cost calibration.
// Reports aggregate errors only; no generated logit fixtures.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/reference.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace {
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Hip(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}
struct Arena {
  std::vector<void*> allocations;
  template<typename T>
  T* Make(std::size_t count) {
    T* p = nullptr;
    Hip(hipMalloc(&p, count * sizeof(T)));
    allocations.push_back(p);
    return p;
  }
  ~Arena() {
    for (auto* p : allocations)
      (void)hipFree(p);
  }
};
struct Trace {
  std::vector<float> norm, fused, attention, hidden, head, ffn_input,
      ffn_output;
  q::MtpTrace spans;
  explicit Trace(const q::Config& c)
      : norm(c.HcDim()),
        fused(c.HcDim()),
        attention(c.HcDim()),
        hidden(c.HcDim()),
        head(c.hidden_size),
        ffn_input(c.hidden_size),
        ffn_output(c.hidden_size),
        spans{norm, fused, attention, hidden, head, ffn_input, ffn_output} {}
};
void Compare(std::span<const float> actual, std::span<const float> expected,
             const char* stage, double limit) {
  Require(actual.size() == expected.size(), "oracle shape mismatch");
  double error = 0, norm = 0, maximum = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    Require(std::isfinite(actual[i]) && std::isfinite(expected[i]),
            std::string(stage) + " nonfinite value");
    const double delta = double(actual[i]) - expected[i];
    error += delta * delta;
    norm += double(expected[i]) * expected[i];
    maximum = std::max(maximum, std::abs(delta));
  }
  const auto relative = std::sqrt(error / std::max(norm, 1e-30));
  std::printf("  %s relative_rms=%.7f max_abs=%.7f\n", stage, relative,
              maximum);
  Require(relative <= limit,
          std::string(stage) + " exceeds scalar oracle tolerance");
}
}  // namespace

void AuditMtp(q::rocm::Executor& exec, const q::rocm::DeviceModel& device,
              const q::ModelWeights& weights, const q::MtpWeights& mtp,
              const gufo::tokenization::QwenTokenizer& tokenizer,
              const std::filesystem::path& path) {
  const auto& c = exec.config();
  const auto stream = exec.stream();
  std::string error;
  // Verify the upload split byte for byte against the ORIGINAL GGUF rows.
  const auto& combined = mtp.block.nextn_eh_proj;
  const auto half_row = combined.RowBytes() / 2;
  for (unsigned part = 0; part < 2; ++part) {
    const auto& tensor =
        part ? device.mtp().nextn_fc_hidden : device.mtp().nextn_fc_embedding;
    std::vector<std::uint8_t> bytes(half_row * combined.rows);
    Hip(hipMemcpy(bytes.data(), tensor.data, bytes.size(),
                  hipMemcpyDeviceToHost));
    for (std::size_t r = 0; r < combined.rows; ++r) {
      const auto* original = static_cast<const std::uint8_t*>(combined.data) +
                             r * combined.RowBytes() + part * half_row;
      Require(std::memcmp(bytes.data() + r * half_row, original, half_row) == 0,
              "split projection changed GGUF weight bytes");
    }
  }
  Arena arena;
  auto* input = arena.Make<float>(c.HcDim());
  auto* head_input = arena.Make<float>(c.hidden_size);
  auto* logits = arena.Make<float>(c.vocab_size);
  auto* quantized =
      arena.Make<std::uint8_t>(qfn_mmq_q8_1_bytes(1, c.hidden_size));
  std::vector<float> all_logits(c.vocab_size), initial(c.HcDim());
  // Direct adversarial norm check: unequal HC magnitudes distinguish a
  // single 10240-wide denominator from four independent denominators.
  for (std::size_t i = 0; i < initial.size(); ++i)
    initial[i] = std::sin(float(i) * 0.013F) * float(1U << (i / c.hidden_size));
  std::vector<float> expected_norm(initial), actual_norm(initial.size());
  q::cpu::RmsNorm(expected_norm,
                  static_cast<const float*>(mtp.block.nextn_hnorm.data),
                  c.rms_eps);
  Hip(hipMemcpyAsync(input, initial.data(), initial.size() * sizeof(float),
                     hipMemcpyHostToDevice, stream));
  q::rocm::RmsNormRows(input, device.mtp().nextn_hnorm.f32(), input, 1,
                       c.HcDim(), 1, c.rms_eps, stream);
  Hip(hipMemcpyAsync(actual_norm.data(), input, initial.size() * sizeof(float),
                     hipMemcpyDeviceToHost, stream));
  Hip(hipStreamSynchronize(stream));
  Compare(actual_norm, expected_norm, "adversarial-full-hidden-norm", 2e-6);
  std::array<std::unique_ptr<q::rocm::Session>, 2> sessions, serial;
  std::array<std::unique_ptr<q::ReferenceModel>, 2> references;
  std::array<std::int32_t, 2> next{42, 43};
  auto vision =
      gufo::models::qwen::vision::Encoder::Open(path, {}, c.hidden_size);
  for (unsigned sequence = 0; sequence < 2; ++sequence) {
    sessions[sequence] =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 64, &error);
    serial[sequence] =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 64, &error);
    references[sequence] = std::make_unique<q::ReferenceModel>(
        weights, nullptr, 64, q::ReferenceModel::Storage::kDecode);
    Require(sessions[sequence] && serial[sequence], error);
    gufo::models::qwen::vision::RopeLayout rope;
    if (sequence == 1) {
      Require(vision != nullptr,
              "MTP image oracle requires the vision sidecar");
      auto prompt = std::make_shared<gufo::models::qwen::vision::Prompt>();
      rope.images.push_back({0, 2, 2});
      prompt->rope = rope;
      gufo::core::Image pixels;
      pixels.width = pixels.height = 64;
      pixels.pixels.assign(64 * 64 * 3, 127);
      prompt->images.push_back({std::move(pixels), rope.images.front()});
      sessions[sequence]->ConfigureVision(prompt, vision, stream);
      serial[sequence]->ConfigureVision(prompt, vision, stream);
    }
    auto origin =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 64, &error);
    Require(origin != nullptr, error);
    const auto encoded = tokenizer.Encode(
        sequence == 0
            ? "The capital of France is Paris. Explain why the sky is blue."
            : "Write the word red repeatedly, separated by spaces. red red "
              "red");
    const std::vector<std::int32_t> prefix(encoded.begin(), encoded.end());
    const auto kept =
        std::min<std::size_t>(prefix.size(), exec.max_speculative());
    std::vector<float> kept_hidden(kept * c.HcDim());
    Require(exec.Forward(*origin, prefix, 0, nullptr,
                         q::rocm::Executor::ForwardMode::kPrefill, &error) &&
                exec.CopyTrunkHidden(*origin, kept_hidden, &error),
            error);
    std::copy_n(kept_hidden.end() - c.HcDim(), c.HcDim(), initial.begin());
    origin.reset();
    Hip(hipMemcpyAsync(input, initial.data(), initial.size() * sizeof(float),
                       hipMemcpyHostToDevice, stream));
    for (unsigned step = 0; step < 4; ++step) {
      const std::int32_t token = sequence == 1 && step < 3
                                     ? gufo::models::qwen::vision::kImageToken
                                     : 42 + step + sequence;
      Trace cpu(c), gpu(c), replay(c);
      q::MtpCandidateLogits candidates;
      Require(exec.MtpForward(*sessions[sequence], std::span(&token, 1),
                              step == 0 ? 0 : -1,
                              {.candidates = &candidates, .trace = &gpu.spans},
                              &error, step == 0 ? input : nullptr),
              error);
      Require(exec.MtpForward(*serial[sequence], std::span(&token, 1),
                              step == 0 ? 0 : -1, {.trace = &replay.spans},
                              &error, step == 0 ? input : nullptr),
              error);
      Require(gpu.hidden == replay.hidden, "MTP head evaluation altered carry");
      Require(references[sequence]->MtpStep(mtp, token, initial, {}, &cpu.spans,
                                            &rope, &error, &gpu.spans),
              error);
      initial = gpu.hidden;
      std::printf("MTP oracle sequence=%u step=%u image=%u\n", sequence, step,
                  sequence);
      // Independent scalar execution includes storage rounding. Remaining
      // differences come from reductions and nonlinear approximations.
      Compare(gpu.norm, cpu.norm, "full-hidden-norm", 2e-6);
      Compare(gpu.fused, cpu.fused, "split-fusion", 0.002);
      Compare(gpu.attention, cpu.attention, "attention-state", 0.002);
      Compare(gpu.hidden, cpu.hidden, "recursive-carry", 0.00005);
      Compare(gpu.head, cpu.head, "head-mixer", 0.00005);
      // Full Q8 vocabulary baseline uses the exact production dot arithmetic;
      // independent host sorting checks candidate selection and vocabulary
      // tail.
      Hip(hipMemcpyAsync(head_input, gpu.head.data(),
                         gpu.head.size() * sizeof(float), hipMemcpyHostToDevice,
                         stream));
      Require(qfn_mmq_quantize_q8_1(head_input, quantized, 1, c.hidden_size,
                                    stream) == 0 &&
                  qfn_mmq_q8_0_dense_vec_preq(device.output().data, nullptr,
                                              quantized, logits, c.vocab_size,
                                              1, c.hidden_size, stream) == 0,
              "full Q8 oracle projection failed");
      Hip(hipMemcpyAsync(all_logits.data(), logits,
                         all_logits.size() * sizeof(float),
                         hipMemcpyDeviceToHost, stream));
      Hip(hipStreamSynchronize(stream));
      std::vector<std::uint32_t> ids(c.vocab_size);
      std::iota(ids.begin(), ids.end(), 0);
      std::partial_sort(ids.begin(), ids.begin() + candidates.size, ids.end(),
                        [&](auto a, auto b) {
                          return all_logits[a] != all_logits[b]
                                     ? all_logits[a] > all_logits[b]
                                     : a < b;
                        });
      for (std::size_t i = 0; i < candidates.size; ++i)
        Require(candidates.ids[i] == ids[i] &&
                    candidates.logits[i] == all_logits[ids[i]],
                "MTP candidates differ from full Q8 head");
      next[sequence] = candidates.ids[0];
    }
  }
  // Continue two unrelated caches and recursive hidden streams in one batch.
  for (unsigned step = 0; step < 3; ++step) {
    std::array<q::rocm::Executor::MtpBatchItem, 2> batch{};
    for (unsigned i = 0; i < 2; ++i)
      batch[i] = {sessions[i].get(), std::span(&next[i], 1), -1};
    Require(exec.MtpForwardBatch(batch, &error), error);
    std::array<q::MtpCandidateLogits, 2> candidates;
    std::array<q::rocm::Executor::MtpHeadItem, 2> heads{};
    std::array<std::int32_t, 2> greedy{};
    for (unsigned i = 0; i < 2; ++i)
      heads[i] = {
          sessions[i].get(),
          {.token = &greedy[i],
           .candidates = step == 2 || i == 1 ? &candidates[i] : nullptr}};
    Require(exec.MtpHeads(heads, &error), error);
    for (unsigned i = 0; i < 2; ++i) {
      q::MtpCandidateLogits scalar;
      Require(exec.MtpForward(*serial[i], std::span(&next[i], 1), -1,
                              {.candidates = &scalar}, &error),
              error);
      Require(greedy[i] == static_cast<std::int32_t>(scalar.ids[0]),
              "batched greedy head differs from full Q8 candidates");
      if (step == 2 || i == 1)
        Require(
            scalar.size == candidates[i].size &&
                scalar.ids == candidates[i].ids &&
                scalar.logits == candidates[i].logits,
            "batched MTP body/head differs from independent single session");
      next[i] = scalar.ids[0];
    }
  }
  Require(vision->ResidentBytes() == 0,
          "MTP encoded pixels instead of embedding shifted text IDs");
  // Identical known trunk rows must build identical predictor KV regardless
  // of prefill segmentation. Supply the final row separately so obsolete
  // residuals from cache-only catch-up cannot mask a KV discrepancy.
  constexpr unsigned replay_rows = 33;
  std::vector<float> replay_hidden((replay_rows + 1) * c.HcDim());
  for (unsigned row = 0; row <= replay_rows; ++row) {
    for (std::size_t h = 0; h < c.HcDim(); ++h)
      replay_hidden[row * c.HcDim() + h] =
          initial[h] + 0.01F * std::sin(float(h + row) * 0.17F);
  }
  auto* replay_input = arena.Make<float>(replay_hidden.size());
  Hip(hipMemcpyAsync(replay_input, replay_hidden.data(),
                     replay_hidden.size() * sizeof(float),
                     hipMemcpyHostToDevice, stream));
  std::array<std::int32_t, replay_rows + 1> replay_tokens;
  std::iota(replay_tokens.begin(), replay_tokens.end(), 42);
  q::MtpCandidateLogits replay_reference;
  for (const bool kv_only : {false, true}) {
    for (const auto chunk : {1U, 8U, 9U, 32U, 33U}) {
      auto replay_session =
          exec.CreateSession(gufo::core::SessionMode::kSpeculative, 64, &error);
      Require(replay_session != nullptr, error);
      for (unsigned offset = 0; offset < replay_rows; offset += chunk) {
        const auto count = std::min(chunk, replay_rows - offset);
        Require(exec.MtpForward(*replay_session,
                                std::span(replay_tokens).subspan(offset, count),
                                0, {.kv_only = kv_only}, &error,
                                replay_input + offset * c.HcDim()),
                error);
      }
      if (kv_only) {
        q::MtpCandidateLogits unused;
        q::rocm::Executor::MtpHeadItem head{replay_session.get(),
                                            {.candidates = &unused}};
        Require(!exec.MtpHeads(std::span(&head, 1), &error),
                "head accepted an omitted predictor residual");
        Require(
            !exec.MtpForward(*replay_session, std::span(replay_tokens).last(1),
                             -1, {}, &error),
            "recursive proposal accepted an omitted predictor residual");
        q::rocm::Executor::MtpBatchItem body{
            replay_session.get(), std::span(replay_tokens).last(1), -1};
        Require(!exec.MtpForwardBatch(std::span(&body, 1), &error),
                "batched proposal accepted an omitted predictor residual");
      }
      q::MtpCandidateLogits actual;
      Require(exec.MtpForward(*replay_session, std::span(replay_tokens).last(1),
                              0, {.candidates = &actual}, &error,
                              replay_input + replay_rows * c.HcDim()),
              error);
      if (!kv_only && chunk == 1)
        replay_reference = actual;
      else
        Require(actual.ids == replay_reference.ids &&
                    actual.logits == replay_reference.logits,
                "prefill segmentation changed MTP candidate probabilities");
      std::printf("MTP cache replay chunk=%u kv_only=%d candidates_exact=1\n",
                  chunk, kv_only);
    }
  }
  // Short catch-up compacts only the final FFN rows. Compare it against
  // full independent predictor execution, including the one-row arithmetic
  // tail that must retain its original shape.
  for (unsigned i = 0; i < 2; ++i) {
    sessions[i]->ConfigureVision(nullptr, nullptr, stream);
    serial[i]->ConfigureVision(nullptr, nullptr, stream);
  }
  for (const auto widths :
       {std::array{2U, 7U}, std::array{8U, 8U}, std::array{3U, 7U}}) {
    std::array<std::vector<std::int32_t>, 2> tokens;
    std::array<q::rocm::Executor::MtpBatchItem, 2> batch;
    std::array<q::MtpCandidateLogits, 2> expected, actual;
    std::array<q::rocm::Executor::MtpHeadItem, 2> heads;
    for (unsigned i = 0; i < 2; ++i) {
      tokens[i].resize(widths[i]);
      std::iota(tokens[i].begin(), tokens[i].end(), 42 + 9 * i);
      for (auto* session : {sessions[i].get(), serial[i].get()})
        Require(exec.Forward(*session, tokens[i], 0, nullptr,
                             q::rocm::Executor::ForwardMode::kDecode, &error),
                error);
      batch[i] = {sessions[i].get(), tokens[i], 0};
      heads[i] = {sessions[i].get(), {.candidates = &actual[i]}};
      Require(exec.MtpForward(*serial[i], tokens[i], 0,
                              {.candidates = &expected[i]}, &error),
              error);
    }
    Require(exec.MtpForwardBatch(batch, &error) && exec.MtpHeads(heads, &error),
            error);
    for (unsigned i = 0; i < 2; ++i) {
      Require(actual[i].ids == expected[i].ids &&
                  actual[i].logits == expected[i].logits,
              "short catch-up changed full-head candidates");
      const auto token = static_cast<std::int32_t>(expected[i].ids[0]);
      Trace full(c), tail(c);
      Require(exec.MtpForward(*serial[i], {&token, 1}, -1,
                              {.trace = &full.spans}, &error) &&
                  exec.MtpForward(*sessions[i], {&token, 1}, -1,
                                  {.trace = &tail.spans}, &error),
              error);
      Require(full.norm == tail.norm && full.fused == tail.fused &&
                  full.attention == tail.attention &&
                  full.ffn_input == tail.ffn_input &&
                  full.ffn_output == tail.ffn_output &&
                  full.hidden == tail.hidden && full.head == tail.head,
              "short catch-up changed recursive predictor stages");
    }
  }
  std::puts("MTP short catch-up: ragged/full candidates and carry exact");
  // A full predictor forward is the independent execution control for
  // headless catch-up. Only its final residual is carried; all KV/indexer
  // rows must still survive. Cross the sparse-attention boundary and then
  // compare recursive proposals, not only the final argmax. Cover the minimum
  // 96-row tail tile, a two-tile tail, and aligned/ragged large chunks.
  std::vector<unsigned> counts{224U, 257U, 2047U, 2048U};
  for (auto& count : counts)
    count = std::min(exec.max_batch(), count);
  counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
  for (const unsigned count : counts) {
    Require(count > 32, "catch-up audit requires --batch greater than 32");
    const unsigned rounds = c.indexer_top_k / count + 2;
    const unsigned capacity = rounds * (count + 1) + 1;
    auto full = exec.CreateSession(gufo::core::SessionMode::kSpeculative,
                                   capacity, &error);
    auto tail = exec.CreateSession(gufo::core::SessionMode::kSpeculative,
                                   capacity, &error);
    Require(full && tail, error);
    std::vector<float> hidden(std::size_t(count) * c.HcDim());
    for (std::size_t i = 0; i < hidden.size(); ++i)
      hidden[i] = initial[i % initial.size()] *
                  (0.8F + float((i / initial.size()) % 11) * 0.04F);
    auto* source = arena.Make<float>(hidden.size());
    Hip(hipMemcpy(source, hidden.data(), hidden.size() * sizeof(float),
                  hipMemcpyHostToDevice));
    const auto pattern = tokenizer.Encode("Red, blue. Explain virtual memory.");
    Require(!pattern.empty(), "empty catch-up audit prompt");
    std::vector<std::int32_t> tokens(count);
    for (unsigned round = 0; round < rounds; ++round) {
      for (unsigned i = 0; i < count; ++i)
        tokens[i] = pattern[(round * count + i) % pattern.size()];
      const auto exact = [&](const auto& expected, const auto& actual,
                             const char* stage) {
        if (expected != actual) {
          std::fprintf(stderr, "catch-up round=%u rows=%u stage=%s\n", round,
                       count, stage);
          Compare(actual, expected, stage, 0.0);
        }
      };
      Trace full_row(c), tail_row(c);
      q::MtpCandidateLogits expected, actual;
      Require(
          exec.MtpForward(*full, tokens, 0,
                          {.candidates = &expected, .trace = &full_row.spans},
                          &error, source) &&
              exec.MtpForward(*tail, tokens, 0, {.trace = &tail_row.spans},
                              &error, source),
          error);
      exact(full_row.norm, tail_row.norm, "wide hidden norm");
      exact(full_row.fused, tail_row.fused, "wide fusion");
      exact(full_row.attention, tail_row.attention, "wide attention");
      exact(full_row.ffn_input, tail_row.ffn_input, "wide FFN input");
      exact(full_row.ffn_output, tail_row.ffn_output, "wide FFN output");
      exact(full_row.hidden, tail_row.hidden, "wide recursive carry");
      exact(full_row.head, tail_row.head, "wide head mixer");
      const std::array head{
          q::rocm::Executor::MtpHeadItem{tail.get(), {.candidates = &actual}}};
      Require(exec.MtpHeads(head, &error), error);
      Require(expected.size == actual.size && expected.ids == actual.ids &&
                  expected.logits == actual.logits,
              "headless catch-up changed full-head candidates");
      const auto next = static_cast<std::int32_t>(expected.ids[0]);
      Trace a(c), b(c);
      Require(
          exec.MtpForward(*full, {&next, 1}, -1, {.trace = &a.spans}, &error) &&
              exec.MtpForward(*tail, {&next, 1}, -1, {.trace = &b.spans},
                              &error),
          error);
      exact(a.norm, b.norm, "catch-up hidden norm");
      exact(a.fused, b.fused, "catch-up fusion");
      exact(a.attention, b.attention, "catch-up attention");
      exact(a.hidden, b.hidden, "catch-up recursive carry");
      exact(a.head, b.head, "catch-up head mixer");
    }
    std::puts("MTP catch-up: full/tail candidates and recursive stages exact");
  }
  std::puts(
      "MTP oracle PASS: full-width norm, byte-exact split, attention, "
      "recursive carry, full-Q8 head, image IDs and independent batches");
}

// Model-level cost calibration. Restore identical real states for every
// shape, warm allocation and C1 graph capture, then measure each stage.
// Report the median complete cycle of three warmed samples: allocator stalls
// on large cohorts can outlive the first pass after changing width.
// Timings never enter the production sampler at runtime.
void AuditMtpCosts(q::rocm::Executor& exec,
                   const gufo::tokenization::QwenTokenizer& tokenizer,
                   std::span<const std::uint32_t> depths,
                   std::uint32_t selected_concurrency) {
  using Clock = std::chrono::steady_clock;
  const auto elapsed = [](auto start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
  };
  const auto pattern = tokenizer.Encode(
      "Virtual memory maps pages to storage. Red, green, blue. ");
  std::string error;
  const auto vocab = exec.config().vocab_size;
  std::vector<float> logits(vocab);
  for (const auto depth : depths) {
    const auto prefix_size = depth + 32;
    const auto capacity = prefix_size + 64;
    std::vector<std::int32_t> prefix(prefix_size);
    for (std::size_t i = 0; i < prefix.size(); ++i)
      prefix[i] = pattern[i % pattern.size()];
    auto base = exec.CreateSession(gufo::core::SessionMode::kSpeculative,
                                   capacity, &error);
    Require(base != nullptr, error);
    for (std::size_t offset = 0; offset < prefix.size();) {
      const auto n =
          std::min<std::size_t>(exec.max_batch(), prefix.size() - offset);
      Require(
          exec.Forward(*base, std::span(prefix).subspan(offset, n), 0, nullptr,
                       q::rocm::Executor::ForwardMode::kPrefill, &error),
          error);
      const auto next = prefix[(offset + n) % prefix.size()];
      Require(
          exec.MtpForward(*base, std::span(&next, 1),
                          std::min<std::size_t>(n, exec.max_speculative()) - 1,
                          {}, &error),
          error);
      offset += n;
    }
    std::vector<std::uint8_t> common(exec.SnapshotBytes(*base, 0));
    Require(exec.SaveSnapshot(*base, 0, common, &error), error);
    base.reset();
    for (const unsigned concurrency : {1, 2, 4, 6, 8}) {
      if (selected_concurrency != 0 && selected_concurrency != concurrency)
        continue;
      std::vector<std::unique_ptr<q::rocm::Session>> sessions;
      std::vector<std::vector<std::uint8_t>> snapshots;
      std::array<std::array<std::int32_t, 16>, 8> tails{};
      std::array<std::int32_t, 8> anchors{};
      const unsigned count = concurrency;
      for (unsigned i = 0; i < count; ++i) {
        sessions.push_back(exec.CreateSession(
            gufo::core::SessionMode::kSpeculative, capacity, &error));
        Require(sessions.back() != nullptr, error);
        q::rocm::Executor::SnapshotInfo info;
        Require(exec.RestoreSnapshot(*sessions.back(), common, &info, &error),
                error);
        const auto suffix = tokenizer.Encode(
            std::string{"Request "} + std::to_string(i) + ": explain " +
            std::array{"gravity", "a queue", "photosynthesis", "sorting",
                       "the alphabet", "Paris", "astronomy", "a mutex"}[i]);
        for (std::size_t j = 0; j < 16; ++j)
          tails[i][j] = suffix[j % suffix.size()];
        Require(exec.Forward(*sessions.back(), tails[i], 1, logits.data(),
                             q::rocm::Executor::ForwardMode::kPrefill, &error),
                error);
        anchors[i] =
            std::max_element(logits.begin(), logits.end()) - logits.begin();
        Require(exec.MtpForward(*sessions.back(), std::span(&anchors[i], 1), 7,
                                {}, &error),
                error);
        snapshots.emplace_back(exec.SnapshotBytes(*sessions.back(), 8));
        Require(
            exec.SaveSnapshot(*sessions.back(), 8, snapshots.back(), &error),
            error);
      }
      for (const unsigned width : {1, 2, 3, 4, 5, 6, 7, 8}) {
        std::array<std::array<double, 3>, 3> samples{};
        for (unsigned repetition = 0; repetition < 5; ++repetition) {
          std::vector<std::vector<std::int32_t>> replays(concurrency);
          std::vector<q::rocm::Executor::MtpBatchItem> catchup;
          std::array<std::int32_t, 8> next = anchors;
          for (unsigned i = 0; i < concurrency; ++i) {
            q::rocm::Executor::SnapshotInfo info;
            Require(exec.RestoreSnapshot(*sessions[i], snapshots[i], &info,
                                         &error, width - 1),
                    error);
            exec.MtpRewind(*sessions[i], sessions[i]->position() - width);
            replays[i].assign(tails[i].end() - width + 1, tails[i].end());
            replays[i].push_back(anchors[i]);
            catchup.push_back({sessions[i].get(), replays[i],
                               static_cast<std::int32_t>(8 - width)});
          }
          const auto catchup_start = Clock::now();
          Require(exec.MtpForwardBatch(catchup, &error), error);
          const double catchup_ms = elapsed(catchup_start);
          std::vector<std::vector<std::int32_t>> chains(concurrency);
          for (unsigned i = 0; i < concurrency; ++i)
            chains[i].push_back(anchors[i]);
          const auto proposal_start = Clock::now();
          for (unsigned step = 1; step < width; ++step) {
            if (step > 1) {
              std::vector<q::rocm::Executor::MtpBatchItem> body;
              for (unsigned i = 0; i < concurrency; ++i)
                body.push_back({sessions[i].get(), {&next[i], 1}, -1});
              Require(exec.MtpForwardBatch(body, &error), error);
            }
            std::vector<q::rocm::Executor::MtpHeadItem> heads;
            for (unsigned i = 0; i < concurrency; ++i)
              heads.push_back({sessions[i].get(), {.token = &next[i]}});
            Require(exec.MtpHeads(heads, &error), error);
            for (unsigned i = 0; i < concurrency; ++i)
              chains[i].push_back(next[i]);
          }
          const double proposal_ms = elapsed(proposal_start);
          const auto target_start = Clock::now();
          if (concurrency == 1) {
            Require(exec.Forward(*sessions[0], chains[0], width,
                                 width == 1 ? logits.data() : nullptr,
                                 width == 1
                                     ? q::rocm::Executor::ForwardMode::kDecode
                                     : q::rocm::Executor::ForwardMode::kVerify,
                                 &error),
                    error);
          } else {
            std::vector<q::rocm::Executor::BatchItem> items;
            for (unsigned i = 0; i < concurrency; ++i)
              items.push_back({sessions[i].get(), chains[i], width > 1});
            Require(exec.ForwardBatch(items, &error), error);
          }
          for (unsigned i = 0; i < concurrency; ++i) {
            if (concurrency > 1)
              Require(exec.SelectBatchLogits(
                          i * width, width,
                          width == 1 ? logits.data() : nullptr, &error),
                      error);
            if (width > 1)
              Require(exec.Rollback(*sessions[i], width, &error, logits.data()),
                      error);
          }
          const double target_ms = elapsed(target_start);
          if (repetition >= 2)
            samples[repetition - 2] = {catchup_ms, proposal_ms, target_ms};
        }
        const auto total = [](const auto& sample) {
          return sample[0] + sample[1] + sample[2];
        };
        std::sort(
            samples.begin(), samples.end(),
            [&](const auto& a, const auto& b) { return total(a) < total(b); });
        const auto& sample = samples[1];
        std::printf(
            "MTP_COST depth=%u C=%u width=%u catchup_ms=%.4f "
            "proposal_ms=%.4f target_ms=%.4f total_ms=%.4f\n",
            depth, concurrency, width, sample[0], sample[1], sample[2],
            total(sample));
        std::fflush(stdout);
      }
    }
  }
}
