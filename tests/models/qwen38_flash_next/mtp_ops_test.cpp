#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

struct DeviceDelete {
  void operator()(void* pointer) const { (void)hipFree(pointer); }
};

template<typename T>
auto Allocate(std::size_t count) {
  T* pointer = nullptr;
  CheckHip(hipMalloc(&pointer, count * sizeof(T)), "allocate");
  return std::unique_ptr<T, DeviceDelete>(pointer);
}

void CheckPenaltyArgmax(std::uint32_t vocab) {
  namespace sampling = gufo::sampling;
  // Ragged vocabularies end at the allocation boundary, with every MTP width.
  for (unsigned rows = 1; rows <= 7; ++rows) {
    for (unsigned mode = 0; mode < 5; ++mode) {
      sampling::SamplingConfig config{
          .seed = 73,
          .repeat_penalty = mode == 1 ? 0.7F : 1.3F,
          .repeat_last_n = mode == 2 ? 0U : 3U,
          .frequency_penalty = mode == 1 ? -0.4F : 0.2F,
          .presence_penalty = mode == 1 ? -1.5F : 1.5F};
      if (mode >= 3) {
        config.repeat_penalty = 1;
        config.frequency_penalty = 0;
        config.presence_penalty = 1.0e-8F;
      }
      sampling::SamplerState sampler(config,
                                     std::vector<sampling::TokenId>{0, 1, 0});
      sampler.Accept(0);
      sampler.Accept(vocab - 1);
      std::vector<float> logits(std::size_t(rows) * vocab);
      std::vector<sampling::TokenPenalty> counts;
      q::GreedyPenaltyRows batch{};
      std::vector<int> expected(rows, -1);
      for (unsigned row = 0; row < rows; ++row) {
        auto values =
            std::span(logits).subspan(std::size_t(row) * vocab, vocab);
        for (unsigned i = 0; i < vocab; ++i)
          values[i] =
              float(int((i * 7919U + row * 31U) % 65521U) - 32768) / 32768;
        values[0] = values[vocab - 1] = mode == 2 ? -1.e35F : 1.0F;
        if (mode == 2)
          std::fill(values.begin(), values.end(), -1.e35F);
        if (mode >= 3) {
          std::fill(values.begin(), values.end(), -100.0F);
          // FP32 penalty arithmetic incorrectly picks token zero here.
          values[0] = values[1] = 1.0F;
          values[vocab - 1] = std::numeric_limits<float>::infinity();
        }
        if (mode == 4)
          std::fill(values.begin(), values.end(),
                    std::numeric_limits<float>::quiet_NaN());
        const auto penalties = sampler.penalties();
        counts.insert(counts.end(), penalties.begin(), penalties.end());
        batch.offsets[row + 1] = counts.size();
        try {
          expected[row] = sampler.Sample(values);
        } catch (const std::runtime_error&) {
          if (mode != 4)
            throw;
        }
        sampler.Accept((row + 2) % vocab);
      }
      auto device_logits = Allocate<float>(logits.size());
      auto device_counts = Allocate<sampling::TokenPenalty>(counts.size());
      auto partial =
          Allocate<q::PenaltyArgmaxCandidate>(rows * q::kArgmaxParts);
      auto output = Allocate<q::ArgmaxCandidate>(rows);
      CheckHip(hipMemcpy(device_logits.get(), logits.data(),
                         logits.size() * sizeof(float), hipMemcpyHostToDevice),
               "penalty logits");
      CheckHip(
          hipMemcpy(device_counts.get(), counts.data(),
                    counts.size() * sizeof(counts[0]), hipMemcpyHostToDevice),
          "penalty histories");
      batch.penalties = device_counts.get();
      q::PenalizedArgmax(device_logits.get(), batch, config.repeat_penalty,
                         config.frequency_penalty, config.presence_penalty,
                         partial.get(), output.get(), rows, vocab, nullptr);
      std::vector<q::ArgmaxCandidate> actual(rows);
      CheckHip(hipMemcpy(actual.data(), output.get(), rows * sizeof(actual[0]),
                         hipMemcpyDeviceToHost),
               "penalty predictions");
      for (unsigned row = 0; row < rows; ++row) {
        if (actual[row].index != expected[row] ||
            (expected[row] < 0 && std::isfinite(actual[row].value)))
          throw std::runtime_error("penalty argmax differs from CPU: vocab=" +
                                   std::to_string(vocab) +
                                   " row=" + std::to_string(row) +
                                   " mode=" + std::to_string(mode));
      }
      std::vector<float> retained(logits.size());
      CheckHip(
          hipMemcpy(retained.data(), device_logits.get(),
                    retained.size() * sizeof(float), hipMemcpyDeviceToHost),
          "retained logits");
      if (std::memcmp(retained.data(), logits.data(),
                      logits.size() * sizeof(float)))
        throw std::runtime_error("penalty selection modified frontier logits");
    }
  }
  std::cout << "penalty argmax CPU exact: vocab=" << vocab << " widths=1..7\n";
}

void CheckArgmax(std::uint32_t vocab) {
  constexpr std::uint32_t rows = 10;
  constexpr std::int32_t guard = -1234567;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> logits(static_cast<std::size_t>(rows) * vocab);
  for (std::uint32_t t = 0; t < rows; ++t) {
    float* row = logits.data() + static_cast<std::size_t>(t) * vocab;
    for (std::uint32_t i = 0; i < vocab; ++i) {
      row[i] = -static_cast<float>((i * 7919U + t * 31U) % 65521U);
    }
    switch (t) {
      case 0:
        row[vocab - 1] = 10.0F;
        break;
      case 1:
        row[0] = row[vocab - 1] = 10.0F;
        break;
      case 2:
        row[vocab / 3] = row[vocab - 1] = inf;
        break;
      case 3:
        std::fill_n(row, vocab, -inf);
        break;
      case 4:
        row[0] = nan;
        break;
      case 5:
        row[vocab - 1] = nan;
        row[vocab / 2] = 10.0F;
        break;
      case 6:
        std::fill_n(row, vocab, nan);
        break;
      case 7:
        for (std::uint32_t i = 0; i < vocab; ++i)
          row[i] = i % 2 ? 0.0F : -0.0F;
        break;
      case 8:
        row[vocab / 2] = row[vocab - 1] = 10.0F;
        break;
      default:
        break;
    }
  }
  auto device_logits = Allocate<float>(logits.size());
  auto scratch = Allocate<q::ArgmaxCandidate>(rows * q::kArgmaxParts);
  auto output = Allocate<std::int32_t>(rows + 2);
  const auto greedy_scratch_size = 2 * rows * ((vocab - 1) / 4096 + 1);
  auto greedy_scratch = Allocate<float>(greedy_scratch_size);
  auto greedy_ids = Allocate<std::uint32_t>(rows);
  auto greedy_output = Allocate<q::ArgmaxCandidate>(rows + 2);
  std::vector<q::ArgmaxCandidate> predictions(rows + 2, {-1234567.0F, guard});
  std::vector<std::int32_t> actual(rows + 2, guard);
  CheckHip(hipMemcpy(output.get(), actual.data(),
                     actual.size() * sizeof(actual[0]), hipMemcpyHostToDevice),
           "initialize output");
  CheckHip(hipMemcpy(greedy_output.get(), predictions.data(),
                     predictions.size() * sizeof(predictions[0]),
                     hipMemcpyHostToDevice),
           "initialize greedy output");
  // Reuse the captured selection with changed logits, as the draft executor
  // does.
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
  CheckHip(hipStreamCreate(&stream), "create stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "capture");
  q::Argmax(device_logits.get(), scratch.get(), output.get() + 1, rows, vocab,
            stream);
  gufo::hip::LaunchBatchedGPUArgmax(
      device_logits.get(), greedy_ids.get(), rows, vocab,
      {greedy_scratch.get(), greedy_scratch_size}, stream);
  q::GatherArgmaxCandidates(device_logits.get(), greedy_ids.get(),
                            greedy_output.get() + 1, rows, vocab, stream);
  CheckHip(hipStreamEndCapture(stream, &graph), "finish capture");
  CheckHip(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
           "instantiate");
  for (int replay = 0; replay < 2; ++replay) {
    CheckHip(hipMemcpyAsync(device_logits.get(), logits.data(),
                            logits.size() * sizeof(float),
                            hipMemcpyHostToDevice, stream),
             "upload logits");
    CheckHip(hipGraphLaunch(executable, stream), "select");
    CheckHip(hipMemcpyAsync(actual.data(), output.get(),
                            actual.size() * sizeof(actual[0]),
                            hipMemcpyDeviceToHost, stream),
             "download tokens");
    CheckHip(hipMemcpyAsync(predictions.data(), greedy_output.get(),
                            predictions.size() * sizeof(predictions[0]),
                            hipMemcpyDeviceToHost, stream),
             "download greedy predictions");
    CheckHip(hipStreamSynchronize(stream), "synchronize");
    if (actual.front() != guard || actual.back() != guard) {
      throw std::runtime_error("argmax overwrote output guard");
    }
    if (predictions.front().index != guard ||
        predictions.back().index != guard ||
        predictions.front().value != -1234567.0F ||
        predictions.back().value != -1234567.0F)
      throw std::runtime_error("greedy argmax overwrote output guard");
    for (std::uint32_t t = 0; t < rows; ++t) {
      const float* row = logits.data() + static_cast<std::size_t>(t) * vocab;
      const auto expected =
          static_cast<std::int32_t>(std::max_element(row, row + vocab) - row);
      if (actual[t + 1] != expected) {
        throw std::runtime_error(
            "argmax mismatch: vocab=" + std::to_string(vocab) + " row=" +
            std::to_string(t) + " expected=" + std::to_string(expected) +
            " actual=" + std::to_string(actual[t + 1]));
      }
      const auto& prediction = predictions[t + 1];
      gufo::sampling::SamplerState sampler;
      std::uint32_t greedy = 0;
      bool finite = true;
      try {
        greedy = sampler.Sample({row, vocab});
      } catch (const std::runtime_error&) {
        finite = false;
      }
      if (!finite) {
        if (std::isfinite(prediction.value))
          throw std::runtime_error("greedy argmax accepted nonfinite row");
      } else if (prediction.index != static_cast<std::int32_t>(greedy) ||
                 std::bit_cast<std::uint32_t>(prediction.value) !=
                     std::bit_cast<std::uint32_t>(row[greedy])) {
        throw std::runtime_error(
            "greedy argmax disagrees with sampler: vocab=" +
            std::to_string(vocab) + " row=" + std::to_string(t));
      }
    }
    std::reverse(logits.begin(), logits.end());
  }
  CheckHip(hipGraphExecDestroy(executable), "destroy executable");
  CheckHip(hipGraphDestroy(graph), "destroy graph");
  CheckHip(hipStreamDestroy(stream), "destroy stream");
  std::cout << "MTP argmax vocab=" << vocab
            << ": CPU oracle and graph replay passed\n";
}

void CheckCandidates(std::uint32_t vocab) {
  constexpr auto kCandidates = gufo::models::qwen38_flash_next::kMtpCandidates;
  const auto workspace_size = q::MtpCandidateWorkspaceSize(vocab);
  auto device_ids = Allocate<std::uint32_t>(workspace_size + 2);
  auto scratch_ids = Allocate<std::uint32_t>(workspace_size + 2);
  // Match the executor's packed result: selection consumes the input IDs
  // before the final scores overwrite unused selection workspace.
  auto* device_scores =
      reinterpret_cast<float*>(device_ids.get() + 1 + kCandidates);
  auto device_logits = Allocate<float>(vocab);
  std::vector<float> logits(vocab);
  std::vector<std::uint32_t> expected(vocab);
  const auto count = std::min<std::size_t>(vocab, kCandidates);
  const auto result_count = std::min<std::size_t>(vocab, kCandidates);
  std::vector<float> scores(result_count);
  std::vector<std::uint32_t> ids(workspace_size + 2, UINT32_MAX);
  CheckHip(hipMemcpy(device_ids.get(), ids.data(), ids.size() * 4,
                     hipMemcpyHostToDevice),
           "candidate ID guards");
  CheckHip(hipMemcpy(scratch_ids.get(), ids.data(), ids.size() * 4,
                     hipMemcpyHostToDevice),
           "candidate scratch guards");
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
  CheckHip(hipStreamCreate(&stream), "candidate stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "candidate capture");
  q::MtpTopCandidates(device_logits.get(), device_ids.get() + 1,
                      scratch_ids.get() + 1, device_scores, vocab, stream);
  CheckHip(hipStreamEndCapture(stream, &graph), "candidate capture end");
  CheckHip(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
           "candidate graph");

  const auto score = [&](std::uint32_t id) {
    return std::isfinite(logits[id]) ? logits[id]
                                     : -std::numeric_limits<float>::infinity();
  };
  const auto better = [&](auto a, auto b) {
    return score(a) == score(b) ? a < b : score(a) > score(b);
  };
  for (unsigned mode = 0; mode < 4; ++mode) {
    for (std::uint32_t i = 0; i < vocab; ++i) {
      logits[i] = mode == 0   ? static_cast<float>((i * 7919U) % 257)
                  : mode == 1 ? (i % 2 ? 0.0F : -0.0F)
                  : mode == 2 ? std::numeric_limits<float>::quiet_NaN()
                              : (i >= vocab - count ? 10.0F : -100.0F);
    }
    if (mode == 0) {
      logits[0] = std::numeric_limits<float>::infinity();
      logits[vocab - 1] = std::numeric_limits<float>::quiet_NaN();
    }
    std::iota(expected.begin(), expected.end(), 0);
    std::sort(expected.begin(), expected.end(), better);
    CheckHip(hipMemcpyAsync(device_logits.get(), logits.data(), vocab * 4,
                            hipMemcpyHostToDevice, stream),
             "candidate upload");
    CheckHip(hipGraphLaunch(executable, stream), "candidate replay");
    CheckHip(hipMemcpyAsync(ids.data(), device_ids.get(), ids.size() * 4,
                            hipMemcpyDeviceToHost, stream),
             "candidate IDs");
    CheckHip(hipStreamSynchronize(stream), "candidate synchronization");
    if (ids.front() != UINT32_MAX || ids.back() != UINT32_MAX)
      throw std::runtime_error("MTP candidate ID guard changed");
    for (std::size_t i = 0; i < count; ++i)
      if (ids[i + 1] != expected[i])
        throw std::runtime_error("MTP candidate IDs disagree with CPU");

    CheckHip(hipMemcpyAsync(scores.data(), device_scores, scores.size() * 4,
                            hipMemcpyDeviceToHost, stream),
             "candidate scores");
    CheckHip(hipStreamSynchronize(stream), "rescoring synchronization");
    if (ids.front() != UINT32_MAX || ids.back() != UINT32_MAX)
      throw std::runtime_error("MTP candidate output guard changed");
    for (std::size_t i = 0; i < result_count; ++i) {
      if (ids[i + 1] != expected[i] ||
          std::bit_cast<std::uint32_t>(scores[i]) !=
              std::bit_cast<std::uint32_t>(score(expected[i])))
        throw std::runtime_error(
            "MTP candidate scores disagree with CPU: mode=" +
            std::to_string(mode) + " rank=" + std::to_string(i) + " ID=" +
            std::to_string(ids[i + 1]) + "/" + std::to_string(expected[i]) +
            " score=" + std::to_string(scores[i]) + "/" +
            std::to_string(score(expected[i])));
    }
    CheckHip(hipMemcpy(ids.data(), scratch_ids.get(), ids.size() * 4,
                       hipMemcpyDeviceToHost),
             "candidate scratch download");
    if (ids.front() != UINT32_MAX || ids.back() != UINT32_MAX)
      throw std::runtime_error("MTP candidate scratch guard changed");
  }

  CheckHip(hipGraphExecDestroy(executable), "candidate graph destroy");
  CheckHip(hipGraphDestroy(graph), "candidate source graph destroy");
  CheckHip(hipStreamDestroy(stream), "candidate stream destroy");
  std::cout << "MTP candidates vocab=" << vocab
            << ": exact ordering, ties, guards and graph replay passed\n";
}

}  // namespace

int main() {
  try {
    CheckPenaltyArgmax(257);
    CheckPenaltyArgmax(248320);
    CheckArgmax(1);
    CheckArgmax(257);
    CheckArgmax(248320);
    for (const unsigned vocab :
         {1, 63, 64, 65, 255, 256, 257, 1024, 1025, 16385, 248320}) {
      CheckCandidates(vocab);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
