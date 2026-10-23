// Compares the GPU encoder with the float32 CPU reference on fixed token
// sequences, checks that packing sequences into one batch does not change
// any embedding beyond rounding, and checks that concurrent calls take device
// turns one batch at a time.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/models/embedding_gemma2/hip/encoder_runtime.hpp"
#include "src/models/embedding_gemma2/loader.hpp"
#include "src/models/embedding_gemma2/reference/cpu_encoder.hpp"

namespace embedding_gemma2 = gufo::models::embedding_gemma2;

namespace {

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL embedding_gemma2_encoder_hip_test: " << message << '\n';
    std::exit(1);
  }
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * b[i];
    na += double(a[i]) * a[i];
    nb += double(b[i]) * b[i];
  }
  return dot / std::sqrt(na * nb);
}

std::vector<std::uint32_t> RandomTokens(std::size_t length, unsigned seed) {
  std::mt19937 random(seed);
  std::vector<std::uint32_t> tokens{2};
  while (tokens.size() + 1 < length) {
    tokens.push_back(1000 + random() % 100000);
  }
  tokens.push_back(1);
  return tokens;
}

// A long call holds the device for one batch at a time. A short call that
// arrives during the first batch runs before the second, and neither result
// changes. The cancellation hook runs at the start of each turn, so it records
// the order of turns.
void CheckTurns(embedding_gemma2::hip::EncoderRuntime& runtime) {
  const std::vector<std::vector<std::uint32_t>> long_call = {
      RandomTokens(8000, 11), RandomTokens(8000, 12), RandomTokens(8000, 13)};
  const std::vector<std::vector<std::uint32_t>> short_call = {
      RandomTokens(16, 14)};
  std::string error;
  std::vector<std::vector<float>> long_alone;
  std::vector<std::vector<float>> short_alone;
  Check(runtime.Embed(long_call, &long_alone, &error), error);
  Check(runtime.Embed(short_call, &short_alone, &error), error);

  std::mutex mutex;
  std::string order;
  std::jthread short_thread;
  std::vector<std::vector<float>> short_result;
  bool short_ok = false;
  const auto on_long_turn = [&] {
    {
      const std::lock_guard lock(mutex);
      order += 'L';
      if (order.size() > 1) {
        return false;
      }
    }
    short_thread = std::jthread([&] {
      std::string short_error;
      short_ok = runtime.Embed(short_call, &short_result, &short_error, [&] {
        const std::lock_guard lock(mutex);
        order += 'S';
        return false;
      });
    });
    // The short call queues for a turn while this batch still holds one.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return false;
  };
  std::vector<std::vector<float>> long_result;
  Check(runtime.Embed(long_call, &long_result, &error, on_long_turn), error);
  short_thread.join();
  Check(short_ok, "the concurrent short call succeeds");
  std::cout << "turn order " << order << '\n';
  Check(order == "LSLL", "a short call runs between the batches of a long one");
  for (std::size_t i = 0; i < long_call.size(); ++i) {
    Check(Cosine(long_result[i], long_alone[i]) > 0.9999,
          "interleaving must not change a long-call embedding");
  }
  Check(Cosine(short_result[0], short_alone[0]) > 0.9999,
        "interleaving must not change a short-call embedding");
}

}  // namespace

int main(int argc, char** argv) {
  const char* configured = std::getenv("EMBEDDING_GEMMA2_MODEL_ROOT");
  const std::filesystem::path model_root =
      argc > 1
          ? argv[1]
          : (configured != nullptr ? configured : "models/embeddinggemma-2");
  if (!std::filesystem::exists(model_root / "model.safetensors")) {
    std::cerr << "SKIP embedding_gemma2_encoder_hip_test: external model "
                 "unavailable\n";
    return 77;
  }

  std::string error;
  auto runtime = embedding_gemma2::hip::EncoderRuntime::Create(
      model_root.string(), &error);
  if (runtime == nullptr) {
    // A missing device is a skip. Any other failure is real.
    if (error.find("no ROCm-capable device") != std::string::npos ||
        error.find("invalid device ordinal") != std::string::npos) {
      std::cerr << "SKIP embedding_gemma2_encoder_hip_test: " << error << '\n';
      return 77;
    }
    Check(false, error);
  }

  const std::vector<std::vector<std::uint32_t>> sequences = {
      RandomTokens(9, 1), RandomTokens(24, 2), RandomTokens(3, 3),
      RandomTokens(48, 4)};
  std::vector<std::vector<float>> batched;
  Check(runtime->Embed(sequences, &batched, &error), error);
  Check(batched.size() == sequences.size(), "one embedding per sequence");

  const embedding_gemma2::LoadResult loaded =
      embedding_gemma2::LoadModelDirectory(model_root.string());
  Check(loaded.ok, loaded.error);
  const embedding_gemma2::reference::CpuEncoder cpu(loaded.config.text,
                                                    *loaded.store);
  for (std::size_t i = 0; i < sequences.size(); ++i) {
    Check(batched[i].size() == 768, "embedding width");
    double norm = 0.0;
    for (const float value : batched[i]) {
      norm += double(value) * value;
    }
    Check(std::abs(norm - 1.0) < 1e-3, "embeddings are unit length");
    const std::vector<float> expected = cpu.Embed(sequences[i]);
    const double cosine = Cosine(batched[i], expected);
    std::cout << std::setprecision(8);
    std::cout << "sequence " << i << " cosine vs CPU reference " << cosine
              << '\n';
    Check(cosine > 0.9995, "GPU embedding disagrees with the CPU reference");

    std::vector<std::vector<float>> single;
    Check(runtime->Embed({sequences[i]}, &single, &error), error);
    const double alone = Cosine(single[0], batched[i]);
    std::cout << "sequence " << i << " cosine alone vs batched " << alone
              << '\n';
    // The tuned GEMM may pick a different algorithm for a different batch
    // size, which changes float32 accumulation order but not the result by
    // more than rounding. A leak between sequences would be far larger.
    Check(alone > 0.9999, "batching must not change an embedding");
  }

  CheckTurns(*runtime);
  Check(!runtime->device_lost(), "no device failure was reported");

  Check(!runtime->Embed({{}}, &batched, &error), "empty input is rejected");
  Check(!runtime->Embed({{2, 262144}}, &batched, &error),
        "out-of-range token is rejected");
  std::cout << "PASS embedding_gemma2_encoder_hip_test\n";
  return 0;
}
