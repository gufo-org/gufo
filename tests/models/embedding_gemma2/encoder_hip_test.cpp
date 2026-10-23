// Compares the GPU encoder with the float32 CPU reference on fixed token
// sequences, checks that packing sequences into one batch does not change
// any embedding beyond rounding, and checks that concurrent calls share
// batches while a long call gives up the device after each of its batches, and
// that a call with images runs in a batch of its own.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
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
// changes. The cancellation hook runs before each batch that takes the call's
// sequences, so it records the order of batches.
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

// Checks that running beside other calls left an embedding unchanged.
void CheckUnchanged(std::string_view what, const std::vector<float>& together,
                    const std::vector<float>& alone) {
  const double cosine = Cosine(together, alone);
  std::cout << what << " cosine vs alone " << cosine << '\n';
  Check(cosine > 0.9999,
        "running beside another call must not change an embedding");
}

// Embeds a long blocker call on its own thread and runs `start` on this one
// as soon as the blocker's batch is about to run. The hook only raises a flag,
// because it runs with the scheduler's lock held. The calls that `start` begins
// queue once the lock is free, while the 8,000-token batch holds the device.
void EmbedBehindBlocker(embedding_gemma2::hip::EncoderRuntime& runtime,
                        unsigned seed, const std::function<void()>& start) {
  const std::vector<std::vector<std::uint32_t>> blocker = {
      RandomTokens(8000, seed)};
  std::atomic<bool> batch_starting{false};
  std::vector<std::vector<float>> result;
  std::string error;
  bool ok = false;
  std::jthread blocker_thread([&] {
    ok = runtime.Embed(blocker, &result, &error, [&] {
      batch_starting = true;
      return false;
    });
  });
  while (!batch_starting) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  start();
  blocker_thread.join();
  Check(ok, error);
}

// Two calls that queue behind a busy device share its next batch, and
// sharing does not change their embeddings. The batch's cancellation hooks
// run on the thread that runs it, so hooks of both calls on one thread mean
// one shared batch.
void CheckSharedBatch(embedding_gemma2::hip::EncoderRuntime& runtime) {
  const std::vector<std::vector<std::uint32_t>> first_call = {
      RandomTokens(100, 22), RandomTokens(40, 23)};
  const std::vector<std::vector<std::uint32_t>> second_call = {
      RandomTokens(64, 24)};
  std::string error;
  std::vector<std::vector<float>> first_alone;
  std::vector<std::vector<float>> second_alone;
  Check(runtime.Embed(first_call, &first_alone, &error), error);
  Check(runtime.Embed(second_call, &second_alone, &error), error);

  std::mutex mutex;
  std::vector<std::thread::id> first_hooks;
  std::vector<std::thread::id> second_hooks;
  std::vector<std::vector<float>> first_result;
  std::vector<std::vector<float>> second_result;
  bool first_ok = false;
  bool second_ok = false;
  std::jthread first_thread;
  std::jthread second_thread;
  const auto record = [&](std::vector<std::thread::id>* hooks) {
    const std::lock_guard lock(mutex);
    hooks->push_back(std::this_thread::get_id());
    return false;
  };
  EmbedBehindBlocker(runtime, 21, [&] {
    first_thread = std::jthread([&] {
      std::string first_error;
      first_ok = runtime.Embed(first_call, &first_result, &first_error,
                               [&] { return record(&first_hooks); });
    });
    second_thread = std::jthread([&] {
      std::string second_error;
      second_ok = runtime.Embed(second_call, &second_result, &second_error,
                                [&] { return record(&second_hooks); });
    });
  });
  first_thread.join();
  second_thread.join();
  Check(first_ok && second_ok, "the queued calls succeed");
  Check(first_hooks.size() == 1 && second_hooks.size() == 1 &&
            first_hooks[0] == second_hooks[0],
        "two queued calls share one batch");
  for (std::size_t i = 0; i < first_alone.size(); ++i) {
    CheckUnchanged("shared batch, first call sequence " + std::to_string(i),
                   first_result[i], first_alone[i]);
  }
  CheckUnchanged("shared batch, second call", second_result[0],
                 second_alone[0]);
}

// One sequence with a 6 x 6 patch image of random pixels, four soft tokens.
embedding_gemma2::EncoderSequence ImageSequence(unsigned seed) {
  std::mt19937 random(seed);
  std::uniform_real_distribution<float> pixel(0.0F, 1.0F);
  embedding_gemma2::ImagePatches patches{.rows = 6, .columns = 6, .pixels = {}};
  patches.pixels.resize(std::size_t{patches.rows} * patches.columns * 768);
  for (float& value : patches.pixels) {
    value = pixel(random);
  }
  std::vector<std::uint32_t> tokens = {2, 255999};
  const std::size_t offset = tokens.size();
  tokens.insert(tokens.end(), patches.soft_tokens(),
                embedding_gemma2::kImageTokenId);
  tokens.push_back(258882);
  const std::vector<std::uint32_t> text = RandomTokens(24, seed);
  tokens.insert(tokens.end(), text.begin() + 1, text.end());
  return {.tokens = std::move(tokens),
          .images = {{.offset = offset, .patches = std::move(patches)}}};
}

// An image call and a text call that queue together behind a busy device run
// in separate batches, in arrival order, and both get their alone results. The
// first image call loads the vision tower in its batch. The cancellation hooks
// record the order of the batches and the thread that runs each. In the cold
// run the text call starts from the image call's first hook, so it queues while
// the image call's batch runs. A warm image call forms its batch in that hook,
// so the warm text call starts 50 ms after the image call instead.
void CheckImageAndTextCalls(embedding_gemma2::hip::EncoderRuntime& runtime) {
  const std::vector<embedding_gemma2::EncoderSequence> image_call = {
      ImageSequence(32)};
  const std::vector<std::vector<std::uint32_t>> text_call = {
      RandomTokens(64, 33)};
  std::string error;

  struct Hooks {
    std::string order;
    std::vector<std::thread::id> image_threads;
    std::vector<std::thread::id> text_threads;
  };
  struct Results {
    std::vector<std::vector<float>> image;
    std::vector<std::vector<float>> text;
    std::string image_error;
    std::string text_error;
    bool image_ok = false;
    bool text_ok = false;
  };
  const auto run_together = [&](Hooks* hooks, Results* results,
                                bool text_from_image_hook) {
    std::mutex mutex;
    std::jthread image_thread;
    std::jthread text_thread;
    std::function<void()> start_text;
    bool text_started = false;
    const auto record = [&](char call, std::vector<std::thread::id>* threads) {
      const std::lock_guard lock(mutex);
      hooks->order += call;
      threads->push_back(std::this_thread::get_id());
      if (call == 'I' && text_from_image_hook &&
          !std::exchange(text_started, true)) {
        start_text();
      }
      return false;
    };
    start_text = [&] {
      text_thread = std::jthread([&] {
        results->text_ok =
            runtime.Embed(text_call, &results->text, &results->text_error,
                          [&] { return record('T', &hooks->text_threads); });
      });
    };
    EmbedBehindBlocker(runtime, 31, [&] {
      image_thread = std::jthread([&] {
        results->image_ok = runtime.EmbedMultimodal(
            image_call, &results->image, &results->image_error,
            [&] { return record('I', &hooks->image_threads); });
      });
      if (!text_from_image_hook) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        start_text();
      }
    });
    image_thread.join();
    text_thread.join();
    Check(results->image_ok, results->image_error);
    Check(results->text_ok, results->text_error);
  };

  const auto check_separate_batches = [](const Hooks& hooks, const char* what) {
    Check(hooks.order == "IT" && hooks.image_threads.size() == 1 &&
              hooks.text_threads.size() == 1 &&
              hooks.image_threads != hooks.text_threads,
          what);
  };

  // The first image call loads the vision tower in its own batch, and the text
  // call arrives while that batch runs.
  Hooks cold_hooks;
  Results cold;
  run_together(&cold_hooks, &cold, true);
  std::cout << "image and text batch order " << cold_hooks.order << '\n';
  check_separate_batches(cold_hooks,
                         "the image call that loads the vision tower and the "
                         "text call run in separate batches, in arrival order");

  std::vector<std::vector<float>> image_alone;
  std::vector<std::vector<float>> text_alone;
  Check(runtime.EmbedMultimodal(image_call, &image_alone, &error), error);
  Check(runtime.Embed(text_call, &text_alone, &error), error);

  Hooks warm_hooks;
  Results warm;
  run_together(&warm_hooks, &warm, false);
  check_separate_batches(
      warm_hooks,
      "an image call and a text call run in separate batches, in arrival "
      "order");

  CheckUnchanged("image call that loads the vision tower", cold.image[0],
                 image_alone[0]);
  CheckUnchanged("text call behind the vision load", cold.text[0],
                 text_alone[0]);
  CheckUnchanged("image call in its own batch", warm.image[0], image_alone[0]);
  CheckUnchanged("text call behind an image call", warm.text[0], text_alone[0]);
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
  CheckSharedBatch(*runtime);
  if (runtime->config().supported_vision) {
    CheckImageAndTextCalls(*runtime);
  } else {
    std::cout << "SKIP image and text calls: the checkpoint has no supported "
                 "vision tower\n";
  }
  Check(!runtime->device_lost(), "no device failure was reported");

  Check(!runtime->Embed({{}}, &batched, &error), "empty input is rejected");
  Check(!runtime->Embed({{2, 262144}}, &batched, &error),
        "out-of-range token is rejected");
  std::cout << "PASS embedding_gemma2_encoder_hip_test\n";
  return 0;
}
