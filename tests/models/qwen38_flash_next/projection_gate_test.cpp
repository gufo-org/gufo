// GPU gates for the optional runtime residual projection: a loaded but
// disabled vector matches an executor without one, per-request toggle
// sequencing under captured decode graphs, and per-session isolation in
// multi-session batch decode. All compare full-vocabulary logit rows
// bitwise; a mismatch fails the run.
//
// projection_gate_test --model FIRST_SHARD.gguf
//                       --projection-vector CV.gguf [--projection-layers LO-HI]
//                       [--context N] [--steps N]
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/projection.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace rocm = q::rocm;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
  } else {
    std::printf("ok: %s\n", what.c_str());
  }
}

bool SameRows(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Decodes `steps` continuation tokens with the per-step projection pattern,
// returning one vocabulary row per step.
std::vector<float> DecodePattern(rocm::Executor& executor,
                                 rocm::Session& session,
                                 const std::vector<std::int32_t>& tokens,
                                 std::span<const std::int32_t> feed,
                                 const std::vector<bool>& pattern,
                                 std::uint32_t vocab, std::string* error) {
  std::vector<float> rows(static_cast<std::size_t>(feed.size()) * vocab);
  for (std::size_t i = 0; i < feed.size(); ++i) {
    session.SetProjectionEnabled(pattern[i % pattern.size()]);
    float* out = rows.data() + static_cast<std::size_t>(i) * vocab;
    if (!executor.Forward(session, std::span<const std::int32_t>(&feed[i], 1),
                          1, out, rocm::Executor::ForwardMode::kDecode,
                          error)) {
      return {};
    }
  }
  (void)tokens;
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string vector_path;
  std::string layers;
  std::uint32_t context = 4096;
  std::size_t steps = 8;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--projection-vector") {
      vector_path = next();
    } else if (arg == "--projection-layers") {
      layers = next();
    } else if (arg == "--context") {
      context = static_cast<std::uint32_t>(std::stoul(next()));
    } else if (arg == "--steps") {
      steps = std::stoul(next());
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty() || vector_path.empty()) {
    std::fprintf(stderr,
                 "usage: projection_gate_test --model M.gguf "
                 "--projection-vector CV.gguf [--projection-layers LO-HI] "
                 "[--context N] [--steps N]\n");
    return 2;
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
  q::ProjectionVector projection;
  std::string warning;
  if (!q::LoadControlVector(vector_path, c.num_layers, c.hidden_size,
                            "qwen4exp", &projection, &error, &warning)) {
    std::fprintf(stderr, "projection vector failed: %s\n", error.c_str());
    return 1;
  }
  if (!warning.empty()) {
    std::fprintf(stderr, "warning: %s\n", warning.c_str());
  }
  if (!q::SelectProjectionLayers(layers, &projection, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 2;
  }
  std::printf("projection layers %u..%u\n", projection.first_layer,
              projection.last_layer);

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
  auto device =
      q::rocm::DeviceModel::Upload(*weights, *reader, nullptr, nullptr, &error);
  if (!device) {
    std::fprintf(stderr, "upload failed: %s\n", error.c_str());
    return 1;
  }
  const auto create_executor = [&](const q::ProjectionVector* vector) {
    rocm::Executor::Options options;
    options.max_batch = 64;
    options.max_context = context;
    options.max_logit_rows = 8;
    options.projection = vector;
    return rocm::Executor::Create(*device, ngram.get(), options, &error);
  };

  const std::string prompt_text =
      "The capital of France is Paris. The largest planet in the solar "
      "system is Jupiter. Water boils at 100 degrees Celsius at sea level.";
  std::vector<std::int32_t> tokens;
  for (auto id : tokenizer->Encode(prompt_text)) {
    tokens.push_back(static_cast<std::int32_t>(id));
  }
  const std::string cont_text =
      " Photosynthesis converts sunlight into chemical energy stored as "
      "glucose inside plant cells.";
  std::vector<std::int32_t> feed;
  for (auto id : tokenizer->Encode(cont_text)) {
    feed.push_back(static_cast<std::int32_t>(id));
  }
  if (feed.size() > steps) {
    feed.resize(steps);
  }
  // Gates 1b/1c compare the first two decode steps.
  if (feed.size() < 2) {
    std::fprintf(stderr, "--steps must be at least 2\n");
    return 2;
  }
  const auto prefill_with = [&](rocm::Executor& e, rocm::Session& s,
                                bool enabled, std::vector<float>& last_row) {
    s.SetProjectionEnabled(enabled);
    last_row.assign(c.vocab_size, 0.0F);
    return e.Forward(s, tokens, 1, last_row.data(),
                     rocm::Executor::ForwardMode::kPrefill, &error);
  };
  const std::vector<bool> off(feed.size(), false);

  // Gate 0: with the vector loaded but disabled, prefill and captured decode
  // are bitwise identical to an executor that never loaded one. The
  // unloaded executor is released before the gates below allocate theirs.
  std::vector<float> unloaded_last;
  std::vector<float> unloaded_rows;
  {
    auto unloaded = create_executor(nullptr);
    if (!unloaded) {
      std::fprintf(stderr, "executor failed: %s\n", error.c_str());
      return 1;
    }
    auto s = unloaded->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     context, &error);
    if (!s || !prefill_with(*unloaded, *s, false, unloaded_last)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    unloaded_rows =
        DecodePattern(*unloaded, *s, tokens, feed, off, c.vocab_size, &error);
  }
  auto executor = create_executor(&projection);
  if (!executor) {
    std::fprintf(stderr, "executor failed: %s\n", error.c_str());
    return 1;
  }
  const auto prefill = [&](rocm::Session& s, bool enabled,
                           std::vector<float>& last_row) {
    return prefill_with(*executor, s, enabled, last_row);
  };
  {
    std::vector<float> last;
    auto s = executor->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     context, &error);
    if (!s || !prefill(*s, false, last)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const auto rows =
        DecodePattern(*executor, *s, tokens, feed, off, c.vocab_size, &error);
    Check(SameRows(last, unloaded_last),
          "gate 0a: disabled prefill matches an unloaded executor");
    Check(!rows.empty() && SameRows(rows, unloaded_rows),
          "gate 0b: disabled decode matches an unloaded executor");
  }

  // Gate 1: an alternating toggle pattern replays bitwise-identically on a
  // fresh session (deterministic graph replay across projection states) and
  // stays bitwise-identical to an all-off session up to the first enabled
  // step, then diverges.
  {
    const std::vector<bool> pattern = {false, true,  true,  false,
                                       true,  false, false, true};
    std::vector<float> last;
    auto s1 = executor->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      context, &error);
    if (!prefill(*s1, false, last)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const auto rows1 = DecodePattern(*executor, *s1, tokens, feed, pattern,
                                     c.vocab_size, &error);
    auto s2 = executor->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      context, &error);
    if (!prefill(*s2, false, last)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const auto rows2 = DecodePattern(*executor, *s2, tokens, feed, pattern,
                                     c.vocab_size, &error);
    Check(!rows1.empty() && SameRows(rows1, rows2),
          "gate 1a: alternating pattern replays bitwise-identically");

    auto s3 = executor->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      context, &error);
    if (!prefill(*s3, false, last)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const auto rows3 =
        DecodePattern(*executor, *s3, tokens, feed, off, c.vocab_size, &error);
    std::vector<float> prefix1(rows1.begin(),
                               rows1.begin() + static_cast<long>(c.vocab_size));
    std::vector<float> prefix3(rows3.begin(),
                               rows3.begin() + static_cast<long>(c.vocab_size));
    Check(SameRows(prefix1, prefix3),
          "gate 1b: step before the first enabled step matches all-off");
    std::vector<float> on1(rows1.begin() + static_cast<long>(c.vocab_size),
                           rows1.begin() + 2 * static_cast<long>(c.vocab_size));
    std::vector<float> on3(rows3.begin() + static_cast<long>(c.vocab_size),
                           rows3.begin() + 2 * static_cast<long>(c.vocab_size));
    Check(!SameRows(on1, on3), "gate 1c: the first enabled step diverges");
  }

  // Gate 2: multi-session batch decode isolates the per-session projection.
  // For every combination, the batched rows must equal the rows of a session
  // decoding the same tokens alone with the same flag sequence, and two
  // sessions with different flags must disagree from the first step.
  const bool combos[][2] = {
      {false, false}, {true, true}, {false, true}, {true, false}};
  for (const auto& combo : combos) {
    std::vector<float> last;
    auto a = executor->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     context, &error);
    auto b = executor->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     context, &error);
    if (!prefill(*a, combo[0], last) || !prefill(*b, combo[1], last)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    // Isolated references replaying the same flags.
    auto c1 = executor->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      context, &error);
    auto c2 = executor->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      context, &error);
    if (!prefill(*c1, combo[0], last) || !prefill(*c2, combo[1], last)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const std::size_t flip_at = feed.size() / 2;
    std::vector<bool> pa(feed.size(), combo[0]);
    std::vector<bool> pb(feed.size(), combo[1]);
    pa[flip_at] = !pa[flip_at];
    pb[flip_at] = !pb[flip_at];
    const auto ref_a =
        DecodePattern(*executor, *c1, tokens, feed, pa, c.vocab_size, &error);
    const auto ref_b =
        DecodePattern(*executor, *c2, tokens, feed, pb, c.vocab_size, &error);

    // Batched decode; the middle round flips both flags so a later batch
    // sees flipped states (sequential batches over shared captured graphs).
    std::vector<float> rows(static_cast<std::size_t>(feed.size()) * 2 *
                            c.vocab_size);
    for (std::size_t i = 0; i < feed.size(); ++i) {
      a->SetProjectionEnabled(pa[i]);
      b->SetProjectionEnabled(pb[i]);
      const rocm::Executor::BatchItem items[2] = {
          {a.get(), std::span<const std::int32_t>(&feed[i], 1), false},
          {b.get(), std::span<const std::int32_t>(&feed[i], 1), false}};
      if (!executor->ForwardBatch(items, &error)) {
        std::fprintf(stderr, "batch failed: %s\n", error.c_str());
        return 1;
      }
      if (!executor->SelectBatchLogits(
              0, 2,
              rows.data() + static_cast<std::size_t>(i) * 2 * c.vocab_size,
              &error)) {
        std::fprintf(stderr, "select failed: %s\n", error.c_str());
        return 1;
      }
    }
    const auto label =
        std::string(combo[0] ? "on" : "off") + "/" + (combo[1] ? "on" : "off");
    // Every round's rows must equal an isolated session that replayed the
    // exact same flag sequence, including the flipped round.
    bool a_ok = true;
    bool b_ok = true;
    for (std::size_t i = 0; i < feed.size(); ++i) {
      const float* row_a = rows.data() + (i * 2) * c.vocab_size;
      const float* row_b = rows.data() + (i * 2 + 1) * c.vocab_size;
      a_ok = a_ok &&
             SameRows(std::vector<float>(row_a, row_a + c.vocab_size),
                      std::vector<float>(
                          ref_a.begin() + static_cast<long>(i * c.vocab_size),
                          ref_a.begin() +
                              static_cast<long>((i + 1) * c.vocab_size)));
      b_ok = b_ok &&
             SameRows(std::vector<float>(row_b, row_b + c.vocab_size),
                      std::vector<float>(
                          ref_b.begin() + static_cast<long>(i * c.vocab_size),
                          ref_b.begin() +
                              static_cast<long>((i + 1) * c.vocab_size)));
    }
    Check(a_ok,
          "gate 2 [" + label + "]: session A batched rows match isolated");
    Check(b_ok,
          "gate 2 [" + label + "]: session B batched rows match isolated");
    if (combo[0] != combo[1]) {
      const float* row_a = rows.data();
      const float* row_b = rows.data() + c.vocab_size;
      Check(!SameRows(std::vector<float>(row_a, row_a + c.vocab_size),
                      std::vector<float>(row_b, row_b + c.vocab_size)),
            "gate 2 [" + label + "]: differing flags diverge");
    } else {
      const float* row_a = rows.data();
      const float* row_b = rows.data() + c.vocab_size;
      Check(SameRows(std::vector<float>(row_a, row_a + c.vocab_size),
                     std::vector<float>(row_b, row_b + c.vocab_size)),
            "gate 2 [" + label + "]: equal flags agree");
    }
  }

  if (failures != 0) {
    std::printf("%d projection gate check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("all projection gates passed\n");
  return 0;
}
