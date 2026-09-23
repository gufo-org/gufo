#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace {
using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Case {
  std::vector<Token> tokens;
  std::vector<std::vector<float>> logits;
  std::size_t prompt_size;
};

// Teacher forcing keeps quantization comparisons on identical input prefixes.
// Free-running text would conflate different inputs with numerical differences.
constexpr std::array<const char*, 3> kTexts{
    "Virtual memory lets an operating system give each process its own address "
    "space. Pages map virtual addresses to physical memory. A page fault "
    "occurs "
    "when the requested page is unavailable and the operating system must act.",
    "A train leaves the station at noon and travels at sixty kilometers per "
    "hour. "
    "Another train leaves one hour later at ninety kilometers per hour. To "
    "find "
    "when they meet, compare the distances each train has traveled.",
    "template <typename T> class RingBuffer { public: bool push(const T& "
    "value) "
    "{ if (count == capacity) return false; data[tail] = value; "
    "tail = (tail + 1) % capacity; ++count; return true; } };",
};

std::vector<float> Logits(Executor& executor) {
  const auto view = executor.CopyLastLogits();
  Expect(!view.empty() && std::ranges::all_of(
                              view, [](float v) { return std::isfinite(v); }),
         "target logits must be finite and nonempty");
  return {view.begin(), view.end()};
}

bool ByteEqual(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::string Fingerprint(std::span<const float> values) {
  Expect(
      !values.empty() &&
          std::ranges::all_of(values, [](float v) { return std::isfinite(v); }),
      "fingerprinted values must be finite and nonempty");
  return gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(values.data()),
       values.size_bytes()});
}

void CheckPrefillReplay(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& model) {
  std::string error;
  auto executor = Executor::Create(model, &error, 12288);
  Expect(executor != nullptr, error);
  constexpr std::array<std::uint32_t, 5> layers{6, 20, 34, 48, 62};
  const auto text = executor->GetTokenizer().Encode(
      std::string(kTexts[0]) + "\n" + kTexts[1] + "\n" + kTexts[2]);
  Expect(!text.empty(), "prefill fixture must tokenize");
  // A deep ragged suffix also checks restored attention state and the feature
  // taps consumed by DFlash2 after prefill uses temporary KV layouts.
  for (const auto [depth, length] :
       {std::pair{0U, 64U}, std::pair{0U, 128U}, std::pair{0U, 257U},
        std::pair{0U, 2048U}, std::pair{8192U, 1025U}}) {
    const std::uint32_t end = depth + length;
    std::vector<Token> tokens(end + 2);
    for (std::size_t i = 0; i < tokens.size(); ++i)
      tokens[i] = text[i % text.size()];
    const auto prompt = std::span(tokens).subspan(depth, length);
    executor->Reset();
    executor->SetPromptHiddenCapture(false);
    std::unique_ptr<gufo::hip::QwenGpuSnapshot> prefix;
    if (depth != 0) {
      (void)executor->ForwardPromptBatch(std::span(tokens).first(depth));
      prefix = executor->SaveSnapshot(depth);
      Expect(prefix != nullptr, "deep prefill snapshot");
    }
    executor->SetPromptHiddenCapture(true, layers);
    (void)executor->ForwardPromptBatch(prompt, depth);
    const auto expected = Logits(*executor);
    const auto features = Fingerprint(executor->GetPromptHiddenStates());
    Expect(executor->GetPromptHiddenStates().size() ==
               length * layers.size() * executor->GetConfig().hidden_size,
           "prefill must capture every requested feature row");
    if (prefix)
      executor->RestoreSnapshot(*prefix);
    else
      executor->Reset();
    (void)executor->ForwardPromptBatch(prompt, depth);
    Expect(ByteEqual(Logits(*executor), expected) &&
               Fingerprint(executor->GetPromptHiddenStates()) == features,
           "repeated matrix prefill changed logits or features");
    std::cout << "prefill fingerprint depth=" << depth << " tokens=" << length
              << " logits=" << Fingerprint(expected) << " features=" << features
              << '\n';

    const auto snapshot = executor->SaveSnapshot(end);
    const auto suffix = std::span(tokens).subspan(end);
    (void)executor->ForwardVerificationChunk(suffix, end, true);
    std::vector<std::vector<float>> verified;
    for (std::size_t row = 0; row < suffix.size(); ++row) {
      const auto logits = executor->CopyVerificationLogits(row);
      verified.emplace_back(logits.begin(), logits.end());
    }
    const auto hidden = executor->GetVerificationHiddenStates();
    const std::vector<float> verified_features(hidden.begin(), hidden.end());
    executor->RestoreSnapshot(*snapshot);
    for (std::size_t row = 0; row < suffix.size(); ++row) {
      // Changing taps after the first capture must replace its copy nodes.
      const bool reverse_taps = length == 64 && row == 1;
      if (reverse_taps) {
        auto reversed = layers;
        std::reverse(reversed.begin(), reversed.end());
        executor->SetPromptHiddenCapture(true, reversed);
      }
      (void)executor->ForwardToken(suffix[row], end + row);
      const auto last_hidden = executor->CopyLastHidden();
      const auto reference =
          std::span(verified_features)
              .subspan(row * last_hidden.size(), last_hidden.size());
      std::vector<float> expected_features(reference.begin(), reference.end());
      if (reverse_taps) {
        const auto width = executor->GetConfig().hidden_size;
        for (std::size_t layer = 0; layer < layers.size(); ++layer) {
          std::copy_n(reference.data() + (layers.size() - 1 - layer) * width,
                      width, expected_features.data() + layer * width);
        }
      }
      Expect(ByteEqual(Logits(*executor), verified[row]) &&
                 ByteEqual(last_hidden, expected_features),
             "verification after matrix prefill changed logits or features");
      std::cout << "prefill continuation tokens=" << length << " row=" << row
                << " logits=" << Fingerprint(verified[row])
                << " features=" << Fingerprint(last_hidden) << '\n';
    }
  }
}

void CheckVerificationFeatures(Executor& executor,
                               std::span<const std::vector<float>> expected) {
  const auto actual = executor.GetVerificationHiddenStates();
  const std::size_t width = expected.front().size();
  Expect(actual.size() == expected.size() * width,
         "verification feature dimensions changed");
  for (std::size_t row = 0; row < expected.size(); ++row) {
    Expect(ByteEqual(actual.subspan(row * width, width), expected[row]),
           "verification features differ from scalar decoding");
  }
  Expect(ByteEqual(executor.CopyLastHidden(), expected.back()),
         "last verification features differ from scalar decoding");
}

void CheckMixedContextBatch(const Executor& owner, bool replay) {
  for (const auto storage : {gufo::hip::QwenKvCacheStorage::kFp16,
                             gufo::hip::QwenKvCacheStorage::kFp32}) {
    auto policy = gufo::hip::QwenExecutionPolicy::Production();
    policy.kv_cache_storage = storage;
    Executor short_session(owner.GetSharedModel(), 32, policy);
    Executor long_session(owner.GetSharedModel(), 64, policy);
    Executor longest_session(owner.GetSharedModel(), 128, policy);
    const std::array<Executor*, 3> sessions{&short_session, &long_session,
                                            &longest_session};
    constexpr std::array<std::uint32_t, 3> prefix_sizes{5, 9, 13};
    constexpr std::uint32_t continuation = 4;
    std::array<std::vector<Token>, 3> tokens;
    std::array<std::vector<std::vector<float>>, 3> expected;
    for (std::size_t row = 0; row < sessions.size(); ++row) {
      auto& session = *sessions[row];
      tokens[row] = owner.GetTokenizer().Encode(kTexts[row]);
      Expect(tokens[row].size() >= prefix_sizes[row] + continuation,
             "mixed-context fixture has too few tokens");
      session.Reset();
      (void)session.ForwardPromptBatch(
          std::span(tokens[row]).first(prefix_sizes[row]));
      auto snapshot = session.SaveSnapshot(prefix_sizes[row]);
      for (std::uint32_t step = 0; step < continuation; ++step) {
        const auto position = prefix_sizes[row] + step;
        (void)session.ForwardToken(tokens[row][position], position);
        expected[row].push_back(Logits(session));
      }
      session.RestoreSnapshot(*snapshot);
      if (replay)
        session.SaveState(prefix_sizes[row]);
    }
    for (std::uint32_t step = 0; step < continuation; ++step) {
      std::array<gufo::hip::QwenGpuBatchItem, 3> items;
      for (std::size_t row = 0; row < sessions.size(); ++row) {
        const auto position = prefix_sizes[row] + step;
        items[row] = {sessions[row], tokens[row][position], position};
      }
      // Hand replayed state to each possible coordinator. The longest prefix
      // crosses the replay ring; all rows still use one scalar oracle.
      const auto coordinator = step % sessions.size();
      std::rotate(items.begin(), items.begin() + coordinator, items.end());
      const auto predictions = Executor::ForwardTokenBatch(items);
      Expect(predictions.size() == sessions.size(),
             "mixed-context batch returned the wrong number of rows");
      for (std::size_t row = 0; row < sessions.size(); ++row) {
        const auto actual = Logits(*sessions[row]);
        const bool exact = ByteEqual(actual, expected[row][step]);
        if (!exact) {
          const auto comparison =
              gufo::testing::CompareLogits(actual, expected[row][step]);
          std::cerr << "mixed-context storage=" << static_cast<int>(storage)
                    << " replay=" << replay << " row=" << row
                    << " step=" << step
                    << " max_abs=" << comparison.max_abs_diff << '\n';
        }
        Expect(exact, "mixed-context batching changed target logits");
        const auto& logits = expected[row][step];
        const auto next = static_cast<Token>(std::ranges::max_element(logits) -
                                             logits.begin());
        const std::size_t batch_row =
            (row + sessions.size() - coordinator) % sessions.size();
        Expect(predictions[batch_row] == next,
               "mixed-context batch returned a different token");
      }
      if (replay) {
        for (std::size_t row = 0; row < sessions.size(); ++row) {
          sessions[row]->RestoreState();
          sessions[row]->CommitVerificationChunk(
              std::span(tokens[row]).subspan(prefix_sizes[row], step + 1),
              prefix_sizes[row]);
        }
      }
    }
    std::cout << "mixed-context batch: storage=" << static_cast<int>(storage)
              << " replay=" << replay << " all 12 full-logit rows exact\n";
  }
}

void CheckConcurrencyWidths(const Executor& owner) {
  constexpr std::size_t count = 8;
  constexpr std::array<std::uint32_t, 5> taps{6, 20, 34, 48, 62};
  std::array<std::unique_ptr<Executor>, count> sessions;
  std::array<std::unique_ptr<gufo::hip::QwenGpuSnapshot>, count> snapshots;
  std::array<gufo::hip::QwenGpuBatchItem, count> items;
  std::array<std::vector<Token>, count> tokens;
  std::array<std::array<std::vector<float>, 9>, count> expected_logits;
  std::array<std::array<std::vector<float>, 9>, count> expected_features;
  for (std::size_t row = 0; row < count; ++row) {
    // The C8 coordinator has enough FFN scratch for all verification logits;
    // smaller coordinators exercise the bounded-allocation fallback.
    sessions[row] = std::make_unique<Executor>(owner.GetSharedModel(),
                                               row == count / 2 ? 1024
                                               : row % 2 == 0   ? 64
                                                                : 128);
    auto& session = *sessions[row];
    session.SetPromptHiddenCapture(true, taps);
    const std::string text = kTexts[row % kTexts.size()];
    tokens[row] = owner.GetTokenizer().Encode(text + "\n" + text);
    const auto position = static_cast<std::uint32_t>(5 + 4 * row);
    Expect(tokens[row].size() >= position + expected_logits[row].size(),
           "concurrency fixture has too few tokens");
    (void)session.ForwardPromptBatch(std::span(tokens[row]).first(position));
    snapshots[row] = session.SaveSnapshot(position);
    items[row] = {&session, tokens[row][position], position};
    for (std::size_t step = 0; step < expected_logits[row].size(); ++step) {
      (void)session.ForwardToken(tokens[row][position + step], position + step);
      expected_logits[row][step] = Logits(session);
      const auto features = session.CopyLastHidden();
      expected_features[row][step].assign(features.begin(), features.end());
    }
  }
  // Reuse eight scalar oracles across the required serving widths. Unequal
  // prefixes and different prompts expose accidental sharing of request state.
  for (const std::size_t width : {2U, 4U, 6U, 8U}) {
    for (std::size_t row = 0; row < width; ++row)
      sessions[row]->RestoreSnapshot(*snapshots[row]);
    const auto predictions =
        Executor::ForwardTokenBatch(std::span(items).first(width));
    Expect(predictions.size() == width, "concurrency width changed");
    for (std::size_t row = 0; row < width; ++row) {
      Expect(ByteEqual(Logits(*sessions[row]), expected_logits[row][0]) &&
                 ByteEqual(sessions[row]->CopyLastHidden(),
                           expected_features[row][0]),
             "concurrent decoding changed target logits or draft features");
      const auto& logits = expected_logits[row][0];
      Expect(predictions[row] ==
                 static_cast<Token>(std::ranges::max_element(logits) -
                                    logits.begin()),
             "concurrent decoding changed the target prediction");
    }
    std::cout << "concurrency width=" << width << " logits/features exact=1\n";

    for (const std::size_t cohort_rows : {0U, 9U, 10U, 11U, 12U, 13U, 14U, 15U,
                                          16U, 28U, 32U, 42U, 48U, 56U, 64U}) {
      if ((cohort_rows > 16 && width != count) ||
          (cohort_rows > 0 && cohort_rows <= 16 && width != 2))
        continue;
      std::array<gufo::hip::QwenGpuVerificationItem, count> verification_items;
      for (std::size_t row = 0; row < width; ++row) {
        const auto position = items[row].position;
        const std::size_t length =
            cohort_rows != 0 ? cohort_rows / width + (row < cohort_rows % width)
                             : (3 * row + width - 2) % 8 + 1;
        sessions[row]->RestoreSnapshot(*snapshots[row]);
        sessions[row]->SaveState(position);
        verification_items[row] = {
            sessions[row].get(),
            std::span(tokens[row]).subspan(position, length), position, true};
      }
      // Rotate the coordinator, vary chunk lengths and cross eight total rows.
      // Attention capacity and recurrent replay still belong to each sequence.
      const std::size_t rotation = width / 2;
      auto batch = std::span(verification_items).first(width);
      std::rotate(batch.begin(), batch.begin() + rotation, batch.end());
      const auto memory_before = batch.front().executor->GetMemoryUsage();
      const auto verified = Executor::ForwardVerificationBatch(batch);
      const auto memory_after = batch.front().executor->GetMemoryUsage();
      if (width == count && cohort_rows == 64) {
        const auto retained_limit =
            8 * owner.GetConfig().vocab_size * sizeof(float);
        Expect(memory_after.TotalBytes() - memory_before.TotalBytes() <=
                   retained_limit,
               "a coordinator must not retain the whole cohort's logits");
      }
      Expect(verified.size() == width, "verification cohort width changed");
      for (std::size_t index = 0; index < width; ++index) {
        const std::size_t row = (index + rotation) % width;
        auto& session = *sessions[row];
        const auto& item = batch[index];
        Expect(verified[index].size() == item.tokens.size(),
               "verification changed a sequence length");
        CheckVerificationFeatures(
            session,
            std::span(expected_features[row]).first(item.tokens.size()));
        for (std::size_t step = 0; step < item.tokens.size(); ++step) {
          const auto& logits = expected_logits[row][step];
          Expect(ByteEqual(session.CopyVerificationLogits(step), logits) &&
                     verified[index][step] ==
                         static_cast<Token>(std::ranges::max_element(logits) -
                                            logits.begin()),
                 "concurrent verification changed logits or predictions");
        }
        const std::size_t committed = item.tokens.size() / 2 + 1;
        session.RestoreState();
        session.CommitVerificationChunk(item.tokens.first(committed),
                                        item.position);
        (void)session.ForwardToken(tokens[row][item.position + committed],
                                   item.position + committed);
        Expect(ByteEqual(Logits(session), expected_logits[row][committed]) &&
                   ByteEqual(session.CopyLastHidden(),
                             expected_features[row][committed]),
               "concurrent verification replay changed continuation state");
      }
      std::cout << "verification concurrency=" << width
                << " cohort_rows=" << cohort_rows
                << " logits/features/replay exact=1\n";
    }
  }
  for (std::size_t row = 0; row < count; ++row) {
    sessions[row]->RestoreSnapshot(*snapshots[row]);
    sessions[row]->SaveState(items[row].position);
  }
  // Append chunks without resetting the rollback snapshot. Requests leave
  // independently, including a final single-session continuation.
  for (std::size_t stage = 0; stage < 4; ++stage) {
    const auto active = count >> stage;
    const auto offset = stage * 2;
    std::vector<gufo::hip::QwenGpuVerificationItem> chunks;
    for (std::size_t row = 0; row < active; ++row) {
      const auto position =
          items[row].position + static_cast<std::uint32_t>(offset);
      chunks.push_back({sessions[row].get(),
                        std::span(tokens[row]).subspan(position, 2), position,
                        true});
    }
    const auto rotation = stage % active;
    std::rotate(chunks.begin(), chunks.begin() + rotation, chunks.end());
    const auto predictions = Executor::ForwardVerificationBatch(chunks);
    for (std::size_t index = 0; index < active; ++index) {
      const auto row = (index + rotation) % active;
      CheckVerificationFeatures(
          *sessions[row], std::span(expected_features[row]).subspan(offset, 2));
      for (std::size_t local = 0; local < 2; ++local) {
        const auto& expected = expected_logits[row][offset + local];
        Expect(
            ByteEqual(sessions[row]->CopyVerificationLogits(local), expected) &&
                predictions[index][local] ==
                    static_cast<Token>(std::ranges::max_element(expected) -
                                       expected.begin()),
            "appended verification chunk changed logits or features");
      }
    }
    for (std::size_t row = active / 2; row < active; ++row) {
      const auto position = items[row].position;
      const auto committed = offset + 1;
      auto& session = *sessions[row];
      session.RestoreState();
      session.CommitVerificationChunk(
          std::span(tokens[row]).subspan(position, committed), position);
      (void)session.ForwardToken(tokens[row][position + committed],
                                 position + committed);
      const auto actual = Logits(session);
      if (!ByteEqual(actual, expected_logits[row][committed])) {
        const auto difference = gufo::testing::CompareLogits(
            actual, expected_logits[row][committed]);
        std::cerr << "chunk replay stage=" << stage << " row=" << row
                  << " committed=" << committed
                  << " max_abs=" << difference.max_abs_diff << '\n';
      }
      Expect(ByteEqual(actual, expected_logits[row][committed]) &&
                 ByteEqual(session.CopyLastHidden(),
                           expected_features[row][committed]),
             "rollback across verification chunks changed continuation state");
    }
  }
  std::cout << "appended verification: C8/4/2/1 logits/features/replay exact\n";
}

void CheckWideCache(Executor& reference) {
  // Exercise the 2^32-element K/V boundary without filling the context.
  Executor wide(reference.GetSharedModel(), 262144);
  auto tokens = reference.GetTokenizer().Encode(kTexts[0]);
  constexpr std::size_t prefix = 24;
  constexpr std::size_t continuation = 3;
  Expect(tokens.size() >= prefix + continuation,
         "wide-cache fixture has too few tokens");
  reference.Reset();
  (void)reference.ForwardPromptBatch(std::span(tokens).first(prefix));
  auto expected_prefill = Logits(reference);
  std::array<std::vector<float>, continuation> expected;
  for (std::size_t step = 0; step < continuation; ++step) {
    (void)reference.ForwardToken(tokens[prefix + step], prefix + step);
    expected[step] = Logits(reference);
  }
  wide.Reset();
  (void)wide.ForwardPromptBatch(std::span(tokens).first(prefix));
  Expect(ByteEqual(Logits(wide), expected_prefill),
         "wide-cache prefill changed target logits");
  wide.SaveState(prefix);
  for (std::size_t step = 0; step < continuation; ++step) {
    (void)wide.ForwardToken(tokens[prefix + step], prefix + step);
    Expect(ByteEqual(Logits(wide), expected[step]),
           "wide-cache scalar decode changed target logits");
  }
  wide.RestoreState();
  const auto predictions = wide.ForwardVerificationChunk(
      std::span(tokens).subspan(prefix, continuation), prefix, true);
  Expect(predictions.size() == continuation,
         "wide-cache verifier returned the wrong number of rows");
  for (std::size_t step = 0; step < continuation; ++step) {
    Expect(ByteEqual(wide.CopyVerificationLogits(step), expected[step]),
           "wide-cache verification changed target logits");
  }
  std::cout
      << "context=262144: prefill, scalar and verification logits exact\n";
}

enum class CheckMode { kAll, kPrefill, kConcurrency };

std::vector<Case> Capture(const char* path, bool check_replay,
                          CheckMode mode = CheckMode::kAll) {
  std::string error;
  auto owner = gufo::core::GgufReader::OpenFile(path, &error);
  Expect(owner != nullptr, error);
  auto executor = Executor::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(owner)), &error,
      128);
  Expect(executor != nullptr, error);
  Expect(executor->GetConfig().hidden_size == 5120 &&
             executor->GetConfig().vocab_size == 248320,
         "quality fixture requires Qwen3.8 27B");
  if (mode == CheckMode::kPrefill) {
    CheckPrefillReplay(executor->GetSharedModel());
    return {};
  }
  if (mode == CheckMode::kConcurrency) {
    CheckConcurrencyWidths(*executor);
    return {};
  }
  if (check_replay) {
    constexpr std::array<std::uint32_t, 5> target_layers{6, 20, 34, 48, 62};
    executor->SetPromptHiddenCapture(true, target_layers);
  }
  std::vector<Case> cases;
  for (const auto* text : kTexts) {
    // The middle case crosses the 16-row replay ring after three tokens.
    Case row{.tokens = executor->GetTokenizer().Encode(text),
             .logits = {},
             .prompt_size = cases.size() == 1 ? 29U : 24U};
    const std::size_t prompt_size = row.prompt_size;
    // Exercise a complete DFlash2 verification block, including the anchor.
    constexpr std::size_t continuation = 8;
    Expect(row.tokens.size() >= prompt_size + continuation,
           "quality fixture has too few tokens");
    row.tokens.resize(prompt_size + continuation);
    executor->Reset();
    (void)executor->ForwardPromptBatch(
        std::span(row.tokens).first(prompt_size));
    row.logits.push_back(Logits(*executor));
    std::vector<std::vector<float>> expected_features;
    auto snapshot = executor->SaveSnapshot(prompt_size);
    for (std::size_t pos = prompt_size; pos < row.tokens.size(); ++pos) {
      (void)executor->ForwardToken(row.tokens[pos], pos);
      row.logits.push_back(Logits(*executor));
      if (check_replay) {
        const auto features = executor->CopyLastHidden();
        Expect(
            features.size() == 5 * executor->GetConfig().hidden_size &&
                std::ranges::all_of(
                    features, [](float value) { return std::isfinite(value); }),
            "scalar target features must be finite and complete");
        expected_features.emplace_back(features.begin(), features.end());
      }
    }
    if (check_replay) {
      executor->RestoreSnapshot(*snapshot);
      const auto suffix = std::span(row.tokens).subspan(prompt_size);
      // Adaptive drafting exercises every width. Reuse one scalar oracle
      // and snapshot instead of loading another model or adding a suite.
      if (cases.empty()) {
        for (std::size_t width = 2; width < suffix.size(); ++width) {
          executor->RestoreSnapshot(*snapshot);
          const auto predictions = executor->ForwardVerificationChunk(
              suffix.first(width), prompt_size, true);
          Expect(predictions.size() == width, "verification width mismatch");
          for (std::size_t index = 0; index < width; ++index) {
            Expect(ByteEqual(executor->CopyVerificationLogits(index),
                             row.logits[index + 1]),
                   "adaptive-width verification changed target logits");
          }
          CheckVerificationFeatures(*executor,
                                    std::span(expected_features).first(width));
          std::cout << "verification width=" << width << " exact=1\n";
        }
        executor->RestoreSnapshot(*snapshot);
      }
      executor->SaveState(prompt_size);
      const auto predictions =
          executor->ForwardVerificationChunk(suffix, prompt_size, true);
      Expect(predictions.size() == suffix.size(),
             "verification returned the wrong number of rows");
      for (std::size_t index = 0; index < suffix.size(); ++index) {
        const auto logits = executor->CopyVerificationLogits(index);
        const auto comparison =
            gufo::testing::CompareLogits(logits, row.logits[index + 1]);
        std::cout << "verification case=" << cases.size() << " row=" << index
                  << " exact=" << ByteEqual(logits, row.logits[index + 1])
                  << " max_abs=" << comparison.max_abs_diff << '\n';
        Expect(ByteEqual(logits, row.logits[index + 1]),
               "batched verification changed target logits");
      }
      CheckVerificationFeatures(*executor, expected_features);
      std::cout << "verification features case=" << cases.size()
                << " rows=" << suffix.size() << " exact=1\n";
      executor->RestoreState();
      constexpr std::size_t committed = 5;
      executor->CommitVerificationChunk(suffix.first(committed), prompt_size);
      for (std::size_t index = committed; index < suffix.size(); ++index) {
        (void)executor->ForwardToken(suffix[index], prompt_size + index);
        Expect(ByteEqual(Logits(*executor), row.logits[index + 1]),
               "rejected draft replay changed target logits");
      }
      executor->RestoreSnapshot(*snapshot);
      for (std::size_t pos = prompt_size; pos < row.tokens.size(); ++pos) {
        (void)executor->ForwardToken(row.tokens[pos], pos);
        Expect(ByteEqual(Logits(*executor), row.logits[pos - prompt_size + 1]),
               "snapshot continuation changed target logits");
      }
      executor->Reset();
      (void)executor->ForwardPromptBatch(
          std::span(row.tokens).first(prompt_size));
      Expect(ByteEqual(Logits(*executor), row.logits.front()),
             "repeated prefill changed target logits");
      executor->Reset();
      for (std::size_t pos = 0; pos < prompt_size; ++pos) {
        (void)executor->ForwardToken(row.tokens[pos], pos,
                                     pos + 1 == prompt_size);
      }
      const auto comparison =
          gufo::testing::CompareLogits(Logits(*executor), row.logits.front());
      std::cout << "prefill case=" << cases.size()
                << " top1_match=" << comparison.top1_match
                << " rmse=" << comparison.root_mean_square_error
                << " max_abs=" << comparison.max_abs_diff << '\n';
      Expect(comparison.finite && comparison.top1_match,
             "prefill/scalar target choice differs");
    }
    cases.push_back(std::move(row));
  }
  if (check_replay) {
    CheckMixedContextBatch(*executor, false);
    CheckMixedContextBatch(*executor, true);
    CheckConcurrencyWidths(*executor);
    CheckWideCache(*executor);
    CheckPrefillReplay(executor->GetSharedModel());
  }
  return cases;
}

double LogNormalizer(std::span<const float> logits) {
  const double maximum = *std::ranges::max_element(logits);
  double sum = 0;
  for (const double value : logits)
    sum += std::exp(value - maximum);
  return maximum + std::log(sum);
}

void CompareReference(const std::vector<Case>& candidate,
                      const std::vector<Case>& reference) {
  Expect(candidate.size() == reference.size(), "reference case count differs");
  double total_kl = 0, total_tv = 0, total_nll_delta = 0;
  std::size_t rows = 0, labels = 0, top1 = 0;
  for (std::size_t c = 0; c < reference.size(); ++c) {
    Expect(candidate[c].tokens == reference[c].tokens,
           "reference and candidate tokenization differ");
    for (std::size_t r = 0; r < reference[c].logits.size(); ++r) {
      const auto& p = reference[c].logits[r];
      const auto& q = candidate[c].logits[r];
      Expect(p.size() == q.size(), "reference vocabulary differs");
      const double log_p = LogNormalizer(p), log_q = LogNormalizer(q);
      double kl = 0, tv = 0;
      for (std::size_t t = 0; t < p.size(); ++t) {
        const double probability = std::exp(p[t] - log_p);
        kl += probability * ((p[t] - log_p) - (q[t] - log_q));
        tv += std::abs(probability - std::exp(q[t] - log_q)) * 0.5;
      }
      const auto cmp = gufo::testing::CompareLogits(p, q);
      total_kl += kl;
      total_tv += tv;
      top1 += cmp.top1_match;
      ++rows;
      if (r + 1 < reference[c].logits.size()) {
        const auto label = reference[c].tokens[reference[c].prompt_size + r];
        total_nll_delta += (log_q - q[label]) - (log_p - p[label]);
        ++labels;
      }
      std::cout << "reference case=" << c << " row=" << r
                << " top1_match=" << cmp.top1_match << " kl=" << kl
                << " tv=" << tv << " rmse=" << cmp.root_mean_square_error
                << '\n';
    }
  }
  // These are measurements, not a claim that different quantizations must
  // produce identical distributions or that Gufo BF16 is independent truth.
  std::cout << "reference rows=" << rows << " top1=" << top1
            << " mean_kl=" << total_kl / rows << " mean_tv=" << total_tv / rows
            << " mean_nll_delta=" << total_nll_delta / labels << '\n';
}
}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* model = argc > 1 ? argv[1] : std::getenv("GUFO_QWEN27B_MODEL");
    if (model == nullptr) {
      std::cout << "Set GUFO_QWEN27B_MODEL for the Qwen27B target check.\n";
      return 77;
    }
    if (argc == 3 && std::string_view(argv[2]) == "--prefill-only") {
      (void)Capture(model, true, CheckMode::kPrefill);
      return 0;
    }
    if (argc == 3 && std::string_view(argv[2]) == "--concurrency-only") {
      (void)Capture(model, true, CheckMode::kConcurrency);
      return 0;
    }
    const auto candidate = Capture(model, true);
    if (argc > 2) {
      // Load one model at a time: a reference check must not require both
      // targets to remain resident in unified memory.
      CompareReference(candidate, Capture(argv[2], false));
    }
    std::cout << "Qwen27B target checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
