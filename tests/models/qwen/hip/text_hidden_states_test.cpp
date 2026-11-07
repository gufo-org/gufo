// Qwen hidden-state encoding (`TextModelRunner::EncodeHiddenStates`) on a
// real checkpoint.
//
// GUFO_QWEN_HIDDEN_MODEL names a qwen35 GGUF (MOSS-Image uses its Qwen3.5-9B
// text encoder); without it the test skips with 77. It checks shape, repeated
// positive/empty-prompt isolation, cancellation, invalid input and that a
// concurrent session on the same weights is unaffected.
//
// GUFO_QWEN_HIDDEN_REFERENCE optionally names a file written by
// tools/qwen27b/hidden_state_reference.py from the same checkpoint's
// unquantized Hugging Face weights. The adapter's rows are then compared with
// the reference's pre-norm rows, and the final norm, applied once to the
// adapter's rows, with the reference's normalized rows. That pins the
// documented semantics: no normalization inside the runner, exactly one in
// the consumer. GUFO_QWEN_HIDDEN_MIN_COSINE overrides the per-row cosine
// floor (default 0.99, chosen for BF16/F16 checkpoints).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/ext/text.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/runner.hpp"

namespace {

using gufo::ext::TextHiddenStates;
using gufo::ext::TextHiddenStateStatus;
using gufo::ext::TextRunnerToken;

constexpr std::uint32_t kContext = 4096;
constexpr std::size_t kDecodeSteps = 4;

void Expect(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

// MOSS-Image framing; enable_thinking=True leaves the think block open.
std::string Frame(std::string_view prompt) {
  return "<|im_start|>user\n" + std::string(prompt) +
         "<|im_end|>\n<|im_start|>assistant\n<think>\n";
}

struct ReferenceCase {
  std::vector<TextRunnerToken> tokens;
  std::vector<float> pre_norm;
  std::vector<float> normalized;
};

struct Reference {
  std::size_t width{0};
  float epsilon{0};
  std::vector<float> scale;
  std::vector<ReferenceCase> cases;
};

template<typename T>
void Read(std::ifstream& in, T* values, std::size_t count) {
  in.read(reinterpret_cast<char*>(values),
          static_cast<std::streamsize>(count * sizeof(T)));
  Expect(static_cast<bool>(in), "hidden-state reference is truncated");
}

// Layout written by tools/qwen27b/hidden_state_reference.py, little endian:
// "GQHS", u32 version 1, u32 cases, u32 width, f32 epsilon, f32 scale[width],
// then per case u32 rows, u32 tokens[rows], f32 pre_norm[rows * width],
// f32 normalized[rows * width].
Reference LoadReference(const char* path) {
  std::ifstream in(path, std::ios::binary);
  Expect(static_cast<bool>(in), "cannot open the hidden-state reference");
  std::array<char, 4> magic{};
  Read(in, magic.data(), magic.size());
  Expect(std::string_view(magic.data(), magic.size()) == "GQHS",
         "not a Gufo Qwen hidden-state reference");
  std::uint32_t header[3]{};
  Read(in, header, 3);
  Expect(header[0] == 1, "unsupported hidden-state reference version");
  Reference reference;
  reference.width = header[2];
  Read(in, &reference.epsilon, 1);
  reference.scale.resize(reference.width);
  Read(in, reference.scale.data(), reference.width);
  for (std::uint32_t c = 0; c < header[1]; ++c) {
    std::uint32_t rows = 0;
    Read(in, &rows, 1);
    ReferenceCase item;
    item.tokens.resize(rows);
    Read(in, item.tokens.data(), rows);
    item.pre_norm.resize(std::size_t{rows} * reference.width);
    Read(in, item.pre_norm.data(), item.pre_norm.size());
    item.normalized.resize(item.pre_norm.size());
    Read(in, item.normalized.data(), item.normalized.size());
    reference.cases.push_back(std::move(item));
  }
  return reference;
}

TextHiddenStates Encode(const gufo::ext::TextModelRunner& runner,
                        std::span<const TextRunnerToken> tokens) {
  TextHiddenStates states;
  std::string error;
  const auto status = runner.EncodeHiddenStates(
      {.tokens = tokens, .is_cancelled = {}}, &states, &error);
  Expect(status == TextHiddenStateStatus::kOk, "encoding failed: " + error);
  Expect(states.rows == tokens.size() &&
             states.values.size() == states.rows * states.width,
         "encoded states are not rows x width");
  Expect(std::ranges::all_of(states.values,
                             [](float v) { return std::isfinite(v); }),
         "encoded states must be finite");
  return states;
}

bool SameBytes(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size_bytes()) == 0);
}

std::vector<float> Logits(gufo::hip::QwenGpuExecutor& executor) {
  const auto view = executor.CopyLastLogits();
  return {view.begin(), view.end()};
}

// Greedy prefill plus a few decode steps on a session executor, optionally
// encoding on the runner between prefill and decode.
std::vector<std::vector<float>> RunSession(
    gufo::hip::QwenGpuExecutor& session,
    std::span<const TextRunnerToken> prompt,
    const std::function<void()>& between) {
  std::vector<std::vector<float>> logits;
  session.Reset();
  auto token = session.ForwardPromptBatch(prompt, 0, true);
  logits.push_back(Logits(session));
  if (between)
    between();
  for (std::size_t step = 0; step < kDecodeSteps; ++step) {
    token = session.ForwardToken(
        token, static_cast<std::uint32_t>(prompt.size() + step));
    logits.push_back(Logits(session));
  }
  return logits;
}

double Cosine(std::span<const float> a, std::span<const float> b) {
  double dot = 0, aa = 0, bb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += double{a[i]} * b[i];
    aa += double{a[i]} * a[i];
    bb += double{b[i]} * b[i];
  }
  return dot / std::sqrt(aa * bb);
}

double Rms(std::span<const float> a) {
  double sum = 0;
  for (const float v : a)
    sum += double{v} * v;
  return std::sqrt(sum / static_cast<double>(a.size()));
}

std::vector<float> NormalizeOnce(std::span<const float> row,
                                 const Reference& reference) {
  const double inverse =
      1.0 / std::sqrt(Rms(row) * Rms(row) + reference.epsilon);
  std::vector<float> out(row.size());
  for (std::size_t i = 0; i < row.size(); ++i)
    out[i] = static_cast<float>(row[i] * inverse * reference.scale[i]);
  return out;
}

void CompareWithReference(const gufo::ext::TextModelRunner& runner,
                          const Reference& reference, std::size_t width) {
  Expect(reference.width == width,
         "hidden-state reference width does not match the GGUF");
  const char* floor_text = std::getenv("GUFO_QWEN_HIDDEN_MIN_COSINE");
  const double floor = floor_text != nullptr ? std::atof(floor_text) : 0.99;
  for (std::size_t c = 0; c < reference.cases.size(); ++c) {
    const auto& item = reference.cases[c];
    const auto states = Encode(runner, item.tokens);
    double min_pre = 1, min_once = 1, min_twice = 1, max_rms_ratio = 0;
    double min_rms_ratio = 1e30, min_unnormalized = 1;
    for (std::size_t row = 0; row < states.rows; ++row) {
      const auto at = [&](std::span<const float> values) {
        return values.subspan(row * width, width);
      };
      const auto ours = at(states.values);
      const auto pre = at(item.pre_norm);
      const auto normalized = at(item.normalized);
      const auto once = NormalizeOnce(ours, reference);
      const auto twice = NormalizeOnce(once, reference);
      min_pre = std::min(min_pre, Cosine(ours, pre));
      min_once = std::min(min_once, Cosine(once, normalized));
      min_twice = std::min(min_twice, Cosine(twice, normalized));
      min_unnormalized = std::min(min_unnormalized, Cosine(ours, normalized));
      const double ratio = Rms(ours) / Rms(pre);
      min_rms_ratio = std::min(min_rms_ratio, ratio);
      max_rms_ratio = std::max(max_rms_ratio, ratio);
    }
    std::cout << "reference case=" << c << " rows=" << states.rows
              << " min_cos_pre_norm=" << min_pre
              << " min_cos_norm_once=" << min_once
              << " min_cos_norm_twice=" << min_twice
              << " min_cos_raw_vs_normalized=" << min_unnormalized
              << " rms_ratio=[" << min_rms_ratio << ", " << max_rms_ratio
              << "]\n";
    Expect(min_pre >= floor, "adapter rows differ from the pre-norm reference");
    Expect(min_once >= floor,
           "adapter rows normalized once differ from the normalized "
           "reference");
    // Unnormalized rows keep the residual stream's magnitude; a runner that
    // already applied the norm would land near the normalized scale instead.
    Expect(min_rms_ratio > 0.9 && max_rms_ratio < 1.1,
           "adapter rows are not on the pre-norm scale");
  }
}

void Run(const char* model_path) {
  std::string error;
  auto owner = gufo::core::GgufReader::OpenFile(model_path, &error);
  Expect(owner != nullptr, error);
  auto model = gufo::hip::QwenGpuModel::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(owner)), &error);
  Expect(model != nullptr, error);
  const std::size_t width = model->GetConfig().hidden_size;
  const auto runner = gufo::models::qwen::CreateQwenTextRunner(
      model, kContext, nullptr, gufo::speculative::SpeculativeOptions{},
      gufo::speculative::DFlashDraftPolicy::kAdaptive, {}, {});

  const auto encoding = runner->HiddenStateEncoding();
  Expect(encoding.has_value() && encoding->width == width &&
             encoding->max_tokens == kContext,
         "Qwen runner must report its hidden-state width and limit");

  const auto positive = runner->Tokenize(
      Frame("A red fox reading a newspaper on a park bench, watercolor."));
  const auto empty = runner->Tokenize(Frame(""));
  const auto a = Encode(*runner, positive);
  const auto e = Encode(*runner, empty);
  Expect(a.width == width && e.width == width, "row width is hidden_size");
  Expect(SameBytes(Encode(*runner, positive).values, a.values) &&
             SameBytes(Encode(*runner, empty).values, e.values) &&
             SameBytes(Encode(*runner, positive).values, a.values),
         "repeated positive/empty encoding must be bit-identical");

  // A session on the same weights: encoding between its prefill and decode
  // must not change a single logit.
  gufo::hip::QwenGpuExecutor session(model, kContext);
  const auto baseline = RunSession(session, positive, {});
  const auto interleaved = RunSession(session, positive, [&] {
    (void)Encode(*runner, empty);
    (void)Encode(*runner, positive);
  });
  Expect(baseline.size() == interleaved.size(), "session step count");
  for (std::size_t step = 0; step < baseline.size(); ++step)
    Expect(SameBytes(baseline[step], interleaved[step]),
           "hidden-state encoding changed a concurrent session's logits");
  Expect(SameBytes(Encode(*runner, positive).values, a.values),
         "session activity changed hidden-state encoding");

  // Past the first 1,024-token step the private executor is rebuilt larger
  // and the capture spans several prefill chunks; neither may change results.
  std::vector<TextRunnerToken> long_prompt;
  while (long_prompt.size() < 2500)
    long_prompt.insert(long_prompt.end(), positive.begin(), positive.end());
  const auto long_states = Encode(*runner, long_prompt);
  Expect(SameBytes(Encode(*runner, positive).values, a.values) &&
             SameBytes(Encode(*runner, long_prompt).values, long_states.values),
         "a larger private executor changed hidden-state encoding");

  TextHiddenStates states;
  Expect(runner->EncodeHiddenStates(
             {.tokens = positive, .is_cancelled = [] { return true; }}, &states,
             &error) == TextHiddenStateStatus::kCancelled &&
             states.values.empty(),
         "cancelled encoding reports kCancelled without states");
  int polls = 0;
  const auto status = runner->EncodeHiddenStates(
      {.tokens = positive, .is_cancelled = [&polls] { return ++polls > 3; }},
      &states, &error);
  Expect(status == TextHiddenStateStatus::kCancelled ||
             status == TextHiddenStateStatus::kOk,
         "mid-pass cancellation either stops or completes");
  Expect(SameBytes(Encode(*runner, positive).values, a.values),
         "encoding after cancellation must be unaffected");

  Expect(runner->EncodeHiddenStates({}, &states, &error) ==
             TextHiddenStateStatus::kInvalidInput,
         "an empty prompt is invalid input");
  const std::vector<TextRunnerToken> outside{
      static_cast<TextRunnerToken>(model->GetConfig().vocab_size)};
  Expect(runner->EncodeHiddenStates({.tokens = outside, .is_cancelled = {}},
                                    &states, &error) ==
             TextHiddenStateStatus::kInvalidInput,
         "a token outside the vocabulary is invalid input");
  const std::vector<TextRunnerToken> too_long(kContext + 1, positive.front());
  Expect(runner->EncodeHiddenStates({.tokens = too_long, .is_cancelled = {}},
                                    &states, &error) ==
             TextHiddenStateStatus::kInvalidInput,
         "a prompt above max_tokens is invalid input");

  if (const char* path = std::getenv("GUFO_QWEN_HIDDEN_REFERENCE"))
    CompareWithReference(*runner, LoadReference(path), width);
  else
    std::cout << "GUFO_QWEN_HIDDEN_REFERENCE unset: reference comparison "
                 "skipped\n";
  std::cout << "positive rows=" << a.rows << " empty rows=" << e.rows
            << " width=" << width << '\n';
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* model =
        argc > 1 ? argv[1] : std::getenv("GUFO_QWEN_HIDDEN_MODEL");
    if (model == nullptr) {
      std::cout << "Set GUFO_QWEN_HIDDEN_MODEL for the Qwen hidden-state "
                   "check.\n";
      return 77;
    }
    Run(model);
    std::cout << "Qwen hidden-state checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
