// Gemma 4 sessions with images: prefix reuse must never cross a changed
// image, resumed sessions must stay inside the prefill envelope of fresh
// ones (target_test.cpp), snapshots must restore bit for bit with their
// images, and prefill chunks must not split an image. Requires
// GUFO_GEMMA4_MODEL and GUFO_GEMMA4_MMPROJ; exits 77 without them.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/vision/encoder.hpp"
#include "src/models/gemma4/vision/prompt.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/image_fixture.hpp"

namespace g4 = gufo::models::gemma4;
namespace vision = gufo::models::gemma4::vision;
using gemma4_test::Prepared;
using gemma4_test::Require;

namespace {

constexpr std::uint32_t kContext = 4096;
constexpr std::uint32_t kSoftTokens = 70;

std::vector<float> Logits(const g4::Session& session) {
  return {session.Logits().begin(), session.Logits().end()};
}

double MeanKl(const std::vector<float>& ref, const std::vector<float>& got) {
  double ma = -INFINITY, mb = -INFINITY;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    ma = std::max<double>(ma, ref[i]);
    mb = std::max<double>(mb, got[i]);
  }
  double za = 0, zb = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    za += std::exp(ref[i] - ma);
    zb += std::exp(got[i] - mb);
  }
  double kl = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double pa = ref[i] - ma - std::log(za);
    kl += std::exp(pa) * (pa - (got[i] - mb - std::log(zb)));
  }
  return kl;
}

/// Prefill envelope of target_test.cpp (resumed vs fresh prefill).
constexpr double kPrefillKl = 0.015;

std::size_t Argmax(const std::vector<float>& v) {
  return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) -
                                  v.begin());
}

void RequireClose(const std::vector<float>& want, const std::vector<float>& got,
                  const std::string& what) {
  const double kl = MeanKl(want, got);
  std::cout << what << ": KL " << kl << '\n';
  Require(kl < kPrefillKl && Argmax(want) == Argmax(got), what + " diverges");
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_GEMMA4_MODEL");
  const char* mmproj = std::getenv("GUFO_GEMMA4_MMPROJ");
  if (model_path == nullptr || mmproj == nullptr || *model_path == '\0' ||
      *mmproj == '\0') {
    std::cerr << "SKIP: set GUFO_GEMMA4_MODEL and GUFO_GEMMA4_MMPROJ\n";
    return 77;
  }
  return gemma4_test::Run([&] {
    std::string error;
    g4::ModelOptions options;
    options.max_context = kContext;
    const auto model = g4::Model::Load(model_path, options, &error);
    Require(model != nullptr, error);
    vision::Encoder encoder(mmproj, model->config().hidden_size);
    const auto prepare = [&](std::uint32_t seed) {
      return gemma4_test::Prepare(
          *model, encoder, seed, "Which colour fills it? Answer with one word.",
          kContext, kSoftTokens);
    };
    const auto a = prepare(1);
    const auto b = prepare(2);
    Require(a.prompt.tokens == b.prompt.tokens &&
                a.spans[0].identity != b.spans[0].identity,
            "fixture images must share tokens and differ in identity");

    const auto fresh = [&](const Prepared& p) {
      auto session = model->CreateSession(kContext, &error);
      Require(
          session && session->Sync(p.prompt.tokens, p.spans, p.embed(), &error),
          error);
      return Logits(*session);
    };
    const auto logits_a = fresh(a);
    const auto logits_b = fresh(b);
    const double between = MeanKl(logits_a, logits_b);
    std::cout << "image A vs image B: KL " << between << '\n';
    Require(between > 10 * kPrefillKl && Argmax(logits_a) != Argmax(logits_b),
            "the image does not decide the answer");

    // Same tokens, other image: the session must re-evaluate from the image.
    auto session = model->CreateSession(kContext, &error);
    Require(session->Sync(a.prompt.tokens, a.spans, a.embed(), &error), error);
    Require(Logits(*session) == logits_a, "repeat of a fresh sync differs");
    Require(session->Sync(b.prompt.tokens, b.spans, b.embed(), &error), error);
    RequireClose(logits_b, Logits(*session), "image A then image B");
    Require(session->Sync(a.prompt.tokens, a.spans, a.embed(), &error), error);
    RequireClose(logits_a, Logits(*session), "back to image A");

    // Extending past the image reuses it; the result equals a fresh sync.
    auto longer = a;
    const auto more = model->Tokenize(" Please answer briefly.");
    longer.prompt.tokens.insert(longer.prompt.tokens.end() - 1, more.begin(),
                                more.end());
    Require(session->Sync(longer.prompt.tokens, longer.spans, longer.embed(),
                          &error),
            error);
    RequireClose(fresh(longer), Logits(*session), "extension after an image");

    // Snapshots keep their images.
    const auto snapshot = session->SaveSnapshot(&error);
    Require(snapshot != nullptr, error);
    auto restored = model->CreateSession(kContext, &error);
    Require(restored->RestoreSnapshot(*snapshot, &error), error);
    Require(std::ranges::equal(restored->Images(), session->Images()) &&
                Logits(*restored) == Logits(*session),
            "snapshot lost its images");
    Require(restored->Sync(b.prompt.tokens, b.spans, b.embed(), &error), error);
    RequireClose(logits_b, Logits(*restored), "restored, then image B");

    // A prefill chunk ending inside the image moves before it.
    g4::ModelOptions small = options;
    // Ends inside the image, yet holds a whole image.
    small.prefill_chunk =
        std::max(a.spans[0].rows, a.spans[0].offset + a.spans[0].rows / 2);
    Require(small.prefill_chunk > a.spans[0].offset &&
                small.prefill_chunk < a.spans[0].offset + a.spans[0].rows,
            "fixture image does not cross the chunk");
    const auto chunked = g4::Model::Load(model_path, small, &error);
    Require(chunked != nullptr, error);
    auto split = chunked->CreateSession(kContext, &error);
    Require(split->Sync(a.prompt.tokens, a.spans, a.embed(), &error), error);
    RequireClose(logits_a, Logits(*split), "prefill chunk ending in an image");
  });
}
