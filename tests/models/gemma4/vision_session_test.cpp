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

namespace g4 = gufo::models::gemma4;
namespace vision = gufo::models::gemma4::vision;
using gemma4_test::Require;

namespace {

constexpr std::uint32_t kContext = 4096;
constexpr std::uint32_t kSoftTokens = 70;

/// The prompt builder decodes PNG/JPEG, so the fixture writes a small
/// stored-deflate PNG of a nearly solid colour.
std::vector<std::uint8_t> Png(std::uint32_t seed) {
  constexpr std::uint32_t kSide = 96;
  std::vector<std::uint8_t> raw;
  for (std::uint32_t y = 0; y < kSide; ++y) {
    raw.push_back(0);  // filter: none
    for (std::uint32_t x = 0; x < kSide; ++x) {
      // Seed 1 is red, seed 2 blue, with a faint texture.
      const auto texture = static_cast<std::uint8_t>((x * 7 + y * 3) % 16);
      raw.push_back(static_cast<std::uint8_t>(seed == 1 ? 220 : 20) + texture);
      raw.push_back(static_cast<std::uint8_t>(30 + texture));
      raw.push_back(static_cast<std::uint8_t>(seed == 1 ? 20 : 220) + texture);
    }
  }
  const auto crc = [](const std::uint8_t* data, std::size_t size) {
    std::uint32_t c = 0xffffffffU;
    for (std::size_t i = 0; i < size; ++i) {
      c ^= data[i];
      for (int k = 0; k < 8; ++k)
        c = (c >> 1) ^ (0xedb88320U & (0U - (c & 1U)));
    }
    return c ^ 0xffffffffU;
  };
  std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
  const auto be32 = [](std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int s = 24; s >= 0; s -= 8)
      out.push_back(static_cast<std::uint8_t>(v >> s));
  };
  const auto chunk = [&](const char* type,
                         const std::vector<std::uint8_t>& body) {
    be32(png, static_cast<std::uint32_t>(body.size()));
    std::vector<std::uint8_t> typed(type, type + 4);
    typed.insert(typed.end(), body.begin(), body.end());
    png.insert(png.end(), typed.begin(), typed.end());
    be32(png, crc(typed.data(), typed.size()));
  };
  std::vector<std::uint8_t> header;
  be32(header, kSide);
  be32(header, kSide);
  header.insert(header.end(), {8, 2, 0, 0, 0});
  chunk("IHDR", header);
  // zlib stream of stored deflate blocks with an Adler-32 trailer.
  std::vector<std::uint8_t> z = {0x78, 0x01};
  for (std::size_t at = 0; at < raw.size(); at += 65535) {
    const auto size = static_cast<std::uint16_t>(
        std::min<std::size_t>(65535, raw.size() - at));
    z.push_back(at + size == raw.size() ? 1 : 0);
    z.push_back(static_cast<std::uint8_t>(size));
    z.push_back(static_cast<std::uint8_t>(size >> 8));
    z.push_back(static_cast<std::uint8_t>(~size));
    z.push_back(static_cast<std::uint8_t>(~size >> 8));
    z.insert(z.end(), raw.begin() + at, raw.begin() + at + size);
  }
  std::uint32_t a = 1, b = 0;
  for (const auto byte : raw) {
    a = (a + byte) % 65521;
    b = (b + a) % 65521;
  }
  be32(z, (b << 16) | a);
  chunk("IDAT", z);
  chunk("IEND", {});
  return png;
}

struct Prepared {
  vision::Prompt prompt;
  std::vector<g4::ImageSpan> spans;
  std::vector<std::shared_ptr<const vision::Encoder::Embedding>> embeddings;
  g4::ImageEmbeddings embed() const {
    return [this](std::size_t i) { return embeddings.at(i)->data(); };
  }
};

Prepared Prepare(const g4::Model& model, vision::Encoder& encoder,
                 std::uint32_t seed, const std::string& question) {
  std::vector<gufo::tokenization::ChatMessage> messages(1);
  messages[0].content =
      "Here is a picture that I generated for a small experiment about "
      "colour perception in synthetic images: " +
      question;
  messages[0].images.push_back(
      {messages[0].content.find(':') + 1,
       std::make_shared<const std::vector<std::uint8_t>>(Png(seed))});
  Prepared p;
  p.prompt = vision::Prepare(model.tokenizer(), messages, {}, {},
                             encoder.identity(), kContext, kSoftTokens);
  for (const auto& image : p.prompt.images) {
    p.embeddings.push_back(encoder.Encode(image.pixels));
    p.spans.push_back(
        {image.span.offset, image.span.rows, image.prefix_identity});
  }
  return p;
}

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
    const auto a = Prepare(*model, encoder, 1,
                           "Which colour fills it? Answer with one word.");
    const auto b = Prepare(*model, encoder, 2,
                           "Which colour fills it? Answer with one word.");
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
