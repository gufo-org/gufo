// Model-backed GPU qualification of the Gemma 4 target against the scalar
// reference on the same GGUF: decode arithmetic (FP32 activations) must be
// essentially exact, prefill (Q8_1 activations) must stay inside the
// envelope measured at introduction, and session prefix extension and
// snapshot restore past a sliding-ring wrap must be bitwise stable.
// Requires GUFO_GEMMA4_MODEL; exits 77 without it.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/kernels/rocm/executor.hpp"
#include "src/models/gemma4/reference.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

/// Limits recorded when the runtime was introduced (2026-09-26); they are
/// never loosened. Observed: decode mean KL 2.4e-7, prefill mean KL 0.0084.
struct Limits {
  double decode;
  double prefill;
  /// Bulk prefill vs exact rows over a realistic conversation that wraps
  /// the sliding ring (fixtures/long_conversation.txt).
  double long_prefill;
};
/// Dense targets. The long-prefill limit was set 2026-09-27 at about twice
/// the values observed under rounding-only kernel changes (0.056-0.076); a
/// broken window, ring or tile gives values above 1.
constexpr Limits kDense{1e-5, 0.015, 0.15};
/// Mixture-of-experts targets (26B-A4B, recorded 2026-09-28). Top-8 routing
/// is discrete: exact FP32 decodes that differ only in reduction order flip
/// near-tied expert choices and land between mean KL 1.3e-6 and 2.7e-5
/// (single positions up to 3e-4), so decode gets 1e-4. Prefill runs binary16
/// activations (observed 3.2e-5 to 4.0e-5); the conversation, whose exact
/// rows move with the same flips, 0.016 to 0.029.
constexpr Limits kExperts{1e-4, 1e-3, 0.06};

struct Stats {
  double mean_kl{0.0};
  double max_kl{0.0};
  std::size_t top1{0};
  std::size_t rows{0};
};

Stats Compare(const std::vector<float>& ref, const std::vector<float>& got,
              std::size_t vocab) {
  Stats s;
  s.rows = ref.size() / vocab;
  for (std::size_t r = 0; r < s.rows; ++r) {
    const float* a = &ref[r * vocab];
    const float* b = &got[r * vocab];
    double ma = -INFINITY, mb = -INFINITY;
    std::size_t aa = 0, ab = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
      Require(std::isfinite(b[i]), "non-finite GPU logit");
      if (a[i] > ma) {
        ma = a[i];
        aa = i;
      }
      if (b[i] > mb) {
        mb = b[i];
        ab = i;
      }
    }
    double za = 0.0, zb = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
      za += std::exp(a[i] - ma);
      zb += std::exp(b[i] - mb);
    }
    const double la = std::log(za) + ma, lb = std::log(zb) + mb;
    double kl = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
      const double pa = a[i] - la;
      kl += std::exp(pa) * (pa - (b[i] - lb));
    }
    s.mean_kl += kl / static_cast<double>(s.rows);
    s.max_kl = std::max(s.max_kl, kl);
    s.top1 += aa == ab ? 1 : 0;
  }
  return s;
}

std::vector<float> GpuLogits(g4::Model& model,
                             const std::vector<g4::TokenId>& tokens,
                             std::size_t chunk) {
  std::string error;
  auto session = model.CreateSession(0, &error);
  Require(session != nullptr, error);
  std::vector<float> all, part;
  for (std::size_t begin = 0; begin < tokens.size(); begin += chunk) {
    const std::size_t count = std::min(chunk, tokens.size() - begin);
    Require(session->EvaluateAll(std::span(tokens).subspan(begin, count), &part,
                                 &error),
            error);
    all.insert(all.end(), part.begin(), part.end());
  }
  return all;
}

/// Tokens from autoregressive evaluation and from DecodeStep (greedy by
/// default).
std::vector<g4::TokenId> Generate(
    g4::Model& model, std::span<const g4::TokenId> prompt, std::size_t count,
    bool speculative, gufo::sampling::SamplingConfig config = {},
    g4::Session::SpeculativeStats* stats_out = nullptr) {
  std::string error;
  auto session = model.CreateSession(0, &error);
  Require(session && session->Sync(prompt, &error), error);
  gufo::sampling::SamplerState sampler(config, {});
  std::vector<g4::TokenId> out;
  if (!speculative) {
    while (out.size() < count) {
      const auto token =
          static_cast<g4::TokenId>(sampler.Sample(session->Logits()));
      sampler.Accept(static_cast<gufo::sampling::TokenId>(token));
      out.push_back(token);
      Require(session->Evaluate(token, &error), error);
    }
    return out;
  }
  while (out.size() < count) {
    g4::Session::DecodeResult step;
    Require(session->DecodeStep(count - out.size(), sampler, &step, &error,
                                false) &&
                !step.tokens.empty(),
            error);
    out.insert(out.end(), step.tokens.begin(), step.tokens.end());
  }
  const auto& stats = session->Statistics();
  std::cout << "speculative: cycles " << stats.cycles << ", drafted "
            << stats.drafted << ", accepted " << stats.accepted << ", copied "
            << stats.copied << ", copies accepted " << stats.copied_accepted
            << ", siblings " << stats.siblings << ", siblings accepted "
            << stats.siblings_accepted << '\n';
  if (stats_out != nullptr) {
    *stats_out = stats;
  }
  return out;
}

/// Greedy speculation against autoregressive decode, logits included: after
/// every cycle the session's logits (those of the row that chose its last
/// token: a chain row or an accepted sibling's) equal decode's bit for bit.
void CheckSpeculativeLogits(g4::Model& model,
                            std::span<const g4::TokenId> prompt,
                            std::size_t count,
                            g4::Session::SpeculativeStats* stats_out) {
  std::string error;
  auto session = model.CreateSession(0, &error);
  Require(session && session->Sync(prompt, &error), error);
  gufo::sampling::SamplerState sampler({}, {});
  std::vector<g4::TokenId> ar;
  std::vector<std::vector<float>> ar_logits;
  while (ar.size() < count) {
    const auto logits = session->Logits();
    ar_logits.emplace_back(logits.begin(), logits.end());
    const auto token = static_cast<g4::TokenId>(sampler.Sample(logits));
    sampler.Accept(static_cast<gufo::sampling::TokenId>(token));
    ar.push_back(token);
    Require(session->Evaluate(token, &error), error);
  }
  // A fresh session: re-syncing this one would keep its cached prompt
  // keys, written by the forward of a different width.
  session = model.CreateSession(0, &error);
  Require(session && session->Sync(prompt, &error), error);
  sampler = gufo::sampling::SamplerState({}, {});
  std::vector<g4::TokenId> spec;
  while (spec.size() < count) {
    g4::Session::DecodeResult step;
    Require(session->DecodeStep(count - spec.size(), sampler, &step, &error,
                                false) &&
                !step.tokens.empty(),
            error);
    spec.insert(spec.end(), step.tokens.begin(), step.tokens.end());
    const auto logits = session->Logits();
    Require(std::equal(logits.begin(), logits.end(),
                       ar_logits[spec.size() - 1].begin(),
                       ar_logits[spec.size() - 1].end()),
            "speculative logits differ from decode at token " +
                std::to_string(spec.size() - 1));
  }
  Require(spec == ar, "greedy MTP differs from AR");
  const auto& stats = session->Statistics();
  stats_out->siblings += stats.siblings;
  stats_out->siblings_accepted += stats.siblings_accepted;
}

/// The fixture conversation: turns separated by "@@ user" / "@@ model".
std::vector<g4::TokenId> ConversationTokens(const g4::Model& model) {
  std::ifstream in(std::string(GUFO_GEMMA4_FIXTURES) +
                   "/long_conversation.txt");
  Require(in.good(), "long_conversation.txt is missing");
  std::vector<gufo::tokenization::ChatMessage> messages;
  std::string line;
  std::string text;
  auto role = gufo::tokenization::ChatRole::kUser;
  bool open = false;
  const auto flush = [&] {
    if (open) {
      while (!text.empty() && text.back() == '\n') {
        text.pop_back();
      }
      messages.emplace_back(role, text);
    }
    text.clear();
  };
  while (std::getline(in, line)) {
    if (line == "@@ user" || line == "@@ model") {
      flush();
      role = line == "@@ user" ? gufo::tokenization::ChatRole::kUser
                               : gufo::tokenization::ChatRole::kAssistant;
      open = true;
    } else {
      text += line + "\n";
    }
  }
  flush();
  std::string error;
  const auto rendered = g4::ChatTemplate::Render(messages, {}, {}, &error);
  Require(rendered.has_value(), error);
  return model.Tokenize(rendered->text);
}

std::vector<g4::TokenId> PromptTokens(const g4::Model& model,
                                      const std::string& text) {
  gufo::tokenization::ChatMessage message(gufo::tokenization::ChatRole::kUser,
                                          text);
  std::string error;
  const auto rendered =
      g4::ChatTemplate::Render(std::span(&message, 1), {}, {}, &error);
  Require(rendered.has_value(), error);
  return model.Tokenize(rendered->text);
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GEMMA4_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: GUFO_GEMMA4_MODEL is not set\n";
    return 77;
  }
  const char* draft = std::getenv("GUFO_GEMMA4_MTP_MODEL");
  return gemma4_test::Run([&] {
    std::string error;
    g4::ModelOptions options;
    // A small prefill chunk keeps the sliding ring short (window + 256), so
    // the snapshot check below wraps it inside this context.
    options.max_context = 3072;
    // 240 rows plus an absorbed remainder of up to 16 make 256-row forwards.
    options.prefill_chunk = 240;
    auto model = g4::Model::Load(path, options, &error);
    Require(model != nullptr, error);
    const std::size_t vocab = model->VocabSize();
    const auto tokens =
        PromptTokens(*model, "Write one sentence about the sea near Genoa.");
    Require(tokens.size() > 16, "prompt too short for the prefill path");

    // Independent scalar reference over the same weights.
    const auto weights = g4::ModelWeights::Bind(model->reader(), &error);
    Require(weights.has_value(), error);
    g4::Reference reference(*weights, g4::Reference::Storage::kHalfKv);
    std::vector<float> oracle;
    reference.Forward(tokens, &oracle, nullptr);

    const auto decode = GpuLogits(*model, tokens, 1);
    const auto chunked = GpuLogits(*model, tokens, 8);
    const auto prefill = GpuLogits(*model, tokens, tokens.size());
    const Stats d = Compare(oracle, decode, vocab);
    const Stats c = Compare(oracle, chunked, vocab);
    const Stats p = Compare(oracle, prefill, vocab);
    std::cout << "decode:  mean KL " << d.mean_kl << ", max " << d.max_kl
              << ", top-1 " << d.top1 << "/" << d.rows << '\n'
              << "8-row:   mean KL " << c.mean_kl << ", top-1 " << c.top1 << "/"
              << c.rows << '\n'
              << "prefill: mean KL " << p.mean_kl << ", max " << p.max_kl
              << ", top-1 " << p.top1 << "/" << p.rows << '\n';
    const Limits& limits = model->config().HasExperts() ? kExperts : kDense;
    Require(d.mean_kl < limits.decode && d.top1 == d.rows,
            "decode diverges from the reference");
    Require(c.mean_kl < limits.decode && c.top1 == c.rows,
            "small-batch rows diverge from the reference");
    Require(p.mean_kl < limits.prefill, "prefill exceeds its KL envelope");

    // Extending a synced prompt matches evaluating the tokens one by one.
    const std::size_t split = tokens.size() - 4;
    const std::span<const g4::TokenId> head(tokens.data(), split);
    auto a = model->CreateSession(0, &error);
    auto b = model->CreateSession(0, &error);
    Require(a && b, error);
    Require(a->Sync(head, &error) && b->Sync(head, &error), error);
    for (std::size_t i = split; i < tokens.size(); ++i) {
      Require(a->Evaluate(tokens[i], &error), error);
    }
    Require(b->Sync(tokens, &error), error);
    const auto as_vector = [](std::span<const float> v) {
      return std::vector<float>(v.begin(), v.end());
    };
    const Stats ext =
        Compare(as_vector(a->Logits()), as_vector(b->Logits()), vocab);
    Require(ext.top1 == 1 && ext.mean_kl < limits.decode,
            "prefix extension diverges from token-by-token evaluation");
    // Rewinding to a shorter prompt reuses the prefix and matches a fresh sync.
    Require(b->Sync(head, &error), error);
    auto fresh = model->CreateSession(0, &error);
    Require(fresh && fresh->Sync(head, &error), error);
    const Stats rw =
        Compare(as_vector(fresh->Logits()), as_vector(b->Logits()), vocab);
    Require(rw.top1 == 1 && rw.mean_kl < limits.prefill,
            "rewound session diverges from a fresh one");

    // A snapshot taken after the ring wrapped restores into a fresh session
    // (typed and serialized) and continues bitwise like the original.
    std::string text;
    for (int i = 0; i < 90; ++i) {
      text += "Ship " + std::to_string(i) +
              " left Genoa at dawn carrying salt, wine and letters. ";
    }
    const auto long_prompt = PromptTokens(*model, text);
    Require(long_prompt.size() > model->SessionRingSlots() + 64 &&
                long_prompt.size() + 16 < options.max_context,
            "snapshot prompt (" + std::to_string(long_prompt.size()) +
                " tokens) does not wrap the sliding ring (" +
                std::to_string(model->SessionRingSlots()) + " slots)");
    auto original = model->CreateSession(0, &error);
    Require(original && original->Sync(long_prompt, &error), error);
    const auto snapshot = original->SaveSnapshot(&error);
    Require(snapshot != nullptr &&
                snapshot->SizeBytes() == original->SnapshotBytes(),
            error);
    std::vector<std::uint8_t> bytes(snapshot->SizeBytes());
    Require(snapshot->CopyTo(bytes), "snapshot payload copy failed");
    std::vector<std::uint8_t> streamed;
    Require(snapshot->Stream([&](std::span<const std::uint8_t> piece) {
      streamed.insert(streamed.end(), piece.begin(), piece.end());
    }) && streamed == bytes,
            "streamed snapshot payload differs from its copy");
    auto typed = model->CreateSession(0, &error);
    auto serialized = model->CreateSession(0, &error);
    Require(typed && typed->RestoreSnapshot(*snapshot, &error), error);
    Require(serialized && serialized->RestoreSnapshot(bytes, &error), error);
    for (int step = 0; step <= 8; ++step) {
      const auto logits = as_vector(original->Logits());
      Require(as_vector(typed->Logits()) == logits &&
                  as_vector(serialized->Logits()) == logits,
              "restored session diverges at step " + std::to_string(step));
      if (step == 8) {
        break;
      }
      const auto next = static_cast<g4::TokenId>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      Require(original->Evaluate(next, &error) &&
                  typed->Evaluate(next, &error) &&
                  serialized->Evaluate(next, &error),
              error);
    }
    // Snapshots of one history share their position blocks. Shared and
    // freshly copied snapshots of equal states hold equal payloads, and a
    // restore that skips the blocks its cache still holds continues exactly
    // like one from the serialized payload, also after the session rewrote
    // positions inside them or wrapped the sliding ring over them.
    const auto payload = [](const g4::SessionSnapshot& s) {
      std::vector<std::uint8_t> out(s.SizeBytes());
      Require(s.CopyTo(out), "snapshot payload copy failed");
      return out;
    };
    const auto from_bytes = [&] {
      auto session = model->CreateSession(0, &error);
      Require(session && session->RestoreSnapshot(bytes, &error), error);
      return session;
    };
    {
      const auto plain = from_bytes();
      for (const g4::TokenId token :
           std::span(original->Tokens()).subspan(long_prompt.size())) {
        Require(plain->Evaluate(token, &error), error);
      }
      const auto reference = plain->SaveSnapshot(&error);
      const auto shared = original->SaveSnapshot(&error);
      const auto adopted = typed->SaveSnapshot(&error);
      Require(reference && shared && adopted, error);
      Require(payload(*shared) == payload(*reference) &&
                  payload(*adopted) == payload(*reference),
              "shared snapshot payload differs from a fresh copy");
    }
    std::string other;
    for (int i = 0; i < 60; ++i) {
      other += "Harbour " + std::to_string(i) +
               " counted barrels of oil, crates of lemons and coils of rope. ";
    }
    const auto other_tokens = PromptTokens(*model, other);
    const std::size_t n1 = long_prompt.size();
    const std::size_t window = model->config().sliding_window;
    const std::size_t ring = model->SessionRingSlots();
    // The first whole sliding block (64 positions) of the snapshot's window,
    // and enough tokens past the snapshot to overwrite 16 of its ring slots.
    const std::size_t sliding_block = (n1 - (window - 1) + 63) / 64 * 64;
    const std::size_t wrap = sliding_block + ring + 16 - n1;
    Require(sliding_block + 64 <= n1 && other_tokens.size() >= wrap &&
                n1 + wrap + 16 < options.max_context,
            "snapshot test cannot wrap the ring over a sliding block");
    const auto check_restore = [&](std::span<const g4::TokenId> prompt,
                                   const std::string& what) {
      Require(original->Sync(prompt, &error), error);
      Require(original->RestoreSnapshot(*snapshot, &error), error);
      const auto plain = from_bytes();
      for (int step = 0; step < 4; ++step) {
        const auto logits = as_vector(plain->Logits());
        Require(as_vector(original->Logits()) == logits,
                what + ": restore diverges at step " + std::to_string(step));
        const auto next = static_cast<g4::TokenId>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        Require(
            original->Evaluate(next, &error) && plain->Evaluate(next, &error),
            error);
      }
      const auto shared = original->SaveSnapshot(&error);
      const auto reference = plain->SaveSnapshot(&error);
      Require(shared && reference && payload(*shared) == payload(*reference),
              what + ": shared snapshot payload differs from a fresh copy");
    };
    check_restore(long_prompt, "held blocks");
    std::vector<g4::TokenId> rewritten(long_prompt.begin(),
                                       long_prompt.end() - 200);
    rewritten.insert(rewritten.end(), other_tokens.begin(),
                     other_tokens.begin() + 100);
    check_restore(rewritten, "rewritten blocks");
    std::vector<g4::TokenId> wrapped = long_prompt;
    wrapped.insert(wrapped.end(), other_tokens.begin(),
                   other_tokens.begin() + static_cast<std::ptrdiff_t>(wrap));
    check_restore(wrapped, "wrapped ring");

    std::vector<std::uint8_t> corrupt = bytes;
    corrupt.pop_back();
    Require(!serialized->RestoreSnapshot(corrupt, &error),
            "truncated snapshot was accepted");
    original.reset();
    typed.reset();
    serialized.reset();

    // Sync's lookahead (the generation prompt after a cache boundary) runs
    // in the prompt's last forward while the session ends at the boundary.
    // The adopting Sync keeps the KV of a whole-prompt Sync, so greedy
    // decoding continues bitwise; its first logits come from a two-row head
    // projection. A snapshot taken at the boundary restores and extends.
    {
      constexpr std::size_t kTail = 7;
      const std::span<const g4::TokenId> head(long_prompt.data(),
                                              long_prompt.size() - kTail);
      const std::span<const g4::TokenId> tail(long_prompt.data() + head.size(),
                                              kTail);
      auto whole = model->CreateSession(0, &error);
      auto ahead = model->CreateSession(0, &error);
      Require(whole && whole->Sync(long_prompt, &error), error);
      Require(ahead && ahead->Sync(head, {}, {}, &error, tail) &&
                  ahead->Position() == head.size(),
              "lookahead sync: " + error);
      const auto boundary = ahead->SaveSnapshot(&error);
      Require(boundary != nullptr, error);
      Require(ahead->Sync(long_prompt, &error) &&
                  ahead->Position() == long_prompt.size(),
              "lookahead adoption: " + error);
      const auto first = as_vector(whole->Logits());
      const Stats adopted = Compare(first, as_vector(ahead->Logits()), vocab);
      Require(adopted.top1 == 1 && adopted.mean_kl < limits.decode,
              "adopted lookahead logits diverge");
      for (int step = 0; step < 8; ++step) {
        const auto logits = as_vector(whole->Logits());
        const auto next = static_cast<g4::TokenId>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        Require(whole->Evaluate(next, &error) && ahead->Evaluate(next, &error),
                error);
        Require(as_vector(ahead->Logits()) == as_vector(whole->Logits()),
                "adopted lookahead diverges at step " + std::to_string(step));
      }
      auto restored = model->CreateSession(0, &error);
      Require(restored && restored->RestoreSnapshot(*boundary, &error) &&
                  restored->Position() == head.size() &&
                  restored->Sync(long_prompt, &error),
              "boundary snapshot: " + error);
      const Stats resumed =
          Compare(first, as_vector(restored->Logits()), vocab);
      Require(resumed.top1 == 1 && resumed.mean_kl < limits.prefill,
              "boundary snapshot diverges after its tail");
    }

    // Bulk prefill over many attention tiles and a wrapped ring stays close
    // to the exact small-batch rows on a realistic conversation.
    {
      const auto conversation = ConversationTokens(*model);
      Require(conversation.size() > model->SessionRingSlots() + 64 &&
                  conversation.size() + 16 < options.max_context,
              "conversation (" + std::to_string(conversation.size()) +
                  " tokens) does not wrap the sliding ring");
      const auto exact = GpuLogits(*model, conversation, 8);
      const auto bulk = GpuLogits(*model, conversation, conversation.size());
      const Stats lp = Compare(exact, bulk, vocab);
      std::cout << "long prefill vs exact rows (" << lp.rows
                << " tokens): mean KL " << lp.mean_kl << ", max " << lp.max_kl
                << ", top-1 " << lp.top1 << "/" << lp.rows << '\n';
      Require(lp.mean_kl < limits.long_prefill,
              "long prefill exceeds its KL envelope");
    }
    // Reported only: the repeated snapshot prompt is ill-conditioned (many
    // near-tied predictions), so any rounding change moves this value
    // (llama.cpp's teacher-forced logits sit at mean KL 2.93 from the same
    // exact rows).
    {
      const auto exact = GpuLogits(*model, long_prompt, 8);
      const auto bulk = GpuLogits(*model, long_prompt, long_prompt.size());
      const Stats lp = Compare(exact, bulk, vocab);
      std::cout << "repetitive prefill vs exact rows (" << lp.rows
                << " tokens, reported): mean KL " << lp.mean_kl << ", top-1 "
                << lp.top1 << "/" << lp.rows << '\n';
    }

    if (draft == nullptr || *draft == '\0') {
      std::cout
          << "note: GUFO_GEMMA4_MTP_MODEL not set; speculation unchecked\n";
      return;
    }
    model.reset();
    options.mtp_model_path = draft;
    // The longest chain the engine allows: 16-row verification.
    options.draft_tokens = g4::rocm::kMaxDraftTokens;
    // Seeded sampled replay is exact when the calibrated policy restarts its
    // calibration with every request; the shared default learns across
    // requests, which changes later draft counts and so their random draws.
    options.draft_calibration = g4::DraftCalibrationScope::kRequest;
    auto mtp = g4::Model::Load(path, options, &error);
    Require(mtp != nullptr, error);
    // With a drafter, verification rows round exactly like decode...
    const auto single = GpuLogits(*mtp, tokens, 1);
    Require(single == GpuLogits(*mtp, tokens, 5) &&
                single == GpuLogits(*mtp, tokens, 16),
            "verification rows differ from single-token decode");
    // ...so greedy speculation reproduces autoregressive output.
    for (const char* text :
         {"Write one sentence about the sea near Genoa.",
          "List the first twelve prime numbers, separated by commas.",
          "Explain in three sentences why the sky is blue."}) {
      const auto prompt = PromptTokens(*mtp, text);
      const auto ar = Generate(*mtp, prompt, 64, false);
      const auto spec = Generate(*mtp, prompt, 64, true);
      Require(ar == spec, std::string("greedy MTP differs from AR: ") + text);
    }
    // Siblings: an accepted one's row and the keys it leaves in the cache
    // continue exactly as decode would.
    {
      g4::Session::SpeculativeStats sibling_stats;
      for (const char* text :
           {"Write a paragraph about the history of Genoa's harbour.",
            "Describe an autumn morning in a mountain village.",
            "Explain how a lighthouse lens concentrates light."}) {
        CheckSpeculativeLogits(*mtp, PromptTokens(*mtp, text), 96,
                               &sibling_stats);
      }
      std::cout << "siblings: verified " << sibling_stats.siblings
                << ", accepted " << sibling_stats.siblings_accepted << '\n';
      Require(sibling_stats.siblings_accepted > 0,
              "greedy MTP accepted no siblings");
    }
    // Batched decoding: sessions sharing a forward reproduce their own
    // single-session output (greedy), for plain and speculative steps.
    {
      const char* texts[] = {
          "Write one sentence about the sea near Genoa.",
          "List the first twelve prime numbers, separated by commas.",
          "Explain in three sentences why the sky is blue."};
      std::vector<std::vector<g4::TokenId>> want;
      std::vector<std::unique_ptr<g4::Session>> sessions;
      for (const char* text : texts) {
        const auto prompt = PromptTokens(*mtp, text);
        want.push_back(Generate(*mtp, prompt, 48, false));
        sessions.push_back(mtp->CreateSession(0, &error));
        Require(sessions.back() && sessions.back()->Sync(prompt, &error),
                error);
      }
      // Plain decode: one token per session per forward.
      std::vector<std::vector<g4::TokenId>> got(std::size(texts));
      std::vector<gufo::sampling::SamplerState> samplers;
      for (std::size_t i = 0; i < std::size(texts); ++i) {
        samplers.emplace_back(gufo::sampling::SamplingConfig{},
                              std::span<const gufo::sampling::TokenId>{});
      }
      std::vector<g4::Session*> raw;
      for (auto& session : sessions) {
        raw.push_back(session.get());
      }
      while (got[0].size() < 48) {
        std::vector<g4::TokenId> next;
        for (std::size_t i = 0; i < raw.size(); ++i) {
          const auto token =
              static_cast<g4::TokenId>(samplers[i].Sample(raw[i]->Logits()));
          samplers[i].Accept(static_cast<gufo::sampling::TokenId>(token));
          got[i].push_back(token);
          next.push_back(token);
        }
        Require(g4::Session::EvaluateBatch(raw, next, &error), error);
      }
      Require(got == want, "batched decode differs from single sessions");
      // Speculative decode: every session drafts and all verify together.
      std::vector<std::vector<g4::TokenId>> spec(std::size(texts));
      for (std::size_t i = 0; i < std::size(texts); ++i) {
        Require(sessions[i]->Sync(PromptTokens(*mtp, texts[i]), &error), error);
        samplers[i] = gufo::sampling::SamplerState(
            gufo::sampling::SamplingConfig{},
            std::span<const gufo::sampling::TokenId>{});
      }
      std::vector<g4::Session::DecodeResult> results(std::size(texts));
      while (true) {
        std::vector<g4::Session::BatchDecode> batch;
        for (std::size_t i = 0; i < std::size(texts); ++i) {
          if (spec[i].size() < 48) {
            batch.push_back({.session = raw[i],
                             .max_tokens = 48 - spec[i].size(),
                             .sampler = &samplers[i],
                             .result = &results[i],
                             .stop_at_eos = false,
                             .error = {}});
          }
        }
        if (batch.empty()) {
          break;
        }
        Require(g4::Session::DecodeBatch(batch), batch.front().error);
        for (std::size_t i = 0; i < std::size(texts); ++i) {
          if (spec[i].size() < 48) {
            spec[i].insert(spec[i].end(), results[i].tokens.begin(),
                           results[i].tokens.end());
          }
        }
      }
      Require(spec == want, "batched speculative decode differs from AR");
    }
    // Prompt lookup: repeating a passage from the prompt proposes copies,
    // verified like drafts, so greedy output still equals AR.
    const auto copy_prompt = PromptTokens(
        *mtp,
        "Repeat the following paragraph word for word, with nothing else:\n\n"
        "A harbour town usually begins with a sheltered cove and a handful of "
        "families who fish close to shore. The first boats are small, pulled "
        "up on the beach at night, and the village lives by the rhythm of the "
        "seasons: anchovies in spring, sardines in summer, storms that keep "
        "everyone ashore in winter.");
    g4::Session::SpeculativeStats copy_stats;
    const auto copy_ar = Generate(*mtp, copy_prompt, 64, false);
    const auto copy_spec =
        Generate(*mtp, copy_prompt, 64, true, {}, &copy_stats);
    Require(copy_ar == copy_spec,
            "greedy MTP with prompt lookup differs from AR");
    Require(copy_stats.copied_accepted > 0,
            "prompt lookup proposed no accepted copies");
    // Sampled MTP (a chat front end's sampler): drafts are sampled and
    // verified by p/q rejection; a seed replays the same tokens.
    gufo::sampling::SamplingConfig chat;
    chat.temperature = 1.0F;
    chat.top_k = 64;
    chat.top_p = 0.95F;
    chat.repeat_penalty = 1.05F;
    chat.seed = 7;
    const auto prompt =
        PromptTokens(*mtp, "Write a short story about a lighthouse keeper.");
    g4::Session::SpeculativeStats stats;
    const auto first = Generate(*mtp, prompt, 96, true, chat, &stats);
    const auto again = Generate(*mtp, prompt, 96, true, chat);
    Require(first.size() == 96 && first == again,
            "sampled MTP does not replay its seed");
    Require(stats.accepted > 0, "sampled MTP accepted no drafts");
    // Siblings depend on the rows a batch leaves: exact replay drafts none.
    Require(stats.siblings == 0, "exact-replay sampled MTP drafted siblings");
    // Sampled copies are point-mass proposals under the same rule.
    g4::Session::SpeculativeStats sampled_copy_stats;
    const auto sampled_copy =
        Generate(*mtp, copy_prompt, 64, true, chat, &sampled_copy_stats);
    Require(sampled_copy == Generate(*mtp, copy_prompt, 64, true, chat),
            "sampled MTP with prompt lookup does not replay its seed");
    Require(sampled_copy_stats.copied_accepted > 0,
            "sampled prompt lookup accepted no copies");
    // The shared default verifies sampled siblings by recursive rejection
    // sampling: some are drafted and some accepted.
    mtp.reset();
    options.draft_calibration = g4::DraftCalibrationScope::kShared;
    auto shared = g4::Model::Load(path, options, &error);
    Require(shared != nullptr, error);
    g4::Session::SpeculativeStats shared_stats;
    for (int i = 0; i < 3; ++i) {
      chat.seed = 7 + i;
      Require(
          Generate(*shared, prompt, 96, true, chat, &shared_stats).size() == 96,
          "shared sampled MTP stopped early");
    }
    std::cout << "sampled siblings: verified " << shared_stats.siblings
              << ", accepted " << shared_stats.siblings_accepted << '\n';
    Require(shared_stats.siblings > 0 && shared_stats.siblings_accepted > 0,
            "shared sampled MTP verified or accepted no siblings");
  });
}
