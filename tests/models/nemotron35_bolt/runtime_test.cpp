// Session-level checks of the ROCm runtime on the pinned model: incremental
// decode (the captured single-token graph) against one bulk prefill, exact
// snapshot restore into another session, and invalidation. With the LiLiCorr
// drafter (second argument or GUFO_NEMOTRON_BOLT_DRAFTER) also: a verified
// block committed in part against appending that part, a discarded
// verification, drafts from a restored snapshot, and draft-and-verify rounds
// against plain greedy decoding. Skips without a model path (argument or
// GUFO_NEMOTRON_BOLT_GGUF).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/nemotron35_bolt/drafter.hpp"
#include "src/models/nemotron35_bolt/engine.hpp"
#include "src/models/nemotron35_bolt/speculative.hpp"

namespace nb = gufo::models::nemotron35_bolt;

namespace {

constexpr int kSkip = 77;
int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++failures;
  }
}

void Require(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    std::exit(1);
  }
}

double Kl(const std::vector<float>& p_logits,
          const std::vector<float>& q_logits) {
  auto log_softmax = [](const std::vector<float>& v) {
    const float m = *std::max_element(v.begin(), v.end());
    double z = 0;
    for (float x : v)
      z += std::exp(double(x) - m);
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
      out[i] = double(v[i]) - m - std::log(z);
    return out;
  };
  const auto p = log_softmax(p_logits);
  const auto q = log_softmax(q_logits);
  double kl = 0;
  for (std::size_t i = 0; i < p.size(); ++i)
    kl += std::exp(p[i]) * (p[i] - q[i]);
  return kl;
}

std::size_t Argmax(const std::vector<float>& v) {
  return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) -
                                  v.begin());
}

/// Device heads of real rows, from a verified block and from a decode step,
/// draw the full rows' tokens for top-p and top-k sampling.
void CheckHeads(nb::Model& model, const std::vector<std::int32_t>& prompt,
                const std::vector<std::int32_t>& block) {
  std::string error;
  const std::size_t vocab = model.config().vocab_size;
  auto session = model.CreateSession(4096, &error);
  std::vector<float> logits(vocab);
  Require(session && session->Append(prompt, logits, &error),
          "prefill: " + error);
  for (const auto& config :
       {gufo::sampling::SamplingConfig{.temperature = 1.0F, .top_p = 0.95F},
        gufo::sampling::SamplingConfig{.temperature = 0.6F, .top_k = 20}}) {
    const auto filter = *config.logit_head_filter();
    std::vector<float> rows(block.size() * vocab);
    std::vector<gufo::sampling::LogitHead> heads(block.size());
    Require(session->Verify(block, rows, filter, heads, &error) &&
                session->Commit(0, &error),
            "verify: " + error);
    nb::Frontier step;
    step.logits.resize(vocab);
    const std::int32_t one[] = {block[0]};
    Require(session->Append(one, step.logits, filter, &step.head, &error),
            "decode: " + error);
    heads.push_back(step.head);
    rows.insert(rows.end(), step.logits.begin(), step.logits.end());
    std::size_t summarized = 0;
    bool same = true;
    for (std::size_t r = 0; r < heads.size(); ++r) {
      summarized += !heads[r].candidates.empty();
      const std::span<const float> row(rows.data() + r * vocab, vocab);
      for (std::int64_t seed = 0; seed < 16; ++seed) {
        auto seeded = config;
        seeded.seed = seed;
        gufo::sampling::SamplerState a(seeded), b(seeded);
        same = same && a.Sample(row) == b.Sample(row, heads[r]) &&
               a.rng_state() == b.rng_state();
      }
    }
    std::printf("device heads (T %.1f, top-k %d): %zu of %zu rows\n",
                config.temperature, config.top_k, summarized, heads.size());
    Check(summarized * 2 > heads.size(), "most rows get a device head");
    Check(same, "device heads draw the full rows' tokens");
    Require(session->Reset(&error) && session->Append(prompt, logits, &error),
            "restart: " + error);
  }
}

/// Greedy tokens from `logits` onward, one decode step at a time.
std::vector<std::int32_t> Greedy(nb::Session& session,
                                 std::vector<float> logits, std::size_t count) {
  std::vector<std::int32_t> out;
  std::string error;
  for (std::size_t i = 0; i < count; ++i) {
    const auto next = static_cast<std::int32_t>(Argmax(logits));
    out.push_back(next);
    const std::int32_t one[] = {next};
    Require(session.Append(one, logits, &error), "greedy: " + error);
  }
  return out;
}

void CheckDrafter(nb::Model& model, const std::vector<std::int32_t>& prompt,
                  const std::vector<float>& prompt_logits) {
  std::string error;
  const std::size_t vocab = model.config().vocab_size;
  // A block verified and committed in part leaves the session where
  // appending that part does.
  auto a = model.CreateSession(4096, &error);
  auto b = model.CreateSession(4096, &error);
  Require(a && b && a->draft() != nullptr, "sessions with drafter: " + error);
  std::vector<float> logits(vocab);
  Require(
      a->Append(prompt, logits, &error) && b->Append(prompt, logits, &error),
      "prefill: " + error);
  const auto greedy = Greedy(*b, prompt_logits, 6);  // b advances by 6
  Require(b->Reset(&error) && b->Append(prompt, logits, &error),
          "restart: " + error);
  std::vector<float> rows(4 * vocab);
  Require(a->Verify(std::span(greedy).first(4), rows, &error),
          "verify: " + error);
  Require(a->Commit(2, &error), "commit: " + error);
  Check(a->Position() == prompt.size() + 2, "commit advances by the kept part");
  Require(b->Append(std::span(greedy).first(2), {}, &error),
          "append: " + error);
  std::vector<float> after_a(vocab), after_b(vocab);
  const std::int32_t next[] = {greedy[2]};
  Require(a->Append(next, after_a, &error) && b->Append(next, after_b, &error),
          "continue: " + error);
  const double kl = Kl(after_b, after_a);
  std::printf("commit 2 of 4 vs append 2: KL %.2e\n", kl);
  Check(after_a == after_b,
        "a partly committed block matches appending its kept part");
  // The verify rows are the next-token rows of the block's prefixes.
  Check(Argmax(std::vector<float>(rows.begin(), rows.begin() + vocab)) ==
            static_cast<std::size_t>(greedy[1]),
        "verify row 0 predicts the next greedy token");

  // A discarded verification leaves no trace.
  auto c = model.CreateSession(4096, &error);
  Require(c && c->Append(prompt, logits, &error), "prefill: " + error);
  Require(c->Verify(std::span(greedy).first(3),
                    std::span(rows).first(3 * vocab), &error) &&
              c->Commit(0, &error),
          "discard: " + error);
  std::vector<float> fresh(vocab);
  Require(c->Append(std::span(greedy).first(2), {}, &error) &&
              c->Append(next, fresh, &error),
          "after discard: " + error);
  Check(fresh == after_b, "a discarded verify is forgotten");

  // Drafts depend only on the restored state.
  nb::SessionSnapshot saved;
  Require(c->Reset(&error) && c->Append(prompt, logits, &error) &&
              c->Save(&saved, &error),
          "save: " + error);
  Check(!saved.draft.empty(), "snapshots carry the drafter context");
  auto d = model.CreateSession(4096, &error);
  Require(d && d->Restore(saved, &error), "restore: " + error);
  nb::DraftLattice from_live, from_restored;
  const auto anchor = static_cast<std::int32_t>(greedy[0]);
  Require(
      c->draft()->Draft(anchor, c->Position(), &from_live, {}, &error) &&
          d->draft()->Draft(anchor, d->Position(), &from_restored, {}, &error),
      "draft: " + error);
  Check(from_live.candidates == from_restored.candidates &&
            from_live.scores == from_restored.scores,
        "a restored session drafts exactly as the original");

  // Greedy draft-and-verify rounds emit plain greedy decoding's tokens: a
  // verified row equals the row of decoding its token alone.
  constexpr std::size_t kTokens = 48;
  Require(b->Reset(&error) && b->Append(prompt, logits, &error),
          "restart: " + error);
  const auto reference = Greedy(*b, prompt_logits, kTokens);
  auto e = model.CreateSession(4096, &error);
  nb::Frontier frontier;
  frontier.logits.resize(vocab);
  Require(e && e->Append(prompt, frontier.logits, &error), "prefill: " + error);
  gufo::sampling::SamplerState sampler;
  std::vector<std::int32_t> spec;
  std::size_t accepted = 0;
  std::size_t rounds = 0;
  while (spec.size() < kTokens) {
    nb::SpeculativeRound round;
    Require(nb::SpeculativeStep(
                *e, sampler, frontier,
                static_cast<std::uint32_t>(kTokens - spec.size()),
                [](std::int32_t) { return false; }, false, &round, &error),
            "round: " + error);
    spec.insert(spec.end(), round.tokens.begin(), round.tokens.end());
    accepted += round.accepted;
    ++rounds;
  }
  std::size_t same = 0;
  while (same < kTokens && spec[same] == reference[same])
    ++same;
  std::printf(
      "draft rounds: %zu for %zu tokens, %zu accepted; same as plain "
      "greedy for %zu tokens\n",
      rounds, spec.size(), accepted, same);
  Check(e->Position() == prompt.size() + spec.size(),
        "rounds run every emitted token");
  Check(same == kTokens, "draft-and-verify follows plain greedy decoding");
  Check(accepted >= kTokens / 4, "the drafter's proposals are accepted");
}

/// `rows` sessions decoded together through DecodeBatch against the same
/// sessions decoded one token at a time: bit-identical rows and the same
/// positions, and with a drafter the same drafts afterwards.
void CheckBatch(nb::Model& model, const std::vector<std::int32_t>& prompt,
                std::size_t rows) {
  const std::size_t kRows = rows;
  constexpr int kSteps = 8;
  std::string error;
  const std::size_t vocab = model.config().vocab_size;
  std::vector<std::unique_ptr<nb::Session>> batched, serial;
  std::vector<std::vector<float>> logits(kRows, std::vector<float>(vocab));
  std::vector<std::vector<float>> reference(kRows, std::vector<float>(vocab));
  for (std::size_t r = 0; r < kRows; ++r) {
    // Different lengths, so the rows sit at different positions.
    const auto part = std::span(prompt).first(prompt.size() - r);
    batched.push_back(model.CreateSession(4096, &error));
    serial.push_back(model.CreateSession(4096, &error));
    Require(batched.back() && serial.back() &&
                batched.back()->Append(part, logits[r], &error) &&
                serial.back()->Append(part, reference[r], &error),
            "batch prefill: " + error);
  }
  std::vector<nb::Session*> sessions;
  for (auto& s : batched)
    sessions.push_back(s.get());
  const gufo::sampling::LogitHeadFilter filter{
      .temperature = 1.0F, .keep = 0, .top_p = 0.95F, .minimum = 1};
  std::vector<const gufo::sampling::LogitHeadFilter*> filters(kRows, nullptr);
  filters[1] = &filter;
  std::vector<gufo::sampling::LogitHead> heads(kRows);
  double worst = 0.0;
  bool same_tokens = true;
  bool exact = true;
  for (int step = 0; step < kSteps; ++step) {
    std::vector<std::int32_t> tokens(kRows);
    for (std::size_t r = 0; r < kRows; ++r)
      tokens[r] = static_cast<std::int32_t>(Argmax(reference[r]));
    std::vector<std::span<float>> outputs;
    for (auto& row : logits)
      outputs.emplace_back(row);
    Require(nb::Session::DecodeBatch(sessions, tokens, outputs, filters, heads,
                                     &error),
            "batch decode: " + error);
    for (std::size_t r = 0; r < kRows; ++r) {
      const std::int32_t one[] = {tokens[r]};
      Require(serial[r]->Append(one, reference[r], &error),
              "serial decode: " + error);
      worst = std::max(worst, Kl(reference[r], logits[r]));
      same_tokens = same_tokens && Argmax(logits[r]) == Argmax(reference[r]);
      exact = exact && logits[r] == reference[r];
    }
    Check(!heads[1].candidates.empty() && heads[0].candidates.empty(),
          "batched rows are summarized with their own filters");
  }
  std::printf("%zu batched vs serial rows: worst KL %.2e, %s\n", kRows, worst,
              exact ? "bit-identical" : "rows differ");
  Check(exact && same_tokens, "batched rows equal serial rows");
  for (std::size_t r = 0; r < kRows; ++r)
    Check(batched[r]->Position() == serial[r]->Position(),
          "batched sessions advance one position per step");
  if (batched[0]->draft() != nullptr) {
    bool drafts = true;
    for (std::size_t r = 0; r < kRows; ++r) {
      nb::DraftLattice a, b;
      const auto anchor = static_cast<std::int32_t>(Argmax(reference[r]));
      Require(batched[r]->draft()->Draft(anchor, batched[r]->Position(), &a, {},
                                         &error) &&
                  serial[r]->draft()->Draft(anchor, serial[r]->Position(), &b,
                                            {}, &error),
              "draft: " + error);
      drafts = drafts && a.candidates == b.candidates;
    }
    Check(drafts, "batched sessions feed their drafter contexts");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path = argc > 1 ? argv[1] : "";
  if (model_path.empty()) {
    if (const char* env = std::getenv("GUFO_NEMOTRON_BOLT_GGUF"))
      model_path = env;
  }
  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    std::fprintf(stderr,
                 "SKIP: set GUFO_NEMOTRON_BOLT_GGUF to the Nemotron 3.5 Bolt "
                 "GGUF\n");
    return kSkip;
  }
  std::string drafter_path = argc > 2 ? argv[2] : "";
  if (drafter_path.empty()) {
    if (const char* env = std::getenv("GUFO_NEMOTRON_BOLT_DRAFTER"))
      drafter_path = env;
  }
  std::string error;
  nb::ModelOptions options;
  options.max_context = 4096;
  auto model = nb::Model::Load(model_path, options, &error);
  Require(model != nullptr, "load: " + error);
  const std::size_t vocab = model->config().vocab_size;

  const auto ids = model->tokenizer().Encode(
      "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nName three prime "
      "numbers and explain why they are prime.<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n",
      true);
  const std::vector<std::int32_t> prompt(ids.begin(), ids.end());
  Require(prompt.size() > 8, "prompt tokenizes");

  // Bulk prefill of the whole prompt.
  auto bulk = model->CreateSession(4096, &error);
  Require(bulk != nullptr, "session: " + error);
  std::vector<float> bulk_logits(vocab);
  Require(bulk->Append(prompt, bulk_logits, &error), "bulk prefill: " + error);

  // The same prompt one token at a time through the decode graph.
  auto step = model->CreateSession(4096, &error);
  Require(step != nullptr, "session: " + error);
  std::vector<float> step_logits(vocab);
  for (const std::int32_t token : prompt) {
    const std::int32_t one[] = {token};
    Require(step->Append(one, step_logits, &error), "decode step: " + error);
  }
  const double kl = Kl(bulk_logits, step_logits);
  std::printf("bulk vs incremental: KL %.2e, top-1 %zu vs %zu\n", kl,
              Argmax(bulk_logits), Argmax(step_logits));
  Check(Argmax(bulk_logits) == Argmax(step_logits) && kl < 2e-2,
        "incremental decode matches the bulk prefill");
  Check(step->Position() == prompt.size(), "position counts every token");

  // Snapshot after the prompt, continue greedily, then replay from the
  // snapshot in a fresh session: the same kernels must give identical bits.
  nb::SessionSnapshot saved;
  Require(bulk->Save(&saved, &error), "save: " + error);
  std::vector<std::int32_t> continuation;
  std::vector<std::vector<float>> first_run;
  std::vector<float> logits = bulk_logits;
  for (int i = 0; i < 6; ++i) {
    const auto next = static_cast<std::int32_t>(Argmax(logits));
    continuation.push_back(next);
    const std::int32_t one[] = {next};
    Require(bulk->Append(one, logits, &error), "continue: " + error);
    first_run.push_back(logits);
  }
  auto restored = model->CreateSession(4096, &error);
  Require(restored != nullptr, "session: " + error);
  Require(restored->Restore(saved, &error), "restore: " + error);
  Check(restored->Position() == prompt.size(), "restored position");
  bool identical = true;
  for (std::size_t i = 0; i < continuation.size(); ++i) {
    const std::int32_t one[] = {continuation[i]};
    Require(restored->Append(one, logits, &error), "replay: " + error);
    identical = identical && logits == first_run[i];
  }
  Check(identical, "restored session reproduces the continuation exactly");

  // Invalidation empties the session before its next append.
  restored->Invalidate();
  Check(restored->Position() == 0, "invalidate resets the position");
  Require(restored->Append(prompt, logits, &error),
          "after invalidate: " + error);
  Check(logits == bulk_logits, "invalidated session restarts from scratch");

  CheckHeads(*model, prompt,
             std::vector<std::int32_t>(prompt.end() - 8, prompt.end()));
  for (std::size_t rows = 2; rows <= 8; ++rows)
    CheckBatch(*model, prompt, rows);

  if (!drafter_path.empty()) {
    Require(model->LoadDrafter(drafter_path, &error), "drafter: " + error);
    CheckDrafter(*model, prompt, bulk_logits);
    CheckBatch(*model, prompt, 3);
  } else {
    std::printf("no drafter: set GUFO_NEMOTRON_BOLT_DRAFTER to check it\n");
  }

  if (failures != 0)
    return 1;
  std::printf("runtime checks passed\n");
  return 0;
}
