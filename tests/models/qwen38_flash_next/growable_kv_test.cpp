// K/V rows that grow with the tokens a session holds. A fresh session maps
// nothing until it writes; its results must equal a session whose rows were
// already backed by an earlier, longer request, across chunk boundaries
// (2 MiB: 3,640.9 K/V rows, 16,384 pooled-key tokens), MTP verification and
// rollback, snapshot restore, cancellation and batched decoding. A small
// budget then has to refuse growth cleanly, leave its peers intact, and reuse
// the rows an idle session no longer holds.
//
// growable_kv_test --model FIRST.gguf --mtp-model MTP.gguf [--batch-only]
#include <algorithm>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;
using gufo::core::SessionMode;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireExact(std::span<const float> expected,
                  std::span<const float> actual, const std::string& message) {
  Require(!expected.empty() && expected.size() == actual.size() &&
              std::memcmp(expected.data(), actual.data(),
                          expected.size_bytes()) == 0,
          message);
}

constexpr std::uint32_t kContext = 20000;
// The first row of the second K/V chunk and the first pooled-key chunk
// boundary in tokens.
constexpr std::uint32_t kKvChunkRow = 3641;
constexpr std::uint32_t kBlockChunkToken = 16384;

std::vector<std::int32_t> Prompt(const qfn::Model& model, std::size_t tokens,
                                 const std::string& salt) {
  const auto pattern = model.Tokenize(
      salt +
      ": the ledger lists each shipment, its weight in kilograms and the "
      "port that received it. Rotterdam 412, Gdansk 88, Valencia 1301. "
      "def total(rows):\n    return sum(r['kg'] for r in rows)\n");
  std::vector<std::int32_t> prompt(tokens);
  for (std::size_t i = 0; i < tokens; ++i)
    prompt[i] = pattern[i % pattern.size()];
  return prompt;
}

std::unique_ptr<qfn::Session> Create(qfn::Model& model,
                                     std::uint32_t context = kContext) {
  std::string error;
  auto session =
      model.CreateSession(SessionMode::kSpeculative, context, &error);
  Require(session != nullptr, error);
  return session;
}

/// A session whose rows an earlier request already backed to `tokens`.
std::unique_ptr<qfn::Session> Pregrown(qfn::Model& model,
                                       std::uint32_t tokens) {
  auto session = Create(model);
  std::string error;
  Require(session->Sync(Prompt(model, tokens, "warm"), &error), error);
  session->Reset();
  return session;
}

/// Syncs both sessions through the same prefixes (identical pass
/// boundaries) and compares the logits after each.
void SyncBoth(qfn::Session& fresh, qfn::Session& reference,
              std::span<const std::int32_t> prompt,
              std::span<const std::uint32_t> ends, const char* what) {
  std::string error;
  for (const auto end : ends) {
    Require(fresh.Sync(prompt.first(end), &error), error);
    Require(reference.Sync(prompt.first(end), &error), error);
    RequireExact(reference.Logits(), fresh.Logits(),
                 std::string(what) + ": prefill logits differ at " +
                     std::to_string(end));
  }
}

/// Greedy MTP decoding of both sessions, compared token by token.
void DecodeBoth(qfn::Session& fresh, qfn::Session& reference, std::size_t steps,
                const char* what) {
  std::string error;
  const auto history = [](const qfn::Session& s) {
    return std::vector<sampling::TokenId>(s.Tokens().begin(), s.Tokens().end());
  };
  sampling::SamplerState a({.seed = 7}, history(fresh));
  sampling::SamplerState b({.seed = 7}, history(reference));
  for (std::size_t i = 0; i < steps; ++i) {
    qfn::Session::DecodeResult x, y;
    Require(fresh.DecodeStep(8, a, &x, &error, false), error);
    Require(reference.DecodeStep(8, b, &y, &error, false), error);
    Require(x.tokens == y.tokens, std::string(what) + ": tokens differ");
    RequireExact(reference.Logits(), fresh.Logits(),
                 std::string(what) + ": decode logits differ");
  }
  Require(fresh.Statistics().drafted == reference.Statistics().drafted &&
              fresh.Statistics().accepted == reference.Statistics().accepted,
          std::string(what) + ": drafts differ");
}

std::vector<std::uint8_t> Bytes(const qfn::Session& session) {
  std::string error;
  const auto snapshot =
      session.SaveSnapshot(&error, qfn::Session::SnapshotMode::kMaterialized);
  Require(snapshot != nullptr, error);
  return {snapshot->bytes().begin(), snapshot->bytes().end()};
}

void CheckGrowth(qfn::Model& model) {
  auto reference = Pregrown(model, 18000);
  auto fresh = Create(model);
  const auto empty = fresh->AllocatedBytes();
  Require(reference->AllocatedBytes() > empty + (200ULL << 20),
          "a fresh session already backs its K/V rows");
  const auto prompt = Prompt(model, 17000, "growth");
  // Passes ending just before, on and after the first K/V chunk boundary,
  // then across the first pooled-key chunk boundary.
  const std::uint32_t ends[] = {
      kKvChunkRow - 3,       kKvChunkRow + 1,        9000,
      kBlockChunkToken - 20, kBlockChunkToken + 300, 17000};
  SyncBoth(*fresh, *reference, prompt, ends, "growth");
  DecodeBoth(*fresh, *reference, 12, "growth");
  Require(Bytes(*fresh) == Bytes(*reference), "grown state bytes differ");
  Require(fresh->AllocatedBytes() > empty + (200ULL << 20),
          "growth did not back the written rows");
  std::cout << "growth across K/V and pooled-key chunks: logits, tokens and "
               "state exact; backed "
            << (fresh->AllocatedBytes() - empty) / (1 << 20) << " MiB\n";
}

void CheckRollbackAtBoundary(qfn::Model& model) {
  auto reference = Pregrown(model, 8000);
  auto fresh = Create(model);
  // Verification batches of up to eight rows straddle row 3,641; rejected
  // drafts are cut back across it.
  const auto prompt = Prompt(model, kKvChunkRow - 5, "rollback");
  const std::uint32_t ends[] = {kKvChunkRow - 5};
  SyncBoth(*fresh, *reference, prompt, ends, "rollback");
  DecodeBoth(*fresh, *reference, 16, "rollback");
  Require(fresh->Position() > kKvChunkRow, "decoding did not cross the chunk");
  Require(Bytes(*fresh) == Bytes(*reference), "rollback state bytes differ");
  std::cout << "MTP verification and rollback across a chunk: exact, drafted "
            << fresh->Statistics().drafted << " accepted "
            << fresh->Statistics().accepted << '\n';
}

void CheckRestore(qfn::Model& model) {
  std::string error;
  auto source = Create(model);
  Require(source->Sync(Prompt(model, 9000, "restore"), &error), error);
  const auto snapshot =
      source->SaveSnapshot(&error, qfn::Session::SnapshotMode::kMaterialized);
  Require(snapshot != nullptr, error);
  auto fresh = Create(model);
  const auto empty = fresh->AllocatedBytes();
  Require(fresh->RestoreSnapshot(*snapshot, &error), error);
  Require(fresh->AllocatedBytes() > empty, "restore did not back its rows");
  RequireExact(source->Logits(), fresh->Logits(), "restored logits differ");
  const auto restored = Bytes(*fresh);
  Require(restored == Bytes(*source), "restored state bytes differ");
  // A longer session restores the same snapshot over rows it already backs.
  auto longer = Pregrown(model, 15000);
  Require(longer->RestoreSnapshot(*snapshot, &error), error);
  Require(Bytes(*longer) == restored, "restore over backed rows differs");
  DecodeBoth(*fresh, *source, 8, "restore");
  std::cout << "restore into a session with no backed rows: exact\n";
}

void CheckCancellation(qfn::Model& model) {
  std::string error;
  const auto prompt = Prompt(model, 17500, "cancel");
  auto reference = Create(model);
  int checks = 0;
  reference->SetCancellationCheck([&] { return ++checks < 0; });
  Require(reference->Sync(prompt, &error), error);
  reference->SetCancellationCheck({});
  // The second pass (rows 16,384 on, past the pooled-key chunk) holds about
  // the last half of the checks: cancel in it, after it has grown.
  const int total = checks;
  auto fresh = Create(model);
  checks = 0;
  fresh->SetCancellationCheck([&] { return ++checks > total * 3 / 4; });
  Require(!fresh->Sync(prompt, &error), "the long prefill was not cancelled");
  fresh->SetCancellationCheck({});
  fresh->Reset();
  Require(fresh->Sync(prompt, &error), error);
  RequireExact(reference->Logits(), fresh->Logits(),
               "prefill after a cancelled growth differs");
  DecodeBoth(*fresh, *reference, 4, "cancellation");
  std::cout << "cancellation mid-growth, then a full prefill: exact\n";
}

void CheckBatch(qfn::Model& model, bool pregrown) {
  std::string error;
  // Lengths around both chunk boundaries and short ones; batched decoding
  // grows each member independently.
  const std::uint32_t lengths[] = {
      100, kKvChunkRow - 4, 3700, 5000, 7279, 300, 4096, 9000};
  std::vector<std::unique_ptr<qfn::Session>> serial, batched;
  std::vector<sampling::SamplerState> serial_samplers, batch_samplers;
  for (std::size_t i = 0; i < 8; ++i) {
    const auto prompt = Prompt(model, lengths[i], "batch " + std::to_string(i));
    serial.push_back(pregrown ? Pregrown(model, 10000) : Create(model));
    batched.push_back(pregrown ? Pregrown(model, 10000) : Create(model));
    Require(serial.back()->Sync(prompt, &error), error);
    Require(batched.back()->Sync(prompt, &error), error);
    const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
    serial_samplers.emplace_back(sampling::SamplingConfig{.seed = 11}, history);
    batch_samplers.emplace_back(sampling::SamplingConfig{.seed = 11}, history);
  }
  // Draft lengths follow each cohort's policy, so a batched cycle and a
  // lone cycle may commit different counts. Bring each serial twin to the
  // batched position, then compare the committed tokens and logits.
  for (unsigned cycle = 0; cycle < 6; ++cycle) {
    std::vector<qfn::Session::DecodeResult> actual(8);
    std::vector<qfn::Session::DecodeRequest> requests;
    for (std::size_t i = 0; i < 8; ++i)
      requests.push_back(
          {batched[i].get(), 8, &batch_samplers[i], &actual[i], false});
    Require(qfn::Session::DecodeBatch(requests, &error), error);
    for (std::size_t i = 0; i < 8; ++i) {
      while (serial[i]->Position() < batched[i]->Position()) {
        qfn::Session::DecodeResult step;
        Require(serial[i]->DecodeStep(
                    batched[i]->Position() - serial[i]->Position(),
                    serial_samplers[i], &step, &error, false, 8),
                error);
      }
      Require(std::ranges::equal(serial[i]->Tokens(), batched[i]->Tokens()),
              std::string(pregrown ? "pregrown" : "growing") +
                  " batch changed tokens of member " + std::to_string(i) +
                  " in cycle " + std::to_string(cycle) + " at position " +
                  std::to_string(batched[i]->Position()));
      RequireExact(
          serial[i]->Logits(), batched[i]->Logits(),
          "batched growth changed logits of member " + std::to_string(i));
    }
  }
  Require(batched[1]->Position() > kKvChunkRow && batched[4]->Position() > 7282,
          "batched members did not cross their chunk boundaries");
  std::cout << "eight concurrent sessions of mixed lengths: tokens and logits "
               "exact\n";
}

void CheckBudget(const char* path, const char* mtp_path,
                 std::span<const float> expected, std::uint32_t tokens,
                 std::size_t per_session, std::size_t full_claim) {
  std::string error;
  // Room for two sessions of `tokens` tokens and not a third.
  auto model = qfn::Model::Load(path,
                                {.max_context = kContext,
                                 .mtp_model_path = mtp_path,
                                 .decode_concurrency = 4,
                                 .kv_cache_bytes = per_session * 5 / 2},
                                &error);
  Require(model != nullptr, error);
  // The startup claim charges a share of the budget, not the full context.
  const auto claim = model->SessionBytes(SessionMode::kSpeculative, kContext);
  Require(claim < full_claim && claim + per_session > full_claim / 4,
          "a budgeted session claims " + std::to_string(claim) + " of " +
              std::to_string(full_claim) + " bytes");
  const auto prompt = Prompt(*model, tokens, "budget");
  auto a = Create(*model);
  auto b = Create(*model);
  auto c = Create(*model);
  Require(a->Sync(prompt, &error), error);
  RequireExact(expected, a->Logits(), "budgeted prefill differs");
  Require(b->Sync(prompt, &error), error);
  const auto borrowed = b->SaveSnapshot(&error);
  Require(borrowed != nullptr, error);
  const auto b_bytes = Bytes(*b);
  Require(!c->Sync(prompt, &error), "growth past the budget succeeded");
  Require(error.find("K/V memory exhausted") != std::string::npos,
          "unexpected budget error: " + error);
  // The refusal touched nothing else: a keeps decoding as before.
  sampling::SamplerState sampler(
      {.seed = 3},
      std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
  qfn::Session::DecodeResult step;
  Require(a->DecodeStep(8, sampler, &step, &error, false), error);
  // b goes idle: c reuses the rows b no longer holds, and b's borrowed
  // checkpoint keeps its bytes.
  b->Reset();
  Require(c->Sync(prompt, &error), error);
  RequireExact(expected, c->Logits(), "prefill in released rows differs");
  Require(std::equal(borrowed->bytes().begin(), borrowed->bytes().end(),
                     b_bytes.begin(), b_bytes.end()),
          "releasing idle rows changed a borrowed checkpoint");
  std::cout << "budget " << (per_session * 5 / 2) / (1 << 20)
            << " MiB: third session refused cleanly, peers intact, idle rows "
               "reused, borrowed checkpoint intact\n";
}

}  // namespace

int main(int argc, char** argv) {
  const bool batch_only =
      argc == 6 && std::string_view(argv[5]) == "--batch-only";
  if ((argc != 5 && !batch_only) || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr << "Usage: growable_kv_test --model FIRST.gguf --mtp-model "
                 "MTP.gguf [--batch-only]\n";
    return 77;
  }
  try {
    std::string error;
    std::vector<float> budget_logits;
    std::size_t budget_session = 0;
    std::size_t full_claim = 0;
    constexpr std::uint32_t kBudgetTokens = 6000;
    {
      auto model = qfn::Model::Load(argv[2],
                                    {.max_context = kContext,
                                     .mtp_model_path = argv[4],
                                     .decode_concurrency = 8},
                                    &error);
      Require(model != nullptr, error);
      if (!batch_only) {
        CheckGrowth(*model);
        CheckRollbackAtBoundary(*model);
        CheckRestore(*model);
        CheckCancellation(*model);
      }
      CheckBatch(*model, true);
      CheckBatch(*model, false);
      auto reference = Create(*model);
      const auto empty = reference->AllocatedBytes();
      Require(reference->Sync(Prompt(*model, kBudgetTokens, "budget"), &error),
              error);
      budget_logits.assign(reference->Logits().begin(),
                           reference->Logits().end());
      budget_session = reference->AllocatedBytes() - empty;
      full_claim = model->SessionBytes(SessionMode::kSpeculative, kContext);
    }
    CheckBudget(argv[2], argv[4], budget_logits, kBudgetTokens, budget_session,
                full_claim);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
