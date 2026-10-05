// Session snapshot round trips: a restored session must continue exactly
// like the session it was captured from, in memory and through the
// persistent byte form, at a prompt boundary and mid-decode.
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Decoded {
  std::vector<std::int32_t> tokens;
  qfn::Session::SpeculativeStats stats;
};

Decoded Decode(qfn::Session& session, std::size_t count,
               sampling::SamplerState& sampler, std::size_t budget) {
  std::string error;
  const auto before = session.Statistics();
  Decoded out;
  while (out.tokens.size() < count) {
    qfn::Session::DecodeResult step;
    Require(session.DecodeStep(std::min(budget, count - out.tokens.size()),
                               sampler, &step, &error, false),
            error);
    Require(!step.tokens.empty(), "empty decode step");
    out.tokens.insert(out.tokens.end(), step.tokens.begin(), step.tokens.end());
  }
  const auto after = session.Statistics();
  out.stats = {after.cycles - before.cycles, after.drafted - before.drafted,
               after.accepted - before.accepted};
  return out;
}

/// Decodes `count` tokens past EOS with a fresh seeded sampler.
Decoded Decode(qfn::Session& session, std::size_t count,
               const sampling::SamplingConfig& config, std::size_t budget) {
  const auto history = session.Tokens();
  const std::vector<sampling::TokenId> initial(history.begin(), history.end());
  sampling::SamplerState sampler(config, initial);
  return Decode(session, count, sampler, budget);
}

void RequireSame(const Decoded& expected, const Decoded& actual,
                 const std::string& what) {
  Require(expected.tokens == actual.tokens, what + ": tokens differ");
  Require(expected.stats.drafted == actual.stats.drafted &&
              expected.stats.accepted == actual.stats.accepted,
          what + ": draft acceptance differs");
}

double Millis(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

std::vector<std::uint8_t> Payload(const qfn::SessionSnapshot& snapshot) {
  std::vector<std::uint8_t> result(snapshot.SizeBytes());
  Require(snapshot.CopyTo(result), "materialize snapshot");
  return result;
}

std::vector<std::uint8_t> Payload(qfn::Session& session) {
  std::string error;
  auto snapshot = session.SaveSnapshot(&error);
  Require(snapshot != nullptr, error);
  return Payload(*snapshot);
}

void CheckPrefillCheckpoint(const std::shared_ptr<qfn::Model>& model,
                            std::span<const std::int32_t> prompt,
                            gufo::core::SessionMode mode, std::size_t warm = 0,
                            std::uint32_t suffix = 7) {
  std::string error;
  auto source = model->CreateSession(mode, 8192, &error);
  auto whole = model->CreateSession(mode, 8192, &error);
  auto split = model->CreateSession(mode, 8192, &error);
  Require(source && whole && split, error);
  if (warm)
    Require(source->Sync(prompt.first(warm), &error) &&
                whole->Sync(prompt.first(warm), &error) &&
                split->Sync(prompt.first(warm), &error),
            error);
  const auto boundary = static_cast<std::uint32_t>(prompt.size() - suffix);
  const auto claim = source->PrefillCheckpointBytes(boundary);
  std::unique_ptr<qfn::SessionSnapshot> checkpoint;
  double capture_ms = -1;
  Require(
      source->SyncThrough(prompt, boundary, &checkpoint, &error, &capture_ms),
      error);
  Require(std::isfinite(capture_ms) && capture_ms > 0,
          "in-pass checkpoint capture time was not reported");
  Require(checkpoint != nullptr, "in-pass checkpoint was not returned");
  Require(checkpoint->SizeBytes() == claim,
          "in-pass checkpoint differs from its admission claim");
  Require(whole->Sync(prompt.first(boundary), &error) &&
              whole->Sync(prompt, &error),
          error);
  Require(source->Logits().size() == whole->Logits().size() &&
              std::memcmp(source->Logits().data(), whole->Logits().data(),
                          source->Logits().size_bytes()) == 0,
          "in-pass checkpoint changed split-pass logits");
  const auto continued =
      Decode(*source, 8,
             {.temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = 73}, 8);
  RequireSame(
      continued,
      Decode(*whole, 8,
             {.temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = 73}, 8),
      "in-pass sampled continuation");
  Require(std::memcmp(source->Logits().data(), whole->Logits().data(),
                      source->Logits().size_bytes()) == 0,
          "in-pass sampled continuation changed logits");
  Require(split->Sync(prompt.first(boundary), &error), error);
  const auto expected = Payload(*split);
  source->Reset();
  const auto actual = Payload(*checkpoint);
  Require(source->RestoreSnapshot(*checkpoint, &error), error);
  Require(Payload(*source) == actual,
          "in-pass checkpoint did not survive source reset");
  Require(actual == expected,
          "in-pass boundary differs from a pass ending there");
  auto raw = model->CreateSession(mode, 8192, &error);
  Require(raw && raw->RestoreSnapshot(actual, &error), error);
  Require(source->Sync(prompt, &error), error);
  Require(raw->Sync(prompt, &error), error);
  Require(Payload(*source) == Payload(*raw),
          "borrowed and serialized boundaries resumed differently");
  Require(split->Sync(prompt, &error), error);
  Require(Payload(*source) == Payload(*split),
          "restored boundary did not resume exactly");
  std::cout << "in-pass checkpoint mode=" << static_cast<int>(mode)
            << " tokens=" << prompt.size() << " boundary=" << boundary
            << " warm=" << warm << " restore exact\n";
}

void CheckBorrowedSnapshots(const std::shared_ptr<qfn::Model>& model,
                            std::span<const std::int32_t> prompt,
                            gufo::core::SessionMode mode) {
  std::string error;
  auto source = model->CreateSession(mode, 8192, &error);
  Require(source != nullptr, error);
  Require(source->Sync(prompt, &error), error);
  const auto frontier_bytes = Payload(*source);
  auto eager =
      source->SaveSnapshot(&error, qfn::Session::SnapshotMode::kMaterialized);
  Require(eager != nullptr, error);
  Require(Payload(*eager) == frontier_bytes,
          "export capture differs from borrowed checkpoint bytes");
  auto frontier = source->SaveSnapshot(&error);
  Require(frontier != nullptr, error);
  source->Reset();
  Require(source->RestoreSnapshot(*frontier, &error), error);
  Require(Payload(*source) == frontier_bytes,
          "rewind after logical reset changed checkpoint rows");

  std::vector<std::int32_t> extended(prompt.begin(), prompt.end());
  extended.insert(extended.end(), prompt.begin(), prompt.begin() + 9);
  Require(source->Sync(extended, &error), error);
  const auto boundary_bytes = Payload(*source);
  auto boundary = source->SaveSnapshot(&error);
  Require(boundary != nullptr, error);
  extended.insert(extended.end(), prompt.begin() + 9, prompt.begin() + 16);
  Require(source->Sync(extended, &error), error);

  // Rewind without reading either borrowed payload. Preserve B's suffix,
  // overwrite it with a different branch, then restore B over that branch.
  Require(source->RestoreSnapshot(*frontier, &error), error);
  std::vector<std::int32_t> branch(prompt.begin(), prompt.end());
  branch.insert(branch.end(), prompt.begin() + 20, prompt.begin() + 36);
  Require(source->Sync(branch, &error), error);
  auto other_branch = source->SaveSnapshot(&error);
  Require(other_branch != nullptr, error);
  const auto other_bytes = Payload(*source);
  auto fork = model->CreateSession(mode, 8192, &error);
  Require(fork != nullptr, error);
  Require(fork->RestoreSnapshot(*boundary, &error), error);
  Require(Payload(*fork) == boundary_bytes,
          "foreign fork lost its live prefix or detached tail");
  auto fork_boundary = fork->SaveSnapshot(&error);
  Require(fork_boundary != nullptr, error);
  // A fresh checkpoint still has no host KV. Fork from it while the source
  // continues above those rows; neither path may materialize its host prefix.
  auto live_branch = source->SaveSnapshot(&error);
  Require(live_branch != nullptr, error);
  auto fork_reader = std::async(std::launch::async, [&] {
    std::string fork_error;
    Require(fork->RestoreSnapshot(*live_branch, &fork_error), fork_error);
    return Payload(*fork);
  });
  auto appended = branch;
  appended.insert(appended.end(), prompt.begin() + 36, prompt.begin() + 40);
  Require(source->Sync(appended, &error), error);
  Require(fork_reader.get() == other_bytes,
          "concurrent device fork changed committed checkpoint rows");
  // Retained shared rows still belong to the target session: resetting and
  // overwriting that prefix must protect them even without another capture.
  fork->Reset();
  Require(fork->Sync(prompt.subspan(20, 32), &error), error);
  Require(Payload(*fork_boundary) == boundary_bytes,
          "reset after sibling fork overwrote retained checkpoint rows");
  Require(fork->RestoreSnapshot(*fork_boundary, &error), error);
  Require(Payload(*fork) == boundary_bytes,
          "sibling fork checkpoint did not restore its protected suffix");
  Require(source->RestoreSnapshot(*boundary, &error), error);
  Require(Payload(*source) == boundary_bytes,
          "borrowed restore differs from complete snapshot");
  Require(source->RestoreSnapshot(*other_branch, &error), error);
  Require(Payload(*source) == other_bytes,
          "second branch lost its overwritten rows");

  // Disk/fork readers can copy committed rows while this session appends.
  auto reader =
      std::async(std::launch::async, [&] { return Payload(*frontier); });
  branch.insert(branch.end(), prompt.begin() + 36, prompt.begin() + 52);
  Require(source->Sync(branch, &error), error);
  Require(reader.get() == frontier_bytes,
          "concurrent append changed borrowed snapshot");
  source->Reset();
  Require(Payload(*boundary) == boundary_bytes,
          "reset discarded borrowed checkpoint rows");
  Require(Payload(*other_branch) == other_bytes,
          "reset discarded a detached branch");

  Require(source->Sync(prompt, &error), error);
  auto overwritten = source->SaveSnapshot(&error);
  Require(overwritten != nullptr, error);
  source->Reset();
  Require(source->Sync(prompt.subspan(20, 32), &error), error);
  Require(Payload(*overwritten) == frontier_bytes,
          "first write after reset overwrote a borrowed checkpoint");
  Require(source->RestoreSnapshot(*overwritten, &error), error);
  auto at_decode = source->SaveSnapshot(&error);
  Require(at_decode != nullptr, error);
  const auto decode_bytes = Payload(*source);
  auto decode_reader =
      std::async(std::launch::async, [&] { return Payload(*at_decode); });
  (void)Decode(*source, 16,
               {.temperature = 0.8F, .top_k = 40, .top_p = 0.9F, .seed = 7}, 8);
  Require(decode_reader.get() == decode_bytes,
          "concurrent graph capture/decode changed checkpoint bytes");

  auto survivor = source->SaveSnapshot(&error);
  Require(survivor != nullptr, error);
  const auto survivor_bytes = Payload(*source);
  auto final_reader =
      std::async(std::launch::async, [&] { return Payload(*survivor); });
  source.reset();
  Require(final_reader.get() == survivor_bytes,
          "concurrent source destruction changed borrowed rows");
  Require(Payload(*survivor) == survivor_bytes,
          "source destruction discarded borrowed rows");
  Require(fork->RestoreSnapshot(*survivor, &error), error);
  Require(Payload(*fork) == survivor_bytes,
          "foreign restore after source destruction lost checkpoint rows");
  std::cout << "borrowed checkpoints: rewind, branches, reader, reset, "
               "destruction pass\n";
}

void CheckCheckpointRetention(const std::shared_ptr<qfn::Model>& model,
                              std::span<const std::int32_t> prompt,
                              gufo::core::SessionMode mode) {
  std::string error;
  auto source = model->CreateSession(mode, 8192, &error);
  Require(source && source->Sync(prompt.first(32), &error), error);
  const auto expected = Payload(*source);
  auto small = source->SaveSnapshot(&error);
  Require(small != nullptr, error);
  Require(source->OwnsSnapshot(*small), "checkpoint lost its source identity");
  const auto before = small->DeviceBytes();
  Require(source->Sync(prompt, &error), error);
  auto large = source->SaveSnapshot(&error);
  Require(large != nullptr, error);
  source->Reset();
  Require(source->Sync(prompt.first(16), &error), error);
  Require(large->DeviceBytes() > before + (80ULL << 20),
          "deep checkpoint did not preserve its overwritten rows");
  large.reset();
  // Count owned backing blocks rather than HIP's global heap cache. A short
  // prefix may retain bounded packing slack, but never the deep-only rows.
  const auto after = small->DeviceBytes();
  Require(after <= before + (8ULL << 20),
          "short checkpoint retained the discarded deep allocation");
  Require(source->RestoreSnapshot(*small, &error), error);
  Require(Payload(*source) == expected,
          "discarding a deep checkpoint changed its shared short prefix");
  std::cout << "checkpoint retention mode=" << static_cast<int>(mode)
            << " additional_bytes=" << after - before
            << " short prefix exact\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr
        << "Usage: snapshot_test --model FIRST.gguf --mtp-model MTP.gguf\n";
    return 77;
  }
  try {
    std::string error;
    constexpr std::uint32_t kContext = 8192;
    auto model = qfn::Model::Load(argv[2],
                                  {.max_context = kContext,
                                   .mtp_model_path = argv[4],
                                   .max_draft_tokens = 7},
                                  &error);
    Require(model != nullptr, error);
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty prompt pattern");
    // Partial prefill batch and three unpooled raw indexer rows at the end
    // of the ring. Continuing (and speculative rollback) crosses its wrap.
    std::vector<std::int32_t> prompt(4095);
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = pattern[i % pattern.size()];
    CheckPrefillCheckpoint(model, prompt,
                           gufo::core::SessionMode::kAutoregressive);
    CheckPrefillCheckpoint(model, prompt,
                           gufo::core::SessionMode::kSpeculative);
    std::vector<std::int32_t> overflow_prompt(prompt);
    overflow_prompt.insert(overflow_prompt.end(), prompt.begin(),
                           prompt.begin() + 2051);
    std::vector<std::int32_t> warm_prompt(prompt);
    warm_prompt.insert(warm_prompt.end(), prompt.begin(), prompt.begin() + 33);
    for (const auto mode : {gufo::core::SessionMode::kAutoregressive,
                            gufo::core::SessionMode::kSpeculative}) {
      CheckPrefillCheckpoint(model, warm_prompt, mode, prompt.size());
      CheckPrefillCheckpoint(model, overflow_prompt, mode, prompt.size());
      CheckPrefillCheckpoint(
          model, std::span<const std::int32_t>(prompt).first(2040), mode);
      CheckPrefillCheckpoint(model, prompt, mode, 0, 0);
      CheckPrefillCheckpoint(
          model, std::span<const std::int32_t>(warm_prompt).first(2055), mode);
    }
    CheckBorrowedSnapshots(model, prompt,
                           gufo::core::SessionMode::kAutoregressive);
    CheckBorrowedSnapshots(model, prompt,
                           gufo::core::SessionMode::kSpeculative);
    CheckCheckpointRetention(model, prompt,
                             gufo::core::SessionMode::kAutoregressive);
    CheckCheckpointRetention(model, prompt,
                             gufo::core::SessionMode::kSpeculative);
    const sampling::SamplingConfig config{
        .temperature = 0.8F, .top_k = 40, .top_p = 0.9F, .seed = 7};
    constexpr std::size_t kTokens = 48;

    auto origin = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        kContext, &error);
    Require(origin != nullptr, error);
    Require(origin->Sync(prompt, &error), error);
    const std::vector<float> prompt_logits(origin->Logits().begin(),
                                           origin->Logits().end());

    auto start = std::chrono::steady_clock::now();
    auto at_prompt = origin->SaveSnapshot(&error);
    Require(at_prompt != nullptr, error);
    const double save_ms = Millis(start);
    Require(at_prompt->SizeBytes() == origin->SnapshotBytes(),
            "snapshot size differs from the estimate");
    std::cout << "snapshot tokens=" << prompt.size()
              << " bytes=" << at_prompt->SizeBytes() << " save_ms=" << save_ms
              << "\n";

    const Decoded expected = Decode(*origin, kTokens, config, 8);
    Require(expected.stats.drafted > 0, "MTP did not draft");

    // Restore into a fresh session.
    auto restored = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        kContext, &error);
    Require(restored != nullptr, error);
    start = std::chrono::steady_clock::now();
    Require(restored->RestoreSnapshot(*at_prompt, &error), error);
    std::cout << "restore_ms=" << Millis(start) << "\n";
    Require(std::equal(prompt.begin(), prompt.end(), restored->Tokens().begin(),
                       restored->Tokens().end()),
            "restored tokens differ");
    Require(restored->Position() == prompt.size(), "restored position");
    Require(std::memcmp(prompt_logits.data(), restored->Logits().data(),
                        prompt_logits.size() * sizeof(float)) == 0,
            "restored logits differ");
    RequireSame(expected, Decode(*restored, kTokens, config, 8),
                "fresh session restore");

    // The persistent byte form restores the same way.
    std::vector<std::uint8_t> bytes(at_prompt->SizeBytes());
    Require(at_prompt->CopyTo(bytes), "snapshot copy");
    auto persisted = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        kContext, &error);
    Require(persisted != nullptr, error);
    Require(persisted->RestoreSnapshot(bytes, &error), error);
    RequireSame(expected, Decode(*persisted, kTokens, config, 8),
                "persistent restore");

    // Restoring over a session that already decoded discards its state.
    Require(origin->RestoreSnapshot(*at_prompt, &error), error);
    RequireSame(expected, Decode(*origin, kTokens, config, 8),
                "restore over a used session");

    // Single-token decoding from a restored session also matches.
    Require(origin->RestoreSnapshot(*at_prompt, &error), error);
    const Decoded single = Decode(*origin, 16, config, 1);
    Require(restored->RestoreSnapshot(*at_prompt, &error), error);
    Require(single.tokens == Decode(*restored, 16, config, 1).tokens,
            "single-token decode differs after restore");

    // A mid-decode snapshot carries the draft block's caught-up state and
    // the partial batch of kept trunk rows.
    Require(origin->RestoreSnapshot(*at_prompt, &error), error);
    const Decoded first_half = Decode(*origin, 16, config, 8);
    auto mid = origin->SaveSnapshot(&error);
    Require(mid != nullptr, error);
    const Decoded second_half = Decode(*origin, kTokens, config, 8);
    Require(restored->RestoreSnapshot(*mid, &error), error);
    Require(restored->Position() == prompt.size() + first_half.tokens.size(),
            "mid-decode position");
    // Controller state is part of a stochastic continuation's replay.
    RequireSame(second_half, Decode(*restored, kTokens, config, 8),
                "mid-decode restore");

    // Capture a rejected proposal before its residual has been evaluated.
    // The context snapshot and the request-owned sampler must replay together.
    auto pending = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        kContext, &error);
    const auto short_prompt = std::span(prompt).first(16);
    Require(pending && pending->Sync(short_prompt, &error), error);
    const std::vector<sampling::TokenId> short_history(short_prompt.begin(),
                                                       short_prompt.end());
    sampling::SamplerState sampler({.temperature = 2.0F, .seed = 73},
                                   short_history);
    bool rejected = false;
    for (unsigned attempt = 0; attempt < 32 && !rejected; ++attempt) {
      qfn::Session::DecodeResult step;
      Require(pending->DecodeStep(2, sampler, &step, &error, false), error);
      rejected = step.tokens.size() == 1;
    }
    Require(rejected, "snapshot test did not exercise a deferred residual");
    auto residual_snapshot = pending->SaveSnapshot(&error);
    Require(residual_snapshot != nullptr, error);
    auto replay_sampler = sampler;
    const auto residual_expected = Decode(*pending, 16, sampler, 8);
    Require(restored->RestoreSnapshot(*residual_snapshot, &error), error);
    RequireSame(residual_expected, Decode(*restored, 16, replay_sampler, 8),
                "deferred residual restore");
    Require(sampler.rng_state() == replay_sampler.rng_state(),
            "deferred residual restore changed RNG");

    // Extending the restored context keeps the prefix.
    Require(restored->RestoreSnapshot(*at_prompt, &error), error);
    std::vector<std::int32_t> extended = prompt;
    extended.insert(extended.end(), expected.tokens.begin(),
                    expected.tokens.end());
    Require(restored->Sync(extended, &error), error);
    Require(origin->RestoreSnapshot(*at_prompt, &error), error);
    Require(origin->Sync(extended, &error), error);
    Require(std::memcmp(origin->Logits().data(), restored->Logits().data(),
                        prompt_logits.size() * sizeof(float)) == 0,
            "extension after restore differs");

    // Cached state from older prefill arithmetic must be rebuilt.
    auto incompatible = bytes;
    const std::uint32_t old_version = qfn::Session::kSnapshotPayloadVersion - 1;
    std::memcpy(incompatible.data() + 8, &old_version, sizeof(old_version));
    Require(!restored->RestoreSnapshot(incompatible, &error),
            "incompatible snapshot version accepted");
    Require(restored->Position() == extended.size(),
            "incompatible snapshot disturbed the session");

    // Rejections: truncated payload, and a context too small to hold it.
    Require(!restored->RestoreSnapshot(
                std::span<const std::uint8_t>(bytes).first(bytes.size() / 2),
                &error),
            "truncated payload accepted");
    Require(restored->Position() == extended.size(),
            "rejected payload disturbed the session");
    auto small = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        2048, &error);
    Require(small != nullptr, error);
    Require(!small->RestoreSnapshot(*at_prompt, &error),
            "oversized snapshot accepted");
    std::cout << "snapshot round trips exact=1\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
