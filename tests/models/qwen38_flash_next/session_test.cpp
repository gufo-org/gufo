#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/models/qwen38_flash_next/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"
#include "tests/models/qwen38_flash_next/mtp_replay.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireExact(std::span<const float> expected,
                  std::span<const float> actual, const std::string& message) {
  Require(expected.size() == actual.size() &&
              std::memcmp(expected.data(), actual.data(),
                          expected.size_bytes()) == 0,
          message);
}

void CheckSnapshotDuringGraphCapture(const std::shared_ptr<qfn::Model>& model,
                                     bool warm_storage,
                                     bool protect_rows = false) {
  std::string error;
  const auto mode = gufo::core::SessionMode::kAutoregressive;
  auto decoding = model->CreateSession(mode, 128, &error);
  auto reference = model->CreateSession(mode, 128, &error);
  auto frozen = model->CreateSession(mode, 128, &error);
  Require(decoding && reference && frozen, error);
  const auto prompt = model->Tokenize("Continue: red, blue, red, blue,");
  for (auto* session : {decoding.get(), reference.get(), frozen.get()})
    Require(session->Sync(prompt, &error), error);
  auto expected_snapshot =
      (warm_storage ? frozen : reference)->SaveSnapshot(&error);
  Require(expected_snapshot != nullptr, error);
  std::unique_ptr<qfn::SessionSnapshot> prefix_checkpoint;
  if (protect_rows) {
    prefix_checkpoint = frozen->SaveSnapshot(&error);
    Require(prefix_checkpoint != nullptr, error);
    auto extended = prompt;
    extended.push_back(prompt.front());
    Require(frozen->Sync(extended, &error), error);
    expected_snapshot =
        frozen->SaveSnapshot(&error, qfn::Session::SnapshotMode::kMaterialized);
    Require(expected_snapshot != nullptr, error);
  }

  // Warm the one-token shape; its next execution records the decode graph.
  Require(decoding->Evaluate(prompt.front(), &error) &&
              reference->Evaluate(prompt.front(), &error),
          error);
  std::unique_ptr<qfn::SessionSnapshot> concurrent_snapshot;
  std::string snapshot_error;
  unsigned checkpoints = 0;
  decoding->SetCancellationCheck([&] {
    // Forward checks once before capture and again at its first layer.
    if (++checkpoints == 2) {
      try {
        concurrent_snapshot = std::async(std::launch::async, [&] {
                                return frozen->SaveSnapshot(&snapshot_error);
                              }).get();
      } catch (const std::exception& e) {
        snapshot_error = e.what();
        throw;
      }
    }
    return false;
  });
  const bool evaluated = decoding->Evaluate(prompt.back(), &error);
  Require(evaluated, error + "; peer snapshot: " + snapshot_error);
  decoding->SetCancellationCheck({});
  Require(concurrent_snapshot != nullptr, snapshot_error);
  // Restore shares the executor's stream and follows its completed forward.
  // The captured checkpoint must preserve its suffix before the rewind.
  if (prefix_checkpoint)
    Require(frozen->RestoreSnapshot(*prefix_checkpoint, &error), error);
  Require(concurrent_snapshot->bytes().size() ==
                  expected_snapshot->bytes().size() &&
              std::memcmp(concurrent_snapshot->bytes().data(),
                          expected_snapshot->bytes().data(),
                          expected_snapshot->bytes().size()) == 0,
          "decode graph capture changed a peer snapshot");
  Require(reference->Evaluate(prompt.back(), &error), error);
  RequireExact(reference->Logits(), decoding->Logits(),
               "snapshot capture changed decode logits");
  Require(decoding->Evaluate(prompt.front(), &error) &&
              reference->Evaluate(prompt.front(), &error),
          error);
  RequireExact(reference->Logits(), decoding->Logits(),
               "snapshot capture changed graph replay logits");
  std::cout << "peer snapshot bytes, graph capture and replay logits exact\n";
}

void CheckFailureRecovery(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  auto session = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      4096, &error);
  Require(session != nullptr, error);
  const auto initial_bytes = session->AllocatedBytes();
  const auto& c = model->config();
  const auto checkpoint_bytes =
      std::size_t{c.num_layers - c.num_layers / c.full_attention_interval} *
      c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim * sizeof(float);
  Require(model->SessionBytes(gufo::core::SessionMode::kSpeculative, 4096) >
              initial_bytes + checkpoint_bytes,
          "unused deep rollback state was allocated eagerly");
  const auto pattern =
      model->Tokenize("Explain virtual memory in a short sentence.");
  std::vector<std::int32_t> prompt(2048);
  for (std::size_t i = 0; i < prompt.size(); ++i)
    prompt[i] = pattern[i % pattern.size()];
  Require(session->Sync(prompt, &error), error);
  const std::vector<float> expected(session->Logits().begin(),
                                    session->Logits().end());
  const auto snapshot = session->SaveSnapshot(&error);
  Require(snapshot != nullptr, error);
  Require(snapshot->SizeBytes() < 200ULL * 1024 * 1024,
          "short prompt snapshot retains a full prefill hidden buffer");

  auto peer = model->CreateSession(gufo::core::SessionMode::kAutoregressive, 64,
                                   &error);
  Require(peer && peer->Sync(pattern, &error), error);
  const std::vector<float> peer_expected(peer->Logits().begin(),
                                         peer->Logits().end());
  unsigned checks = 0;
  for (const unsigned stop_at : {3U, 4U, 6U}) {
    peer->Reset();
    checks = 0;
    // Exercise both sides of PLE and reuse its mapped reader buffer in
    // another request immediately after cancellation.
    session->SetCancellationCheck([&] { return ++checks >= stop_at; });
    Require(!session->Evaluate(prompt.front(), &error) && !session->IsValid(),
            "cancelled mutation left a reusable session");
    Require(session->Tokens().empty() && session->Logits().empty() &&
                !session->SaveSnapshot(&error),
            "poisoned state exposed reusable tokens, logits or a snapshot");
    session->SetCancellationCheck({});
    Require(peer->Sync(pattern, &error), error);
    RequireExact(peer_expected, peer->Logits(),
                 "cancelled forward left pinned inputs in use");
    Require(session->Sync(prompt, &error) && session->IsValid(), error);
    RequireExact(expected, session->Logits(),
                 "Sync reused partially mutated state");
  }

  checks = 0;
  session->SetCancellationCheck([&] { return ++checks >= 3; });
  Require(!session->RestoreSnapshot(*snapshot, &error),
          "snapshot restoration ignored cancellation");
  session->SetCancellationCheck({});
  Require(session->Sync(prompt, &error), error);
  RequireExact(expected, session->Logits(),
               "cancelled restore leaked device state");
  Require(session->AllocatedBytes() == initial_bytes,
          "reset retained rollback allocations");
  std::cout
      << "cancelled forward/restore recover exactly; rollback starts empty\n";
}

void CheckRollbackReuse(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  const auto prompt = model->Tokenize("Continue: red, blue, red, blue,");
  const auto context = static_cast<std::uint32_t>(prompt.size() + 4);
  auto session = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      context, &error);
  Require(session != nullptr, error);
  const auto empty_bytes = session->AllocatedBytes();
  Require(session->Sync(prompt, &error), error);
  sampling::SamplerState sampler(
      {.seed = 73},
      std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
  qfn::Session::DecodeResult step;
  Require(session->DecodeStep(2, sampler, &step, &error, false), error);
  const auto warm_bytes = session->AllocatedBytes();
  Require(warm_bytes > empty_bytes &&
              warm_bytes <= model->SessionBytes(
                                gufo::core::SessionMode::kSpeculative, context),
          "rollback allocation exceeded its configured bound");
  session->ResetDraftPolicy();
  const auto warm = session->SaveSnapshot(&error);
  Require(warm && session->RestoreSnapshot(*warm, &error), error);
  Require(session->AllocatedBytes() == warm_bytes,
          "restore discarded reusable rollback or allocated deeper rows");
  auto replay_sampler = sampler;
  qfn::Session::DecodeResult expected, actual;
  Require(session->DecodeStep(2, sampler, &expected, &error, false), error);
  const std::vector<float> logits(session->Logits().begin(),
                                  session->Logits().end());
  Require(session->RestoreSnapshot(*warm, &error) &&
              session->DecodeStep(2, replay_sampler, &actual, &error, false),
          error);
  Require(actual.tokens == expected.tokens &&
              sampler.rng_state() == replay_sampler.rng_state(),
          "reused rollback changed continuation replay");
  RequireExact(logits, session->Logits(), "reused rollback changed logits");
  while (session->Position() < context)
    Require(session->Evaluate(prompt.front(), &error), error);
  const auto full = session->SaveSnapshot(&error);
  Require(full && session->RestoreSnapshot(*full, &error), error);
  Require(session->AllocatedBytes() == empty_bytes,
          "restore kept rollback that the restored context cannot use");
  session->Reset();
  Require(session->AllocatedBytes() == empty_bytes,
          "explicit reset retained rollback");
  std::cout
      << "rollback restore is bounded, reuses warm rows and replays exactly\n";
}

void CheckImageSnapshotAttachment(const std::shared_ptr<qfn::Model>& model) {
  namespace vision = gufo::models::qwen::vision;
  if (!model->VisionEncoder()) {
    std::cout
        << "image snapshot attachment: matching vision sidecar unavailable\n";
    return;
  }
  const auto image = [](bool blue) {
    auto prompt = std::make_shared<vision::Prompt>();
    const vision::ImageGrid grid{0, 2, 2};
    prompt->tokens = {vision::kImageToken, vision::kImageToken,
                      vision::kImageToken, vision::kImageToken, 42};
    prompt->rope.images.push_back(grid);
    gufo::core::Image pixels;
    pixels.width = pixels.height = 64;
    pixels.pixels.resize(64 * 64 * 3);
    for (std::size_t i = blue ? 2 : 0; i < pixels.pixels.size(); i += 3)
      pixels.pixels[i] = 255;
    prompt->images.push_back(
        {std::make_shared<const gufo::core::Image>(std::move(pixels)), grid});
    prompt->images.back().prefix_identity.fill(blue ? 2 : 1);
    prompt->cache_identity.assign(32, blue ? 2 : 1);
    return prompt;
  };
  const auto red = image(false), blue = image(true);
  const std::vector<std::int32_t> tokens(red->tokens.begin(),
                                         red->tokens.end());
  std::string error;
  auto origin = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      64, &error);
  auto restored = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      64, &error);
  auto text = model->CreateSession(
      model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                      : gufo::core::SessionMode::kAutoregressive,
      64, &error);
  Require(origin && restored && text, error);
  origin->ConfigureVision(red);
  Require(origin->Sync(std::span(tokens).first(2), &error), error);
  const auto snapshot = origin->SaveSnapshot(&error);
  Require(snapshot != nullptr, error);
  Require(!restored->RestoreSnapshot(*snapshot, &error),
          "image snapshot restored without pixels");
  restored->ConfigureVision(blue);
  Require(restored->Sync(std::span(tokens).first(2), &error), error);
  Require(!restored->RestoreSnapshot(*snapshot, &error),
          "image snapshot inherited another image's pixels");
  restored->ConfigureVision(red);
  Require(restored->RestoreSnapshot(*snapshot, &error) &&
              restored->Sync(tokens, &error) && origin->Sync(tokens, &error),
          error);
  RequireExact(origin->Logits(), restored->Logits(),
               "attached image snapshot changed continuation logits");

  const std::array<std::int32_t, 2> plain{42, 43};
  Require(text->Sync(plain, &error), error);
  const auto text_snapshot = text->SaveSnapshot(&error);
  Require(text_snapshot && restored->RestoreSnapshot(*text_snapshot, &error),
          error);
  Require(text->Evaluate(44, &error) && restored->Evaluate(44, &error), error);
  RequireExact(text->Logits(), restored->Logits(),
               "text snapshot retained stale image input");
  std::cout
      << "image snapshot attachment, replacement and text restore exact\n";
}

void CheckPrefillChunks(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  const auto pattern = model->Tokenize(
      "Virtual memory maps pages to physical storage. "
      "A train travels sixty kilometers per hour. "
      "Continue red, green, blue, red, green, blue. ");
  Require(!pattern.empty(), "empty chunk fixture");
  for (const auto [length, boundary] :
       {std::pair{136U, 94U}, std::pair{136U, 103U}, std::pair{136U, 104U},
        std::pair{136U, 127U}, std::pair{136U, 128U}, std::pair{136U, 135U},
        std::pair{2048U, 1025U}, std::pair{4096U, 2048U}}) {
    std::vector<std::int32_t> tokens(length);
    for (std::size_t i = 0; i < tokens.size(); ++i)
      tokens[i] = pattern[i % pattern.size()];
    auto bulk = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        6145, &error);
    auto split = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        6145, &error);
    Require(bulk && split, error);
    Require(bulk->Sync(tokens, &error) &&
                split->Sync(std::span(tokens).first(boundary), &error) &&
                split->Sync(tokens, &error),
            error);
    std::vector<sampling::TokenId> history(tokens.begin(), tokens.end());
    for (unsigned step = 0; step < 4; ++step) {
      const auto logits = bulk->Logits();
      RequireExact(
          logits, split->Logits(),
          "prefill chunking changed logits: length=" + std::to_string(length) +
              " boundary=" + std::to_string(boundary) +
              " step=" + std::to_string(step));
      const auto token = static_cast<std::int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      Require(bulk->Evaluate(token, &error) && split->Evaluate(token, &error),
              error);
      history.push_back(token);
    }
    if (model->HasMtp()) {
      const sampling::SamplingConfig config{
          .temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = 73};
      sampling::SamplerState a(config, history), b(config, history);
      for (unsigned generated = 0; generated < 16;) {
        qfn::Session::DecodeResult first, second;
        Require(
            bulk->DecodeStep(16 - generated, a, &first, &error, false) &&
                split->DecodeStep(16 - generated, b, &second, &error, false),
            error);
        Require(!first.tokens.empty() && first.tokens == second.tokens &&
                    a.rng_state() == b.rng_state(),
                "prefill chunking changed sampled MTP replay: length=" +
                    std::to_string(length) +
                    " boundary=" + std::to_string(boundary));
        RequireExact(bulk->Logits(), split->Logits(),
                     "prefill chunking changed MTP target logits");
        generated += first.tokens.size();
      }
    }
    std::cout << "prefill chunks: length=" << length << " boundary=" << boundary
              << " four logit rows and sampled MTP replay exact\n"
              << std::flush;
  }
}

// A prefetched chunk serves the next chunk's leading tokens when a cache
// frontier or step budget ends that chunk before or after the hint.
void CheckPrefetchHints(const std::shared_ptr<qfn::Model>& model) {
  const auto& c = model->config();
  if (c.ple_layer < 0)
    return;
  std::string error;
  const auto pattern = model->Tokenize(
      "Virtual memory maps pages to physical storage. "
      "A train travels sixty kilometers per hour. ");
  std::vector<std::int32_t> tokens(2048), other(2048);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = pattern[i % pattern.size()];
    other[i] = pattern[(i + 1025) % pattern.size()];
  }
  Require(other[0] != tokens[1024], "diverging hint shares its first token");
  const auto mode = model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                                    : gufo::core::SessionMode::kAutoregressive;
  auto bulk = model->CreateSession(mode, 6145, &error);
  Require(bulk && bulk->Sync(tokens, &error), error);
  const std::span all(tokens);
  struct Case {
    const char* name;
    std::span<const std::int32_t> hint;
    std::size_t middle;  ///< a frontier before the full prompt, or 0
    std::size_t wasted;  ///< hinted tokens no chunk claims
  };
  for (const auto& [name, hint, middle, wasted] : {
           Case{"exact", all.subspan(1024), 0, 0},
           Case{"shorter", all.subspan(1024, 512), 0, 0},
           Case{"longer", all.subspan(1024), 1536, 512},
           Case{"diverging", std::span<const std::int32_t>(other).first(1024),
                0, 1024},
       }) {
    auto split = model->CreateSession(mode, 6145, &error);
    Require(split != nullptr, error);
    Require(split->Sync(all.first(1024), &error, hint), error);
    if (middle != 0)
      Require(split->Sync(all.first(middle), &error), error);
    Require(split->Sync(tokens, &error), error);
    const auto rows = split->NgramRowTotals();
    Require(
        rows.rows == (tokens.size() + wasted) * c.ple_heads &&
            rows.rows == rows.dedup_rows + rows.cache_hits + rows.disk_reads,
        std::string("prefetch hint read unexpected rows: ") + name);
    RequireExact(bulk->Logits(), split->Logits(),
                 std::string("prefetch hint changed logits: ") + name);
    std::cout << "prefetch hint: " << name << " exact\n" << std::flush;
  }
}

void CheckMtpCacheReplay(const std::shared_ptr<qfn::Model>& model) {
  namespace qt = gufo::tokenization;
  std::string error;
  for (const auto seed : {1U, 2U}) {
    std::string system = "Case " + std::to_string(seed) + ". ";
    for (unsigned i = 0; i < 32; ++i)
      system += "Follow the user instruction carefully and answer accurately. ";
    const std::array messages{
        qt::ChatMessage{qt::ChatRole::kSystem, system},
        qt::ChatMessage{qt::ChatRole::kUser,
                        "Name three rivers in Europe and one fact about each."},
        qt::ChatMessage{qt::ChatRole::kAssistant,
                        "The Danube, the Rhine and the Loire."},
        qt::ChatMessage{qt::ChatRole::kUser,
                        "Now write a short story about one of them."}};
    const auto encoded = qt::QwenChatTemplate::RenderAndTokenize(
        model->tokenizer(), messages, {.enable_thinking = false}, &error);
    Require(encoded.has_value(), error);
    const std::vector<std::int32_t> prompt(encoded->begin(), encoded->end());
    Require(prompt.size() == 355, "MTP cache replay fixture changed");
    const sampling::SamplingConfig config{
        .temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = seed};
    const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
    std::vector<std::int32_t> expected;
    std::vector<std::size_t> expected_widths;
    std::uint64_t expected_rng = 0;
    // The server's immutable frontier excludes the seven-token generation
    // suffix. Rebuilding the same prompt must not depend on whether a
    // previous turn already supplied its first 320 tokens.
    for (const auto boundary : {320U, 348U}) {
      auto session = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                          6145, &error);
      Require(session &&
                  session->Sync(std::span(prompt).first(boundary), &error) &&
                  session->Sync(prompt, &error),
              error);
      const auto snapshot = session->SaveSnapshot(&error);
      Require(snapshot != nullptr, error);
      for (unsigned restore = 0; restore < 2; ++restore) {
        if (restore)
          Require(session->RestoreSnapshot(*snapshot, &error), error);
        session->ResetDraftPolicy();
        sampling::SamplerState sampler(config, history);
        std::vector<std::int32_t> tokens;
        std::vector<std::size_t> widths;
        while (tokens.size() < 200) {
          qfn::Session::DecodeResult step;
          Require(session->DecodeStep(200 - tokens.size(), sampler, &step,
                                      &error, false),
                  error);
          Require(!step.tokens.empty(), "MTP replay made no progress");
          tokens.insert(tokens.end(), step.tokens.begin(), step.tokens.end());
          widths.push_back(step.tokens.size());
        }
        if (expected.empty()) {
          expected = tokens;
          expected_widths = widths;
          expected_rng = sampler.rng_state();
        } else {
          Require(tokens == expected && widths == expected_widths &&
                      sampler.rng_state() == expected_rng,
                  "MTP cache replay changed tokens, acceptance or RNG: seed=" +
                      std::to_string(seed) +
                      " boundary=" + std::to_string(boundary) +
                      " restore=" + std::to_string(restore));
        }
        std::cout << "MTP cache replay seed=" << seed
                  << " boundary=" << boundary << " restore=" << restore
                  << " tokens=200 exact=1\n"
                  << std::flush;
      }
    }
  }
}

void CheckDraftPolicySnapshots(const std::shared_ptr<qfn::Model>& model) {
  if (!model->HasMtp())
    return;
  namespace qt = gufo::tokenization;
  std::string error;
  struct Case {
    const char* name;
    bool thinking;
    sampling::SamplingConfig config;
  };
  // Explicit model presets keep this fixture independent of their resolver.
  const std::array cases{
      Case{"greedy", false, {.temperature = 0.0F, .seed = 1}},
      Case{"thinking-default",
           true,
           {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F, .seed = 1}},
      Case{"nonthinking-default",
           false,
           {.temperature = 0.7F,
            .top_k = 20,
            .top_p = 0.8F,
            .seed = 1,
            .presence_penalty = 1.5F}}};
  for (const auto& test : cases) {
    for (const std::string text :
         {"Write a detailed story about a traveller finding a lost map in "
          "a library. Include a puzzle and its solution.",
          "เล่าเรื่องสั้นภาษาไทยเกี่ยวกับแมวสีส้มที่พบแผนที่"
          "ในห้องสมุดและออกเดินทางตามหาเมืองที่หายไป"}) {
      const std::array messages{
          qt::ChatMessage{qt::ChatRole::kSystem,
                          "Follow the instruction and answer accurately."},
          qt::ChatMessage{qt::ChatRole::kUser, text}};
      const auto encoded = qt::QwenChatTemplate::RenderAndTokenize(
          model->tokenizer(), messages, {.enable_thinking = test.thinking},
          &error);
      Require(encoded.has_value(), error);
      const std::vector<std::int32_t> prompt(encoded->begin(), encoded->end());
      auto live = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                       4096, &error);
      Require(live && live->Sync(prompt, &error), error);
      sampling::SamplerState sampler(
          test.config,
          std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
      // Train the actual controller before saving. Comparing two restored
      // sessions would miss a restore that resets their learned state equally.
      const auto trained = gufo::test::DecodeMtpReplay(*live, sampler, 128);
      Require(trained.drafted > 0, "policy snapshot did not exercise MTP");
      const auto snapshot = live->SaveSnapshot(&error);
      Require(snapshot != nullptr, error);
      std::vector<std::uint8_t> bytes(snapshot->SizeBytes());
      Require(snapshot->CopyTo(bytes), "serialize learned MTP policy");
      auto saved_sampler = sampler;
      const auto expected = gufo::test::DecodeMtpReplay(*live, sampler, 64);
      const std::vector<float> expected_logits(live->Logits().begin(),
                                               live->Logits().end());
      for (const bool serialized : {false, true}) {
        auto restored = model->CreateSession(
            gufo::core::SessionMode::kSpeculative, 4096, &error);
        Require(restored != nullptr, error);
        Require(serialized ? restored->RestoreSnapshot(bytes, &error)
                           : restored->RestoreSnapshot(*snapshot, &error),
                error);
        auto replay_sampler = saved_sampler;
        Require(gufo::test::DecodeMtpReplay(*restored, replay_sampler, 64) ==
                    expected,
                "restored MTP policy differs from uninterrupted generation");
        RequireExact(expected_logits, restored->Logits(),
                     "restored MTP policy changed final target logits");
      }
    }
    std::cout << "policy snapshot preset=" << test.name
              << " uninterrupted_RAM_serialized_exact=1\n";
  }
  std::cout << "learned MTP policy: uninterrupted, RAM and serialized "
               "greedy and default-sampled continuation are exact\n";
}

void CheckBatchFailureIsolation(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  const auto prompt = model->Tokenize("Repeat red, blue, red, blue,");
  for (const auto mode : {gufo::core::SessionMode::kAutoregressive,
                          gufo::core::SessionMode::kSpeculative}) {
    auto first = model->CreateSession(mode, 256, &error);
    auto second = model->CreateSession(mode, 256, &error);
    Require(first && second && first->Sync(prompt, &error), error);
    const auto snapshot = first->SaveSnapshot(&error);
    Require(snapshot && second->RestoreSnapshot(*snapshot, &error), error);
    const auto token = prompt.front();
    const std::array baseline{
        qfn::Session::AdvanceRequest{first.get(), token},
        qfn::Session::AdvanceRequest{second.get(), token}};
    Require(qfn::Session::EvaluateBatch(baseline, &error), error);
    const std::vector<float> expected(second->Logits().begin(),
                                      second->Logits().end());
    Require(first->RestoreSnapshot(*snapshot, &error) &&
                second->RestoreSnapshot(*snapshot, &error),
            error);
    unsigned checks = 0;
    first->SetCancellationCheck([&] { return ++checks >= 12; });
    std::array<qfn::Session::BatchOutcome, 2> outcomes;
    const std::array cancelled{
        qfn::Session::AdvanceRequest{first.get(), token, &outcomes[0]},
        qfn::Session::AdvanceRequest{second.get(), token, &outcomes[1]}};
    Require(!qfn::Session::EvaluateBatch(cancelled, &error),
            "batch cancellation not reported");
    Require(!outcomes[0].completed && !outcomes[0].error.empty() &&
                outcomes[1].completed && second->IsValid() && !first->IsValid(),
            "cancellation must invalidate only the partially executed session");
    RequireExact(expected, second->Logits(),
                 "cancelled peer changed target logits");
    first->SetCancellationCheck({});
    Require(first->RestoreSnapshot(*snapshot, &error) &&
                second->RestoreSnapshot(*snapshot, &error),
            error);
    const std::array invalid{
        qfn::Session::AdvanceRequest{first.get(), -1, &outcomes[0]},
        qfn::Session::AdvanceRequest{second.get(), token, &outcomes[1]}};
    Require(!qfn::Session::EvaluateBatch(invalid, &error) && first->IsValid() &&
                second->IsValid() && outcomes[1].completed,
            "invalid preparation poisoned a peer or the untouched session");
    RequireExact(expected, second->Logits(),
                 "invalid peer changed target logits");
    Require(first->RestoreSnapshot(*snapshot, &error) &&
                second->RestoreSnapshot(*snapshot, &error),
            error);
    const sampling::SamplingConfig config{
        .temperature = 0.7F, .top_p = 0.9F, .seed = 73};
    sampling::SamplerState control(config), a(config), b(config);
    qfn::Session::DecodeResult reference, ignored, actual;
    Require(second->DecodeStep(3, control, &reference, &error, false), error);
    const std::vector<float> speculative_logits(second->Logits().begin(),
                                                second->Logits().end());
    Require(second->RestoreSnapshot(*snapshot, &error), error);
    const std::array invalid_decode{
        qfn::Session::DecodeRequest{first.get(), 0, &a, &ignored, false,
                                    &outcomes[0]},
        qfn::Session::DecodeRequest{second.get(), 3, &b, &actual, false,
                                    &outcomes[1]}};
    Require(!qfn::Session::DecodeBatch(invalid_decode, &error) &&
                first->IsValid() && outcomes[1].completed &&
                actual.tokens == reference.tokens &&
                b.rng_state() == control.rng_state(),
            "invalid decode peer changed sampling replay");
    RequireExact(speculative_logits, second->Logits(),
                 "invalid decode peer changed logits");
    if (mode == gufo::core::SessionMode::kSpeculative) {
      Require(first->RestoreSnapshot(*snapshot, &error) &&
                  second->RestoreSnapshot(*snapshot, &error),
              error);
      a = sampling::SamplerState(config);
      b = sampling::SamplerState(config);
      checks = 0;
      first->SetCancellationCheck([&] { return ++checks >= 6; });
      const std::array mtp_cancel{
          qfn::Session::DecodeRequest{first.get(), 3, &a, &ignored, false,
                                      &outcomes[0]},
          qfn::Session::DecodeRequest{second.get(), 3, &b, &actual, false,
                                      &outcomes[1]}};
      Require(!qfn::Session::DecodeBatch(mtp_cancel, &error) &&
                  !outcomes[0].completed && outcomes[1].completed &&
                  second->IsValid() && actual.tokens == reference.tokens &&
                  b.rng_state() == control.rng_state(),
              "cancelled MTP peer changed sampling replay");
      RequireExact(speculative_logits, second->Logits(),
                   "cancelled MTP peer changed logits");
    }
  }
  std::cout << "batch preparation and mid-execution cancellation isolate peers "
               "and sampling replay\n";
}

void CheckBatchedSessions(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  std::size_t sampled_rejections = 0;
  std::size_t sampled_acceptances = 0;
  std::size_t sampled_small_head = 0;
  std::vector<std::unique_ptr<qfn::Session>> serial, batched;
  std::vector<sampling::SamplerState> serial_samplers, batch_samplers;
  const auto cases = gufo::test::QwenSamplingCases();
  for (std::size_t i = 0; i < 8; ++i) {
    serial.push_back(model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        6145, &error));
    batched.push_back(model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        6145, &error));
    Require(serial.back() && batched.back(), error);
    // Thai members start on the full head (their prompts bring new tokens
    // above the draft prefix) and table those tokens, while the others use
    // the small draft head from the start, so cohorts mix both heads.
    const bool thai = i == 3 || i == 6;
    auto prompt = model->Tokenize(
        thai ? "\u0e04\u0e33\u0e02\u0e2d " + std::to_string(i) +
                   ": \u0e15\u0e48\u0e2d\u0e44\u0e1b \u0e41\u0e14\u0e07, "
                   "\u0e19\u0e49\u0e33\u0e40\u0e07\u0e34\u0e19, "
                   "\u0e19\u0e49\u0e33\u0e40\u0e07\u0e34\u0e19, "
                   "\u0e41\u0e14\u0e07,"
             : "Batch request " + std::to_string(i) +
                   ": Continue red, blue, blue, red,");
    if (thai) {
      Require(std::ranges::any_of(prompt,
                                  [](std::int32_t token) {
                                    return token >= static_cast<std::int32_t>(
                                                        qfn::kMtpDraftVocab);
                                  }),
              "Thai batch prompt stays inside the draft vocabulary");
    }
    // One member crosses the sparse-attention boundary while the others
    // remain short; interleaving must not share positions or pooling state.
    if (i == 1) {
      const auto pattern = prompt;
      prompt.resize(4095);
      for (std::size_t t = pattern.size(); t < prompt.size(); ++t) {
        prompt[t] = pattern[t % pattern.size()];
      }
    }
    Require(serial.back()->Sync(prompt, &error), error);
    auto snapshot = serial.back()->SaveSnapshot(&error);
    Require(snapshot && batched.back()->RestoreSnapshot(*snapshot, &error),
            error);
    std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
    auto config = cases[(i * 3) % cases.size()].config;
    if (i % 2 == 0) {
      config.temperature = 0;
      config.repeat_penalty = i == 0 ? 0.7F : 1.3F;
      config.repeat_last_n = 3;
      config.frequency_penalty = i == 0 ? -0.4F : 0.2F;
      config.presence_penalty = i == 0 ? -1.5F : 1.5F;
    }
    serial_samplers.emplace_back(config, history);
    batch_samplers.emplace_back(config, history);
  }
  // Exercise the AR serving entrypoint, including a session crossing 4K.
  for (const std::size_t width : {2, 4, 6, 8}) {
    // Reuse cohorts with different tokens and positions in the same
    // allocations.
    for (unsigned replay = 0; replay < 2; ++replay) {
      std::vector<qfn::Session::AdvanceRequest> advances;
      for (std::size_t i = 0; i < width; ++i) {
        const auto token = static_cast<std::int32_t>(
            std::max_element(serial[i]->Logits().begin(),
                             serial[i]->Logits().end()) -
            serial[i]->Logits().begin());
        Require(serial[i]->Evaluate(token, &error), error);
        advances.push_back({batched[i].get(), token});
        serial_samplers[i].Accept(token);
        batch_samplers[i].Accept(token);
      }
      Require(qfn::Session::EvaluateBatch(advances, &error), error);
      for (std::size_t i = 0; i < width; ++i) {
        RequireExact(serial[i]->Logits(), batched[i]->Logits(),
                     "batched AR frontier differs at C" +
                         std::to_string(width) + " row " + std::to_string(i));
      }
    }
    // Different budgets create ragged chains and include ordinary one-token
    // decoding in the same batch as speculative verification.
    for (unsigned cycle = 0; cycle < 2; ++cycle) {
      std::vector<qfn::Session::DecodeResult> expected(width), actual(width);
      std::vector<qfn::Session::DecodeRequest> requests;
      std::vector<qfn::Session::SpeculativeStats> before;
      for (std::size_t i = 0; i < width; ++i) {
        const std::size_t budget = 1 + (i * 3 + cycle * 5 + 7) % 8;
        before.push_back(serial[i]->Statistics());
        // Unplanned greedy rows price drafts at the cohort width.
        Require(serial[i]->DecodeStep(budget, serial_samplers[i], &expected[i],
                                      &error, false,
                                      static_cast<std::uint32_t>(width)),
                error);
        requests.push_back(
            {batched[i].get(), budget, &batch_samplers[i], &actual[i], false});
      }
      if (cycle % 2 != 0) {
        std::reverse(requests.begin(), requests.end());
      }
      Require(qfn::Session::DecodeBatch(requests, &error), error);
      for (std::size_t i = 0; i < width; ++i) {
        const auto a = serial[i]->Statistics();
        const auto b = batched[i]->Statistics();
        Require(expected[i].tokens == actual[i].tokens &&
                    expected[i].stop == actual[i].stop &&
                    serial_samplers[i].rng_state() ==
                        batch_samplers[i].rng_state() &&
                    a.cycles == b.cycles && a.drafted == b.drafted &&
                    a.accepted == b.accepted &&
                    a.small_head_cycles == b.small_head_cycles &&
                    a.lookup == b.lookup &&
                    a.lookup_accepted == b.lookup_accepted &&
                    std::equal(serial[i]->Tokens().begin(),
                               serial[i]->Tokens().end(),
                               batched[i]->Tokens().begin(),
                               batched[i]->Tokens().end()),
                "batched decode tokens, RNG or acceptance differ");
        RequireExact(serial[i]->Logits(), batched[i]->Logits(),
                     "batched MTP frontier differs at C" +
                         std::to_string(width) + " row " + std::to_string(i) +
                         " cycle " + std::to_string(cycle));
        Require(std::equal(serial_samplers[i].history().begin(),
                           serial_samplers[i].history().end(),
                           batch_samplers[i].history().begin(),
                           batch_samplers[i].history().end()),
                "batched sampling history differs");
        // Probe a copy so the test also compares pending residual draws at
        // the final cycle, without consuming either request's real state.
        auto serial_next = serial_samplers[i];
        auto batch_next = batch_samplers[i];
        Require(serial_next.Sample(serial[i]->Logits()) ==
                        batch_next.Sample(batched[i]->Logits()) &&
                    serial_next.rng_state() == batch_next.rng_state(),
                "batched next draw or deferred residual differs");
        if (serial_samplers[i].config().uses_random_sampling()) {
          const auto drafted = a.drafted - before[i].drafted;
          const auto accepted = a.accepted - before[i].accepted;
          sampled_acceptances += accepted;
          sampled_small_head +=
              a.small_head_cycles - before[i].small_head_cycles;
          if (accepted < drafted) {
            ++sampled_rejections;
            // stop_at_eos is false: a sampled rejection always defers its
            // correction. Retrieving it must not draw again from p.
            Require(serial_next.rng_state() == serial_samplers[i].rng_state(),
                    "sampled rejection lost its deferred correction");
          }
        }
      }
    }
    std::cout << "batch C" << width << " AR_MTP_logits_RNG_exact=1\n"
              << std::flush;
  }
  Require(
      sampled_acceptances > 0 && sampled_rejections > 0,
      "batch test must exercise sampled acceptance and residual correction");
  // Sampled members draft through the small head on this eight-session model.
  Require(sampled_small_head > 0,
          "batched sampled members never used the small draft head");
  std::cout << "batch sampled_accepted=" << sampled_acceptances
            << " rejected_cycles=" << sampled_rejections
            << " sampled_small_head_cycles=" << sampled_small_head
            << " history_and_residual_exact=1\n"
            << std::flush;
  // A complete state comparison catches recurrent/hidden differences that a
  // short output comparison could miss.
  for (std::size_t i = 0; i < serial.size(); ++i) {
    const auto a = serial[i]->SaveSnapshot(&error);
    const auto b = batched[i]->SaveSnapshot(&error);
    Require(a && b &&
                std::equal(a->bytes().begin(), a->bytes().end(),
                           b->bytes().begin(), b->bytes().end()),
            "batched snapshot state differs: " + std::to_string(i));
  }
  std::cout << "batch independent_state_exact=1\n" << std::flush;
}

void CheckLookupBatch(const std::shared_ptr<qfn::Model>& model) {
  namespace qt = gufo::tokenization;
  std::string error;
  std::vector<std::unique_ptr<qfn::Session>> serial, batched;
  std::vector<sampling::SamplerState> samplers, twins;
  for (std::size_t i = 0; i < 8; ++i) {
    std::string text =
        "Repeat the following text exactly, without an introduction or a "
        "code fence:\n\n";
    for (unsigned n = 0; n < 24; ++n)
      text += i % 2 ? "Monday, Tuesday, Wednesday, Thursday, Friday, "
                    : "red, blue, green, yellow, orange, purple, ";
    const std::array messages{qt::ChatMessage{qt::ChatRole::kUser, text}};
    const auto encoded = qt::QwenChatTemplate::RenderAndTokenize(
        model->tokenizer(), messages, {.enable_thinking = false}, &error);
    Require(encoded.has_value(), error);
    const std::vector<std::int32_t> prompt(encoded->begin(), encoded->end());
    serial.push_back(model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                          2048, &error));
    batched.push_back(model->CreateSession(
        gufo::core::SessionMode::kSpeculative, 2048, &error));
    Require(
        serial.back() && batched.back() && serial.back()->Sync(prompt, &error),
        error);
    const auto snapshot = serial.back()->SaveSnapshot(&error);
    Require(snapshot && batched.back()->RestoreSnapshot(*snapshot, &error),
            error);
    sampling::SamplingConfig config;
    config.temperature = i % 2 ? 0.7F : 0.0F;
    config.seed = 19 + i;
    config.repeat_penalty = 1;
    config.frequency_penalty = 0;
    config.presence_penalty = 0;
    if (i % 4 == 1) {
      // Mix default sampled verification with unbounded sampling and greedy
      // peers, retaining exact per-session logits and replay across C2..C8.
      config.top_k = 20;
      config.top_p = 0.8F;
      config.presence_penalty = 1.5F;
    }
    samplers.emplace_back(
        config, std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
    twins.push_back(samplers.back());
  }
  for (const std::size_t width : {2, 4, 6, 8}) {
    for (unsigned cycle = 0; cycle < 4; ++cycle) {
      std::vector<qfn::Session::DecodeResult> expected(width), actual(width);
      std::vector<qfn::Session::DecodeRequest> requests;
      for (std::size_t i = 0; i < width; ++i) {
        const std::size_t budget = cycle == 2 && i % 3 == 0 ? 3 : 8;
        // Unplanned greedy rows price drafts at the cohort width.
        Require(serial[i]->DecodeStep(budget, samplers[i], &expected[i], &error,
                                      false, static_cast<std::uint32_t>(width)),
                error);
        requests.push_back(
            {batched[i].get(), budget, &twins[i], &actual[i], false});
      }
      if (cycle % 2)
        std::reverse(requests.begin(), requests.end());
      Require(qfn::Session::DecodeBatch(requests, &error), error);
      for (std::size_t i = 0; i < width; ++i) {
        const auto a = serial[i]->Statistics();
        const auto b = batched[i]->Statistics();
        Require(expected[i].tokens == actual[i].tokens &&
                    expected[i].stop == actual[i].stop &&
                    samplers[i].rng_state() == twins[i].rng_state() &&
                    a.drafted == b.drafted && a.accepted == b.accepted &&
                    a.lookup == b.lookup &&
                    a.lookup_accepted == b.lookup_accepted,
                "lookup changed tokens, RNG or proposals with batch peers");
        RequireExact(serial[i]->Logits(), batched[i]->Logits(),
                     "lookup changed batched target logits");
        auto left = samplers[i], right = twins[i];
        Require(left.Sample(serial[i]->Logits()) ==
                        right.Sample(batched[i]->Logits()) &&
                    left.rng_state() == right.rng_state(),
                "lookup changed the deferred sampled correction");
      }
    }
    std::cout << "lookup C" << width << " tokens_logits_RNG_exact=1\n"
              << std::flush;
  }
  std::uint64_t sampled_copies = 0, greedy_copies = 0;
  for (std::size_t i = 0; i < serial.size(); ++i) {
    const auto stats = serial[i]->Statistics();
    (i % 2 ? sampled_copies : greedy_copies) += stats.lookup_accepted;
    const auto a = serial[i]->SaveSnapshot(&error);
    const auto b = batched[i]->SaveSnapshot(&error);
    Require(a && b && a->bytes().size() == b->bytes().size() &&
                std::equal(a->bytes().begin(), a->bytes().end(),
                           b->bytes().begin()),
            "lookup batched snapshot state differs");
  }
  Require(sampled_copies != 0 && greedy_copies != 0,
          "lookup batch check did not accept sampled and greedy copies");
  std::cout << "lookup sampled_copies=" << sampled_copies
            << " greedy_copies=" << greedy_copies << " snapshots_exact=1\n"
            << std::flush;
  // Serving can reuse the entire prefix without another Sync. Resetting
  // request policy must skip lookup for exactly the first speculative step.
  serial[0]->ResetDraftPolicy();
  batched[0]->ResetDraftPolicy();
  bool resumed_lookup = false;
  for (unsigned step = 0; step < 32; ++step) {
    const auto before = serial[0]->Statistics().lookup;
    qfn::Session::DecodeResult a, b;
    Require(serial[0]->DecodeStep(8, samplers[0], &a, &error, false) &&
                batched[0]->DecodeStep(8, twins[0], &b, &error, false),
            error);
    Require(
        a.tokens == b.tokens && samplers[0].rng_state() == twins[0].rng_state(),
        "lookup request reset changed replay");
    const auto copied = serial[0]->Statistics().lookup - before;
    if (step == 0)
      Require(copied == 0, "reused request copied " + std::to_string(copied) +
                               " tokens on its first step");
    else if (copied != 0) {
      resumed_lookup = true;
      break;
    }
  }
  Require(resumed_lookup, "request reset permanently disabled lookup");
  // Reuse a session whose old lookup index contains a different passage,
  // then rewind both sessions into the middle of a sampled copy. Restoring
  // only into a fresh session would miss stale index and resume state.
  const auto checkpoint = serial[1]->SaveSnapshot(&error);
  Require(checkpoint != nullptr, error);
  const auto saved_sampler = samplers[1];
  for (unsigned replay = 0; replay < 2; ++replay) {
    Require(serial[1]->RestoreSnapshot(*checkpoint, &error) &&
                batched[0]->RestoreSnapshot(*checkpoint, &error),
            error);
    auto left = saved_sampler, right = saved_sampler;
    for (const std::size_t budget : {1, 2, 8, 3, 8}) {
      qfn::Session::DecodeResult a, b;
      Require(serial[1]->DecodeStep(budget, left, &a, &error, false) &&
                  batched[0]->DecodeStep(budget, right, &b, &error, false),
              error);
      Require(a.tokens == b.tokens && a.stop == b.stop &&
                  left.rng_state() == right.rng_state(),
              "lookup restore into a used session changed sampled replay");
      RequireExact(serial[1]->Logits(), batched[0]->Logits(),
                   "lookup restore changed target logits");
    }
  }
  std::cout << "lookup restore_used_session_and_rewind_exact=1\n" << std::flush;
}

// The draft table's lifecycle on the device: rare tokens join it, the small
// head comes back once a window passes without a miss, tabled rare drafts map
// back to real IDs, a restore into a session with a populated table matches a
// fresh one, and a mixed batch matches isolated sessions.
std::vector<std::int32_t> RepeatTokens(std::vector<std::int32_t> pattern,
                                       std::size_t size) {
  std::vector<std::int32_t> out(size);
  for (std::size_t t = 0; t < size; ++t)
    out[t] = pattern[t % pattern.size()];
  return out;
}

std::size_t RareTokens(std::span<const std::int32_t> tokens) {
  return static_cast<std::size_t>(
      std::ranges::count_if(tokens, [](std::int32_t token) {
        return token >= static_cast<std::int32_t>(qfn::kMtpDraftVocab);
      }));
}

// The reduced draft head serves one-session greedy requests (and planned
// batches); on a multi-session model an unplanned greedy request keeps the
// full head, so this check needs a one-session model. Greedy prompt lookup can
// supply a whole chain without any draft head, so head use is judged per cycle,
// for the cycles that draft through the MTP head.
void CheckDraftVocabLifecycle(const std::shared_ptr<qfn::Model>& model) {
  if (!model->HasMtp())
    return;
  Require(model->DecodeConcurrency() == 1,
          "the draft-vocabulary lifecycle needs a one-session model");
  std::string error;
  const auto create = [&] {
    auto session = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                        6145, &error);
    Require(session != nullptr, error);
    return session;
  };
  // Every rare token appears in the first copy of the pattern; the copies
  // after it run well past the window without a miss.
  const auto repeated = [&](const std::string& text) {
    const auto pattern = model->Tokenize(text);
    Require(RareTokens(pattern) * 3 >= pattern.size(),
            "lifecycle pattern is not mostly rare tokens");
    return RepeatTokens(pattern,
                        pattern.size() * 2 + qfn::kMtpDraftVocabWindow + 64);
  };
  const auto thai = repeated(
      "แมวสีส้มนอน"
      "บนหลังคา สุน"
      "ัขสีดำวิ่งใ"
      "นสวน ");
  const auto hindi = repeated(
      "नारंगी बिल्ल"
      "ी छत पर सोती ह"
      "ै, काला कुत्त"
      "ा बगीचे में द"
      "ौड़ता है। ");
  const auto decode = [&](qfn::Session& session,
                          sampling::SamplerState& sampler, unsigned cycles) {
    std::vector<std::int32_t> out;
    for (unsigned c = 0; c < cycles; ++c) {
      qfn::Session::DecodeResult step;
      Require(session.DecodeStep(8, sampler, &step, &error, false), error);
      out.insert(out.end(), step.tokens.begin(), step.tokens.end());
    }
    return out;
  };
  // Cycles whose chain starts with an MTP draft, and how many of those used
  // the small head. A cycle of prompt-lookup copies alone drafts nothing.
  struct HeadUse {
    std::uint64_t drafting{0};
    std::uint64_t small_head{0};
  };
  const auto decode_counted = [&](qfn::Session& session,
                                  sampling::SamplerState& sampler,
                                  unsigned cycles, HeadUse* use) {
    std::vector<std::int32_t> out;
    for (unsigned c = 0; c < cycles; ++c) {
      const auto before = session.Statistics();
      qfn::Session::DecodeResult step;
      Require(session.DecodeStep(8, sampler, &step, &error, false), error);
      out.insert(out.end(), step.tokens.begin(), step.tokens.end());
      const auto& after = session.Statistics();
      if (after.drafted - before.drafted > after.lookup - before.lookup) {
        ++use->drafting;
        use->small_head += after.small_head_cycles - before.small_head_cycles;
      }
    }
    return out;
  };
  const auto history = [](std::span<const std::int32_t> tokens) {
    return std::vector<sampling::TokenId>(tokens.begin(), tokens.end());
  };

  // Rare tokens in the prompt were misses, but the window has passed, so the
  // greedy continuation drafts its rare tokens through the table. It also
  // repeats the prompt, so lookup copies supply some of its chains.
  auto t = create();
  Require(t->Sync(thai, &error), error);
  sampling::SamplerState greedy({}, history(thai));
  HeadUse tabled;
  const auto thai_out = decode_counted(*t, greedy, 8, &tabled);
  const auto& ts = t->Statistics();
  Require(tabled.drafting > 0, "no tabled cycle drafted through the MTP head");
  Require(tabled.small_head == tabled.drafting,
          "tabled text did not return to the small head");
  Require(RareTokens(thai_out) * 3 >= thai_out.size(),
          "the repeated Thai continuation has too few rare tokens");
  // Wrong table IDs would reject every drafted rare token. Accepted copies
  // are not drafts of the head, so they are left out.
  Require((ts.accepted - ts.lookup_accepted) * 2 >= ts.drafted - ts.lookup,
          "small-head drafts of tabled tokens were rejected");
  // Later restores into this session replace its statistics.
  const auto tabled_copies = ts.lookup;

  // A second request with its own table, then a restore of each into the
  // other's populated session against a restore into a fresh one.
  auto h = create();
  Require(h->Sync(hindi, &error), error);
  sampling::SamplerState hindi_greedy({}, history(hindi));
  (void)decode(*h, hindi_greedy, 4);
  const auto thai_snapshot = t->SaveSnapshot(&error);
  const auto hindi_snapshot = h->SaveSnapshot(&error);
  Require(thai_snapshot && hindi_snapshot, error);
  for (const bool sampled : {false, true}) {
    for (auto* pair : {&thai_snapshot, &hindi_snapshot}) {
      auto& used = pair == &thai_snapshot ? h : t;
      auto fresh = create();
      Require(used->RestoreSnapshot(**pair, &error) &&
                  fresh->RestoreSnapshot(**pair, &error),
              error);
      const auto restored = used->Tokens();
      const sampling::SamplingConfig config{
          .temperature = sampled ? 0.8F : 0.0F, .seed = 91};
      sampling::SamplerState a(config, history(restored));
      sampling::SamplerState b(config, history(restored));
      HeadUse use;
      const auto expected = decode_counted(*fresh, a, 4, &use);
      const auto actual = decode(*used, b, 4);
      const auto& fs = fresh->Statistics();
      const auto& us = used->Statistics();
      Require(expected == actual && a.rng_state() == b.rng_state() &&
                  fs.drafted == us.drafted && fs.accepted == us.accepted &&
                  fs.small_head_cycles == us.small_head_cycles,
              "restore into a populated draft table differs from a fresh "
              "session");
      // The restored Thai table keeps drafting through the small head.
      // Greedy lookup may copy every chain; sampled cycles always draft.
      Require(pair != &thai_snapshot || (use.small_head == use.drafting &&
                                         (!sampled || use.drafting > 0)),
              "a restored draft table left the small head");
      RequireExact(fresh->Logits(), used->Logits(),
                   "restore into a populated draft table changed logits");
    }
  }

  // Switching to penalties preserves the target/predictor checkpoint.
  // Greedy and sampled penalized proposals may both use the table. Both must
  // replay exactly in populated and fresh sessions.
  for (const float temperature : {0.0F, 0.8F}) {
    auto fresh = create();
    Require(t->RestoreSnapshot(*thai_snapshot, &error) &&
                fresh->RestoreSnapshot(*thai_snapshot, &error),
            error);
    t->ResetDraftPolicy();
    fresh->ResetDraftPolicy();
    const sampling::SamplingConfig config{
        .temperature = temperature, .seed = 73, .presence_penalty = 1.5F};
    sampling::SamplerState a(config, history(t->Tokens()));
    auto b = a;
    const auto expected = gufo::test::DecodeMtpReplay(*fresh, a, 32);
    const auto actual = gufo::test::DecodeMtpReplay(*t, b, 32);
    Require(expected == actual && actual.drafted > 0,
            "penalized replay differs after a populated draft table");
    Require(t->Statistics().small_head_cycles > 0 &&
                fresh->Statistics().small_head_cycles > 0,
            "penalized replay did not exercise the reduced head");
    RequireExact(fresh->Logits(), t->Logits(),
                 "penalized replay changed target logits");
  }

  // A mixed batch: two members on the small head (one with rare rows
  // tabled, one with specials only) and one whose new rare tokens keep it on
  // the full head. Every member must match its isolated twin.
  const auto english = model->Tokenize("Continue the list: red, blue, blue,");
  auto fresh_script = model->Tokenize(
      "คำขอ: ต่อไป "
      "แดง, น้ำเงิน,");
  Require(RareTokens(fresh_script) > 0, "full-head member has no rare token");
  std::vector<std::unique_ptr<qfn::Session>> serial, batched;
  std::vector<sampling::SamplerState> serial_samplers, batch_samplers;
  for (std::size_t i = 0; i < 3; ++i) {
    serial.push_back(create());
    batched.push_back(create());
    if (i == 0) {
      Require(serial[i]->RestoreSnapshot(*thai_snapshot, &error) &&
                  batched[i]->RestoreSnapshot(*thai_snapshot, &error),
              error);
    } else {
      const auto& prompt = i == 1 ? english : fresh_script;
      Require(
          serial[i]->Sync(prompt, &error) && batched[i]->Sync(prompt, &error),
          error);
    }
    const sampling::SamplingConfig config{
        .temperature = i < 2 ? 0.0F : 0.3F,
        .seed = static_cast<std::int64_t>(17 + i)};
    serial_samplers.emplace_back(config, history(serial[i]->Tokens()));
    batch_samplers.emplace_back(config, history(batched[i]->Tokens()));
  }
  for (unsigned cycle = 0; cycle < 3; ++cycle) {
    const auto tabled_before = batched[0]->Statistics();
    std::vector<qfn::Session::DecodeResult> expected(3), actual(3);
    std::vector<qfn::Session::DecodeRequest> requests;
    for (std::size_t i = 0; i < 3; ++i) {
      // Unplanned greedy rows price drafts at the cohort width.
      Require(serial[i]->DecodeStep(8, serial_samplers[i], &expected[i], &error,
                                    false, 3),
              error);
      requests.push_back(
          {batched[i].get(), 8, &batch_samplers[i], &actual[i], false});
    }
    Require(qfn::Session::DecodeBatch(requests, &error), error);
    for (std::size_t i = 0; i < 3; ++i) {
      const auto& a = serial[i]->Statistics();
      const auto& b = batched[i]->Statistics();
      Require(
          expected[i].tokens == actual[i].tokens &&
              serial_samplers[i].rng_state() == batch_samplers[i].rng_state() &&
              a.drafted == b.drafted && a.accepted == b.accepted &&
              a.small_head_cycles == b.small_head_cycles,
          "mixed-head batch member " + std::to_string(i) +
              " differs from its isolated session");
      RequireExact(
          serial[i]->Logits(), batched[i]->Logits(),
          "mixed-head batch changed logits of member " + std::to_string(i));
    }
    // The tabled member repeats its prompt, so lookup may copy its whole
    // chain; a chain it drafts must use the small head.
    const auto& tabled_after = batched[0]->Statistics();
    Require(tabled_after.drafted - tabled_before.drafted ==
                    tabled_after.lookup - tabled_before.lookup ||
                tabled_after.small_head_cycles ==
                    tabled_before.small_head_cycles + 1,
            "the tabled batch member drafted over the full head");
    if (cycle == 0) {
      Require(batched[1]->Statistics().small_head_cycles == 1 &&
                  batched[2]->Statistics().small_head_cycles == 0,
              "the batch did not mix small-head and full-head members");
    }
  }
  std::cout << "draft table: small head returns after the window ("
            << tabled.small_head << "/" << tabled.drafting
            << " drafting cycles, " << tabled_copies
            << " lookup copies), tabled rare drafts map back, restores into "
               "used tables and mixed batches are exact\n";
}

// On a two-session model, sampled requests draft through the small head
// alone and in batches, and each request's head choice and table stay its
// own: an isolated run, peers joining and leaving, and reordered pairs give
// identical tokens (reasoning included), RNG, acceptance, head use and
// logits. Restores into used sessions match fresh ones. Plain greedy uses
// the small head alone and in a jointly planned batch, keeps the full head
// beside a sampled peer, and its text is the same either way.
void CheckSampledDraftHead(const std::shared_ptr<qfn::Model>& model) {
  namespace qt = gufo::tokenization;
  if (!model->HasMtp())
    return;
  Require(model->DecodeConcurrency() == 2,
          "the sampled draft-head check needs a two-session model");
  std::string error;
  const auto create = [&] {
    auto session = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                        6145, &error);
    Require(session != nullptr, error);
    return session;
  };
  const auto chat = [&](const std::string& user, bool thinking) {
    const std::array messages{qt::ChatMessage{qt::ChatRole::kUser, user}};
    const auto encoded = qt::QwenChatTemplate::RenderAndTokenize(
        model->tokenizer(), messages, {.enable_thinking = thinking}, &error);
    Require(encoded.has_value(), error);
    return std::vector<std::int32_t>(encoded->begin(), encoded->end());
  };
  auto json = std::make_shared<sampling::TokenConstraint>();
  json->grammar = sampling::JsonConstraint::Compile(
      gufo::json::parse(R"({"type":"object","properties":{"summary":)"
                        R"({"type":"string"}},"required":["summary"],)"
                        R"("additionalProperties":false})"),
      true);
  json->vocabulary = std::make_shared<sampling::ConstraintVocabulary>(
      model->tokenizer().GetVocabSize(), [&](std::uint32_t id) {
        const auto& tokenizer = model->tokenizer();
        return sampling::ConstraintVocabulary::Piece{
            tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
      });
  // Pi-like defaults, sampled penalties, a Thai request that starts on the
  // full head, and a constrained JSON request.
  struct Member {
    std::vector<std::int32_t> prompt;
    sampling::SamplingConfig config;
  };
  const std::array<Member, 4> members{{
      {chat("Write a Python function that merges overlapping intervals, "
            "then explain its complexity.",
            true),
       {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F, .seed = 11}},
      {chat("List practical tips for reviewing a pull request.", true),
       {.temperature = 0.8F,
        .seed = 12,
        .frequency_penalty = 0.2F,
        .presence_penalty = 1.5F}},
      {chat("เขียนกลอน"
            "สั้นๆ เกี่"
            "ยวกับฝนที"
            "่ตกในเมือ"
            "ง",
            true),
       {.temperature = 0.8F, .top_k = 20, .seed = 13}},
      {chat("Summarize the benefits of unit tests as JSON.", false),
       {.temperature = 0.8F, .seed = 14, .constraint = json}},
  }};
  Require(RareTokens(members[2].prompt) > 0, "Thai member has no rare token");
  constexpr std::size_t kCycles = 8;
  struct Run {
    std::vector<std::unique_ptr<qfn::Session>> sessions;
    std::vector<sampling::SamplerState> samplers;
    std::vector<std::vector<std::int32_t>> generated;
    std::vector<std::size_t> full_head_drafts;
  };
  const auto start = [&] {
    Run run;
    for (const auto& member : members) {
      run.sessions.push_back(create());
      Require(run.sessions.back()->Sync(member.prompt, &error), error);
      run.samplers.emplace_back(
          member.config, std::vector<sampling::TokenId>(member.prompt.begin(),
                                                        member.prompt.end()));
      run.generated.emplace_back();
      run.full_head_drafts.push_back(0);
    }
    return run;
  };
  // Small-head batches and mixed small/full-head batches seen in a schedule.
  struct BatchUse {
    std::size_t all_small{0};
    std::size_t mixed{0};
  };
  const auto step = [&](Run& run, std::span<const std::size_t> group,
                        BatchUse* use) {
    std::vector<qfn::Session::SpeculativeStats> before;
    std::vector<qfn::Session::DecodeResult> results(group.size());
    std::vector<qfn::Session::DecodeRequest> requests;
    for (std::size_t g = 0; g < group.size(); ++g) {
      const auto i = group[g];
      before.push_back(run.sessions[i]->Statistics());
      requests.push_back(
          {run.sessions[i].get(), 8, &run.samplers[i], &results[g], false});
    }
    if (requests.size() == 1) {
      Require(run.sessions[group[0]]->DecodeStep(8, run.samplers[group[0]],
                                                 &results[0], &error, false),
              error);
    } else {
      Require(qfn::Session::DecodeBatch(requests, &error), error);
    }
    std::size_t drafting = 0, small = 0;
    for (std::size_t g = 0; g < group.size(); ++g) {
      const auto i = group[g];
      const auto& after = run.sessions[i]->Statistics();
      Require(!results[g].tokens.empty(), "sampled step made no progress");
      run.generated[i].insert(run.generated[i].end(), results[g].tokens.begin(),
                              results[g].tokens.end());
      if (after.drafted - before[g].drafted > after.lookup - before[g].lookup) {
        ++drafting;
        const auto used_small =
            after.small_head_cycles - before[g].small_head_cycles;
        small += used_small;
        run.full_head_drafts[i] += used_small == 0;
      }
    }
    if (use && group.size() > 1 && drafting == group.size()) {
      use->all_small += small == drafting;
      use->mixed += small != 0 && small != drafting;
    }
  };
  // Each member decodes kCycles steps in every schedule.
  using Group = std::vector<std::size_t>;
  const std::vector<Group> isolated = [&] {
    std::vector<Group> groups;
    for (std::size_t i = 0; i < members.size(); ++i)
      for (std::size_t c = 0; c < kCycles; ++c)
        groups.push_back({i});
    return groups;
  }();
  // Peers join and leave: member 0 starts alone, the Thai member joins and
  // leaves, then member 1; later pairs mix penalties, the constraint and
  // the Thai member, and some members finish alone.
  const std::vector<Group> overlapping{{0},    {0},    {0, 2}, {0, 2}, {0, 2},
                                       {0, 1}, {0, 1}, {0},    {1, 3}, {1, 3},
                                       {1, 2}, {1, 2}, {1},    {1, 3}, {3},
                                       {2, 3}, {2, 3}, {2},    {3},    {3}};
  const std::vector<Group> reordered = [&] {
    std::vector<Group> groups;
    for (std::size_t c = 0; c < kCycles; ++c) {
      groups.push_back(c % 2 ? Group{1, 0} : Group{0, 1});
      groups.push_back(c % 2 ? Group{2, 3} : Group{3, 2});
    }
    return groups;
  }();
  const auto play = [&](const std::vector<Group>& schedule, BatchUse* use) {
    std::array<std::size_t, 4> count{};
    for (const auto& group : schedule)
      for (const auto i : group)
        ++count[i];
    Require(std::ranges::all_of(count, [](auto c) { return c == kCycles; }),
            "schedule gives a member the wrong number of cycles");
    auto run = start();
    for (const auto& group : schedule)
      step(run, group, use);
    return run;
  };
  const auto reference = play(isolated, nullptr);
  BatchUse overlap_use, reordered_use;
  for (const auto* schedule : {&overlapping, &reordered}) {
    const auto run = play(
        *schedule, schedule == &overlapping ? &overlap_use : &reordered_use);
    for (std::size_t i = 0; i < members.size(); ++i) {
      const auto& a = reference.sessions[i]->Statistics();
      const auto& b = run.sessions[i]->Statistics();
      Require(reference.generated[i] == run.generated[i] &&
                  reference.samplers[i].rng_state() ==
                      run.samplers[i].rng_state() &&
                  std::ranges::equal(reference.samplers[i].history(),
                                     run.samplers[i].history()) &&
                  a.cycles == b.cycles && a.drafted == b.drafted &&
                  a.accepted == b.accepted &&
                  a.small_head_cycles == b.small_head_cycles &&
                  a.lookup == b.lookup &&
                  a.lookup_accepted == b.lookup_accepted,
              "sampled member " + std::to_string(i) +
                  " depends on its overlap schedule");
      RequireExact(
          reference.sessions[i]->Logits(), run.sessions[i]->Logits(),
          "overlap schedule changed logits of member " + std::to_string(i));
    }
  }
  // The small head served every sampled kind alone, the Thai member also
  // drafted over the full head, and batches of both kinds ran.
  std::array<std::uint64_t, 4> small_cycles{};
  for (std::size_t i = 0; i < members.size(); ++i) {
    const auto& stats = reference.sessions[i]->Statistics();
    small_cycles[i] = stats.small_head_cycles;
    Require(stats.drafted > stats.lookup, "a sampled member never drafted");
  }
  Require(small_cycles[0] > 0 && small_cycles[1] > 0 && small_cycles[3] > 0,
          "a sampled member never used the small head on two sessions");
  Require(reference.full_head_drafts[2] > 0,
          "the Thai member never drafted over the full head");
  Require(overlap_use.all_small > 0 && overlap_use.mixed > 0 &&
              reordered_use.mixed + reordered_use.all_small > 0,
          "overlapping schedules did not run small-head and mixed batches");

  // Restore each table into the other's used session and into a fresh one,
  // then decode the used pair as a batch and the fresh pair alone.
  const auto english_snapshot = reference.sessions[0]->SaveSnapshot(&error);
  const auto thai_snapshot = reference.sessions[2]->SaveSnapshot(&error);
  Require(english_snapshot && thai_snapshot, error);
  const std::array snapshots{english_snapshot.get(), thai_snapshot.get()};
  std::array<std::unique_ptr<qfn::Session>, 2> fresh{create(), create()};
  std::array<qfn::Session*, 2> used{reference.sessions[2].get(),
                                    reference.sessions[0].get()};
  std::vector<sampling::SamplerState> fresh_samplers, used_samplers;
  for (std::size_t k = 0; k < 2; ++k) {
    Require(fresh[k]->RestoreSnapshot(*snapshots[k], &error) &&
                used[k]->RestoreSnapshot(*snapshots[k], &error),
            error);
    const auto tokens = fresh[k]->Tokens();
    const std::vector<sampling::TokenId> history(tokens.begin(), tokens.end());
    fresh_samplers.emplace_back(members[k * 2].config, history);
    used_samplers.emplace_back(members[k * 2].config, history);
  }
  const auto restored_small = fresh[0]->Statistics().small_head_cycles;
  for (std::size_t c = 0; c < 4; ++c) {
    std::array<qfn::Session::DecodeResult, 2> expected, actual;
    std::vector<qfn::Session::DecodeRequest> requests;
    for (std::size_t k = 0; k < 2; ++k) {
      Require(fresh[k]->DecodeStep(8, fresh_samplers[k], &expected[k], &error,
                                   false),
              error);
      requests.push_back({used[k], 8, &used_samplers[k], &actual[k], false});
    }
    Require(qfn::Session::DecodeBatch(requests, &error), error);
    for (std::size_t k = 0; k < 2; ++k) {
      const auto& a = fresh[k]->Statistics();
      const auto& b = used[k]->Statistics();
      Require(
          expected[k].tokens == actual[k].tokens &&
              fresh_samplers[k].rng_state() == used_samplers[k].rng_state() &&
              a.drafted == b.drafted && a.accepted == b.accepted &&
              a.small_head_cycles == b.small_head_cycles,
          "restore into a used session differs from a fresh one");
      RequireExact(fresh[k]->Logits(), used[k]->Logits(),
                   "restore into a used session changed logits");
    }
  }
  Require(fresh[0]->Statistics().small_head_cycles > restored_small,
          "the restored English member left the small head");

  // Alone, plain greedy drafts through the small head on two sessions as on
  // one; beside a sampled peer it keeps the full head and makes the same
  // text; a jointly planned greedy pair uses the small head.
  const auto english = chat("Count from one to thirty in words.", false);
  std::array<std::unique_ptr<qfn::Session>, 3> greedy{create(), create(),
                                                      create()};
  std::vector<sampling::SamplerState> greedy_samplers;
  for (auto& session : greedy) {
    Require(session->Sync(english, &error), error);
    greedy_samplers.emplace_back(
        sampling::SamplingConfig{},
        std::vector<sampling::TokenId>(english.begin(), english.end()));
  }
  std::vector<std::int32_t> alone_tokens, mixed_tokens;
  for (std::size_t c = 0; c < 4; ++c) {
    qfn::Session::DecodeResult alone;
    Require(greedy[0]->DecodeStep(8, greedy_samplers[0], &alone, &error, false),
            error);
    alone_tokens.insert(alone_tokens.end(), alone.tokens.begin(),
                        alone.tokens.end());
  }
  const auto& alone = greedy[0]->Statistics();
  Require(alone.drafted > alone.lookup && alone.small_head_cycles > 0,
          "plain greedy kept the full head alone on two sessions");
  auto sampled_peer = create();
  Require(sampled_peer->Sync(members[0].prompt, &error), error);
  sampling::SamplerState sampled_peer_sampler(
      members[0].config,
      std::vector<sampling::TokenId>(members[0].prompt.begin(),
                                     members[0].prompt.end()));
  while (mixed_tokens.size() < alone_tokens.size()) {
    std::array<qfn::Session::DecodeResult, 2> results;
    std::array<qfn::Session::DecodeRequest, 2> requests{{
        {greedy[2].get(), 8, &greedy_samplers[2], &results[0], false},
        {sampled_peer.get(), 8, &sampled_peer_sampler, &results[1], false},
    }};
    Require(qfn::Session::DecodeBatch(requests, &error), error);
    Require(!results[0].tokens.empty(), "mixed greedy step made no progress");
    mixed_tokens.insert(mixed_tokens.end(), results[0].tokens.begin(),
                        results[0].tokens.end());
  }
  const auto& mixed = greedy[2]->Statistics();
  Require(mixed.drafted > mixed.lookup && mixed.small_head_cycles == 0,
          "plain greedy reached the small head beside a sampled peer");
  Require(std::equal(alone_tokens.begin(), alone_tokens.end(),
                     mixed_tokens.begin()),
          "the small head changed greedy text");
  auto peer = create();
  Require(peer->Sync(english, &error), error);
  sampling::SamplerState peer_sampler(
      sampling::SamplingConfig{},
      std::vector<sampling::TokenId>(english.begin(), english.end()));
  for (std::size_t c = 0; c < 4; ++c) {
    std::array<qfn::Session::DecodeResult, 2> results;
    std::array<qfn::Session::DecodeRequest, 2> requests{{
        {greedy[1].get(), 8, &greedy_samplers[1], &results[0], false},
        {peer.get(), 8, &peer_sampler, &results[1], false},
    }};
    Require(qfn::Session::DecodeBatch(requests, &error), error);
  }
  Require(greedy[1]->Statistics().small_head_cycles > 0,
          "a jointly planned greedy pair left the small head");
  std::cout << "sampled draft head: two sessions small_head_cycles="
            << small_cycles[0] << "," << small_cycles[1] << ","
            << small_cycles[2] << "," << small_cycles[3]
            << " thai_full_head_drafts=" << reference.full_head_drafts[2]
            << " overlap all_small=" << overlap_use.all_small
            << " mixed=" << overlap_use.mixed
            << " reordered all_small=" << reordered_use.all_small
            << " mixed=" << reordered_use.mixed
            << " schedules_and_restores_exact=1 greedy_alone_small_head="
            << alone.small_head_cycles << " greedy_text_exact=1\n"
            << std::flush;
}

void CheckExecutionModes(const std::shared_ptr<qfn::Model>& model,
                         std::uint32_t prompt_tokens = 4096,
                         bool with_image = false) {
  using gufo::core::SessionMode;
  namespace vision = gufo::models::qwen::vision;
  std::string error;
  auto ar = model->CreateSession(SessionMode::kAutoregressive,
                                 prompt_tokens + 16, &error);
  auto mtp = model->CreateSession(SessionMode::kSpeculative, prompt_tokens + 16,
                                  &error);
  Require(ar && mtp, error);
  Require(ar->AllocatedBytes() < mtp->AllocatedBytes(),
          "AR allocated predictor state");
  const auto pattern = model->Tokenize(
      "Review this transaction log and identify the failed request.\n"
      "def retry(items):\n"
      "    return [item for item in items if item['status'] != 200]\n"
      "{\"request\":\"tools.read\",\"status\":503,\"retry_after\":2.5}\n"
      "The train travels 60 km in 45 minutes. Explain the calculation.\n"
      "Café, e\u0301, 日本語, العربية. Keep Unicode and literal <tags> "
      "intact.\n");
  Require(!pattern.empty(), "empty execution-mode fixture");
  std::vector<std::int32_t> prompt(prompt_tokens);
  for (std::size_t i = 0; i < prompt.size(); ++i)
    prompt[i] = pattern[i % pattern.size()];
  if (with_image) {
    Require(model->VisionEncoder() != nullptr && prompt_tokens >= 4,
            "image prefill requires its encoder and four image tokens");
    auto image = std::make_shared<vision::Prompt>();
    const vision::ImageGrid grid{0, 2, 2};
    std::fill_n(prompt.begin(), 4, vision::kImageToken);
    image->tokens.assign(prompt.begin(), prompt.end());
    image->rope.images.push_back(grid);
    gufo::core::Image pixels;
    pixels.width = pixels.height = 64;
    pixels.pixels.resize(64 * 64 * 3);
    for (std::size_t i = 0; i < pixels.pixels.size(); ++i)
      pixels.pixels[i] = static_cast<std::uint8_t>((i * 37 + i / 64) % 256);
    image->images.push_back(
        {std::make_shared<const gufo::core::Image>(std::move(pixels)), grid});
    image->images.back().prefix_identity.fill(3);
    image->cache_identity.assign(32, 3);
    ar->ConfigureVision(image);
    mtp->ConfigureVision(image);
  }
  Require(ar->Sync(prompt, &error) && mtp->Sync(prompt, &error), error);
  RequireExact(ar->Logits(), mtp->Logits(),
               "execution mode changes target prefill logits");
  const auto ar_snapshot = ar->SaveSnapshot(&error);
  const auto mtp_snapshot = mtp->SaveSnapshot(&error);
  Require(ar_snapshot && mtp_snapshot &&
              ar_snapshot->SizeBytes() < mtp_snapshot->SizeBytes(),
          "AR snapshot includes predictor state");
  const auto anchor = static_cast<std::int32_t>(
      std::max_element(ar->Logits().begin(), ar->Logits().end()) -
      ar->Logits().begin());
  const std::array<qfn::Session::AdvanceRequest, 2> mixed{
      {{ar.get(), anchor}, {mtp.get(), anchor}}};
  Require(qfn::Session::EvaluateBatch(mixed, &error), error);
  RequireExact(ar->Logits(), mtp->Logits(),
               "mixed execution modes contaminate target logits");
  const sampling::SamplingConfig config{
      .temperature = 0.8F, .top_k = 20, .top_p = 0.95F, .seed = 73};
  sampling::SamplerState sampler(config);
  qfn::Session::DecodeResult step;
  Require(ar->DecodeStep(8, sampler, &step, &error, false), error);
  Require(step.tokens.size() == 1 && ar->Statistics().drafted == 0,
          "AR session with a resident sidecar executed speculative decoding");
  const std::vector<float> expected(ar->Logits().begin(), ar->Logits().end());
  for (bool serialized : {false, true}) {
    if (serialized) {
      std::vector<std::uint8_t> bytes(ar_snapshot->SizeBytes());
      Require(ar_snapshot->CopyTo(bytes), "serialize AR frontier");
      Require(ar->RestoreSnapshot(bytes, &error), error);
    } else {
      Require(ar->RestoreSnapshot(*ar_snapshot, &error), error);
    }
    Require(ar->Evaluate(anchor, &error), error);
    sampling::SamplerState replay(config);
    qfn::Session::DecodeResult resumed;
    Require(ar->DecodeStep(8, replay, &resumed, &error, false), error);
    Require(resumed.tokens == step.tokens &&
                replay.rng_state() == sampler.rng_state(),
            "AR frontier changed sampled replay");
    RequireExact(expected, ar->Logits(),
                 "AR frontier changed resumed target logits");
  }
  Require(!mtp->RestoreSnapshot(*ar_snapshot, &error) &&
              !ar->RestoreSnapshot(*mtp_snapshot, &error),
          "snapshots crossed execution modes");
  std::cout << "execution_tokens=" << prompt_tokens << " image=" << with_image
            << " execution_modes=independent AR_predictor_bytes=0 "
               "mixed_batch_exact=1 RAM_serialized_sampled_replay=1\n"
            << std::flush;
}

void CheckServingEos(const std::shared_ptr<qfn::Model>& model) {
  namespace server = gufo::server;
  using Backend = server::InferenceBackend;
  Backend backend;
  std::string error;
  server::TextSpeculativeConfig options;
  options.backend = server::TextSpeculativeBackend::kMtp;
  Require(backend.load(model, &error, 128, 2, {}, {}, options), error);
  // Synthetic grammar pieces make the boundary deterministic: any ordinary
  // first token completes the empty JSON object; only stop tokens may follow.
  // Actual token IDs, EOS recognition, GPU execution and checkpoints stay real.
  auto constraint = std::make_shared<sampling::TokenConstraint>();
  constraint->grammar = sampling::JsonConstraint::Compile(
      gufo::json::parse(
          R"({"type":"object","properties":{},"additionalProperties":false})"),
      true);
  constraint->vocabulary = std::make_shared<sampling::ConstraintVocabulary>(
      model->VocabSize(), [&](std::uint32_t token) {
        return sampling::ConstraintVocabulary::Piece{
            .text = model->IsStopToken(token) ? "" : "{}",
            .stop = model->IsStopToken(token),
        };
      });
  sampling::SamplingConfig config;
  config.constraint = std::move(constraint);
  config.logprobs = 2;
  const std::array<std::string, 2> prompts{
      "EOS fixture one: continue red, blue,",
      "EOS fixture two: continue green, yellow,"};
  std::array<Backend::Result, 2> references;
  for (std::size_t row = 0; row < prompts.size(); ++row) {
    references[row] = backend.complete(prompts[row], 8, config);
    Require(references[row].logprobs.size() == 2 &&
                model->IsStopToken(
                    references[row].logprobs.back().entries.front().id),
            "constrained EOS must retain its raw token probability");
    Require(
        references[row].tokens.size() == 1 &&
            !model->IsStopToken(references[row].tokens.front()),
        "scalar MTP executed EOS or trailing work after the forced boundary");
    const auto replay = backend.complete(prompts[row], 8, config);
    Require(
        replay.tokens == references[row].tokens && replay.prefill_tokens == 0,
        "scalar EOS checkpoint did not replay exactly without prefill");
  }
  std::array<std::future<Backend::Result>, 2> pending;
  std::latch ready(pending.size());
  for (std::size_t row = 0; row < pending.size(); ++row) {
    pending[row] = std::async(
        std::launch::async, [&backend, &ready, &prompts, &config, row] {
          ready.arrive_and_wait();
          return backend.complete(prompts[row], 8, config);
        });
  }
  std::size_t batched = 0;
  for (std::size_t row = 0; row < pending.size(); ++row) {
    const auto result = pending[row].get();
    Require(result.logprobs.size() == 2 &&
                model->IsStopToken(result.logprobs.back().entries.front().id),
            "batched cached EOS must retain token metadata");
    Require(result.tokens == references[row].tokens,
            "batched EOS accounting differs from the scalar checkpoint");
    batched += result.physical_execution_width > 1;
  }
  Require(batched > 0, "EOS fixture did not exercise batched MTP");
  // Completions explicitly ignoring EOS must retain their requested budget.
  const auto ignored_request =
      backend.start_complete(prompts.front(), 8, config, {}, false, true);
  const auto ignored = ignored_request->Wait({}, {});
  Require(
      ignored.tokens.size() == 8 &&
          std::all_of(ignored.tokens.begin() + 1, ignored.tokens.end(),
                      [&](auto token) { return model->IsStopToken(token); }),
      "MTP ignored the request's explicit ignore_eos policy");
  std::latch mixed_ready(pending.size());
  for (std::size_t row = 0; row < pending.size(); ++row) {
    pending[row] = std::async(
        std::launch::async, [&backend, &mixed_ready, &prompts, &config, row] {
          mixed_ready.arrive_and_wait();
          const auto request = backend.start_complete(prompts[row], 8, config,
                                                      {}, false, row == 1);
          return request->Wait({}, {});
        });
  }
  const auto stopped = pending[0].get();
  const auto continuing = pending[1].get();
  Require(
      stopped.tokens == references[0].tokens && continuing.tokens.size() == 8 &&
          std::all_of(continuing.tokens.begin() + 1, continuing.tokens.end(),
                      [&](auto token) { return model->IsStopToken(token); }) &&
          (stopped.physical_execution_width > 1 ||
           continuing.physical_execution_width > 1),
      "mixed MTP batch did not respect each row's EOS policy");
  std::cout << "serving_eos scalar_and_batch_exact=1 checkpoint_replay=1 "
               "ignore_eos_budget=8 batched_requests="
            << batched << '\n'
            << std::flush;
}

void CheckServingSampling(const std::shared_ptr<qfn::Model>& model) {
  namespace server = gufo::server;
  using Backend = server::InferenceBackend;
  Backend ar, mtp;
  std::string error;
  Require(ar.load(model, &error, 6145, 2), error);
  server::TextSpeculativeConfig options;
  options.backend = server::TextSpeculativeBackend::kMtp;
  Require(mtp.load(model, &error, 6145, 2, {}, {}, options), error);
  struct Reference {
    gufo::test::SamplingCase test;
    std::string prompt;
    Backend::Result ar, mtp;
  };
  std::vector<Reference> references;
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    const std::string prompt = "Sampling " + std::string(test.name) +
                               ": Continue red, blue, blue, red,";
    auto ordinary = ar.complete(prompt, 8, test.config);
    auto speculative = mtp.complete(prompt, 8, test.config);
    // Exercise the serving handoff against the model session directly.
    // Replaying HTTP alone would not detect a consistently dropped residual.
    auto direct = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        6145, &error);
    const auto encoded = model->Tokenize(prompt);
    Require(direct && direct->Sync(encoded, &error), error);
    const std::vector<sampling::TokenId> history(encoded.begin(),
                                                 encoded.end());
    sampling::SamplerState direct_sampler(test.config, history);
    std::vector<sampling::TokenId> direct_tokens;
    while (direct_tokens.size() < 8) {
      qfn::Session::DecodeResult step;
      Require(direct->DecodeStep(8 - direct_tokens.size(), direct_sampler,
                                 &step, &error),
              error);
      direct_tokens.insert(direct_tokens.end(), step.tokens.begin(),
                           step.tokens.end());
      if (step.stop)
        break;
      Require(!step.tokens.empty(),
              "direct serving reference made no progress");
    }
    const auto direct_stats = direct->Statistics();
    Require(direct_tokens == speculative.tokens &&
                direct_stats.drafted == speculative.draft_tokens &&
                direct_stats.accepted == speculative.draft_accepted_tokens,
            "serving lost sampling state: " + std::string(test.name));
    Require(!ordinary.tokens.empty() && ordinary.draft_tokens == 0 &&
                speculative.draft_tokens > 0,
            "sampling strategy did not exercise AR and MTP");
    Require(ordinary.physical_execution_width == 1 &&
                speculative.physical_execution_width == 1,
            "serial completion reported a batched execution");
    if (!test.config.uses_random_sampling()) {
      Require(ordinary.tokens == speculative.tokens,
              "serving greedy AR/MTP mismatch: " + std::string(test.name));
    }
    for (const bool use_mtp : {false, true}) {
      auto& backend = use_mtp ? mtp : ar;
      const auto& expected = use_mtp ? speculative : ordinary;
      const auto replay = backend.complete(prompt, 8, test.config);
      Require(
          replay.tokens == expected.tokens &&
              replay.draft_tokens == expected.draft_tokens &&
              replay.draft_accepted_tokens == expected.draft_accepted_tokens,
          "serving replay changed sampling or draft acceptance");
    }
    references.push_back(
        {test, prompt, std::move(ordinary), std::move(speculative)});
    std::cout << "serving sampling=" << test.name << " replay_exact=1\n"
              << std::flush;
  }
  // Every strategy also runs with shared projections in a concurrent pair.
  for (const bool use_mtp : {false, true}) {
    auto& backend = use_mtp ? mtp : ar;
    std::size_t batched_requests = 0;
    for (std::size_t offset = 0; offset < references.size(); offset += 2) {
      std::array<std::future<Backend::Result>, 2> pending;
      std::latch ready(pending.size());
      for (std::size_t row = 0; row < pending.size(); ++row) {
        const auto* saved = &references[(offset + row) % references.size()];
        pending[row] =
            std::async(std::launch::async, [&backend, &ready, saved] {
              ready.arrive_and_wait();
              return backend.complete(saved->prompt, 8, saved->test.config);
            });
      }
      for (std::size_t row = 0; row < pending.size(); ++row) {
        const auto& saved = references[(offset + row) % references.size()];
        const auto& expected = use_mtp ? saved.mtp : saved.ar;
        const auto actual = pending[row].get();
        Require(actual.tokens == expected.tokens &&
                    (!saved.test.config.uses_random_sampling() ||
                     (actual.draft_tokens == expected.draft_tokens &&
                      actual.draft_accepted_tokens ==
                          expected.draft_accepted_tokens)),
                "interleaving changed serving sampling or acceptance");
        if (actual.physical_execution_width > 1) {
          Require(actual.physical_execution_width == 2 &&
                      actual.execution_plan == "batched-w2",
                  "paired completion reported an incorrect execution plan");
          ++batched_requests;
        }
      }
    }
    Require(batched_requests > 0,
            "concurrent sampling checks did not report batched execution");
    // Greedy/AR prefixes are budget-independent. Sampled MTP consumes
    // proposal/rejection draws, so each budget must replay its own result.
    for (const auto index : {std::size_t{0}, references.size() - 2}) {
      const auto& saved = references[index];
      for (const auto budget : {1U, 2U, 3U}) {
        const auto actual =
            backend.complete(saved.prompt, budget, saved.test.config);
        if (use_mtp && saved.test.config.uses_random_sampling()) {
          const auto replay =
              backend.complete(saved.prompt, budget, saved.test.config);
          Require(
              actual.tokens.size() <= budget &&
                  actual.tokens == replay.tokens &&
                  actual.draft_tokens == replay.draft_tokens &&
                  actual.draft_accepted_tokens == replay.draft_accepted_tokens,
              "sampled MTP budget or replay differs");
        } else {
          Require(actual.tokens.size() == std::min<std::size_t>(
                                              budget, saved.ar.tokens.size()) &&
                      std::equal(actual.tokens.begin(), actual.tokens.end(),
                                 saved.ar.tokens.begin()),
                  "output budget changed the AR/greedy prefix");
        }
      }
    }
    std::cout << "serving strategies=" << references.size()
              << " mtp=" << use_mtp << " batched_requests=" << batched_requests
              << " C2_and_budgets_exact=1\n"
              << std::flush;
  }
}

void CheckLogprobs(const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  const auto prompt = model->Tokenize(
      "Continue this list: red blue green red blue green red blue green");
  const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  for (const bool mtp : {false, true}) {
    for (const float temperature : {0.F, 0.7F}) {
      const auto mode = mtp ? gufo::core::SessionMode::kSpeculative
                            : gufo::core::SessionMode::kAutoregressive;
      auto observed = model->CreateSession(mode, 256, &error);
      auto plain = model->CreateSession(mode, 256, &error);
      auto reference = model->CreateSession(
          gufo::core::SessionMode::kAutoregressive, 256, &error);
      Require(observed && plain && reference, error);
      Require(observed->Sync(prompt, &error) && plain->Sync(prompt, &error) &&
                  reference->Sync(prompt, &error),
              error);
      sampling::SamplingConfig config;
      config.temperature = temperature;
      config.top_k = 20;
      config.seed = 73;
      sampling::SamplerState a(config, history);
      config.logprobs = 5;
      sampling::SamplerState b(config, history);
      std::size_t generated = 0;
      while (generated < 32) {
        qfn::Session::DecodeResult expected, actual;
        const auto budget = std::min<std::size_t>(mtp ? 8 : 1, 32 - generated);
        Require(plain->DecodeStep(budget, a, &expected, &error, false) &&
                    observed->DecodeStep(budget, b, &actual, &error, false),
                error);
        Require(
            expected.tokens == actual.tokens && a.rng_state() == b.rng_state(),
            "logprobs changed selected tokens or RNG");
        Require(actual.logprobs.size() == actual.tokens.size(),
                "missing accepted-token probabilities");
        for (std::size_t i = 0; i < actual.tokens.size(); ++i) {
          const auto logits = reference->Logits();
          const double maximum =
              *std::max_element(logits.begin(), logits.end());
          double sum = 0;
          for (const float value : logits)
            sum += std::exp(double(value) - maximum);
          const auto check = [&](const sampling::TokenLogprob& entry) {
            const double expected_lp =
                (double(logits[entry.token]) - maximum) - std::log(sum);
            Require(std::abs(entry.logprob - expected_lp) < 1e-5,
                    "reported probability disagrees with teacher-forced target "
                    "logits");
          };
          Require(actual.logprobs[i].selected.token ==
                      static_cast<sampling::TokenId>(actual.tokens[i]),
                  "probability token order differs");
          check(actual.logprobs[i].selected);
          for (const auto& entry : actual.logprobs[i].top)
            check(entry);
          Require(reference->Evaluate(actual.tokens[i], &error), error);
        }
        generated += actual.tokens.size();
      }
      const auto x = observed->Statistics(), y = plain->Statistics();
      Require(x.drafted == y.drafted && x.accepted == y.accepted,
              "observation changed speculative acceptance");
      if (mtp)
        Require(x.drafted > 0, "logprob fixture must execute drafts");
      std::cout << "logprobs mtp=" << mtp << " temperature=" << temperature
                << " tokens=" << generated << " drafted=" << x.drafted << '\n';
    }
  }
}

int main(int argc, char** argv) {
  const bool logprobs_only =
      argc == 6 && std::string_view(argv[5]) == "--logprobs-only";
  const bool lookup_only =
      argc == 6 && std::string_view(argv[5]) == "--lookup-only";
  const bool execution_only =
      argc == 7 && std::string_view(argv[5]) == "--execution-only";
  std::uint32_t execution_tokens = 4096;
  const bool batch_only =
      argc == 6 && std::string_view(argv[5]) == "--batch-only";
  const bool prefill_only =
      argc == 6 && std::string_view(argv[5]) == "--prefill-only";
  const bool sampling_only =
      argc == 6 && std::string_view(argv[5]) == "--sampling-only";
  const bool cache_only =
      argc == 6 && std::string_view(argv[5]) == "--cache-only";
  const bool eos_only = argc == 6 && std::string_view(argv[5]) == "--eos-only";
  if ((argc != 5 && !batch_only && !prefill_only && !sampling_only &&
       !cache_only && !eos_only && !execution_only && !lookup_only &&
       !logprobs_only) ||
      std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr
        << "Usage: session_test --model FIRST.gguf --mtp-model MTP.gguf "
           "[--batch-only | --prefill-only | --sampling-only | "
           "--cache-only | --eos-only | --lookup-only | --logprobs-only | "
           "--execution-only TOKENS]\n";
    return 77;
  }
  if (execution_only) {
    const std::string_view value(argv[6]);
    const auto [end, ec] = std::from_chars(
        value.data(), value.data() + value.size(), execution_tokens);
    if (ec != std::errc{} || end != value.data() + value.size() ||
        execution_tokens == 0 || execution_tokens > 262128) {
      std::cerr << "--execution-only requires 1..262128 prompt tokens\n";
      return 2;
    }
  }
  try {
    std::string error;
    auto model = qfn::Model::Load(
        argv[2],
        {.max_context = execution_only ? execution_tokens + 16 : 6145,
         .mtp_model_path = argv[4],
         .max_draft_tokens = 7,
         .decode_concurrency = 8},
        &error);
    Require(model != nullptr, error);
    if (logprobs_only) {
      CheckLogprobs(model);
      CheckServingEos(model);
      return 0;
    }
    if (lookup_only) {
      CheckLookupBatch(model);
      return 0;
    }
    if (execution_only) {
      CheckExecutionModes(model, execution_tokens);
      return 0;
    }
    if (eos_only) {
      CheckServingEos(model);
      return 0;
    }
    if (cache_only) {
      CheckMtpCacheReplay(model);
      CheckDraftPolicySnapshots(model);
      return 0;
    }
    CheckSnapshotDuringGraphCapture(model, false);
    CheckSnapshotDuringGraphCapture(model, true);
    CheckSnapshotDuringGraphCapture(model, true, true);
    CheckExecutionModes(model);
    if (sampling_only) {
      CheckServingEos(model);
      CheckServingSampling(model);
      return 0;
    }
    if (!batch_only) {
      CheckExecutionModes(model, 2048, true);
      CheckPrefillChunks(model);
      CheckPrefetchHints(model);
    }
    if (prefill_only)
      return 0;
    if (!batch_only) {
      CheckMtpCacheReplay(model);
      CheckDraftPolicySnapshots(model);
    }
    CheckFailureRecovery(model);
    CheckRollbackReuse(model);
    CheckImageSnapshotAttachment(model);
    CheckBatchFailureIsolation(model);
    CheckBatchedSessions(model);
    CheckLookupBatch(model);
    // The checks above batch up to eight sessions; the draft-vocabulary
    // lifecycle needs one (see CheckDraftVocabLifecycle) and the sampled
    // draft-head check two.
    const auto load = [&](std::uint32_t decode_concurrency) {
      model.reset();
      model = qfn::Model::Load(argv[2],
                               {.max_context = 6145,
                                .mtp_model_path = argv[4],
                                .max_draft_tokens = 7,
                                .decode_concurrency = decode_concurrency},
                               &error);
      Require(model != nullptr, error);
    };
    load(1);
    CheckDraftVocabLifecycle(model);
    load(2);
    CheckSampledDraftHead(model);
    if (batch_only)
      return 0;
    load(8);
    CheckServingEos(model);
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty prompt pattern");
    std::vector<std::int32_t> prompt(4096);
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = pattern[i % pattern.size()];
    const std::array<sampling::SamplingConfig, 4> configs{{
        {.seed = 1},
        {.temperature = 0.7F, .seed = 1},
        {.temperature = 1.0F, .top_p = 0.95F, .seed = 73},
        {.temperature = 0.8F,
         .top_k = 40,
         .top_p = 0.9F,
         .min_p = 0.01F,
         .min_keep = 2,
         .seed = 1,
         .repeat_penalty = 1.1F,
         .repeat_last_n = 16,
         .frequency_penalty = 0.2F,
         .presence_penalty = 0.1F},
    }};
    struct Replay {
      unsigned depth;
      std::size_t config;
      std::vector<float> prefill_logits;
      std::vector<float> final_logits;
      std::vector<std::int32_t> tokens;
      std::uint64_t rng;
    };
    std::vector<Replay> replays;
    for (const auto depth : {16U, 4096U}) {
      const auto prefix = std::span(prompt).first(depth);
      const std::vector<sampling::TokenId> history(prefix.begin(),
                                                   prefix.end());
      for (std::size_t c = 0; c < configs.size(); ++c) {
        const bool sampled = configs[c].uses_random_sampling();
        auto ar = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                       6145, &error);
        auto mtp = model->CreateSession(
            model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                            : gufo::core::SessionMode::kAutoregressive,
            6145, &error);
        Require(ar && mtp, error);
        Require(ar->Sync(prefix, &error), error);
        // Prefill the other session before decoding either: retained target
        // hidden rows must remain private to their session.
        Require(mtp->Sync(prefix, &error), error);
        RequireExact(ar->Logits(), mtp->Logits(),
                     "interleaving changed prefill logits");
        Replay saved{depth, c,  {ar->Logits().begin(), ar->Logits().end()},
                     {},    {}, 0};
        sampling::SamplerState a(configs[c], history), b(configs[c], history);
        std::vector<std::int32_t> reference, candidate;
        std::vector<std::size_t> interleaved_widths;
        const std::size_t token_count = c <= 1 ? 32 : 8;
        while (candidate.size() < token_count) {
          qfn::Session::DecodeResult decoded;
          Require(mtp->DecodeStep(token_count - candidate.size(), b, &decoded,
                                  &error, false),
                  error);
          Require(!decoded.tokens.empty(), "empty MTP step");
          interleaved_widths.push_back(decoded.tokens.size());
          candidate.insert(candidate.end(), decoded.tokens.begin(),
                           decoded.tokens.end());
          while (reference.size() < candidate.size()) {
            if (sampled) {
              // Independent target replay checks every committed token's
              // recurrent/cache state, including residual corrections.
              const auto token = candidate[reference.size()];
              Require(ar->Evaluate(token, &error), error);
              reference.push_back(token);
              a.Accept(static_cast<sampling::TokenId>(token));
              continue;
            }
            qfn::Session::DecodeResult single;
            Require(ar->DecodeStep(1, a, &single, &error, false), error);
            reference.insert(reference.end(), single.tokens.begin(),
                             single.tokens.end());
          }
          double squared = 0.0;
          float max_diff = 0.0F;
          for (std::size_t i = 0; i < ar->Logits().size(); ++i) {
            Require(std::isfinite(ar->Logits()[i]) &&
                        std::isfinite(mtp->Logits()[i]),
                    "non-finite logits");
            const float d = ar->Logits()[i] - mtp->Logits()[i];
            squared += static_cast<double>(d) * d;
            max_diff = std::max(max_diff, std::abs(d));
          }
          std::cout << "frontier depth=" << depth << " config=" << c
                    << " tokens=" << candidate.size()
                    << " width=" << decoded.tokens.size()
                    << " max_diff=" << max_diff
                    << " rmse=" << std::sqrt(squared / ar->Logits().size())
                    << " ar_rng=" << a.rng_state()
                    << " mtp_rng=" << b.rng_state() << '\n';
          if (reference != candidate) {
            std::cout << "AR:";
            for (auto t : reference)
              std::cout << ' ' << t;
            std::cout << "\nMTP:";
            for (auto t : candidate)
              std::cout << ' ' << t;
            std::cout << '\n';
          }
          std::cout << std::flush;
          Require(reference == candidate,
                  "AR/MTP token mismatch at depth " + std::to_string(depth) +
                      " config " + std::to_string(c) + " token " +
                      std::to_string(reference.size()));
          Require(max_diff == 0.0F, "AR/MTP frontier logits differ");
          if (!sampled) {
            Require(a.rng_state() == b.rng_state(),
                    "greedy AR/MTP RNG mismatch");
          }
          Require(ar->Position() == mtp->Position(),
                  "AR/MTP position mismatch");
        }
        saved.final_logits.assign(ar->Logits().begin(), ar->Logits().end());
        saved.tokens = reference;
        saved.rng = b.rng_state();
        // Greedy and seeded sampling exercise fresh-load determinism; the
        // remaining configurations cover filters and penalties above.
        if (c <= 1)
          replays.push_back(std::move(saved));
        const auto stats = mtp->Statistics();
        Require(mtp->Sync(prefix, &error), error);
        Require(mtp->Position() == depth, "prefix reset failed");
        sampling::SamplerState replay_sampler(configs[c], history);
        std::vector<std::int32_t> replay;
        std::vector<std::size_t> isolated_widths;
        while (replay.size() < token_count) {
          qfn::Session::DecodeResult decoded;
          Require(mtp->DecodeStep(token_count - replay.size(), replay_sampler,
                                  &decoded, &error, false),
                  error);
          Require(!decoded.tokens.empty(), "empty replay step");
          isolated_widths.push_back(decoded.tokens.size());
          replay.insert(replay.end(), decoded.tokens.begin(),
                        decoded.tokens.end());
        }
        const auto after = mtp->Statistics();
        Require(replay == candidate && isolated_widths == interleaved_widths &&
                    after.drafted - stats.drafted == stats.drafted &&
                    after.accepted - stats.accepted == stats.accepted,
                "interleaving changed MTP proposals or acceptance");
        std::cout << "depth=" << depth << " config=" << c
                  << " tokens=" << token_count
                  << " exact=1 drafted=" << stats.drafted
                  << " accepted=" << stats.accepted << '\n'
                  << std::flush;
      }
    }
    // Recreate device weights, executor and plans. Reverse request order so
    // a previous shape cannot silently determine this request's arithmetic.
    model.reset();
    model = qfn::Model::Load(argv[2],
                             {.max_context = 6145,
                              .mtp_model_path = argv[4],
                              .max_draft_tokens = 7,
                              .decode_concurrency = 8},
                             &error);
    Require(model != nullptr, error);
    std::reverse(replays.begin(), replays.end());
    for (const auto& saved : replays) {
      const auto prefix = std::span(prompt).first(saved.depth);
      const std::vector<sampling::TokenId> history(prefix.begin(),
                                                   prefix.end());
      for (const bool speculative : {false, true}) {
        const bool sampled = configs[saved.config].uses_random_sampling();
        auto session = model->CreateSession(
            model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                            : gufo::core::SessionMode::kAutoregressive,
            6145, &error);
        Require(session && session->Sync(prefix, &error), error);
        RequireExact(saved.prefill_logits, session->Logits(),
                     "fresh load changed prefill logits");
        sampling::SamplerState sampler(configs[saved.config], history);
        std::vector<std::int32_t> tokens;
        constexpr std::array<std::size_t, 4> budgets{2, 8, 1, 4};
        std::size_t step = 0;
        while (tokens.size() < saved.tokens.size()) {
          if (sampled && !speculative) {
            const auto token = saved.tokens[tokens.size()];
            Require(session->Evaluate(token, &error), error);
            tokens.push_back(token);
            continue;
          }
          const auto budget = sampled ? saved.tokens.size() - tokens.size()
                              : speculative ? budgets[step++ % budgets.size()]
                                            : 1;
          qfn::Session::DecodeResult decoded;
          Require(session->DecodeStep(
                      std::min(budget, saved.tokens.size() - tokens.size()),
                      sampler, &decoded, &error, false),
                  error);
          Require(!decoded.tokens.empty(), "empty fresh-load replay step");
          tokens.insert(tokens.end(), decoded.tokens.begin(),
                        decoded.tokens.end());
        }
        Require(tokens == saved.tokens,
                "fresh load or draft width changed tokens");
        RequireExact(saved.final_logits, session->Logits(),
                     "fresh load or draft width changed final logits");
        if (!sampled || speculative) {
          Require(sampler.rng_state() == saved.rng,
                  "fresh load or draft width changed RNG");
        }
        std::cout << "fresh_load depth=" << saved.depth
                  << " config=" << saved.config << " mtp=" << speculative
                  << " tokens=" << tokens.size() << " bitwise_exact=1\n"
                  << std::flush;
      }
    }
    CheckServingSampling(model);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
