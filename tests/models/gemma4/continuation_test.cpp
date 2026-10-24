// Component-cache continuation of Gemma 4 sessions: a checkpoint captured
// through the adapter and restored into another slot, or into its own slot
// after the session wrapped its ring, reproduces the logits and greedy
// continuation of the session it came from.
//
// With GUFO_GEMMA4_MTP_MODEL, greedy MTP cycles after a restore also draft and
// accept exactly as before it: the drafter reads the restored hidden state.
//
// A checkpoint holds up to kContinuationImages image records; a session with
// more is not captured. A completion waited after its slot is destroyed
// returns the result the slot settled.
//
// Requires GUFO_GEMMA4_MODEL; exits 77 without it.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/cache/checkpoint.hpp"
#include "src/core/hip/transfer_pool.hpp"
#include "src/core/sampling.hpp"
#include "src/models/gemma4/continuation_adapter.hpp"
#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/kernels/rocm/executor.hpp"
#include "src/models/gemma4/vision/encoder.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/image_fixture.hpp"

namespace g4 = gufo::models::gemma4;
namespace cache = gufo::cache;
using gemma4_test::Require;

namespace {
class RecordingGuard final : public cache::MutationGuard {
public:
  struct Write {
    cache::ComponentId id;
    cache::Rows first, end;
  };
  void BeforeOverwrite(cache::ComponentId id, cache::Rows first,
                       cache::Rows end) override {
    writes.push_back({id, first, end});
    count.fetch_add(1, std::memory_order_release);
  }
  void BeforeRelease(cache::ComponentId, cache::Rows,
                     cache::Rows) noexcept override {}
  std::vector<Write> writes;
  // Readable from other threads while the slot executes.
  std::atomic<std::size_t> count{0};
};

struct Fixture {
  cache::ResourceLedger ledger{
      cache::ResourceLimits{1ULL << 36, 1ULL << 36, 1ULL << 36, 1ULL << 36}};
  g4::ContinuationAdapter& adapter;
  gufo::hip::TransferPool& pool;
  // Registered host buffers, as serving's backing pool provides: the
  // adapter gathers their copies in one launch. Pageable ones copy singly.
  bool pinned{false};

  // Each transfer seals its pooled stream; it returns to the pool once the
  // stream and its completion are both released.
  void Settle(const std::function<cache::Completion(cache::Stream&)>& submit) {
    auto stream = pool.TryAcquire();
    Require(stream != nullptr, "no transfer stream");
    Require(submit(*stream).Wait() == cache::TransferResult::kSucceeded,
            "transfer failed");
  }

  cache::Payload Save(cache::Slot& slot, const cache::PayloadRequest& request) {
    auto pool =
        ledger.Reserve(cache::ResourceCategory::kBackingFree, request.bytes)
            .Convert();
    auto assigned = pool.ReserveBacking(request.category);
    std::shared_ptr<std::vector<std::byte>> buffer(
        new std::vector<std::byte>(request.bytes), [](auto* bytes) {
          if (hipHostUnregister(bytes->data()) != hipSuccess) {
            // Pageable buffer: nothing was registered.
          }
          delete bytes;
        });
    if (pinned && !buffer->empty())
      Require(hipHostRegister(buffer->data(), buffer->size(),
                              hipHostRegisterDefault) == hipSuccess,
              "host registration failed");
    Settle([&](cache::Stream& stream) {
      return request.category == cache::ResourceCategory::kPrivateState
                 ? adapter.CapturePrivate(slot, request.component, *buffer,
                                          stream)
                 : adapter.CopyRowsOut(slot, request.component, request.first,
                                       request.end, *buffer, stream);
    });
    return cache::Payload::Committed(assigned.Convert(), buffer);
  }
  // Private state optionally arrives in two pieces, as disk restores send it.
  void Restore(cache::Slot& slot, const cache::Checkpoint& checkpoint,
               bool pieces) {
    std::vector<cache::ComponentPosition> positions;
    for (const auto& c : checkpoint.Components())
      positions.push_back(c.position);
    adapter.BeginRestore(slot, positions);
    const auto bytes = [](const auto& payload) {
      return std::span<const std::byte>(
          *std::static_pointer_cast<const std::vector<std::byte>>(
              payload.Owner()));
    };
    for (const auto& c : checkpoint.Components()) {
      const auto id = c.descriptor.id;
      for (const auto& chunk : c.chunks)
        Settle([&](cache::Stream& stream) {
          return adapter.CopyRowsIn(slot, id, chunk.First(), chunk.End(),
                                    bytes(chunk.Storage()), stream);
        });
      if (c.tail) {
        const auto end = c.position.valid_rows;
        Settle([&](cache::Stream& stream) {
          return adapter.CopyRowsIn(slot, id,
                                    end - end % c.descriptor.rows_per_chunk,
                                    end, bytes(*c.tail), stream);
        });
      }
      if (c.private_state) {
        const auto all = bytes(*c.private_state);
        const auto half = pieces ? all.size() / 2 : 0;
        if (half)
          Settle([&](cache::Stream& stream) {
            return adapter.LoadPrivatePiece(slot, id, half, all.subspan(half),
                                            stream);
          });
        Settle([&](cache::Stream& stream) {
          return pieces ? adapter.LoadPrivatePiece(slot, id, 0, all.first(half),
                                                   stream)
                        : adapter.LoadPrivate(slot, id, all, stream);
        });
      }
    }
    Require(adapter.Validate(slot, positions), "restore did not validate");
  }
  // In-place restore: keeps the rows ResidentRows reports and loads the rest
  // around them. Returns the bytes loaded.
  std::size_t RestoreKeeping(cache::Slot& slot,
                             const cache::Checkpoint& checkpoint,
                             std::vector<cache::ResidentRange>* kept_out) {
    std::vector<cache::ComponentPosition> positions;
    for (const auto& c : checkpoint.Components())
      positions.push_back(c.position);
    std::vector<cache::ResidentRange> kept;
    for (auto range : adapter.ResidentRows(slot, positions)) {
      const auto& c = *std::ranges::find_if(
          checkpoint.Components(),
          [&](const auto& x) { return x.descriptor.id == range.id; });
      range.first =
          std::max(range.first,
                   cache::FirstChunkRow(c.descriptor, c.position.valid_rows));
      if (range.first < range.end)
        kept.push_back(range);
    }
    adapter.BeginRestore(slot, positions, kept);
    std::size_t loaded = 0;
    for (const auto& c : checkpoint.Components()) {
      const auto id = c.descriptor.id;
      const auto k = std::ranges::find(kept, id, &cache::ResidentRange::id);
      const auto row = c.descriptor.row_bytes;
      const auto around = [&](const cache::Payload& payload, cache::Rows first,
                              cache::Rows end) {
        const auto& bytes =
            *std::static_pointer_cast<const std::vector<std::byte>>(
                payload.Owner());
        for (const auto& [a, b] :
             {std::pair{first, k == kept.end() ? end : std::min(end, k->first)},
              std::pair{k == kept.end() ? end : std::max(first, k->end),
                        end}}) {
          if (a >= b)
            continue;
          Settle([&](cache::Stream& stream) {
            return adapter.CopyRowsIn(
                slot, id, a, b,
                std::span(bytes).subspan((a - first) * row, (b - a) * row),
                stream);
          });
          loaded += (b - a) * row;
        }
      };
      for (const auto& chunk : c.chunks)
        around(chunk.Storage(), chunk.First(), chunk.End());
      if (c.tail) {
        const auto end = c.position.valid_rows;
        around(*c.tail, end - end % c.descriptor.rows_per_chunk, end);
      }
      if (c.private_state) {
        const auto& bytes =
            *std::static_pointer_cast<const std::vector<std::byte>>(
                c.private_state->Owner());
        Settle([&](cache::Stream& stream) {
          return adapter.LoadPrivate(slot, id, bytes, stream);
        });
        loaded += bytes.size();
      }
    }
    Require(adapter.Validate(slot, positions), "in-place restore failed");
    if (kept_out != nullptr)
      *kept_out = std::move(kept);
    return loaded;
  }
  // Captures `slot` and requires every chunk, tail and private state to equal
  // `original` byte for byte.
  void RequireSameState(cache::Slot& slot, const cache::Checkpoint& original) {
    auto history = cache::ExecutionHistory::Cold(
        ledger, adapter.Components(), adapter.CompatibilityIdentity());
    const auto again = history.Capture(
        {original.Tokens(), cache::InputIdentity(original.Tokens().size()),
         adapter.Positions(slot), cache::CheckpointPurpose::kPrompt, 1},
        [&](const cache::PayloadRequest& request) {
          return Save(slot, request);
        });
    const auto bytes = [](const cache::Payload& payload) {
      return *std::static_pointer_cast<const std::vector<std::byte>>(
          payload.Owner());
    };
    const auto a = original.Components(), b = again->Components();
    Require(a.size() == b.size(), "recaptured component count differs");
    for (std::size_t i = 0; i < a.size(); ++i) {
      Require(a[i].chunks.size() == b[i].chunks.size() &&
                  bool(a[i].tail) == bool(b[i].tail) &&
                  bool(a[i].private_state) == bool(b[i].private_state),
              "recaptured component layout differs");
      for (std::size_t k = 0; k < a[i].chunks.size(); ++k)
        Require(a[i].chunks[k].First() == b[i].chunks[k].First() &&
                    bytes(a[i].chunks[k].Storage()) ==
                        bytes(b[i].chunks[k].Storage()),
                "recaptured rows differ in component " + std::to_string(i));
      if (a[i].tail)
        Require(bytes(*a[i].tail) == bytes(*b[i].tail),
                "recaptured tail differs in component " + std::to_string(i));
      if (a[i].private_state)
        Require(bytes(*a[i].private_state) == bytes(*b[i].private_state),
                "recaptured private state differs in component " +
                    std::to_string(i));
    }
  }
};

struct Continuation {
  std::vector<g4::TokenId> tokens;
  std::vector<std::vector<float>> logits;
  bool operator==(const Continuation&) const = default;
};
// Greedy decoding, recording the logits each token was chosen from.
Continuation Greedy(g4::Session& session, int steps) {
  Continuation result;
  std::string error;
  for (int i = 0; i < steps; ++i) {
    const auto logits = session.Logits();
    result.logits.emplace_back(logits.begin(), logits.end());
    const auto token = static_cast<g4::TokenId>(
        std::ranges::max_element(logits) - logits.begin());
    result.tokens.push_back(token);
    Require(session.Evaluate(token, &error), error);
  }
  return result;
}
// Each cycle's emitted tokens, drafts and accepted drafts.
struct Cycles {
  std::vector<std::vector<g4::TokenId>> tokens;
  std::vector<std::uint64_t> drafted, accepted;
  bool operator==(const Cycles&) const = default;
};
Cycles Speculate(g4::Session& session, int cycles) {
  Cycles result;
  gufo::sampling::SamplerState sampler({}, {});
  std::string error;
  for (int i = 0; i < cycles; ++i) {
    const auto before = session.Statistics();
    g4::Session::DecodeResult decoded;
    Require(session.DecodeStep(16, sampler, &decoded, &error), error);
    result.tokens.push_back(decoded.tokens);
    result.drafted.push_back(session.Statistics().drafted - before.drafted);
    result.accepted.push_back(session.Statistics().accepted - before.accepted);
  }
  return result;
}
std::vector<g4::TokenId> LongPrompt(const g4::Model& model,
                                    std::size_t tokens) {
  std::string text = "Notes from the harbour office.";
  std::vector<g4::TokenId> result;
  for (int line = 0; result.size() < tokens; ++line) {
    text += " Ship " + std::to_string(line * 37 % 1000) + " left berth " +
            std::to_string(line % 17) + " at " + std::to_string(line % 24) +
            ":" + std::to_string(line * 7 % 60) + " carrying " +
            std::to_string(line * 13 % 500) + " crates.";
    result = model.Tokenize(text);
  }
  result.resize(tokens);
  return result;
}
}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GEMMA4_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: GUFO_GEMMA4_MODEL is not set\n";
    return 77;
  }
  return gemma4_test::Run([&] {
    std::string error;
    g4::ModelOptions options;
    // A small prefill chunk keeps the ring short (window + 256 slots), so
    // the prompt below wraps it.
    options.max_context = 4096;
    options.prefill_chunk = 240;
    auto model = g4::Model::Load(path, options, &error);
    Require(model != nullptr, error);
    const auto ring = model->SessionRingSlots();
    g4::ContinuationAdapter adapter(model, options.max_context, {'t'});
    gufo::hip::TransferPool pool(2);
    Fixture f{.adapter = adapter, .pool = pool, .pinned = true};
    RecordingGuard guard_a, guard_b;
    auto slot_a = adapter.CreateSlot(guard_a);
    auto slot_b = adapter.CreateSlot(guard_b);

    // A prompt past the ring: forwards evict window rows, and the guard
    // hears of those evictions besides the rows written.
    const auto prompt = LongPrompt(*model, ring + 700);
    auto& a = adapter.GetSession(*slot_a);
    Require(a.Sync(prompt, &error), error);
    const auto& layout = adapter.Layout();
    const auto window = std::ranges::find_if(layout, [](const auto& c) {
      return c.descriptor.kind == cache::ComponentKind::kWindowRows;
    });
    const auto global = std::ranges::find_if(layout, [](const auto& c) {
      return c.part == g4::ContinuationPart::kKey &&
             c.descriptor.kind == cache::ComponentKind::kAppendRows;
    });
    Require(window != layout.end() && global != layout.end(),
            "layout lacks window or global rows");
    const auto writes = [&](cache::ComponentId id) {
      return std::ranges::count_if(guard_a.writes,
                                   [&](const auto& w) { return w.id == id; });
    };
    Require(writes(window->descriptor.id) > writes(global->descriptor.id),
            "ring evictions were not guarded");

    auto history = cache::ExecutionHistory::Cold(
        f.ledger, adapter.Components(), adapter.CompatibilityIdentity());
    const std::vector<cache::Token> cached(prompt.begin(), prompt.end());
    const auto checkpoint = history.Capture(
        {cached, cache::InputIdentity(cached.size()),
         adapter.Positions(*slot_a), cache::CheckpointPurpose::kPrompt, 1},
        [&](const cache::PayloadRequest& request) {
          return f.Save(*slot_a, request);
        });
    const auto n = prompt.size();
    for (const auto& c : checkpoint->Components())
      if (c.descriptor.kind == cache::ComponentKind::kWindowRows)
        Require(
            c.chunks.front().First() == cache::FirstChunkRow(c.descriptor, n),
            "window chunks start before the window");
    const auto reference = Greedy(a, 6);

    f.Restore(*slot_b, *checkpoint, false);
    f.RequireSameState(*slot_b, *checkpoint);
    auto& b = adapter.GetSession(*slot_b);
    Require(b.Position() == n, "restored position differs");
    Require(Greedy(b, 6) == reference, "restore into another slot diverges");

    // Right after the reply, the original slot still holds every row: an
    // in-place restore loads only the private state.
    std::size_t private_bytes = 0;
    for (const auto& c : checkpoint->Components())
      private_bytes += c.descriptor.state_bytes;
    Require(f.RestoreKeeping(*slot_a, *checkpoint, nullptr) == private_bytes,
            "in-place restore after a short reply loaded rows");
    f.RequireSameState(*slot_a, *checkpoint);
    Require(Greedy(adapter.GetSession(*slot_a), 6) == reference,
            "in-place restore after a short reply diverges");

    // The original slot moves a whole ring further, then restores in place.
    auto further = prompt;
    further.insert(further.end(), reference.tokens.begin(),
                   reference.tokens.end());
    const auto more = LongPrompt(*model, ring + 64);
    further.insert(further.end(), more.begin() + 1, more.end());
    Require(a.Sync(further, &error), error);
    std::vector<cache::ResidentRange> kept;
    const auto loaded = f.RestoreKeeping(*slot_a, *checkpoint, &kept);
    // Global rows stay; the window left the ring and loads again.
    for (const auto& k : kept)
      Require(adapter.Layout()[k.id.value - 1].descriptor.kind ==
                      cache::ComponentKind::kAppendRows &&
                  k.first == 0 && k.end == n,
              "kept rows outside the global layers after a ring wrap");
    Require(loaded > private_bytes, "the evicted window was not reloaded");
    f.RequireSameState(*slot_a, *checkpoint);
    f.Restore(*slot_a, *checkpoint, true);
    f.RequireSameState(*slot_a, *checkpoint);
    Require(Greedy(adapter.GetSession(*slot_a), 6) == reference,
            "restore into the wrapped original slot diverges");

    // A frontier adopted at a stop inside a prompt lookahead keeps that row's
    // hidden state (the MTP drafter's input), as one evaluated directly does.
    {
      const auto hidden =
          std::ranges::find(layout, g4::ContinuationPart::kHidden,
                            &g4::ContinuationComponent::part);
      const auto read_hidden = [&](cache::Slot& slot) {
        std::vector<float> values(hidden->descriptor.state_bytes /
                                  sizeof(float));
        Require(hipHostRegister(values.data(), hidden->descriptor.state_bytes,
                                hipHostRegisterDefault) == hipSuccess,
                "host registration failed");
        f.Settle([&](cache::Stream& stream) {
          return adapter.CapturePrivate(
              slot, hidden->descriptor.id,
              std::as_writable_bytes(std::span(values)), stream);
        });
        (void)hipHostUnregister(values.data());
        return values;
      };
      const auto text = LongPrompt(*model, 600);
      const std::span<const g4::TokenId> head(text.data(), text.size() - 40);
      const std::span<const g4::TokenId> tail(text.data() + head.size(), 40);
      const std::vector<std::size_t> stops{5, 23};
      // Stops that descend, repeat or reach past the lookahead are refused.
      for (const std::vector<std::size_t>& bad :
           {std::vector<std::size_t>{23, 5}, std::vector<std::size_t>{5, 5},
            std::vector<std::size_t>{5, tail.size()}}) {
        std::string refused;
        Require(!b.Sync(head, {}, {}, &refused, tail, bad) && !refused.empty(),
                "malformed lookahead stops accepted");
      }
      Require(b.Sync(head, {}, {}, &error, tail, stops) &&
                  b.LookaheadTokens() == tail.size(),
              "lookahead with stops: " + error);
      for (const std::size_t stop : stops) {
        const auto prefix = std::span(text).first(head.size() + stop);
        Require(
            b.Sync(prefix, &error) && b.LookaheadTokens() == tail.size() - stop,
            "lookahead stop not adopted: " + error);
        Require(a.Sync(prefix, &error), error);
        const auto adopted = read_hidden(*slot_b);
        const auto direct = read_hidden(*slot_a);
        double dot = 0, aa = 0, dd = 0;
        for (std::size_t i = 0; i < adopted.size(); ++i) {
          dot += double{adopted[i]} * direct[i];
          aa += double{adopted[i]} * adopted[i];
          dd += double{direct[i]} * direct[i];
        }
        const double cosine = dot / std::sqrt(aa * dd);
        std::cout << "adopted hidden at stop " << stop << ": cosine " << cosine
                  << "\n";
        Require(cosine > 0.999, "adopted hidden state differs");
      }
    }

    // A capture taken while another slot prefills reads only its own
    // session's writes, and equals one taken with the device idle.
    {
      f.Restore(*slot_a, *checkpoint, false);
      const auto started = guard_b.count.load(std::memory_order_acquire);
      std::string prefill_error;
      bool prefilled = false;
      std::thread prefill([&] {
        prefilled = b.Sync(LongPrompt(*model, ring + 700), &prefill_error);
      });
      while (guard_b.count.load(std::memory_order_acquire) == started)
        std::this_thread::yield();
      auto during = cache::ExecutionHistory::Cold(
          f.ledger, adapter.Components(), adapter.CompatibilityIdentity());
      const auto concurrent = during.Capture(
          {cached, cache::InputIdentity(cached.size()),
           adapter.Positions(*slot_a), cache::CheckpointPurpose::kPrompt, 1},
          [&](const cache::PayloadRequest& request) {
            return f.Save(*slot_a, request);
          });
      prefill.join();
      Require(prefilled, prefill_error);
      f.RequireSameState(*slot_a, *concurrent);
      f.RequireSameState(*slot_a, *checkpoint);
    }

    // A checkpoint with an image restores the image records with its rows:
    // the restored session keeps the image when the prompt grows past it,
    // and evaluates from it again when the image changes, exactly as the
    // session it came from does.
    if (const char* mmproj = std::getenv("GUFO_GEMMA4_MMPROJ");
        mmproj != nullptr && *mmproj != '\0') {
      g4::vision::Encoder encoder(mmproj, model->config().hidden_size);
      const auto prepare = [&](std::uint32_t seed) {
        return gemma4_test::Prepare(
            *model, encoder, seed,
            "Which colour fills it? Answer with one word.", options.max_context,
            70);
      };
      const auto red = prepare(1), blue = prepare(2);
      const auto logits = [](const g4::Session& session) {
        return std::vector<float>(session.Logits().begin(),
                                  session.Logits().end());
      };
      auto& original = adapter.GetSession(*slot_b);
      Require(original.Sync(red.prompt.tokens, red.spans, red.embed(), &error),
              error);
      auto images = cache::ExecutionHistory::Cold(
          f.ledger, adapter.Components(), adapter.CompatibilityIdentity());
      const std::vector<cache::Token> pictured_tokens(red.prompt.tokens.begin(),
                                                      red.prompt.tokens.end());
      const auto pictured = images.Capture(
          {pictured_tokens, cache::InputIdentity(pictured_tokens.size()),
           adapter.Positions(*slot_b), cache::CheckpointPurpose::kPrompt, 1},
          [&](const cache::PayloadRequest& request) {
            return f.Save(*slot_b, request);
          });
      f.Restore(*slot_a, *pictured, false);
      f.RequireSameState(*slot_a, *pictured);
      auto& restored = adapter.GetSession(*slot_a);
      Require(std::ranges::equal(restored.Images(), original.Images()) &&
                  logits(restored) == logits(original),
              "restore lost the image");
      auto longer = red.prompt.tokens;
      const auto more = model->Tokenize(" Please answer briefly.");
      longer.insert(longer.end() - 1, more.begin(), more.end());
      const auto same = [&](const std::vector<g4::TokenId>& prompt,
                            const gemma4_test::Prepared& images,
                            const char* what) {
        Require(original.Sync(prompt, images.spans, images.embed(), &error) &&
                    restored.Sync(prompt, images.spans, images.embed(), &error),
                error);
        Require(logits(restored) == logits(original), what);
      };
      same(longer, red, "restored image prefix extends differently");
      same(blue.prompt.tokens, blue, "restored session misses a changed image");
      // In place: the slot holds the image rows of the blue prompt.
      f.RestoreKeeping(*slot_b, *pictured, nullptr);
      f.RequireSameState(*slot_b, *pictured);
      Require(
          std::ranges::equal(original.Images(), std::span(red.spans).first(1)),
          "in-place restore kept the other image");
      std::cout << "images: a " << red.prompt.tokens.size()
                << "-token checkpoint with a " << red.spans[0].rows
                << "-row image restores into another slot and in place\n";
    }

    // A completion outlives its slot: the slot settles its transfers before
    // it is destroyed, and a later wait returns that result.
    {
      RecordingGuard guard;
      auto doomed = adapter.CreateSlot(guard);
      Require(
          adapter.GetSession(*doomed).Sync(std::span(prompt).first(64), &error),
          error);
      const auto hidden =
          std::ranges::find(layout, g4::ContinuationPart::kHidden,
                            &g4::ContinuationComponent::part);
      std::vector<float> values(hidden->descriptor.state_bytes / sizeof(float));
      Require(hipHostRegister(values.data(), hidden->descriptor.state_bytes,
                              hipHostRegisterDefault) == hipSuccess,
              "host registration failed");
      auto stream = pool.TryAcquire();
      Require(stream != nullptr, "no transfer stream");
      auto completion = adapter.CapturePrivate(
          *doomed, hidden->descriptor.id,
          std::as_writable_bytes(std::span(values)), *stream);
      doomed.reset();
      Require(completion.Wait() == cache::TransferResult::kSucceeded,
              "a capture waited after its slot was destroyed failed");
      (void)hipHostUnregister(values.data());
    }

    // A checkpoint holds every image record up to kContinuationImages, and a
    // restored session evaluates only an image past them, as the original
    // does. A session with more images is not captured. One-row images with
    // zero embeddings sit at odd positions, each followed by a text token.
    {
      const std::size_t most = g4::kContinuationImages;
      const auto bytes =
          std::size_t{model->config().hidden_size} * sizeof(float);
      void* embedding = nullptr;
      Require(hipMalloc(&embedding, bytes) == hipSuccess &&
                  hipMemset(embedding, 0, bytes) == hipSuccess,
              "image embedding allocation failed");
      const g4::ImageEmbeddings embed = [&](std::size_t) {
        return static_cast<const float*>(embedding);
      };
      const auto many_tokens = LongPrompt(*model, 2 * most + 4);
      std::vector<g4::ImageSpan> many;
      for (std::size_t i = 0; i <= most; ++i) {
        g4::ImageSpan span{static_cast<std::uint32_t>(1 + 2 * i), 1};
        std::memcpy(span.identity.data(), &i, sizeof(i));
        many.push_back(span);
      }
      const auto few_tokens = std::span(many_tokens).first(2 * most + 2);
      const auto few = std::span(many).first(most);
      Require(a.Sync(few_tokens, few, embed, &error), error);
      auto pictured = cache::ExecutionHistory::Cold(
          f.ledger, adapter.Components(), adapter.CompatibilityIdentity());
      const std::vector<cache::Token> few_cached(few_tokens.begin(),
                                                 few_tokens.end());
      const auto all = pictured.Capture(
          {few_cached, cache::InputIdentity(few_cached.size()),
           adapter.Positions(*slot_a), cache::CheckpointPurpose::kPrompt, 1},
          [&](const cache::PayloadRequest& request) {
            return f.Save(*slot_a, request);
          });
      f.Restore(*slot_b, *all, false);
      Require(std::ranges::equal(b.Images(), few),
              "restore lost image records");
      Require(a.Sync(many_tokens, many, embed, &error) &&
                  b.Sync(many_tokens, many, embed, &error),
              error);
      Require(std::ranges::equal(a.Logits(), b.Logits()),
              "the restored session evaluated more than the new image");
      bool refused = false;
      try {
        auto over = cache::ExecutionHistory::Cold(
            f.ledger, adapter.Components(), adapter.CompatibilityIdentity());
        const std::vector<cache::Token> many_cached(many_tokens.begin(),
                                                    many_tokens.end());
        (void)over.Capture(
            {many_cached, cache::InputIdentity(many_cached.size()),
             adapter.Positions(*slot_a), cache::CheckpointPurpose::kPrompt, 1},
            [&](const cache::PayloadRequest& request) {
              return f.Save(*slot_a, request);
            });
      } catch (const std::length_error&) {
        refused = true;
      }
      Require(refused, "a session with more image records was captured");
      (void)hipFree(embedding);
      std::cout << "images: " << most
                << " image records restore; one more refuses the capture\n";
    }

    Require(adapter.Invalidate(*slot_b), "invalidate refused");
    Require(adapter.GetSession(*slot_b).Position() == 0 && slot_b->IsValid(),
            "invalidated slot is not empty and valid");
    std::cout << "continuation: " << n << "-token checkpoint, ring " << ring
              << ", 6 greedy tokens match after both restores\n";

    const char* draft = std::getenv("GUFO_GEMMA4_MTP_MODEL");
    if (draft == nullptr || *draft == '\0')
      return;
    model.reset();
    options.mtp_model_path = draft;
    options.draft_tokens = g4::rocm::kMaxDraftTokens;
    // Fixed-length chains make every draft depend on the drafter's first
    // input, the restored hidden state.
    options.draft_policy = g4::DraftPolicy::kFixed;
    auto mtp = g4::Model::Load(path, options, &error);
    Require(mtp != nullptr, error);
    g4::ContinuationAdapter speculative(mtp, options.max_context, {'m'});
    Fixture g{.adapter = speculative, .pool = pool};
    RecordingGuard guard_c, guard_d;
    auto slot_c = speculative.CreateSlot(guard_c);
    auto slot_d = speculative.CreateSlot(guard_d);
    // Ordinary prose: prompt lookup has nothing to copy.
    const auto story = mtp->Tokenize(
        "<bos><|turn>user\nWrite a short story about a lighthouse keeper "
        "who finds a message in a bottle.<turn|>\n<|turn>model\n");
    const std::vector<cache::Token> story_tokens(story.begin(), story.end());
    auto& c = speculative.GetSession(*slot_c);
    Require(c.Sync(story, &error), error);
    auto mtp_history =
        cache::ExecutionHistory::Cold(g.ledger, speculative.Components(),
                                      speculative.CompatibilityIdentity());
    const auto mtp_checkpoint = mtp_history.Capture(
        {story_tokens, cache::InputIdentity(story_tokens.size()),
         speculative.Positions(*slot_c), cache::CheckpointPurpose::kPrompt, 1},
        [&](const cache::PayloadRequest& request) {
          return g.Save(*slot_c, request);
        });
    const auto expected = Speculate(c, 4);
    Require(expected.drafted.front() > 0,
            "the first cycle drafted nothing from the restored state");
    g.Restore(*slot_d, *mtp_checkpoint, true);
    g.RequireSameState(*slot_d, *mtp_checkpoint);
    auto& d = speculative.GetSession(*slot_d);
    d.BeginRequest();
    Require(Speculate(d, 4) == expected, "restored MTP cycles diverge");
    std::cout << "mtp cycles (accepted/drafted):";
    for (std::size_t i = 0; i < expected.drafted.size(); ++i)
      std::cout << ' ' << expected.accepted[i] << '/' << expected.drafted[i];
    std::cout << ", identical after restore\n";
  });
}
