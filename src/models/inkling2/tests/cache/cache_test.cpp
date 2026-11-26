// GPU test of the cache layer, without a server.
//
//   inkling2_cache_test --model GGUF --ids ids.bin [--n N] [--tail 48]
//
// 1. Capture/restore: prefill N tokens in slot A, capture a checkpoint through
// the adapter (rows,
//    window rows, private state, in 4 KiB pieces for the private parts),
//    continue A for `tail` tokens recording every token's logits; restore the
//    checkpoint into a fresh slot B, run the same tokens: logits must be
//    BIT-identical. N > 1,536 so the SWA rings have wrapped.
// 2. Layered prefill: prefill the same N tokens in slot C with PrefillYield (8
// groups), a yield
//    callback that decodes on a second session D; C's logits equal A's and D's
//    logits equal an uninterrupted run of D (bit-identical).
// 3. The mutation guard is called before rows are overwritten, and captures
// work inside the
//    guard callback (as the cache does).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "src/models/inkling2/cache_adapter.hpp"
#include "src/models/inkling2/engine.hpp"

// An external extension target may not link gufo_cache (the `gufo` binary has
// it), so this tool carries its own copy of the 40 lines of
// cache/completion.cpp (gufo v0.35.0, same behaviour).
namespace gufo::cache {
Completion::Completion(std::unique_ptr<CompletionSignal> signal)
    : signal_(std::move(signal)) {}
Completion::~Completion() {
  (void)Wait();
}
Completion::Completion(Completion&& other) noexcept
    : signal_(std::move(other.signal_)),
      result_(std::exchange(other.result_, {})) {}
Completion& Completion::operator=(Completion&& other) noexcept {
  if (this != &other) {
    (void)Wait();
    signal_ = std::move(other.signal_);
    result_ = std::exchange(other.result_, {});
  }
  return *this;
}
bool Completion::Ready() const noexcept {
  return result_.has_value() || !signal_ || signal_->Ready();
}
TransferResult Completion::Wait() noexcept {
  if (!result_)
    result_ = signal_ ? signal_->Wait() : TransferResult::kFailed;
  return *result_;
}
}  // namespace gufo::cache

namespace i2 = gufo::models::inkling2;
namespace gc = gufo::cache;

namespace {
class NullStream final : public gc::Stream {
public:
  gc::TransferResult Synchronize() noexcept override {
    return gc::TransferResult::kSucceeded;
  }
};

struct CountingGuard final : gc::MutationGuard {
  std::size_t overwrites = 0, releases = 0, resets = 0, reentrant_captures = 0;
  i2::Inkling2CacheAdapter* adapter = nullptr;
  gc::Slot* slot = nullptr;
  bool capture_inside = false;
  void BeforeOverwrite(gc::ComponentId id, gc::Rows first,
                       gc::Rows end) override {
    ++overwrites;
    if (capture_inside && adapter && slot && first < end) {
      // The cache preserves what a checkpoint borrows from inside this
      // callback.
      const auto pos = adapter->Positions(*slot);
      const auto have = pos[id.value - 1].valid_rows;
      if (first < have && id.value > 3) {
        const auto stop = std::min<gc::Rows>(end, have);
        std::vector<std::byte> buf((stop - first) * i2::kKvRowBytes);
        NullStream st;
        auto done = adapter->CopyRowsOut(*slot, id, first, stop, buf, st);
        if (done.Wait() == gc::TransferResult::kSucceeded)
          ++reentrant_captures;
      }
    }
  }
  void BeforeRelease(gc::ComponentId, gc::Rows, gc::Rows) noexcept override {
    ++releases;
  }
  void AfterReset() noexcept override { ++resets; }
};

struct Checkpoint {
  std::vector<gc::ComponentPosition> positions;
  std::vector<std::vector<std::byte>>
      data;  // per component: rows [lo, hi) or private bytes
};

#define REQUIRE(cond, ...)                                                   \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                                     \
      std::fprintf(stderr, "\n");                                            \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

Checkpoint Capture(i2::Inkling2CacheAdapter& a, gc::Slot& slot) {
  NullStream st;
  Checkpoint cp;
  cp.positions = a.Positions(slot);
  const auto comps = a.Components();
  cp.data.resize(comps.size());
  for (std::size_t i = 0; i < comps.size(); ++i) {
    const auto& d = comps[i];
    const auto [lo, hi] =
        i2::StateLayout::RestoreRange(d, cp.positions[i].valid_rows);
    if (d.kind == gc::ComponentKind::kPrivateState) {
      cp.data[i].resize(d.state_bytes);
      // in 4 KiB pieces (the bounded-serialization path) except the 64-byte
      // metadata
      for (std::size_t off = 0; off < d.state_bytes; off += 4096) {
        const auto n = std::min<std::size_t>(4096, d.state_bytes - off);
        auto c = a.CapturePrivatePiece(
            slot, d.id, off, std::span<std::byte>(cp.data[i]).subspan(off, n),
            st);
        REQUIRE(c.Wait() == gc::TransferResult::kSucceeded,
                "private capture %zu", i);
      }
    } else {
      cp.data[i].resize((hi - lo) * d.row_bytes);
      // in 256-row chunks, like the cache
      for (gc::Rows r = lo; r < hi; r += 256) {
        const auto e = std::min<gc::Rows>(r + 256, hi);
        auto c = a.CopyRowsOut(
            slot, d.id, r, e,
            std::span<std::byte>(cp.data[i])
                .subspan((r - lo) * d.row_bytes, (e - r) * d.row_bytes),
            st);
        REQUIRE(c.Wait() == gc::TransferResult::kSucceeded, "rows capture %zu",
                i);
      }
    }
  }
  return cp;
}

void Restore(i2::Inkling2CacheAdapter& a, gc::Slot& slot,
             const Checkpoint& cp) {
  NullStream st;
  a.BeginRestore(slot, cp.positions);
  const auto comps = a.Components();
  for (std::size_t i = 0; i < comps.size(); ++i) {
    const auto& d = comps[i];
    const auto [lo, hi] =
        i2::StateLayout::RestoreRange(d, cp.positions[i].valid_rows);
    if (d.kind == gc::ComponentKind::kPrivateState) {
      for (std::size_t off = 0; off < d.state_bytes; off += 4096) {
        const auto n = std::min<std::size_t>(4096, d.state_bytes - off);
        auto c = a.LoadPrivatePiece(
            slot, d.id, off,
            std::span<const std::byte>(cp.data[i]).subspan(off, n), st);
        REQUIRE(c.Wait() == gc::TransferResult::kSucceeded, "private load %zu",
                i);
      }
    } else {
      for (gc::Rows r = lo; r < hi; r += 256) {
        const auto e = std::min<gc::Rows>(r + 256, hi);
        auto c = a.CopyRowsIn(
            slot, d.id, r, e,
            std::span<const std::byte>(cp.data[i])
                .subspan((r - lo) * d.row_bytes, (e - r) * d.row_bytes),
            st);
        REQUIRE(c.Wait() == gc::TransferResult::kSucceeded, "rows load %zu", i);
      }
    }
  }
  REQUIRE(a.Validate(slot, cp.positions), "Validate refused the restore");
}

std::vector<float> Run(i2::Engine& e, i2::Session& s,
                       std::span<const std::int32_t> t,
                       const i2::PrefillYield* y = nullptr) {
  std::vector<float> all;
  const auto V = e.config().n_vocab_text;
  e.Forward(
      s, t,
      [&](std::size_t, std::size_t count, const float* d) {
        all.insert(all.end(), d, d + count * V);
      },
      y);
  return all;
}

// The M2 prefill kernels add the expert outputs with f32 atomics, so two
// identical prefills can differ in the last bits. `g_noise` is the measured
// run-to-run difference of an identical sequence (0 when the engine is
// deterministic, e.g. INKLING2_SLOWPREFILL=1); comparisons are exact when it is
// 0 and within 8x it otherwise. A wrong or stale state row moves logits by far
// more than that.
double g_noise = 0;
double g_worst = 0;
double MaxDiff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size())
    return 1e30;
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
  return m;
}
bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  const double d = MaxDiff(a, b);
  g_worst = std::max(g_worst, d);
  return g_noise == 0 ? d == 0 : d <= 8 * g_noise;
}
}  // namespace

int main(int argc, char** argv) {
  std::string model, idsf;
  // Two passes and a 600-token third: more than the ring rows (1024 + ubatch),
  // so the SWA rings wrap.
  std::size_t n = 2 * i2::Engine::kUbatch + 600, tail = 48;
  std::size_t chunk_prompt = 1853;  // over 1K keys, under one ubatch
  bool chunks_only = false;
  std::string dspark;  // drafter GGUF: its context is checkpointed too
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&] {
      return i + 1 < argc ? std::string(argv[++i]) : std::string();
    };
    if (k == "--model")
      model = next();
    else if (k == "--ids")
      idsf = next();
    else if (k == "--n")
      n = std::strtoull(next().c_str(), nullptr, 10);
    else if (k == "--tail")
      tail = std::strtoull(next().c_str(), nullptr, 10);
    else if (k == "--chunk-prompt")
      chunk_prompt = std::strtoull(next().c_str(), nullptr, 10);
    else if (k == "--chunks-only")
      chunks_only = true;
    else if (k == "--dspark")
      dspark = next();
  }
  if (model.empty() || idsf.empty()) {
    std::fprintf(stderr,
                 "usage: inkling2_cache_test --model GGUF --ids ids.bin [--n "
                 "N] [--tail T] [--chunk-prompt P] [--chunks-only] [--dspark "
                 "DRAFT]\n");
    return 2;
  }
  std::ifstream f(idsf, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  const auto raw = ss.str();
  std::vector<std::int32_t> ids(raw.size() / 4);
  std::memcpy(ids.data(), raw.data(), ids.size() * 4);
  if (ids.size() < n + tail + 400) {
    std::fprintf(stderr, "need %zu ids\n", n + tail + 400);
    return 2;
  }

  std::string err;
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader =
      gufo::core::GgufReader::OpenFile(model, &err, open);
  if (!reader) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  auto engine = i2::Engine::Create(
      reader, static_cast<std::uint32_t>(n + tail + 512), &err);
  if (!engine) {
    std::fprintf(stderr, "engine: %s\n", err.c_str());
    return 1;
  }
  const auto V = engine->config().n_vocab_text;
  if (!dspark.empty() && !engine->AttachDrafter(dspark, &err)) {
    std::fprintf(stderr, "drafter: %s\n", err.c_str());
    return 1;
  }

  CountingGuard guard;
  gufo::models::inkling2::Inkling2CacheAdapter adapter(
      std::shared_ptr<i2::Engine>(std::move(engine)),
      gc::Identity{'t', 'e', 's', 't'}, false, i2::Engine::kUbatch);
  auto& eng = *adapter.engine_for_test();
  const std::size_t U = eng.ubatch();
  guard.adapter = &adapter;

  const std::span<const std::int32_t> prompt(ids.data(), n),
      more(ids.data() + n, tail);

  // ---- 0. chunk invariance ----
  // A restored prefix plus a tail prefill must equal one prefill of the whole
  // prompt, bit for bit: every prompt chunk, however short, takes the same
  // per-token arithmetic. Splits cover 1-token tails, the decode and
  // verification widths (up to 64 rows), the GEMM tile edges and many short
  // chunks.
  {
    const std::size_t P = std::min<std::size_t>(chunk_prompt, n);
    const std::span<const std::int32_t> cp_prompt(ids.data(), P);
    const std::span<const std::int32_t> cont(ids.data() + P, 4);
    auto finish = [&](i2::Session& s) {
      std::vector<float> all = s.logits();
      for (const auto t : cont) {
        eng.Step(s, t, nullptr);
        const auto& l = s.logits();
        all.insert(all.end(), l.begin(), l.end());
      }
      return all;
    };
    auto slot_r = adapter.CreateSlot(guard);
    auto& sr = adapter.GetSession(*slot_r);
    eng.Forward(sr, cp_prompt);
    const auto want = finish(sr);
    std::vector<std::vector<std::size_t>> splits;
    for (const std::size_t k :
         {1,  2,   3,   7,   8,   9,   31,  44,   63,   64,
          65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025})
      if (k < P)
        splits.push_back({P - k, k});
    for (const std::size_t k : {2047, 2048, 2049, 3000})
      if (k < P)
        splits.push_back({k, P - k});
    splits.push_back({P / 3, P / 3, P - 2 * (P / 3)});
    {
      std::vector<std::size_t> many;
      for (std::size_t at = 0; at < P; at += 37)
        many.push_back(std::min<std::size_t>(37, P - at));
      splits.push_back(many);
    }
    int bad = 0;
    for (const auto& sp : splits) {
      auto slot_s = adapter.CreateSlot(guard);
      auto& sx = adapter.GetSession(*slot_s);
      std::size_t at = 0;
      for (const auto len : sp) {
        eng.Forward(sx, cp_prompt.subspan(at, len));
        at += len;
      }
      const auto got = finish(sx);
      const double d = MaxDiff(got, want);
      std::string desc;
      for (const auto len : sp)
        desc += (desc.empty() ? "" : "+") + std::to_string(len);
      if (sp.size() > 4)
        desc = std::to_string(sp.size()) + " chunks of " +
               std::to_string(sp.front());
      std::fprintf(stderr, "%s chunks %s: max |dlogit| %.3e\n",
                   d == 0 ? "same" : "DIFF", desc.c_str(), d);
      bad += d != 0;
    }
    // The server's case: a checkpoint of the prefix restored into a fresh
    // slot, then the tail.
    for (const std::size_t k : {1, 44}) {
      auto slot_p = adapter.CreateSlot(guard);
      auto& sp = adapter.GetSession(*slot_p);
      eng.Forward(sp, cp_prompt.subspan(0, P - k));
      const auto pre = Capture(adapter, *slot_p);
      auto slot_q = adapter.CreateSlot(guard);
      Restore(adapter, *slot_q, pre);
      auto& sq = adapter.GetSession(*slot_q);
      eng.Forward(sq, cp_prompt.subspan(P - k, k));
      const double d = MaxDiff(finish(sq), want);
      std::fprintf(stderr, "%s restore %zu + %zu: max |dlogit| %.3e\n",
                   d == 0 ? "same" : "DIFF", P - k, k, d);
      bad += d != 0;
    }
    REQUIRE(bad == 0, "%d chunkings or restores differ from one prefill", bad);
    std::fprintf(stderr,
                 "PASS chunk invariance: %zu chunkings and 2 restores of %zu "
                 "tokens equal one prefill, logits and %zu decode steps bit "
                 "for bit\n",
                 splits.size(), P, cont.size());
    if (chunks_only)
      return 0;
  }

  // ---- 1. capture / restore ----
  auto slot_a = adapter.CreateSlot(guard);
  guard.slot = slot_a.get();
  auto& sa = adapter.GetSession(*slot_a);
  guard.capture_inside = true;
  for (std::size_t at = 0; at < n; at += U) {
    const auto len = std::min<std::size_t>(U, n - at);
    (void)Run(eng, sa, prompt.subspan(at, len));
  }
  guard.capture_inside = false;
  std::fprintf(stderr,
               "A prefilled %zu tokens; guard overwrites=%zu, in-callback "
               "captures=%zu\n",
               n, guard.overwrites, guard.reentrant_captures);
  REQUIRE(guard.overwrites > 0, "the guard was never called");
  REQUIRE(guard.reentrant_captures > 0,
          "no capture ran inside a guard callback");
  const std::vector<float> logits_at_n = sa.logits();
  auto cp = Capture(adapter, *slot_a);
  std::size_t cp_bytes = 0;
  for (const auto& d : cp.data)
    cp_bytes += d.size();
  std::fprintf(stderr, "checkpoint at %zu: %zu components, %.1f MiB\n", n,
               cp.data.size(), cp_bytes / 1048576.0);
  const auto ref = Run(eng, sa, more);  // A continues
  {
    // noise floor: the same prompt in another slot, no cache involved
    auto slot_n = adapter.CreateSlot(guard);
    auto& sn = adapter.GetSession(*slot_n);
    for (std::size_t at = 0; at < n; at += U)
      (void)Run(eng, sn, prompt.subspan(at, std::min<std::size_t>(U, n - at)));
    const auto l0 = sn.logits();
    const auto m2 = Run(eng, sn, more);
    g_noise = std::max(MaxDiff(l0, logits_at_n), MaxDiff(m2, ref));
    std::fprintf(stderr,
                 "run-to-run noise (max |dlogit|, identical prefill + "
                 "continuation): %.3e%s\n",
                 g_noise,
                 g_noise == 0 ? " (deterministic: comparisons are bit-exact)"
                              : " (comparisons use 8x)");
  }

  auto slot_b = adapter.CreateSlot(guard);
  Restore(adapter, *slot_b, cp);
  auto& sb = adapter.GetSession(*slot_b);
  REQUIRE(sb.position() == n, "restored position %zu", sb.position());
  REQUIRE(sb.logits() == logits_at_n,
          "restored logits differ from the captured ones");
  if (eng.has_drafter()) {
    // The drafter's context came back with the checkpoint: proposals at the
    // restored frontier equal those of a session that prefilled the prompt.
    auto slot_o = adapter.CreateSlot(guard);
    auto& so = adapter.GetSession(*slot_o);
    for (std::size_t at = 0; at < n; at += U)
      eng.Forward(so, prompt.subspan(at, std::min<std::size_t>(U, n - at)));
    REQUIRE(eng.DraftReady(sb), "no drafter context after a restore");
    std::int32_t pa[7] = {}, pb[7] = {};
    float ca[7] = {}, cb[7] = {};
    const int na = eng.Propose(so, more[0], 7, pa, ca);
    const int nb = eng.Propose(sb, more[0], 7, pb, cb);
    REQUIRE(na == nb && std::memcmp(pa, pb, sizeof pa) == 0 &&
                std::memcmp(ca, cb, sizeof ca) == 0,
            "proposals after a restore differ (%d vs %d)", na, nb);
    std::fprintf(stderr,
                 "PASS drafter context restored: %d proposals identical\n", na);
  }
  const auto got = Run(eng, sb, more);
  REQUIRE(got.size() == ref.size(), "size");
  REQUIRE(Same(got, ref),
          "logits after a restore differ from the uninterrupted run");
  std::fprintf(stderr,
               "PASS restore: %zu continuation tokens identical (exact when "
               "the engine is deterministic, else within the noise floor)\n",
               tail);

  // restore into a slot that already holds another sequence (the cache reuses
  // slots)
  {
    auto slot_x = adapter.CreateSlot(guard);
    auto& sx = adapter.GetSession(*slot_x);
    (void)Run(eng, sx, std::span<const std::int32_t>(ids.data() + 100, 700));
    Restore(adapter, *slot_x, cp);
    const auto again = Run(eng, sx, more);
    REQUIRE(Same(again, ref), "restore into a used slot differs");
    std::fprintf(stderr, "PASS restore into a used slot\n");
    // Invalidate returns it to a clean empty slot
    REQUIRE(adapter.Invalidate(*slot_x), "Invalidate");
    REQUIRE(sx.position() == 0 && sx.logits().empty(),
            "slot not empty after Invalidate");
  }

  // ---- 2. layered prefill ----
  auto slot_c = adapter.CreateSlot(guard);
  auto& sc = adapter.GetSession(*slot_c);
  auto slot_d = adapter.CreateSlot(guard);
  auto& sd = adapter.GetSession(*slot_d);
  const std::span<const std::int32_t> dprompt(ids.data() + n + tail + 100, 64);
  (void)Run(eng, sd, dprompt);
  std::vector<std::vector<float>> d_yielded;
  std::size_t yields = 0;
  i2::PrefillYield y;
  y.groups = 8;
  y.yield = [&] {
    ++yields;
    const std::int32_t tok = ids[n + tail + 200 + yields];
    eng.Decode(sd, tok);
    d_yielded.push_back(sd.logits());
  };
  std::vector<float> c_logits;
  for (std::size_t at = 0; at < n; at += U) {
    const auto len = std::min<std::size_t>(U, n - at);
    (void)Run(eng, sc, prompt.subspan(at, len), &y);
  }
  c_logits = sc.logits();
  REQUIRE(yields >= 7, "expected at least 7 yields, got %zu", yields);
  REQUIRE(Same(c_logits, logits_at_n),
          "layered prefill logits differ from the unsplit prefill");
  // continuing C must also match A (KV rows written across yields are intact)
  const auto c_more = Run(eng, sc, more);
  REQUIRE(Same(c_more, ref), "tokens after a layered prefill differ");
  // D without interleaving
  auto slot_d2 = adapter.CreateSlot(guard);
  auto& sd2 = adapter.GetSession(*slot_d2);
  (void)Run(eng, sd2, dprompt);
  for (std::size_t k = 0; k < d_yielded.size(); ++k) {
    const std::int32_t tok = ids[n + tail + 200 + k + 1];
    eng.Decode(sd2, tok);
    const auto l = sd2.logits();
    REQUIRE(Same(l, d_yielded[k]), "decode during a yield differs at yield %zu",
            k);
  }
  std::fprintf(stderr,
               "PASS layered prefill: %zu yields, prefill and interleaved "
               "decodes identical (exact when the engine is deterministic, "
               "else within the noise floor)\n",
               yields);

  // ---- 3. layered prefill whose yield runs a MULTI-token prefill on another
  // session ---- Another Forward replaces the per-ubatch dense work items and
  // the shared scratch; the yielded prefill must restore them (T differs: 70
  // against a full pass and the last one).
  {
    auto slot_e = adapter.CreateSlot(guard);
    auto& se = adapter.GetSession(*slot_e);
    auto slot_f = adapter.CreateSlot(guard);
    auto& sf = adapter.GetSession(*slot_f);
    auto slot_g = adapter.CreateSlot(guard);
    auto& sg = adapter.GetSession(*slot_g);
    std::size_t yields2 = 0;
    std::vector<std::vector<float>> e_log;
    i2::PrefillYield y2;
    y2.groups = 6;
    // 70 tokens per yield, taken from anywhere in the id file.
    auto yield_ids = [&](std::size_t k) {
      return (n + tail + 400 + k * 70) % (ids.size() - 70);
    };
    y2.yield = [&] {
      const std::size_t off = yield_ids(yields2);
      ++yields2;
      e_log.push_back(
          Run(eng, se, std::span<const std::int32_t>(ids.data() + off, 70)));
    };
    for (std::size_t at = 0; at < n; at += U) {
      const auto len = std::min<std::size_t>(U, n - at);
      (void)Run(eng, sf, prompt.subspan(at, len), &y2);
    }
    REQUIRE(Same(sf.logits(), logits_at_n),
            "layered prefill (prefill in yield) logits differ");
    for (std::size_t k = 0; k < e_log.size(); ++k) {
      const std::size_t off = yield_ids(k);
      const auto l =
          Run(eng, sg, std::span<const std::int32_t>(ids.data() + off, 70));
      REQUIRE(Same(l, e_log[k]), "prefill during a yield differs at yield %zu",
              k);
    }
    std::fprintf(stderr,
                 "PASS layered prefill with %zu yielded 70-token prefills, "
                 "identical (exact when the engine is deterministic, else "
                 "within the noise floor)\n",
                 yields2);
  }

  // ---- 4. in-pass prefill checkpoints ----
  // Single Forward call over two ubatches and 600 tokens of a third (the pass
  // writes more rows than the SWA ring holds, so it wraps)
  // capturing interior and ubatch-end boundaries in place, the last one after
  // the wrap. Restoring each checkpoint must yield bit-identical logits,
  // continuations, and (with --dspark) draft proposals to stopping prefill at
  // that boundary.
  const std::size_t kInPassLen = 2 * U + 600;
  if (n < kInPassLen) {
    std::fprintf(stderr, "SKIP in-pass checkpoints: needs --n %zu\n",
                 kInPassLen);
  } else {
    const std::span<const std::int32_t> ip_prompt(ids.data(), kInPassLen);
    const std::size_t ip_len = ip_prompt.size();
    const std::vector<std::size_t> bounds = {700,     1500,  U,
                                             U + 552, 2 * U, 2 * U + 104};
    const std::vector<i2::Block> blocks = {
        i2::Block::kIdle, i2::Block::kAfterHeader, i2::Block::kText,
        i2::Block::kIdle, i2::Block::kText,        i2::Block::kIdle};
    REQUIRE(ip_len > eng.ring_rows() && bounds.back() > eng.ring_rows(),
            "in-pass test no longer wraps the %u-row ring", eng.ring_rows());
    for (const std::size_t b : bounds)
      REQUIRE(eng.CanCaptureInPass(0, ip_len, b),
              "CanCaptureInPass rejected %zu", b);
    // A boundary whose FirstChunkRow would be evicted from the SWA ring must be
    // rejected: from offset U + 1 a pass ends at 2 U + 1, and the window of
    // boundary U + 2 starts U - 1024 rows in, one row more than the ring holds.
    REQUIRE(!eng.CanCaptureInPass(U + 1, U, U + 2),
            "CanCaptureInPass accepted evicted window boundary");

    auto slot_ip = adapter.CreateSlot(guard);
    std::vector<Checkpoint> cps(bounds.size());
    std::vector<bool> seen(bounds.size(), false);
    adapter.PrefillThrough(*slot_ip, ip_prompt, bounds, blocks,
                           [&](std::size_t idx) {
                             REQUIRE(idx < bounds.size() && !seen[idx],
                                     "bad checkpoint index %zu", idx);
                             const auto pos = adapter.Positions(*slot_ip);
                             REQUIRE(pos.front().valid_rows == bounds[idx],
                                     "checkpoint position %zu != %zu",
                                     pos.front().valid_rows, bounds[idx]);
                             cps[idx] = Capture(adapter, *slot_ip);
                             seen[idx] = true;
                           });
    for (std::size_t i = 0; i < bounds.size(); ++i)
      REQUIRE(seen[i], "checkpoint %zu not delivered", i);

    for (std::size_t i = 0; i < bounds.size(); ++i) {
      const std::size_t b = bounds[i];
      auto slot_ref = adapter.CreateSlot(guard);
      auto& sref = adapter.GetSession(*slot_ref);
      eng.Forward(sref, ip_prompt.first(b));
      const auto ref_logits = sref.logits();
      const auto cont = std::span<const std::int32_t>(
          ids.data() + b, std::min<std::size_t>(24, ids.size() - b));
      const auto ref_cont = Run(eng, sref, cont);

      auto slot_res = adapter.CreateSlot(guard);
      Restore(adapter, *slot_res, cps[i]);
      auto& sres = adapter.GetSession(*slot_res);
      REQUIRE(sres.position() == b, "restored in-pass pos %zu != %zu",
              sres.position(), b);
      REQUIRE(static_cast<i2::Inkling2Slot&>(*slot_res).block == blocks[i],
              "restored in-pass block mismatch at %zu", b);
      REQUIRE(Same(sres.logits(), ref_logits),
              "in-pass checkpoint logits differ at %zu", b);
      if (eng.has_drafter()) {
        auto slot_dref = adapter.CreateSlot(guard);
        auto& sdref = adapter.GetSession(*slot_dref);
        eng.Forward(sdref, ip_prompt.first(b));
        REQUIRE(eng.DraftReady(sres), "no drafter context at in-pass %zu", b);
        std::int32_t pa[7] = {}, pb[7] = {};
        float ca[7] = {}, cb[7] = {};
        const int na = eng.Propose(sdref, cont[0], 7, pa, ca);
        const int nb = eng.Propose(sres, cont[0], 7, pb, cb);
        REQUIRE(na == nb && std::memcmp(pa, pb, sizeof pa) == 0 &&
                    std::memcmp(ca, cb, sizeof ca) == 0,
                "in-pass drafter proposals differ at %zu (%d vs %d)", b, na,
                nb);
      }
      const auto got_cont = Run(eng, sres, cont);
      REQUIRE(Same(got_cont, ref_cont),
              "in-pass continuation logits differ at %zu", b);
    }
    std::fprintf(stderr,
                 "PASS in-pass checkpoints: %zu boundaries (interior + "
                 "ubatch-end, SWA wrap%s) identical\n",
                 bounds.size(), eng.has_drafter() ? " + DSpark" : "");
  }
  (void)V;
  std::fprintf(stderr, "worst compared max |dlogit| %.3e (noise %.3e)\n",
               g_worst, g_noise);
  std::fprintf(stderr,
               "ALL PASS (guard overwrites=%zu releases=%zu resets=%zu)\n",
               guard.overwrites, guard.releases, guard.resets);
  return 0;
}
