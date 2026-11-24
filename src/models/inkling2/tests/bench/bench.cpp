// Speed bench for the Inkling 2 Mini HIP engine.
//
//   inkling2_bench --model GGUF --text FILE --ctxs 64,4096,16384,32000
//   [--decode 64] [--ctx 32768]
//                  [--batch 2,3,4] [--verify] [--spec 2,3,4]
//
// --batch B...: after the single-session lines, for each B: B sessions
// (different text slices), each prefilled with C tokens, then --decode steps of
// Engine::DecodeBatch (one token per session per step, greedy). Prints the
// aggregate and per-session tok/s. --verify also runs the same sessions through
// single-token Forward calls and reports the largest logit difference and token
// agreement.
//
// --spec R...: speculative verification. A reference session decodes --decode
// greedy tokens one at a time; another replays them in Engine::Verify steps of
// R rows whose drafts are the reference tokens, one of them corrupted at
// random so that a prefix is kept (CommitVerify). Every kept row's logits are
// compared byte for byte with the reference; prints the mismatches and the
// verification step time.
//
// For every context length C: fresh session, timed prefill of C tokens (one
// Forward call, only the last logits are read back), then --decode greedy
// single-token Forward calls (timed). Prints prefill tok/s and decode tok/s
// (ms/token) at that context.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/inkling2/engine.hpp"
#include "src/models/inkling2/text/tokenizer.hpp"

int main(int argc, char** argv) {
  std::string model, text, ids_in, ctxs = "64,4096", batches, specs, dspark;
  int depth = 3;
  bool verify = false, same = false;
  int decode = 64;
  unsigned ctx = 32768;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--model")
      model = next();
    else if (k == "--text")
      text = next();
    else if (k == "--ids")
      ids_in = next();
    else if (k == "--ctxs")
      ctxs = next();
    else if (k == "--decode")
      decode = std::atoi(next().c_str());
    else if (k == "--batch")
      batches = next();
    else if (k == "--verify")
      verify = true;
    else if (k == "--spec")
      specs = next();
    else if (k == "--dspark")
      dspark = next();
    else if (k == "--depth")
      depth = std::atoi(next().c_str());
    else if (k == "--same")
      same = true;  // identical prompts in all sessions (debug)
    else if (k == "--ctx")
      ctx = static_cast<unsigned>(std::strtoul(next().c_str(), nullptr, 10));
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  std::vector<int> lens;
  {
    std::stringstream ss(ctxs);
    std::string t;
    while (std::getline(ss, t, ','))
      lens.push_back(std::atoi(t.c_str()));
  }
  std::string err;
  std::vector<std::int32_t> ids;
  if (!ids_in.empty()) {
    std::ifstream f(ids_in, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string raw = ss.str();
    ids.resize(raw.size() / 4);
    std::memcpy(ids.data(), raw.data(), ids.size() * 4);
  } else {
    auto tok = inkling2::text::Tokenizer::CreateFromGguf(model, &err);
    if (!tok) {
      std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
      return 1;
    }
    std::ifstream f(text, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    {
      const auto e = tok->Encode(ss.str().substr(0, 400000), false);
      ids.assign(e.begin(), e.end());
    }
  }
  std::vector<int> bs;
  {
    std::stringstream ss(batches);
    std::string t;
    while (std::getline(ss, t, ','))
      bs.push_back(std::atoi(t.c_str()));
  }
  const int maxb = bs.empty() ? 1 : *std::max_element(bs.begin(), bs.end());
  const int need =
      *std::max_element(lens.begin(), lens.end()) + (maxb - 1) * 997;
  if (static_cast<int>(ids.size()) < need) {
    std::fprintf(stderr, "only %zu tokens\n", ids.size());
    return 1;
  }
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader =
      gufo::core::GgufReader::OpenFile(model, &err, open);
  if (!reader) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  auto engine = gufo::models::inkling2::Engine::Create(reader, ctx, &err);
  if (!engine) {
    std::fprintf(stderr, "engine: %s\n", err.c_str());
    return 1;
  }
  if (!dspark.empty() && !engine->AttachDrafter(dspark, &err)) {
    std::fprintf(stderr, "dspark: %s\n", err.c_str());
    return 1;
  }
  auto session = engine->NewSession();
  const std::size_t V = engine->config().n_vocab_text;
  // ---- DSpark greedy speculative decoding vs plain greedy decoding ----
  if (!dspark.empty()) {
    auto amax = [](std::span<const float> l) {
      return static_cast<std::int32_t>(std::max_element(l.begin(), l.end()) -
                                       l.begin());
    };
    for (int C : lens) {
      const std::span<const std::int32_t> prompt(ids.data(),
                                                 static_cast<std::size_t>(C));
      auto ref = engine->NewSession();
      engine->Forward(*ref, prompt);
      std::vector<std::int32_t> want{amax(ref->logits())};
      const auto r0 = std::chrono::steady_clock::now();
      while (static_cast<int>(want.size()) < decode + 8) {
        engine->Decode(*ref, want.back());
        want.push_back(amax(ref->logits()));
      }
      const double ar =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - r0)
              .count() /
          (decode + 7);
      auto sp = engine->NewSession();
      engine->Forward(*sp, prompt);
      std::vector<std::int32_t> got{amax(sp->logits())};
      std::span<const float> out;
      long drafted = 0, accepted = 0, cycles = 0;
      // calibration: [bin of the draft probability] -> (seen, accepted), for
      // drafts whose predecessors were all accepted
      long cal[10][2] = {};
      double t_prop = 0, t_ver = 0, t_com = 0;
      using SC = std::chrono::steady_clock;
      auto ms = [](SC::time_point a, SC::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      const auto s0 = std::chrono::steady_clock::now();
      while (static_cast<int>(got.size()) < decode) {
        const std::int32_t anchor = got.back();
        std::int32_t d[8] = {};
        float conf[8] = {};
        int n = 0;
        const auto c0 = SC::now();
        if (engine->DraftReady(*sp)) {
          n = std::min<int>(depth,
                            static_cast<int>(engine->VerifyRows(*sp)) - 1);
          if (n > 0)
            engine->Propose(*sp, anchor, n, d, conf);
        }
        const auto c1 = SC::now();
        std::vector<std::int32_t> rows{anchor};
        rows.insert(rows.end(), d, d + n);
        engine->Verify(*sp, rows, &out);
        const auto c2 = SC::now();
        int j = 0;
        while (j < n &&
               amax(std::span<const float>(out.data() + j * V, V)) == d[j])
          ++j;
        for (int i = 0; i < j; ++i)
          got.push_back(d[i]);
        for (int i = 0; i <= j && i < n; ++i) {
          const int b = std::clamp(static_cast<int>(conf[i] * 10), 0, 9);
          ++cal[b][0];
          cal[b][1] += i < j;
        }
        got.push_back(amax(std::span<const float>(out.data() + j * V, V)));
        engine->CommitVerify(*sp, static_cast<std::size_t>(j + 1), out);
        const auto c3 = SC::now();
        t_prop += ms(c0, c1);
        t_ver += ms(c1, c2);
        t_com += ms(c2, c3);
        drafted += n;
        accepted += j;
        ++cycles;
      }
      const double st =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - s0)
              .count();
      int same = 0;
      while (same < static_cast<int>(got.size()) &&
             same < static_cast<int>(want.size()) && got[same] == want[same])
        ++same;
      std::printf(
          "dspark ctx %6d depth %d  tokens %zu identical-prefix %d  accepted "
          "%ld/%ld (%.1f%%), %.2f tokens/cycle  %.2f tok/s (AR %.2f)\n",
          C, depth, got.size(), same, accepted, drafted,
          drafted ? 100.0 * accepted / drafted : 0.0,
          static_cast<double>(got.size() - 1) / cycles, (got.size() - 1) / st,
          1.0 / ar);
      std::printf(
          "  per cycle: propose %.2f ms, verify %.2f ms, commit %.2f ms\n",
          t_prop / cycles, t_ver / cycles, t_com / cycles);
      std::printf("  calibration (draft p -> accepted):");
      for (int b = 0; b < 10; ++b)
        if (cal[b][0])
          std::printf(" %.1f:%ld/%ld", b / 10.0, cal[b][1], cal[b][0]);
      std::printf("\n");
      std::fflush(stdout);
    }
    return 0;
  }
  for (int C : (batches.empty() || !verify) ? lens : std::vector<int>{}) {
    engine->Reset(*session);
    const auto t0 = std::chrono::steady_clock::now();
    engine->Forward(*session, std::span<const std::int32_t>(
                                  ids.data(), static_cast<std::size_t>(C)));
    const auto t1 = std::chrono::steady_clock::now();
    const double pf = std::chrono::duration<double>(t1 - t0).count();
    std::int32_t next = static_cast<std::int32_t>(
        std::max_element(session->logits().begin(), session->logits().end()) -
        session->logits().begin());
    // warm-up decode steps (graph capture etc.), then timed
    for (int i = 0; i < 3; ++i) {
      engine->Decode(*session, next);
      next = static_cast<std::int32_t>(
          std::max_element(session->logits().begin(), session->logits().end()) -
          session->logits().begin());
    }
    const auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < decode; ++i) {
      engine->Decode(*session, next);
      next = static_cast<std::int32_t>(
          std::max_element(session->logits().begin(), session->logits().end()) -
          session->logits().begin());
    }
    const auto t3 = std::chrono::steady_clock::now();
    const double dc = std::chrono::duration<double>(t3 - t2).count();
    std::printf(
        "ctx %6d  prefill %8.1f tok/s (%.2f s)   decode %6.2f tok/s (%.2f "
        "ms/tok)  [V=%zu]\n",
        C, C / pf, pf, decode / dc, dc * 1000 / decode, V);
    std::fflush(stdout);
  }
  // ---- speculative verification ----
  if (!specs.empty()) {
    std::vector<int> rs;
    {
      std::stringstream ss(specs);
      std::string t;
      while (std::getline(ss, t, ','))
        rs.push_back(std::atoi(t.c_str()));
    }
    auto amax = [](std::span<const float> l) {
      return static_cast<std::int32_t>(std::max_element(l.begin(), l.end()) -
                                       l.begin());
    };
    for (int C : lens) {
      auto ref = engine->NewSession();
      const std::span<const std::int32_t> prompt(ids.data(),
                                                 static_cast<std::size_t>(C));
      engine->Forward(*ref, prompt);
      std::vector<std::int32_t> toks;
      std::vector<std::vector<float>> lg;
      toks.push_back(amax(ref->logits()));
      const auto r0 = std::chrono::steady_clock::now();
      for (int i = 0; i < decode + 64; ++i) {
        engine->Decode(*ref, toks.back());
        lg.push_back(ref->logits());
        toks.push_back(amax(ref->logits()));
      }
      const double ar =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - r0)
              .count() /
          (decode + 64);
      for (int R : rs) {
        auto sp = engine->NewSession();
        engine->Forward(*sp, prompt);
        std::uint32_t rng = 12345u + static_cast<std::uint32_t>(R);
        long rows = 0, bad = 0, steps = 0;
        double vt = 0;
        std::span<const float> out;
        int i = 0;
        while (i < decode) {
          const int n =
              std::min<int>(R, static_cast<int>(engine->VerifyRows(*sp)));
          std::vector<std::int32_t> t(toks.begin() + i, toks.begin() + i + n);
          rng = rng * 1664525u + 1013904223u;
          // the first wrong draft (n: none)
          const int wrong = n > 1 ? 1 + static_cast<int>((rng >> 8) % n) : n;
          if (wrong < n)
            t[wrong] = (t[wrong] + 1) % static_cast<std::int32_t>(V);
          const auto v0 = std::chrono::steady_clock::now();
          engine->Verify(*sp, t, &out);
          vt += std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              v0)
                    .count();
          ++steps;
          const int keep = wrong;
          for (int j = 0; j < keep; ++j) {
            ++rows;
            if (std::memcmp(out.data() + static_cast<std::size_t>(j) * V,
                            lg[i + j].data(), V * 4) != 0)
              ++bad;
          }
          engine->CommitVerify(*sp, static_cast<std::size_t>(keep), out);
          i += keep;
        }
        // one more plain decode after the last commit must match too
        engine->Decode(*sp, toks[i]);
        ++rows;
        if (std::memcmp(sp->logits().data(), lg[i].data(), V * 4) != 0)
          ++bad;
        // the same steps without copying the rows (device argmax only)
        double vt2 = 0;
        {
          auto s2 = engine->NewSession();
          engine->Forward(*s2, prompt);
          std::int32_t am[64];
          for (int k = 0; k < 16; ++k) {
            const int n2 =
                std::min<int>(R, static_cast<int>(engine->VerifyRows(*s2)));
            std::vector<std::int32_t> t2(toks.begin() + k,
                                         toks.begin() + k + n2);
            const auto w0 = std::chrono::steady_clock::now();
            engine->Verify(*s2, t2, nullptr, am);
            vt2 += std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - w0)
                       .count();
            engine->CommitVerify(*s2, 1, {});
          }
          vt2 /= 16;
        }
        std::printf(
            "spec R=%d ctx %6d  rows %ld mismatched %ld  verify %.2f ms/step "
            "(argmax only %.2f ms, decode %.2f ms)\n",
            R, C, rows, bad, vt * 1000 / steps, vt2 * 1000, ar * 1000);
        std::fflush(stdout);
      }
    }
    return 0;
  }
  // ---- batched decode ----
  using gufo::models::inkling2::Session;
  auto argmax = [](const std::vector<float>& l) {
    return static_cast<std::int32_t>(std::max_element(l.begin(), l.end()) -
                                     l.begin());
  };
  for (int B : bs) {
    for (int C : lens) {
      std::vector<std::unique_ptr<Session>> a,
          r;  // batched, reference (verify)
      std::vector<std::int32_t> nxt(B), nref(B);
      for (int b = 0; b < B; ++b) {
        a.push_back(engine->NewSession());
        const std::span<const std::int32_t> prompt(
            ids.data() + (same ? 0 : b * 997), static_cast<std::size_t>(C));
        engine->Forward(*a[b], prompt);
        nxt[b] = argmax(a[b]->logits());
        if (verify) {
          r.push_back(engine->NewSession());
          engine->Forward(*r[b], prompt);
          nref[b] = argmax(r[b]->logits());
        }
      }
      std::vector<Session*> sp;
      for (auto& x : a)
        sp.push_back(x.get());
      auto step = [&]() {
        engine->DecodeBatch(sp, nxt);
        for (int b = 0; b < B; ++b)
          nxt[b] = argmax(a[b]->logits());
      };
      double maxdiff = 0;
      long agree = 0, total = 0;
      for (int i = 0; i < 3; ++i)
        step();
      if (verify) {
        // reference: same tokens fed one session at a time; the batched
        // sessions are ahead by 3 steps, so replay
        for (int i = 0; i < 3; ++i)
          for (int b = 0; b < B; ++b) {
            engine->Decode(*r[b], nref[b]);
            nref[b] = argmax(r[b]->logits());
          }
      }
      const auto t2 = std::chrono::steady_clock::now();
      for (int i = 0; i < decode; ++i) {
        step();
        if (verify) {
          for (int b = 0; b < B; ++b) {
            engine->Decode(*r[b], nref[b]);
            nref[b] = argmax(r[b]->logits());
            const auto& x = a[b]->logits();
            const auto& y = r[b]->logits();
            for (std::size_t v = 0; v < x.size(); ++v)
              maxdiff = std::max(maxdiff, (double)std::abs(x[v] - y[v]));
            agree += nxt[b] == nref[b];
            ++total;
          }
        }
      }
      const auto t3 = std::chrono::steady_clock::now();
      const double dc = std::chrono::duration<double>(t3 - t2).count();
      if (verify) {
        std::printf(
            "batch %d ctx %6d  verify: max |dlogit| %.3e, argmax agree "
            "%ld/%ld\n",
            B, C, maxdiff, agree, total);
      } else {
        std::printf(
            "batch %d ctx %6d  aggregate %7.2f tok/s (%.2f ms/step)  per "
            "session %6.2f tok/s\n",
            B, C, B * decode / dc, dc * 1000 / decode, decode / dc);
      }
      std::fflush(stdout);
    }
  }
  return 0;
}
