// Speculative verification and DSpark against one-token decoding (GPU).
//
//   inkling2_spec_test --model GGUF [--dspark DRAFT.gguf] [--ids IDS]
//                      [--prompt 600] [--tokens 48]
//
// 1. A reference session decodes --tokens greedy tokens one at a time. Another
//    replays them through Engine::Verify in steps of 2 to 64 rows whose
//    proposals are the reference tokens with one corrupted per step, and keeps
//    the rows up to it (CommitVerify). Every kept row's logits, and those of a
//    one-token decode after the last commit, must equal the reference's byte
//    for byte.
// 2. With --dspark: greedy speculative decoding of a real chat prompt with the
//    drafter's proposals must produce plain decoding's tokens and keep at
//    least --min-acceptance of the proposals.
// 3. Several sessions at different positions (different attention buckets)
//    in one step: Engine::DecodeBatch, and Engine::VerifyBatch with mixed
//    widths and corruptions; every kept row must equal that session's lone
//    one-token decode byte for byte.
// 4. First, on the fresh engine (so the batch graphs are captured there):
//    sessions decoding in the yields of another session's layered prefill,
//    in batches of 2 to Engine::kMaxDecodeBatch - 1 (DecodeBatch) and in
//    batched verification. The prefill's logits must equal an unsplit
//    prefill's, and every decoded row the session's lone decode, byte for
//    byte.
// Exit 0 on success, 1 on a mismatch, 77 (skip) without a model.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/inkling2/engine.hpp"
#include "src/models/inkling2/text/chat_template.hpp"
#include "src/models/inkling2/text/tokenizer.hpp"

namespace {
using gufo::models::inkling2::Engine;

std::int32_t Argmax(std::span<const float> l) {
  return static_cast<std::int32_t>(std::max_element(l.begin(), l.end()) -
                                   l.begin());
}
}  // namespace

int main(int argc, char** argv) {
  std::string model, dspark, ids_in;
  int prompt = 600, tokens = 48;
  double min_acceptance = 0.2;  // 26% on the built-in prompt (UD-Q4_K_XL)
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--model")
      model = next();
    else if (k == "--dspark")
      dspark = next();
    else if (k == "--ids")
      ids_in = next();
    else if (k == "--prompt")
      prompt = std::atoi(next().c_str());
    else if (k == "--tokens")
      tokens = std::atoi(next().c_str());
    else if (k == "--min-acceptance")
      min_acceptance = std::atof(next().c_str());
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (model.empty()) {
    std::fprintf(stderr, "skip: no --model\n");
    return 77;
  }
  std::vector<std::int32_t> ids;
  if (!ids_in.empty()) {
    std::ifstream f(ids_in, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string raw = ss.str();
    ids.resize(raw.size() / 4);
    std::memcpy(ids.data(), raw.data(), ids.size() * 4);
  }
  for (int i = static_cast<int>(ids.size()); i < prompt; ++i)
    ids.push_back(1000 + (i * 7919) % 50000);  // filler prompt
  ids.resize(static_cast<std::size_t>(prompt));
  std::string err;
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader =
      gufo::core::GgufReader::OpenFile(model, &err, open);
  if (!reader) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  auto engine = Engine::Create(reader, 8192, &err);
  if (!engine) {
    std::fprintf(stderr, "engine: %s\n", err.c_str());
    return 1;
  }
  if (!dspark.empty() && !engine->AttachDrafter(dspark, &err)) {
    std::fprintf(stderr, "dspark: %s\n", err.c_str());
    return 1;
  }
  const std::size_t V = engine->config().n_vocab_text;
  const std::span<const std::int32_t> p(ids);
  int failures = 0;
  using gufo::models::inkling2::Session;

  {
    // 4. decoding inside a layered prefill's yields
    constexpr int kD = static_cast<int>(Engine::kMaxDecodeBatch) - 1;
    std::vector<std::vector<std::int32_t>> dp(kD);
    std::vector<std::unique_ptr<Session>> ds;
    for (int g = 0; g < kD; ++g) {
      dp[g] = ids;
      for (int i = 0; i < 150 * g; ++i)
        dp[g].push_back(3000 + (i * 7907 + g * 13) % 40000);
      ds.push_back(engine->NewSession());
      engine->Forward(*ds.back(), dp[g]);
    }
    std::vector<std::vector<std::int32_t>> dt(kD);        // tokens each decoded
    std::vector<std::vector<std::vector<float>>> dl(kD);  // and their logits
    for (int g = 0; g < kD; ++g)
      dt[g].push_back(Argmax(ds[g]->logits()));
    std::vector<std::int32_t> long_prompt = ids;
    for (int i = static_cast<int>(ids.size()); i < 5000; ++i)
      long_prompt.push_back(1500 + (i * 6151) % 55000);
    int round = 0;
    long verify_rows = 0;
    std::uint32_t rng = 4242u;
    std::span<const float> vout;
    gufo::models::inkling2::PrefillYield yield;
    yield.groups = 12;
    yield.yield = [&] {
      // widths 2 .. kD, then a batched verification of up to 8 rows
      const int w = 2 + round % (kD - 1);
      std::vector<Session*> sp;
      std::vector<std::int32_t> t;
      for (int g = 0; g < w; ++g) {
        sp.push_back(ds[g].get());
        t.push_back(dt[g].back());
      }
      engine->DecodeBatch(sp, t);
      for (int g = 0; g < w; ++g) {
        dl[g].push_back(ds[g]->logits());
        dt[g].push_back(Argmax(ds[g]->logits()));
      }
      if (round % 3 == 2) {
        // two rows per session for four sessions, the second a guess that
        // may be wrong; keep the rows up to the first wrong guess
        std::vector<Session*> vp;
        std::vector<std::size_t> cnt;
        std::vector<std::int32_t> vt;
        for (int g = 0; g < 4; ++g) {
          vp.push_back(ds[g].get());
          cnt.push_back(2);
          rng = rng * 1664525u + 1013904223u;
          vt.push_back(dt[g].back());
          vt.push_back(static_cast<std::int32_t>(1000 + (rng >> 8) % 50000));
        }
        engine->VerifyBatch(vp, cnt, vt, &vout, nullptr);
        for (int g = 0; g < 4; ++g) {
          const auto r0 = vout.subspan(static_cast<std::size_t>(2 * g) * V, V);
          const auto r1 =
              vout.subspan(static_cast<std::size_t>(2 * g + 1) * V, V);
          const bool second = Argmax(r0) == vt[2 * g + 1];
          dl[g].emplace_back(r0.begin(), r0.end());
          dt[g].push_back(Argmax(r0));
          if (second) {
            dl[g].emplace_back(r1.begin(), r1.end());
            dt[g].push_back(Argmax(r1));
          }
          verify_rows += second ? 2 : 1;
          engine->CommitVerify(
              *vp[g], second ? 2 : 1,
              vout.subspan(static_cast<std::size_t>(2 * g) * V, 2 * V));
        }
      }
      ++round;
    };
    auto pre = engine->NewSession();
    engine->Forward(*pre, long_prompt, {}, &yield);
    const std::vector<float> yielded = pre->logits();
    auto plain = engine->NewSession();
    engine->Forward(*plain, long_prompt);
    long bad = std::memcmp(yielded.data(), plain->logits().data(), V * 4) != 0;
    long rows = 0;
    for (int g = 0; g < kD; ++g) {
      auto lone = engine->NewSession();
      engine->Forward(*lone, dp[g]);
      for (std::size_t i = 0; i < dl[g].size(); ++i) {
        engine->Decode(*lone, dt[g][i]);
        ++rows;
        bad += std::memcmp(lone->logits().data(), dl[g][i].data(), V * 4) != 0;
      }
    }
    std::printf(
        "decode in prefill yields: %d yields, %ld decoded rows (%ld "
        "verified), %ld mismatched (prefill logits included)\n",
        round, rows, verify_rows, bad);
    failures += bad != 0 || round == 0;
  }

  // reference: one token at a time
  auto ref = engine->NewSession();
  engine->Forward(*ref, p);
  std::vector<std::int32_t> want{Argmax(ref->logits())};
  std::vector<std::vector<float>> lg;
  for (int i = 0; i < tokens + 64; ++i) {
    engine->Decode(*ref, want.back());
    lg.push_back(ref->logits());
    want.push_back(Argmax(ref->logits()));
  }

  // 2 to 8 rows (DSpark), and the wide copy steps (int8 WMMA projections,
  // grouped MoE, chunked attention)
  for (int R : {2, 3, 4, 5, 6, 7, 8, 9, 16, 33, 64}) {
    auto sp = engine->NewSession();
    engine->Forward(*sp, p);
    std::uint32_t rng = 12345u + static_cast<std::uint32_t>(R);
    long rows = 0, bad = 0;
    int i = 0;
    std::span<const float> out;
    while (i < tokens) {
      const int n = std::min<int>(R, static_cast<int>(engine->VerifyRows(*sp)));
      std::vector<std::int32_t> t(want.begin() + i, want.begin() + i + n);
      rng = rng * 1664525u + 1013904223u;
      const int wrong = n > 1 ? 1 + static_cast<int>((rng >> 8) % n) : n;
      if (wrong < n)
        t[wrong] = (t[wrong] + 1) % static_cast<std::int32_t>(V);
      engine->Verify(*sp, t, &out);
      for (int j = 0; j < wrong; ++j) {
        ++rows;
        bad += std::memcmp(out.data() + static_cast<std::size_t>(j) * V,
                           lg[i + j].data(), V * 4) != 0;
      }
      engine->CommitVerify(*sp, static_cast<std::size_t>(wrong), out);
      i += wrong;
    }
    engine->Decode(*sp, want[i]);
    ++rows;
    bad += std::memcmp(sp->logits().data(), lg[i].data(), V * 4) != 0;
    std::printf("verify R=%d: %ld rows, %ld mismatched\n", R, rows, bad);
    failures += bad != 0;
  }

  {
    // Session 0 is `ref`'s prompt; the others are longer filler prompts.
    constexpr int kS = 4;
    const int lens[kS] = {prompt, prompt + 1700, prompt + 3900, prompt + 900};
    std::vector<std::vector<std::int32_t>> pr(kS), wt(kS);
    std::vector<std::vector<std::vector<float>>> rl(kS);
    for (int g = 0; g < kS; ++g) {
      pr[g] = ids;
      for (int i = prompt; i < lens[g]; ++i)
        pr[g].push_back(2000 + (i * 104729 + g * 31) % 60000);
      auto r = engine->NewSession();
      engine->Forward(*r, pr[g]);
      wt[g].push_back(Argmax(r->logits()));
      for (int i = 0; i < tokens + 8; ++i) {
        engine->Decode(*r, wt[g].back());
        rl[g].push_back(r->logits());
        wt[g].push_back(Argmax(r->logits()));
      }
    }
    // plain batched decode
    {
      std::vector<std::unique_ptr<gufo::models::inkling2::Session>> ss;
      std::vector<gufo::models::inkling2::Session*> sp;
      for (int g = 0; g < kS; ++g) {
        ss.push_back(engine->NewSession());
        engine->Forward(*ss.back(), pr[g]);
        sp.push_back(ss.back().get());
      }
      long rows = 0, bad = 0;
      for (int i = 0; i < 12; ++i) {
        std::int32_t t[kS];
        for (int g = 0; g < kS; ++g)
          t[g] = wt[g][i];
        engine->DecodeBatch(sp, t);
        for (int g = 0; g < kS; ++g) {
          ++rows;
          bad +=
              std::memcmp(sp[g]->logits().data(), rl[g][i].data(), V * 4) != 0;
        }
      }
      std::printf("decode batch: %ld rows, %ld mismatched\n", rows, bad);
      failures += bad != 0;
    }
    // batched verification, widths and corruptions per session
    {
      std::vector<std::unique_ptr<gufo::models::inkling2::Session>> ss;
      for (int g = 0; g < kS; ++g) {
        ss.push_back(engine->NewSession());
        engine->Forward(*ss.back(), pr[g]);
      }
      std::uint32_t rng = 777u;
      int at[kS] = {};
      long rows = 0, bad = 0, steps = 0;
      std::span<const float> out;
      while (*std::min_element(at, at + kS) < tokens) {
        std::vector<gufo::models::inkling2::Session*> sp;
        std::vector<std::size_t> cnt;
        std::vector<int> who, wrong;
        std::vector<std::int32_t> t;
        std::size_t total = 0;
        for (int g = 0; g < kS; ++g) {
          rng = rng * 1664525u + 1013904223u;
          if (at[g] >= tokens || (rng >> 28) == 0)
            continue;  // sometimes a session sits a step out
          const std::size_t room = Engine::kMaxBatchVerifyRows - total;
          if (room == 0)
            break;
          rng = rng * 1664525u + 1013904223u;
          const int n = static_cast<int>(std::min<std::size_t>(
              {1 + (rng >> 8) % 4, room, engine->VerifyRows(*ss[g])}));
          rng = rng * 1664525u + 1013904223u;
          const int w = n > 1 ? 1 + static_cast<int>((rng >> 8) % n) : n;
          for (int j = 0; j < n; ++j)
            t.push_back(j == w ? (wt[g][at[g] + j] + 1) %
                                     static_cast<std::int32_t>(V)
                               : wt[g][at[g] + j]);
          sp.push_back(ss[g].get());
          cnt.push_back(static_cast<std::size_t>(n));
          who.push_back(g);
          wrong.push_back(w);
          total += static_cast<std::size_t>(n);
        }
        if (sp.empty())
          continue;
        engine->VerifyBatch(sp, cnt, t, &out, nullptr);
        ++steps;
        std::size_t r0 = 0;
        for (std::size_t k = 0; k < sp.size(); ++k) {
          const int g = who[k];
          for (int j = 0; j < wrong[k]; ++j) {
            ++rows;
            bad += std::memcmp(out.data() + (r0 + j) * V,
                               rl[g][at[g] + j].data(), V * 4) != 0;
          }
          r0 += cnt[k];
        }
        // the device copy for the first session, host rows for the others
        r0 = 0;
        for (std::size_t k = 0; k < sp.size(); ++k) {
          const auto mine = out.subspan(r0 * V, cnt[k] * V);
          engine->CommitVerify(*sp[k], static_cast<std::size_t>(wrong[k]),
                               k == 0 ? std::span<const float>{} : mine);
          at[who[k]] += wrong[k];
          r0 += cnt[k];
        }
      }
      for (int g = 0; g < kS; ++g) {
        engine->Decode(*ss[g], wt[g][at[g]]);
        ++rows;
        bad += std::memcmp(ss[g]->logits().data(), rl[g][at[g]].data(),
                           V * 4) != 0;
      }
      std::printf("verify batch: %ld steps, %ld rows, %ld mismatched\n", steps,
                  rows, bad);
      failures += bad != 0;
    }
  }

  if (!dspark.empty()) {
    // a real chat prompt: filler tokens would give the drafter nothing to
    // predict
    auto tok = inkling2::text::Tokenizer::CreateFromGguf(*reader, &err);
    inkling2::text::PromptTokens pt;
    inkling2::text::RenderOptions ro;
    ro.generation_prefix = inkling2::text::RenderOptions::Prefix::kHeader;
    if (!tok || !inkling2::text::RenderAndTokenize(
                    *tok,
                    {inkling2::text::Message::Text(
                        "user",
                        "Write a C function that reverses a singly linked "
                        "list in place. Explain each step in a comment.")},
                    {}, ro, &pt, &err)) {
      std::fprintf(stderr, "chat prompt: %s\n", err.c_str());
      return 1;
    }
    const std::vector<std::int32_t> chat(pt.ids.begin(), pt.ids.end());
    auto plain = engine->NewSession();
    engine->Forward(*plain, chat);
    std::vector<std::int32_t> want{Argmax(plain->logits())};
    while (static_cast<int>(want.size()) < tokens + 8) {
      engine->Decode(*plain, want.back());
      want.push_back(Argmax(plain->logits()));
    }
    auto sp = engine->NewSession();
    engine->Forward(*sp, chat);
    std::vector<std::int32_t> got{Argmax(sp->logits())};
    std::span<const float> out;
    long drafted = 0, accepted = 0;
    while (static_cast<int>(got.size()) < tokens) {
      std::int32_t d[8] = {};
      int n = 0;
      if (engine->DraftReady(*sp)) {
        n = std::min<int>(7, static_cast<int>(engine->VerifyRows(*sp)) - 1);
        if (n > 0)
          n = engine->Propose(*sp, got.back(), n, d);
      }
      std::vector<std::int32_t> rows{got.back()};
      rows.insert(rows.end(), d, d + n);
      engine->Verify(*sp, rows, &out);
      int j = 0;
      while (j < n &&
             Argmax(out.subspan(static_cast<std::size_t>(j) * V, V)) == d[j])
        ++j;
      for (int k = 0; k < j; ++k)
        got.push_back(d[k]);
      got.push_back(Argmax(out.subspan(static_cast<std::size_t>(j) * V, V)));
      engine->CommitVerify(*sp, static_cast<std::size_t>(j + 1), out);
      drafted += n;
      accepted += j;
    }
    int same = 0;
    const int n = std::min(static_cast<int>(got.size()), tokens + 8);
    while (same < n && got[same] == want[same])
      ++same;
    const double rate =
        drafted ? static_cast<double>(accepted) / static_cast<double>(drafted)
                : 0.0;
    std::printf(
        "dspark: %d tokens, %d equal to plain decoding, %ld of %ld "
        "proposals accepted (%.1f%%, at least %.1f%% required)\n",
        n, same, accepted, drafted, 100.0 * rate, 100.0 * min_acceptance);
    failures += same != n || rate < min_acceptance;
  }
  if (!dspark.empty()) {
    // proposals of several sessions in one drafter pass equal lone ones
    std::vector<std::unique_ptr<gufo::models::inkling2::Session>> ss;
    std::vector<std::int32_t> anchors;
    for (int g = 0; g < 3; ++g) {
      std::vector<std::int32_t> q = ids;
      for (int i = 0; i < 300 * g; ++i)
        q.push_back(ids[static_cast<std::size_t>(i) % ids.size()]);
      ss.push_back(engine->NewSession());
      engine->Forward(*ss.back(), q);
      anchors.push_back(Argmax(ss.back()->logits()));
    }
    std::int32_t solo[3][8] = {}, both[3][8] = {};
    float csolo[3][8] = {}, cboth[3][8] = {};
    int nsolo[3] = {};
    for (int g = 0; g < 3; ++g)
      nsolo[g] = engine->Propose(*ss[g], anchors[g], 7, solo[g], csolo[g]);
    std::vector<Engine::ProposeRequest> rq(3);
    for (int g = 0; g < 3; ++g)
      rq[g] = {.session = ss[g].get(),
               .anchor = anchors[g],
               .n = 7,
               .out = both[g],
               .confidence = cboth[g]};
    engine->ProposeBatch(rq);
    int bad = 0;
    for (int g = 0; g < 3; ++g)
      bad += rq[g].done != nsolo[g] ||
             std::memcmp(solo[g], both[g], sizeof solo[g]) != 0 ||
             std::memcmp(csolo[g], cboth[g], sizeof csolo[g]) != 0;
    std::printf("propose batch: 3 sessions, %d differ from lone proposals\n",
                bad);
    failures += bad != 0;
  }
  {
    // GPU linear sampling vs host FP64 reference + ReadVerifyRows check
    auto s = engine->NewSession();
    engine->Forward(*s, p);
    std::vector<std::int32_t> toks(want.begin(), want.begin() + 8);
    std::span<const float> out;
    std::int32_t gpu_toks[8] = {};
    Engine::LinearSample ls[8];
    for (int j = 0; j < 8; ++j)
      ls[j] = {.temperature = 1.0f,
               .min_p = j % 2 ? 0.05f : 0.0f,
               .u = 0.07 + 0.11 * j};
    engine->Verify(*s, toks, &out, nullptr,
                   std::span<const Engine::LinearSample>(ls, 8), gpu_toks);
    int ls_bad = 0;
    for (int j = 0; j < 8; ++j) {
      const auto row = out.subspan(static_cast<std::size_t>(j) * V, V);
      double mx = -INFINITY;
      for (float v : row)
        if (std::isfinite(v) && v > mx)
          mx = v;
      const double thr =
          ls[j].min_p > 0 ? mx + ls[j].temperature *
                                     std::log(static_cast<double>(ls[j].min_p))
                          : -INFINITY;
      double z = 0.0;
      for (float v : row)
        if (std::isfinite(v) && v >= thr)
          z += std::exp((static_cast<double>(v) - mx) / ls[j].temperature);
      const double target = ls[j].u * z;
      double run = 0.0;
      int cpu_tok = static_cast<int>(V) - 1;
      for (std::size_t i = 0; i < V; ++i) {
        if (std::isfinite(row[i]) && row[i] >= thr) {
          run +=
              std::exp((static_cast<double>(row[i]) - mx) / ls[j].temperature);
          if (target < run) {
            cpu_tok = static_cast<int>(i);
            break;
          }
        }
      }
      if (gpu_toks[j] != cpu_tok)
        ++ls_bad;
    }
    // ReadVerifyRows must copy the row from the device again: Verify's
    // output lives in the same host buffer, so poison the row first.
    const std::vector<float> want3(out.begin() + 3 * V, out.begin() + 4 * V);
    std::fill_n(const_cast<float*>(out.data()) + 3 * V, V, std::nanf(""));
    std::span<const float> row3;
    engine->ReadVerifyRows(3, 1, &row3);
    ls_bad +=
        row3.size() != V || std::memcmp(row3.data(), want3.data(), V * 4) != 0;
    engine->CommitVerify(*s, 1, {});
    std::printf("gpu linear sample + lazy row read: 8 rows, %d errors\n",
                ls_bad);
    failures += ls_bad != 0;
  }
  std::printf(failures ? "FAIL\n" : "ALL PASS\n");
  return failures ? 1 : 0;
}
