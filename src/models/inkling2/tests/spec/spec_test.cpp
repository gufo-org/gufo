// Speculative verification and DSpark against one-token decoding (GPU).
//
//   inkling2_spec_test --model GGUF [--dspark DRAFT.gguf] [--ids IDS]
//                      [--prompt 600] [--tokens 48]
//
// 1. A reference session decodes --tokens greedy tokens one at a time. Another
//    replays them through Engine::Verify in steps of 2, 3 and 4 rows whose
//    proposals are the reference tokens with one corrupted per step, and keeps
//    the rows up to it (CommitVerify). Every kept row's logits, and those of a
//    one-token decode after the last commit, must equal the reference's byte
//    for byte.
// 2. With --dspark: greedy speculative decoding with the drafter's proposals
//    must produce the reference's tokens.
// Exit 0 on success, 1 on a mismatch, 77 (skip) without a model.
#include <algorithm>
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

  // reference: one token at a time
  auto ref = engine->NewSession();
  engine->Forward(*ref, p);
  std::vector<std::int32_t> want{Argmax(ref->logits())};
  std::vector<std::vector<float>> lg;
  for (int i = 0; i < tokens + 8; ++i) {
    engine->Forward(*ref, std::span<const std::int32_t>(&want.back(), 1));
    lg.push_back(ref->logits());
    want.push_back(Argmax(ref->logits()));
  }

  int failures = 0;
  for (int R = 2; R <= 4; ++R) {
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
    engine->Forward(*sp, std::span<const std::int32_t>(&want[i], 1));
    ++rows;
    bad += std::memcmp(sp->logits().data(), lg[i].data(), V * 4) != 0;
    std::printf("verify R=%d: %ld rows, %ld mismatched\n", R, rows, bad);
    failures += bad != 0;
  }

  if (!dspark.empty()) {
    auto sp = engine->NewSession();
    engine->Forward(*sp, p);
    std::vector<std::int32_t> got{Argmax(sp->logits())};
    std::span<const float> out;
    long drafted = 0, accepted = 0;
    while (static_cast<int>(got.size()) < tokens) {
      std::int32_t d[8] = {};
      int n = 0;
      if (engine->DraftReady(*sp)) {
        n = std::min<int>(3, static_cast<int>(engine->VerifyRows(*sp)) - 1);
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
    while (same < static_cast<int>(got.size()) && got[same] == want[same])
      ++same;
    std::printf(
        "dspark: %zu tokens, %d equal to plain decoding, %ld of %ld "
        "proposals accepted\n",
        got.size(), same, accepted, drafted);
    failures += same != static_cast<int>(got.size());
  }
  std::printf(failures ? "FAIL\n" : "ALL PASS\n");
  return failures ? 1 : 0;
}
