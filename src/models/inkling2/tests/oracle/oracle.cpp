// Teacher-forcing oracle dump for the Inkling 2 Mini HIP engine.
//
//   inkling2_oracle --model GGUF --text FILE --n N --out dump.bin [--ids-out
//   ids.bin]
//                   [--ids ids.bin] [--ref refdump.bin] [--tap hidden.bin]
//
// Tokenizes the first N tokens of FILE (plain text, no special-token parsing)
// or reads the ids from --ids, evaluates them through the engine and writes,
// per position, the logsumexp of the soft-capped logits over the text
// vocabulary, the top-64 (id, logit) and, when --ref is given, our logits at
// the reference's top-64 ids. compare_dumps.py (beside this
// file) scores two dumps.
// llama_dump.cpp writes the same format from llama.cpp's libllama.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/inkling2/engine.hpp"
#include "src/models/inkling2/text/tokenizer.hpp"

namespace {
constexpr int kTop = 64;
constexpr char kMagic[4] = {'I', '2', 'O', 'R'};

struct Args {
  std::string model, text, out, ids_out, ids_in, ref, tap;
  bool tokenize_only = false;
  bool single = false;  // feed one token per Forward call (the decode path)
  std::size_t n = 1024;
  std::uint32_t ctx = 0;
};

bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  *out = ss.str();
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--model")
      a.model = next();
    else if (k == "--text")
      a.text = next();
    else if (k == "--n")
      a.n = std::strtoull(next().c_str(), nullptr, 10);
    else if (k == "--out")
      a.out = next();
    else if (k == "--ids-out")
      a.ids_out = next();
    else if (k == "--ids")
      a.ids_in = next();
    else if (k == "--ref")
      a.ref = next();
    else if (k == "--tap")
      a.tap = next();
    else if (k == "--tokenize-only")
      a.tokenize_only = true;
    else if (k == "--single")
      a.single = true;
    else if (k == "--ctx")
      a.ctx =
          static_cast<std::uint32_t>(std::strtoul(next().c_str(), nullptr, 10));
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (a.model.empty() || (a.out.empty() && !a.tokenize_only) ||
      (a.text.empty() && a.ids_in.empty())) {
    std::fprintf(stderr,
                 "usage: inkling2_oracle --model GGUF (--text FILE --n N | "
                 "--ids F) --out F [...]\n");
    return 2;
  }
  std::string err;
  std::vector<std::int32_t> ids;
  if (!a.ids_in.empty()) {
    std::string raw;
    if (!ReadFile(a.ids_in, &raw)) {
      std::fprintf(stderr, "cannot read %s\n", a.ids_in.c_str());
      return 1;
    }
    ids.resize(raw.size() / 4);
    std::memcpy(ids.data(), raw.data(), ids.size() * 4);
    if (a.n && a.n < ids.size())
      ids.resize(a.n);
  } else {
    auto tok = inkling2::text::Tokenizer::CreateFromGguf(a.model, &err);
    if (!tok) {
      std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
      return 1;
    }
    std::string text;
    if (!ReadFile(a.text, &text)) {
      std::fprintf(stderr, "cannot read %s\n", a.text.c_str());
      return 1;
    }
    const auto all = tok->Encode(text, false);
    if (all.size() < a.n) {
      std::fprintf(stderr, "text has only %zu tokens\n", all.size());
      return 1;
    }
    ids.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(a.n));
    std::fprintf(stderr, "tokenized %zu tokens (text has %zu)\n", ids.size(),
                 all.size());
  }
  if (!a.ids_out.empty()) {
    std::ofstream f(a.ids_out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(ids.data()),
            static_cast<std::streamsize>(ids.size() * 4));
  }

  if (a.tokenize_only)
    return 0;
  // reference ids per position (top-64), if given
  std::vector<std::int32_t> ref_ids;
  if (!a.ref.empty()) {
    std::string raw;
    if (!ReadFile(a.ref, &raw) || raw.size() < 16) {
      std::fprintf(stderr, "cannot read ref\n");
      return 1;
    }
    std::uint32_t n_pos, top;
    std::memcpy(&n_pos, raw.data() + 8, 4);
    std::memcpy(&top, raw.data() + 12, 4);
    const std::size_t rec = 4 + top * 4 * 3;  // lse, ids, logits, logits_at_ref
    if (raw.size() < 16 + static_cast<std::size_t>(n_pos) * rec) {
      std::fprintf(stderr, "ref dump is truncated\n");
      return 1;
    }
    ref_ids.resize(static_cast<std::size_t>(n_pos) * top);
    for (std::uint32_t p = 0; p < n_pos; ++p)
      std::memcpy(&ref_ids[static_cast<std::size_t>(p) * top],
                  raw.data() + 16 + p * rec + 4, top * 4);
    if (n_pos < ids.size()) {
      std::fprintf(stderr, "ref has fewer positions than ids\n");
      return 1;
    }
  }

  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader =
      gufo::core::GgufReader::OpenFile(a.model, &err, open);
  if (!reader) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const std::uint32_t ctx =
      a.ctx ? a.ctx : static_cast<std::uint32_t>(ids.size() + 16);
  auto engine = gufo::models::inkling2::Engine::Create(reader, ctx, &err);
  if (!engine) {
    std::fprintf(stderr, "engine: %s\n", err.c_str());
    return 1;
  }
  const auto t1 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "loaded %.1f GiB in %.1f s\n",
               engine->weight_bytes() / 1073741824.0,
               std::chrono::duration<double>(t1 - t0).count());
  auto session = engine->NewSession();

  std::vector<std::vector<float>> taps;
  std::vector<std::string> tap_names;
  if (!a.tap.empty()) {
    engine->SetTap(
        [&](int visit, const char* what, const std::vector<float>& row) {
          if (taps.size() < 48) {
            taps.push_back(row);
            tap_names.push_back(std::to_string(visit) + ":" + what);
          }
        });
  }

  std::ofstream out(a.out, std::ios::binary);
  const std::uint32_t version = 1,
                      n_pos = static_cast<std::uint32_t>(ids.size()),
                      top = kTop;
  out.write(kMagic, 4);
  out.write(reinterpret_cast<const char*>(&version), 4);
  out.write(reinterpret_cast<const char*>(&n_pos), 4);
  out.write(reinterpret_cast<const char*>(&top), 4);

  std::vector<int> order;
  auto sink = [&](std::size_t first, std::size_t count, const float* data) {
    // the engine's tap only fires for the first Forward call; ignore later ones
    const std::size_t V = engine->config().n_vocab_text;
    for (std::size_t r = 0; r < count; ++r) {
      const float* l = data + r * V;
      double mx = -1e30;
      for (std::size_t i = 0; i < V; ++i)
        mx = std::max<double>(mx, l[i]);
      double sum = 0;
      for (std::size_t i = 0; i < V; ++i)
        sum += std::exp(static_cast<double>(l[i]) - mx);
      const float lse = static_cast<float>(mx + std::log(sum));
      order.resize(V);
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(
          order.begin(), order.begin() + kTop, order.end(),
          [&](int x, int y) { return l[x] > l[y] || (l[x] == l[y] && x < y); });
      std::int32_t tid[kTop];
      float tl[kTop], at_ref[kTop];
      for (int i = 0; i < kTop; ++i) {
        tid[i] = order[i];
        tl[i] = l[order[i]];
      }
      for (int i = 0; i < kTop; ++i)
        at_ref[i] =
            ref_ids.empty() ? tl[i] : l[ref_ids[(first + r) * kTop + i]];
      out.write(reinterpret_cast<const char*>(&lse), 4);
      out.write(reinterpret_cast<const char*>(tid), sizeof tid);
      out.write(reinterpret_cast<const char*>(tl), sizeof tl);
      out.write(reinterpret_cast<const char*>(at_ref), sizeof at_ref);
    }
  };
  const auto t2 = std::chrono::steady_clock::now();
  if (a.single) {
    // decode path: one token per call (T = 1 kernels)
    for (std::size_t i = 0; i < ids.size(); ++i) {
      engine->Decode(*session, ids[i]);
      sink(i, 1, session->logits().data());
    }
  } else {
    engine->Forward(*session, ids, sink);
  }
  const auto t3 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t3 - t2).count();
  std::fprintf(stderr, "forward %zu tokens in %.1f s = %.1f tok/s\n",
               ids.size(), secs, ids.size() / secs);
  if (!a.tap.empty()) {
    std::ofstream f(a.tap, std::ios::binary);
    for (std::size_t i = 0; i < taps.size(); ++i) {
      f.write(reinterpret_cast<const char*>(taps[i].data()),
              static_cast<std::streamsize>(taps[i].size() * 4));
      std::fprintf(stderr, "tap %s\n", tap_names[i].c_str());
    }
  }
  return 0;
}
