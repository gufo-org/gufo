// Binds real Inkling 2 Mini GGUFs and checks every tensor. Usage:
//   inkling2_weights_test [--equal] FILE.gguf...
// With --equal all files must hold byte-identical tensors under the
// layout-independent names (the two Unsloth layouts of one quant do). With no
// file the test skips, so it passes where no model is installed.

#include "src/models/inkling2/weights.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/models/inkling2/tensor_ref.hpp"

namespace {

using namespace gufo::models::inkling2;

int g_failures = 0;

#define CHECK(cond, ...)                                                    \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                   #cond);                                                  \
      std::fprintf(stderr, "  " __VA_ARGS__);                               \
      std::fprintf(stderr, "\n");                                           \
      ++g_failures;                                                         \
    }                                                                       \
  } while (false)

struct Loaded {
  std::string path;
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::optional<Inkling2Weights> weights;
};

bool Contains(const std::string& text, const char* part) {
  return text.find(part) != std::string::npos;
}

/// Independent expectations per quant, from the artifact census.
void CheckQuant(const Loaded& f) {
  const auto& w = *f.weights;
  const bool nvfp4 = Contains(f.path, "UD-Q4_K_XL");
  const bool iq4nl = Contains(f.path, "IQ4_NL");
  const bool q8 = Contains(f.path, "Q8_0") && !iq4nl;
  for (std::uint32_t il = 0; il < 24; ++il) {
    const auto& l = w.layers[il];
    const bool tail = il >= 20;  // layers 20-23 are 8-bit in the mixed quants
    CHECK(l.is_global == (il % 6 == 5), "layer %u", il);
    CHECK(l.is_core == (il >= 6 && il < 18), "layer %u", il);
    CHECK(static_cast<bool>(l.attn_res_query) == (il != 0), "layer %u", il);
    CHECK(l.attn_res_query.ne[1] == (l.is_global && il != 0 ? 3U : 1U),
          "layer %u", il);
    CHECK(static_cast<bool>(l.attn_norm) == !l.is_global, "layer %u", il);
    CHECK(static_cast<bool>(l.attn_k_conv_proj) == l.is_global, "layer %u", il);
    CHECK(static_cast<bool>(l.msg_query[kStreamFfn]) == l.is_core, "layer %u",
          il);
    if (nvfp4) {
      const auto want = tail ? GgmlType::kQ8_0 : kNvfp4;
      CHECK(l.ffn_gate_up_exps.type == want && l.ffn_down_exps.type == want,
            "layer %u", il);
      CHECK(static_cast<bool>(l.ffn_gate_up_scale) == !tail, "layer %u", il);
      CHECK(static_cast<bool>(l.ffn_down_scale) == !tail, "layer %u", il);
    } else if (iq4nl) {
      const auto want = tail ? GgmlType::kQ8_0 : GgmlType::kIQ4_NL;
      CHECK(l.ffn_gate_up_exps.type == want && l.ffn_down_exps.type == want,
            "layer %u", il);
    } else if (q8) {
      CHECK(l.ffn_gate_up_exps.type == GgmlType::kQ8_0, "layer %u", il);
      CHECK(l.attn_q.type == GgmlType::kQ8_0, "layer %u", il);
    }
  }
  CHECK(w.token_embd.type == GgmlType::kQ8_0 || !(iq4nl || q8 || nvfp4));
}

/// NVFP4 sizes come from the extension; the reader's reserved span (up to the
/// next tensor) may exceed them only by alignment padding.
void CheckSizing(const Loaded& f) {
  std::size_t nvfp4_tensors = 0;
  std::uint64_t total = 0;
  ForEachTensor(*f.weights, [&](const std::string& name, const TensorRef& t) {
    CHECK(t.data != nullptr && t.size_bytes > 0, "%s", name.c_str());
    if (t.type == kNvfp4) {
      ++nvfp4_tensors;
      CHECK(t.size_bytes == Nvfp4Bytes(t.Elements()), "%s", name.c_str());
    }
    if (const auto* info = f.reader->FindTensor(name)) {
      CHECK(info->size_bytes >= t.size_bytes &&
                info->size_bytes - t.size_bytes < 32,
            "%s: reserved %zu, sized %zu", name.c_str(), info->size_bytes,
            t.size_bytes);
      total += t.size_bytes;
    } else if (name.find(".msg_query.") == std::string::npos &&
               name.find(".loop.") == std::string::npos) {
      CHECK(false, "%s is not a tensor of the file", name.c_str());
    }
  });
  std::printf(
      "  NVFP4 tensors: %zu, sized tensor bytes: %.2f GiB of %.2f GiB file\n",
      nvfp4_tensors, static_cast<double>(total) / (1ULL << 30),
      static_cast<double>(f.reader->GetSize()) / (1ULL << 30));
  CHECK(total <= f.reader->GetSize());
}

/// Tensors that equal bytes under the common naming: the two layouts of one
/// quant.
void CheckEqual(const Loaded& a, const Loaded& b) {
  std::map<std::string, TensorRef> left;
  ForEachTensor(*a.weights,
                [&](const std::string& n, const TensorRef& t) { left[n] = t; });
  std::size_t compared = 0;
  ForEachTensor(*b.weights, [&](const std::string& n, const TensorRef& t) {
    const auto it = left.find(n);
    CHECK(it != left.end(), "%s only in %s", n.c_str(), b.path.c_str());
    if (it == left.end()) {
      return;
    }
    const auto& u = it->second;
    CHECK(u.type == t.type && u.ne == t.ne && u.size_bytes == t.size_bytes,
          "%s", n.c_str());
    CHECK(std::memcmp(u.data, t.data, t.size_bytes) == 0, "%s differs",
          n.c_str());
    left.erase(it);
    ++compared;
  });
  CHECK(left.empty(), "%zu tensors only in %s", left.size(), a.path.c_str());
  std::printf("equal: %zu tensors byte-identical across %s and %s\n", compared,
              a.path.c_str(), b.path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  bool equal = false;
  std::vector<Loaded> files;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--equal") {
      equal = true;
      continue;
    }
    Loaded f;
    f.path = argv[i];
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    f.reader = gufo::core::GgufReader::OpenFile(
        f.path, &error, {.allow_unsized_tensors = true});
    if (!f.reader) {
      std::fprintf(stderr, "%s: cannot open: %s\n", f.path.c_str(),
                   error.c_str());
      return 1;
    }
    std::string why;
    f.weights = Inkling2Weights::Bind(*f.reader, &why);
    const auto ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    if (!f.weights) {
      std::fprintf(stderr, "%s: refused: %s\n", f.path.c_str(), why.c_str());
      return 1;
    }
    std::printf("%s (open + bind %.0f ms, %llu file tensors)\n%s",
                f.path.c_str(), ms,
                static_cast<unsigned long long>(f.reader->GetTensorCount()),
                Summarize(*f.weights).c_str());
    CheckQuant(f);
    CheckSizing(f);
    files.push_back(std::move(f));
  }
  if (files.empty()) {
    std::puts("inkling2 weights test: no files given, skipped");
    return 0;
  }
  if (equal) {
    for (std::size_t i = 1; i < files.size(); ++i) {
      CheckEqual(files[0], files[i]);
    }
  }
  std::puts(g_failures == 0 ? "inkling2 weights test: OK"
                            : "inkling2 weights test: FAILED");
  return g_failures == 0 ? 0 : 1;
}
