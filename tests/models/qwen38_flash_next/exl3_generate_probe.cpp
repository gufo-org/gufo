// Loads a Flash-Next artifact (GGUF or EXL3 checkpoint directory) through
// Model::Load, prefills a prompt and greedily decodes a few tokens. Prints
// the text, the first step's top logits and the timings.
// Usage: qwen38_flash_next_exl3_generate_probe <model> [tokens] [prompt]
//        qwen38_flash_next_exl3_generate_probe <model> --nll <text> <tokens>
//            <out.tsv>   per-position NLL and argmax of a fixed text
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"

using namespace gufo::models::qwen38_flash_next;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model> [tokens] [prompt]\n", argv[0]);
    return 2;
  }
  const bool nll = argc > 2 && std::string(argv[2]) == "--nll";
  const int steps = argc > 2 && !nll ? std::atoi(argv[2]) : 32;
  const std::string question =
      argc > 3 ? argv[3]
               : "What is the capital of France? Answer in one sentence.";
  const std::string prompt = "<|im_start|>user\n" + question +
                             "<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
                             "</think>\n\n";
  using clock = std::chrono::steady_clock;
  const auto ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const auto t0 = clock::now();
  std::string error;
  ModelOptions options;
  options.max_context = 8192;
  auto model = Model::Load(argv[1], options, &error);
  if (!model) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  const auto t1 = clock::now();
  auto session = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      8192, &error);
  if (!session) {
    std::fprintf(stderr, "session: %s\n", error.c_str());
    return 1;
  }
  if (nll) {
    if (argc < 6) {
      std::fprintf(stderr, "--nll needs <text> <tokens> <out.tsv>\n");
      return 2;
    }
    std::ifstream in(argv[3]);
    std::stringstream text;
    text << in.rdbuf();
    auto ids = model->Tokenize(text.str());
    ids.resize(std::min<std::size_t>(ids.size(), std::atoi(argv[4])));
    std::ofstream out(argv[5]);
    double total = 0.0;
    std::size_t hits = 0;
    if (!session->Evaluate(ids[0], &error)) {
      std::fprintf(stderr, "decode: %s\n", error.c_str());
      return 1;
    }
    for (std::size_t i = 1; i < ids.size(); ++i) {
      const auto logits = session->Logits();
      const float top = *std::max_element(logits.begin(), logits.end());
      double sum = 0.0;
      for (const float v : logits)
        sum += std::exp(static_cast<double>(v - top));
      const double lp = logits[ids[i]] - top - std::log(sum);
      const auto best = static_cast<std::int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      total -= lp;
      hits += best == ids[i];
      out << i << '\t' << ids[i] << '\t' << best << '\t' << -lp << '\n';
      if (!session->Evaluate(ids[i], &error)) {
        std::fprintf(stderr, "decode: %s\n", error.c_str());
        return 1;
      }
    }
    std::printf("nll %.5f over %zu tokens, next-token top1 %.4f\n",
                total / static_cast<double>(ids.size() - 1), ids.size() - 1,
                hits / static_cast<double>(ids.size() - 1));
    return 0;
  }
  if (argc > 2 && std::string(argv[2]) == "--prefill-check") {
    // The prompt pass's final logits against the decode path's, token by
    // token over the same text: KL divergence and top-1 agreement.
    if (argc < 5) {
      std::fprintf(stderr, "--prefill-check needs <text> <tokens>\n");
      return 2;
    }
    std::ifstream in(argv[3]);
    std::stringstream text;
    text << in.rdbuf();
    auto ids = model->Tokenize(text.str());
    ids.resize(std::min<std::size_t>(ids.size(), std::atoi(argv[4])));
    if (!session->Sync(ids, &error)) {
      std::fprintf(stderr, "prefill: %s\n", error.c_str());
      return 1;
    }
    const auto p = session->Logits();
    std::vector<float> prefill(p.begin(), p.end());
    auto other = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      8192, &error);
    for (const auto id : ids) {
      if (!other->Evaluate(id, &error)) {
        std::fprintf(stderr, "decode: %s\n", error.c_str());
        return 1;
      }
    }
    const auto d = other->Logits();
    const auto softmax = [](std::span<const float> v) {
      std::vector<double> out(v.size());
      const float top = *std::max_element(v.begin(), v.end());
      double sum = 0.0;
      for (std::size_t i = 0; i < v.size(); ++i)
        sum += out[i] = std::exp(static_cast<double>(v[i] - top));
      for (auto& x : out)
        x /= sum;
      return out;
    };
    const auto pa = softmax(prefill);
    const auto pb = softmax(d);
    double kl = 0.0;
    for (std::size_t i = 0; i < pa.size(); ++i)
      if (pb[i] > 0 && pa[i] > 0)
        kl += pb[i] * std::log(pb[i] / pa[i]);
    const auto ta =
        std::max_element(prefill.begin(), prefill.end()) - prefill.begin();
    const auto tb = std::max_element(d.begin(), d.end()) - d.begin();
    std::printf(
        "prefill vs decode after %zu tokens: KL %.6f, top1 %s (%ld/%ld)\n",
        ids.size(), kl, ta == tb ? "same" : "differs", static_cast<long>(ta),
        static_cast<long>(tb));
    return 0;
  }
  std::vector<std::int32_t> tokens = model->Tokenize(prompt);
  std::printf("prompt tokens %zu:", tokens.size());
  for (std::size_t i = 0; i < std::min<std::size_t>(tokens.size(), 12); ++i)
    std::printf(" %d", tokens[i]);
  std::printf("\n");
  const auto t2 = clock::now();
  if (!session->Sync(tokens, &error)) {
    std::fprintf(stderr, "prefill: %s\n", error.c_str());
    return 1;
  }
  const auto t3 = clock::now();
  std::vector<std::int32_t> out;
  for (int s = 0; s < steps; ++s) {
    const auto logits = session->Logits();
    if (s == 0) {
      std::vector<int> order(logits.size());
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                        [&](int a, int b) { return logits[a] > logits[b]; });
      std::printf("top5:");
      for (int i = 0; i < 5; ++i)
        std::printf(" %d(%s)=%.3f", order[i],
                    model->TokenText(order[i]).c_str(), logits[order[i]]);
      std::printf("\n");
    }
    const auto best = static_cast<std::int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    out.push_back(best);
    if (model->IsStopToken(best))
      break;
    if (!session->Evaluate(best, &error)) {
      std::fprintf(stderr, "decode: %s\n", error.c_str());
      return 1;
    }
  }
  const auto t4 = clock::now();
  std::printf("text: %s\n", model->Decode(out).c_str());
  std::printf(
      "load %.0f ms, prefill %zu tok %.1f ms (%.1f tok/s), decode %zu "
      "tok %.1f ms (%.2f tok/s)\n",
      ms(t0, t1), tokens.size(), ms(t2, t3),
      tokens.size() / (ms(t2, t3) / 1000.0), out.size(), ms(t3, t4),
      out.size() / (ms(t3, t4) / 1000.0));
  return 0;
}
