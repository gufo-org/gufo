// Binds an EXL3 Flash-Next checkpoint on the host and prints its geometry.
// Usage: qwen38_flash_next_exl3_bind_probe <checkpoint directory>
#include <chrono>
#include <cstdio>
#include <string>

#include "src/models/qwen38_flash_next/exl3/pack.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

using namespace gufo::models::qwen38_flash_next;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <checkpoint directory>\n", argv[0]);
    return 2;
  }
  const auto start = std::chrono::steady_clock::now();
  std::string error;
  auto pack = exl3::Pack::Open(argv[1], &error);
  if (!pack) {
    std::fprintf(stderr, "open: %s\n", error.c_str());
    return 1;
  }
  const auto opened = std::chrono::steady_clock::now();
  HostTensors host;
  auto w = ModelWeights::BindExl3(*pack, host, &error);
  if (!w) {
    std::fprintf(stderr, "bind: %s\n", error.c_str());
    return 1;
  }
  auto mtp = MtpWeights::BindExl3(*pack, w->config, host, &error);
  if (!mtp && !error.empty()) {
    std::fprintf(stderr, "mtp: %s\n", error.c_str());
    return 1;
  }
  const auto bound = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const Config& c = w->config;
  std::printf("tensors %zu identity %s\n", pack->TensorCount(),
              pack->Identity().c_str());
  std::printf("layers %u hidden %u vocab %u ctx %u ple_layer %d rows %llu\n",
              c.num_layers, c.hidden_size, c.vocab_size, c.context_length,
              c.ple_layer, static_cast<unsigned long long>(c.ple_rows));
  std::printf("head K%d expert K%d dense K%d ngram K%d mtp %s\n",
              w->output.exl3->k, w->layers[0].ffn_gate_exps.exl3->k,
              w->layers[0].ssm_qkv.exl3->k, w->ple_exl3 ? w->ple_exl3->k : 0,
              mtp ? "yes" : "no");
  std::size_t host_bytes = 0;
  for (const auto& b : host.buffers)
    host_bytes += b.size();
  std::printf("host conversions %.1f MiB, open %.0f ms, bind %.0f ms\n",
              host_bytes / 1048576.0, ms(start, opened), ms(opened, bound));
  return 0;
}
