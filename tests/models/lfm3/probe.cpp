#include <algorithm>
#include <fstream>
#include <iostream>

#include "src/models/lfm3/engine.hpp"
#include "src/models/lfm3/reference.hpp"
int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: lfm3_probe MODEL cpu|gpu OUTPUT [TOKEN ...]\n";
    return 2;
  }
  using namespace gufo::models::lfm3;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::cerr << error;
    return 1;
  }
  auto w = Weights::Bind(*reader, &error);
  auto tokenizer = Tokenizer::Load(*reader, &error);
  if (!w || !tokenizer) {
    std::cerr << error;
    return 1;
  }
  std::vector<std::uint32_t> tokens;
  for (int i = 4; i < argc; ++i)
    tokens.push_back(std::stoul(argv[i]));
  if (tokens.empty())
    tokens = tokenizer->Encode("The capital of France is");
  for (auto id : tokens)
    std::cerr << id << ' ';
  std::cerr << '\n';
  std::vector<float> logits(w->config.vocab);
  if (std::string_view(argv[2]) == "cpu") {
    Reference reference(*w, tokens.size() + 16);
    for (std::size_t i = 0; i < tokens.size(); ++i)
      reference.Step(tokens[i], i + 1 == tokens.size() ? std::span(logits)
                                                       : std::span<float>{});
  } else {
#if defined(ENGINE_ENABLE_HIP)
    auto model = Model::Load(argv[1], &error);
    if (!model) {
      std::cerr << error;
      return 1;
    }
    auto session = model->CreateSession(tokens.size() + 16, &error);
    if (!session || !session->Forward(tokens, logits, &error)) {
      std::cerr << error;
      return 1;
    }
#else
    std::cerr << "HIP disabled";
    return 1;
#endif
  }
  std::ofstream out(argv[3], std::ios::binary);
  out.write(reinterpret_cast<const char*>(logits.data()),
            logits.size() * sizeof(float));
  std::vector<std::uint32_t> order(logits.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::partial_sort(order.begin(), order.begin() + 10, order.end(),
                    [&](auto a, auto b) { return logits[a] > logits[b]; });
  for (int i = 0; i < 10; ++i)
    std::cerr << order[i] << ' ' << logits[order[i]] << ' '
              << tokenizer->Piece(order[i]) << '\n';
}
