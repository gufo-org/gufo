// Runs the ROCm runtime over a prompt: dumps float32 logits for every prompt
// position (the layout compare_logits.py reads), optionally continues
// greedily, and reports prefill and decode rates.
//
//   nemotron35_bolt_runtime_probe --model MODEL.gguf
//       (--tokens 1,2,3 | --text TEXT | --text-file PATH) [--dump logits.bin]
//       [--generate N]
//       [--context N] [--step] [--depth N]
//       [--drafter DRAFT.gguf [--temperature T]]
// --step appends the prompt one token at a time, so the dumped rows come
// from the decode path instead of one prefill. --depth N prefills the first
// N prompt tokens as a cached prefix, timed on their own, before the timed
// prefill of the rest (pp at depth N).
// With a drafter, generation runs draft-and-verify rounds and reports the
// acceptance; temperature 0 (the default) decodes greedily either way.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/nemotron35_bolt/engine.hpp"
#include "src/models/nemotron35_bolt/speculative.hpp"

namespace nb = gufo::models::nemotron35_bolt;

namespace {

std::vector<std::int32_t> ParseTokens(const std::string& text) {
  std::vector<std::int32_t> out;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t end = text.find(',', start);
    out.push_back(std::atoi(text.substr(start, end - start).c_str()));
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return out;
}

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, tokens_arg, text, dump, drafter_path;
  float temperature = 0.0F;
  std::uint32_t generate = 0;
  std::uint32_t context = 8192;
  bool step = false;
  std::size_t depth = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--model")
      model_path = value();
    else if (arg == "--tokens")
      tokens_arg = value();
    else if (arg == "--text")
      text = value();
    else if (arg == "--text-file") {
      std::ifstream in(value(), std::ios::binary);
      text.assign(std::istreambuf_iterator<char>(in), {});
    } else if (arg == "--dump")
      dump = value();
    else if (arg == "--drafter")
      drafter_path = value();
    else if (arg == "--temperature")
      temperature = std::stof(value());
    else if (arg == "--generate")
      generate = static_cast<std::uint32_t>(std::stoul(value()));
    else if (arg == "--context")
      context = static_cast<std::uint32_t>(std::stoul(value()));
    else if (arg == "--step")
      step = true;
    else if (arg == "--depth")
      depth = std::stoul(value());
    else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty() || (tokens_arg.empty() && text.empty())) {
    std::fprintf(stderr,
                 "usage: %s --model MODEL.gguf (--tokens 1,2,3 | --text TEXT | "
                 "--text-file PATH) [--dump logits.bin] [--generate N] "
                 "[--context N] [--step] [--depth N]\n",
                 argv[0]);
    return 2;
  }

  std::string error;
  const auto load_start = std::chrono::steady_clock::now();
  nb::ModelOptions options;
  options.max_context = context;
  auto model = nb::Model::Load(model_path, options, &error);
  if (!model) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  std::printf("loaded %.2f GB in %.1f s\n", model->ResidentBytes() / 1e9,
              Seconds(load_start));
  if (!drafter_path.empty() && !model->LoadDrafter(drafter_path, &error)) {
    std::fprintf(stderr, "drafter: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession(context, &error);
  if (!session) {
    std::fprintf(stderr, "session: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::int32_t> tokens;
  if (!tokens_arg.empty()) {
    tokens = ParseTokens(tokens_arg);
  } else {
    for (auto id : model->tokenizer().Encode(text, true))
      tokens.push_back(static_cast<std::int32_t>(id));
  }
  const std::size_t vocab = model->config().vocab_size;
  if (depth > 0) {
    if (depth >= tokens.size()) {
      std::fprintf(stderr, "--depth must leave prompt tokens to prefill\n");
      return 2;
    }
    std::vector<float> last(vocab);
    const auto depth_start = std::chrono::steady_clock::now();
    if (!session->Append(std::span<const std::int32_t>(tokens.data(), depth),
                         last, &error)) {
      std::fprintf(stderr, "depth prefill: %s\n", error.c_str());
      return 1;
    }
    const double seconds = Seconds(depth_start);
    std::printf("depth prefill %zu tokens in %.3f s (%.1f tokens/s)\n", depth,
                seconds, static_cast<double>(depth) / seconds);
    tokens.erase(tokens.begin(),
                 tokens.begin() + static_cast<std::ptrdiff_t>(depth));
  }
  std::vector<float> logits(dump.empty() ? vocab : tokens.size() * vocab);
  const auto prefill_start = std::chrono::steady_clock::now();
  if (step) {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      const std::size_t row = dump.empty() ? 0 : i;
      if (!session->Append(std::span<const std::int32_t>(&tokens[i], 1),
                           std::span<float>(logits.data() + row * vocab, vocab),
                           &error)) {
        std::fprintf(stderr, "prefill: %s\n", error.c_str());
        return 1;
      }
    }
  } else if (!session->Append(tokens, logits, &error)) {
    std::fprintf(stderr, "prefill: %s\n", error.c_str());
    return 1;
  }
  const double prefill = Seconds(prefill_start);
  std::printf("prefill %zu tokens in %.3f s (%.1f tokens/s)\n", tokens.size(),
              prefill, static_cast<double>(tokens.size()) / prefill);
  if (!dump.empty()) {
    std::ofstream out(dump, std::ios::binary);
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * sizeof(float)));
  }

  if (generate > 0) {
    nb::Frontier frontier;
    frontier.logits.assign(logits.end() - static_cast<std::ptrdiff_t>(vocab),
                           logits.end());
    std::vector<float>& row = frontier.logits;
    std::vector<std::int32_t> generated;
    const auto decode_start = std::chrono::steady_clock::now();
    gufo::sampling::SamplingConfig sampling;
    sampling.temperature = temperature;
    sampling.top_p = temperature > 0 ? 0.95F : 1.0F;
    sampling.seed = 1;
    gufo::sampling::SamplerState sampler(sampling);
    std::size_t rounds = 0, drafted = 0, accepted = 0;
    const auto is_stop = [&](std::int32_t t) {
      return model->tokenizer().IsStop(static_cast<nb::TokenId>(t));
    };
    while (model->drafter() != nullptr && generated.size() < generate) {
      nb::SpeculativeRound round;
      if (!nb::SpeculativeStep(
              *session, sampler, frontier,
              static_cast<std::uint32_t>(generate - generated.size()), is_stop,
              false, &round, &error)) {
        std::fprintf(stderr, "decode: %s\n", error.c_str());
        return 1;
      }
      generated.insert(generated.end(), round.tokens.begin(),
                       round.tokens.end());
      ++rounds;
      drafted += round.drafted;
      accepted += round.accepted;
      if (round.stop)
        break;
    }
    if (rounds != 0)
      std::printf("rounds %zu, drafted %zu, accepted %zu (%.2f per round)\n",
                  rounds, drafted, accepted,
                  static_cast<double>(accepted) / static_cast<double>(rounds));
    for (std::uint32_t i = 0; model->drafter() == nullptr && i < generate;
         ++i) {
      const auto next = static_cast<std::int32_t>(
          std::max_element(row.begin(), row.end()) - row.begin());
      generated.push_back(next);
      if (model->tokenizer().IsStop(static_cast<nb::TokenId>(next)))
        break;
      const std::int32_t one[] = {next};
      if (!session->Append(one, row, &error)) {
        std::fprintf(stderr, "decode: %s\n", error.c_str());
        return 1;
      }
    }
    const double decode = Seconds(decode_start);
    std::vector<nb::TokenId> ids(generated.begin(), generated.end());
    std::printf("generated %zu tokens in %.3f s (%.2f tokens/s)\n",
                generated.size(), decode,
                static_cast<double>(generated.size()) / decode);
    std::printf("ids:");
    for (auto id : generated)
      std::printf(" %d", id);
    std::printf("\ntext: %s\n", model->tokenizer().Decode(ids).c_str());
  }
  return 0;
}
