// Evaluates a token file on the GPU runtime and writes teacher-forced logits
// for comparison with tools/gemma4/llama_logits and the CPU reference:
//   gemma4_gpu_probe --model GGUF --tokens T.i32 --logits-out L.g4lg
//       [--chunk N]   (tokens per EvaluateAll call; default: all at once)
//       [--first N --stride N]  (keep only these rows in the output)
// or builds the sequence itself from a chat turn and a generated reply, and
// dumps target features for speculative-drafter studies:
//       --chat TEXT --generate N [--temperature T --top-k K --top-p P
//       --repeat-penalty R --seed S] [--tokens-out T.i32]
//       [--taps 1,12,... --taps-out F.f32]  ([tokens][taps][hidden] float32,
//       the residual after each listed layer)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/sampling.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/engine.hpp"
#include "tests/models/gemma4/logit_file.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

int main(int argc, char** argv) {
  std::string model, tokens_path, logits_out, chat, tokens_out, taps_arg,
      taps_out;
  std::size_t generate = 0;
  gufo::sampling::SamplingConfig sampling;
  std::size_t chunk = 0;
  std::size_t first = 0;
  std::size_t stride = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        std::exit(2);
      return argv[++i];
    };
    if (arg == "--model")
      model = value();
    else if (arg == "--tokens")
      tokens_path = value();
    else if (arg == "--logits-out")
      logits_out = value();
    else if (arg == "--chunk")
      chunk = std::stoul(value());
    else if (arg == "--first")
      first = std::stoul(value());
    else if (arg == "--stride")
      stride = std::stoul(value());
    else if (arg == "--chat")
      chat = value();
    else if (arg == "--generate")
      generate = std::stoul(value());
    else if (arg == "--temperature")
      sampling.temperature = std::stof(value());
    else if (arg == "--top-k")
      sampling.top_k = std::stoul(value());
    else if (arg == "--top-p")
      sampling.top_p = std::stof(value());
    else if (arg == "--repeat-penalty")
      sampling.repeat_penalty = std::stof(value());
    else if (arg == "--seed")
      sampling.seed = std::stoull(value());
    else if (arg == "--tokens-out")
      tokens_out = value();
    else if (arg == "--taps")
      taps_arg = value();
    else if (arg == "--taps-out")
      taps_out = value();
    else {
      std::cerr << "unknown argument " << arg << '\n';
      return 2;
    }
  }
  return gemma4_test::Run([&] {
    Require(
        !model.empty() && (!tokens_path.empty() || !chat.empty()) &&
            (!logits_out.empty() || !taps_out.empty() || !tokens_out.empty()),
        "usage: --model GGUF (--tokens T.i32 | --chat TEXT) "
        "[--logits-out L] [--taps-out F]");
    std::vector<g4::TokenId> tokens;
    if (!tokens_path.empty()) {
      std::ifstream in(tokens_path, std::ios::binary);
      for (std::int32_t t; in.read(reinterpret_cast<char*>(&t), 4);) {
        tokens.push_back(t);
      }
    }
    std::string error;
    g4::ModelOptions options;
    // A chat turn has at most one token per byte of its text.
    options.max_context = static_cast<std::uint32_t>(
        tokens.size() + chat.size() + generate + 4096);
    const auto start = std::chrono::steady_clock::now();
    auto m = g4::Model::Load(model, options, &error);
    Require(m != nullptr, error);
    const auto loaded = std::chrono::steady_clock::now();
    if (!chat.empty()) {
      gufo::tokenization::ChatMessage message(
          gufo::tokenization::ChatRole::kUser, chat);
      const auto rendered =
          g4::ChatTemplate::Render(std::span(&message, 1), {}, {}, &error);
      Require(rendered.has_value(), error);
      tokens = m->Tokenize(rendered->text);
    }
    Require(!tokens.empty(), "no tokens");
    const std::size_t prompt_tokens = tokens.size();
    if (generate > 0) {
      // Autoregressive reply, the target's own continuation.
      auto session = m->CreateSession(0, &error);
      Require(session && session->Sync(tokens, &error), error);
      gufo::sampling::SamplerState sampler(sampling, {});
      for (std::size_t i = 0; i < generate; ++i) {
        const auto token =
            static_cast<g4::TokenId>(sampler.Sample(session->Logits()));
        sampler.Accept(static_cast<gufo::sampling::TokenId>(token));
        tokens.push_back(token);
        if (m->IsStopToken(token)) {
          break;
        }
        Require(session->Evaluate(token, &error), error);
      }
      std::cout << "prompt " << prompt_tokens << " tokens, generated "
                << tokens.size() - prompt_tokens << '\n';
    }
    if (!tokens_out.empty()) {
      std::ofstream out(tokens_out, std::ios::binary);
      out.write(reinterpret_cast<const char*>(tokens.data()),
                static_cast<std::streamsize>(tokens.size() * 4));
    }
    // Residual features after the requested layers, gathered per forward.
    std::vector<std::uint32_t> tap_layers;
    for (std::stringstream list(taps_arg); list.good();) {
      std::string item;
      std::getline(list, item, ',');
      if (!item.empty()) {
        tap_layers.push_back(static_cast<std::uint32_t>(std::stoul(item)));
      }
    }
    const std::size_t hidden = m->config().hidden_size;
    std::map<std::uint32_t, std::vector<float>> taps;
    if (!taps_out.empty()) {
      Require(!tap_layers.empty(), "--taps-out needs --taps");
      m->SetTapSink(
          [&](std::uint32_t layer, const float* rows, std::uint32_t count) {
            if (std::find(tap_layers.begin(), tap_layers.end(), layer) ==
                tap_layers.end()) {
              return;
            }
            auto& dst = taps[layer];
            const std::size_t at = dst.size();
            dst.resize(at + std::size_t{count} * hidden);
            HIP_CHECK(hipMemcpy(dst.data() + at, rows,
                                std::size_t{count} * hidden * sizeof(float),
                                hipMemcpyDeviceToHost));
          });
    }
    auto session = m->CreateSession(0, &error);
    Require(session != nullptr, error);
    gemma4_test::LogitFile file;
    file.vocab = m->VocabSize();
    const std::size_t step = chunk == 0 ? tokens.size() : chunk;
    for (std::size_t begin = 0; begin < tokens.size(); begin += step) {
      const std::size_t count = std::min(step, tokens.size() - begin);
      std::vector<float> logits;
      Require(session->EvaluateAll(std::span(tokens).subspan(begin, count),
                                   &logits, &error),
              error);
      for (std::size_t r = 0; r < count; ++r) {
        const std::size_t p = begin + r;
        if (!logits_out.empty() && p >= first && (p - first) % stride == 0) {
          file.positions.push_back(static_cast<std::uint32_t>(p));
          file.logits.insert(file.logits.end(), logits.begin() + r * file.vocab,
                             logits.begin() + (r + 1) * file.vocab);
        }
      }
    }
    const auto done = std::chrono::steady_clock::now();
    if (!logits_out.empty()) {
      gemma4_test::WriteLogits(logits_out, file);
    }
    if (!taps_out.empty()) {
      m->SetTapSink({});
      std::ofstream out(taps_out, std::ios::binary);
      for (std::size_t p = 0; p < tokens.size(); ++p) {
        for (const std::uint32_t layer : tap_layers) {
          const auto& rows = taps.at(layer);
          Require(rows.size() == tokens.size() * hidden,
                  "tap rows do not cover the sequence");
          out.write(reinterpret_cast<const char*>(rows.data() + p * hidden),
                    static_cast<std::streamsize>(hidden * sizeof(float)));
        }
      }
    }
    std::cout << "load "
              << std::chrono::duration<double>(loaded - start).count() << " s, "
              << tokens.size() << " tokens in "
              << std::chrono::duration<double>(done - loaded).count() << " s\n";
  });
}
