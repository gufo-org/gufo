// Runs the scalar Gemma 4 reference on a prompt and writes its tokens and
// logits for comparison with tools/gemma4/llama_logits and the GPU runtime.
//   gemma4_reference_probe --model GGUF (--text TEXT | --chat MESSAGE)
//       --tokens-out T.i32 --logits-out L.g4lg [--half-kv] [--chunk N]
//       [--image-embd G4VE]
// --image-embd places a G4VE file's encoder rows (gemma4_vision_probe
// --output) on the prompt's run of image soft tokens (with --tokens-in, e.g.
// gemma4_vision_generate_probe's tokens.i32); it needs a single chunk.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/reference.hpp"
#include "src/models/gemma4/tokenizer.hpp"
#include "src/models/gemma4/vision/prompt.hpp"
#include "tests/models/gemma4/logit_file.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

int main(int argc, char** argv) {
  std::string model, text, chat, tokens_in, tokens_out, logits_out, trace_out,
      image_embd;
  bool half_kv = false;
  bool q8 = false;
  std::size_t chunk = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        std::exit(2);
      return argv[++i];
    };
    if (arg == "--model")
      model = value();
    else if (arg == "--text")
      text = value();
    else if (arg == "--chat")
      chat = value();
    else if (arg == "--tokens-in")
      tokens_in = value();
    else if (arg == "--tokens-out")
      tokens_out = value();
    else if (arg == "--logits-out")
      logits_out = value();
    else if (arg == "--half-kv")
      half_kv = true;
    else if (arg == "--q8-activations")
      q8 = true;
    else if (arg == "--trace-out")
      trace_out = value();
    else if (arg == "--chunk")
      chunk = std::stoul(value());
    else if (arg == "--image-embd")
      image_embd = value();
    else {
      std::cerr << "unknown argument " << arg << '\n';
      return 2;
    }
  }
  return gemma4_test::Run([&] {
    Require(!model.empty() && !tokens_out.empty() && !logits_out.empty() &&
                (!text.empty()) + (!chat.empty()) + (!tokens_in.empty()) == 1,
            "usage: --model GGUF (--text T | --chat M | --tokens-in F) "
            "--tokens-out F "
            "--logits-out F [--half-kv] [--chunk N]");
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model, &error);
    Require(reader != nullptr, error);
    const auto weights = g4::ModelWeights::Bind(*reader, &error);
    Require(weights.has_value(), error);
    const auto tokenizer = g4::Tokenizer::CreateFromGguf(*reader, &error);
    Require(tokenizer != nullptr, error);

    std::vector<g4::TokenId> tokens;
    if (!tokens_in.empty()) {
      std::ifstream in(tokens_in, std::ios::binary);
      for (std::int32_t t; in.read(reinterpret_cast<char*>(&t), 4);) {
        tokens.push_back(t);
      }
    } else if (!chat.empty()) {
      gufo::tokenization::ChatMessage message(
          gufo::tokenization::ChatRole::kUser, chat);
      const auto rendered =
          g4::ChatTemplate::Render(std::span(&message, 1), {}, {}, &error);
      Require(rendered.has_value(), error);
      tokens = tokenizer->Encode(rendered->text, false, true);
    } else {
      tokens = tokenizer->Encode(text, true, true);
    }
    gemma4_test::WriteTokens(tokens_out, tokens);

    g4::Reference reference(*weights,
                            q8        ? g4::Reference::Storage::kQ8Activations
                            : half_kv ? g4::Reference::Storage::kHalfKv
                                      : g4::Reference::Storage::kFloat32);
    gemma4_test::LogitFile file;
    file.vocab = weights->vocab_size;
    const std::size_t step = chunk == 0 ? tokens.size() : chunk;
    Require(trace_out.empty() || step == tokens.size(),
            "--trace-out needs a single chunk");
    std::vector<float> image_rows;
    std::vector<g4::Reference::Image> images;
    if (!image_embd.empty()) {
      Require(step == tokens.size(), "--image-embd needs a single chunk");
      std::ifstream in(image_embd, std::ios::binary);
      std::uint32_t header[4] = {};
      in.read(reinterpret_cast<char*>(header), sizeof(header));
      Require(in && header[0] == 0x45563447U &&
                  header[3] == weights->config.hidden_size,
              "--image-embd is not a G4VE file of the model's width");
      image_rows.resize(std::size_t{header[2]} * header[3]);
      in.read(reinterpret_cast<char*>(image_rows.data()),
              static_cast<std::streamsize>(image_rows.size() * 4));
      Require(in.good(), "truncated --image-embd");
      const auto soft =
          gufo::models::gemma4::vision::FindImageTokens(*tokenizer).soft;
      const auto first = std::find(tokens.begin(), tokens.end(), soft);
      const auto end = std::find_if(first, tokens.end(),
                                    [&](g4::TokenId t) { return t != soft; });
      Require(first != tokens.end() &&
                  static_cast<std::size_t>(end - first) == header[2] &&
                  std::find(end, tokens.end(), soft) == tokens.end(),
              "--image-embd rows do not match the prompt's one image");
      images.push_back({static_cast<std::uint32_t>(first - tokens.begin()),
                        header[2], image_rows.data()});
    }
    reference.SetTrace(!trace_out.empty());
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t begin = 0; begin < tokens.size(); begin += step) {
      const std::size_t count = std::min(step, tokens.size() - begin);
      std::vector<float> logits;
      reference.Forward(std::span(tokens).subspan(begin, count), &logits,
                        nullptr, images);
      file.logits.insert(file.logits.end(), logits.begin(), logits.end());
    }
    for (std::size_t p = 0; p < tokens.size(); ++p) {
      file.positions.push_back(static_cast<std::uint32_t>(p));
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    gemma4_test::WriteLogits(logits_out, file);
    if (!trace_out.empty()) {
      std::ofstream t(trace_out, std::ios::binary);
      const std::uint32_t th[3] = {weights->config.num_layers,
                                   static_cast<std::uint32_t>(tokens.size()),
                                   weights->config.hidden_size};
      t.write(reinterpret_cast<const char*>(th), sizeof(th));
      const auto& trace = reference.LayerTrace();
      t.write(reinterpret_cast<const char*>(trace.data()),
              static_cast<std::streamsize>(trace.size() * 4));
    }
    std::cout << tokens.size() << " tokens in " << seconds << " s\n";
  });
}
