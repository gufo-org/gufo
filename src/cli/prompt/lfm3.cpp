#include "src/cli/prompt/lfm3.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "src/models/lfm3/engine.hpp"
#include "src/models/lfm3_vl/prompt.hpp"
namespace gufo::cli {
namespace {
using Clock = std::chrono::steady_clock;
using models::lfm3::Model;
using models::lfm3::Tokenizer;
double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
}  // namespace
int RunLfm3Prompt(const PromptOptions& opt, bool interactive) {
  try {
    if (opt.force_cpu || !opt.speculative_backend.empty())
      throw std::runtime_error("LFM3 supports autoregressive HIP inference");
    if (opt.reasoning_mode != "auto" && opt.reasoning_mode != "on" &&
        opt.reasoning_mode != "off")
      throw std::runtime_error("LFM3 reasoning must be auto, on, or off");
    if (opt.reasoning_effort != "auto" || opt.preserve_thinking != "auto")
      throw std::runtime_error(
          "LFM3 does not support reasoning effort or preserve-thinking "
          "overrides");
    std::string error;
    const auto load = Clock::now();
    auto model = Model::Load(opt.model_path, &error);
    Require(bool(model), error);
    auto reader = core::GgufReader::OpenFile(opt.model_path, &error);
    Require(bool(reader), error);
    const bool vl = reader->GetMetadataString("general.basename") == "LFM3-VL";
    if (vl != !opt.vision_model_path.empty() ||
        (!vl && !opt.image_paths.empty()))
      throw std::invalid_argument(
          "image input requires an LFM3-VL backbone and matching --mmproj");
    if (vl && opt.add_vision_id)
      throw std::invalid_argument("LFM3-VL does not use --add-vision-id");
    if (vl && !opt.use_chat_template)
      throw std::invalid_argument("LFM3-VL requires chat framing");
    std::unique_ptr<models::lfm3_vl::Vision> vision;
    if (vl) {
      auto projector =
          core::GgufReader::OpenFile(opt.vision_model_path, &error);
      if (!projector ||
          projector->GetMetadataString("general.basename") != "LFM3-VL")
        throw std::invalid_argument(
            "--mmproj does not match the LFM3-VL backbone");
      vision = models::lfm3_vl::Vision::Load(opt.vision_model_path);
    }
    std::cerr << "[ModelLoad]: " << Seconds(load) << " s\n";
    auto session = model->CreateSession(
        std::min<std::uint32_t>(8192, model->config().context), &error);
    Require(bool(session), error);
    const auto& tokenizer = model->tokenizer();
    std::vector<std::pair<std::string, std::string>> history;
    if (!opt.system_prompt.empty())
      history.emplace_back("system", opt.system_prompt);
    std::vector<tokenization::ChatMessage> visual_history;
    if (!opt.system_prompt.empty())
      visual_history.emplace_back(tokenization::ChatRole::kSystem,
                                  opt.system_prompt);
    bool attach_images = true;
    std::vector<std::uint32_t> processed;
    std::vector<float> logits(model->config().vocab);
    std::string input = opt.prompt_text;
    do {
      if (interactive) {
        std::cout << ">>> User: " << std::flush;
        if (!std::getline(std::cin, input) || input == "exit")
          break;
      }
      history.emplace_back("user", input);
      const auto rendered =
          opt.use_chat_template
              ? Tokenizer::Render(history, opt.reasoning_mode != "off")
              : input;
      auto tokens =
          vision ? std::vector<std::uint32_t>{} : tokenizer.Encode(rendered);
      models::lfm3_vl::Prompt visual;
      if (vision) {
        tokenization::ChatMessage message(tokenization::ChatRole::kUser, input);
        if (attach_images)
          for (const auto& path : opt.image_paths)
            message.images.push_back(
                {0, std::make_shared<const std::vector<std::uint8_t>>(
                        core::ReadImageFile(path))});
        attach_images = false;
        visual_history.push_back(std::move(message));
        ReasoningOptions reasoning;
        reasoning.enabled = opt.reasoning_mode != "off";
        visual = models::lfm3_vl::Prepare(
            tokenizer, visual_history, {}, reasoning,
            std::min<std::uint32_t>(8192, model->config().context));
        tokens = visual.tokens;
      }
      if (tokens.empty())
        tokens.push_back(tokenizer.Bos());
      const bool prefix =
          tokens.size() > processed.size() &&
          std::equal(processed.begin(), processed.end(), tokens.begin());
      if (!prefix) {
        Require(session->Reset(&error), error);
        processed.clear();
      }
      const auto prefill = Clock::now();
      std::size_t cursor = processed.size();
      for (const auto& image : visual.images) {
        const auto rows =
            std::size_t(image.picture.height) * image.picture.width / 4;
        if (cursor >= image.offset + rows)
          continue;
        if (cursor < image.offset) {
          Require(session->Forward(
                      std::span(tokens).subspan(cursor, image.offset - cursor),
                      {}, &error),
                  error);
          cursor = image.offset;
        }
        auto embeddings = vision->Encode(image.picture);
        const auto hidden = model->config().hidden;
        Require(
            session->ForwardEmbeddings(
                std::span(embeddings).subspan((cursor - image.offset) * hidden),
                false, {}, &error),
            error);
        cursor = image.offset + rows;
      }
      Require(
          session->Forward(std::span(tokens).subspan(cursor), logits, &error),
          error);
      const auto pp = Seconds(prefill);
      processed = tokens;
      sampling::SamplerState sampler(opt.sampling, tokens);
      std::vector<std::uint32_t> generated;
      const auto decode = Clock::now();
      for (std::size_t i = 0; i < opt.max_tokens; ++i) {
        const auto token = sampler.Sample(logits);
        if (tokenizer.IsStop(token))
          break;
        generated.push_back(token);
        sampler.Accept(token);
        std::cout << tokenizer.Piece(token) << std::flush;
        if (session->Position() >= 8192)
          break;
        Require(session->Forward(std::span(&token, 1), logits, &error), error);
        processed.push_back(token);
      }
      const auto tg = Seconds(decode);
      std::cout << '\n';
      if (opt.verbose)
        std::cerr << "LFM3 pp " << tokens.size() << " tokens " << pp
                  << " s; tg " << generated.size() << " tokens " << tg << " s ("
                  << generated.size() / tg << " tok/s)\n";
      history.emplace_back("assistant", tokenizer.Decode(generated));
      if (vision)
        visual_history.emplace_back(tokenization::ChatRole::kAssistant,
                                    tokenizer.Decode(generated));
    } while (interactive);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "LFM3: " << e.what() << '\n';
    return 1;
  }
}
int RunLfm3Bench(const BenchOptions& opt) {
  try {
    if (opt.concurrency != std::vector<std::size_t>{1} ||
        !opt.speculative_backend.empty() || opt.enable_yarn ||
        opt.validate_prefill_tokens)
      throw std::runtime_error(
          "LFM3 bench supports C1 autoregressive timing; use lfm3_probe for "
          "logit validation");
    std::string error;
    auto model = Model::Load(opt.model_path, &error);
    Require(bool(model), error);
    auto metadata = core::GgufReader::OpenFile(opt.model_path, &error);
    Require(bool(metadata), error);
    const auto model_name =
        metadata->GetMetadataString("general.basename") == "LFM3-VL"
            ? "LFM3-VL-2B"
            : "LFM3-1.7B";
    auto max = [](const auto& v) {
      return v.empty() ? std::size_t(0) : *std::max_element(v.begin(), v.end());
    };
    auto context =
        max(opt.n_depths) + std::max(max(opt.n_prompts), max(opt.n_gens)) + 1;
    if (context > model->config().context)
      throw std::runtime_error("LFM3 benchmark exceeds model context");
    auto session = model->CreateSession(context, &error);
    Require(bool(session), error);
    std::vector<float> logits(model->config().vocab);
    auto seed = model->tokenizer().Encode(
        "The quick brown fox jumps over the lazy dog. ");
    auto fill = [&](std::size_t n) {
      std::vector<std::uint32_t> ids(n);
      for (std::size_t i = 0; i < n; ++i)
        ids[i] = seed[i % seed.size()];
      return ids;
    };
    Require(session->Forward(seed, logits, &error),
            error);  // Warm both prompt and decode routes.
    auto token = std::uint32_t(std::max_element(logits.begin(), logits.end()) -
                               logits.begin());
    Require(session->Forward(std::span(&token, 1), logits, &error), error);
    std::cout << "| model | test | depth | repetition | tokens/s | seconds "
                 "|\n| :-- | :-- | --: | --: | --: | --: |\n";
    for (auto depth : opt.n_depths) {
      auto prepare = [&] {
        Require(session->Reset(&error), error);
        if (depth)
          Require(session->Forward(fill(depth), {}, &error), error);
      };
      // Warm the actual prompt shapes, including a full 128-token chunk,
      // before any timed request. Session scratch is already allocated.
      for (auto n : opt.n_prompts) {
        if (!n)
          continue;
        prepare();
        Require(session->Forward(fill(n), {}, &error), error);
      }
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        for (auto n : opt.n_prompts) {
          if (!n)
            continue;
          prepare();
          auto prompt = fill(n);
          const auto start = Clock::now();
          Require(session->Forward(prompt, {}, &error), error);
          const auto dt = Seconds(start);
          std::cout << "| " << model_name << " | pp" << n << " | " << depth
                    << " | " << r << " | " << n / dt << " | " << dt << " |\n";
        }
        for (auto n : opt.n_gens) {
          if (!n)
            continue;
          prepare();
          auto prompt = fill(1);
          Require(session->Forward(prompt, logits, &error), error);
          sampling::SamplerState sampler(opt.sampling, prompt);
          const auto start = Clock::now();
          for (std::size_t i = 0; i < n; ++i) {
            token = sampler.Sample(logits);
            sampler.Accept(token);
            Require(session->Forward(std::span(&token, 1), logits, &error),
                    error);
          }
          const auto dt = Seconds(start);
          std::cout << "| " << model_name << " | tg" << n << " | " << depth
                    << " | " << r << " | " << n / dt << " | " << dt << " |\n";
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "LFM3: " << e.what() << '\n';
    return 1;
  }
}
}  // namespace gufo::cli
