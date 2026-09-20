#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_tts/hip/talker_runtime.hpp"
#include "src/models/qwen3_tts/tokenizer.hpp"

namespace {

namespace qwen3_tts_hip = gufo::models::qwen3_tts::hip;
namespace qwen3_tts = gufo::models::qwen3_tts;

constexpr std::string_view kText =
    "The boy who lived. Mr. and Mrs. Dursley, of number four, Privet Drive, "
    "were proud to say that they were perfectly normal, thank you very much. "
    "They were the last people you'd expect to be involved in anything strange "
    "or mysterious, because they just didn't hold with such nonsense.";

struct NpyFloat {
  std::vector<std::size_t> shape;
  std::vector<float> values;
};

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_talker_hip_prefill_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

std::uint16_t ReadU16(std::istream& input) {
  unsigned char bytes[2]{};
  input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t ReadU32(std::istream& input) {
  unsigned char bytes[4]{};
  input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::vector<std::size_t> ParseShape(std::string_view header) {
  const std::size_t shape_key = header.find("'shape'");
  const std::size_t open = shape_key == std::string_view::npos
                               ? shape_key
                               : header.find('(', shape_key);
  const std::size_t close =
      open == std::string_view::npos ? open : header.find(')', open);
  if (open == std::string_view::npos || close == std::string_view::npos) {
    return {};
  }
  std::vector<std::size_t> shape;
  std::size_t position = open + 1;
  while (position < close) {
    while (position < close &&
           (header[position] == ' ' || header[position] == ',')) {
      ++position;
    }
    if (position == close) {
      break;
    }
    std::size_t end = position;
    while (end < close && header[end] >= '0' && header[end] <= '9') {
      ++end;
    }
    if (end == position) {
      return {};
    }
    shape.push_back(static_cast<std::size_t>(
        std::stoull(std::string(header.substr(position, end - position)))));
    position = end;
  }
  return shape;
}

NpyFloat ReadNpyFloat(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "cannot open " + path.string());
  char magic[6]{};
  input.read(magic, sizeof(magic));
  Check(std::string_view(magic, sizeof(magic)) == "\x93NUMPY",
        "invalid NPY magic");
  const int major = input.get();
  const int minor = input.get();
  (void)minor;
  Check(major == 1 || major == 2 || major == 3, "unsupported NPY version");
  const std::size_t header_size = major == 1 ? ReadU16(input) : ReadU32(input);
  std::string header(header_size, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  Check(header.find("'descr': '<f4'") != std::string::npos ||
            header.find("\"descr\": \"<f4\"") != std::string::npos,
        "NPY tensor must be little-endian float32");
  Check(header.find("fortran_order': False") != std::string::npos ||
            header.find("\"fortran_order\": false") != std::string::npos,
        "NPY tensor must be C contiguous");

  NpyFloat result;
  result.shape = ParseShape(header);
  Check(!result.shape.empty(), "cannot parse NPY shape");
  std::size_t elements = 1;
  for (const std::size_t dimension : result.shape) {
    Check(dimension == 0 ||
              elements <= std::numeric_limits<std::size_t>::max() / dimension,
          "NPY shape overflow");
    elements *= dimension;
  }
  result.values.resize(elements);
  input.read(reinterpret_cast<char*>(result.values.data()),
             static_cast<std::streamsize>(elements * sizeof(float)));
  Check(
      input.gcount() == static_cast<std::streamsize>(elements * sizeof(float)),
      "truncated NPY payload");
  return result;
}

std::size_t Argmax(std::span<const float> values) {
  return static_cast<std::size_t>(
      std::max_element(values.begin(), values.end()) - values.begin());
}

std::uint32_t SelectOfficialGreedy(
    std::span<const float> logits,
    std::span<const std::uint32_t> generated_first_codes,
    std::size_t generation_step) {
  std::vector<float> processed(logits.begin(), logits.end());
  constexpr std::uint32_t kCodecEos = 2150;
  constexpr float kRepetitionPenalty = 1.05F;
  if (generation_step < 2 && kCodecEos < processed.size()) {
    processed[kCodecEos] = -std::numeric_limits<float>::infinity();
  }
  for (std::size_t token = 2048; token < processed.size(); ++token) {
    if (token != kCodecEos) {
      processed[token] = -std::numeric_limits<float>::infinity();
    }
  }
  for (const std::uint32_t token : generated_first_codes) {
    if (token < processed.size()) {
      processed[token] = processed[token] < 0.0F
                             ? processed[token] * kRepetitionPenalty
                             : processed[token] / kRepetitionPenalty;
    }
  }
  return static_cast<std::uint32_t>(Argmax(processed));
}

struct Comparison {
  double cosine{0.0};
  double mean_absolute_error{0.0};
  float maximum_absolute_error{0.0F};
};

Comparison Compare(std::span<const float> actual,
                   std::span<const float> expected) {
  Check(actual.size() == expected.size(), "comparison shape mismatch");
  double dot = 0.0;
  double actual_norm = 0.0;
  double expected_norm = 0.0;
  double absolute_error = 0.0;
  float maximum_error = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    dot += static_cast<double>(actual[index]) * expected[index];
    actual_norm += static_cast<double>(actual[index]) * actual[index];
    expected_norm += static_cast<double>(expected[index]) * expected[index];
    const float error = std::abs(actual[index] - expected[index]);
    absolute_error += error;
    maximum_error = std::max(maximum_error, error);
  }
  return {
      .cosine = dot / std::sqrt(actual_norm * expected_norm),
      .mean_absolute_error =
          absolute_error / static_cast<double>(actual.size()),
      .maximum_absolute_error = maximum_error,
  };
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path model_root =
      argc > 1 ? argv[1]
               : "/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice";
  const std::filesystem::path artifacts =
      argc > 2 ? argv[2]
               : "/home/fbozzo/projects/strix-halo.cpp/artifacts/qwen3_tts";
  const std::filesystem::path embeddings_path =
      artifacts / "internals" / "prefill_inputs_embeds.npy";
  if (!std::filesystem::is_regular_file(model_root / "model.safetensors") ||
      !std::filesystem::is_regular_file(embeddings_path)) {
    std::cerr << "SKIP qwen3_tts_talker_hip_prefill_test: external model or "
                 "artifacts are unavailable\n";
    return 77;
  }

  const NpyFloat embeddings = ReadNpyFloat(embeddings_path);
  const NpyFloat expected_layer0 =
      ReadNpyFloat(artifacts / "internals" / "layer0_output.npy");
  const NpyFloat expected_logits =
      ReadNpyFloat(artifacts / "internals" / "prefill_logits_lastpos.npy");
  const NpyFloat expected_cached_logits =
      ReadNpyFloat(artifacts / "internals" / "cached_step_logits.npy");
  const NpyFloat expected_cached_hidden =
      ReadNpyFloat(artifacts / "internals" / "cached_step_past_hidden.npy");
  const NpyFloat expected_cached_input =
      ReadNpyFloat(artifacts / "internals" / "cached_step_inputs_embeds.npy");
  const NpyFloat expected_predictor_logits = ReadNpyFloat(
      artifacts / "internals" / "predictor_first_frame_logits.npy");
  Check(embeddings.shape == std::vector<std::size_t>({75, 2048}),
        "frozen embedding shape");

  std::string error;
  auto runtime =
      qwen3_tts_hip::TalkerHipRuntime::Create(model_root.string(), 512, &error);
  Check(runtime != nullptr, "create HIP runtime: " + error);

  qwen3_tts::Tokenizer tokenizer;
  Check(qwen3_tts::Tokenizer::Load(model_root, &tokenizer, &error),
        "load tokenizer: " + error);
  std::vector<std::uint32_t> input_ids;
  Check(tokenizer.EncodeAssistantPrompt(kText, &input_ids, &error),
        "encode prompt: " + error);
  qwen3_tts_hip::CustomVoicePromptOutput prompt;
  const auto prompt_start = std::chrono::steady_clock::now();
  Check(runtime->BuildCustomVoicePrompt(input_ids, {}, "vivian", "english",
                                        &prompt, &error),
        "build prompt: " + error);
  const auto prompt_end = std::chrono::steady_clock::now();
  Check(prompt.tokens == embeddings.shape[0], "native prompt token count");
  const Comparison prompt_comparison =
      Compare(prompt.embeddings, embeddings.values);
  std::cout << "prompt cosine=" << prompt_comparison.cosine
            << " mae=" << prompt_comparison.mean_absolute_error
            << " max_abs=" << prompt_comparison.maximum_absolute_error << " ms="
            << std::chrono::duration<double, std::milli>(prompt_end -
                                                         prompt_start)
                   .count()
            << '\n';

  qwen3_tts_hip::TalkerPrefillOutput output;
  const auto prefill_start = std::chrono::steady_clock::now();
  Check(runtime->Prefill(prompt.embeddings, prompt.tokens, &output, &error),
        "HIP prefill: " + error);
  const auto prefill_end = std::chrono::steady_clock::now();

  const Comparison layer0 =
      Compare(output.layer0_output, expected_layer0.values);
  const Comparison logits = Compare(output.logits, expected_logits.values);
  std::cout << "layer0 cosine=" << layer0.cosine
            << " mae=" << layer0.mean_absolute_error
            << " max_abs=" << layer0.maximum_absolute_error << '\n';
  std::cout << "logits cosine=" << logits.cosine
            << " mae=" << logits.mean_absolute_error
            << " max_abs=" << logits.maximum_absolute_error
            << " expected_argmax=" << Argmax(expected_logits.values)
            << " actual_argmax=" << Argmax(output.logits) << " prefill_ms="
            << std::chrono::duration<double, std::milli>(prefill_end -
                                                         prefill_start)
                   .count()
            << '\n';

  Check(prompt_comparison.cosine > 0.99, "prompt embedding cosine parity");
  Check(layer0.cosine > 0.99, "layer 0 cosine parity");
  Check(logits.cosine > 0.95, "final logit cosine parity");
  Check(Argmax(output.logits) == Argmax(expected_logits.values),
        "first greedy codec token parity");
  const std::vector<std::vector<std::uint32_t>> expected_frames = {
      {1995, 1159, 355, 22, 1174, 1093, 625, 1814, 1058, 905, 1846, 1247, 1677,
       889, 812, 901},
      {1028, 1836, 568, 89, 1191, 84, 1118, 431, 962, 837, 4, 262, 1012, 15,
       331, 299},
      {899, 1937, 453, 1626, 674, 1345, 640, 386, 589, 1389, 1861, 1151, 459,
       356, 1660, 750},
      {1546, 1473, 177, 1881, 1621, 408, 220, 855, 102, 1261, 135, 943, 986,
       340, 6, 439},
      {1546, 1114, 544, 807, 1621, 1605, 220, 314, 1009, 797, 930, 917, 1221,
       1449, 246, 1206},
  };
  std::vector<std::uint32_t> generated_first_codes;
  double predictor_ms = 0.0;
  double cached_talker_ms = 0.0;
  std::size_t matching_codes = 0;
  std::size_t matching_first_codes = 0;
  qwen3_tts_hip::TalkerPrefillOutput current = std::move(output);
  for (std::size_t frame = 0; frame < expected_frames.size(); ++frame) {
    const std::uint32_t first_code =
        SelectOfficialGreedy(current.logits, generated_first_codes, frame);
    generated_first_codes.push_back(first_code);
    std::vector<std::uint32_t> codes;
    const auto predictor_start = std::chrono::steady_clock::now();
    if (frame == 0) {
      qwen3_tts_hip::CodePredictorOutput predictor_output;
      Check(runtime->PredictCodeFrameTrace(current.last_hidden, first_code,
                                           &predictor_output, &error),
            "HIP code predictor trace: " + error);
      codes = std::move(predictor_output.codes);
      const Comparison predictor_logits =
          Compare(predictor_output.logits, expected_predictor_logits.values);
      std::cout << "predictor_logits cosine=" << predictor_logits.cosine
                << " mae=" << predictor_logits.mean_absolute_error
                << " max_abs=" << predictor_logits.maximum_absolute_error
                << '\n';
      Check(predictor_logits.cosine > 0.95,
            "first-frame predictor logit cosine parity");
    } else {
      Check(runtime->PredictCodeFrame(current.last_hidden, first_code, &codes,
                                      &error),
            "HIP code predictor: " + error);
    }
    const auto predictor_end = std::chrono::steady_clock::now();
    predictor_ms += std::chrono::duration<double, std::milli>(predictor_end -
                                                              predictor_start)
                        .count();
    for (std::size_t group = 0; group < codes.size(); ++group) {
      matching_codes += codes[group] == expected_frames[frame][group] ? 1 : 0;
    }
    matching_first_codes +=
        codes.front() == expected_frames[frame].front() ? 1 : 0;
    if (codes != expected_frames[frame]) {
      std::cerr << "frame " << frame << " expected=";
      for (const std::uint32_t code : expected_frames[frame]) {
        std::cerr << code << ',';
      }
      std::cerr << " actual=";
      for (const std::uint32_t code : codes) {
        std::cerr << code << ',';
      }
      std::cerr << '\n';
      Check(frame != 0, "first greedy codec frame parity");
    }
    if (frame + 1 < expected_frames.size()) {
      if (frame == 0) {
        std::vector<float> frame_embedding;
        Check(runtime->BuildCodeFrameEmbedding(codes, prompt.tts_pad,
                                               &frame_embedding, &error),
              "build codec frame embedding: " + error);
        const Comparison cached_input =
            Compare(frame_embedding, expected_cached_input.values);
        std::cout << "cached_input cosine=" << cached_input.cosine
                  << " mae=" << cached_input.mean_absolute_error
                  << " max_abs=" << cached_input.maximum_absolute_error << '\n';
      }
      qwen3_tts_hip::TalkerPrefillOutput next;
      const auto talker_start = std::chrono::steady_clock::now();
      Check(runtime->DecodeCodeFrame(codes, prompt.tts_pad, &next, &error),
            "HIP cached talker: " + error);
      const auto talker_end = std::chrono::steady_clock::now();
      cached_talker_ms +=
          std::chrono::duration<double, std::milli>(talker_end - talker_start)
              .count();
      if (frame == 0) {
        const Comparison cached_logits =
            Compare(next.logits, expected_cached_logits.values);
        const Comparison cached_hidden =
            Compare(next.last_hidden, expected_cached_hidden.values);
        std::cout << "cached_hidden cosine=" << cached_hidden.cosine
                  << " mae=" << cached_hidden.mean_absolute_error
                  << " max_abs=" << cached_hidden.maximum_absolute_error
                  << '\n';
        std::cout << "cached_logits cosine=" << cached_logits.cosine
                  << " mae=" << cached_logits.mean_absolute_error
                  << " max_abs=" << cached_logits.maximum_absolute_error
                  << " expected_argmax="
                  << Argmax(expected_cached_logits.values)
                  << " actual_argmax=" << Argmax(next.logits) << '\n';
      }
      current = std::move(next);
    }
  }
  std::cout << "predictor_ms_total=" << predictor_ms
            << " cached_talker_ms_total=" << cached_talker_ms
            << " matching_codes=" << matching_codes << '/'
            << expected_frames.size() * expected_frames.front().size()
            << " matching_first_codes=" << matching_first_codes << '/'
            << expected_frames.size() << '\n';

  // Cross the historical frame-38 divergence and attention's 64/128-token
  // boundaries. Short eight-frame controls did not exercise this failure.
  for (const bool sample : {false, true}) {
    const qwen3_tts::SamplingOptions sampling{
        .sample = sample,
        .seed = 42,
        .predictor_sample = sample,
    };
    qwen3_tts_hip::TalkerGenerationOutput first;
    Check(runtime->Generate(prompt, 64, sampling, &first, &error),
          "HIP generation: " + error);
    qwen3_tts_hip::TalkerGenerationOutput repeated;
    Check(runtime->Generate(prompt, 64, sampling, &repeated, &error),
          "repeated HIP generation: " + error);
    Check(first.frames > 0 && first.code_groups == 16, "HIP generation shape");
    const auto mismatch =
        std::mismatch(first.codes.begin(), first.codes.end(),
                      repeated.codes.begin(), repeated.codes.end());
    Check(first.codes == repeated.codes && first.frames == repeated.frames,
          "HIP generation replay differs at frame " +
              std::to_string(
                  std::distance(first.codes.begin(), mismatch.first) / 16) +
              (sample ? " (sampled)" : " (greedy)"));
    std::cout << "replay_sampled=" << sample << " frames=" << first.frames
              << '\n';
  }
  std::cout << "PASS qwen3_tts_talker_hip_prefill_test\n";
  return 0;
}
