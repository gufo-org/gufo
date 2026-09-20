#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_tts/hip/speech_decoder_runtime.hpp"

namespace {

namespace qwen3_tts_hip = gufo::models::qwen3_tts::hip;

struct NpyFloat {
  std::vector<std::size_t> shape;
  std::vector<float> values;
};

struct Comparison {
  double cosine{0.0};
  double mean_absolute_error{0.0};
  float maximum_absolute_error{0.0F};
};

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_speech_decoder_hip_test: " << message << '\n';
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
  const std::size_t key = header.find("'shape'");
  const std::size_t open =
      key == std::string_view::npos ? key : header.find('(', key);
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
  (void)input.get();
  Check(major == 1 || major == 2 || major == 3, "unsupported NPY version");
  const std::size_t header_size = major == 1 ? ReadU16(input) : ReadU32(input);
  std::string header(header_size, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  Check(header.find("'descr': '<f4'") != std::string::npos,
        "NPY tensor must be little-endian float32");
  Check(header.find("fortran_order': False") != std::string::npos,
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

std::vector<float> ChannelMajorToRows(const NpyFloat& tensor,
                                      std::size_t channels,
                                      std::size_t length) {
  Check(tensor.shape == std::vector<std::size_t>({1, channels, length}),
        "unexpected official convolution tensor shape");
  std::vector<float> rows(length * channels);
  for (std::size_t time = 0; time < length; ++time) {
    for (std::size_t channel = 0; channel < channels; ++channel) {
      rows[time * channels + channel] = tensor.values[channel * length + time];
    }
  }
  return rows;
}

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
    const float difference = std::abs(actual[index] - expected[index]);
    absolute_error += difference;
    maximum_error = std::max(maximum_error, difference);
  }
  return {
      .cosine = dot / std::sqrt(actual_norm * expected_norm),
      .mean_absolute_error =
          absolute_error / static_cast<double>(actual.size()),
      .maximum_absolute_error = maximum_error,
  };
}

Comparison Report(std::string_view name, std::span<const float> actual,
                  std::span<const float> expected) {
  const Comparison comparison = Compare(actual, expected);
  std::cout << name << " cosine=" << comparison.cosine
            << " mae=" << comparison.mean_absolute_error
            << " max_abs=" << comparison.maximum_absolute_error << '\n';
  return comparison;
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path model_root =
      argc > 1 ? argv[1]
               : "/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice";
  const std::filesystem::path artifacts =
      argc > 2 ? argv[2]
               : std::filesystem::path(__FILE__)
                         .parent_path()
                         .parent_path()
                         .parent_path()
                         .parent_path() /
                     "artifacts/qwen3_tts/speech_decoder_f32_rocm";
  if (!std::filesystem::is_regular_file(model_root / "model.safetensors") ||
      !std::filesystem::is_regular_file(artifacts / "waveform.npy")) {
    std::cerr << "SKIP qwen3_tts_speech_decoder_hip_test: external model or "
                 "decoder artifacts are unavailable\n";
    return 77;
  }

  const std::vector<std::uint32_t> codes = {
      1995, 1159, 355, 22,   1174, 1093, 625,  1814, 1058, 905,  1846, 1247,
      1677, 889,  812, 901,  1028, 1836, 568,  89,   1191, 84,   1118, 431,
      962,  837,  4,   262,  1012, 15,   331,  299,  899,  1937, 453,  1626,
      674,  1345, 640, 386,  589,  1389, 1861, 1151, 459,  356,  1660, 750,
      1546, 1473, 177, 1881, 1621, 408,  220,  855,  102,  1261, 135,  943,
      986,  340,  6,   439,  1546, 1114, 544,  807,  1621, 1605, 220,  314,
      1009, 797,  930, 917,  1221, 1449, 246,  1206,
  };
  constexpr std::size_t frames = 5;

  std::string error;
  auto runtime = qwen3_tts_hip::SpeechDecoderHipRuntime::Create(
      model_root.string(), &error);
  Check(runtime != nullptr, "create speech decoder: " + error);
  qwen3_tts_hip::SpeechDecoderOutput output;
  qwen3_tts_hip::SpeechDecoderTrace trace;
  Check(runtime->Decode(codes, frames, &output, &trace, &error),
        "decode speech: " + error);
  Check(output.samples.size() == frames * 1920, "waveform sample count");
  Check(output.sample_rate == 24000, "waveform sample rate");
  Check(trace.upsample_outputs.size() == 4, "upsample trace count");
  Check(trace.decoder_outputs.size() == 7, "decoder trace count");

  const auto quantizer = ChannelMajorToRows(
      ReadNpyFloat(artifacts / "quantizer_output.npy"), 512, frames);
  const auto pre_conv = ChannelMajorToRows(
      ReadNpyFloat(artifacts / "pre_conv_output.npy"), 1024, frames);
  const NpyFloat transformer =
      ReadNpyFloat(artifacts / "transformer_output.npy");
  Check(transformer.shape == std::vector<std::size_t>({1, frames, 1024}),
        "official transformer tensor shape");

  const Comparison quantizer_comparison =
      Report("quantizer", trace.quantizer_output, quantizer);
  const Comparison pre_conv_comparison =
      Report("pre_conv", trace.pre_conv_output, pre_conv);
  const Comparison transformer_comparison =
      Report("transformer", trace.transformer_output, transformer.values);

  const std::array<std::size_t, 4> upsample_lengths{10, 10, 20, 20};
  for (std::size_t index = 0; index < trace.upsample_outputs.size(); ++index) {
    const std::size_t stage = index / 2;
    const std::size_t block = index % 2;
    const auto expected = ChannelMajorToRows(
        ReadNpyFloat(artifacts / ("upsample_" + std::to_string(stage) + "_" +
                                  std::to_string(block) + "_output.npy")),
        1024, upsample_lengths[index]);
    (void)Report(
        "upsample_" + std::to_string(stage) + "_" + std::to_string(block),
        trace.upsample_outputs[index], expected);
  }

  const std::array<std::size_t, 7> decoder_channels{1536, 768, 384, 192,
                                                    96,   96,  1};
  const std::array<std::size_t, 7> decoder_lengths{20,   160,  800, 3200,
                                                   9600, 9600, 9600};
  Comparison decoder_output;
  for (std::size_t index = 0; index < trace.decoder_outputs.size(); ++index) {
    const auto expected = ChannelMajorToRows(
        ReadNpyFloat(artifacts /
                     ("decoder_" + std::to_string(index) + "_output.npy")),
        decoder_channels[index], decoder_lengths[index]);
    decoder_output = Report("decoder_" + std::to_string(index),
                            trace.decoder_outputs[index], expected);
  }
  const NpyFloat waveform = ReadNpyFloat(artifacts / "waveform.npy");
  Check(waveform.shape == std::vector<std::size_t>({1, 1, 9600}),
        "official waveform shape");
  const Comparison waveform_comparison =
      Report("waveform", output.samples, waveform.values);
  std::cout << "decode_ms=" << output.decode_milliseconds << '\n';

  Check(quantizer_comparison.cosine > 0.9999,
        "quantizer cosine against official ROCm");
  Check(pre_conv_comparison.cosine > 0.999,
        "pre-convolution cosine against official ROCm");
  Check(transformer_comparison.cosine > 0.99,
        "transformer cosine against official ROCm");
  Check(decoder_output.cosine > 0.98,
        "output convolution cosine against official ROCm");
  Check(waveform_comparison.cosine > 0.98,
        "waveform cosine against official ROCm");
  Check(waveform_comparison.mean_absolute_error < 1.0e-5,
        "waveform mean absolute error against official ROCm");
  Check(waveform_comparison.maximum_absolute_error < 1.0e-4,
        "waveform maximum absolute error against official ROCm");
  std::vector<std::uint32_t> long_codes(320 * 16);
  for (std::size_t i = 0; i < long_codes.size(); ++i) {
    long_codes[i] = codes[i % codes.size()];
  }
  qwen3_tts_hip::SpeechDecoderOutput full;
  qwen3_tts_hip::SpeechDecoderOutput repeated;
  Check(runtime->Decode(long_codes, 320, &full, nullptr, &error) &&
            runtime->Decode(long_codes, 320, &repeated, nullptr, &error),
        "decode long replay: " + error);
  Check(full.samples == repeated.samples,
        "waveform decoder repeats exactly beyond the historical frame-38 gap");
  for (const std::size_t prefix : {8U, 32U}) {
    qwen3_tts_hip::SpeechDecoderOutput early;
    Check(runtime->Decode(std::span(long_codes).first(prefix * 16), prefix,
                          &early, nullptr, &error),
          "decode causal prefix: " + error);
    const auto comparison = Report(
        "causal_prefix_" + std::to_string(prefix), early.samples,
        std::span<const float>(full.samples).first(early.samples.size()));
    Check(comparison.mean_absolute_error < 1.0e-5 &&
              comparison.maximum_absolute_error < 1.0e-4,
          "early audio preserves the established waveform quality gate");
  }
  std::vector<float> incremental;
  std::size_t begin = 0;
  for (const std::size_t end : {8U, 24U, 64U, 299U, 301U, 320U}) {
    qwen3_tts_hip::SpeechDecoderOutput part;
    Check(runtime->DecodeIncremental(std::span(long_codes).first(end * 16), end,
                                     begin, &part, &error),
          "incremental decoder: " + error);
    Check(part.samples.size() == (end - begin) * 1920,
          "incremental decoder emits only new samples");
    (void)Report("chunk_" + std::to_string(end), part.samples,
                 std::span<const float>(full.samples)
                     .subspan(begin * 1920, part.samples.size()));
    incremental.insert(incremental.end(), part.samples.begin(),
                       part.samples.end());
    begin = end;
  }
  const auto incremental_comparison =
      Report("incremental", incremental, full.samples);
  Check(incremental_comparison.mean_absolute_error < 1.0e-5 &&
            incremental_comparison.maximum_absolute_error < 1.0e-4,
        "incremental state retains offline waveform quality");
  for (const std::size_t reference : {31U, 300U, 303U}) {
    qwen3_tts_hip::SpeechDecoderOutput cold, cached;
    Check(runtime->DecodeAfterReference(long_codes, 320, reference, &cold,
                                        &error),
          "capture reference state: " + error);
    Check(runtime->DecodeAfterReference(long_codes, 320, reference, &cached,
                                        &error),
          "restore reference state: " + error);
    Check(cold.samples == cached.samples && cold.code_frames == 320 - reference,
          "cached reference emits exact suffix audio");
    Check(std::equal(cold.samples.begin(), cold.samples.end(),
                     incremental.begin() + reference * 1920),
          "reference reuse matches independently decoded full history");
  }
  // A new voice must replace the saved state, and an intervening unrelated
  // request must not modify its immutable reference frontier.
  auto other_codes = long_codes;
  other_codes[0] = (other_codes[0] + 1) % 2048;
  qwen3_tts_hip::SpeechDecoderOutput other_full, other_suffix, restored;
  Check(runtime->DecodeIncremental(other_codes, 320, 0, &other_full, &error) &&
            runtime->DecodeAfterReference(other_codes, 320, 31, &other_suffix,
                                          &error) &&
            runtime->DecodeIncremental(std::span(long_codes).first(8 * 16), 8,
                                       0, &restored, &error) &&
            runtime->DecodeAfterReference(other_codes, 320, 31, &restored,
                                          &error),
        "replace and independently reuse reference state: " + error);
  Check(other_suffix.samples == restored.samples &&
            std::equal(restored.samples.begin(), restored.samples.end(),
                       other_full.samples.begin() + 31 * 1920),
        "reference identity and unrelated request isolation");
  qwen3_tts_hip::SpeechDecoderOutput reset_output;
  Check(runtime->DecodeIncremental(std::span(long_codes).first(8 * 16), 8, 0,
                                   &reset_output, &error),
        error);
  Check(std::equal(reset_output.samples.begin(), reset_output.samples.end(),
                   full.samples.begin()),
        "a new request clears decoder history");
  long_codes[0] = (long_codes[0] + 1) % 2048;
  Check(!runtime->DecodeIncremental(std::span(long_codes).first(9 * 16), 9, 8,
                                    &reset_output, &error),
        "changed codec prefixes cannot reuse decoder state");
  std::cout << "PASS qwen3_tts_speech_decoder_hip_test\n";
  return 0;
}
