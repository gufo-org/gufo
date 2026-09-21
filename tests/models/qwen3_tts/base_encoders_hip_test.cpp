#include <algorithm>
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

#include "src/models/qwen3_tts/audio.hpp"
#include "src/models/qwen3_tts/hip/encoder_ops.hpp"
#include "src/models/qwen3_tts/hip/encoder_support.hpp"
#include "src/models/qwen3_tts/hip/speaker_encoder_runtime.hpp"
#include "src/models/qwen3_tts/hip/speech_decoder_runtime.hpp"
#include "src/models/qwen3_tts/hip/speech_encoder_runtime.hpp"

namespace {

namespace qwen3_tts = gufo::models::qwen3_tts;
namespace qwen3_tts_hip = gufo::models::qwen3_tts::hip;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_base_encoders_hip_test: " << message << '\n';
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
  std::vector<std::size_t> shape;
  if (open == std::string_view::npos || close == std::string_view::npos) {
    return shape;
  }
  for (std::size_t position = open + 1; position < close;) {
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
    shape.push_back(
        std::stoull(std::string(header.substr(position, end - position))));
    position = end;
  }
  return shape;
}

template<typename Element>
std::vector<Element> ReadNpy(const std::filesystem::path& path,
                             std::string_view descriptor,
                             std::vector<std::size_t>* shape) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "cannot open " + path.string());
  char magic[6]{};
  input.read(magic, sizeof(magic));
  Check(std::string_view(magic, sizeof(magic)) == "\x93NUMPY",
        "invalid NPY magic");
  const int major = input.get();
  (void)input.get();
  const std::size_t header_size = major == 1 ? ReadU16(input) : ReadU32(input);
  std::string header(header_size, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  Check(header.find(std::string("'descr': '") + std::string(descriptor) +
                    "'") != std::string::npos,
        "unexpected NPY dtype");
  *shape = ParseShape(header);
  std::size_t elements = 1;
  for (const std::size_t dimension : *shape) {
    Check(dimension == 0 ||
              elements <= std::numeric_limits<std::size_t>::max() / dimension,
          "NPY shape overflow");
    elements *= dimension;
  }
  std::vector<Element> values(elements);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(Element)));
  Check(input.gcount() ==
            static_cast<std::streamsize>(values.size() * sizeof(Element)),
        "truncated NPY tensor");
  if (header.find("'fortran_order': True") != std::string::npos &&
      shape->size() > 1) {
    std::vector<Element> c_order(values.size());
    for (std::size_t destination = 0; destination < values.size();
         ++destination) {
      std::size_t remainder = destination;
      std::vector<std::size_t> coordinates(shape->size());
      for (std::size_t axis = shape->size(); axis-- > 0;) {
        coordinates[axis] = remainder % (*shape)[axis];
        remainder /= (*shape)[axis];
      }
      std::size_t source = 0;
      std::size_t source_stride = 1;
      for (std::size_t axis = 0; axis < shape->size(); ++axis) {
        source += coordinates[axis] * source_stride;
        source_stride *= (*shape)[axis];
      }
      c_order[destination] = values[source];
    }
    values = std::move(c_order);
  }
  return values;
}

qwen3_tts::AudioBuffer ReadWav(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "cannot open reference WAV");
  input.seekg(0, std::ios::end);
  const auto length = input.tellg();
  Check(length > 0, "reference WAV is empty");
  input.seekg(0, std::ios::beg);
  std::vector<std::byte> bytes(static_cast<std::size_t>(length));
  input.read(reinterpret_cast<char*>(bytes.data()), length);
  qwen3_tts::AudioBuffer audio;
  std::string error;
  Check(qwen3_tts::DecodeWav(bytes, &audio, &error), error);
  return audio;
}

double Cosine(std::span<const float> left, std::span<const float> right) {
  Check(left.size() == right.size(), "embedding shape mismatch");
  double dot = 0.0;
  double left_norm = 0.0;
  double right_norm = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    dot += static_cast<double>(left[index]) * right[index];
    left_norm += static_cast<double>(left[index]) * left[index];
    right_norm += static_cast<double>(right[index]) * right[index];
  }
  return dot / std::sqrt(left_norm * right_norm);
}

std::vector<float> ToFrameMajor(std::span<const float> values,
                                std::size_t channels, std::size_t frames) {
  Check(values.size() == channels * frames, "trace tensor shape mismatch");
  std::vector<float> result(values.size());
  for (std::size_t channel = 0; channel < channels; ++channel) {
    for (std::size_t frame = 0; frame < frames; ++frame) {
      result[frame * channels + channel] = values[channel * frames + frame];
    }
  }
  return result;
}

void ReportTraceCosine(const std::filesystem::path& artifacts,
                       std::string_view name, std::span<const float> actual,
                       std::size_t channels, std::size_t frames,
                       bool expected_is_frame_major) {
  // Prefer the float32 reference when it has been captured: the native encoder
  // emulates bfloat16 activations, and comparing against a bfloat16 oracle
  // hides whether that emulation is the whole story.
  std::filesystem::path path =
      artifacts / "speech_encoder" / (std::string(name) + "_float32.npy");
  if (!std::filesystem::is_regular_file(path)) {
    path = artifacts / "speech_encoder" / (std::string(name) + ".npy");
  }
  if (!std::filesystem::is_regular_file(path)) {
    return;
  }
  std::vector<std::size_t> shape;
  const std::vector<float> raw = ReadNpy<float>(path, "<f4", &shape);
  if (raw.size() != channels * frames || actual.size() != raw.size()) {
    std::cout << name << "_cosine=shape-mismatch(" << raw.size() << " vs "
              << channels * frames << '/' << actual.size() << ")\n";
    return;
  }
  std::vector<float> expected =
      expected_is_frame_major ? raw : ToFrameMajor(raw, channels, frames);
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double difference = actual[index] - expected[index];
    numerator += difference * difference;
    denominator += static_cast<double>(expected[index]) * expected[index];
  }
  std::cout << name << "_cosine=" << Cosine(actual, expected) << " relL2="
            << (denominator > 0.0 ? std::sqrt(numerator / denominator) : 0.0)
            << '\n';
}

/// Compares one frame-major speaker-encoder stage against the channel-major
/// float32 oracle written by `probe_speaker_encoder.py`.
void ReportSpeakerStage(const std::filesystem::path& artifacts,
                        std::string_view name, std::span<const float> actual,
                        std::size_t channels, std::size_t frames) {
  const std::filesystem::path path =
      artifacts / "speaker_encoder" / (std::string(name) + "_float32.npy");
  if (!std::filesystem::is_regular_file(path) || actual.empty()) {
    return;
  }
  std::vector<std::size_t> shape;
  const std::vector<float> raw = ReadNpy<float>(path, "<f4", &shape);
  if (raw.size() != channels * frames) {
    std::cout << "speaker_" << name << "_cosine=shape-mismatch(" << raw.size()
              << " vs " << channels * frames << ")\n";
    return;
  }
  const std::vector<float> expected =
      frames == 1 ? raw : ToFrameMajor(raw, channels, frames);
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double difference = actual[index] - expected[index];
    numerator += difference * difference;
    denominator += static_cast<double>(expected[index]) * expected[index];
  }
  std::cout << "speaker_" << name << "_cosine=" << Cosine(actual, expected)
            << " relL2="
            << (denominator > 0.0 ? std::sqrt(numerator / denominator) : 0.0)
            << '\n';
}

void CheckSoftmaxReplay() {
  constexpr std::size_t channels = 1536;
  for (const std::size_t rows : {1U, 31U, 33U, 65U, 127U, 257U, 1025U}) {
    std::vector<float> input(rows * channels), expected(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = 7.0F * std::sin(static_cast<float>(i) * 0.37F);
    for (std::size_t channel = 0; channel < channels; ++channel) {
      double maximum = -std::numeric_limits<double>::infinity(), sum = 0;
      for (std::size_t row = 0; row < rows; ++row)
        maximum = std::max(maximum, double(input[row * channels + channel]));
      for (std::size_t row = 0; row < rows; ++row)
        sum += std::exp(double(input[row * channels + channel]) - maximum);
      for (std::size_t row = 0; row < rows; ++row)
        expected[row * channels + channel] =
            std::exp(double(input[row * channels + channel]) - maximum) / sum;
    }
    qwen3_tts_hip::EncoderDeviceBuffer<float> device(input.size());
    std::vector<float> actual(input.size()), first;
    for (int repetition = 0; repetition < 8; ++repetition) {
      qwen3_tts_hip::RequireEncoderHip(
          hipMemcpy(device.get(), input.data(), input.size() * sizeof(float),
                    hipMemcpyHostToDevice),
          "softmax input");
      qwen3_tts_hip::LaunchSoftmaxOverRows(device.get(), rows, channels,
                                           nullptr);
      qwen3_tts_hip::RequireEncoderHip(
          hipMemcpy(actual.data(), device.get(), actual.size() * sizeof(float),
                    hipMemcpyDeviceToHost),
          "softmax output");
      for (std::size_t i = 0; i < actual.size(); ++i)
        Check(std::isfinite(actual[i]) &&
                  std::abs(actual[i] - expected[i]) < 3e-6F,
              "speaker softmax differs from independent FP64 normalization");
      if (repetition == 0)
        first = actual;
      else
        Check(actual == first, "speaker softmax replay changed bits");
    }
  }
  std::cout << "speaker softmax: FP64 oracle and exact replay passed\n";
}

}  // namespace

int main(int argc, char** argv) {
  CheckSoftmaxReplay();
  if (argc == 2 && std::string_view(argv[1]) == "--ops")
    return 0;
  Check(argc == 1 || argc == 3,
        "usage: base_encoders_hip_test [model artifacts]");
  const std::filesystem::path model_root =
      argc == 3
          ? argv[1]
          : "/home/fbozzo/projects/audio.cpp/models/Qwen3-TTS-12Hz-1.7B-Base";
  const std::filesystem::path artifacts =
      argc == 3 ? argv[2] : "artifacts/qwen3_tts/base";
  if (!std::filesystem::is_regular_file(model_root / "model.safetensors") ||
      !std::filesystem::is_regular_file(artifacts / "reference.wav")) {
    std::cerr << "SKIP qwen3_tts_base_encoders_hip_test: model or artifacts "
                 "are unavailable\n";
    return 77;
  }
  const qwen3_tts::AudioBuffer audio = ReadWav(artifacts / "reference.wav");
  std::string error;

  std::vector<std::size_t> speaker_shape;
  const std::vector<float> expected_speaker = ReadNpy<float>(
      artifacts / "reference_speaker_embedding.npy", "<f4", &speaker_shape);
  auto speaker = qwen3_tts_hip::SpeakerEncoderHipRuntime::Create(
      model_root.string(), &error);
  Check(speaker != nullptr, error);
  qwen3_tts_hip::SpeakerEncoderOutput speaker_output;
  qwen3_tts_hip::SpeakerEncoderTrace speaker_trace;
  Check(speaker->Encode(audio, &speaker_output, &error, &speaker_trace), error);
  const double speaker_cosine =
      Cosine(speaker_output.embedding, expected_speaker);
  std::cout << "speaker_embedding_cosine=" << speaker_cosine << '\n';
  // Stage-by-stage against the float32 oracle from probe_speaker_encoder.py,
  // which is what localises a mismatch to one block instead of the whole
  // ECAPA-TDNN. Absent files are skipped so the gate still runs without them.
  ReportSpeakerStage(artifacts, "mel", speaker_trace.features, 128,
                     speaker_trace.frames);
  ReportSpeakerStage(artifacts, "block0", speaker_trace.initial, 512,
                     speaker_trace.frames);
  ReportSpeakerStage(artifacts, "block1", speaker_trace.block1, 512,
                     speaker_trace.frames);
  ReportSpeakerStage(artifacts, "block2", speaker_trace.block2, 512,
                     speaker_trace.frames);
  ReportSpeakerStage(artifacts, "block3", speaker_trace.block3, 512,
                     speaker_trace.frames);
  ReportSpeakerStage(artifacts, "mfa", speaker_trace.aggregate, 1536,
                     speaker_trace.frames);
  ReportSpeakerStage(artifacts, "asp", speaker_trace.pooled, 3072, 1);
  ReportSpeakerStage(artifacts, "fc", speaker_trace.embedding, 2048, 1);
  Check(speaker_shape == std::vector<std::size_t>({2048}),
        "speaker oracle shape");
  // The native encoder emulates the official bfloat16 activations, and the
  // float32 oracle shows that emulation is worth about 2e-3 relative L2. A
  // threshold near 1 is what caught the Res2Net residual aliasing that 0.98
  // silently admitted.
  const bool speaker_parity = speaker_cosine > 0.9999;
  qwen3_tts_hip::SpeakerEncoderOutput speaker_replay;
  Check(speaker->Encode(audio, &speaker_replay, &error), error);
  Check(speaker_replay.embedding == speaker_output.embedding,
        "speaker encoding changes without diagnostic synchronization");
  speaker.reset();
  speaker = qwen3_tts_hip::SpeakerEncoderHipRuntime::Create(model_root.string(),
                                                            &error);
  Check(speaker != nullptr, error);
  Check(speaker->Encode(audio, &speaker_replay, &error), error);
  Check(speaker_replay.embedding == speaker_output.embedding,
        "speaker encoding changes after runtime reload");
  speaker.reset();

  std::vector<std::size_t> code_shape;
  const std::vector<std::int32_t> expected_codes = ReadNpy<std::int32_t>(
      artifacts / "reference_codes.npy", "<i4", &code_shape);
  auto speech = qwen3_tts_hip::SpeechEncoderHipRuntime::Create(
      model_root.string(), &error);
  Check(speech != nullptr, error);
  qwen3_tts_hip::SpeechEncoderOutput speech_output;
  qwen3_tts_hip::SpeechEncoderTrace speech_trace;
  Check(speech->Encode(audio, &speech_output, &error, &speech_trace), error);
  qwen3_tts_hip::SpeechEncoderOutput speech_replay;
  Check(speech->Encode(audio, &speech_replay, &error), error);
  Check(speech_replay.codes == speech_output.codes,
        "speech encoding changes without diagnostic synchronization");
  for (std::size_t index = 0; index < speech_trace.layers.size(); ++index) {
    const auto& layer = speech_trace.layers[index];
    std::string name = "layer";
    name += static_cast<char>('0' + (index / 10));
    name += static_cast<char>('0' + (index % 10));
    ReportTraceCosine(artifacts, name, layer.values, layer.channels,
                      layer.frames, false);
  }
  ReportTraceCosine(artifacts, "convolutional", speech_trace.convolutional, 512,
                    speech_trace.convolutional_frames, false);
  ReportTraceCosine(artifacts, "transformer", speech_trace.transformer, 512,
                    speech_trace.convolutional_frames, true);
  ReportTraceCosine(artifacts, "downsample", speech_trace.downsample, 512,
                    speech_trace.output_frames, false);
  ReportTraceCosine(artifacts, "semantic_projection",
                    speech_trace.semantic_projection, 256,
                    speech_trace.output_frames, false);
  ReportTraceCosine(artifacts, "acoustic_projection",
                    speech_trace.acoustic_projection, 256,
                    speech_trace.output_frames, false);
  Check(code_shape == std::vector<std::size_t>(
                          {speech_output.frames, speech_output.code_groups}),
        "speech code shape parity");
  std::size_t matching = 0;
  for (std::size_t index = 0; index < expected_codes.size(); ++index) {
    matching += speech_output.codes[index] ==
                        static_cast<std::uint32_t>(expected_codes[index])
                    ? 1
                    : 0;
  }
  const double match_rate =
      static_cast<double>(matching) / expected_codes.size();
  std::cout << "speech_code_match_rate=" << match_rate << " (" << matching
            << '/' << expected_codes.size() << ")\n";
  std::cout << "speech_group_match_rates=";
  std::vector<double> group_match_rates(speech_output.code_groups);
  for (std::size_t group = 0; group < speech_output.code_groups; ++group) {
    std::size_t group_matching = 0;
    for (std::size_t frame = 0; frame < speech_output.frames; ++frame) {
      const std::size_t index = frame * speech_output.code_groups + group;
      group_matching +=
          speech_output.codes[index] ==
                  static_cast<std::uint32_t>(expected_codes[index])
              ? 1
              : 0;
    }
    group_match_rates[group] =
        static_cast<double>(group_matching) / speech_output.frames;
    std::cout << (group == 0 ? "" : ",") << group_match_rates[group];
  }
  std::cout << '\n';
  std::cout << "speech_first_codes=";
  for (std::size_t index = 0;
       index < std::min<std::size_t>(16, speech_output.codes.size()); ++index) {
    std::cout << (index == 0 ? "" : ",") << speech_output.codes[index];
  }
  std::cout << '\n';
  std::vector<std::uint32_t> expected_unsigned(expected_codes.size());
  std::ranges::transform(
      expected_codes, expected_unsigned.begin(),
      [](std::int32_t code) { return static_cast<std::uint32_t>(code); });
  auto decoder = qwen3_tts_hip::SpeechDecoderHipRuntime::Create(
      model_root.string(), &error);
  Check(decoder != nullptr, error);
  qwen3_tts_hip::SpeechDecoderOutput expected_audio;
  qwen3_tts_hip::SpeechDecoderOutput actual_audio;
  Check(decoder->Decode(expected_unsigned, speech_output.frames,
                        &expected_audio, nullptr, &error),
        error);
  Check(decoder->Decode(speech_output.codes, speech_output.frames,
                        &actual_audio, nullptr, &error),
        error);
  const double reconstruction_cosine =
      Cosine(actual_audio.samples, expected_audio.samples);
  double reconstruction_mae = 0.0;
  for (std::size_t index = 0; index < actual_audio.samples.size(); ++index) {
    reconstruction_mae +=
        std::abs(actual_audio.samples[index] - expected_audio.samples[index]);
  }
  reconstruction_mae /= actual_audio.samples.size();
  std::cout << "speech_reconstruction_cosine=" << reconstruction_cosine
            << " mae=" << reconstruction_mae << '\n';
  Check(match_rate > 0.55, "speech tokenizer aggregate code parity");
  Check(group_match_rates[0] > 0.95 && group_match_rates[1] > 0.90,
        "speech tokenizer leading codebook parity");
  Check(reconstruction_cosine > 0.93,
        "speech tokenizer reconstructed waveform parity");
  Check(reconstruction_mae < 0.02,
        "speech tokenizer reconstructed waveform error");
  Check(speaker_parity, "speaker embedding cosine parity");

  std::cout << "PASS qwen3_tts_base_encoders_hip_test\n";
  return 0;
}
