// Stage oracles against the pinned upstream CPU path
// (reference/run_official.py): tokenizer ids, teacher-forced logits, the
// acoustic velocity and latents from fixed tokens and noise, and audio from
// fixed latents. Exact sampling cannot match across runtimes, so each stage
// starts from upstream inputs.
//
// YUE2_ORACLE_DIR holds the oracle's .npy/.json outputs; YUE2_MODEL_DIR and
// YUE2_VAE_DIR the checkpoints. Without them the test skips (77).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "src/models/yue2/config.hpp"
#include "src/models/yue2/hip/backbone.hpp"
#include "src/models/yue2/hip/vae.hpp"
#include "src/models/yue2/protocol.hpp"
#include "src/models/yue2/tokenizer.hpp"

namespace fs = std::filesystem;
namespace yue2 = gufo::models::yue2;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  std::cout << (condition ? "ok   " : "FAIL ") << message << '\n';
  if (!condition)
    ++failures;
}

std::string ReadFile(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

// Minimal .npy reader for little-endian <f4 and <i4 arrays.
template<typename T>
std::vector<T> ReadNpy(const fs::path& path, std::vector<std::size_t>* shape) {
  const std::string bytes = ReadFile(path);
  if (bytes.size() < 10 || bytes.compare(0, 6, "\x93NUMPY") != 0)
    throw std::runtime_error("not an npy file: " + path.string());
  const std::size_t header =
      static_cast<unsigned char>(bytes[8]) |
      (static_cast<std::size_t>(static_cast<unsigned char>(bytes[9])) << 8U);
  const std::string dict = bytes.substr(10, header);
  const std::string want =
      sizeof(T) == 4 && std::is_floating_point_v<T> ? "'<f4'" : "'<i4'";
  if (dict.find(want) == std::string::npos)
    throw std::runtime_error("unexpected npy dtype in " + path.string());
  shape->clear();
  const std::size_t open = dict.find('(');
  const std::size_t close = dict.find(')');
  std::size_t count = 1;
  for (std::size_t i = open + 1; i < close;) {
    while (i < close && (dict[i] < '0' || dict[i] > '9'))
      ++i;
    if (i >= close)
      break;
    std::size_t value = 0;
    while (i < close && dict[i] >= '0' && dict[i] <= '9')
      value = value * 10 + static_cast<std::size_t>(dict[i++] - '0');
    shape->push_back(value);
    count *= value;
  }
  std::vector<T> data(count);
  std::memcpy(data.data(), bytes.data() + 10 + header, count * sizeof(T));
  return data;
}

template<typename T>
std::vector<T> ReadNpy(const fs::path& path) {
  std::vector<std::size_t> shape;
  return ReadNpy<T>(path, &shape);
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  return dot / std::sqrt(na * nb);
}

double RelativeError(const std::vector<float>& actual,
                     const std::vector<float>& expected) {
  double error = 0, norm = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double d = static_cast<double>(actual[i]) - expected[i];
    error += d * d;
    norm += static_cast<double>(expected[i]) * expected[i];
  }
  return std::sqrt(error / norm);
}

std::vector<std::uint32_t> ToIds(const std::vector<std::int32_t>& values) {
  return {values.begin(), values.end()};
}

void TestTokenizer(const yue2::TextTokenizer& tokenizer, const fs::path& dir) {
  const auto oracle = gufo::json::parse(ReadFile(dir / "tokens.json"));
  for (const auto& [name, entry] : oracle.members()) {
    std::vector<std::uint32_t> expected;
    for (const auto& id : entry.find("ids")->items())
      expected.push_back(static_cast<std::uint32_t>(id.as_size(0)));
    Expect(tokenizer.Encode(entry.member_str("text")) == expected,
           "tokenizer matches upstream on '" + name + "'");
  }
}

}  // namespace

int main() {
  const char* oracle_env = std::getenv("YUE2_ORACLE_DIR");
  const char* model_env = std::getenv("YUE2_MODEL_DIR");
  const char* vae_env = std::getenv("YUE2_VAE_DIR");
  if (oracle_env == nullptr || model_env == nullptr || vae_env == nullptr) {
    std::cout << "SKIP: set YUE2_ORACLE_DIR, YUE2_MODEL_DIR and YUE2_VAE_DIR\n";
    return 77;
  }
  const fs::path oracle(oracle_env), model(model_env), vae_dir(vae_env);
  try {
    std::string error;
    auto tokenizer = yue2::TextTokenizer::Load(model / "qwen.tiktoken", &error);
    Expect(tokenizer != nullptr, "tokenizer loads " + error);
    TestTokenizer(*tokenizer, oracle);

    // The prompt protocol reproduces the planning prefix.
    const auto plan = gufo::json::parse(ReadFile(oracle / "plan.json"));
    const auto* request = plan.find("request");
    const yue2::SongPrompt prompt{
        request->member_str("style"), request->member_str("lyrics"),
        *yue2::ParsePlanning(request->member_str("cot"))};
    const auto abc_ids = ToIds(ReadNpy<std::int32_t>(oracle / "abc_ids.npy"));
    Expect(yue2::PlanPrefix(*tokenizer, prompt) ==
               ToIds(ReadNpy<std::int32_t>(oracle / "plan_prefix.npy")),
           "plan prefix matches upstream");
    const auto prefix = yue2::SemanticPrefix(*tokenizer, prompt, abc_ids);
    Expect(prefix == ToIds(ReadNpy<std::int32_t>(oracle / "prefix.npy")),
           "semantic prefix matches upstream");

    yue2::BackboneConfig config;
    yue2::GenerationDefaults defaults;
    Expect(yue2::LoadBackboneConfig(model, &config, &defaults, &error),
           "backbone config " + error);
    auto backbone = yue2::hip::Backbone::Create(model, config, &error);
    Expect(backbone != nullptr, "backbone loads " + error);
    if (!backbone)
      return 1;

    // Teacher-forced logits over the full vocabulary at the oracle's rows.
    const auto sequence =
        ToIds(ReadNpy<std::int32_t>(oracle / "tf_sequence.npy"));
    const auto rows_signed = ReadNpy<std::int32_t>(oracle / "tf_rows.npy");
    const std::vector<std::uint32_t> rows(rows_signed.begin(),
                                          rows_signed.end());
    std::vector<std::size_t> shape;
    const auto expected_logits =
        ReadNpy<float>(oracle / "tf_logits.npy", &shape);
    std::vector<float> logits;
    backbone->TeacherForce(sequence, rows,
                           {0, static_cast<std::uint32_t>(config.vocabulary)},
                           &logits);
    double worst_cosine = 1.0;
    std::size_t argmax_agree = 0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
      const auto begin = static_cast<std::ptrdiff_t>(r * shape[1]);
      std::vector<float> a(logits.begin() + begin,
                           logits.begin() + begin + shape[1]);
      std::vector<float> b(expected_logits.begin() + begin,
                           expected_logits.begin() + begin + shape[1]);
      worst_cosine = std::min(worst_cosine, Cosine(a, b));
      argmax_agree += std::max_element(a.begin(), a.end()) - a.begin() ==
                      std::max_element(b.begin(), b.end()) - b.begin();
    }
    std::cout << "     teacher-forced logits: worst cosine " << worst_cosine
              << ", argmax agrees on " << argmax_agree << "/" << rows.size()
              << '\n';
    Expect(worst_cosine > 0.999, "teacher-forced logits cosine > 0.999");
    Expect(argmax_agree * 10 >= rows.size() * 9,
           "teacher-forced argmax agrees on 90% of rows");

    // Acoustic velocity and latents from the upstream tokens and noise.
    const auto codec = ReadNpy<std::int32_t>(oracle / "semantic.npy");
    std::vector<std::uint32_t> ar = prefix;
    for (const auto code : codec)
      ar.push_back(static_cast<std::uint32_t>(code) + yue2::kCodecOffset);
    ar.push_back(yue2::kMusicEnd);
    auto cache = backbone->NewCache(ar.size());
    backbone->Append(*cache, ar, {}, nullptr);
    std::vector<float> noise = ReadNpy<float>(oracle / "noise.npy");
    for (float& value : noise) {
      // Upstream solves from the BF16 noise.
      std::uint32_t bits;
      std::memcpy(&bits, &value, 4);
      bits = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;
      std::memcpy(&value, &bits, 4);
    }
    std::vector<float> velocity;
    backbone->Velocity(*cache, noise, 20.0, &velocity);
    const auto expected_velocity = ReadNpy<float>(oracle / "velocity_t1.npy");
    const double velocity_cosine = Cosine(velocity, expected_velocity);
    std::cout << "     velocity(t=1): cosine " << velocity_cosine
              << ", relative error "
              << RelativeError(velocity, expected_velocity) << '\n';
    Expect(velocity_cosine > 0.999, "velocity cosine > 0.999");
    std::vector<float> latents;
    Expect(backbone->Synthesize(*cache, noise, defaults.ode_steps, {}, &latents,
                                &error),
           "acoustic solve " + error);
    const auto expected_latents = ReadNpy<float>(oracle / "latent.npy");
    const double latent_cosine = Cosine(latents, expected_latents);
    std::cout << "     latents: cosine " << latent_cosine << ", relative error "
              << RelativeError(latents, expected_latents) << '\n';
    Expect(latent_cosine > 0.999, "latent cosine > 0.999");
    cache.reset();
    backbone.reset();

    // Audio from the upstream latents.
    auto vae = yue2::hip::VaeDecoder::Create(vae_dir, &error);
    Expect(vae != nullptr, "VAE loads " + error);
    std::vector<float> audio;
    Expect(vae->Decode(expected_latents, {}, &audio, &error),
           "VAE decode " + error);
    const auto channel_major =
        ReadNpy<float>(oracle / "audio_tiled.npy", &shape);
    std::vector<float> expected_audio(audio.size());
    for (std::size_t t = 0; t < shape[1]; ++t)
      for (std::size_t c = 0; c < 2; ++c)
        expected_audio[2 * t + c] = channel_major[c * shape[1] + t];
    Expect(audio.size() == 2 * shape[1], "VAE output length");
    const double audio_error = RelativeError(audio, expected_audio);
    std::cout << "     audio: relative error " << audio_error << '\n';
    Expect(audio_error < 1e-3, "VAE audio relative error < 1e-3");
  } catch (const std::exception& exception) {
    std::cout << "FAIL exception: " << exception.what() << '\n';
    return 1;
  }
  return failures == 0 ? 0 : 1;
}
