#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>

#include "src/core/gguf_reader.hpp"
#include "src/models/minimax_h3/attention.hip.hpp"
#include "src/models/moss_image/hip/ops.hpp"
#include "src/models/moss_image/model.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen_image_21/weights.hpp"

namespace gufo::models::moss_image {
namespace {

using qwen_image_21::Weights;
using qwen_image_21::hip::Activation;
using qwen_image_21::hip::Norm;
using Clock = std::chrono::steady_clock;
using hip::Matrix;
using hip::Runtime;

// transformer/config.json of the released dense model.
constexpr int kDim = 3072;
constexpr int kHeads = 24;
constexpr int kHeadDim = kDim / kHeads;
constexpr int kLayers = 42;
constexpr int kLatentChannels = 128;
constexpr int kTextDim = 4096;
constexpr int kFreqDim = 256;
constexpr int kRopeBands = kHeadDim / 8;
constexpr int kMaxTextTokens = 4096;

double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

void Cancel(const CancellationCheck& cancelled) {
  if (cancelled && cancelled())
    throw std::runtime_error("image request cancelled");
}

std::vector<qwen_image_21::SafetensorsSource> Sources(
    const std::filesystem::path& root) {
  std::vector<qwen_image_21::SafetensorsSource> files;
  const auto manifest = qwen_image_21::ReadJson(
      root / "transformer/diffusion_pytorch_model.safetensors.index.json");
  const auto* weight_map = manifest.find("weight_map");
  if (!weight_map || !weight_map->is_object())
    throw std::runtime_error("missing MOSS-Image transformer weight map");
  std::vector<std::string> shards;
  for (const auto& [name, file] : weight_map->members()) {
    (void)name;
    if (!file.is_string() ||
        std::filesystem::path(file.str()).filename().string() != file.str())
      throw std::runtime_error("invalid MOSS-Image shard path");
    if (std::ranges::find(shards, file.str()) == shards.end())
      shards.push_back(file.str());
  }
  for (const auto& shard : shards)
    files.push_back({root / "transformer" / shard, "transformer."});
  files.push_back({root / "vae/diffusion_pytorch_model.safetensors", "vae."});
  // Only the final RMSNorm of the text encoder: the encoder itself runs from
  // the GGUF on the Qwen engine, which returns pre-norm hidden states.
  const auto text = qwen_image_21::ReadJson(
      root / "text_encoder/model.safetensors.index.json");
  const auto* text_map = text.find("weight_map");
  const auto* norm =
      text_map ? text_map->find("language_model.norm.weight") : nullptr;
  if (!norm || !norm->is_string())
    throw std::runtime_error("missing MOSS-Image text encoder norm");
  files.push_back({root / "text_encoder" / norm->str(), "text_encoder."});
  return files;
}

void CheckConfig(const std::filesystem::path& root) {
  const auto dit = qwen_image_21::ReadJson(root / "transformer/config.json");
  for (auto [key, value] : {std::pair{"dim", kDim},
                            {"ffn_dim", 8192},
                            {"num_heads", kHeads},
                            {"num_layers", kLayers},
                            {"text_dim", kTextDim},
                            {"freq_dim", kFreqDim},
                            {"latent_channels", kLatentChannels}}) {
    const auto* v = dit.find(key);
    if (!v || !v->is_number() || v->as_double() != value)
      throw std::runtime_error(std::string("unsupported MOSS-Image ") + key);
  }
  const auto vae = qwen_image_21::ReadJson(root / "vae/config.json");
  if (vae.member_str("_class_name") != "AutoencoderKLFlux2")
    throw std::runtime_error("MOSS-Image expects the FLUX.2 VAE");
}

// Four-axis RoPE table (r, f, h, w) of MOSSImageTransformer._RoPE: image rows
// first, then text. Runtime::Rope reads (cos, sin) per rotated pair.
std::vector<float> RopeTable(int height, int width, int text_length) {
  std::array<float, kRopeBands> bands;
  for (int b = 0; b < kRopeBands; ++b)
    bands[b] = 1.0F / std::pow(10000.0F, static_cast<float>(b) / kRopeBands);
  const int rows = height * width + text_length;
  std::vector<float> table(static_cast<std::size_t>(rows) * kHeadDim);
  for (int row = 0; row < rows; ++row) {
    std::array<float, 4> axes{};
    if (row < height * width) {
      axes = {0.0F, static_cast<float>(text_length),
              static_cast<float>(row / width) - (height - 1) / 2.0F,
              static_cast<float>(row % width) - (width - 1) / 2.0F};
    } else {
      axes[1] = static_cast<float>(row - height * width);
    }
    for (int axis = 0; axis < 4; ++axis)
      for (int b = 0; b < kRopeBands; ++b) {
        const float phase = axes[axis] * bands[b];
        const auto pair = static_cast<std::size_t>(row) * (kHeadDim / 2) +
                          axis * kRopeBands + b;
        table[pair * 2] = std::cos(phase);
        table[pair * 2 + 1] = std::sin(phase);
      }
  }
  return table;
}

// Dense noncausal attention over the joint image+text sequence on the
// MiniMax-H3 WMMA kernel (head dimension 128, head-major inputs).
Matrix DenseAttention(Runtime& rt, const Matrix& q, const Matrix& k,
                      const Matrix& v) {
  const int rows = q.rows;
  auto qh = rt.New(rows, kDim), kh = rt.New(rows, kDim),
       vh = rt.New(rows, kDim);
  qwen_image_21::hip::ReorderHeads(q.data, qh.data, rows, kHeads, kHeads,
                                   kHeadDim, false, rt.stream());
  qwen_image_21::hip::ReorderHeads(k.data, kh.data, rows, kHeads, kHeads,
                                   kHeadDim, false, rt.stream());
  qwen_image_21::hip::ReorderHeads(v.data, vh.data, rows, kHeads, kHeads,
                                   kHeadDim, false, rt.stream());
  // Rows beyond 4096 stage the values in a packed copy.
  const int padded = (rows + 31) / 32 * 32;
  Matrix packed;
  if (rows > 4096)
    packed = rt.New(padded * kHeads, kHeadDim);
  auto out = rt.New(rows, kDim);
  ::gufo::minimax_h3::dit_ops::LaunchWmmaAttention(
      qh.data, kh.data, vh.data, out.data, packed.data,
      static_cast<std::uint32_t>(rows), kHeads, rt.stream());
  qwen_image_21::hip::Check(hipGetLastError());
  return out;
}

// Per-head RMSNorm, then rotate-half RoPE with BF16-rounded cos/sin and
// products (Runtime::Rope mode 2).
Matrix NormRope(Runtime& rt, const Matrix& x, const std::string& norm,
                const Matrix& rope) {
  const auto normalized =
      rt.Normalize(x.Reshape(x.rows * kHeads, kHeadDim), Norm::kRms, norm)
          .Reshape(x.rows, kDim);
  return rt.Rope(normalized, kHeads, kHeadDim, rope, 2);
}

Matrix Attention(Runtime& rt, const Matrix& q, const Matrix& k, const Matrix& v,
                 const std::string& p, const Matrix& rope) {
  return rt.Attention(NormRope(rt, q, p + ".q_norm", rope),
                      NormRope(rt, k, p + ".k_norm", rope), v, kHeads, kHeads,
                      kHeadDim);
}

// MOSSImageTransformer._TextProjection over one prompt's encoder output.
Matrix ProjectText(Runtime& rt, const Matrix& context, const Matrix& rope) {
  auto x = rt.Linear(context, "transformer.text_embedder.proj");
  for (int block = 0; block < 2; ++block) {
    const auto p = "transformer.text_embedder.blocks." + std::to_string(block);
    const auto qkv =
        rt.Linear(rt.Normalize(x, Norm::kRms, p + ".norm1"), p + ".qkv");
    const auto attention =
        Attention(rt, rt.Columns(qkv, 0, kDim), rt.Columns(qkv, kDim, kDim),
                  rt.Columns(qkv, 2 * kDim, kDim), p, rope);
    x = rt.Add(x, rt.Linear(attention, p + ".proj"));
    const auto y = rt.Normalize(x, Norm::kRms, p + ".norm2");
    const auto up = rt.Linear(y, p + ".ffn_up");
    const auto gate = rt.Linear(y, p + ".ffn_gate");
    x = rt.Add(
        x, rt.Linear(hip::Situ(rt, up, 0, gate, 0, up.cols), p + ".ffn_down"));
  }
  return x;
}

struct Conditioning {
  Matrix text;  // projected text tokens [length, dim]
  Matrix rope;  // image + text RoPE table
};

Conditioning PrepareText(Runtime& rt, const Matrix& encoded, int height,
                         int width) {
  Conditioning c;
  c.rope = rt.UploadFloat(RopeTable(height, width, encoded.rows),
                          height * width + encoded.rows, kHeadDim);
  c.text = ProjectText(rt, encoded, c.rope.Slice(height * width, encoded.rows));
  return c;
}

// One transformer forward: [height*width, 128] latents to velocity.
std::vector<float> Transform(Runtime& rt, const Matrix& latent,
                             const Conditioning& c, const Matrix& mod,
                             const Matrix& head, int image_tokens,
                             int skip_layer, const CancellationCheck& cancelled,
                             const Observer& observer, const std::string& tag) {
  auto video = rt.Linear(latent, "transformer.video_embedder");
  auto x = rt.Concat(std::array{video, c.text});
  for (int layer = 0; layer < kLayers; ++layer) {
    if (layer == skip_layer)
      continue;
    Cancel(cancelled);
    const auto p = "transformer.layers." + std::to_string(layer);
    const auto y = hip::ShiftScale(
        rt, rt.Normalize(x, Norm::kRms, p + ".norm1"), mod, 0, 1);
    const auto q =
        NormRope(rt, rt.Linear(y, p + ".proj.q"), p + ".q_norm", c.rope);
    const auto k =
        NormRope(rt, rt.Linear(y, p + ".proj.k"), p + ".k_norm", c.rope);
    const auto v = rt.Linear(y, p + ".proj.v");
    const auto o = rt.Linear(DenseAttention(rt, q, k, v), p + ".proj.o");
    x = hip::GatedResidual(rt, x, o, mod, 2);
    const auto up_gate =
        rt.Linear(hip::ShiftScale(rt, rt.Normalize(x, Norm::kRms, p + ".norm2"),
                                  mod, 3, 4),
                  p + ".proj.ffn_up_gate");
    const int ffn = up_gate.cols / 2;
    const auto down = rt.Linear(hip::Situ(rt, up_gate, 0, up_gate, ffn, ffn),
                                p + ".proj.ffn_down");
    x = hip::GatedResidual(rt, x, down, mod, 5);
    rt.Observe(observer, tag + ".block." + std::to_string(layer), x);
  }
  auto out = hip::ShiftScale(rt,
                             rt.Normalize(x.Slice(0, image_tokens), Norm::kRms,
                                          "transformer.final_norm_video"),
                             head, 0, 1);
  out = rt.Linear(out, "transformer.final_linear_video");
  rt.Observe(observer, tag + ".output", out);
  return rt.Download(out);
}

// FLUX.2 VAE decoder (diffusers AutoencoderKLFlux2.decoder), channels-last.
Matrix Group(Runtime& rt, const Matrix& x, const std::string& name, bool silu) {
  return hip::GroupNorm(rt, x, rt.Weight(name + ".weight"),
                        rt.Weight(name + ".bias"), 32, 1e-6F, silu);
}

Matrix Resnet(Runtime& rt, const Weights& weights, const Matrix& x, int height,
              int width, const std::string& p) {
  auto h =
      rt.Conv(Group(rt, x, p + ".norm1", true), height, width, p + ".conv1");
  h = rt.Conv(Group(rt, h, p + ".norm2", true), height, width, p + ".conv2");
  const auto shortcut =
      weights.Contains(p + ".conv_shortcut.weight")
          ? rt.Conv(x, height, width, p + ".conv_shortcut", 1, 0)
          : x;
  return rt.Add(shortcut, h);
}

Matrix VaeAttention(Runtime& rt, const Matrix& x, const std::string& p) {
  const auto h = Group(rt, x, p + ".group_norm", false);
  const auto attention =
      rt.Attention(rt.Linear(h, p + ".to_q"), rt.Linear(h, p + ".to_k"),
                   rt.Linear(h, p + ".to_v"), 1, 1, x.cols);
  return rt.Add(x, rt.Linear(attention, p + ".to_out.0"));
}

Image DecodeVae(Runtime& rt, const Weights& weights,
                const std::vector<float>& latent, int height, int width,
                const CancellationCheck& cancelled, const Observer& observer) {
  // Undo the latent BatchNorm and the 2x2 patching on the host in FP32:
  // channel c*4 + i*2 + j of latent pixel (y, x) is pixel (2y+i, 2x+j) of
  // VAE channel c.
  const auto& mean = weights.Get("vae.bn.running_mean");
  const auto& var = weights.Get("vae.bn.running_var");
  if (mean.bf16 || var.bf16)
    throw std::runtime_error("MOSS-Image expects FP32 VAE batch statistics");
  const auto* m = static_cast<const float*>(mean.data);
  const auto* v = static_cast<const float*>(var.data);
  const int vh = height * 2, vw = width * 2;
  std::vector<float> z(static_cast<std::size_t>(vh) * vw * 32);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int ch = 0; ch < kLatentChannels; ++ch) {
        const float value =
            latent[(static_cast<std::size_t>(y) * width + x) * kLatentChannels +
                   ch] *
                std::sqrt(v[ch] + 1e-4F) +
            m[ch];
        const int c = ch / 4, i = (ch / 2) % 2, j = ch % 2;
        z[(static_cast<std::size_t>(2 * y + i) * vw + 2 * x + j) * 32 + c] =
            value;
      }
  int h = vh, w = vw;
  auto x = rt.Upload(z, h * w, 32);
  x = rt.Conv(x, h, w, "vae.post_quant_conv", 1, 0);
  x = rt.Conv(x, h, w, "vae.decoder.conv_in");
  x = Resnet(rt, weights, x, h, w, "vae.decoder.mid_block.resnets.0");
  x = VaeAttention(rt, x, "vae.decoder.mid_block.attentions.0");
  x = Resnet(rt, weights, x, h, w, "vae.decoder.mid_block.resnets.1");
  rt.Observe(observer, "vae.decoder.mid", x);
  for (int block = 0; block < 4; ++block) {
    Cancel(cancelled);
    const auto p = "vae.decoder.up_blocks." + std::to_string(block);
    for (int layer = 0; layer < 3; ++layer)
      x = Resnet(rt, weights, x, h, w, p + ".resnets." + std::to_string(layer));
    if (block < 3) {
      // Nearest 2x upsampling folded into the 3x3 convolution.
      x = rt.Conv(x, h, w, p + ".upsamplers.0.conv", 1, 1, 2);
      h *= 2;
      w *= 2;
    }
    rt.Observe(observer, "vae.decoder.block." + std::to_string(block), x);
  }
  x = rt.Conv(Group(rt, x, "vae.decoder.conv_norm_out", true), h, w,
              "vae.decoder.conv_out");
  rt.Observe(observer, "vae.decoded", x);
  if (x.cols != 3)
    throw std::runtime_error("expected RGB VAE output");
  const auto pixels = rt.Download(x);
  Image image{w, h,
              std::vector<std::uint8_t>(static_cast<std::size_t>(w) * h * 4)};
  for (std::size_t p = 0; p < static_cast<std::size_t>(w) * h; ++p) {
    for (int c = 0; c < 3; ++c) {
      const float value = pixels[p * 3 + c];
      if (!std::isfinite(value))
        throw std::runtime_error("nonfinite image output");
      // VaeImageProcessor: (x / 2 + 0.5).clamp(0, 1) * 255, rounded.
      const float unit = std::clamp(value / 2.0F + 0.5F, 0.0F, 1.0F);
      image.rgba[p * 4 + c] =
          static_cast<std::uint8_t>(std::nearbyint(unit * 255.0F));
    }
    image.rgba[p * 4 + 3] = 255;
  }
  return image;
}

}  // namespace

struct Model::Impl {
  Weights weights;
  Runtime runtime;
  std::shared_ptr<const core::GgufReader> reader;
  std::unique_ptr<gufo::hip::QwenGpuExecutor> encoder;
  Options options;
  std::mutex mutex;

  Impl(const std::filesystem::path& root, Options opts)
      : weights(Weights::FromFiles(
            Sources(root), qwen_image_21::ReadJson(root / "vae/config.json"))),
        runtime(weights),
        options(std::move(opts)) {
    CheckConfig(root);
    std::string error;
    auto opened = core::GgufReader::OpenFile(options.text_encoder, &error);
    if (!opened)
      throw std::runtime_error("cannot open MOSS-Image text encoder: " + error);
    reader = std::shared_ptr<const core::GgufReader>(std::move(opened));
    encoder = gufo::hip::QwenGpuExecutor::CreateFromGguf(reader, &error,
                                                         kMaxTextTokens);
    if (!encoder)
      throw std::runtime_error("cannot load MOSS-Image text encoder: " + error);
    if (encoder->GetConfig().hidden_size != kTextDim)
      throw std::runtime_error("MOSS-Image text encoder width mismatch");
  }

  // Qwen3_5Model.last_hidden_state of the chat-formatted prompt, including
  // the final RMSNorm, as BF16 rows.
  Matrix EncodeText(const std::string& prompt) {
    std::string text(tokenization::kImStart);
    text.append("user\n").append(prompt).append(tokenization::kImEnd);
    text.append("\n").append(tokenization::kImStart).append("assistant\n");
    // enable_thinking=True leaves the think block open.
    text.append("<think>\n");
    const auto ids = encoder->GetTokenizer().Encode(text);
    if (ids.empty() || ids.size() > kMaxTextTokens)
      throw std::invalid_argument("MOSS-Image prompt exceeds 4096 tokens");
    encoder->Reset();
    encoder->SetPromptHiddenCapture(true);
    (void)encoder->ForwardPromptBatch(ids, 0, false);
    const auto hidden = encoder->GetPromptHiddenStates();
    std::vector<float> rows(hidden.begin(), hidden.end());
    encoder->SetPromptHiddenCapture(false);
    encoder->Reset();
    auto x = runtime.Upload(rows, static_cast<int>(ids.size()), kTextDim);
    return runtime.Normalize(x, Norm::kZeroRms,
                             "text_encoder.language_model.norm");
  }
};

Model::Model(const std::filesystem::path& root, Options options)
    : impl_(std::make_unique<Impl>(root, std::move(options))) {}
Model::~Model() = default;

Result Model::Generate(const Request& request,
                       const CancellationCheck& cancelled,
                       const Observer& observer) {
  moss_image::ValidateRequest(request);
  std::unique_lock lock(impl_->mutex);
  Cancel(cancelled);
  const auto start = Clock::now();
  auto& rt = impl_->runtime;
  const auto& options = impl_->options;
  Result result;
  try {
    const int height = request.height / 16, width = request.width / 16;
    const int tokens = height * width;
    const bool guided = options.guidance_scale != 1.0F;
    const auto positive = impl_->EncodeText(request.prompt);
    rt.Observe(observer, "text.positive", positive);
    const auto cond = PrepareText(rt, positive, height, width);
    Conditioning uncond;
    if (guided) {
      const auto negative = impl_->EncodeText("");
      rt.Observe(observer, "text.negative", negative);
      uncond = PrepareText(rt, negative, height, width);
    }
    rt.Synchronize();
    result.metrics.prompt_ms = Milliseconds(start);

    // Noise in FP32, [tokens, channels]. Options::initial_noise replays a
    // reference draw stored as raw FP32 [channels, height, width].
    std::vector<float> latent(static_cast<std::size_t>(tokens) *
                              kLatentChannels);
    if (!options.initial_noise.empty()) {
      std::vector<float> planar(latent.size());
      std::ifstream file(options.initial_noise, std::ios::binary);
      file.read(reinterpret_cast<char*>(planar.data()),
                static_cast<std::streamsize>(planar.size() * sizeof(float)));
      if (!file || file.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("initial noise does not match the size");
      for (int t = 0; t < tokens; ++t)
        for (int c = 0; c < kLatentChannels; ++c)
          latent[static_cast<std::size_t>(t) * kLatentChannels + c] =
              planar[static_cast<std::size_t>(c) * tokens + t];
    } else {
      std::mt19937_64 rng(request.seed);
      std::normal_distribution<float> normal(0, 1);
      for (float& value : latent)
        value = normal(rng);
    }
    if (observer.Wants("noise"))
      observer.write("noise", latent, tokens, kLatentChannels);
    const auto sigmas = FlowSigmas(request.steps);
    if (observer.Wants("sigmas"))
      observer.write("sigmas", sigmas, 1, static_cast<int>(sigmas.size()));
    const auto denoise_start = Clock::now();
    for (std::size_t step = 0; step + 1 < sigmas.size(); ++step) {
      Cancel(cancelled);
      // Timestep sinusoid in FP64 as in training, fed to the MLP in BF16.
      const double t = static_cast<double>(sigmas[step] * 1000.0F);
      std::vector<float> embedding(kFreqDim);
      for (int i = 0; i < kFreqDim / 2; ++i) {
        const double freq =
            std::pow(10000.0, -static_cast<double>(i) / (kFreqDim / 2));
        embedding[i] = static_cast<float>(std::cos(t * freq));
        embedding[kFreqDim / 2 + i] = static_cast<float>(std::sin(t * freq));
      }
      const auto time =
          rt.Linear(rt.Activate(rt.Linear(rt.Upload(embedding, 1, kFreqDim),
                                          "transformer.time_embedding.0"),
                                Activation::kSilu),
                    "transformer.time_embedding.2");
      const auto mod = rt.Linear(time, "transformer.shared_block_modulation");
      const auto head = rt.Linear(time, "transformer.video_head_modulation");
      const auto x = rt.Upload(latent, tokens, kLatentChannels);
      const auto tag = "dit." + std::to_string(step);
      auto prediction = Transform(rt, x, cond, mod, head, tokens, -1, cancelled,
                                  observer, tag + ".cond");
      if (guided) {
        const auto negative = Transform(rt, x, uncond, mod, head, tokens,
                                        options.uncond_skip_layer, cancelled,
                                        observer, tag + ".uncond");
        for (std::size_t i = 0; i < prediction.size(); ++i)
          prediction[i] = negative[i] + options.guidance_scale *
                                            (prediction[i] - negative[i]);
      }
      const float dt = sigmas[step + 1] - sigmas[step];
      for (std::size_t i = 0; i < latent.size(); ++i)
        latent[i] += dt * prediction[i];
      if (observer.Wants("latent." + std::to_string(step)))
        observer.write("latent." + std::to_string(step), latent, tokens,
                       kLatentChannels);
    }
    rt.Synchronize();
    result.metrics.denoise_ms = Milliseconds(denoise_start);
    const auto vae_start = Clock::now();
    result.image = DecodeVae(rt, impl_->weights, latent, height, width,
                             cancelled, observer);
    result.metrics.vae_ms = Milliseconds(vae_start);
    result.metrics.total_ms = Milliseconds(start);
    result.metrics.resident_weight_bytes = rt.WeightBytes();
    return result;
  } catch (...) {
    (void)hipStreamSynchronize(rt.stream());
    throw;
  }
}

}  // namespace gufo::models::moss_image
