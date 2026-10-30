#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen_image_21/hip/memory.hpp"
#include "src/models/qwen_image_21/hip/runtime.hpp"
#include "src/models/qwen_image_21/tokenizer.hpp"

namespace gufo::models::qwen_image_21 {
namespace {

using hip::Activation;
using hip::Matrix;
using hip::Norm;
using hip::Runtime;
using Clock = std::chrono::steady_clock;

float Bf16(float x) {
  auto bits = std::bit_cast<std::uint32_t>(x);
  bits = (bits + 0x7fffU + ((bits >> 16) & 1)) & 0xffff0000U;
  return std::bit_cast<float>(bits);
}

double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

void Cancel(const CancellationCheck& cancelled) {
  if (cancelled && cancelled())
    throw std::runtime_error("image request cancelled");
}

/// Observe a second pass (negative prompt, init image) under its own names.
Observer Prefixed(const Observer& observer, std::string prefix) {
  if (!observer.write)
    return {};
  Observer result;
  result.write = [observer, prefix](std::string_view name,
                                    std::span<const float> values, int rows,
                                    int cols) {
    observer.write(prefix + std::string(name), values, rows, cols);
  };
  result.include = [observer, prefix](std::string_view name) {
    return observer.Wants(prefix + std::string(name));
  };
  return result;
}

Matrix Concat(Runtime& rt, std::initializer_list<Matrix> values) {
  return rt.Concat(std::span(values.begin(), values.size()));
}

struct Visual {
  Matrix embedded;
  std::array<Matrix, 3> deep;
};

Matrix VisionMerge(Runtime& rt, const Matrix& x, const std::string& name,
                   bool postshuffle) {
  auto norm = rt.Normalize(postshuffle ? x.Reshape(x.rows / 4, x.cols * 4) : x,
                           Norm::kLayer, name + ".norm");
  auto flat = norm.Reshape(x.rows / 4, x.cols * 4);
  return rt.Linear(
      rt.Activate(rt.Linear(flat, name + ".linear_fc1"), Activation::kGelu),
      name + ".linear_fc2");
}

Visual EncodeVision(Runtime& rt, const Image& image,
                    const CancellationCheck& cancelled,
                    const Observer& observer) {
  const int height = image.height / 16, width = image.width / 16;
  if (height < 2 || width < 2 || height % 2 || width % 2)
    throw std::invalid_argument("vision input must be aligned to 32 pixels");
  constexpr int patch_width = 3 * 2 * 16 * 16;
  std::vector<float> patches(static_cast<std::size_t>(height) * width *
                             patch_width);
  std::vector<float> frequencies(static_cast<std::size_t>(height) * width * 72);
  for (int token = 0; token < height * width; ++token) {
    const int py = (token / 4 / (width / 2)) * 2 + (token / 2) % 2;
    const int px = ((token / 4) % (width / 2)) * 2 + token % 2;
    for (int c = 0; c < 3; ++c)
      for (int t = 0; t < 2; ++t)
        for (int y = 0; y < 16; ++y)
          for (int x = 0; x < 16; ++x) {
            const auto pixel =
                (static_cast<std::size_t>(py * 16 + y) * image.width + px * 16 +
                 x) *
                4;
            const int alpha = image.rgba[pixel + 3];
            // PIL's paste over white uses DIV255 rounded integer arithmetic.
            const int product =
                image.rgba[pixel + c] * alpha + 255 * (255 - alpha) + 128;
            const int color = (product + (product >> 8)) >> 8;
            patches[static_cast<std::size_t>(token) * patch_width +
                    ((c * 2 + t) * 16 + y) * 16 + x] =
                (static_cast<float>(color) / 255.0F - 0.5F) / 0.5F;
          }
    for (int pair = 0; pair < 36; ++pair) {
      const int position = pair < 18 ? py : px;
      const float angle = position / std::pow(10000.0F, (pair % 18) / 18.0F);
      frequencies[(static_cast<std::size_t>(token) * 36 + pair) * 2] =
          std::cos(angle);
      frequencies[(static_cast<std::size_t>(token) * 36 + pair) * 2 + 1] =
          std::sin(angle);
    }
  }
  const std::string prefix = "text_encoder.model.visual.";
  auto x = rt.Linear(rt.Upload(patches, height * width, patch_width),
                     prefix + "patch_embed.proj");
  x = rt.VisionPosition(x, height, width);
  rt.Observe(observer, "vision.embedding", x);
  const auto rope = rt.UploadFloat(frequencies, height * width, 72);
  Visual output;
  for (int layer = 0; layer < 27; ++layer) {
    Cancel(cancelled);
    const auto p = prefix + "blocks." + std::to_string(layer);
    const auto norm = rt.Normalize(x, Norm::kLayer, p + ".norm1");
    const auto qkv = rt.Linear(norm, p + ".attn.qkv");
    const auto q = rt.Rope(rt.Columns(qkv, 0, 1152), 16, 72, rope, 1);
    const auto k = rt.Rope(rt.Columns(qkv, 1152, 1152), 16, 72, rope, 1);
    const auto v = rt.Columns(qkv, 2304, 1152);
    x = rt.Add(x,
               rt.Linear(rt.Attention(q, k, v, 16, 16, 72), p + ".attn.proj"));
    x = rt.Add(x, rt.Linear(rt.Activate(rt.Linear(rt.Normalize(x, Norm::kLayer,
                                                               p + ".norm2"),
                                                  p + ".mlp.linear_fc1"),
                                        Activation::kGeluTanh),
                            p + ".mlp.linear_fc2"));
    rt.Observe(observer, "vision.block." + std::to_string(layer), x);
    if (layer == 8 || layer == 16 || layer == 24) {
      const int index = layer / 8 - 1;
      output.deep[index] = VisionMerge(
          rt, x, prefix + "deepstack_merger_list." + std::to_string(index),
          true);
    }
  }
  output.embedded = VisionMerge(rt, x, prefix + "merger", false);
  rt.Observe(observer, "vision.output", output.embedded);
  return output;
}

constexpr std::string_view kSystemText =
    "system\nComprehend and analyze the provided prompt.";
constexpr std::string_view kNewline = "\n";
constexpr auto kSystemStorage =
    tokenization::ConcatControlText<tokenization::kImStart, kSystemText,
                                    tokenization::kImEnd, kNewline>();
constexpr std::string_view kSystem(kSystemStorage.data(),
                                   kSystemStorage.size());

struct EncodedPrompt {
  Matrix hidden;
  std::vector<std::uint32_t> ids;
  std::vector<Image> images;
  std::size_t drop{0};
};

EncodedPrompt EncodePrompt(Runtime& rt, const Tokenizer& tokenizer,
                           const Request& request,
                           const CancellationCheck& cancelled,
                           const Observer& observer) {
  EncodedPrompt output;
  std::string text(kSystem);
  text.append(tokenization::kImStart).append("user\n");
  for (std::size_t i = 0; i < request.images.size(); ++i) {
    if (i)
      text += ' ';
    text += "<image" + std::to_string(i + 1) + ">";
    text.append(tokenization::kVisionStart)
        .append(tokenization::kImagePad)
        .append(tokenization::kVisionEnd);
    const auto& image = request.images[i];
    const double ratio = static_cast<double>(image.width) / image.height;
    const double width = std::sqrt(1024.0 * 1024 * ratio);
    const double height = width / ratio;
    const int rw = static_cast<int>(std::nearbyint(width / 32)) * 32;
    const int rh = static_cast<int>(std::nearbyint(height / 32)) * 32;
    if (rw < 32 || rh < 32 || rw > 8192 || rh > 8192)
      throw std::invalid_argument(
          "reference image aspect ratio exceeds limits");
    output.images.push_back(ResizeImage(image, rw, rh));
  }
  text += request.prompt.empty() ? " " : request.prompt;
  text.append(tokenization::kImEnd).append("\n");
  text.append(tokenization::kImStart).append("assistant\n");
  const auto raw_ids = tokenizer.Encode(text);
  std::vector<int> image_positions;
  std::vector<std::array<int, 3>> positions;
  std::vector<Visual> visuals;
  std::size_t image_index = 0;
  int position = 0;
  for (const auto id : raw_ids) {
    if (id != 151655) {
      output.ids.push_back(id);
      positions.push_back({position, position, position});
      ++position;
      continue;
    }
    if (image_index >= output.images.size())
      throw std::invalid_argument("prompt contains an unmatched image token");
    const auto& image = output.images[image_index++];
    const int h = image.height / 32, w = image.width / 32;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        image_positions.push_back(static_cast<int>(output.ids.size()));
        output.ids.push_back(151655);
        positions.push_back({position, position + y, position + x});
      }
    position += std::max(h, w);
    visuals.push_back(EncodeVision(rt, image, cancelled, observer));
  }
  if (image_index != output.images.size())
    throw std::logic_error("image prompt layout mismatch");
  if (output.ids.size() > 16384)
    throw std::invalid_argument("image prompt exceeds 16384 encoder tokens");
  output.drop = tokenizer.Encode(kSystem).size();
  const std::string prefix = "text_encoder.model.language_model.";
  auto x = rt.Embed(rt.Weight(prefix + "embed_tokens.weight"), output.ids);
  std::array<Matrix, 3> deep;
  if (!visuals.empty()) {
    std::vector<Matrix> embeds;
    for (const auto& v : visuals)
      embeds.push_back(v.embedded);
    x = rt.ReplaceRows(x, rt.Concat(embeds), image_positions, false);
    for (int i = 0; i < 3; ++i) {
      embeds.clear();
      for (const auto& v : visuals)
        embeds.push_back(v.deep[i]);
      deep[i] = rt.Concat(embeds);
    }
  }
  std::vector<float> frequencies(output.ids.size() * 128);
  for (std::size_t token = 0; token < positions.size(); ++token)
    for (int pair = 0; pair < 64; ++pair) {
      const int axis = pair < 60 ? pair % 3 : 0;
      const float angle =
          positions[token][axis] / std::pow(5000000.0F, pair / 64.0F);
      frequencies[(token * 64 + pair) * 2] = std::cos(angle);
      frequencies[(token * 64 + pair) * 2 + 1] = std::sin(angle);
    }
  const auto rope = rt.UploadFloat(frequencies, x.rows, 128);
  std::vector<int> limits(x.rows);
  std::iota(limits.begin(), limits.end(), 1);
  rt.Observe(observer, "text.embedding", x);
  for (int layer = 0; layer < 36; ++layer) {
    Cancel(cancelled);
    const auto p = prefix + "layers." + std::to_string(layer);
    const auto norm = rt.Normalize(x, Norm::kRms, p + ".input_layernorm");
    const auto q = rt.Rope(rt.Normalize(rt.Linear(norm, p + ".self_attn.q_proj")
                                            .Reshape(x.rows * 32, 128),
                                        Norm::kRms, p + ".self_attn.q_norm")
                               .Reshape(x.rows, 4096),
                           32, 128, rope, 2);
    const auto k = rt.Rope(
        rt.Normalize(
              rt.Linear(norm, p + ".self_attn.k_proj").Reshape(x.rows * 8, 128),
              Norm::kRms, p + ".self_attn.k_norm")
            .Reshape(x.rows, 1024),
        8, 128, rope, 2);
    const auto v = rt.Linear(norm, p + ".self_attn.v_proj");
    x = rt.Add(x, rt.Linear(rt.Attention(q, k, v, 32, 8, 128, limits),
                            p + ".self_attn.o_proj"));
    const auto norm2 =
        rt.Normalize(x, Norm::kRms, p + ".post_attention_layernorm");
    x = rt.Add(x, rt.Linear(rt.SwiGlu(rt.Linear(norm2, p + ".mlp.gate_proj"),
                                      rt.Linear(norm2, p + ".mlp.up_proj")),
                            p + ".mlp.down_proj"));
    if (layer < 3 && !visuals.empty())
      x = rt.ReplaceRows(x, deep[layer], image_positions, true);
    rt.Observe(observer, "text.block." + std::to_string(layer), x);
  }
  // The diffusion checkpoint expects pre-final-norm hidden states.
  output.hidden = x.Slice(static_cast<int>(output.drop),
                          x.rows - static_cast<int>(output.drop));
  rt.Observe(observer, "text.output", output.hidden);
  return output;
}

Matrix VaeResidual(Runtime& rt, const Matrix& x, int height, int width,
                   const std::string& name, const Weights& weights) {
  Matrix shortcut = x;
  if (weights.Contains(name + ".conv_shortcut.weight"))
    shortcut = rt.Conv(x, height, width, name + ".conv_shortcut", 1, 0);
  auto h = rt.Conv(rt.Normalize(x, Norm::kVae, name + ".norm1", 1e-6F, true),
                   height, width, name + ".conv1");
  h = rt.Conv(rt.Normalize(h, Norm::kVae, name + ".norm2", 1e-6F, true), height,
              width, name + ".conv2");
  return rt.Add(h, shortcut);
}

Matrix VaeMid(Runtime& rt, const Matrix& x, int height, int width,
              const std::string& p, const Weights& weights) {
  auto h = VaeResidual(rt, x, height, width, p + ".resnets.0", weights);
  const int channels = h.cols;
  auto qkv = rt.Conv(rt.Normalize(h, Norm::kVae, p + ".attentions.0.norm"),
                     height, width, p + ".attentions.0.to_qkv", 1, 0);
  auto attention = rt.Attention(
      rt.Columns(qkv, 0, channels), rt.Columns(qkv, channels, channels),
      rt.Columns(qkv, channels * 2, channels), 1, 1, channels);
  h = rt.Add(h,
             rt.Conv(attention, height, width, p + ".attentions.0.proj", 1, 0));
  return VaeResidual(rt, h, height, width, p + ".resnets.1", weights);
}

std::array<float, 64> VaeConstants(const Weights& weights, const char* key) {
  std::array<float, 64> values;
  const auto& items = weights.VaeConfig().find(key)->items();
  for (int i = 0; i < 64; ++i)
    values[i] = static_cast<float>(items[i].as_double());
  return values;
}

Matrix EncodeVae(Runtime& rt, const Weights& weights, const Image& image,
                 const CancellationCheck& cancelled, const Observer& observer) {
  std::vector<float> input(image.rgba.size());
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(image.rgba[i]) / 255.0F * 2.0F - 1.0F;
  int height = image.height, width = image.width;
  auto x = rt.Conv(rt.Upload(input, height * width, 4), height, width,
                   "vae.encoder.conv_in");
  rt.Observe(observer, "vae.encoder.input", x);
  constexpr std::array<int, 5> channels{96, 192, 384, 768, 768};
  for (int block = 0; block < 5; ++block) {
    Cancel(cancelled);
    const auto p = "vae.encoder.down_blocks." + std::to_string(block);
    const auto skip =
        rt.DownShortcut(x, height, width, channels[block],
                        block > 0 && block < 4 ? 2 : 1, block < 4 ? 2 : 1);
    for (int layer = 0; layer < 2; ++layer)
      x = VaeResidual(rt, x, height, width,
                      p + ".resnets." + std::to_string(layer), weights);
    if (block < 4) {
      // ZeroPad2d((0,1,0,1)) followed by valid stride-2 convolution.
      // The output has H/2 by W/2, unlike symmetric padding.
      x = rt.Conv(x, height, width, p + ".downsampler.resample.1", 2, -1);
      height /= 2;
      width /= 2;
    }
    x = rt.Add(x, skip);
    rt.Observe(observer, "vae.encoder.block." + std::to_string(block), x);
  }
  x = VaeMid(rt, x, height, width, "vae.encoder.mid_block", weights);
  x = rt.Conv(rt.Normalize(x, Norm::kVae, "vae.encoder.norm_out", 1e-6F, true),
              height, width, "vae.encoder.conv_out");
  x = rt.Conv(x, height, width, "vae.quant_conv", 1, 0);
  x = rt.Columns(x, 0, 64);
  x = rt.LatentScale(x, VaeConstants(weights, "latents_mean"),
                     VaeConstants(weights, "latents_std"), true);
  rt.Observe(observer, "vae.encoded", x);
  return x;
}

Image DecodeVae(Runtime& rt, const Weights& weights, const Matrix& latent,
                int height, int width, const CancellationCheck& cancelled,
                const Observer& observer) {
  auto x = rt.LatentScale(latent, VaeConstants(weights, "latents_mean"),
                          VaeConstants(weights, "latents_std"), false);
  x = rt.Conv(x, height, width, "vae.post_quant_conv", 1, 0);
  x = rt.Conv(x, height, width, "vae.decoder.conv_in");
  x = VaeMid(rt, x, height, width, "vae.decoder.mid_block", weights);
  rt.Observe(observer, "vae.decoder.mid", x);
  constexpr std::array<int, 5> channels{1152, 1152, 576, 288, 144};
  for (int block = 0; block < 5; ++block) {
    Cancel(cancelled);
    const auto p = "vae.decoder.up_blocks." + std::to_string(block);
    Matrix skip;
    if (block < 4)
      skip =
          rt.UpShortcut(x, height, width, channels[block], block < 3 ? 2 : 1);
    for (int layer = 0; layer < 3; ++layer)
      x = VaeResidual(rt, x, height, width,
                      p + ".resnets." + std::to_string(layer), weights);
    if (block < 4) {
      // The first and only image frame skips all temporal convolutions.
      x = rt.Conv(x, height, width, p + ".upsampler.resample.1", 1, 1, 2);
      height *= 2;
      width *= 2;
      x = rt.Add(x, skip);
    }
    rt.Observe(observer, "vae.decoder.block." + std::to_string(block), x);
  }
  x = rt.Conv(rt.Normalize(x, Norm::kVae, "vae.decoder.norm_out", 1e-6F, true),
              height, width, "vae.decoder.conv_out");
  rt.Observe(observer, "vae.decoded", x);
  if (x.cols != 4)
    throw std::runtime_error("expected RGBA VAE output");
  const auto pixels = rt.Download(x);
  Image image{width, height, std::vector<std::uint8_t>(pixels.size())};
  for (std::size_t i = 0; i < pixels.size(); ++i) {
    if (!std::isfinite(pixels[i]))
      throw std::runtime_error("nonfinite image output");
    // Diffusers divides and adds in the VAE dtype before converting to FP32.
    const float v = Bf16(Bf16(std::clamp(pixels[i], -1.0F, 1.0F) / 2) + 0.5F);
    image.rgba[i] = static_cast<std::uint8_t>(std::nearbyint(v * 255));
  }
  return image;
}

struct DiffusionContext {
  Matrix prefix;
  Matrix rope;
  std::vector<int> limits;
  std::array<Matrix, 32> keys, values;
  // Leading condition rows taken from `base`'s caches instead of `prefix`.
  // Block-causal rows never see later tokens, so a prefix that is identical
  // in both prompts (the reference images before the prompt text) has
  // identical K/V for the negative pass. `rope` and `limits` cover only the
  // own rows; limits count keys in the full sequence.
  const DiffusionContext* base{nullptr};
  int shared_rows{0};
  // Prefill already filled keys/values; every step then runs target rows.
  bool prefilled{false};
};

/// Leading encoder tokens shared by two prompts with the same references,
/// ending after the last shared reference image; 0 without one. The
/// remaining tail is never empty.
int SharedConditionTokens(const EncodedPrompt& a, const EncodedPrompt& b) {
  const std::size_t length =
      std::min(a.ids.size() - a.drop, b.ids.size() - b.drop);
  std::size_t common = 0;
  while (common + 1 < length &&
         a.ids[a.drop + common] == b.ids[b.drop + common])
    ++common;
  std::size_t shared = 0;
  for (std::size_t i = 0; i < common; ++i)
    if (a.ids[a.drop + i] == 151655 &&
        (i + 1 == common || a.ids[a.drop + i + 1] != 151655))
      shared = i + 1;
  return static_cast<int>(shared);
}

DiffusionContext PrepareDiffusion(Runtime& rt, const Weights& weights,
                                  const EncodedPrompt& prompt, int height,
                                  int width, const CancellationCheck& cancelled,
                                  const Observer& observer,
                                  const DiffusionContext* base = nullptr,
                                  int shared_tokens = 0) {
  DiffusionContext context;
  context.base = shared_tokens ? base : nullptr;
  auto text = rt.Linear(
      rt.Activate(rt.Linear(rt.Normalize(prompt.hidden, Norm::kZeroRms,
                                         "transformer.txt_in.text_norm"),
                            "transformer.txt_in.in_layer"),
                  Activation::kGeluTanh),
      "transformer.txt_in.out_layer");
  std::vector<Matrix> parts;
  std::vector<std::array<int, 3>> positions;
  int position = 0;
  int cursor = 0;
  std::size_t image_index = 0;
  const int length = text.rows;
  while (cursor < length) {
    if (cursor == shared_tokens)
      context.shared_rows = static_cast<int>(positions.size());
    const bool shared = cursor < shared_tokens;
    if (prompt.ids[prompt.drop + cursor] == 151655) {
      const auto& image = prompt.images.at(image_index++);
      const int slots = (image.height / 32) * (image.width / 32);
      const int h = image.height / 16, w = image.width / 16;
      if (shared && cursor + slots > shared_tokens)
        throw std::logic_error("shared condition ends inside an image");
      if (!shared)
        parts.push_back(
            rt.Linear(EncodeVae(rt, weights, image, cancelled, observer),
                      "transformer.img_in"));
      const int end = static_cast<int>(positions.size()) + h * w;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          positions.push_back({position, y - (h - h / 2), x - (w - w / 2)});
          context.limits.push_back(end);
        }
      position += std::max(h, w);
      cursor += slots;
    } else {
      const int start = cursor;
      const int stop = shared ? shared_tokens : length;
      while (cursor < stop && prompt.ids[prompt.drop + cursor] != 151655) {
        positions.push_back({position, position, position});
        context.limits.push_back(static_cast<int>(positions.size()));
        ++position;
        ++cursor;
      }
      if (!shared)
        parts.push_back(text.Slice(start, cursor - start));
    }
  }
  if (shared_tokens && (!context.base || shared_tokens >= length ||
                        context.shared_rows > context.base->prefix.rows))
    throw std::logic_error("invalid shared condition prefix");
  context.prefix = rt.Concat(parts);
  const int end = static_cast<int>(positions.size()) + height * width;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      positions.push_back(
          {position, y - (height - height / 2), x - (width - width / 2)});
      context.limits.push_back(end);
    }
  if (context.shared_rows) {
    positions.erase(positions.begin(), positions.begin() + context.shared_rows);
    context.limits.erase(context.limits.begin(),
                         context.limits.begin() + context.shared_rows);
  }
  std::vector<float> rope(positions.size() * 128);
  for (std::size_t row = 0; row < positions.size(); ++row)
    for (int pair = 0; pair < 64; ++pair) {
      const int axis = pair < 8 ? 0 : (pair < 36 ? 1 : 2);
      const int first = axis == 0 ? 0 : (axis == 1 ? 8 : 36);
      const int count = axis == 0 ? 8 : 28;
      const float angle =
          positions[row][axis] /
          std::pow(10000.0F, static_cast<float>(pair - first) / count);
      rope[(row * 64 + pair) * 2] = std::cos(angle);
      rope[(row * 64 + pair) * 2 + 1] = std::sin(angle);
    }
  context.rope = rt.UploadFloat(rope, static_cast<int>(positions.size()), 128);
  return context;
}

/// DiT condition rows of a prompt: text tokens plus 4 rows per image token.
int ConditionRows(const EncodedPrompt& prompt) {
  int rows = static_cast<int>(prompt.ids.size() - prompt.drop);
  for (const auto& image : prompt.images)
    rows += 3 * (image.height / 32) * (image.width / 32);
  return rows;
}

// About one 1 MP reference: 0-1 references keep every weight resident.
constexpr int kReleaseEncoderRows = 6144;
// Host RAM left to Windows and other processes when shared GPU memory grows.
constexpr std::uint64_t kHostReserveBytes = 3ULL << 30;
constexpr std::uint64_t kMinimumFreeBytes = 512ULL << 20;

/// DiT rows of encoder tokens [begin, end): 4 per image token, 1 otherwise.
int RowsOf(const EncodedPrompt& prompt, std::size_t begin, std::size_t end) {
  int rows = 0;
  for (std::size_t i = prompt.drop + begin; i < prompt.drop + end; ++i)
    rows += prompt.ids[i] == 151655 ? 4 : 1;
  return rows;
}

/// GPU memory a large request still needs after the text encoder left:
/// missing DiT/VAE weights, both condition caches, the larger of the prefill
/// and per-step working sets, the bounded pool cache and a margin.
std::uint64_t LargeRequestBytes(const Runtime& rt, const EncodedPrompt& prompt,
                                const EncodedPrompt* negative,
                                int target_rows) {
  constexpr std::uint64_t kCacheRow = 32ULL * 2 * 4096 * 2;
  // Activations of one row through a DiT block (about 19 row-sized
  // 4096-wide BF16 tensors including the 12288-wide MLP).
  constexpr std::uint64_t kWorkRow = 19ULL * 4096 * 2;
  const std::uint64_t rows = ConditionRows(prompt);
  std::uint64_t cache = rows;
  if (negative) {
    const auto shared = SharedConditionTokens(prompt, *negative);
    cache += RowsOf(*negative, shared, negative->ids.size() - negative->drop);
  }
  const std::uint64_t prefill = 3 * rows * 4096 * 2 + 4096 * kWorkRow;
  const std::uint64_t step = static_cast<std::uint64_t>(target_rows) * kWorkRow;
  return rt.MissingBytes("transformer.") + rt.MissingBytes("vae.") +
         cache * kCacheRow + std::max(prefill, step) + (2ULL << 30) +
         (1ULL << 30);
}
constexpr std::size_t kDefaultCacheBytes = 8ULL << 30;
constexpr std::size_t kLargeRequestCacheBytes = 2ULL << 30;

/// Fill the condition K/V caches before the first step, without the target
/// rows. Block-causal condition rows never attend to the target and use the
/// timestep-zero modulation, so this matches the joint first step; 4096-row
/// chunks bound the temporaries of multi-reference prompts (a joint first
/// step over 38k rows holds about 6 GB). Cached blocks of the growing key
/// spans are returned after every layer.
void Prefill(Runtime& rt, DiffusionContext& context, float sigma,
             const CancellationCheck& cancelled) {
  constexpr int kChunk = 4096;
  const int rows = context.prefix.rows;
  std::vector<std::pair<int, int>> chunks;
  for (int start = 0; start < rows; start += kChunk)
    chunks.emplace_back(start, std::min(kChunk, rows - start));
  // The fused attention kernel needs at least 128 query rows.
  if (chunks.size() > 1 && chunks.back().second < 128) {
    chunks[chunks.size() - 2].second += chunks.back().second;
    chunks.pop_back();
  }
  if (rows < 1 || (context.shared_rows && !context.base->prefilled))
    throw std::logic_error("invalid condition prefill");
  const float timestep = Bf16(Bf16(sigma * 1000.0F) / 1000.0F);
  std::vector<float> time(512);
  for (int row = 0; row < 2; ++row)
    for (int i = 0; i < 128; ++i) {
      const float angle = (row == 0 ? timestep * 1000 : 0) *
                          std::exp(-std::log(10000.0F) * i / 128.0F);
      time[row * 256 + i] = std::cos(angle);
      time[row * 256 + 128 + i] = std::sin(angle);
    }
  const auto temb = rt.Linear(
      rt.Activate(
          rt.Linear(rt.Upload(time, 2, 256),
                    "transformer.time_text_embed.timestep_embedder.linear_1"),
          Activation::kSilu),
      "transformer.time_text_embed.timestep_embedder.linear_2");
  const auto factors =
      rt.ModulationFactors(rt.Linear(rt.Activate(temb, Activation::kSilu),
                                     "transformer.modulation.1"),
                           true);
  auto x = rt.Concat(std::array{context.prefix});
  const int shared = context.shared_rows;
  for (int layer = 0; layer < 32; ++layer) {
    Cancel(cancelled);
    const auto p = "transformer.transformer_blocks." + std::to_string(layer);
    // Image rows attend to their whole image block, which can extend past a
    // chunk: first project every chunk, then attend.
    auto queries = rt.New(rows, 4096), keys = rt.New(rows, 4096),
         values = rt.New(rows, 4096);
    for (const auto [start, count] : chunks) {
      const auto rope = context.rope.Slice(start, count);
      // Every row is a condition row: modulation part with timestep zero.
      const auto norm =
          rt.NormalizeModulate(x.Slice(start, count), factors, 0, count);
      rt.CopyRows(queries, start,
                  rt.NormalizeRope(rt.Linear(norm, p + ".attn.to_q"),
                                   p + ".attn.norm_q", 32, rope));
      rt.CopyRows(keys, start,
                  rt.NormalizeRope(rt.Linear(norm, p + ".attn.to_k"),
                                   p + ".attn.norm_k", 32, rope));
      rt.CopyRows(values, start, rt.Linear(norm, p + ".attn.to_v"));
    }
    // The caches stay in the attention kernel's head-major layout, so later
    // steps read them in place (no per-step concatenation or reordering).
    auto head_keys = rt.HeadLayout(keys), head_values = rt.HeadLayout(values);
    keys = {};
    values = {};
    // The last layer's condition outputs are never read.
    for (const auto [start, count] :
         layer == 31 ? decltype(chunks){} : chunks) {
      const std::vector<int> limits(context.limits.begin() + start,
                                    context.limits.begin() + start + count);
      const int span = *std::max_element(limits.begin(), limits.end()) - shared;
      std::vector<Runtime::HeadKeys> segments;
      if (shared)
        segments.push_back(
            {context.base->keys[layer], context.base->values[layer], shared});
      segments.push_back({head_keys, head_values, span});
      const auto chunk = x.Slice(start, count);
      const auto attention = rt.Linear(
          rt.SegmentedAttention(queries.Slice(start, count), segments, limits),
          p + ".attn.to_out.0");
      auto out = rt.Modulate(attention, factors, 1, count, &chunk);
      const auto norm2 = rt.NormalizeModulate(out, factors, 2, count);
      const auto mlp = rt.Linear(
          rt.GatedLinear(norm2, p + ".img_mlp.gate_layer", p + ".img_mlp.proj"),
          p + ".img_mlp.out");
      out = rt.Modulate(mlp, factors, 3, count, &out);
      rt.CopyRows(x, start, out);
    }
    context.keys[layer] = std::move(head_keys);
    context.values[layer] = std::move(head_values);
    rt.Trim();
  }
  context.prefilled = true;
}

Matrix Denoise(Runtime& rt, DiffusionContext& context, const Matrix& latent,
               float sigma, int step, const CancellationCheck& cancelled,
               const Observer& observer) {
  const int prefix = step == 0 && !context.prefilled ? context.prefix.rows : 0;
  auto x = rt.Linear(latent, "transformer.img_in");
  if (prefix)
    x = Concat(rt, {context.prefix, x});
  const auto rope = prefix
                        ? context.rope
                        : context.rope.Slice(context.prefix.rows, latent.rows);
  const float timestep = Bf16(Bf16(sigma * 1000.0F) / 1000.0F);
  std::vector<float> time(512);
  for (int row = 0; row < 2; ++row)
    for (int i = 0; i < 128; ++i) {
      const float angle = (row == 0 ? timestep * 1000 : 0) *
                          std::exp(-std::log(10000.0F) * i / 128.0F);
      time[row * 256 + i] = std::cos(angle);
      time[row * 256 + 128 + i] = std::sin(angle);
    }
  auto temb = rt.Linear(
      rt.Activate(
          rt.Linear(rt.Upload(time, 2, 256),
                    "transformer.time_text_embed.timestep_embedder.linear_1"),
          Activation::kSilu),
      "transformer.time_text_embed.timestep_embedder.linear_2");
  auto activated = rt.Activate(temb, Activation::kSilu);
  const auto modulation = rt.Linear(activated, "transformer.modulation.1");
  rt.Observe(observer, "dit." + std::to_string(step) + ".input", x);
  rt.Observe(observer, "dit." + std::to_string(step) + ".modulation",
             modulation);
  const auto factors = rt.ModulationFactors(modulation, true);
  for (int layer = 0; layer < 32; ++layer) {
    Cancel(cancelled);
    const auto p = "transformer.transformer_blocks." + std::to_string(layer);
    const auto norm = rt.NormalizeModulate(x, factors, 0, prefix);
    const auto q = rt.NormalizeRope(rt.Linear(norm, p + ".attn.to_q"),
                                    p + ".attn.norm_q", 32, rope);
    auto k = rt.NormalizeRope(rt.Linear(norm, p + ".attn.to_k"),
                              p + ".attn.norm_k", 32, rope);
    auto v = rt.Linear(norm, p + ".attn.to_v");
    Matrix attended;
    if (context.prefilled) {
      // Head-major caches read in place: shared reference rows, own
      // condition rows, then this step's target rows.
      std::vector<Runtime::HeadKeys> segments;
      if (context.shared_rows)
        segments.push_back({context.base->keys[layer],
                            context.base->values[layer], context.shared_rows});
      segments.push_back(
          {context.keys[layer], context.values[layer], context.prefix.rows});
      segments.push_back({rt.HeadLayout(k), rt.HeadLayout(v), latent.rows});
      attended = rt.SegmentedAttention(q, segments);
    } else if (prefix) {
      // Copy only prefix rows: a view would pin full target K/V for all layers.
      context.keys[layer] = rt.Concat(std::array{k.Slice(0, prefix)});
      context.values[layer] = rt.Concat(std::array{v.Slice(0, prefix)});
      if (context.shared_rows) {
        k = Concat(
            rt, {context.base->keys[layer].Slice(0, context.shared_rows), k});
        v = Concat(
            rt, {context.base->values[layer].Slice(0, context.shared_rows), v});
      }
    } else if (context.shared_rows) {
      k = Concat(rt, {context.base->keys[layer].Slice(0, context.shared_rows),
                      context.keys[layer], k});
      v = Concat(rt, {context.base->values[layer].Slice(0, context.shared_rows),
                      context.values[layer], v});
    } else {
      k = Concat(rt, {context.keys[layer], k});
      v = Concat(rt, {context.values[layer], v});
    }
    if (!context.prefilled)
      attended = rt.Attention(q, k, v, 32, 32, 128,
                              prefix ? context.limits : std::vector<int>{});
    const auto attention = rt.Linear(attended, p + ".attn.to_out.0");
    x = rt.Modulate(attention, factors, 1, prefix, &x);
    const auto norm2 = rt.NormalizeModulate(x, factors, 2, prefix);
    const auto mlp = rt.Linear(
        rt.GatedLinear(norm2, p + ".img_mlp.gate_layer", p + ".img_mlp.proj"),
        p + ".img_mlp.out");
    x = rt.Modulate(mlp, factors, 3, prefix, &x);
    rt.Observe(
        observer,
        "dit." + std::to_string(step) + ".block." + std::to_string(layer), x);
  }
  const auto scale = rt.Linear(activated, "transformer.norm_out.linear");
  rt.Observe(observer, "dit." + std::to_string(step) + ".output_scale", scale);
  x = rt.NormalizeModulate(x.Slice(prefix, latent.rows),
                           rt.ModulationFactors(scale, false), 0, 0);
  rt.Observe(observer, "dit." + std::to_string(step) + ".output_norm", x);
  x = rt.Linear(x, "transformer.proj_out");
  rt.Observe(observer, "dit." + std::to_string(step) + ".output", x);
  return x;
}

}  // namespace

struct Model::Impl {
  Weights weights;
  Tokenizer tokenizer;
  Runtime runtime;
  std::mutex mutex;
  std::atomic<bool> lost{false};
  std::function<void(const std::string&)> log;
  explicit Impl(const std::filesystem::path& root)
      : weights(root), tokenizer(root), runtime(weights) {}
};

Model::Model(const std::filesystem::path& root)
    : impl_(std::make_unique<Impl>(root)) {}
Model::~Model() = default;

std::vector<std::uint32_t> Model::Tokenize(std::string_view text) const {
  return impl_->tokenizer.Encode(text);
}

Result Model::Generate(const Request& request,
                       const CancellationCheck& cancelled,
                       const Observer& observer) {
  ValidateRequest(request);
  std::unique_lock lock(impl_->mutex);
  if (DeviceLost())
    throw std::runtime_error("Qwen-Image GPU context lost; restart the server");
  Cancel(cancelled);
  const auto start = Clock::now();
  auto& rt = impl_->runtime;
  Result result;
  struct CacheLimitReset {
    Runtime& rt;
    ~CacheLimitReset() { rt.SetCacheLimit(kDefaultCacheBytes); }
  } cache_limit_reset{rt};
  const auto Log = [this](const std::string& line) {
    if (impl_->log)
      impl_->log(line);
  };
  // Memory per phase: device budget, pool cache and resident weights.
  const auto report = [&](std::string_view phase) {
    if (!impl_->log)
      return;
    rt.Synchronize();
    Log("event=memory phase=" + std::string(phase) + " " +
        hip::QueryDeviceMemory().Describe() +
        " pool_cached_mib=" + std::to_string(rt.CachedBytes() >> 20) +
        " weights_mib=" + std::to_string(rt.WeightBytes() >> 20));
  };
  // Large requests stop cleanly when the shared segment runs out anyway.
  const auto GuardMemory = [&](int step) {
    const auto memory = hip::QueryDeviceMemory();
    if (memory.Available(kHostReserveBytes) < kMinimumFreeBytes) {
      Log("event=memory_exhausted step=" + std::to_string(step) + " " +
          memory.Describe());
      throw InsufficientMemory("Qwen-Image ran out of GPU memory at step " +
                               std::to_string(step) +
                               "; use fewer or smaller reference images");
    }
  };
  try {
    const auto prompt =
        EncodePrompt(rt, impl_->tokenizer, request, cancelled, observer);
    const int height = request.height / 16, width = request.width / 16;
    // True CFG: a second prompt; its boundaries are observed under
    // "negative.".
    const bool guided = request.guidance > 1;
    const auto negative_observer = Prefixed(observer, "negative.");
    std::optional<EncodedPrompt> negative;
    if (guided) {
      Request text;
      text.prompt = request.negative_prompt;
      text.images = request.images;
      negative = EncodePrompt(rt, impl_->tokenizer, text, cancelled,
                              negative_observer);
    }
    // Every DiT condition row keeps 32 layers of K and V (512 KiB per row).
    // With several references these caches outgrow the memory beside all
    // weights, so the text encoder (16 GB) leaves until the next request.
    const int condition_rows = ConditionRows(prompt);
    result.metrics.condition_rows = condition_rows;
    report("encoded");
    const bool large = condition_rows > kReleaseEncoderRows;
    if (large) {
      rt.Evict("text_encoder.model.language_model.");
      rt.Evict("text_encoder.model.visual.");
      result.metrics.released_text_encoder = true;
      // Freed temporaries stay cached up to this bound (8 GB otherwise).
      rt.SetCacheLimit(kLargeRequestCacheBytes);
      // Refuse before the caches fill instead of failing inside a kernel:
      // an exhausted shared segment ends in a launch failure or a driver
      // reset, not in an allocation error.
      const auto need = LargeRequestBytes(
          rt, prompt, negative ? &*negative : nullptr, height * width);
      const auto memory = hip::QueryDeviceMemory();
      const auto available = memory.Available(kHostReserveBytes);
      Log("event=memory_check need_mib=" + std::to_string(need >> 20) +
          " available_mib=" + std::to_string(available >> 20) + " " +
          memory.Describe());
      if (need > available)
        throw InsufficientMemory(
            "Qwen-Image needs about " + std::to_string(need >> 30) +
            " GiB of GPU memory for " + std::to_string(condition_rows) +
            " condition rows, but only " + std::to_string(available >> 30) +
            " GiB are free; use fewer or smaller reference images");
      report("released");
    }
    std::optional<DiffusionContext> context;
    context.emplace(PrepareDiffusion(rt, impl_->weights, prompt, height, width,
                                     cancelled, observer));
    std::optional<DiffusionContext> negative_context;
    if (guided) {
      // The references before the prompt text share their K/V with the
      // positive pass; only the negative text rows get their own cache.
      negative_context.emplace(
          PrepareDiffusion(rt, impl_->weights, *negative, height, width,
                           cancelled, negative_observer, &*context,
                           SharedConditionTokens(prompt, *negative)));
      result.metrics.negative_condition_rows = negative_context->prefix.rows;
      negative.reset();
    }
    const bool img2img =
        request.init_image && request.strength <= kFullStrength;
    std::optional<Image> init_image;
    Matrix init_latent;
    if (img2img) {
      init_image =
          ResizeImage(*request.init_image, request.width, request.height);
      init_latent = EncodeVae(rt, impl_->weights, *init_image, cancelled,
                              Prefixed(observer, "init."));
    }
    // Drop cached encoder and VAE-encoder temporaries before the caches fill.
    rt.Trim();
    result.metrics.prompt_ms = Milliseconds(start);
    report("prepared");
    // Generate independent full-precision normal draws; seed replay never
    // depends on admission order, wall time, or another request's RNG.
    // A generated reference can otherwise reuse exactly the same noise when
    // edited at the same size and seed. Bind edit noise to its reference
    // pixels, including later edits of an edit, without changing generation or
    // replay.
    std::mt19937_64 rng(InitialNoiseSeed(request.seed, prompt.images,
                                         init_image ? &*init_image : nullptr));
    std::normal_distribution<float> normal(0, 1);
    std::vector<float> noise(static_cast<std::size_t>(height) * width * 64);
    for (float& value : noise)
      value = normal(rng);
    auto latent = rt.Upload(noise, height * width, 64);
    rt.Observe(observer, "noise", latent);
    const auto denoise_start = Clock::now();
    const auto sigmas =
        DenoiseSigmas(request.steps, request.strength, height * width);
    if (observer.Wants("sigmas"))
      observer.write("sigmas", sigmas, 1, static_cast<int>(sigmas.size()));
    if (img2img) {
      latent = rt.NoiseMix(latent, init_latent, sigmas[0]);
      rt.Observe(observer, "init.latent", latent);
    }
    if (large) {
      // Condition caches in head-major layout, filled chunk by chunk.
      Prefill(rt, *context, sigmas[0], cancelled);
      if (negative_context)
        Prefill(rt, *negative_context, sigmas[0], cancelled);
      report("prefilled");
    }
    for (int step = 0; step < request.steps; ++step) {
      if (large)
        GuardMemory(step);
      auto velocity = Denoise(rt, *context, latent, sigmas[step], step,
                              cancelled, observer);
      if (guided) {
        const auto negative =
            Denoise(rt, *negative_context, latent, sigmas[step], step,
                    cancelled, negative_observer);
        velocity = rt.Guidance(velocity, negative, request.guidance);
        rt.Observe(observer, "dit." + std::to_string(step) + ".guided",
                   velocity);
      }
      latent = rt.Euler(latent, velocity, sigmas[step + 1] - sigmas[step]);
      rt.Synchronize();
      rt.Observe(observer, "latent." + std::to_string(step), latent);
      if (step == 0)
        report("step0");
    }
    rt.Synchronize();
    result.metrics.denoise_ms = Milliseconds(denoise_start);
    // The condition caches and DiT temporaries are done; give the VAE
    // decoder their memory.
    negative_context.reset();
    context.reset();
    rt.Trim();
    report("denoised");
    const auto vae_start = Clock::now();
    result.image = DecodeVae(rt, impl_->weights, latent, height, width,
                             cancelled, observer);
    result.metrics.vae_ms = Milliseconds(vae_start);
    result.metrics.total_ms = Milliseconds(start);
    result.metrics.resident_weight_bytes = rt.WeightBytes();
    return result;
  } catch (const std::exception& error) {
    // All launches on this request's stream must finish before temporary
    // buffers return to the pool and the next request starts.
    const auto status = hipStreamSynchronize(rt.stream());
    // A failed HIP or hipBLAS(Lt) call leaves this process unable to run
    // later requests (they fail with "unspecified launch failure").
    const std::string_view what = error.what();
    if (status != hipSuccess || what.starts_with("Qwen-Image HIP") ||
        what.starts_with("Qwen-Image hipBLAS"))
      impl_->lost.store(true, std::memory_order_release);
    throw;
  } catch (...) {
    (void)hipStreamSynchronize(rt.stream());
    throw;
  }
}

void Model::SetLog(std::function<void(const std::string&)> log) {
  impl_->log = std::move(log);
}

bool Model::DeviceLost() const {
  return impl_->lost.load(std::memory_order_acquire);
}

}  // namespace gufo::models::qwen_image_21
