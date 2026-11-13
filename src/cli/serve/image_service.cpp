#include "src/cli/serve/image_service.hpp"

#include <png.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <variant>

#include "gufo/ext/registry.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/core/image.hpp"
#include "src/ext/image.hpp"

namespace gufo::server {
namespace {

template<typename T>
std::string Shortest(T value) {
  char text[32];
  return {text, std::to_chars(text, text + sizeof(text), value).ptr};
}

void CheckImage(const ext::ImageBuffer& image) {
  if (image.width <= 0 || image.height <= 0 ||
      static_cast<std::uint64_t>(image.width) * image.height >
          core::kMaxImagePixels ||
      image.rgba.size() !=
          static_cast<std::size_t>(image.width) * image.height * 4)
    throw std::invalid_argument("invalid RGBA image dimensions");
}

struct PngOutput {
  std::vector<std::uint8_t> bytes;
  std::size_t used{0};
};

void WritePng(png_structp png, png_bytep data, png_size_t size) {
  auto& output = *static_cast<PngOutput*>(png_get_io_ptr(png));
  if (size > output.bytes.size() - output.used)
    png_error(png, "output PNG exceeds its allocation");
  std::memcpy(output.bytes.data() + output.used, data, size);
  output.used += size;
}

std::string LoraList(const std::vector<ext::ImageLora>& loras) {
  std::ostringstream list;
  for (const auto& lora : loras)
    if (lora.scale != 0)
      list << (list.tellp() > 0 ? "," : "") << lora.name << ':' << lora.scale;
  return list.tellp() > 0 ? list.str() : "none";
}

}  // namespace

ext::ImageBuffer DecodeImage(std::span<const std::uint8_t> bytes) {
  if (bytes.size() > core::kMaxEncodedImageBytes || bytes.size() < 8)
    throw std::invalid_argument("invalid or oversized input image");
  if (png_sig_cmp(bytes.data(), 0, 8) != 0) {
    const auto rgb = core::DecodeImage(bytes);
    ext::ImageBuffer image{
        static_cast<int>(rgb.width), static_cast<int>(rgb.height), {}};
    image.rgba.resize(static_cast<std::size_t>(rgb.width) * rgb.height * 4);
    for (std::size_t i = 0; i < image.rgba.size() / 4; ++i) {
      std::memcpy(image.rgba.data() + 4 * i, rgb.pixels.data() + 3 * i, 3);
      image.rgba[4 * i + 3] = 255;
    }
    return image;
  }
  png_image png{};
  png.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_memory(&png, bytes.data(), bytes.size()))
    throw std::invalid_argument("cannot decode PNG image");
  const auto cleanup = std::unique_ptr<png_image, decltype(&png_image_free)>(
      &png, png_image_free);
  if (!png.width || !png.height ||
      static_cast<std::uint64_t>(png.width) * png.height >
          core::kMaxImagePixels)
    throw std::invalid_argument("PNG image dimensions exceed limits");
  png.format = PNG_FORMAT_RGBA;
  ext::ImageBuffer image{
      static_cast<int>(png.width), static_cast<int>(png.height), {}};
  image.rgba.resize(PNG_IMAGE_SIZE(png));
  if (!png_image_finish_read(&png, nullptr, image.rgba.data(), 0, nullptr))
    throw std::invalid_argument("invalid PNG image data");
  return image;
}

std::vector<std::uint8_t> EncodePng(const ext::ImageBuffer& image) {
  CheckImage(image);
  png_image geometry{};
  geometry.width = image.width;
  geometry.height = image.height;
  geometry.format = PNG_FORMAT_RGBA;
  const auto output = std::make_unique<PngOutput>();
  output->bytes.resize(PNG_IMAGE_PNG_SIZE_MAX(geometry));
  auto png = png_create_write_struct(
      PNG_LIBPNG_VER_STRING, nullptr,
      [](png_structp p, png_const_charp) { png_longjmp(p, 1); }, nullptr);
  if (!png)
    throw std::runtime_error("cannot allocate PNG encoder");
  auto info = png_create_info_struct(png);
  if (!info) {
    png_destroy_write_struct(&png, nullptr);
    throw std::runtime_error("cannot allocate PNG metadata");
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    throw std::runtime_error("cannot encode output PNG");
  }
  png_set_write_fn(png, output.get(), WritePng, nullptr);
  png_set_IHDR(png, info, image.width, image.height, 8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_set_sRGB(png, info, PNG_sRGB_INTENT_PERCEPTUAL);
  png_set_compression_level(png, 1);
  png_set_filter(png, PNG_FILTER_TYPE_BASE, PNG_ALL_FILTERS);
  png_write_info(png, info);
  for (int row = 0; row < image.height; ++row)
    png_write_row(png, const_cast<png_bytep>(image.rgba.data() +
                                             static_cast<std::size_t>(row) *
                                                 image.width * 4));
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  output->bytes.resize(output->used);
  return std::move(output->bytes);
}

void ValidateImageRequest(const ext::ImageRequest& request) {
  if (request.prompt.size() > (64U << 10U))
    throw std::invalid_argument("image prompt exceeds 64 KiB");
  if (request.width < 32 || request.height < 32 || request.width % 32 ||
      request.height % 32 || request.width > 4096 || request.height > 4096 ||
      static_cast<std::uint64_t>(request.width) * request.height > 4194304)
    throw std::invalid_argument(
        "image size must be multiples of 32, at most 4096 per side and 4 "
        "megapixels");
  if (request.steps != 0 && (request.steps < 2 || request.steps > 100))
    throw std::invalid_argument("image steps must be between 2 and 100");
  if (request.images.size() > 10)
    throw std::invalid_argument(
        "image edits accept at most 10 reference images");
  std::size_t pixels = 0;
  for (const auto& image : request.images) {
    CheckImage(image);
    pixels += static_cast<std::size_t>(image.width) * image.height;
  }
  if (pixels > core::kMaxImagePixels)
    throw std::invalid_argument(
        "reference images exceed 32 megapixels combined");
  if (request.negative_prompt.size() > (64U << 10U))
    throw std::invalid_argument("negative_prompt exceeds 64 KiB");
  if (request.guidance_scale &&
      !(*request.guidance_scale >= 0 && *request.guidance_scale <= 100))
    throw std::invalid_argument("guidance_scale must be from 0 to 100");
  if (!(request.strength >= 0.01 && request.strength <= 1))
    throw std::invalid_argument("strength must be from 0.01 to 1");
  if (request.strength < 1 && !request.init_image)
    throw std::invalid_argument("strength below 1 requires an input image");
  if (request.init_image)
    CheckImage(*request.init_image);
  if (request.loras.size() > 16)
    throw std::invalid_argument("image requests apply at most 16 LoRAs");
  for (std::size_t i = 0; i < request.loras.size(); ++i) {
    const auto& lora = request.loras[i];
    if (lora.name.empty())
      throw std::invalid_argument("LoRA name must not be empty");
    for (std::size_t j = 0; j < i; ++j)
      if (request.loras[j].name == lora.name)
        throw std::invalid_argument("duplicate LoRA " + lora.name);
    if (!std::isfinite(lora.scale) || std::abs(lora.scale) > 10)
      throw std::invalid_argument(
          "LoRA scale must be finite and at most 10 in magnitude");
  }
}

const ext::ImageProvider& SelectImageProvider(
    const ext::Registry& registry, const std::filesystem::path& root,
    const ext::ParsedOptions& options) {
  const auto* provider = ext::ProbeImageProvider(root.string());
  if (!provider) {
    provider = ext::FindImageProvider("qwen-image-21");
  }
  if (!provider) {
    provider = ext::FindImageProvider("qwen-image-2.1");
  }
  if (!provider) {
    throw std::runtime_error("extension 'qwen_image_21' is not registered");
  }
  if (options.empty())
    return *provider;
  // Explicit options must belong to the provider that claims the model; the
  // parser accepted the union of every available provider's declarations.
  const auto& info = registry.Provider(provider->id);
  ext::ValidateSelectedOptions(registry, ext::kImageServeCommand, info,
                               options);
  for (const auto& [name, value] : options) {
    const auto spec =
        std::ranges::find_if(info.options, [&](const ext::OptionSpec& option) {
          return option.name == name &&
                 ext::OptionApplies(option, ext::kImageServeCommand);
        });
    if (spec == info.options.end() || spec->type != ext::OptionType::kPath)
      continue;
    const auto& path = std::get<std::string>(value);
    std::error_code error;
    if (path.empty() || !std::filesystem::exists(path, error))
      throw std::invalid_argument(name + " does not exist: " + path);
  }
  return *provider;
}

std::shared_ptr<ext::ImageRunner> CreateImageRunner(
    const ext::ImageProvider& provider, const std::filesystem::path& root,
    const ext::ParsedOptions& options) {
  std::string error;
  std::unique_ptr<ext::ImageRunner> runner;
  if (provider.create_with_options) {
    runner = provider.create_with_options(
        {.model_root = root.string(), .options = options}, &error);
  } else if (!options.empty()) {
    throw std::runtime_error("image provider " + provider.id +
                             " declares startup options but has no "
                             "options-aware factory");
  } else if (provider.create) {
    runner = provider.create(root.string(), &error);
  } else {
    throw std::runtime_error("image provider " + provider.id +
                             " cannot run in this build");
  }
  if (!runner) {
    throw std::runtime_error("failed to create runner for " + root.string() +
                             (error.empty() ? "" : ": " + error));
  }
  return runner;
}

ImageService::ImageService(std::shared_ptr<ext::ImageRunner> runner,
                           std::string model_id,
                           std::vector<std::string> model_aliases,
                           ImageLoraOptions loras)
    : model_id_(std::move(model_id)),
      model_aliases_(std::move(model_aliases)),
      loras_(std::move(loras)),
      model_(std::move(runner)) {
  if (model_id_.empty())
    throw std::invalid_argument("image model ID must not be empty");
  if (!model_)
    throw std::invalid_argument("image service requires a runner");
  capabilities_ = model_->Capabilities();
  ValidateLoraOptions(loras_);
  for (const auto& [name, path] : loras_.files) {
    const auto& info = lora_info_[name] = model_->AddLora(name, path.string());
    Logger::Info("image", "event=lora_registered name=" + name +
                              " file=" + path.filename().string() +
                              " format=" + info.format +
                              " targets=" + std::to_string(info.targets) +
                              " rank=" + std::to_string(info.min_rank) + '-' +
                              std::to_string(info.max_rank));
  }
}

ImageService::~ImageService() = default;

void ValidateLoraOptions(const ImageLoraOptions& loras) {
  std::set<std::string_view> names;
  for (const auto& [name, path] : loras.files) {
    (void)path;
    if (name.empty() || name.size() > 64 ||
        name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrs"
                               "tuvwxyz0123456789._-") != std::string::npos)
      throw std::invalid_argument("LoRA name '" + name +
                                  "' must be 1-64 of [A-Za-z0-9._-]");
    if (!names.insert(name).second)
      throw std::invalid_argument("duplicate --lora name '" + name + "'");
  }
  for (const auto& lora : loras.defaults)
    if (!names.contains(lora.name))
      throw std::invalid_argument("--lora-default names unknown adapter '" +
                                  lora.name + "'");
  ext::ImageRequest defaults;
  defaults.loras = loras.defaults;
  ValidateImageRequest(defaults);
}

std::vector<ImageLoraListing> ImageService::LoraListings() const {
  std::map<std::string, int> shared;
  for (const auto& [name, path] : loras_.files)
    ++shared[path.filename().string()];
  std::vector<ImageLoraListing> listings;
  for (const auto& [name, path] : loras_.files) {
    const auto file = path.filename().string();
    if (shared[file] == 1)
      listings.push_back({file, file});
    else
      listings.push_back({name + '/' + file, path.generic_string()});
  }
  return listings;
}

void ImageService::Validate(const Request& request) const {
  if (!request.images.empty() && !capabilities().edits)
    throw std::invalid_argument("model " + model_id_ +
                                " does not support image edits");
  if ((!request.negative_prompt.empty() ||
       request.guidance_scale.value_or(1) > 1) &&
      !capabilities().guidance)
    throw std::invalid_argument(
        "model " + model_id_ +
        " does not support negative_prompt or guidance_scale");
  if (request.strength < 1 && !capabilities().image_to_image)
    throw std::invalid_argument("model " + model_id_ +
                                " does not support image to image (strength)");
  if (request.background == ext::ImageBackground::kTransparent &&
      !capabilities().transparent_background)
    throw std::invalid_argument(
        "model " + model_id_ +
        " does not support transparent backgrounds; use background auto or "
        "opaque");
  model_->ValidateRequest(request);
}

bool ImageService::HasLora(std::string_view name) const {
  return lora_info_.contains(name);
}

const ext::ImageLoraInfo& ImageService::lora_info(std::string_view name) const {
  return lora_info_.find(name)->second;
}

std::vector<ext::ImageLora> ImageService::ActiveLoras() const {
  return model_->ActiveLoras();
}

ImageService::Result ImageService::Generate(
    const Request& request, const CancellationCheck& cancelled) {
  pending_.fetch_add(1, std::memory_order_relaxed);
  const std::unique_ptr<std::atomic<int>, void (*)(std::atomic<int>*)> done(
      &pending_, [](std::atomic<int>* pending) {
        pending->fetch_sub(1, std::memory_order_relaxed);
      });
  std::unique_lock<std::timed_mutex> lock(gate_, std::defer_lock);
  while (!lock.try_lock_for(std::chrono::milliseconds(20)))
    if (cancelled && cancelled())
      throw std::runtime_error("image request cancelled");
  if (cancelled && cancelled())
    throw std::runtime_error("image request cancelled");
  if (!capabilities_ &&
      request.background == ext::ImageBackground::kTransparent) {
    auto legacy = request;
    legacy.prompt =
        "This is an RGBA image with transparency. " + request.prompt +
        " The image has alpha channel and the background is transparent.";
    return model_->Generate(legacy, cancelled);
  }
  return model_->Generate(request, cancelled);
}

std::uint64_t RandomImageSeed() {
  std::random_device entropy;
  return ((static_cast<std::uint64_t>(entropy()) << 21) ^ entropy()) &
         kMaxImageSeed;
}

std::vector<GeneratedImage> GenerateImages(
    ImageService& service, ext::ImageRequest request, std::size_t count,
    std::string_view request_id, std::string_view mode,
    const ext::CancellationCheck& cancelled) {
  std::vector<GeneratedImage> images;
  images.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto result = service.Generate(request, cancelled);
    if (request.background == ext::ImageBackground::kOpaque) {
      for (std::size_t p = 0; p < result.image.rgba.size(); p += 4) {
        const int a = result.image.rgba[p + 3];
        for (int c = 0; c < 3; ++c)
          result.image.rgba[p + c] = static_cast<std::uint8_t>(
              (result.image.rgba[p + c] * a + 255 * (255 - a) + 127) / 255);
        result.image.rgba[p + 3] = 255;
      }
    }
    images.push_back({.png = EncodePng(result.image),
                      .seed = request.seed,
                      .metrics = result.metrics});
    std::ostringstream details;
    details << "request=" << request_id << " event=generated mode=" << mode
            << " seed=" << request.seed << " size=" << request.width << 'x'
            << request.height << " references=" << request.images.size()
            << " steps=" << result.metrics.steps;
    if (request.init_image)
      details << " init_image=" << request.init_image->width << 'x'
              << request.init_image->height << " strength=" << request.strength;
    if (const auto scale = request.guidance_scale
                               ? request.guidance_scale
                               : result.metrics.guidance_scale)
      details << " guidance_scale=" << Shortest(*scale);
    if (!service.loras().files.empty())
      details << " loras=" << LoraList(request.loras)
              << " lora_ms=" << result.metrics.lora_ms;
    details << " prompt_ms=" << result.metrics.prompt_ms
            << " denoise_ms=" << result.metrics.denoise_ms
            << " vae_ms=" << result.metrics.vae_ms
            << " total_ms=" << result.metrics.total_ms
            << " resident_weight_mib="
            << result.metrics.resident_weight_bytes / (1024 * 1024);
    Logger::Info("image", details.str());
    request.seed = (request.seed + 1) & kMaxImageSeed;
  }
  return images;
}

}  // namespace gufo::server
