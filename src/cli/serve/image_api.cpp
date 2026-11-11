#include "src/cli/serve/image_api.hpp"

#include <png.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <variant>

#include "gufo/ext/registry.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/core/base64.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/ext/image.hpp"

namespace gufo::server {
namespace {

json::Value ParseJson(std::string_view text) {
  try {
    return json::parse(text);
  } catch (const std::exception& error) {
    throw std::invalid_argument(error.what());
  }
}

HttpResponse Error(int status, const std::string& message,
                   const std::string& code) {
  auto body = json::Value::object();
  auto error = json::Value::object();
  error["message"] = message;
  error["type"] = status < 500 ? "invalid_request_error" : "server_error";
  error["code"] = code;
  error["param"] = nullptr;
  body["error"] = std::move(error);
  return {.status = status,
          .reason = status == 400   ? "Bad Request"
                    : status == 404 ? "Not Found"
                                    : "Internal Server Error",
          .body = body.dump(),
          .headers = {{"Content-Type", "application/json"}}};
}

std::string_view Trim(std::string_view value) {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
    value.remove_prefix(1);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t'))
    value.remove_suffix(1);
  return value;
}

std::string Parameter(std::string_view value, std::string_view key) {
  // RFC 7578 quoted-string parameters. Filenames are deliberately ignored.
  while (!value.empty()) {
    const auto separator = value.find(';');
    if (separator == std::string_view::npos)
      break;
    value.remove_prefix(separator + 1);
    value = Trim(value);
    const auto equal = value.find('=');
    if (equal == std::string_view::npos)
      break;
    const auto name = Trim(value.substr(0, equal));
    value.remove_prefix(equal + 1);
    std::string result;
    if (!value.empty() && value.front() == '"') {
      value.remove_prefix(1);
      bool closed = false;
      while (!value.empty()) {
        char c = value.front();
        value.remove_prefix(1);
        if (c == '"') {
          closed = true;
          break;
        }
        if (c == '\\') {
          if (value.empty())
            throw std::invalid_argument("truncated multipart escape");
          c = value.front();
          value.remove_prefix(1);
        }
        if (static_cast<unsigned char>(c) < 32)
          throw std::invalid_argument(
              "control character in multipart parameter");
        result += c;
      }
      if (!closed)
        throw std::invalid_argument("unclosed multipart parameter");
    } else {
      const auto end = value.find(';');
      result = Trim(value.substr(0, end));
      value = end == std::string_view::npos ? std::string_view{}
                                            : value.substr(end);
    }
    if (name == key)
      return result;
  }
  return {};
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

void ValidateRequest(const ext::ImageRequest& request) {
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

struct Parsed {
  json::Value fields = json::Value::object();
  std::vector<ext::ImageBuffer> images;
};

Parsed Multipart(const HttpRequest& request) {
  const auto content_type = request.header("content-type");
  if (!content_type.starts_with("multipart/form-data"))
    throw std::invalid_argument("image edits require multipart/form-data");
  const auto boundary = Parameter(content_type, "boundary");
  if (boundary.empty() || boundary.size() > 70 ||
      boundary.find_first_of("\r\n") != std::string::npos)
    throw std::invalid_argument("invalid multipart boundary");
  const std::string marker = "--" + boundary;
  std::string_view body = request.body;
  if (!body.starts_with(marker))
    throw std::invalid_argument("multipart body does not start with boundary");
  body.remove_prefix(marker.size());
  Parsed parsed;
  std::size_t image_bytes = 0, image_pixels = 0, fields = 0;
  for (;;) {
    if (body.starts_with("--")) {
      body.remove_prefix(2);
      if (body != "" && body != "\r\n")
        throw std::invalid_argument("data after final multipart boundary");
      break;
    }
    if (!body.starts_with("\r\n"))
      throw std::invalid_argument("invalid multipart separator");
    body.remove_prefix(2);
    const auto header_end = body.find("\r\n\r\n");
    if (header_end == std::string_view::npos || header_end > 16384)
      throw std::invalid_argument("invalid multipart headers");
    auto headers = body.substr(0, header_end);
    std::string name;
    bool disposition_seen = false;
    while (!headers.empty()) {
      const auto end = headers.find("\r\n");
      const auto line = headers.substr(0, end);
      const auto colon = line.find(':');
      if (colon == std::string_view::npos)
        throw std::invalid_argument("malformed multipart header");
      std::string key(line.substr(0, colon));
      std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (key == "content-disposition") {
        if (disposition_seen)
          throw std::invalid_argument("duplicate multipart disposition");
        disposition_seen = true;
        const auto value = Trim(line.substr(colon + 1));
        if (!value.starts_with("form-data"))
          throw std::invalid_argument("expected form-data disposition");
        name = Parameter(value, "name");
      }
      if (end == std::string_view::npos)
        break;
      headers.remove_prefix(end + 2);
    }
    if (name.empty())
      throw std::invalid_argument("multipart field has no name");
    body.remove_prefix(header_end + 4);
    const std::string delimiter = "\r\n" + marker;
    std::size_t end = 0;
    for (;;) {
      end = body.find(delimiter, end);
      if (end == std::string_view::npos)
        throw std::invalid_argument("missing final multipart boundary");
      const auto suffix = body.substr(end + delimiter.size());
      if (suffix.starts_with("\r\n") || suffix.starts_with("--"))
        break;
      ++end;
    }
    const auto data = body.substr(0, end);
    body.remove_prefix(end + delimiter.size());
    if (++fields > 32)
      throw std::invalid_argument("too many multipart fields");
    if (name == "image" || name == "image[]") {
      if (parsed.images.size() == 10 ||
          data.size() > core::kMaxEncodedImageBytes - image_bytes)
        throw std::invalid_argument("image inputs exceed 10 images or 20 MiB");
      image_bytes += data.size();
      parsed.images.push_back(DecodeImage(std::span(
          reinterpret_cast<const std::uint8_t*>(data.data()), data.size())));
      const auto& decoded = parsed.images.back();
      image_pixels += static_cast<std::size_t>(decoded.width) * decoded.height;
      if (image_pixels > core::kMaxImagePixels)
        throw std::invalid_argument(
            "reference images exceed 32 megapixels combined");
    } else {
      if (name == "mask")
        throw std::invalid_argument(
            "mask uploads are unsupported; provide annotated reference images");
      if (parsed.fields.contains(name))
        throw std::invalid_argument("duplicate field " + name);
      if (data.size() > (64U << 10U))
        throw std::invalid_argument("multipart field too large");
      if (name == "n" || name == "seed" || name == "steps" ||
          name == "stream" || name == "loras") {
        parsed.fields[name] = ParseJson(data);
      } else {
        parsed.fields[name] = std::string(data);
      }
    }
  }
  if (parsed.images.empty())
    throw std::invalid_argument("image edit requires an image");
  return parsed;
}

std::uint64_t Number(const json::Value& fields, const char* name,
                     std::uint64_t fallback, std::uint64_t low,
                     std::uint64_t high) {
  const auto* value = fields.find(name);
  if (!value)
    return fallback;
  if (!value->is_number() || !std::isfinite(value->as_double()) ||
      value->as_double() < static_cast<double>(low) ||
      value->as_double() > static_cast<double>(high) ||
      std::floor(value->as_double()) != value->as_double())
    throw std::invalid_argument(std::string("invalid ") + name);
  return static_cast<std::uint64_t>(value->as_double());
}

std::string String(const json::Value& fields, const char* name,
                   const char* fallback) {
  const auto* value = fields.find(name);
  if (!value)
    return fallback;
  if (!value->is_string())
    throw std::invalid_argument(std::string(name) + " must be a string");
  return value->str();
}

std::vector<ext::ImageLora> ParseLoras(const json::Value& fields,
                                       const ImageService& service) {
  const auto* value = fields.find("loras");
  if (!value)
    return service.loras().defaults;
  if (!value->is_array())
    throw std::invalid_argument("loras must be an array");
  std::vector<ext::ImageLora> loras;
  for (const auto& entry : value->items()) {
    if (!entry.is_object())
      throw std::invalid_argument("loras entries must be objects");
    for (const auto& [key, member] : entry.members()) {
      (void)member;
      if (key != "name" && key != "scale")
        throw std::invalid_argument("unsupported LoRA parameter: " + key);
    }
    const auto* name = entry.find("name");
    if (!name || !name->is_string())
      throw std::invalid_argument("LoRA name must be a string");
    if (!service.HasLora(name->str()))
      throw std::invalid_argument("unknown LoRA adapter: " + name->str());
    auto& lora = loras.emplace_back(ext::ImageLora{name->str()});
    if (const auto* scale = entry.find("scale")) {
      if (!scale->is_number())
        throw std::invalid_argument("LoRA scale must be a number");
      // Converting an out-of-range double to float is undefined.
      const double value = scale->as_double();
      if (!std::isfinite(value) || std::abs(value) > 10)
        throw std::invalid_argument(
            "LoRA scale must be finite and at most 10 in magnitude");
      lora.scale = static_cast<float>(value);
    }
  }
  return loras;
}

std::string LoraList(const std::vector<ext::ImageLora>& loras) {
  std::ostringstream list;
  for (const auto& lora : loras)
    if (lora.scale != 0)
      list << (list.tellp() > 0 ? "," : "") << lora.name << ':' << lora.scale;
  return list.tellp() > 0 ? list.str() : "none";
}

}  // namespace

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
  ValidateRequest(defaults);
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

HttpResponse HandleImageApiRequest(const HttpRequest& request,
                                   ImageService& service) {
  if (request.method != "POST")
    return Error(400, "image endpoints require POST", "invalid_method");
  try {
    Parsed parsed;
    const bool edit = request.path == "/v1/images/edits";
    // Reject before decoding uploads when the model cannot edit at all.
    if (edit && !service.capabilities().edits)
      throw std::invalid_argument("model " + service.model_id() +
                                  " does not support image edits");
    if (edit)
      parsed = Multipart(request);
    else {
      if (!request.header("content-type").starts_with("application/json"))
        throw std::invalid_argument(
            "image generations require application/json");
      parsed.fields = ParseJson(request.body);
    }
    const auto& fields = parsed.fields;
    if (!fields.is_object())
      throw std::invalid_argument("request must be a JSON object");
    // As in the other OpenAI-compatible servers, unknown members are ignored.
    if (!fields.contains("prompt"))
      throw std::invalid_argument("prompt is required");
    const auto model = String(fields, "model", service.model_id().c_str());
    if (!service.serves_model(model))
      return Error(404, "requested image model is not loaded",
                   "model_not_found");
    if (String(fields, "response_format", "b64_json") != "b64_json" ||
        String(fields, "output_format", "png") != "png")
      throw std::invalid_argument(
          "image output supports PNG with response_format b64_json");
    const auto quality = String(fields, "quality", "auto");
    if (quality != "auto" && quality != "standard" && quality != "high")
      throw std::invalid_argument(
          "quality must be auto, standard or high; steps are "
          "checkpoint-specific");
    if (const auto* stream = fields.find("stream");
        stream && (!stream->is_bool() || stream->as_bool()))
      throw std::invalid_argument("streaming image responses are unsupported");
    (void)String(fields, "user", "");
    const auto background = String(fields, "background", "auto");
    ext::ImageRequest generation;
    if (background == "auto")
      generation.background = ext::ImageBackground::kAuto;
    else if (background == "opaque")
      generation.background = ext::ImageBackground::kOpaque;
    else if (background == "transparent")
      generation.background = ext::ImageBackground::kTransparent;
    else
      throw std::invalid_argument(
          "background must be auto, opaque or transparent");
    // The prompt reaches the provider unchanged; model-specific conditioning
    // for the requested background belongs to the provider.
    generation.prompt = String(fields, "prompt", "");
    generation.images = std::move(parsed.images);
    const auto size = String(fields, "size", "auto");
    if (size == "auto") {
      if (!generation.images.empty()) {
        const auto& last = generation.images.back();
        const double ratio = static_cast<double>(last.width) / last.height;
        const double w = std::sqrt(1024.0 * 1024 * ratio);
        generation.width = static_cast<int>(std::nearbyint(w / 32)) * 32;
        generation.height =
            static_cast<int>(std::nearbyint(w / ratio / 32)) * 32;
      }
    } else {
      const auto separator = size.find('x');
      if (separator == std::string::npos)
        throw std::invalid_argument("size must be WIDTHxHEIGHT or auto");
      const auto parse = [](std::string_view s) {
        int n = 0;
        auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), n);
        if (ec != std::errc{} || end != s.data() + s.size())
          throw std::invalid_argument("invalid image size");
        return n;
      };
      generation.width = parse(std::string_view(size).substr(0, separator));
      generation.height = parse(std::string_view(size).substr(separator + 1));
    }
    // Omitted steps keep zero so the runner applies its checkpoint default.
    generation.steps = static_cast<int>(Number(fields, "steps", 0, 2, 100));
    std::random_device entropy;
    generation.seed =
        Number(fields, "seed",
               (static_cast<std::uint64_t>(entropy()) << 21) ^ entropy(), 0,
               (1ULL << 53) - 1);
    generation.loras = ParseLoras(fields, service);
    const auto count = Number(fields, "n", 1, 1, 10);
    ValidateRequest(generation);
    service.Validate(generation);
    auto body = json::Value::object();
    body["created"] = static_cast<double>(std::time(nullptr));
    auto data = json::Value::array();
    for (std::uint64_t i = 0; i < count; ++i) {
      auto result = service.Generate(generation, request.is_cancelled);
      if (generation.background == ext::ImageBackground::kOpaque) {
        for (std::size_t p = 0; p < result.image.rgba.size(); p += 4) {
          const int a = result.image.rgba[p + 3];
          for (int c = 0; c < 3; ++c)
            result.image.rgba[p + c] = static_cast<std::uint8_t>(
                (result.image.rgba[p + c] * a + 255 * (255 - a) + 127) / 255);
          result.image.rgba[p + 3] = 255;
        }
      }
      auto item = json::Value::object();
      item["b64_json"] = core::EncodeBase64(EncodePng(result.image));
      data.push_back(std::move(item));
      std::ostringstream details;
      details << "request=" << request.request_id
              << " event=generated mode=" << (edit ? "edit" : "generate")
              << " seed=" << generation.seed << " size=" << generation.width
              << 'x' << generation.height
              << " references=" << generation.images.size()
              << " steps=" << result.metrics.steps;
      if (!service.loras().files.empty())
        details << " loras=" << LoraList(generation.loras)
                << " lora_ms=" << result.metrics.lora_ms;
      details << " prompt_ms=" << result.metrics.prompt_ms
              << " denoise_ms=" << result.metrics.denoise_ms
              << " vae_ms=" << result.metrics.vae_ms
              << " total_ms=" << result.metrics.total_ms
              << " resident_weight_mib="
              << result.metrics.resident_weight_bytes / (1024 * 1024);
      Logger::Info("image", details.str());
      generation.seed = (generation.seed + 1) & ((1ULL << 53) - 1);
    }
    body["data"] = std::move(data);
    return {.status = 200,
            .reason = "OK",
            .body = body.dump(),
            .headers = {{"Content-Type", "application/json"}}};
  } catch (const std::invalid_argument& error) {
    return Error(400, error.what(), "invalid_request");
  } catch (const std::exception& error) {
    return Error(500, error.what(), "image_generation_failed");
  }
}

HttpResponse HandleImageLorasRequest(const HttpRequest& request,
                                     const ImageService& service) {
  if (request.method != "GET")
    return Error(400, "/sdapi/v1/loras requires GET", "invalid_method");
  const auto active = service.ActiveLoras();
  const auto scale = [](const std::vector<ext::ImageLora>& loras,
                        const std::string& name) -> json::Value {
    for (const auto& lora : loras)
      if (lora.name == name) {
        // Shortest float text, so 0.8 is listed as 0.8.
        char text[32];
        const auto end =
            std::to_chars(text, text + sizeof(text), lora.scale).ptr;
        double value = lora.scale;
        std::from_chars(text, end, value);
        return value;
      }
    return nullptr;
  };
  auto body = json::Value::array();
  const auto listings = service.LoraListings();
  for (std::size_t i = 0; i < listings.size(); ++i) {
    const auto& name = service.loras().files[i].first;
    const auto& info = service.lora_info(name);
    auto metadata = json::Value::object();
    metadata["format"] = info.format;
    metadata["targets"] = info.targets;
    metadata["rank"] =
        std::to_string(info.min_rank) + '-' + std::to_string(info.max_rank);
    auto item = json::Value::object();
    item["name"] = name;
    item["alias"] = name;
    item["path"] = listings[i].path;
    item["display_path"] = listings[i].display_path;
    item["metadata"] = std::move(metadata);
    item["default_scale"] = scale(service.loras().defaults, name);
    item["active_scale"] = scale(active, name);
    body.push_back(std::move(item));
  }
  return {.status = 200,
          .reason = "OK",
          .body = body.dump(),
          .headers = {{"Content-Type", "application/json"}},
          .streaming_body = {},
          .log_details = {}};
}

}  // namespace gufo::server
