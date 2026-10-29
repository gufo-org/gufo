#include "src/cli/serve/image_api.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <ctime>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
namespace {

namespace image = models::qwen_image_21;

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

std::string Base64(std::span<const std::uint8_t> bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve((bytes.size() + 2) / 3 * 4);
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const std::uint32_t word =
        (std::uint32_t(bytes[i]) << 16) |
        (i + 1 < bytes.size() ? std::uint32_t(bytes[i + 1]) << 8 : 0) |
        (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    output += alphabet[(word >> 18) & 63];
    output += alphabet[(word >> 12) & 63];
    output += i + 1 < bytes.size() ? alphabet[(word >> 6) & 63] : '=';
    output += i + 2 < bytes.size() ? alphabet[word & 63] : '=';
  }
  return output;
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

struct Parsed {
  json::Value fields = json::Value::object();
  std::vector<image::Image> images;
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
      parsed.images.push_back(image::DecodeImage(std::span(
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

std::vector<image::Lora> ParseLoras(const json::Value& fields,
                                    const ImageService& service) {
  const auto* value = fields.find("loras");
  if (!value)
    return service.loras().defaults;
  if (!value->is_array())
    throw std::invalid_argument("loras must be an array");
  std::vector<image::Lora> loras;
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
    auto& lora = loras.emplace_back(image::Lora{name->str()});
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

std::string LoraList(const std::vector<image::Lora>& loras) {
  std::ostringstream list;
  for (const auto& lora : loras)
    if (lora.scale != 0)
      list << (list.tellp() > 0 ? "," : "") << lora.name << ':' << lora.scale;
  return list.tellp() > 0 ? list.str() : "none";
}

}  // namespace

ImageService::ImageService(const std::filesystem::path& root,
                           std::string model_id, Runner runner,
                           ImageLoraOptions loras)
    : model_id_(std::move(model_id)),
      runner_(std::move(runner)),
      loras_(std::move(loras)) {
  if (model_id_.empty())
    throw std::invalid_argument("image model ID must not be empty");
  for (const auto& [name, path] : loras_.files) {
    (void)path;
    if (name.empty() || name.size() > 64 ||
        name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrs"
                               "tuvwxyz0123456789._-") != std::string::npos)
      throw std::invalid_argument("LoRA name '" + name +
                                  "' must be 1-64 of [A-Za-z0-9._-]");
    if (!lora_info_.emplace(name, image::LoraInfo{}).second)
      throw std::invalid_argument("duplicate --lora name '" + name + "'");
  }
  for (const auto& lora : loras_.defaults)
    if (!HasLora(lora.name))
      throw std::invalid_argument("--lora-default names unknown adapter '" +
                                  lora.name + "'");
  Request defaults;
  defaults.loras = loras_.defaults;
  image::ValidateRequest(defaults);
  if (!runner_) {
#if defined(ENGINE_ENABLE_HIP)
    model_ = std::make_unique<image::Model>(root);
    for (const auto& [name, path] : loras_.files) {
      const auto& info = lora_info_[name] = model_->AddLora(name, path);
      Logger::Info("image", "event=lora_registered name=" + name +
                                " file=" + path.filename().string() +
                                " format=" + info.format +
                                " targets=" + std::to_string(info.targets) +
                                " rank=" + std::to_string(info.min_rank) + '-' +
                                std::to_string(info.max_rank));
    }
    runner_ = [this](const Request& request,
                     const CancellationCheck& cancelled) {
      return model_->Generate(request, cancelled);
    };
#else
    (void)root;
    throw std::runtime_error("Qwen-Image requires the HIP build");
#endif
  }
}

ImageService::~ImageService() = default;

bool ImageService::HasLora(std::string_view name) const {
  return lora_info_.contains(name);
}

const image::LoraInfo& ImageService::lora_info(std::string_view name) const {
  return lora_info_.find(name)->second;
}

std::vector<image::Lora> ImageService::ActiveLoras() const {
  return model_ ? model_->ActiveLoras() : std::vector<image::Lora>{};
}

ImageService::Result ImageService::Generate(
    const Request& request, const CancellationCheck& cancelled) {
  std::unique_lock<std::timed_mutex> lock(gate_, std::defer_lock);
  while (!lock.try_lock_for(std::chrono::milliseconds(20)))
    if (cancelled && cancelled())
      throw std::runtime_error("image request cancelled");
  if (cancelled && cancelled())
    throw std::runtime_error("image request cancelled");
  return runner_(request, cancelled);
}

HttpResponse HandleImageApiRequest(const HttpRequest& request,
                                   ImageService& service) {
  if (request.method != "POST")
    return Error(400, "image endpoints require POST", "invalid_method");
  try {
    Parsed parsed;
    const bool edit = request.path == "/v1/images/edits";
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
    const std::set<std::string> allowed{
        "prompt",        "model",   "n",          "size", "response_format",
        "output_format", "quality", "background", "user", "seed",
        "steps",         "stream",  "loras"};
    for (const auto& [name, value] : fields.members()) {
      (void)value;
      if (!allowed.contains(name))
        throw std::invalid_argument("unsupported image parameter: " + name);
    }
    if (!fields.contains("prompt"))
      throw std::invalid_argument("prompt is required");
    const auto model = String(fields, "model", service.model_id().c_str());
    if (model != service.model_id())
      return Error(404, "requested image model is not loaded",
                   "model_not_found");
    if (String(fields, "response_format", "b64_json") != "b64_json" ||
        String(fields, "output_format", "png") != "png")
      throw std::invalid_argument(
          "image output supports PNG with response_format b64_json");
    const auto quality = String(fields, "quality", "auto");
    if (quality != "auto" && quality != "standard" && quality != "high")
      throw std::invalid_argument(
          "quality must be auto, standard or high (40 steps)");
    if (const auto* stream = fields.find("stream");
        stream && (!stream->is_bool() || stream->as_bool()))
      throw std::invalid_argument("streaming image responses are unsupported");
    (void)String(fields, "user", "");
    const auto background = String(fields, "background", "auto");
    if (background != "auto" && background != "transparent" &&
        background != "opaque")
      throw std::invalid_argument("invalid image background");
    image::Request generation;
    generation.prompt = String(fields, "prompt", "");
    if (background == "transparent")
      generation.prompt =
          "This is an RGBA image with transparency. " + generation.prompt +
          " The image has alpha channel and the background is transparent.";
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
    generation.steps = static_cast<int>(Number(fields, "steps", 40, 2, 100));
    std::random_device entropy;
    generation.seed =
        Number(fields, "seed",
               (static_cast<std::uint64_t>(entropy()) << 21) ^ entropy(), 0,
               (1ULL << 53) - 1);
    generation.loras = ParseLoras(fields, service);
    const auto count = Number(fields, "n", 1, 1, 10);
    image::ValidateRequest(generation);
    auto body = json::Value::object();
    body["created"] = static_cast<double>(std::time(nullptr));
    auto data = json::Value::array();
    for (std::uint64_t i = 0; i < count; ++i) {
      auto result = service.Generate(generation, request.is_cancelled);
      if (background == "opaque") {
        for (std::size_t p = 0; p < result.image.rgba.size(); p += 4) {
          const int a = result.image.rgba[p + 3];
          for (int c = 0; c < 3; ++c)
            result.image.rgba[p + c] = static_cast<std::uint8_t>(
                (result.image.rgba[p + c] * a + 255 * (255 - a) + 127) / 255);
          result.image.rgba[p + 3] = 255;
        }
      }
      auto item = json::Value::object();
      item["b64_json"] = Base64(image::EncodePng(result.image));
      data.push_back(std::move(item));
      std::ostringstream details;
      details << "request=" << request.request_id
              << " event=generated mode=" << (edit ? "edit" : "generate")
              << " seed=" << generation.seed << " size=" << generation.width
              << 'x' << generation.height
              << " references=" << generation.images.size()
              << " steps=" << generation.steps;
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
  const auto scale = [](const std::vector<image::Lora>& loras,
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
  for (const auto& [name, path] : service.loras().files) {
    const auto& info = service.lora_info(name);
    auto metadata = json::Value::object();
    metadata["format"] = info.format;
    metadata["targets"] = info.targets;
    metadata["rank"] =
        std::to_string(info.min_rank) + '-' + std::to_string(info.max_rank);
    auto item = json::Value::object();
    item["name"] = name;
    item["alias"] = name;
    item["path"] = path.filename().string();
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
