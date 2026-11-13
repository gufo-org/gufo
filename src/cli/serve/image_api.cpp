#include "src/cli/serve/image_api.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <ctime>
#include <optional>
#include <stdexcept>
#include <system_error>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/multipart.hpp"
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

template<typename T>
std::string Shortest(T value) {
  char text[32];
  return {text, std::to_chars(text, text + sizeof(text), value).ptr};
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
                    : status == 503 ? "Service Unavailable"
                                    : "Internal Server Error",
          .body = body.dump(),
          .headers = {{"Content-Type", "application/json"}}};
}

struct Parsed {
  json::Value fields = json::Value::object();
  std::vector<ext::ImageBuffer> images;
};

Parsed Multipart(const HttpRequest& request) {
  Parsed parsed;
  std::size_t image_bytes = 0, image_pixels = 0;
  const MultipartResult form = ParseMultipart(
      request.header("content-type"), request.body,
      {.max_fields = 32, .unique_names = false},
      [&](const MultipartField& field) {
        const std::string name(field.name);
        const std::string_view data = field.content;
        if (name == "image" || name == "image[]") {
          if (parsed.images.size() == 10 ||
              data.size() > core::kMaxEncodedImageBytes - image_bytes)
            throw std::invalid_argument(
                "image inputs exceed 10 images or 20 MiB");
          image_bytes += data.size();
          parsed.images.push_back(DecodeImage(
              std::span(reinterpret_cast<const std::uint8_t*>(data.data()),
                        data.size())));
          const auto& decoded = parsed.images.back();
          image_pixels +=
              static_cast<std::size_t>(decoded.width) * decoded.height;
          if (image_pixels > core::kMaxImagePixels)
            throw std::invalid_argument(
                "reference images exceed 32 megapixels combined");
        } else {
          if (name == "mask")
            throw std::invalid_argument(
                "mask uploads are unsupported; provide annotated reference "
                "images");
          if (parsed.fields.contains(name))
            throw std::invalid_argument("duplicate field " + name);
          if (data.size() > (64U << 10U))
            throw std::invalid_argument("multipart field too large");
          if (name == "n" || name == "seed" || name == "steps" ||
              name == "stream" || name == "loras" || name == "guidance_scale" ||
              name == "strength") {
            parsed.fields[name] = ParseJson(data);
          } else {
            parsed.fields[name] = std::string(data);
          }
        }
        return true;
      });
  switch (form.status) {
    case MultipartStatus::kComplete:
      break;
    case MultipartStatus::kNotFormData:
      throw std::invalid_argument("image edits require multipart/form-data");
    case MultipartStatus::kBadBoundary:
      throw std::invalid_argument("invalid multipart boundary");
    case MultipartStatus::kTooManyFields:
      throw std::invalid_argument("too many multipart fields");
    case MultipartStatus::kMalformed:
      throw std::invalid_argument(form.detail);
    case MultipartStatus::kDuplicateField:
    case MultipartStatus::kStopped:
      // Names may repeat here, and the field handler throws, not stops.
      throw std::logic_error("unexpected multipart result");
  }
  if (parsed.images.empty())
    throw std::invalid_argument("image edit requires an image");
  return parsed;
}

std::optional<double> Real(const json::Value& fields, const char* name,
                           double low, double high) {
  const auto* value = fields.find(name);
  if (!value)
    return std::nullopt;
  if (!value->is_number() || !std::isfinite(value->as_double()) ||
      value->as_double() < low || value->as_double() > high)
    throw std::invalid_argument(std::string(name) + " must be a number from " +
                                Shortest(low) + " to " + Shortest(high));
  return value->as_double();
}

/// The `image` member of a JSON generation: base64 or a data URL. URLs are
/// never fetched.
ext::ImageBuffer InlineImage(const json::Value& value) {
  if (value.is_string()) {
    const auto& text = value.str();
    if (text.starts_with("data:"))
      return DecodeImage(core::ReadImageUrl(text));
    std::vector<std::uint8_t> bytes;
    if (core::DecodeBase64(text, {.max_decoded = core::kMaxEncodedImageBytes},
                           bytes) == core::Base64Status::kOk)
      return DecodeImage(bytes);
  }
  throw std::invalid_argument("image must be base64 or a data URL");
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

}  // namespace

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
    // Flat extensions, as other OpenAI-compatible image servers name them;
    // multipart edits send them as plain form fields (SDK `extra_body`).
    generation.negative_prompt = String(fields, "negative_prompt", "");
    if (const auto scale = Real(fields, "guidance_scale", 0, 100))
      generation.guidance_scale = static_cast<float>(*scale);
    generation.strength = Real(fields, "strength", 0.01, 1).value_or(1);
    if (const auto* image = fields.find("image"); image && !edit)
      generation.init_image = InlineImage(*image);
    if (edit && generation.strength < 1) {
      // Below full strength, the first uploaded image is the start image;
      // any further images stay references. Multipart() refuses an edit
      // without images.
      generation.init_image = std::move(generation.images.front());
      generation.images.erase(generation.images.begin());
    }
    const auto size = String(fields, "size", "auto");
    if (size == "auto") {
      if (generation.init_image) {
        // Keep the start image's canvas, rounded to the 32-pixel grid.
        const auto grid = [](int v) {
          return std::max(32, static_cast<int>(std::nearbyint(v / 32.0)) * 32);
        };
        generation.width = grid(generation.init_image->width);
        generation.height = grid(generation.init_image->height);
      } else if (!generation.images.empty()) {
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
    generation.seed =
        Number(fields, "seed", RandomImageSeed(), 0, kMaxImageSeed);
    generation.loras = ParseLoras(fields, service);
    const auto count = Number(fields, "n", 1, 1, 10);
    ValidateImageRequest(generation);
    service.Validate(generation);
    auto body = json::Value::object();
    body["created"] = static_cast<double>(std::time(nullptr));
    auto data = json::Value::array();
    for (auto& image : GenerateImages(
             service, std::move(generation), count, request.request_id,
             edit ? "edit" : "generate", request.is_cancelled)) {
      auto item = json::Value::object();
      item["b64_json"] = core::EncodeBase64(image.png);
      data.push_back(std::move(item));
    }
    body["data"] = std::move(data);
    return {.status = 200,
            .reason = "OK",
            .body = body.dump(),
            .headers = {{"Content-Type", "application/json"}}};
  } catch (const std::invalid_argument& error) {
    return Error(400, error.what(), "invalid_request");
  } catch (const ext::ImageResourceError& error) {
    return Error(503, error.what(), "insufficient_gpu_memory");
  } catch (const std::exception& error) {
    return Error(500, error.what(), "image_generation_failed");
  }
}

}  // namespace gufo::server
