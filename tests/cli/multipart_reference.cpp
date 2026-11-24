// Frozen copies of the multipart/form-data parsing in 0.33.0
// (src/cli/serve/audio_asr_api.cpp, video_api.cpp and image_api.cpp), which
// multipart_test compares with the Rust parser. Each API's field consumer is
// replaced by a call to on_field (marked "Test hook"); the parsing around it
// is unchanged. Do not change them to match new behavior.
#include "tests/cli/multipart_reference.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::multipart_reference {
namespace asr {
namespace {

constexpr std::size_t kMaximumTextFieldBytes = 16U << 10U;

struct MultipartPart {
  std::string filename;
  std::string body;
};

using Multipart = std::unordered_map<std::string, MultipartPart>;

std::string_view Trim(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

std::string Lower(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return result;
}

std::optional<std::string> Parameter(std::string_view value,
                                     std::string_view name) {
  const std::string target = Lower(name);
  std::size_t begin = 0;
  while (begin < value.size()) {
    const std::size_t semicolon = value.find(';', begin);
    const std::size_t end =
        semicolon == std::string_view::npos ? value.size() : semicolon;
    const std::string_view item = Trim(value.substr(begin, end - begin));
    const std::size_t equals = item.find('=');
    if (equals != std::string_view::npos &&
        Lower(Trim(item.substr(0, equals))) == target) {
      std::string_view parameter = Trim(item.substr(equals + 1U));
      if (parameter.size() >= 2U && parameter.front() == '"' &&
          parameter.back() == '"') {
        parameter.remove_prefix(1);
        parameter.remove_suffix(1);
      }
      if (parameter.find_first_of("\r\n") != std::string_view::npos) {
        return std::nullopt;
      }
      return std::string(parameter);
    }
    if (semicolon == std::string_view::npos) {
      break;
    }
    begin = semicolon + 1U;
  }
  return std::nullopt;
}

bool ExtractBoundary(std::string_view content_type, std::string* boundary) {
  const std::size_t semicolon = content_type.find(';');
  const std::string media_type = Lower(Trim(content_type.substr(0, semicolon)));
  if (media_type != "multipart/form-data") {
    return false;
  }
  const std::optional<std::string> parsed = Parameter(content_type, "boundary");
  if (!parsed.has_value() || parsed->empty() || parsed->size() > 70U ||
      parsed->find_first_of("\r\n") != std::string::npos) {
    return false;
  }
  *boundary = *parsed;
  return true;
}

bool ParseMultipart(std::string_view body, std::string_view boundary,
                    Multipart* output, std::string* error,
                    std::vector<std::string>* order) {
  const std::string delimiter = "--" + std::string(boundary);
  if (!body.starts_with(delimiter)) {
    *error = "multipart body does not start with its boundary";
    return false;
  }
  std::size_t cursor = delimiter.size();
  while (true) {
    if (body.substr(cursor, 2U) == "--") {
      return true;
    }
    if (body.substr(cursor, 2U) != "\r\n") {
      *error = "multipart boundary is malformed";
      return false;
    }
    cursor += 2U;
    const std::size_t headers_end = body.find("\r\n\r\n", cursor);
    if (headers_end == std::string_view::npos ||
        headers_end - cursor > kMaximumTextFieldBytes) {
      *error = "multipart part headers are malformed";
      return false;
    }

    std::string disposition;
    for (std::size_t line_begin = cursor; line_begin < headers_end;) {
      const std::size_t line_end = body.find("\r\n", line_begin);
      const std::size_t bounded_end =
          line_end == std::string_view::npos || line_end > headers_end
              ? headers_end
              : line_end;
      const std::string_view line =
          body.substr(line_begin, bounded_end - line_begin);
      const std::size_t colon = line.find(':');
      if (colon != std::string_view::npos &&
          Lower(Trim(line.substr(0, colon))) == "content-disposition") {
        disposition = std::string(Trim(line.substr(colon + 1U)));
      }
      if (bounded_end == headers_end) {
        break;
      }
      line_begin = bounded_end + 2U;
    }
    const std::optional<std::string> name = Parameter(disposition, "name");
    if (!name.has_value() || name->empty()) {
      *error = "multipart part is missing a field name";
      return false;
    }

    const std::size_t payload_begin = headers_end + 4U;
    const std::string next_marker = "\r\n" + delimiter;
    const std::size_t next = body.find(next_marker, payload_begin);
    if (next == std::string_view::npos) {
      *error = "multipart body is missing a closing boundary";
      return false;
    }
    MultipartPart part{
        .filename = Parameter(disposition, "filename").value_or(""),
        .body = std::string(body.substr(payload_begin, next - payload_begin)),
    };
    if (!output->emplace(*name, std::move(part)).second) {
      *error = "multipart field is duplicated: " + *name;
      return false;
    }
    if (output->size() > 12U) {
      *error = "multipart request contains too many fields";
      return false;
    }
    order->push_back(*name);  // Test hook.
    cursor = next + 2U + delimiter.size();
  }
}

}  // namespace
}  // namespace asr

namespace video {
namespace {

constexpr std::size_t kMaximumMultipartHeadersBytes = 16U << 10U;
constexpr std::size_t kMaximumMultipartFields = 16;

std::string_view TrimAscii(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

bool EqualCaseInsensitive(std::string_view left, std::string_view right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (std::tolower(static_cast<unsigned char>(left[index])) !=
        std::tolower(static_cast<unsigned char>(right[index]))) {
      return false;
    }
  }
  return true;
}

std::vector<std::string_view> HeaderSegments(std::string_view value,
                                             bool* valid) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  bool quoted = false;
  bool escaped = false;
  for (std::size_t index = 0; index < value.size(); ++index) {
    const char character = value[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (quoted && character == '\\') {
      escaped = true;
      continue;
    }
    if (character == '"') {
      quoted = !quoted;
    } else if (character == ';' && !quoted) {
      result.push_back(TrimAscii(value.substr(begin, index - begin)));
      begin = index + 1;
    }
  }
  if (quoted || escaped) {
    *valid = false;
    return {};
  }
  result.push_back(TrimAscii(value.substr(begin)));
  *valid = true;
  return result;
}

bool DecodeHeaderParameter(std::string_view value, std::string* output) {
  value = TrimAscii(value);
  if (value.empty()) {
    output->clear();
    return true;
  }
  if (value.front() != '"') {
    if (value.find_first_of("\"\r\n") != std::string_view::npos) {
      return false;
    }
    *output = value;
    return true;
  }
  if (value.size() < 2 || value.back() != '"') {
    return false;
  }
  output->clear();
  const std::string_view inner = value.substr(1, value.size() - 2);
  bool escaped = false;
  for (const char character : inner) {
    if (escaped) {
      output->push_back(character);
      escaped = false;
    } else if (character == '\\') {
      escaped = true;
    } else if (character == '"' || character == '\r' || character == '\n') {
      return false;
    } else {
      output->push_back(character);
    }
  }
  return !escaped;
}

bool HeaderParameter(std::string_view value, std::string_view parameter,
                     std::string* output, bool* present) {
  bool valid = false;
  const std::vector<std::string_view> segments = HeaderSegments(value, &valid);
  if (!valid || segments.empty()) {
    return false;
  }
  *present = false;
  for (std::size_t index = 1; index < segments.size(); ++index) {
    const std::size_t equals = segments[index].find('=');
    if (equals == std::string_view::npos) {
      return false;
    }
    const std::string_view name = TrimAscii(segments[index].substr(0, equals));
    if (!EqualCaseInsensitive(name, parameter)) {
      continue;
    }
    if (*present ||
        !DecodeHeaderParameter(segments[index].substr(equals + 1), output)) {
      return false;
    }
    *present = true;
  }
  return true;
}

std::optional<std::string> MultipartBoundary(std::string_view content_type) {
  std::string boundary;
  bool present = false;
  if (!HeaderParameter(content_type, "boundary", &boundary, &present) ||
      !present || boundary.empty() || boundary.size() > 200 ||
      boundary.find_first_of("\r\n") != std::string::npos) {
    return std::nullopt;
  }
  return boundary;
}

Status ParseMultipartBody(std::string_view content_type,
                          std::string_view request_body,
                          const FieldFn& on_field, std::string* error) {
  const auto boundary = MultipartBoundary(content_type);
  if (!boundary.has_value()) {
    *error = "multipart video request has an invalid boundary";
    return Status::kMalformed;
  }
  const std::string delimiter = "--" + *boundary;
  const std::string_view body = request_body;
  if (!body.starts_with(delimiter)) {
    *error = "multipart video request does not start with its boundary";
    return Status::kMalformed;
  }

  json::Value result = json::Value::object();
  std::size_t cursor = delimiter.size();
  std::size_t fields = 0;
  while (true) {
    if (body.substr(cursor).starts_with("--")) {
      cursor += 2;
      if (body.substr(cursor).starts_with("\r\n")) {
        cursor += 2;
      }
      if (cursor != body.size()) {
        *error = "multipart video request has data after its final boundary";
        return Status::kMalformed;
      }
      return Status::kComplete;
    }
    if (!body.substr(cursor).starts_with("\r\n")) {
      *error = "multipart video request has an invalid boundary separator";
      return Status::kMalformed;
    }
    cursor += 2;
    const std::size_t headers_end = body.find("\r\n\r\n", cursor);
    if (headers_end == std::string_view::npos ||
        headers_end - cursor > kMaximumMultipartHeadersBytes) {
      *error = "multipart video request has invalid part headers";
      return Status::kMalformed;
    }

    std::string content_disposition;
    std::string_view headers = body.substr(cursor, headers_end - cursor);
    while (!headers.empty()) {
      const std::size_t newline = headers.find("\r\n");
      const std::string_view line = headers.substr(0, newline);
      const std::size_t colon = line.find(':');
      if (colon == std::string_view::npos) {
        *error = "multipart video request has a malformed part header";
        return Status::kMalformed;
      }
      const std::string_view name = TrimAscii(line.substr(0, colon));
      const std::string_view value = TrimAscii(line.substr(colon + 1));
      if (EqualCaseInsensitive(name, "content-disposition")) {
        if (!content_disposition.empty()) {
          *error = "multipart video request repeats Content-Disposition";
          return Status::kMalformed;
        }
        content_disposition = value;
      }
      if (newline == std::string_view::npos) {
        break;
      }
      headers.remove_prefix(newline + 2);
    }

    bool disposition_valid = false;
    const std::vector<std::string_view> disposition_segments =
        HeaderSegments(content_disposition, &disposition_valid);
    std::string field_name;
    std::string filename;
    bool name_present = false;
    bool filename_present = false;
    if (!disposition_valid || disposition_segments.empty() ||
        !EqualCaseInsensitive(disposition_segments.front(), "form-data") ||
        !HeaderParameter(content_disposition, "name", &field_name,
                         &name_present) ||
        !HeaderParameter(content_disposition, "filename", &filename,
                         &filename_present) ||
        !name_present || field_name.empty()) {
      *error = "multipart video request has invalid Content-Disposition";
      return Status::kMalformed;
    }

    const std::size_t content_begin = headers_end + 4;
    const std::string next_boundary = "\r\n" + delimiter;
    const std::size_t content_end = body.find(next_boundary, content_begin);
    if (content_end == std::string_view::npos) {
      *error = "multipart video request is missing a closing boundary";
      return Status::kMalformed;
    }
    if (++fields > kMaximumMultipartFields || result.contains(field_name)) {
      *error = fields > kMaximumMultipartFields
                   ? "multipart video request contains too many fields"
                   : "multipart video request repeats a field";
      return Status::kMalformed;
    }
    const std::string_view content =
        body.substr(content_begin, content_end - content_begin);
    // Test hook in place of the field consumer.
    if (!on_field({field_name, filename_present, std::string(content)})) {
      return Status::kStopped;
    }
    result[field_name] = std::string();
    cursor = content_end + next_boundary.size();
  }
}

}  // namespace
}  // namespace video

namespace image {
namespace {

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

Status Multipart(std::string_view content_type, std::string_view request_body,
                 const FieldFn& on_field) {
  if (!content_type.starts_with("multipart/form-data"))
    throw std::invalid_argument("image edits require multipart/form-data");
  const auto boundary = Parameter(content_type, "boundary");
  if (boundary.empty() || boundary.size() > 70 ||
      boundary.find_first_of("\r\n") != std::string::npos)
    throw std::invalid_argument("invalid multipart boundary");
  const std::string marker = "--" + boundary;
  std::string_view body = request_body;
  if (!body.starts_with(marker))
    throw std::invalid_argument("multipart body does not start with boundary");
  body.remove_prefix(marker.size());
  std::size_t fields = 0;
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
    // Test hook in place of the field consumer.
    if (!on_field({name, false, std::string(data)}))
      return Status::kStopped;
  }
  return Status::kComplete;
}

}  // namespace
}  // namespace image

Result Asr(std::string_view content_type, std::string_view body) {
  std::string boundary;
  if (!asr::ExtractBoundary(content_type, &boundary)) {
    return {.status = Status::kContentType, .message = {}, .fields = {}};
  }
  asr::Multipart parts;
  std::vector<std::string> order;
  std::string error;
  if (!asr::ParseMultipart(body, boundary, &parts, &error, &order)) {
    return {.status = Status::kMalformed, .message = error, .fields = {}};
  }
  Result result{.status = Status::kComplete, .message = {}, .fields = {}};
  for (const std::string& name : order) {
    result.fields.push_back({name, false, parts.at(name).body});
  }
  return result;
}

Result Video(std::string_view content_type, std::string_view body,
             std::size_t stop_at) {
  Result result{.status = Status::kComplete, .message = {}, .fields = {}};
  result.status = video::ParseMultipartBody(
      content_type, body,
      [&](const Field& field) {
        result.fields.push_back(field);
        return result.fields.size() != stop_at + 1;
      },
      &result.message);
  return result;
}

Result Image(std::string_view content_type, std::string_view body,
             std::size_t stop_at) {
  Result result{.status = Status::kComplete, .message = {}, .fields = {}};
  try {
    result.status =
        image::Multipart(content_type, body, [&](const Field& field) {
          result.fields.push_back(field);
          return result.fields.size() != stop_at + 1;
        });
  } catch (const std::invalid_argument& error) {
    result.status = Status::kMalformed;
    result.message = error.what();
  }
  return result;
}

}  // namespace gufo::multipart_reference
