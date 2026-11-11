// Frozen copies of the HTTP request parsing in 0.33.0
// (src/cli/serve/http_server.cpp and websocket.cpp), which http_request_test
// compares with the Rust parser. ParseHead wraps the head-parsing block of
// HttpServer::handle_connection unchanged. Do not change them to match new
// behavior.
#include "tests/cli/http_request_reference.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <sstream>

namespace gufo::http_reference {
namespace {
std::string ToLower(std::string_view s) {
  std::string out(s);
  for (auto& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

std::string UrlDecode(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '+') {
      out += ' ';
    } else if (c == '%' && i + 2 < s.size()) {
      auto hexval = [](char h) {
        if (h >= '0' && h <= '9')
          return h - '0';
        if (h >= 'a' && h <= 'f')
          return h - 'a' + 10;
        if (h >= 'A' && h <= 'F')
          return h - 'A' + 10;
        return -1;
      };
      const int high = hexval(s[i + 1]);
      const int low = hexval(s[i + 2]);
      if (high >= 0 && low >= 0) {
        out += static_cast<char>((high << 4) | low);
        i += 2;
      } else {
        out += c;
      }
    } else {
      out += c;
    }
  }
  return out;
}
}  // namespace

std::optional<std::size_t> ParseContentLength(const HttpRequest& request) {
  std::optional<std::size_t> length;
  for (const auto& [name, value] : request.headers) {
    const auto lowered = ToLower(name);
    // Chunked transfer is not implemented. Never interpret its encoded bytes
    // as an empty or partial inference request.
    if (lowered == "transfer-encoding") {
      return std::nullopt;
    }
    if (lowered != "content-length") {
      continue;
    }
    std::size_t parsed = 0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size() ||
        (length.has_value() && *length != parsed)) {
      return std::nullopt;
    }
    length = parsed;
  }
  return length.value_or(0);
}

bool ExpectsContinue(const HttpRequest& request) {
  return std::ranges::any_of(request.headers, [](const auto& header) {
    return ToLower(header.first) == "expect" &&
           ToLower(header.second) == "100-continue";
  });
}

std::string QueryParam(std::string_view query, const std::string& key) {
  std::string_view remaining = query;
  while (!remaining.empty()) {
    const auto end = remaining.find('&');
    const auto parameter = remaining.substr(0, end);
    const auto equals = parameter.find('=');
    if (UrlDecode(parameter.substr(0, equals)) == key) {
      return equals == std::string_view::npos
                 ? ""
                 : UrlDecode(parameter.substr(equals + 1));
    }
    if (end == std::string_view::npos) {
      break;
    }
    remaining.remove_prefix(end + 1);
  }
  return "";
}

bool HasToken(std::string value, std::string_view token) {
  for (char& c : value)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  for (std::size_t begin = 0; begin < value.size();) {
    auto end = value.find(',', begin);
    if (end == std::string::npos)
      end = value.size();
    auto item = std::string_view(value).substr(begin, end - begin);
    while (!item.empty() && (item.front() == ' ' || item.front() == '\t'))
      item.remove_prefix(1);
    while (!item.empty() && (item.back() == ' ' || item.back() == '\t'))
      item.remove_suffix(1);
    if (item == token)
      return true;
    begin = end + 1;
  }
  return false;
}

Head ParseHead(const std::string& buffer) {
  Head result;
  if (buffer.find("\r\n\r\n") == std::string::npos)
    return result;
  HttpRequest req;
  bool http11 = false;
  {
    const auto delim_pos = buffer.find("\r\n\r\n");
    const std::string headers = buffer.substr(0, delim_pos);
    std::string body = buffer.substr(delim_pos + 4);

    const auto eol = headers.find("\r\n");
    const std::string line =
        (eol == std::string::npos) ? headers : headers.substr(0, eol);
    std::istringstream ls(line);
    std::string target;
    std::string version;
    std::string extra;
    ls >> req.method >> target >> version;
    http11 = version == "HTTP/1.1";
    bool valid_headers = !req.method.empty() && target.starts_with('/') &&
                         (version == "HTTP/1.0" || version == "HTTP/1.1") &&
                         !(ls >> extra);
    const auto qpos = target.find('?');
    if (qpos != std::string::npos) {
      req.query = target.substr(qpos + 1);
      req.path = target.substr(0, qpos);
    } else {
      req.path = target;
    }

    std::size_t cursor = eol == std::string::npos ? headers.size() : eol + 2;
    while (cursor < headers.size()) {
      const std::size_t next = headers.find("\r\n", cursor);
      const std::size_t line_end =
          next == std::string::npos ? headers.size() : next;
      const std::string_view header_line(headers.data() + cursor,
                                         line_end - cursor);
      const std::size_t colon = header_line.find(':');
      // RFC 9112 5.1: reject whitespace before the colon and obsolete
      // line folding instead of storing a name no lookup would match.
      const auto token = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) != 0 ||
               std::string_view("!#$%&'*+-.^_`|~").find(c) !=
                   std::string_view::npos;
      };
      if (colon != std::string_view::npos && colon > 0 &&
          std::ranges::all_of(header_line.substr(0, colon), token)) {
        std::string name(header_line.substr(0, colon));
        std::string_view raw_value = header_line.substr(colon + 1);
        while (!raw_value.empty() && std::isspace(static_cast<unsigned char>(
                                         raw_value.front())) != 0) {
          raw_value.remove_prefix(1);
        }
        while (!raw_value.empty() && std::isspace(static_cast<unsigned char>(
                                         raw_value.back())) != 0) {
          raw_value.remove_suffix(1);
        }
        req.headers.emplace_back(std::move(name), std::string(raw_value));
      } else {
        valid_headers = false;
      }
      if (next == std::string::npos) {
        break;
      }
      cursor = next + 2;
    }

    const auto parsed_length = ParseContentLength(req);
    result.complete = true;
    result.valid = valid_headers && parsed_length.has_value();
    result.len = delim_pos + 4;
    result.method = req.method;
    result.path = req.path;
    result.query = req.query;
    result.http11 = http11;
    result.headers = req.headers;
    result.content_length = parsed_length.value_or(0);
    result.expect_continue = ExpectsContinue(req);
    (void)body;
  }
  return result;
}

}  // namespace gufo::http_reference
