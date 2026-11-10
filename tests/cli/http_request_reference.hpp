#ifndef GUFO_TESTS_CLI_HTTP_REQUEST_REFERENCE_HPP_
#define GUFO_TESTS_CLI_HTTP_REQUEST_REFERENCE_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::http_reference {

// The request fields the 0.33.0 parser read.
struct HttpRequest {
  std::string method;
  std::string path;
  std::string query;
  std::vector<std::pair<std::string, std::string>> headers;
};

// What 0.33.0 made of a buffer: incomplete without CRLF CRLF; otherwise
// valid (accepted, with its framing) or not (400).
struct Head {
  bool complete = false;
  bool valid = false;
  std::size_t len = 0;
  std::string method;
  std::string path;
  std::string query;
  bool http11 = false;
  std::vector<std::pair<std::string, std::string>> headers;
  std::size_t content_length = 0;
  bool expect_continue = false;
};

std::optional<std::size_t> ParseContentLength(const HttpRequest& request);
bool ExpectsContinue(const HttpRequest& request);
std::string QueryParam(std::string_view query, const std::string& key);
bool HasToken(std::string value, std::string_view token);
Head ParseHead(const std::string& buffer);

}  // namespace gufo::http_reference

#endif  // GUFO_TESTS_CLI_HTTP_REQUEST_REFERENCE_HPP_
