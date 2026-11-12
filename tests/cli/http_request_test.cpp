#include <gufo/http.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Headers = std::vector<std::pair<std::string, std::string>>;

// The parsed head as the server uses it: accepted with its framing, or 400.
struct Outcome {
  bool accepted = false;
  std::size_t len = 0;
  std::string method;
  std::string path;
  std::string query;
  bool http11 = false;
  Headers headers;
  std::uint64_t content_length = 0;
  bool expect_continue = false;
  bool operator==(const Outcome&) const = default;
};

std::string Text(GufoHttpSlice slice) {
  return std::string(reinterpret_cast<const char*>(slice.data), slice.len);
}

// Called through the Rust parser, so no exception may leave it.
int AddField(void* opaque, const std::uint8_t* name, std::size_t name_len,
             const std::uint8_t* value, std::size_t value_len) noexcept {
  try {
    static_cast<Headers*>(opaque)->emplace_back(
        std::string(reinterpret_cast<const char*>(name), name_len),
        std::string(reinterpret_cast<const char*>(value), value_len));
    return 0;
  } catch (...) {
    return 1;
  }
}

// The parser's result on the first `len` bytes; `status` reports partial
// heads.
Outcome New(std::string_view buffer, std::size_t known_partial,
            GufoHttpParse* status) {
  GufoHttpHead head{};
  Headers headers;
  *status = gufo_http_parse_head(
      reinterpret_cast<const std::uint8_t*>(buffer.data()), buffer.size(),
      known_partial, &head, AddField, &headers);
  if (*status == GUFO_HTTP_INTERNAL)
    throw std::runtime_error("gufo_http_parse_head failed internally");
  if (*status != GUFO_HTTP_COMPLETE || head.framing_valid == 0)
    return {};
  return {true,
          head.len,
          Text(head.method),
          Text(head.path),
          Text(head.query),
          head.http11 != 0,
          std::move(headers),
          head.content_length,
          head.expect_continue != 0};
}

Outcome New(std::string_view buffer) {
  GufoHttpParse status{};
  return New(buffer, 0, &status);
}

// As the server reads it: one byte more at a time until the head is complete
// or rejected.
Outcome NewByteByByte(std::string_view buffer) {
  std::size_t known = 0;
  for (std::size_t len = 1; len <= buffer.size(); ++len) {
    GufoHttpParse status{};
    Outcome outcome = New(buffer.substr(0, len), known, &status);
    if (status != GUFO_HTTP_PARTIAL)
      return outcome;
    known = len;
  }
  return {};
}
void Check(std::string_view text, bool accepted) {
  const auto outcome = New(text);
  if (outcome.accepted != accepted || NewByteByByte(text) != outcome)
    throw std::runtime_error("HTTP expected outcome differs");
}
}  // namespace
int main() {
  try {
    const std::string valid =
        "POST /path?q=x HTTP/1.1\r\nHost: a\r\nContent-Length: 3\r\nExpect: "
        "100-continue\r\n\r\n";
    Check(valid, true);
    const auto head = New(valid);
    if (head.method != "POST" || head.path != "/path" || head.query != "q=x" ||
        !head.http11 || head.content_length != 3 || !head.expect_continue ||
        head.len != valid.size() || head.headers.size() != 3)
      throw std::runtime_error("HTTP fields differ");
    for (const auto text :
         {"GET  / HTTP/1.1\r\n\r\n", "GET\t/ HTTP/1.1\r\n\r\n",
          "G@T / HTTP/1.1\r\n\r\n",
          "GET /a\x01"
          "b HTTP/1.1\r\n\r\n",
          "GET /\xff HTTP/1.1\r\n\r\n",
          "GET / HTTP/1.1\r\nX: a\x01"
          "b\r\n\r\n",
          "GET / HTTP/1.1\r\nX: a\r\n b\r\n\r\n",
          "POST / HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\n"})
      Check(text, false);
    Check("GET / HTTP/1.1\nHost: a\n\n", true);
    Check("\r\nGET / HTTP/1.1\r\n\r\n", true);
    for (const auto end : {"\r\n", "\n"}) {
      std::string text = std::string("GET / HTTP/1.1") + end;
      for (int i = 0; i < 100; ++i)
        text += std::string("X: a") + end;
      Check(text + end, true);
      Check(text + "X: a" + end + end, false);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
