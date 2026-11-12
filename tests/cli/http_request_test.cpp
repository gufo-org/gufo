// Compares the Rust HTTP request parser (gufo_http_*) with frozen copies of
// the 0.33.0 parsing in tests/cli/http_request_reference.cpp. Request heads
// that follow RFC 9112 strictly (CRLF line ends, single spaces, a token
// method, a target without control bytes in UTF-8, no control bytes in field
// values) must
// parse identically: fields, body length, Expect and head length. Every
// other difference must be one of the documented ones: httparse rejects
// loose whitespace in the request line, non-token methods, control bytes or
// ill-formed UTF-8 in the target, control bytes in field values and more
// than 100 header fields, and accepts bare LF line ends and one empty line
// before the request line.
// Query parameters and header token lists must decode identically. Each head
// is also fed one byte at a time and must give the same result.
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

#include "tests/cli/http_request_reference.hpp"

namespace {

namespace ref = gufo::http_reference;
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

Outcome Old(const std::string& head) {
  const ref::Head parsed = ref::ParseHead(head);
  if (!parsed.complete || !parsed.valid)
    return {};
  return {true,
          parsed.len,
          parsed.method,
          parsed.path,
          parsed.query,
          parsed.http11,
          parsed.headers,
          parsed.content_length,
          parsed.expect_continue};
}

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

// Well-formed UTF-8: httparse hands the target over as text.
bool Utf8(std::string_view text) {
  for (std::size_t i = 0; i < text.size();) {
    const auto lead = static_cast<unsigned char>(text[i]);
    std::size_t len = lead < 0x80                    ? 1
                      : lead >= 0xc2 && lead <= 0xdf ? 2
                      : lead >= 0xe0 && lead <= 0xef ? 3
                      : lead >= 0xf0 && lead <= 0xf4 ? 4
                                                     : 0;
    if (len == 0 || i + len > text.size())
      return false;
    for (std::size_t k = 1; k < len; ++k) {
      const auto c = static_cast<unsigned char>(text[i + k]);
      const unsigned low = k == 1 && lead == 0xe0   ? 0xa0
                           : k == 1 && lead == 0xf0 ? 0x90
                                                    : 0x80;
      const unsigned high = k == 1 && lead == 0xed   ? 0x9f
                            : k == 1 && lead == 0xf4 ? 0x8f
                                                     : 0xbf;
      if (c < low || c > high)
        return false;
    }
    i += len;
  }
  return true;
}

bool Tchar(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
         (c >= 'A' && c <= 'Z') ||
         std::string_view("!#$%&'*+-.^_`|~").find(static_cast<char>(c)) !=
             std::string_view::npos;
}

// The documented ways a head can differ from RFC 9112's strict form.
struct Loose {
  bool bare_lf = false;
  bool leading_empty_line = false;
  // Whitespace, method bytes, or control bytes or ill-formed UTF-8 in the
  // target.
  bool request_line = false;
  bool field_control = false;
  // More than 100 lines after the request line, counted as the parser does.
  bool too_many_fields = false;
};

Loose Classify(std::string_view head) {
  Loose loose;
  for (std::size_t i = 0; i < head.size(); ++i) {
    if (head[i] == '\n' && (i == 0 || head[i - 1] != '\r'))
      loose.bare_lf = true;
  }
  loose.leading_empty_line = head.starts_with("\r\n") || head.starts_with("\n");
  {
    const std::size_t start = head.starts_with("\r\n") ? 2
                              : head.starts_with("\n") ? 1
                                                       : 0;
    std::size_t lines = 0;
    for (std::size_t i = start; i < head.size(); ++i) {
      if (head[i] != '\n')
        continue;
      ++lines;
      if (i + 1 < head.size() && (head[i + 1] == '\n' || head[i + 1] == '\r'))
        break;
    }
    loose.too_many_fields = lines > 101;
  }
  const std::size_t eol = head.find('\n');
  std::string_view line = head.substr(0, eol);
  if (line.ends_with('\r'))
    line.remove_suffix(1);
  const std::size_t first = line.find(' ');
  const std::size_t second =
      first == std::string_view::npos ? first : line.find(' ', first + 1);
  if (first == std::string_view::npos || second == std::string_view::npos ||
      first == 0 || second == first + 1 || second + 1 >= line.size()) {
    loose.request_line = true;
  } else {
    for (std::size_t i = 0; i < first; ++i)
      loose.request_line |= !Tchar(static_cast<unsigned char>(line[i]));
    const std::string_view target = line.substr(first + 1, second - first - 1);
    for (const char byte : target) {
      const auto c = static_cast<unsigned char>(byte);
      loose.request_line |= c < 0x21 || c == 0x7f;
    }
    loose.request_line |= !Utf8(target);
    for (std::size_t i = second + 1; i < line.size(); ++i) {
      const auto c = static_cast<unsigned char>(line[i]);
      loose.request_line |= c <= 0x20 || c == 0x7f;
    }
  }
  if (eol != std::string_view::npos) {
    for (std::size_t i = eol + 1; i < head.size(); ++i) {
      const auto c = static_cast<unsigned char>(head[i]);
      const bool line_end = c == '\n' || (c == '\r' && i + 1 < head.size() &&
                                          head[i + 1] == '\n');
      if ((c < 0x20 && c != '\t' && !line_end) || c == 0x7f)
        loose.field_control = true;
    }
  }
  return loose;
}

std::string Hex(std::string_view text) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  for (const char c : text.substr(0, 48)) {
    const auto byte = static_cast<unsigned char>(c);
    if (byte >= 0x20 && byte < 0x7f && byte != '\\') {
      out += c;
    } else {
      out += "\\x";
      out += kHex[byte >> 4];
      out += kHex[byte & 15];
    }
  }
  return out;
}

struct Totals {
  std::size_t heads = 0;
  std::size_t differing = 0;
  std::size_t failures = 0;
};

void Fail(Totals& totals, const std::string& what, std::string_view head) {
  if (++totals.failures <= 12)
    std::fprintf(stderr, "%s: \"%s\"\n", what.c_str(), Hex(head).c_str());
}

void CompareHead(Totals& totals, const std::string& head) {
  ++totals.heads;
  const Outcome old = Old(head);
  const Outcome now = New(head);
  if (!(NewByteByByte(head) == now))
    Fail(totals, "byte-by-byte parse differs", head);
  if (old == now)
    return;
  ++totals.differing;
  const Loose loose = Classify(head);
  if (old.accepted && !now.accepted) {
    if (!(loose.request_line || loose.field_control || loose.bare_lf ||
          loose.too_many_fields))
      Fail(totals, "rejected now, for no documented reason", head);
  } else if (!old.accepted && now.accepted) {
    if (!(loose.bare_lf || loose.leading_empty_line))
      Fail(totals, "accepted now, for no documented reason", head);
  } else if (!loose.bare_lf) {
    Fail(totals, "parsed differently", head);
  }
}

void Expect(Totals& totals, const std::string& head, bool accepted,
            const Headers& headers = {}) {
  ++totals.heads;
  const Outcome now = New(head);
  if (now.accepted != accepted || (accepted && now.headers != headers))
    Fail(totals, accepted ? "expected acceptance" : "expected rejection", head);
}

std::string Head(std::string_view line,
                 std::initializer_list<std::string_view> fields = {"Host: a"}) {
  std::string head(line);
  head += "\r\n";
  for (const std::string_view field : fields) {
    head += field;
    head += "\r\n";
  }
  return head + "\r\n";
}

void TestHeads(Totals& totals) {
  // The documented differences, one by one.
  for (const char* line :
       {"GET  / HTTP/1.1", "GET\t/ HTTP/1.1", "GET /\tHTTP/1.1",
        "GET\v/ HTTP/1.1", " GET / HTTP/1.1", "GET / HTTP/1.1 ",
        "G@T / HTTP/1.1", "GE(T / HTTP/1.1",
        "GET /a\x01"
        "b HTTP/1.1",
        "GET /a\x7f HTTP/1.1", "GET /\xe9 HTTP/1.1", "GET /\xc3 HTTP/1.1"}) {
    CompareHead(totals, Head(line));
    Expect(totals, Head(line), false);
  }
  using std::string_view_literals::operator""sv;
  for (const std::string_view value :
       {"a\x01"
        "b"sv,
        "a\x7f"sv, "\va\v"sv, "\fa"sv, "a\rb"sv, "a\0b"sv}) {
    const std::string field = std::string("X: ").append(value);
    CompareHead(totals, Head("GET / HTTP/1.1", {field}));
    Expect(totals, Head("GET / HTTP/1.1", {field}), false);
  }
  Expect(totals, "GET / HTTP/1.1\nHost: a\nX: b\n\n", true,
         {{"Host", "a"}, {"X", "b"}});
  Expect(totals, "\r\nGET / HTTP/1.1\r\nHost: a\r\n\r\n", true,
         {{"Host", "a"}});
  CompareHead(totals, "GET / HTTP/1.1\nHost: a\r\n\r\n");
  CompareHead(totals, "GET / HTTP/1.1\r\nHost: a\nb\r\n\r\n");

  // Strict heads: lines, versions, fields, framing and many fields.
  for (const char* line :
       {"GET / HTTP/1.1", "POST /v1/chat/completions?stream=true HTTP/1.1",
        "GET /a?b=c&d HTTP/1.0", "OPTIONS /x HTTP/1.1", "get / HTTP/1.1",
        "M-SEARCH /a#f HTTP/1.1", "GET * HTTP/1.1", "GET http://h/ HTTP/1.1",
        "GET / HTTP/2", "GET / http/1.1", "GET / HTTP/1.10", "GET /",
        "GET / HTTP/1.1 x", "GET /\"<>{}|\\^` HTTP/1.1",
        "GET /\xc3\xa9?q=\xe2\x82\xac HTTP/1.1"}) {
    CompareHead(totals, Head(line));
  }
  for (const char* field : {"X: v",
                            "X:v",
                            "X:   v   ",
                            "X:\tv\t",
                            "X:",
                            "X: a:b",
                            "X: a  b",
                            "X: \xc3\xa9",
                            "x_y: 1",
                            "a!#$%&'*+-.^_`|~z: 1",
                            "X Y: v",
                            "X : v",
                            "X\t: v",
                            ": v",
                            "X@Y: v",
                            "X\xc3\xa9: v",
                            "no colon",
                            "Content-Length: 5",
                            "Content-Length: 05",
                            "Content-Length: +5",
                            "Content-Length: -1",
                            "Content-Length: 0x5",
                            "Content-Length:",
                            "Content-Length: 99999999999999999999",
                            "Transfer-Encoding: chunked",
                            "Expect: 100-continue",
                            "expect: 100-Continue",
                            "Expect: other"}) {
    CompareHead(totals, Head("POST / HTTP/1.1", {"Host: a", field}));
  }
  CompareHead(totals, Head("POST / HTTP/1.1",
                           {"Content-Length: 5", "content-length: 5"}));
  CompareHead(totals, Head("POST / HTTP/1.1",
                           {"Content-Length: 5", "Content-Length: 6"}));
  CompareHead(totals, Head("GET / HTTP/1.1", {"X: a", " b"}));
  CompareHead(totals, Head("GET / HTTP/1.1", {"X: a", "\tb"}));
  CompareHead(totals, "GET / HTTP/1.1\r\n\r\n");
  // Up to 100 fields parse identically; more are rejected (431).
  for (const int count : {99, 100, 101, 1000}) {
    std::string many = "GET / HTTP/1.1\r\n";
    Headers fields;
    for (int i = 0; i < count; ++i) {
      many += "X" + std::to_string(i) + ": v\r\n";
      fields.emplace_back("X" + std::to_string(i), "v");
    }
    CompareHead(totals, many + "\r\n");
    Expect(totals, many + "\r\n", count <= 100, fields);
  }

  // Random mutations of a typical request.
  const std::string base =
      Head("POST /v1/chat/completions?stream=true HTTP/1.1",
           {"Host: localhost:8080", "Content-Type: application/json",
            "Content-Length: 12", "Authorization: Bearer x",
            "Expect: 100-continue"});
  const std::string body = base.substr(0, base.size() - 4);
  constexpr std::array<char, 15> kBytes{' ',    '\t',   '\r', '\n', ':',
                                        '/',    '?',    '%',  '\0', '\x01',
                                        '\x7f', '\xe9', 'A',  '-',  '"'};
  std::mt19937_64 random(9112);
  for (int i = 0; i < 6000; ++i) {
    std::string head = body;
    for (int edit = 1 + static_cast<int>(random() % 3); edit > 0; --edit) {
      const std::size_t at = random() % head.size();
      const char byte = random() % 4 == 0 ? static_cast<char>(random())
                                          : kBytes[random() % kBytes.size()];
      switch (random() % 3) {
        case 0:
          head[at] = byte;
          break;
        case 1:
          head.insert(head.begin() + static_cast<std::ptrdiff_t>(at), byte);
          break;
        default:
          head.erase(at, 1);
          break;
      }
    }
    CompareHead(totals, head + "\r\n\r\n");
  }
}

int QueryParamFailures() {
  int failures = 0;
  const std::string bytes = "ab=&%+4F1g\xc3";
  const std::array<std::string, 6> keys{"", "a", "a b", "A", "ab", "\x4f"};
  std::mt19937_64 random(3986);
  for (int i = 0; i < 20000; ++i) {
    std::string query(random() % 14, ' ');
    for (char& c : query)
      c = bytes[random() % bytes.size()];
    for (const std::string& key : keys) {
      std::string value;
      const GufoHttpBytesFn keep = [](void* opaque, const std::uint8_t* data,
                                      std::size_t len) {
        static_cast<std::string*>(opaque)->assign(
            reinterpret_cast<const char*>(data), len);
      };
      const int found = gufo_http_query_param(
          reinterpret_cast<const std::uint8_t*>(query.data()), query.size(),
          reinterpret_cast<const std::uint8_t*>(key.data()), key.size(), keep,
          &value);
      if (found < 0 || value != ref::QueryParam(query, key)) {
        if (++failures <= 5)
          std::fprintf(stderr, "query \"%s\", key \"%s\": \"%s\"\n",
                       Hex(query).c_str(), Hex(key).c_str(),
                       Hex(value).c_str());
      }
    }
  }
  return failures;
}

int TokenFailures() {
  int failures = 0;
  const std::string bytes = " \t,upgradUPGws\v";
  std::mt19937_64 random(6455);
  for (int i = 0; i < 20000; ++i) {
    std::string value(random() % 16, ' ');
    for (char& c : value)
      c = bytes[random() % bytes.size()];
    if (i % 3 == 0)
      value.insert(random() % (value.size() + 1), "Upgrade");
    for (const std::string_view token : {"upgrade", "websocket", "up"}) {
      const int found = gufo_http_has_token(
          reinterpret_cast<const std::uint8_t*>(value.data()), value.size(),
          reinterpret_cast<const std::uint8_t*>(token.data()), token.size());
      if (found < 0 || (found != 0) != ref::HasToken(value, token)) {
        if (++failures <= 5)
          std::fprintf(stderr, "token \"%s\" in \"%s\": %d\n",
                       std::string(token).c_str(), Hex(value).c_str(), found);
      }
    }
  }
  return failures;
}

}  // namespace

int main() {
  try {
    Totals totals;
    TestHeads(totals);
    const int query_failures = QueryParamFailures();
    const int token_failures = TokenFailures();
    std::printf(
        "http requests: %zu heads, %zu with documented differences, %zu "
        "failures; query parameters %d and token lists %d mismatches\n",
        totals.heads, totals.differing, totals.failures, query_failures,
        token_failures);
    return totals.failures == 0 && query_failures == 0 && token_failures == 0
               ? 0
               : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
