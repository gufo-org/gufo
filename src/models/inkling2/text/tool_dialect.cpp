#include "src/models/inkling2/text/tool_dialect.hpp"

#include <cstdint>

#include "src/models/inkling2/text/json.hpp"

namespace inkling2::text::dialect {

namespace {

enum class R { kOk, kIncomplete, kBad };

bool StartsWithAt(std::string_view t, std::size_t p, std::string_view lit) {
  return t.size() - std::min(p, t.size()) >= lit.size() &&
         t.compare(p, lit.size(), lit) == 0;
}

// Matches `lit` at p. kIncomplete when the text ends while still a prefix of
// lit.
R Expect(std::string_view t, std::size_t* p, std::string_view lit) {
  const std::size_t avail = t.size() - *p;
  if (avail >= lit.size()) {
    if (t.compare(*p, lit.size(), lit) != 0)
      return R::kBad;
    *p += lit.size();
    return R::kOk;
  }
  return t.compare(*p, avail, lit.substr(0, avail)) == 0 ? R::kIncomplete
                                                         : R::kBad;
}

// Skips one JSON string starting at the opening quote.
R SkipString(std::string_view t, std::size_t* p) {
  std::size_t i = *p + 1;
  while (i < t.size()) {
    const char c = t[i];
    if (c == '"') {
      *p = i + 1;
      return R::kOk;
    }
    if (static_cast<unsigned char>(c) < 0x20)
      return R::kBad;
    if (c == '\\') {
      if (i + 1 >= t.size())
        return R::kIncomplete;
      if (t[i + 1] == 'u') {
        if (i + 6 > t.size())
          return R::kIncomplete;
        i += 6;
      } else {
        i += 2;
      }
    } else {
      ++i;
    }
  }
  return R::kIncomplete;
}

R SkipValue(std::string_view t, std::size_t* p, int depth = 0);

R SkipContainer(std::string_view t, std::size_t* p, char open, char close,
                int depth) {
  if (depth > 128)
    return R::kBad;
  std::size_t i = *p + 1;
  bool first = true;
  for (;;) {
    if (i >= t.size())
      return R::kIncomplete;
    if (t[i] == close) {
      *p = i + 1;
      return R::kOk;
    }
    if (!first) {
      if (t[i] != ',')
        return R::kBad;
      ++i;
      if (i >= t.size())
        return R::kIncomplete;
    }
    first = false;
    if (open == '{') {
      if (t[i] != '"')
        return R::kBad;
      if (const auto r = SkipString(t, &i); r != R::kOk)
        return r;
      if (i >= t.size())
        return R::kIncomplete;
      if (t[i] != ':')
        return R::kBad;
      ++i;
      if (i >= t.size())
        return R::kIncomplete;
    }
    if (const auto r = SkipValue(t, &i, depth + 1); r != R::kOk)
      return r;
  }
}

R SkipValue(std::string_view t, std::size_t* p, int depth) {
  if (*p >= t.size())
    return R::kIncomplete;
  const char c = t[*p];
  if (c == '"')
    return SkipString(t, p);
  if (c == '{')
    return SkipContainer(t, p, '{', '}', depth);
  if (c == '[')
    return SkipContainer(t, p, '[', ']', depth);
  for (const std::string_view lit :
       {std::string_view("true"), std::string_view("false"),
        std::string_view("null")}) {
    if (c == lit[0]) {
      const auto r = Expect(t, p, lit);
      return r;
    }
  }
  std::size_t i = *p;
  if (c != '-' && (c < '0' || c > '9'))
    return R::kBad;
  while (i < t.size() &&
         (t[i] == '-' || t[i] == '+' || t[i] == '.' || t[i] == 'e' ||
          t[i] == 'E' || (t[i] >= '0' && t[i] <= '9')))
    ++i;
  if (i >= t.size())
    return R::kIncomplete;  // the number may continue
  *p = i;
  return R::kOk;
}

// Decodes a (possibly truncated) JSON string body into UTF-8, holding back an
// incomplete trailing escape.
std::string DecodePartialString(std::string_view body) {
  std::string out;
  for (std::size_t i = 0; i < body.size();) {
    const char c = body[i];
    if (c != '\\') {
      out.push_back(c);
      ++i;
      continue;
    }
    if (i + 1 >= body.size())
      break;
    const char e = body[i + 1];
    if (e == 'u') {
      if (i + 6 > body.size())
        break;
      json::Value v;
      if (json::Parse("\"" + std::string(body.substr(i, 6)) + "\"", &v))
        out += v.str();
      i += 6;
      continue;
    }
    switch (e) {
      case 'n':
        out.push_back('\n');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      default:
        out.push_back(e);
        break;
    }
    i += 2;
  }
  return out;
}

enum class CallResult { kNotCall, kComplete, kIncomplete, kMalformed };

// Parses one call block starting at the header marker at p.
CallResult ParseCall(std::string_view t, std::size_t p, bool partial,
                     Call* call, std::size_t* end) {
  std::size_t i = p + kHeader.size();
  // header name up to the invoke token
  const std::size_t name_begin = i;
  while (i < t.size() && t[i] != '<')
    ++i;
  if (i >= t.size())
    return CallResult::kIncomplete;  // still in the name
  if (i == name_begin &&
      (StartsWithAt(t, i, kThinking) || StartsWithAt(t, i, kContentText)))
    return CallResult::kNotCall;
  if (i == name_begin && i + 1 < t.size() && t[i + 1] != '|')
    return CallResult::kNotCall;
  const std::string name(t.substr(name_begin, i - name_begin));
  if (name.empty()) {
    // "<|message_model|><|" + something else (or an incomplete control token)
    std::size_t q = i;
    const auto r = Expect(t, &q, kInvokeJson);
    return r == R::kIncomplete ? CallResult::kIncomplete : CallResult::kNotCall;
  }
  if (const auto r = Expect(t, &i, kInvokeJson); r != R::kOk)
    return r == R::kIncomplete ? CallResult::kIncomplete
                               : CallResult::kMalformed;
  const std::string quoted = json::QuoteString(name);
  if (const auto r = Expect(t, &i, "{\"name\":"); r != R::kOk)
    return r == R::kIncomplete ? CallResult::kIncomplete
                               : CallResult::kMalformed;
  if (const auto r = Expect(t, &i, quoted); r != R::kOk)
    return r == R::kIncomplete ? CallResult::kIncomplete
                               : CallResult::kMalformed;
  if (const auto r = Expect(t, &i, ",\"args\":"); r != R::kOk)
    return r == R::kIncomplete ? CallResult::kIncomplete
                               : CallResult::kMalformed;
  call->name = name;
  // args object
  if (i >= t.size())
    return CallResult::kIncomplete;
  if (t[i] != '{')
    return CallResult::kMalformed;
  ++i;
  bool first = true;
  for (;;) {
    if (i >= t.size())
      return CallResult::kIncomplete;
    if (t[i] == '}') {
      ++i;
      break;
    }
    if (!first) {
      if (t[i] != ',')
        return CallResult::kMalformed;
      ++i;
      if (i >= t.size())
        return CallResult::kIncomplete;
    }
    first = false;
    if (t[i] != '"')
      return CallResult::kMalformed;
    const std::size_t key_begin = i;
    if (const auto r = SkipString(t, &i); r != R::kOk)
      return r == R::kIncomplete ? CallResult::kIncomplete
                                 : CallResult::kMalformed;
    json::Value key;
    if (!json::Parse(t.substr(key_begin, i - key_begin), &key))
      return CallResult::kMalformed;
    if (i >= t.size())
      return CallResult::kIncomplete;
    if (t[i] != ':')
      return CallResult::kMalformed;
    ++i;
    if (i >= t.size())
      return CallResult::kIncomplete;
    const std::size_t v_begin = i;
    const auto r = SkipValue(t, &i);
    if (r == R::kBad)
      return CallResult::kMalformed;
    if (r == R::kIncomplete) {
      if (partial && t[v_begin] == '"') {
        call->unfinished = Argument{
            key.str(), DecodePartialString(t.substr(v_begin + 1)), true};
      }
      return CallResult::kIncomplete;
    }
    Argument a;
    a.name = key.str();
    if (t[v_begin] == '"') {
      json::Value s;
      if (!json::Parse(t.substr(v_begin, i - v_begin), &s))
        return CallResult::kMalformed;
      a.value = s.str();
      a.is_string = true;
    } else {
      a.value = std::string(t.substr(v_begin, i - v_begin));
      a.is_string = false;
    }
    call->arguments.push_back(std::move(a));
  }
  if (i >= t.size())
    return CallResult::kIncomplete;
  if (t[i] != '}')
    return CallResult::kMalformed;
  ++i;
  if (const auto r = Expect(t, &i, kEndMessage); r != R::kOk)
    return r == R::kIncomplete ? CallResult::kIncomplete
                               : CallResult::kMalformed;
  *end = i;
  call->complete = true;
  call->valid = true;
  return CallResult::kComplete;
}

}  // namespace

void ScanToolCalls(std::string_view text, bool partial,
                   std::vector<Call>* calls,
                   std::vector<std::pair<std::size_t, std::size_t>>* spans) {
  std::size_t p = text.find(kHeader);
  while (p != std::string_view::npos && p < text.size()) {
    Call call;
    std::size_t end = 0;
    const auto r = ParseCall(text, p, partial, &call, &end);
    if (r == CallResult::kComplete) {
      calls->push_back(std::move(call));
      spans->emplace_back(p, end);
      p = text.find(kHeader, end);
    } else if (r == CallResult::kIncomplete) {
      if (partial && !call.name.empty()) {
        call.complete = false;
        call.valid = true;
        calls->push_back(std::move(call));
      }
      return;
    } else if (r == CallResult::kNotCall) {
      // a thinking or text block: skip to its end
      const auto e = text.find(kEndMessage, p + kHeader.size());
      if (e == std::string_view::npos)
        return;
      p = text.find(kHeader, e + kEndMessage.size());
    } else {  // malformed: treat the header as text and move on
      p = text.find(kHeader, p + kHeader.size());
    }
  }
}

ParsedOutput ParseAssistantOutput(std::string_view text) {
  ParsedOutput out;
  std::size_t p = 0;
  const auto trim_end = [&] {
    if (StartsWithAt(text, p, kEndSampling))
      p = text.size();
  };
  while (p < text.size()) {
    trim_end();
    if (p >= text.size())
      break;
    if (!StartsWithAt(text, p, kHeader)) {
      // stray text outside any block is content
      const auto next = text.find(kHeader, p);
      const auto stop = next == std::string_view::npos ? text.size() : next;
      auto piece = text.substr(p, stop - p);
      if (const auto es = piece.find(kEndSampling);
          es != std::string_view::npos)
        piece = piece.substr(0, es);
      out.content.append(piece);
      p = stop;
      continue;
    }
    const std::size_t body = p + kHeader.size();
    if (StartsWithAt(text, body, kThinking) ||
        StartsWithAt(text, body, kContentText)) {
      const bool think = StartsWithAt(text, body, kThinking);
      const std::size_t start =
          body + (think ? kThinking.size() : kContentText.size());
      auto e = text.find(kEndMessage, start);
      std::string_view piece;
      if (e == std::string_view::npos) {
        piece = text.substr(start);
        if (const auto es = piece.find(kEndSampling);
            es != std::string_view::npos)
          piece = piece.substr(0, es);
        out.truncated = true;
        p = text.size();
      } else {
        piece = text.substr(start, e - start);
        p = e + kEndMessage.size();
      }
      (think ? out.reasoning : out.content).append(piece);
      continue;
    }
    std::vector<Call> calls;
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    ScanToolCalls(text.substr(p), false, &calls, &spans);
    if (!spans.empty() && spans.front().first == 0) {
      const auto& sp = spans.front();
      const std::string_view blk = text.substr(p, sp.second);
      // arguments exactly as written: between "args": and the closing "}" + end
      const auto a = blk.find(",\"args\":", blk.find(kInvokeJson));
      const auto args =
          blk.substr(a + 8, blk.size() - kEndMessage.size() - 1 - (a + 8));
      out.tool_calls.push_back({calls.front().name, std::string(args)});
      p += sp.second;
      continue;
    }
    // an unparsable block: keep it as content so nothing is lost
    out.truncated = true;
    out.content.append(text.substr(p));
    p = text.size();
  }
  return out;
}

}  // namespace inkling2::text::dialect
