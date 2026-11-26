// Expected results of the Rust JSON Schema compiler through its C++ callers:
// JsonSchemaRegex (with ICU's sets), the string and number lexemes,
// IntersectMultipleOf and Format. Each pattern or schema compiles and matches
// the listed texts, or fails with its exception type and message.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/json_schema_lexeme.hpp"
#include "src/core/json_schema_regex.hpp"

namespace {

using gufo::json::Value;
using gufo::sampling::JsonSchemaEmpty;
using gufo::sampling::JsonSchemaLexeme;
using gufo::sampling::JsonSchemaRegex;

constexpr std::uint32_t kAny = UINT32_MAX;
constexpr std::string_view kOk = "ok";

int failures = 0;

std::string Show(std::string_view text) {
  std::string out;
  for (const unsigned char c : text.substr(0, 80)) {
    if (c >= 0x20 && c < 0x7f && c != '\\') {
      out += static_cast<char>(c);
    } else {
      char buffer[8];
      std::snprintf(buffer, sizeof(buffer), "\\x%02x", c);
      out += buffer;
    }
  }
  if (text.size() > 80)
    out += "... (" + std::to_string(text.size()) + " bytes)";
  return out;
}

void Fail(const std::string& what) {
  ++failures;
  std::fprintf(stderr, "json_schema_test: %s\n", what.c_str());
}

// "ok", or which exception the call threw and its message.
template<class F>
std::string Run(F&& call) {
  try {
    call();
    return std::string(kOk);
  } catch (const JsonSchemaEmpty& e) {
    return std::string("empty: ") + e.what();
  } catch (const std::invalid_argument& e) {
    return std::string("invalid: ") + e.what();
  } catch (const std::exception& e) {
    return std::string("other: ") + e.what();
  }
}

std::string Invalid(std::string_view reason) {
  return "invalid: JSON Schema: " + std::string(reason);
}

std::string Empty(std::string_view reason) {
  return "empty: JSON Schema: " + std::string(reason);
}

// ---- Regex ----

std::vector<std::uint32_t> Codepoints(std::string_view utf8) {
  std::vector<std::uint32_t> out;
  for (std::size_t i = 0; i < utf8.size();) {
    const auto lead = static_cast<unsigned char>(utf8[i]);
    const std::size_t width = lead < 0x80   ? 1
                              : lead < 0xe0 ? 2
                              : lead < 0xf0 ? 3
                                            : 4;
    std::uint32_t cp = width == 1 ? lead : lead & (0x7f >> width);
    for (std::size_t k = 1; k < width; ++k)
      cp = (cp << 6) | (static_cast<unsigned char>(utf8[i + k]) & 0x3f);
    out.push_back(cp);
    i += width;
  }
  return out;
}

// 'y' if the regex accepts `text`, 'p' if it is a live prefix, 'd' if a
// codepoint leads to the dead state.
char Match(const JsonSchemaRegex& regex, std::string_view text) {
  auto state = regex.Start();
  for (const auto cp : Codepoints(text)) {
    state = regex.Advance(state, cp);
    if (state == JsonSchemaRegex::kDead)
      return 'd';
  }
  return regex.Accepting(state) ? 'y' : 'p';
}

using Texts = std::vector<std::pair<std::string_view, char>>;

std::shared_ptr<const JsonSchemaRegex> Regex(
    const std::vector<std::string>& patterns, std::uint32_t maximum,
    std::string_view expected, Texts texts = {}) {
  std::string name = "regex";
  for (const auto& pattern : patterns)
    name += " /" + Show(pattern) + "/";
  name += " max " + std::to_string(maximum);
  std::shared_ptr<const JsonSchemaRegex> regex;
  const auto got =
      Run([&] { regex = JsonSchemaRegex::Compile(patterns, maximum); });
  if (got != expected) {
    Fail(name + ": " + got + ", expected " + std::string(expected));
    return nullptr;
  }
  for (const auto& [text, want] : texts)
    if (const char result = regex ? Match(*regex, text) : '-'; result != want)
      Fail(name + " on \"" + Show(text) + "\": " + result + ", expected " +
           want);
  return regex;
}

std::shared_ptr<const JsonSchemaRegex> Anchored(std::string_view atom,
                                                Texts texts) {
  return Regex({"^" + std::string(atom) + "$"}, kAny, kOk, texts);
}

// Classes and properties as ECMAScript's Unicode mode defines them, with
// ICU's sets.
void Classes() {
  Anchored("\\d", {{"", 'p'}, {"5", 'y'}, {"\xd9\xa3", 'd'}, {"a", 'd'}});
  Anchored("\\D", {{"5", 'd'}, {"\xd9\xa3", 'y'}, {"a", 'y'}});
  Anchored("\\w", {{"_", 'y'}, {"Z", 'y'}, {"\xc3\xa9", 'd'}});
  Anchored("\\W", {{"_", 'd'}, {"\xc3\xa9", 'y'}});
  Anchored("\\s", {{" ", 'y'},
                   {"\t", 'y'},
                   {"\n", 'y'},
                   {"\xc2\xa0", 'y'},
                   {"\xef\xbb\xbf", 'y'},
                   {"\xe2\x80\xa8", 'y'},
                   {"\xc2\x85", 'd'},
                   {"a", 'd'}});
  Anchored("\\S", {{"\xc2\x85", 'y'}, {" ", 'd'}, {"\xf0\x9f\x98\x80", 'y'}});
  // The dot excludes line terminators only.
  Anchored(".", {{"\t", 'y'},
                 {"\xc2\x85", 'y'},
                 {"\xf0\x9f\x98\x80", 'y'},
                 {"\n", 'd'},
                 {"\r", 'd'},
                 {"\xe2\x80\xa8", 'd'}});
  Anchored("[^x]", {{"x", 'd'}, {"\n", 'y'}});
  Anchored("[]", {{"", 'p'}, {"a", 'd'}});
  Anchored("[^]", {{"\n", 'y'}, {"\xf0\x9f\x98\x80", 'y'}});
  Anchored("[\\s\\S]", {{"\n", 'y'}});
  Anchored("\\p{L}", {{"\xc3\xa9", 'y'}, {"\xce\xb1", 'y'}, {"5", 'd'}});
  Anchored("\\p{Lu}", {{"A", 'y'}, {"a", 'd'}});
  Anchored("\\P{N}", {{"a", 'y'}, {"5", 'd'}, {"\xd9\xa3", 'd'}});
  Anchored("\\p{Script=Greek}", {{"\xce\xb1", 'y'}, {"a", 'd'}});
  // Escapes and literals.
  Anchored("\\u00e9", {{"\xc3\xa9", 'y'}});
  Anchored("\xc3\xa9", {{"\xc3\xa9", 'y'}});
  Anchored("\\u{1F600}", {{"\xf0\x9f\x98\x80", 'y'}});
  Anchored("\\ud83d\\ude00", {{"\xf0\x9f\x98\x80", 'y'}});
  Anchored("\\u{10FFFF}", {{"\xf4\x8f\xbf\xbf", 'y'}});
  Anchored("\\x41", {{"A", 'y'}});
  Anchored("\\cJ", {{"\n", 'y'}});
  Anchored("\\0", {{std::string_view("\0", 1), 'y'}});
  Anchored("[\\b]", {{"\b", 'y'}});
  Anchored("\\.", {{".", 'y'}, {"a", 'd'}});
  Anchored("\\/", {{"/", 'y'}});
  Anchored("[\\d-]", {{"-", 'y'}, {"5", 'y'}});
  Anchored("[a\\-z]", {{"-", 'y'}, {"z", 'y'}, {"b", 'd'}});
  Anchored("[a-]", {{"-", 'y'}, {"a", 'y'}});
  Anchored("(?:ab)", {{"a", 'p'}, {"ab", 'y'}});
  // Assertions.
  Anchored("\\b", {{"", 'p'}});
  Anchored("\\B", {{"", 'y'}});
  Anchored("^", {{"", 'y'}});
  Anchored("(?=a)", {{"", 'p'}});
  Anchored("(?!b)", {{"", 'y'}});
  Regex({"^(?=a){2}"}, kAny, kOk, {{"a", 'y'}, {"b", 'd'}});
  // Unanchored patterns match anywhere; every pattern must hold.
  Regex({"a"}, kAny, kOk, {{"xax", 'y'}, {"b", 'p'}});
  Regex({"a", "b"}, 4, kOk,
        {{"a", 'p'}, {"ab", 'y'}, {"ba", 'y'}, {"xaxb", 'y'}});
  Regex({"^a", "b$"}, kAny, kOk, {{"ab", 'y'}, {"axb", 'y'}, {"ba", 'd'}});
  Regex({}, kAny, kOk, {{"", 'y'}, {"\xf4\x8f\xbf\xbf", 'y'}});
}

void Fixtures() {
  Anchored("(?!.*--)[a-z0-9-]{3,12}",
           {{"ab", 'p'}, {"abc-def", 'y'}, {"abc--def", 'd'}});
  Anchored("(?=.*\\d)(?=.*[A-Z]).{8,}",
           {{"Passw0rdX", 'y'}, {"password", 'p'}});
  Regex({"\\bword\\b"}, kAny, kOk, {{"a word.", 'y'}, {"sword", 'p'}});
  Regex({"(?:a|ab)(?:c|bcd)d"}, kAny, kOk,
        {{"abc", 'p'}, {"abcd", 'y'}, {"abbcdd", 'y'}, {"acd", 'y'}});
  Anchored("(?:(?:25[0-5]|2[0-4]\\d|1?\\d?\\d)\\.){3}\\d{1,3}",
           {{"255.255.255.255", 'y'}, {"256.1.1.1", 'd'}});
  Anchored("\\$\\d+(?:\\.\\d\\d)?",
           {{"$5", 'y'}, {"$5.2", 'p'}, {"$5.25", 'y'}});
  Anchored("(?:(?:a{0,3}b){0,4}c)+",
           {{"ab", 'p'}, {"bbbc", 'y'}, {"aaabaaabaaabaaabc", 'y'}});
  Anchored("\\p{Lu}\\p{Ll}*(?: \\p{Lu}\\p{Ll}*)*",
           {{"Hello World", 'y'}, {"hello", 'd'}});
  Anchored("[\\u0400-\\u04ff\\s]+",
           {{"\xd0\x9f\xd1\x80 \xd0\xb8", 'y'}, {"a", 'd'}});
}

// Sets of strings match no codepoint, but stay distinct from the empty set.
void StringProperties() {
  Regex({"\\p{RGI_Emoji_Flag_Sequence}"}, kAny, kOk,
        {{"", 'p'}, {"a", 'd'}, {"\xf0\x9f\x87\xae\xf0\x9f\x87\xb9", 'd'}});
  Anchored("\\p{Emoji_Keycap_Sequence}",
           {{"", 'p'}, {"1\xef\xb8\x8f\xe2\x83\xa3", 'd'}});
  Anchored("[^\\p{RGI_Emoji_Flag_Sequence}]",
           {{"", 'p'}, {"a", 'y'}, {"\xe2\x8c\x9a", 'y'}});
  Regex({"[^\\p{RGI_Emoji_Flag_Sequence}]*a[^\\p{RGI_Emoji_Flag_Sequence}]*"},
        3, kOk, {{"x", 'p'}, {"aaaa", 'y'}});
  Anchored("[\\p{RGI_Emoji}a]+",
           {{"\xe2\x8c\x9a", 'y'}, {"aaaa", 'y'}, {"1\xef\xb8\x8f", 'd'}});
  Anchored("(?:[\\p{Basic_Emoji}]|x)*",
           {{"", 'y'}, {"xx", 'y'}, {"\xe2\x8c\x9a", 'y'}});
  Regex({"\\P{RGI_Emoji}"}, kAny, kOk, {{"a", 'y'}, {"\xe2\x8c\x9a", 'p'}});
  for (const char* pattern :
       {"[\\p{RGI_Emoji_Flag_Sequence}-a]", "[a-\\p{Emoji_Keycap_Sequence}]"})
    Regex({pattern}, kAny, Invalid("invalid regex character range"));
}

// Ill-formed UTF-8 in a pattern decodes to U+FFFD per maximal subpart, as
// in ICU.
void IllFormed() {
  constexpr std::string_view kReplacement = "\xef\xbf\xbd";
  const auto times = [&](int n) {
    std::string text;
    for (int i = 0; i < n; ++i)
      text += kReplacement;
    return text;
  };
  for (const auto& [pattern, count] :
       std::initializer_list<std::pair<const char*, int>>{
           {"\xff", 1},
           {"\xe2\x82", 1},
           {"[\xf0\x9f]", 1},
           {"\xc0\xaf", 2},
           {"\xed\xa0\x80", 3}}) {
    const auto accepted = times(count);
    Anchored(pattern, {{accepted, 'y'}});
    if (count > 1)
      Anchored(pattern, {{times(count - 1), 'p'}});
  }
  Anchored("a\xc3", {{"a" + std::string(kReplacement), 'y'}});
}

// Rejections beyond the crate's own tests, as the C++ exception and message;
// \p{...} goes through ICU.
void Rejections() {
  for (const auto& [pattern, reason] :
       std::initializer_list<std::pair<const char*, const char*>>{
           {"(", "truncated regex"},
           {"[", "truncated regex"},
           {"\\", "truncated regex"},
           {"\\p{", "truncated regex"},
           {"a**", "regex repetition has no preceding atom"},
           {"a{", "invalid regex repetition"},
           {"a{,2}", "invalid regex repetition"},
           {"\\k", "unsupported regex escape or backreference"},
           {"\\-", "unsupported regex escape or backreference"},
           {"(?<n>a)", "unsupported regex group, lookbehind or inline flag"},
           {"(?i)a", "unsupported regex group, lookbehind or inline flag"},
           {"\\p{Nope}", "invalid regex Unicode property"},
           {"\\p{Script=Nope}", "invalid regex Unicode property"},
           {"[z-a]", "invalid regex character range"},
           {"\\ud800$", "unpaired surrogate in regex"},
           {"[^\\ud83d]", "unpaired surrogate in regex"},
           {"\\x4$", "invalid regex Unicode escape"},
           {"(\\b)+",
            "regex assertions in unbounded or large repetitions are "
            "unsupported"},
           {"(?=a)*",
            "regex assertions in unbounded or large repetitions are "
            "unsupported"},
           {"^(?:x(?=y)|y(?!x))+$",
            "regex assertions in unbounded or large "
            "repetitions are unsupported"},
       })
    Regex({pattern}, kAny, Invalid(reason));
}

void Limits() {
  std::string empty_groups;
  for (int i = 0; i < 4096; ++i)
    empty_groups += "(?:)";
  Regex({empty_groups}, kAny, kOk, {{"", 'y'}});
  Regex({empty_groups + "a"}, kAny, Invalid("regex exceeds 16384 bytes"));
  Regex({std::string(32, '(') + "a" + std::string(32, ')')}, kAny, kOk,
        {{"a", 'y'}});
  Regex({std::string(33, '(') + std::string(33, ')')}, kAny,
        Invalid("regex nesting exceeds 32 levels"));
  Regex({"a{4294967294}", "a{4294967295}"}, kAny,
        Invalid("regex repetition is too large"));
  Regex({"^[ab]*a[ab]{12}$"}, kAny,
        Invalid("compiled regex exceeds the state budget"));
  // Under a bound the same language fits.
  Regex({"^[ab]*a[ab]{12}$"}, 12, kOk);
  Regex({"(?:(?:(?:a|b|c|d){1,20}x){1,20}y){1,20}"}, kAny, kOk);
}

// The bound caps repetitions and MaximumSuffix; the lexeme counts the rest.
void Bounds() {
  struct Case {
    const char* pattern;
    std::uint32_t maximum, suffix;
    Texts texts;
  };
  for (const auto& [pattern, maximum, suffix, texts] :
       std::initializer_list<Case>{
           {"^a*$", 0, 0, {{"", 'y'}, {"a", 'd'}}},
           {"^a*$", kAny, 0, {{"aaa", 'y'}}},
           {"^a*?$", kAny, 0, {{"aaa", 'y'}}},
           {"^a+$", 0, 0, {{"a", 'd'}}},
           {"^a+$", 2, 1, {{"aaa", 'y'}}},
           {"^a?$", 0, 0, {{"", 'y'}, {"a", 'd'}}},
           {"^a{0}$", kAny, 0, {{"", 'y'}, {"a", 'd'}}},
           {"^a{2}$", 2, 2, {{"a", 'p'}, {"aa", 'y'}}},
           {"^a{1,3}$", 2, 1, {{"aa", 'y'}, {"aaa", 'd'}}},
           {"^a{1,3}$", kAny, 1, {{"aaa", 'y'}}},
           {"^a{2,}$", 2, 2, {{"aa", 'y'}, {"aaa", 'd'}}},
           {"^a{2,}$", kAny, 2, {{"aaa", 'y'}}},
           {"^a{3}$", 3, 3, {{"aaa", 'y'}}},
           {"^a{0,5}$", 3, 0, {{"aaa", 'y'}, {"aaaa", 'd'}}},
           {"^a{33}$", 2, 0, {{"a", 'd'}}},
           {"^a{33}$", kAny, 33, {{"a", 'p'}}},
           // States first reached at the bound have no transitions.
           {"^(?:ab|ba)*$", 1, 0, {{"", 'y'}, {"ab", 'd'}}},
           {"^[ab]*a[ab]{8}$", kAny, 9, {{"ab", 'p'}}},
           {"^[ab]*a[ab]{12}$", 12, 0, {}},
           {"(?:(?:(?:a|b|c|d){1,20}x){1,20}y){1,20}", 12, 4, {}},
           {"(?:(?:(?:a|b|c|d){1,20}x){1,20}y){1,20}", kAny, 3, {}},
       }) {
    const auto regex = Regex({pattern}, maximum, kOk, texts);
    if (regex && regex->MaximumSuffix() != suffix)
      Fail(std::string(pattern) + " max " + std::to_string(maximum) +
           ": MaximumSuffix " + std::to_string(regex->MaximumSuffix()));
  }
}

// CanFinish and CanAdvance from the start state and after "a", including
// lengths far beyond the states (the cycle skip).
void Lengths() {
  struct Finish {
    std::uint32_t low, high;
    bool expected;
  };
  struct Step {
    std::uint32_t first, last, low, high;
    bool expected;
  };
  struct Case {
    const char* pattern;
    std::vector<Finish> start, after_a;
    std::vector<Step> steps;
  };
  constexpr std::uint32_t kAll = 0x10ffff;
  for (const auto& [pattern, start, after_a, steps] :
       std::initializer_list<Case>{
           {"^(?:abc)*$",
            {{0, 0, true},
             {2, 2, false},
             {1, 40, true},
             {1000003, kAny, true},
             {1000003, 1000003, false},
             {4294967290U, kAny, true},
             {kAny, kAny, true},
             {999999937, 999999942, true},
             {999999937, 999999938, false}},
            {{0, 0, false}, {2, 2, true}, {1000001, kAny, true}},
            {{0, kAll, 1, 1, false},
             {0, kAll, 2, 2, true},
             {'a', 'a', 2, 2, true},
             {'b', 'c', 2, 2, false},
             {0, kAll, 1000003, kAny, true},
             {0, kAll, 1000002, 1000002, false},
             {5, 4, 2, 2, false}}},
           {"^(?:ab|ba)*$",
            {{2, 2, true},
             {1000003, 1000003, false},
             {4294967290U, kAny, true},
             {kAny, kAny, false},
             {999999937, 999999938, true}},
            {{0, 0, false}, {2, 2, false}, {1000001, kAny, true}},
            {{'b', 'c', 1, 1, true},
             {0, kAll, 2, 2, false},
             {0, kAll, 1000002, 1000002, false}}},
           {"^[a-c]{2,5}$",
            {{0, 0, false},
             {0, 3, true},
             {7, 1000, false},
             {1000003, kAny, false}},
            {{0, 0, false}, {2, 2, true}, {1000001, kAny, false}},
            {{0, kAll, 0, 0, false},
             {0, kAll, 3, 3, true},
             {'0', '9', 1, 1, false},
             {0, kAll, 1000003, kAny, false}}},
           {"^(?:a|bc)+$",
            {{0, 0, false}, {1000003, 1000003, true}},
            {{0, 0, true}},
            {{'b', 'c', 0, 0, false}, {'b', 'c', 1, 1, true}}},
       }) {
    const auto regex = Regex({pattern}, kAny, kOk);
    if (!regex)
      continue;
    const auto a = regex->Advance(regex->Start(), 'a');
    for (const bool after : {false, true})
      for (const auto& [low, high, expected] : after ? after_a : start)
        if (regex->CanFinish(after ? a : regex->Start(), low, high) != expected)
          Fail(std::string(pattern) + (after ? " after a" : "") +
               ": CanFinish(" + std::to_string(low) + ", " +
               std::to_string(high) + ") != " + (expected ? "true" : "false"));
    for (const auto& [first, last, low, high, expected] : steps)
      if (regex->CanAdvance(regex->Start(), first, last, low, high) != expected)
        Fail(std::string(pattern) + ": CanAdvance(" + std::to_string(first) +
             "-" + std::to_string(last) + ", " + std::to_string(low) + ", " +
             std::to_string(high) + ") != " + (expected ? "true" : "false"));
  }
  // Surrogates and codepoints past U+10FFFF are dead; states are checked.
  const auto regex = Regex({"^[^]*$"}, kAny, kOk);
  if (regex->Advance(0, 0xd800) != JsonSchemaRegex::kDead ||
      regex->Advance(0, 0xdfff) != JsonSchemaRegex::kDead ||
      regex->Advance(0, 0x110000) != JsonSchemaRegex::kDead ||
      regex->Advance(0, 0x10ffff) == JsonSchemaRegex::kDead)
    Fail("^[^]*$: Advance outside the scalars");
  if (const auto got = Run([&] { (void)regex->Accepting(99999); });
      got != "other: JSON Schema regex state out of range")
    Fail("Accepting(99999): " + got);
}

// ---- String lexemes ----

const char* Name(JsonSchemaLexeme::Match match) {
  return match.prefix ? (match.complete ? "PC" : "P")
                      : (match.complete ? "C" : "-");
}

using Checks = std::vector<std::pair<std::string_view, const char*>>;

void CheckAll(const std::string& name, const JsonSchemaLexeme& lexeme,
              Checks checks) {
  for (const auto& [text, expected] : checks)
    if (const auto got = Name(lexeme.Check(text));
        std::string_view(got) != expected)
      Fail(name + " on \"" + Show(text) + "\": " + got + ", expected " +
           expected);
}

std::shared_ptr<const JsonSchemaLexeme> String(std::string_view schema,
                                               std::string_view expected,
                                               Checks checks = {}) {
  std::shared_ptr<const JsonSchemaLexeme> lexeme;
  const auto got = Run(
      [&] { lexeme = JsonSchemaLexeme::String(gufo::json::parse(schema)); });
  const auto name = "string " + std::string(schema);
  if (got != expected) {
    Fail(name + ": " + got + ", expected " + std::string(expected));
    return nullptr;
  }
  if (lexeme)
    CheckAll(name, *lexeme, checks);
  return lexeme;
}

void Strings() {
  // Escapes count as the character they spell; ill-formed UTF-8, unpaired
  // surrogates and control bytes are dead.
  String("{}", kOk,
         {{"\"", "P"},
          {"\"\"", "C"},
          {"\"a\"", "C"},
          {"\"\\u00e9\\u00E9\xc3\xa9\"", "C"},
          {"\"\\ud83d\\ude00\"", "C"},
          {"\"\\u0000\\n\\/\\\"\"", "C"},
          {"\"\xe0\xa0", "P"},
          {"\"\\ud83d\"", "-"},
          {"\"\\ud83d\\u0041\"", "-"},
          {"\"\\udc00\"", "-"},
          {"\"\\x\"", "-"},
          {"\"\x01\"", "-"},
          {"\"\xff\"", "-"},
          {"\"\xc0\xaf\"", "-"},
          {"\"\xed\xa0\x80\"", "-"},
          {"x", "-"},
          {"\"a\"b", "-"}});
  // Lengths count codepoints.
  String(R"({"minLength":2})", kOk,
         {{"\"a\"", "-"},
          {"\"ab\"", "C"},
          {"\"\xc3\xa9\"", "-"},
          {"\"\\ud83d\\ude00\"", "-"},
          {"\"\\u0000\"", "-"}});
  String(R"({"maxLength":3})", kOk,
         {{"\"abc\"", "C"}, {"\"abcd\"", "-"}, {"\"\xf0\x9f\x98\x80\"", "C"}});
  String(R"({"minLength":1,"maxLength":1})", kOk,
         {{"\"\"", "-"}, {"\"\\ud83d\\ude00\"", "C"}, {"\"ab\"", "-"}});
  String(R"({"maxLength":1048576})", kOk, {{"\"a\"", "C"}});
  String(R"({"minLength":1048576})", kOk, {{"\"a", "P"}, {"\"a\"", "-"}});
  // An unanchored pattern keeps any prefix alive; an anchored one does not.
  String(R"({"pattern":"\\d"})", kOk,
         {{"\"ab\"", "-"}, {"\"1.2.3.4\"", "C"}, {"\"\xe0\xa0", "P"}});
  String(R"({"pattern":"^[a-z]+$"})", kOk,
         {{"\"\"", "-"},
          {"\"abc\"", "C"},
          {"\"\xc3\xa9\"", "-"},
          {"\"\xe0\xa0", "-"}});
  String(R"({"pattern":"^\\p{L}{2,}$"})", kOk,
         {{"\"Ab\"", "C"}, {"\"a\"", "-"}, {"\"\xc3\xa9\"", "-"}});
  String(R"({"pattern":"^[\\u00e9a]$"})", kOk,
         {{"\"\xc3\xa9\"", "C"}, {"\"\\u00e9\"", "C"}, {"\"\\u00E9\"", "C"}});
  String(R"({"pattern":"^a","format":"uuid"})", kOk,
         {{"\"a", "P"}, {"\"ab\"", "-"}});
  String(R"({"format":"hostname","minLength":253})", kOk);
  constexpr auto kLengths =
      "string lengths must be nonnegative integers up to 1048576";
  constexpr auto kNoValue =
      "string predicates and lengths have no matching value";
  for (const auto& [schema, error] :
       std::initializer_list<std::pair<const char*, std::string>>{
           {R"({"minLength":4,"maxLength":2})",
            Empty("minLength exceeds maxLength")},
           {R"({"minLength":-1})", Invalid(kLengths)},
           {R"({"maxLength":1.5})", Invalid(kLengths)},
           {R"({"maxLength":1048577})", Invalid(kLengths)},
           {R"({"minLength":"2"})", Invalid(kLengths)},
           {R"({"minLength":null})", Invalid(kLengths)},
           {R"({"pattern":"("})", Invalid("truncated regex")},
           {R"({"pattern":5})", Invalid("pattern must be a string")},
           {R"({"format":"color"})", Invalid("unsupported string format")},
           {R"({"format":null})", Invalid("format must be a string")},
           {R"({"format":"hostname","minLength":254})", Empty(kNoValue)},
           {R"({"minLength":300,"format":"ipv4"})", Empty(kNoValue)},
           {R"({"pattern":"^[0-9]{3}$","format":"ipv4"})", Empty(kNoValue)},
           {R"({"pattern":"^a{3}$","maxLength":2})", Empty(kNoValue)},
       })
    String(schema, error);
}

// The canonical mask state forgets counts that cannot change a token's mask.
void MaskStates() {
  struct Case {
    const char* schema;
    std::string_view a;
    std::size_t a_bytes;
    std::string_view b;
    std::size_t b_bytes;
    bool same;
  };
  for (const auto& [schema, a, a_bytes, b, b_bytes, same] :
       std::initializer_list<Case>{
           // Without a pattern or bounds every count is the same, and
           // completed characters forget their spelling.
           {"{}", "\"", 1, "\"ab", 1, true},
           {"{}", "\"\\u00e9", 1, "\"\xc3\xa9", 1, true},
           {"{}", "\"a", 1, "\"\xc3", 1, false},
           {"{}", "\"\\u00", 1, "\"\\u0", 1, false},
           // Far below minLength the count resets; within a token's reach
           // it stays.
           {R"({"minLength":10})", "\"a", 2, "\"ab", 2, true},
           {R"({"minLength":10})", "\"ab", 9, "\"abc", 9, false},
           {R"({"minLength":10})", "\"abcdefghijk", 1, "\"abcdefghijkl", 1,
            true},
           // Far below maxLength it reaches the minimum.
           {R"({"maxLength":10})", "\"a", 2, "\"abcdefgh", 2, true},
           {R"({"maxLength":10})", "\"a", 2, "\"abcdefghi", 2, false},
           {R"({"pattern":"^a+$","minLength":3})", "\"aaa", 5, "\"aaaa", 5,
            true},
           {R"({"pattern":"^a+$","minLength":3})", "\"aa", 5, "\"aaa", 5,
            false},
       }) {
    const auto lexeme = JsonSchemaLexeme::String(gufo::json::parse(schema));
    const auto state = [&](std::string_view text, std::size_t bytes) {
      std::string encoded;
      for (const unsigned char byte : text)
        if (!lexeme->Advance(encoded, byte).prefix)
          Fail(std::string(schema) + ": \"" + Show(text) + "\" is dead");
      lexeme->CanonicalMaskState(encoded, bytes);
      return encoded;
    };
    if ((state(a, a_bytes) == state(b, b_bytes)) != same)
      Fail(std::string(schema) + ": \"" + Show(a) + "\" and \"" + Show(b) +
           (same ? "\" differ" : "\" coincide"));
  }
}

// ---- Number lexemes ----

void Number(std::string_view schema, bool integer, std::string_view expected,
            Checks checks = {},
            std::initializer_list<std::pair<const char*, bool>> values = {}) {
  std::shared_ptr<const JsonSchemaLexeme> lexeme;
  const auto got = Run([&] {
    lexeme = JsonSchemaLexeme::Number(gufo::json::parse(schema), integer);
  });
  const auto name =
      std::string(integer ? "integer " : "number ") + std::string(schema);
  if (got != expected)
    return Fail(name + ": " + got + ", expected " + std::string(expected));
  if (!lexeme)
    return;
  CheckAll(name, *lexeme, checks);
  for (const auto& [text, accepted] : values)
    if (lexeme->AcceptValue(gufo::json::parse(text)) != accepted)
      Fail(name + ": AcceptValue(" + text +
           ") != " + (accepted ? "true" : "false"));
}

void Numbers() {
  // Values are written as plain decimals: no exponent, no leading zeros, at
  // most 16 significant digits. AcceptValue takes any spelling.
  Number("{}", false, kOk,
         {{"", "P"},
          {"-", "P"},
          {"0", "PC"},
          {"-0", "PC"},
          {"00", "-"},
          {"01", "-"},
          {"1.", "P"},
          {"1.0", "PC"},
          {"-1.5", "PC"},
          {"999.999", "PC"},
          {"1e3", "-"},
          {"1E3", "-"},
          {"+1", "-"},
          {".5", "-"},
          {"--1", "-"},
          {"0.000000000000000000001", "-"}},
         {{"1e3", true}, {"1e-7", true}, {"-1e22", true}, {"\"1\"", false}});
  Number("{}", true, kOk,
         {{"0", "PC"}, {"-0", "PC"}, {"1000", "PC"}, {"1.", "-"}, {"1.0", "-"}},
         {{"1.0", true},
          {"1e1", true},
          {"-1e22", true},
          {"0.5", false},
          {"1e-7", false}});
  Number(R"({"minimum":1,"maximum":10,"multipleOf":0.5})", false, kOk,
         {{"-", "-"},
          {"0", "-"},
          {"0.5", "-"},
          {"1", "PC"},
          {"1.", "P"},
          {"1.0", "PC"},
          {"1.5", "PC"},
          {"10", "PC"},
          {"10.5", "-"},
          {"11", "-"}},
         {{"1e1", true}, {"1e3", false}, {"0.5", false}});
  Number(R"({"minimum":1,"maximum":10,"multipleOf":0.5})", true, kOk,
         {{"1.5", "-"}, {"2.5", "-"}, {"3", "PC"}});
  Number(R"({"exclusiveMinimum":-1.5,"exclusiveMaximum":2.5})", false, kOk,
         {{"-", "P"}, {"-1", "PC"}, {"-1.5", "-"}, {"2", "PC"}, {"2.5", "-"}},
         {{"1e-7", true}, {"-2.5", false}});
  Number(R"({"maximum":1e3,"multipleOf":3})", true, kOk,
         {{"-1", "P"}, {"1", "P"}, {"3", "PC"}, {"10", "P"}, {"1000", "-"}},
         {{"0", true}, {"1e3", false}});
  Number(R"({"minimum":1,"maximum":1})", false, kOk,
         {{"1", "PC"}, {"1.", "P"}, {"1.0", "PC"}, {"1.5", "-"}},
         {{"1.0", true}});
  Number(R"({"minimum":1,"exclusiveMinimum":1})", false, kOk,
         {{"1", "P"}, {"1.0", "P"}, {"1.5", "PC"}},
         {{"1", false}, {"10", true}});
  Number(R"({"minimum":1.1,"maximum":1.9})", false, kOk,
         {{"1", "P"}, {"1.0", "-"}, {"1.5", "PC"}, {"2", "-"}});
  Number(R"({"minimum":123.456,"maximum":123.456})", false, kOk,
         {{"1", "P"}, {"123.456", "PC"}});
  Number(R"({"multipleOf":0.3})", true, kOk,
         {{"1", "P"}, {"3", "PC"}, {"0.3", "-"}});
  Number(R"({"multipleOf":1e-3})", false, kOk, {{"123.456", "PC"}},
         {{"1e-7", false}});
  // Bounds and steps beyond the digit cap.
  Number(R"({"minimum":1e300,"multipleOf":1e-300})", false, kOk,
         {{"0", "-"}, {"1", "P"}, {"1000", "P"}}, {{"1e300", true}});
  Number(R"({"minimum":-1e-300,"maximum":1e-300,"multipleOf":1e-301})", false,
         kOk, {{"0", "PC"}, {"-0", "PC"}, {"1", "-"}});
  Number(R"({"minimum":0.30000000000000004,"maximum":0.4,"multipleOf":0.1})",
         false, kOk, {{"0", "P"}, {"0.3", "-"}, {"0.4", "PC"}});
  constexpr auto kInterval = "numeric constraints describe an empty interval";
  constexpr auto kNoMultiple =
      "numeric constraints contain no multipleOf value";
  for (const auto& [schema, integer, error] :
       std::initializer_list<std::tuple<const char*, bool, std::string>>{
           {R"({"multipleOf":0})", false,
            Invalid("multipleOf must be positive")},
           {R"({"multipleOf":-2})", false,
            Invalid("multipleOf must be positive")},
           {R"({"minimum":"1"})", false,
            Invalid("minimum must be a finite number")},
           {R"({"maximum":null})", false,
            Invalid("maximum must be a finite number")},
           {R"({"multipleOf":true})", false,
            Invalid("multipleOf must be a finite number")},
           {R"({"minimum":2,"maximum":1})", false, Empty(kInterval)},
           {R"({"minimum":1,"exclusiveMaximum":1})", false, Empty(kInterval)},
           {R"({"minimum":0.5,"maximum":0.5})", true, Empty(kInterval)},
           {R"({"minimum":1.1,"maximum":1.9})", true, Empty(kNoMultiple)},
           {R"({"exclusiveMinimum":0,"exclusiveMaximum":0.1,"multipleOf":0.1})",
            false, Empty(kNoMultiple)},
       })
    Number(schema, integer, error);
}

void MultipleOf() {
  for (const auto& [a, b, expected] :
       std::initializer_list<std::tuple<const char*, const char*, std::string>>{
           {"0.4", "0.6", "12e-1"},
           {"3", "5", "15e0"},
           {"3", "3", "3e0"},
           {"0.1", "0.25", "5e-1"},
           {"0.25", "0.1", "5e-1"},
           {"0.1", "0.3", "3e-1"},
           {"1e-3", "7.5", "75e-1"},
           {"123456789.125", "0.3", "14814814695e-1"},
           {"1e300", "7", "7e300"},
           {"1e300", "1e-300", "1e300"},
           {"1e-300", "1e-300", "1e-300"},
           {"0", "3", Invalid("multipleOf must be a positive finite number")},
           {"-2", "3", Invalid("multipleOf must be a positive finite number")},
           {"\"1\"", "3",
            Invalid("multipleOf must be a positive finite number")},
           // The common multiple is read back as JSON, which has no infinity.
           {"1e308", "3", "other: JSON parse error: bad number"},
           {"9007199254740991", "3",
            Invalid(
                "combined multipleOf exceeds the exact schema-number range")},
       }) {
    Value common;
    auto got = Run([&] {
      common = JsonSchemaLexeme::IntersectMultipleOf(gufo::json::parse(a),
                                                     gufo::json::parse(b));
    });
    if (got == kOk)
      got = common.dump(Value::Numbers::kAsParsed);
    if (got != expected)
      Fail(std::string("multipleOf ") + a + " and " + b + ": " + got +
           ", expected " + expected);
  }
}

void Formats() {
  const std::string long_label = "\"" + std::string(64, 'a') + ".com\"";
  for (const auto& [format, pattern, maximum] :
       std::initializer_list<std::tuple<const char*, const char*, int>>{
           {"uuid", "^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$",
            0},
           {"ipv4",
            "^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\\.){3}"
            "(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$",
            0},
           {"email", "^[^@\\s]+@[^@\\s]+$", 0},
           {"time",
            "^(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\\.[0-9]+)?"
            "(?:[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])$",
            0},
           {"hostname",
            "^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
            "(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\\.?$",
            253},
       }) {
    const auto schema = JsonSchemaLexeme::Format(format);
    const auto* length = schema.find("maxLength");
    if (schema.member_str("pattern") != pattern ||
        schema.members().size() != (maximum ? 2U : 1U) ||
        (maximum && (!length || length->as_size(0) != std::size_t(maximum))))
      Fail(std::string("format ") + format + ": " + schema.dump());
  }
  for (const char* format : {"uri", "", "Date"})
    if (const auto got = Run([&] { (void)JsonSchemaLexeme::Format(format); });
        got != Invalid("unsupported string format"))
      Fail(std::string("format \"") + format + "\": " + got);
  // The long formats, through the string lexeme.
  for (const auto& [format, checks] :
       std::initializer_list<std::pair<const char*, Checks>>{
           {"date",
            {{"\"2024-02-29\"", "C"},
             {"\"2000-02-29\"", "C"},
             {"\"2023-02-29\"", "-"},
             {"\"1900-02-29\"", "-"},
             {"\"2024-04-31\"", "-"},
             {"\"2024-13-01\"", "-"}}},
           {"date-time",
            {{"\"2023-12-31T00:00:00+01:00\"", "C"},
             {"\"2023-12-31t23:59:60.5z\"", "C"},
             {"\"2023-12-31 00:00:00Z\"", "-"}}},
           {"duration",
            {{"\"P1W\"", "C"},
             {"\"PT1.5S\"", "C"},
             {"\"P1Y2M3DT4H5M6S\"", "C"},
             {"\"P\"", "-"},
             {"\"PT\"", "-"},
             {"\"P1WT1H\"", "-"}}},
           {"ipv6",
            {{"\"::\"", "C"},
             {"\"::1\"", "C"},
             {"\"2001:db8::8a2e:370:7334\"", "C"},
             {"\"::ffff:192.0.2.1\"", "C"},
             {"\"1:2:3:4:5:6:7:8:9\"", "-"},
             {"\"1::2::3\"", "-"}}},
           {"hostname",
            {{"\"example.com.\"", "C"},
             {"\"-a.com\"", "-"},
             {long_label, "-"}}},
       })
    if (const auto lexeme =
            String(std::string(R"({"format":")") + format + "\"}", kOk))
      CheckAll(std::string("format ") + format, *lexeme, checks);
}

}  // namespace

int main() {
  Classes();
  Fixtures();
  StringProperties();
  IllFormed();
  Rejections();
  Limits();
  Bounds();
  Lengths();
  Strings();
  MaskStates();
  Numbers();
  MultipleOf();
  Formats();
  if (failures) {
    std::fprintf(stderr, "json_schema_test: %d failures\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("json_schema_test: expected results match\n");
  return EXIT_SUCCESS;
}
