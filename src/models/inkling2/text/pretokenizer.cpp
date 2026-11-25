#include "src/models/inkling2/text/pretokenizer.hpp"

#include <cstdint>

#include "src/models/inkling2/text/unicode.hpp"

namespace inkling2::text {

namespace {

using unicode::Category;
namespace uc = unicode;

struct Scanner {
  std::string_view s;
  std::vector<std::uint32_t> run;  // scratch: start offsets of a class run

  [[nodiscard]] bool More(std::size_t p) const { return p < s.size(); }
  [[nodiscard]] uc::Decoded At(std::size_t p) const {
    return uc::DecodeUtf8(s, p);
  }

  static bool IsLetter(std::uint8_t code) {
    const auto c = uc::CategoryOf(code);
    return c >= uc::kLu && c <= uc::kLl;
  }
  static bool IsUpperish(std::uint8_t code) {  // Lu Lt Lm Lo M
    const auto c = uc::CategoryOf(code);
    return c == uc::kLu || c == uc::kLt || c == uc::kLm || c == uc::kLo ||
           c == uc::kMark;
  }
  static bool IsLowerish(std::uint8_t code) {  // Ll Lm Lo M
    const auto c = uc::CategoryOf(code);
    return c == uc::kLl || c == uc::kLm || c == uc::kLo || c == uc::kMark;
  }
  static bool IsNumber(std::uint8_t code) {
    return uc::CategoryOf(code) == uc::kNumber;
  }
  static bool IsNewline(char32_t cp) { return cp == '\r' || cp == '\n'; }

  // (?i:'s|'t|'re|'ve|'m|'ll|'d)? at q; returns the end (q when absent).
  [[nodiscard]] std::size_t Contraction(std::size_t q) const {
    if (q >= s.size() || s[q] != '\'')
      return q;
    const auto lower = [&](std::size_t p) -> int {
      if (p >= s.size())
        return -1;
      const auto c = static_cast<unsigned char>(s[p]);
      return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    };
    const int a = lower(q + 1);
    if (a == 's')
      return q + 2;
    // U+017F LATIN SMALL LETTER LONG S case-folds to 's'.
    if (q + 2 < s.size() && static_cast<unsigned char>(s[q + 1]) == 0xC5 &&
        static_cast<unsigned char>(s[q + 2]) == 0xBF)
      return q + 3;
    if (a == 't')
      return q + 2;
    const int b = lower(q + 2);
    if (a == 'r' && b == 'e')
      return q + 3;
    if (a == 'v' && b == 'e')
      return q + 3;
    if (a == 'm')
      return q + 2;
    if (a == 'l' && b == 'l')
      return q + 3;
    if (a == 'd')
      return q + 2;
    return q;
  }

  // U* Lw+ C? starting at j; npos when it does not match.
  [[nodiscard]] std::size_t Body1(std::size_t j) {
    run.clear();
    std::size_t p = j;
    while (More(p)) {
      const auto d = At(p);
      if (!IsUpperish(uc::Lookup(d.cp)))
        break;
      run.push_back(static_cast<std::uint32_t>(p));
      p += d.len;
    }
    run.push_back(
        static_cast<std::uint32_t>(p));  // run[m]: first non-U position
    for (std::size_t k = run.size(); k-- > 0;) {
      std::size_t q = run[k];
      if (!More(q))
        continue;
      if (!IsLowerish(uc::Lookup(At(q).cp)))
        continue;
      while (More(q)) {
        const auto d = At(q);
        if (!IsLowerish(uc::Lookup(d.cp)))
          break;
        q += d.len;
      }
      return Contraction(q);
    }
    return std::string_view::npos;
  }

  // U+ Lw* C? starting at j.
  [[nodiscard]] std::size_t Body2(std::size_t j) const {
    std::size_t p = j;
    while (More(p)) {
      const auto d = At(p);
      if (!IsUpperish(uc::Lookup(d.cp)))
        break;
      p += d.len;
    }
    if (p == j)
      return std::string_view::npos;
    while (More(p)) {
      const auto d = At(p);
      if (!IsLowerish(uc::Lookup(d.cp)))
        break;
      p += d.len;
    }
    return Contraction(p);
  }

  [[nodiscard]] std::size_t Word(std::size_t i) {
    const auto d = At(i);
    const auto code = uc::Lookup(d.cp);
    const bool can_prefix =
        !IsNewline(d.cp) && !IsLetter(code) && !IsNumber(code);
    if (can_prefix) {
      const auto e = Body1(i + d.len);
      if (e != std::string_view::npos)
        return e;
    }
    auto e = Body1(i);
    if (e != std::string_view::npos)
      return e;
    if (can_prefix) {
      e = Body2(i + d.len);
      if (e != std::string_view::npos)
        return e;
    }
    return Body2(i);
  }

  // Returns the end of the piece starting at i (always > i).
  [[nodiscard]] std::size_t Next(std::size_t i) {
    const auto d = At(i);
    const auto code = uc::Lookup(d.cp);
    // 1 and 2: words.
    if (!IsNumber(code)) {
      const auto e = Word(i);
      if (e != std::string_view::npos)
        return e;
    }
    // 3: up to three numbers.
    if (IsNumber(code)) {
      std::size_t p = i;
      for (int n = 0; n < 3 && More(p); ++n) {
        const auto x = At(p);
        if (!IsNumber(uc::Lookup(x.cp)))
          break;
        p += x.len;
      }
      return p;
    }
    // 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
    {
      std::size_t p = i + (d.cp == ' ' ? 1 : 0);
      const std::size_t start = p;
      while (More(p)) {
        const auto x = At(p);
        const auto c = uc::Lookup(x.cp);
        if (uc::IsSpaceCode(c) || IsLetter(c) || IsNumber(c))
          break;
        p += x.len;
      }
      if (p > start) {
        while (More(p) && (s[p] == '\r' || s[p] == '\n' || s[p] == '/'))
          ++p;
        return p;
      }
    }
    // 5, 6, 7: white space (the code point at i is white space here).
    std::size_t p = i, last_start = i,
                after_last_newline = std::string_view::npos;
    while (More(p)) {
      const auto x = At(p);
      if (!uc::IsSpaceCode(uc::Lookup(x.cp)))
        break;
      last_start = p;
      p += x.len;
      if (IsNewline(x.cp))
        after_last_newline = p;
    }
    if (after_last_newline != std::string_view::npos)
      return after_last_newline;
    if (!More(p))
      return p;  // \s+(?!\S) at the end of the text
    if (last_start > i)
      return last_start;  // leave the last one for the next piece
    if (p == i)
      p = i + d.len;
    return p;  // \s+
  }
};

}  // namespace

void SplitPieces(std::string_view text,
                 std::vector<std::pair<std::uint32_t, std::uint32_t>>* out) {
  Scanner sc{text, {}};
  sc.run.reserve(32);
  std::size_t i = 0;
  while (i < text.size()) {
    const std::size_t e = sc.Next(i);
    out->emplace_back(static_cast<std::uint32_t>(i),
                      static_cast<std::uint32_t>(e));
    i = e;
  }
}

}  // namespace inkling2::text
