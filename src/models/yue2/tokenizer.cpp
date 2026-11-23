#include "src/models/yue2/tokenizer.hpp"

#include <unicode/normalizer2.h>
#include <unicode/stringpiece.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <sstream>

namespace gufo::models::yue2 {
namespace {

enum class Kind : std::uint8_t { kLetter, kNumber, kNewline, kSpace, kOther };

struct CodePoint {
  std::size_t offset;
  UChar32 value;
  Kind kind;
};

Kind Classify(UChar32 value) {
  if (value == '\r' || value == '\n')
    return Kind::kNewline;
  // Rust regex `\s` is the Unicode White_Space property.
  if (u_hasBinaryProperty(value, UCHAR_WHITE_SPACE))
    return Kind::kSpace;
  const auto mask = U_GET_GC_MASK(value);
  if ((mask & U_GC_L_MASK) != 0)
    return Kind::kLetter;
  if ((mask & U_GC_N_MASK) != 0)
    return Kind::kNumber;
  return Kind::kOther;
}

bool IsSpace(Kind kind) {
  return kind == Kind::kSpace || kind == Kind::kNewline;
}

std::vector<CodePoint> CodePoints(std::string_view text) {
  std::vector<CodePoint> points;
  points.reserve(text.size());
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(text.data());
  const auto length = static_cast<std::int32_t>(text.size());
  for (std::int32_t index = 0; index < length;) {
    const std::int32_t start = index;
    UChar32 value = 0;
    U8_NEXT(bytes, index, length, value);
    if (value < 0)
      value = 0xFFFD;
    points.push_back({static_cast<std::size_t>(start), value, Classify(value)});
  }
  return points;
}

// `(?i:s)` also matches U+017F under Unicode simple case folding.
bool FoldsTo(UChar32 value, char ascii) {
  return value == ascii || value == ascii - 'a' + 'A' ||
         (ascii == 's' && value == 0x017F);
}

std::size_t ContractionEnd(const std::vector<CodePoint>& points,
                           std::size_t i) {
  if (points[i].value != '\'' || i + 1 >= points.size())
    return 0;
  const UChar32 first = points[i + 1].value;
  if (FoldsTo(first, 's') || FoldsTo(first, 't'))
    return i + 2;
  if (i + 2 < points.size()) {
    const UChar32 second = points[i + 2].value;
    if ((FoldsTo(first, 'r') && FoldsTo(second, 'e')) ||
        (FoldsTo(first, 'v') && FoldsTo(second, 'e')))
      return i + 3;
  }
  if (FoldsTo(first, 'm'))
    return i + 2;
  if (i + 2 < points.size() && FoldsTo(first, 'l') &&
      FoldsTo(points[i + 2].value, 'l'))
    return i + 3;
  if (FoldsTo(first, 'd'))
    return i + 2;
  return 0;
}

// One match of the Qwen2 pattern at `i`, as an exclusive code point index.
std::size_t PieceEnd(const std::vector<CodePoint>& points, std::size_t i) {
  const std::size_t n = points.size();
  if (const std::size_t end = ContractionEnd(points, i); end != 0)
    return end;
  const auto run = [&](std::size_t from, auto predicate) {
    while (from < n && predicate(points[from].kind))
      ++from;
    return from;
  };
  const auto letter = [](Kind kind) { return kind == Kind::kLetter; };
  // [^\r\n\p{L}\p{N}]?\p{L}+
  if (points[i].kind == Kind::kLetter)
    return run(i, letter);
  if ((points[i].kind == Kind::kSpace || points[i].kind == Kind::kOther) &&
      i + 1 < n && points[i + 1].kind == Kind::kLetter)
    return run(i + 1, letter);
  // \p{N}
  if (points[i].kind == Kind::kNumber)
    return i + 1;
  // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
  const auto other = [](Kind kind) { return kind == Kind::kOther; };
  std::size_t start = i;
  if (points[i].value == ' ' && i + 1 < n && points[i + 1].kind == Kind::kOther)
    start = i + 1;
  if (points[start].kind == Kind::kOther)
    return run(run(start, other),
               [](Kind kind) { return kind == Kind::kNewline; });
  // Only whitespace remains.
  const std::size_t space_end = run(i, IsSpace);
  // \s*[\r\n]+: backtracking ends right after the run's last newline.
  for (std::size_t k = space_end; k > i; --k) {
    if (points[k - 1].kind == Kind::kNewline)
      return k;
  }
  // \s+(?!\S), then \s+.
  if (space_end == n || space_end - i < 2)
    return space_end;
  return space_end - 1;
}

bool DecodeBase64(std::string_view text, std::string* output) {
  static constexpr auto kTable = [] {
    std::array<std::int8_t, 256> table{};
    table.fill(-1);
    constexpr std::string_view kAlphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (std::size_t i = 0; i < kAlphabet.size(); ++i)
      table[static_cast<unsigned char>(kAlphabet[i])] =
          static_cast<std::int8_t>(i);
    return table;
  }();
  output->clear();
  std::uint32_t buffer = 0;
  int bits = 0;
  for (const char c : text) {
    if (c == '=')
      break;
    const std::int8_t value = kTable[static_cast<unsigned char>(c)];
    if (value < 0)
      return false;
    buffer = (buffer << 6U) | static_cast<std::uint32_t>(value);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      output->push_back(static_cast<char>((buffer >> bits) & 0xFFU));
    }
  }
  return true;
}

// Python's `bytes.decode("utf-8", errors="replace")`: one U+FFFD per maximal
// invalid subpart.
std::string ReplaceInvalidUtf8(std::string_view bytes) {
  std::string output;
  output.reserve(bytes.size());
  constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
  std::size_t i = 0;
  while (i < bytes.size()) {
    const auto lead = static_cast<std::uint8_t>(bytes[i]);
    std::size_t length = 0;
    std::uint8_t low = 0x80, high = 0xBF;
    if (lead < 0x80) {
      output.push_back(static_cast<char>(lead));
      ++i;
      continue;
    }
    if (lead >= 0xC2 && lead <= 0xDF) {
      length = 2;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      length = 3;
      if (lead == 0xE0)
        low = 0xA0;
      if (lead == 0xED)
        high = 0x9F;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      length = 4;
      if (lead == 0xF0)
        low = 0x90;
      if (lead == 0xF4)
        high = 0x8F;
    } else {
      output.append(kReplacement);
      ++i;
      continue;
    }
    std::size_t valid = 1;
    while (valid < length && i + valid < bytes.size()) {
      const auto next = static_cast<std::uint8_t>(bytes[i + valid]);
      const std::uint8_t lower = valid == 1 ? low : 0x80;
      const std::uint8_t upper = valid == 1 ? high : 0xBF;
      if (next < lower || next > upper)
        break;
      ++valid;
    }
    if (valid == length) {
      output.append(bytes.substr(i, length));
    } else {
      output.append(kReplacement);
    }
    i += valid;
  }
  return output;
}

}  // namespace

std::vector<std::string_view> SplitQwen2(std::string_view text) {
  const std::vector<CodePoint> points = CodePoints(text);
  std::vector<std::string_view> pieces;
  for (std::size_t i = 0; i < points.size();) {
    const std::size_t end = PieceEnd(points, i);
    const std::size_t begin_byte = points[i].offset;
    const std::size_t end_byte =
        end < points.size() ? points[end].offset : text.size();
    pieces.push_back(text.substr(begin_byte, end_byte - begin_byte));
    i = end;
  }
  return pieces;
}

std::unique_ptr<TextTokenizer> TextTokenizer::Load(
    const std::filesystem::path& ranks_file, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return nullptr;
  };
  std::ifstream input(ranks_file, std::ios::binary);
  if (!input)
    return fail("cannot open " + ranks_file.string());
  std::unique_ptr<TextTokenizer> tokenizer(new TextTokenizer());
  tokenizer->tokens_.resize(kOrdinaryTokens);
  tokenizer->ranks_.reserve(kOrdinaryTokens);
  std::vector<bool> seen(kOrdinaryTokens, false);
  std::string line;
  std::string bytes;
  std::size_t count = 0;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty())
      continue;
    const std::size_t space = line.find(' ');
    if (space == std::string::npos)
      return fail("malformed qwen.tiktoken line");
    std::uint32_t rank = 0;
    std::istringstream number(line.substr(space + 1));
    if (!(number >> rank) || rank >= kOrdinaryTokens || seen[rank] ||
        !DecodeBase64(std::string_view(line).substr(0, space), &bytes) ||
        bytes.empty())
      return fail("qwen.tiktoken must hold unique ranks 0..151642");
    seen[rank] = true;
    tokenizer->tokens_[rank] = bytes;
    if (!tokenizer->ranks_.emplace(bytes, rank).second)
      return fail("qwen.tiktoken repeats a token");
    ++count;
  }
  if (count != kOrdinaryTokens)
    return fail("expected the checkpoint's qwen.tiktoken (151643 tokens)");
  return tokenizer;
}

void TextTokenizer::MergePiece(std::string_view piece,
                               std::vector<std::uint32_t>* output) const {
  if (const auto found = ranks_.find(std::string(piece));
      found != ranks_.end()) {
    output->push_back(found->second);
    return;
  }
  constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();
  // tiktoken's byte_pair_merge: merge the adjacent pair whose bytes have the
  // lowest rank, leftmost on ties, until no pair is a token.
  struct Part {
    std::size_t start;
    std::uint32_t rank;
  };
  const auto rank_of = [&](std::size_t begin, std::size_t end) {
    const auto found =
        ranks_.find(std::string(piece.substr(begin, end - begin)));
    return found == ranks_.end() ? kNone : found->second;
  };
  std::vector<Part> parts;
  parts.reserve(piece.size() + 1);
  for (std::size_t i = 0; i < piece.size(); ++i)
    parts.push_back({i, kNone});
  parts.push_back({piece.size(), kNone});
  for (std::size_t i = 0; i + 2 < parts.size(); ++i)
    parts[i].rank = rank_of(parts[i].start, parts[i + 2].start);
  while (parts.size() > 2) {
    std::uint32_t best = kNone;
    std::size_t index = 0;
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
      if (parts[i].rank < best) {
        best = parts[i].rank;
        index = i;
      }
    }
    if (best == kNone)
      break;
    parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(index) + 1);
    parts[index].rank =
        index + 2 < parts.size()
            ? rank_of(parts[index].start, parts[index + 2].start)
            : kNone;
    if (index > 0)
      parts[index - 1].rank =
          rank_of(parts[index - 1].start, parts[index + 1].start);
  }
  for (std::size_t i = 0; i + 1 < parts.size(); ++i)
    output->push_back(rank_of(parts[i].start, parts[i + 1].start));
}

std::vector<std::uint32_t> TextTokenizer::Encode(std::string_view text) const {
  UErrorCode status = U_ZERO_ERROR;
  const auto* nfc = icu::Normalizer2::getNFCInstance(status);
  std::string normalized;
  if (U_SUCCESS(status)) {
    icu::StringByteSink<std::string> sink(&normalized);
    nfc->normalizeUTF8(
        0,
        icu::StringPiece(text.data(), static_cast<std::int32_t>(text.size())),
        sink, nullptr, status);
  }
  if (U_FAILURE(status))
    normalized.assign(text);
  std::vector<std::uint32_t> ids;
  ids.reserve(normalized.size() / 3 + 1);
  for (const std::string_view piece : SplitQwen2(normalized))
    MergePiece(piece, &ids);
  return ids;
}

std::string TextTokenizer::Decode(std::span<const std::uint32_t> ids) const {
  std::string bytes;
  for (const std::uint32_t id : ids) {
    if (id < tokens_.size())
      bytes.append(tokens_[id]);
  }
  return ReplaceInvalidUtf8(bytes);
}

}  // namespace gufo::models::yue2
