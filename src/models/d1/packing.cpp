#include <algorithm>
#include <stdexcept>

#include "src/models/d1/model.hpp"
#include "src/models/qwen/control_tokens.hpp"
namespace gufo::models::d1 {
std::string Suffix(const lfm3::Tokenizer& tok, const Question& q,
                   std::string body) {
  body += tokenization::kImEnd;
  body += '\n';
  body += tokenization::kImStart;
  body += "assistant\n<think></think>";
  for (std::size_t i = 0; i < q.slot_names.size(); ++i) {
    if (i)
      body += '\n';
    body += q.slot_names[i] + ":";
  }
  // Prompt 1's partial boxes blocks retain the colon+newline token used by
  // the full 16-block read. Do not split a merged token at the marker.
  if (q.type == "boxes" && q.blocks < 16) {
    const auto ids = tok.Encode(body + "\n");
    auto last = std::string(tok.Piece(ids.back()));
    while (!last.empty() && (last.back() == '\n' || last.back() == ' ' ||
                             last.back() == '\r' || last.back() == '\t'))
      last.pop_back();
    if (last.ends_with(':'))
      body += '\n';
  }
  return body;
}
Prompt Pack(const lfm3::Tokenizer& tok, const std::string& prefix,
            const std::string& header, std::span<const std::string> suffixes,
            std::span<const Question> questions) {
  if (suffixes.empty() || suffixes.size() != questions.size() ||
      suffixes.size() > 128)
    throw std::invalid_argument("invalid d1 question pack");
  Prompt p;
  for (std::size_t i = 0; i < suffixes.size(); ++i) {
    const auto text = (i ? header : prefix) + suffixes[i];
    auto ids = tok.Encode(text);
    const auto base = p.tokens.size();
    std::vector<std::size_t> colons;
    if (questions[i].IsBox()) {
      // Slot marker names are template text, so locate them after the lead.
      const auto lead = text.rfind("assistant\n<think></think>");
      if (lead == std::string::npos)
        throw std::invalid_argument("missing box assistant lead");
      std::size_t at =
          lead + std::string_view("assistant\n<think></think>").size();
      for (const auto& name : questions[i].slot_names) {
        const auto marker = text.find(name + ":", at);
        if (marker == std::string::npos)
          throw std::invalid_argument("missing box slot marker");
        at = marker + name.size() + 1;
        colons.push_back(at - 1);
      }
      std::size_t bytes = 0, slot = 0;
      for (std::size_t t = 0; t < ids.size(); ++t) {
        const auto stop = bytes + tok.Piece(ids[t]).size();
        if (slot < colons.size() && bytes <= colons[slot] &&
            colons[slot] < stop) {
          for (std::size_t b = colons[slot] + 1; b < stop; ++b)
            if (text[b] != ' ' && text[b] != '\n' && text[b] != '\r' &&
                text[b] != '\t')
              throw std::invalid_argument(
                  "box marker token includes another slot");
          p.slots.push_back(base + t);
          ++slot;
        }
        bytes = stop;
      }
      if (bytes != text.size() || slot != colons.size())
        throw std::invalid_argument(
            "box slot alignment differs from tokenization");
    } else
      p.slots.push_back(base + ids.size() - 1);
    const auto suffix_tokens = tok.Encode(suffixes[i]).size();
    if (suffix_tokens > ids.size())
      throw std::invalid_argument("question suffix is not inside its prompt");
    const auto state = i ? 0 : ids.size() - suffix_tokens;
    p.groups.insert(p.groups.end(), state, -1);
    p.groups.insert(p.groups.end(), ids.size() - state, std::int32_t(i + 1));
    p.tokens.insert(p.tokens.end(), ids.begin(), ids.end());
  }
  return p;
}
}  // namespace gufo::models::d1
