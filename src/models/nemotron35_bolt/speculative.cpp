#include "src/models/nemotron35_bolt/speculative.hpp"

#include <algorithm>
#include <span>
#include <utility>

#include "src/models/nemotron35_bolt/drafter.hpp"
#include "src/models/nemotron35_bolt/engine.hpp"

namespace gufo::models::nemotron35_bolt {

bool SpeculativeStep(Session& session, sampling::SamplerState& sampler,
                     Frontier& frontier, std::uint32_t max_tokens,
                     const std::function<bool(std::int32_t)>& is_stop,
                     bool keep_rows, SpeculativeRound* out,
                     std::string* error_msg) {
  auto fail = [&](std::string message) {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return false;
  };
  *out = {};
  if (max_tokens == 0)
    return fail("a decode round emits at least one token");
  const std::size_t vocab = frontier.logits.size();
  const auto filter = sampler.config().logit_head_filter();
  const auto first =
      static_cast<std::int32_t>(sampler.Sample(frontier.logits, frontier.head));
  out->tokens.push_back(first);
  if (keep_rows)
    out->rows.assign(frontier.logits.begin(), frontier.logits.end());
  if (is_stop(first)) {
    out->stop = true;
    return true;
  }
  sampler.Accept(first);
  if (session.Position() >= session.MaxContext())
    return fail("the session context is full");

  std::vector<std::int32_t> drafts;
  DraftContext* draft = session.draft();
  const std::uint32_t room = session.MaxContext() - session.Position();
  if (draft != nullptr && max_tokens > 1 && room > 1 &&
      session.Position() > 0) {
    DraftLattice lattice;
    if (!draft->Draft(first, session.Position(), &lattice, {}, error_msg))
      return false;
    drafts = lattice.Walk(std::min(max_tokens - 1, room - 1));
  }
  if (drafts.empty()) {
    const std::int32_t one[] = {first};
    if (filter)
      return session.Append(one, frontier.logits, *filter, &frontier.head,
                            error_msg);
    frontier.head.candidates.clear();
    return session.Append(one, frontier.logits, error_msg);
  }

  std::vector<std::int32_t> block;
  block.reserve(drafts.size() + 1);
  block.push_back(first);
  block.insert(block.end(), drafts.begin(), drafts.end());
  static thread_local std::vector<float> rows;
  static thread_local std::vector<sampling::LogitHead> heads;
  rows.resize(block.size() * vocab);
  heads.resize(block.size());
  for (auto& head : heads)
    head.candidates.clear();
  if (!(filter ? session.Verify(block, rows, *filter, heads, error_msg)
               : session.Verify(block, rows, error_msg)))
    return false;
  std::uint32_t kept = 1;
  for (std::size_t i = 0; i < drafts.size(); ++i) {
    const std::span<const float> row(rows.data() + i * vocab, vocab);
    const auto token = static_cast<std::int32_t>(sampler.Sample(row, heads[i]));
    if (token != drafts[i]) {
      sampler.DeferSample(static_cast<sampling::TokenId>(token));
      break;
    }
    out->tokens.push_back(token);
    if (keep_rows)
      out->rows.insert(out->rows.end(), row.begin(), row.end());
    if (is_stop(token)) {
      // Like a first token that stops, it is reported but never run.
      out->stop = true;
      break;
    }
    sampler.Accept(token);
    ++kept;
  }
  out->drafted = static_cast<std::uint32_t>(drafts.size());
  out->accepted = static_cast<std::uint32_t>(out->tokens.size()) - 1;
  if (!session.Commit(kept, error_msg))
    return false;
  const auto next =
      rows.begin() + static_cast<std::ptrdiff_t>((kept - 1) * vocab);
  std::copy(next, next + static_cast<std::ptrdiff_t>(vocab),
            frontier.logits.begin());
  std::swap(frontier.head, heads[kept - 1]);
  return true;
}

}  // namespace gufo::models::nemotron35_bolt
