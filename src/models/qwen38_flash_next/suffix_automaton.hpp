#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Suffix automaton over the token history, used as a second draft source
/// alongside the MTP predictor.
///
/// Speculative decoding needs "what followed this exact suffix earlier?". A
/// suffix automaton answers that with the longest match rather than a fixed
/// n-gram window, so a repeated phrase of any length still produces a draft,
/// and the walk that finds it is amortized O(1) per appended token. Both
/// properties matter: retrieval-style speculation only pays when it commits to
/// the longest available match instead of falling back to a short window.
///
/// The automaton is host-side on purpose. It is queried once per draft
/// position inside a loop that already drives the predictor forward one token
/// at a time, so a device copy would add synchronization for no overlap, and
/// keeping it host-side makes a request's drafts a pure function of its own
/// token history, which seeded replay requires.
class SuffixAutomaton {
public:
  /// Appends one token to the indexed text. Amortized O(1).
  void Append(std::int32_t token) {
    State cur;
    cur.length = states_[last_].length + 1;
    cur.end = text_size_;
    const auto index = static_cast<std::int32_t>(states_.size());
    states_.push_back(std::move(cur));
    text_.push_back(token);

    std::int32_t p = last_;
    while (p >= 0 && !HasTransition(p, token)) {
      AddTransition(p, token, index);
      p = states_[p].link;
    }
    if (p < 0) {
      states_[index].link = 0;
    } else {
      const std::int32_t q = Transition(p, token);
      if (states_[p].length + 1 == states_[q].length) {
        states_[index].link = q;
      } else {
        State clone = states_[q];
        clone.length = states_[p].length + 1;
        const auto clone_index = static_cast<std::int32_t>(states_.size());
        states_.push_back(std::move(clone));
        while (p >= 0 && Transition(p, token) == q) {
          AssignTransition(p, token, clone_index);
          p = states_[p].link;
        }
        states_[q].link = clone_index;
        states_[index].link = clone_index;
      }
    }
    last_ = index;
    ++text_size_;
  }

  /// Longest suffix of `context` that occurred earlier in the indexed text,
  /// together with the token that followed its first occurrence. `length` is
  /// zero when nothing matched with a usable continuation.
  struct Match {
    std::int32_t length{0};
    std::int32_t next{0};
  };

  [[nodiscard]] Match LongestMatch(const std::vector<std::int32_t>& context,
                                   std::size_t from) const {
    Match best;
    if (from >= context.size() || text_size_ == 0) {
      return best;
    }
    std::int32_t v = 0;
    std::int32_t l = 0;
    for (std::size_t i = from; i < context.size(); ++i) {
      const std::int32_t c = context[i];
      while (v != 0 && !HasTransition(v, c)) {
        v = states_[v].link;
        l = states_[v].length;
      }
      if (!HasTransition(v, c)) {
        v = 0;
        l = 0;
        continue;
      }
      v = Transition(v, c);
      ++l;
      // `end` is where this state's string first occurred, so the token after
      // it is the continuation. A match that ends at the newest token has no
      // continuation yet and is the trivial self-match; reject it and keep the
      // longest earlier occurrence instead.
      const std::int32_t end = states_[v].end;
      if (l > best.length && end + 1 < text_size_) {
        best.length = l;
        best.next = text_[end + 1];
      }
    }
    return best;
  }

  /// Draft continuation for a context that ends at `from`, copying at most
  /// `limit` tokens out of the matched region. A match of length L predicts the
  /// token that followed it plus the rest of the matched run, so the draft is
  /// capped at L+1: longer would be speculation with no evidence behind it.
  [[nodiscard]] std::vector<std::int32_t> Draft(
      const std::vector<std::int32_t>& context, std::size_t from,
      std::int32_t minimum_match, std::size_t limit) const {
    std::vector<std::int32_t> draft;
    if (limit == 0) {
      return draft;
    }
    const Match match = LongestMatch(context, from);
    if (match.length < minimum_match) {
      return draft;
    }
    const std::int32_t end = EndOfMatch(context, from, match.length);
    if (end < 0) {
      return draft;
    }
    const auto count = static_cast<std::size_t>(std::min<std::int32_t>(
        match.length + 1, static_cast<std::int32_t>(limit)));
    draft.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const auto at = static_cast<std::size_t>(end) + 1 + i;
      if (at >= text_size_) {
        break;
      }
      draft.push_back(text_[at]);
    }
    return draft;
  }

  [[nodiscard]] std::size_t size() const noexcept { return text_size_; }
  [[nodiscard]] std::size_t states() const noexcept { return states_.size(); }

private:
  struct State {
    std::int32_t link{-1};
    std::int32_t length{0};
    /// Index of the last token of this state's earliest occurrence.
    std::int32_t end{-1};
    /// Outgoing transitions, kept sorted so a lookup is a short scan. States
    /// carry few transitions on real token streams, and a sorted vector keeps
    /// the whole structure in one allocation.
    std::vector<std::pair<std::int32_t, std::int32_t>> next;
  };

  [[nodiscard]] bool HasTransition(std::int32_t state,
                                   std::int32_t token) const noexcept {
    const auto& edges = states_[state].next;
    const auto it =
        std::lower_bound(edges.begin(), edges.end(), token,
                         [](const std::pair<std::int32_t, std::int32_t>& e,
                            std::int32_t v) { return e.first < v; });
    return it != edges.end() && it->first == token;
  }

  [[nodiscard]] std::int32_t Transition(std::int32_t state,
                                        std::int32_t token) const noexcept {
    const auto& edges = states_[state].next;
    const auto it =
        std::lower_bound(edges.begin(), edges.end(), token,
                         [](const std::pair<std::int32_t, std::int32_t>& e,
                            std::int32_t v) { return e.first < v; });
    return it->second;
  }

  /// Inserts the transition, keeping the edge list sorted.
  void AddTransition(std::int32_t state, std::int32_t token,
                     std::int32_t target) {
    auto& edges = states_[state].next;
    const auto it =
        std::lower_bound(edges.begin(), edges.end(), token,
                         [](const std::pair<std::int32_t, std::int32_t>& e,
                            std::int32_t v) { return e.first < v; });
    edges.insert(it, {token, target});
  }

  /// Points an existing transition at a different state.
  void AssignTransition(std::int32_t state, std::int32_t token,
                        std::int32_t target) {
    auto& edges = states_[state].next;
    const auto it =
        std::lower_bound(edges.begin(), edges.end(), token,
                         [](const std::pair<std::int32_t, std::int32_t>& e,
                            std::int32_t v) { return e.first < v; });
    if (it != edges.end() && it->first == token) {
      it->second = target;
    }
  }

  /// Walks again to recover the end index of the accepted match, so the caller
  /// can copy the continuation out of the indexed text.
  [[nodiscard]] std::int32_t EndOfMatch(
      const std::vector<std::int32_t>& context, std::size_t from,
      std::int32_t target_length) const {
    std::int32_t v = 0;
    std::int32_t l = 0;
    std::int32_t end = -1;
    for (std::size_t i = from; i < context.size() && l < target_length; ++i) {
      const std::int32_t c = context[i];
      while (v != 0 && !HasTransition(v, c)) {
        v = states_[v].link;
        l = states_[v].length;
      }
      if (!HasTransition(v, c)) {
        v = 0;
        l = 0;
        continue;
      }
      v = Transition(v, c);
      ++l;
      if (l == target_length && states_[v].end + 1 < text_size_) {
        end = states_[v].end;
      }
    }
    return end;
  }

  std::vector<State> states_{State{}};
  std::vector<std::int32_t> text_;
  std::size_t text_size_{0};
  std::int32_t last_{0};
};

}  // namespace gufo::models::qwen38_flash_next
