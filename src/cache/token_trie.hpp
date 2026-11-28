#ifndef GUFO_CACHE_TOKEN_TRIE_HPP_
#define GUFO_CACHE_TOKEN_TRIE_HPP_

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/cache/ledger.hpp"
#include "src/cache/types.hpp"

namespace gufo::cache {
// A compressed token trie whose nodes, edges and splits are charged to the
// ledger as kMetadata. Internal to the cache: not part of the extension API.
// A node holds `Records` (anything with empty()); Prune keeps a node while
// its records are not empty. Operations are caller serialized. Only mutation
// lives here: lookups walk Node edges and children themselves.
//
// Invariants callers rely on:
// - Every edge but the root's is non-empty, and each child's map key is the
//   first token of its edge.
// - Place returns the node where the path ends exactly. A split inserts a new
//   node above the existing one, so existing nodes, and pointers to them, are
//   never replaced.
// - Prune erases only record-less leaves and merges a record-less unary node
//   into its surviving child, which keeps its object. It tolerates a failed
//   merge reservation under a full ledger and leaves the unary node.
template<class Records>
class TokenTrie {
public:
  struct Node {
    ResourceCharge metadata;
    Node* parent{};
    std::vector<Token> edge;
    std::map<Token, std::unique_ptr<Node>> children;
    Records records;
    // The charge for this node and its edge.
    std::size_t bytes{};
  };
  TokenTrie() = delete;
  static std::size_t NodeBytes(std::size_t token_capacity) {
    return Add(
        sizeof(Node) + sizeof(typename decltype(Node::children)::value_type),
        Bytes(token_capacity, sizeof(Token)));
  }
  static std::unique_ptr<Node> MakeNode(ResourceLedger& ledger,
                                        std::span<const Token> tokens,
                                        Node* parent) {
    const auto bytes = NodeBytes(tokens.size());
    auto reservation = ledger.Reserve(ResourceCategory::kMetadata, bytes);
    auto node = std::make_unique<Node>();
    node->parent = parent;
    node->edge = Edge(tokens);
    node->bytes = bytes;
    node->metadata = reservation.Convert();
    return node;
  }
  // The node where `tokens` ends below `root`, creating and splitting nodes
  // as needed. All allocation/admission happens before changing an existing
  // edge.
  static Node* Place(ResourceLedger& ledger, Node& root,
                     std::span<const Token> tokens) {
    auto* node = &root;
    while (!tokens.empty()) {
      auto child = node->children.find(tokens.front());
      if (child == node->children.end()) {
        auto fresh = MakeNode(ledger, tokens, node);
        auto* result = fresh.get();
        node->children.emplace(tokens.front(), std::move(fresh));
        return result;
      }
      auto& old = child->second;
      const auto common = static_cast<std::size_t>(
          std::ranges::mismatch(old->edge, tokens).in1 - old->edge.begin());
      if (common == old->edge.size()) {
        tokens = tokens.subspan(common);
        node = old.get();
        continue;
      }
      auto split = MakeNode(ledger, tokens.first(common), node);
      const auto remainder = std::span<const Token>(old->edge).subspan(common);
      const auto narrowed_bytes = NodeBytes(remainder.size());
      auto narrowed_reservation =
          ledger.Reserve(ResourceCategory::kMetadata, narrowed_bytes);
      ResourceCharge narrowed_charge;
      auto suffix = Edge(remainder);
      narrowed_charge = narrowed_reservation.Convert();
      const auto old_key = suffix.front();
      // Preallocate the map entry, so linking the old subtree cannot fail.
      split->children.emplace(old_key, nullptr);
      Node* result = split.get();  // NOLINT(misc-const-correctness): returned
                                   // for record insertion
      if (common < tokens.size()) {
        auto fresh = MakeNode(ledger, tokens.subspan(common), split.get());
        result = fresh.get();
        split->children.emplace(tokens[common], std::move(fresh));
      }
      // Release the former edge before its charge. Shrinking capacity matters
      // when long histories are inserted before their shorter checkpoints.
      // Existing node pointers remain valid throughout the split.
      old->edge = std::move(suffix);
      old->metadata = std::move(narrowed_charge);
      old->bytes = narrowed_bytes;
      old->parent = split.get();
      split->children.at(old_key) = std::move(old);
      child->second = std::move(split);
      return result;
    }
    return node;
  }
  // Remove record-less nodes from `node` toward the root.
  static void Prune(ResourceLedger& ledger, Node* node) {
    while (node->parent && node->records.empty()) {
      auto* parent = node->parent;
      if (node->children.empty()) {
        parent->children.erase(node->edge.front());
      } else if (node->children.size() == 1) {
        // Compact the edge without replacing the surviving record's node.
        // Erasure must still succeed when headroom cannot fund this temporary
        // replacement; a later erasure can retry the harmless unary path.
        auto& child = node->children.begin()->second;
        const auto count = Add(node->edge.size(), child->edge.size());
        const auto bytes = NodeBytes(count);
        try {
          auto reservation = ledger.Reserve(ResourceCategory::kMetadata, bytes);
          std::vector<Token> merged(count);
          if (merged.capacity() != count)
            throw std::logic_error("unexpected token trie edge capacity");
          std::ranges::copy(node->edge, merged.begin());
          std::ranges::copy(
              child->edge,
              merged.begin() + static_cast<std::ptrdiff_t>(node->edge.size()));
          auto charge = reservation.Convert();
          child->edge = std::move(merged);
          child->bytes = bytes;
          child->metadata = std::move(charge);
          child->parent = parent;
          auto promoted = std::move(child);
          parent->children.at(node->edge.front()) = std::move(promoted);
        } catch (const std::bad_alloc&) {
          return;
        }
      } else {
        return;
      }
      node = parent;
    }
  }
  // Destroy every node below `root` without recursion, deepest first.
  static void Clear(Node& root) noexcept {
    auto* node = &root;
    while (true) {
      if (!node->children.empty()) {
        node = node->children.begin()->second.get();
        continue;
      }
      if (node == &root)
        break;
      auto* parent = node->parent;
      parent->children.erase(node->edge.front());
      node = parent;
    }
  }

private:
  // Header-local copies of prefix_index.cpp's checked arithmetic, whose Bytes
  // takes Rows, so the trie stands alone for every user.
  static std::size_t Add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
      throw std::overflow_error("token trie size overflow");
    return a + b;
  }
  static std::size_t Bytes(std::size_t count, std::size_t bytes) {
    if (bytes && count > std::numeric_limits<std::size_t>::max() / bytes)
      throw std::overflow_error("token trie edge size overflow");
    return count * bytes;
  }
  static std::vector<Token> Edge(std::span<const Token> tokens) {
    std::vector<Token> edge(tokens.begin(), tokens.end());
    // The pinned standard library creates exact-capacity range vectors. Fail
    // closed if a different implementation adds unadmitted spare capacity.
    if (edge.capacity() != tokens.size())
      throw std::logic_error("unexpected token trie edge capacity");
    return edge;
  }
};
}  // namespace gufo::cache
#endif
