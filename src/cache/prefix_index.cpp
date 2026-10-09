#include "src/cache/prefix_index.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace gufo::cache {
namespace {
std::size_t Add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("prefix index size overflow");
  return a + b;
}
std::size_t Bytes(Rows rows, std::size_t bytes) {
  if (bytes && rows > std::numeric_limits<std::size_t>::max() / bytes)
    throw std::overflow_error("prefix index transfer size overflow");
  return static_cast<std::size_t>(rows) * bytes;
}
bool Same(const ComponentDescriptor& a, const ComponentDescriptor& b) {
  return a.id == b.id && a.kind == b.kind &&
         a.layout_version == b.layout_version && a.row_bytes == b.row_bytes &&
         a.rows_per_chunk == b.rows_per_chunk && a.state_bytes == b.state_bytes;
}
void ValidateAvailability(std::span<const ComponentAvailability> list,
                          std::span<const ComponentDescriptor> descriptors) {
  for (std::size_t i = 0; i < list.size(); ++i) {
    if (std::ranges::none_of(descriptors,
                             [&](const auto& d) { return d.id == list[i].id; }))
      throw std::invalid_argument("unknown availability component");
    for (std::size_t j = 0; j < i; ++j)
      if (list[i].id == list[j].id)
        throw std::invalid_argument("duplicate availability component");
  }
}
}  // namespace
struct PrefixIndex::Impl {
  struct Record {
    ResourceCharge metadata;
    IndexEntryId id;
    std::shared_ptr<const Checkpoint> checkpoint;
    std::optional<LiveFrontier> live;
    std::vector<ComponentAvailability> availability;
    Rows stable{};
    std::size_t bytes{};
    [[nodiscard]] Rows Boundary() const {
      if (checkpoint)
        return checkpoint->Boundary();
      if (live)
        return live->tokens.size();
      throw std::logic_error("empty prefix index record");
    }
    [[nodiscard]] const Identity& Input() const {
      if (checkpoint)
        return checkpoint->Input();
      if (live)
        return live->input;
      throw std::logic_error("empty prefix index record");
    }
  };
  struct Node {
    ResourceCharge metadata;
    Node* parent{};
    std::vector<Token> edge;
    std::map<Token, std::unique_ptr<Node>> children;
    std::map<std::uint64_t, Record> records;
    std::size_t bytes{};
  };
  struct Tree {
    ResourceCharge metadata;
    std::vector<ComponentDescriptor> descriptors;
    std::unique_ptr<Node> root;
    std::size_t bytes{};
  };
  ResourceLedger* ledger;
  ResourceCharge metadata;
  std::map<Identity, Tree> trees;
  std::map<std::uint64_t, Node*> directory;
  std::uint64_t next{1};
  Impl(ResourceLedger& l, ResourceReservation reservation)
      : ledger(&l), metadata(reservation.Convert()) {}
  ~Impl() {
    for (auto& [identity, tree] : trees) {
      (void)identity;
      auto* node = tree.root.get();
      while (true) {
        if (!node->children.empty()) {
          node = node->children.begin()->second.get();
          continue;
        }
        if (node == tree.root.get())
          break;
        auto* parent = node->parent;
        parent->children.erase(node->edge.front());
        node = parent;
      }
    }
  }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ResourceCharge Charge(std::size_t size) {
    return ledger->Reserve(ResourceCategory::kMetadata, size).Convert();
  }
  static std::size_t NodeBytes(std::size_t token_capacity) {
    return Add(sizeof(Node) + sizeof(decltype(Node::children)::value_type),
               Bytes(token_capacity, sizeof(Token)));
  }
  static std::vector<Token> Edge(std::span<const Token> tokens) {
    std::vector<Token> edge(tokens.begin(), tokens.end());
    // The pinned standard library creates exact-capacity range vectors. Fail
    // closed if a different implementation adds unadmitted spare capacity.
    if (edge.capacity() != tokens.size())
      throw std::logic_error("unexpected prefix edge capacity");
    return edge;
  }
  std::unique_ptr<Node> MakeNode(std::span<const Token> tokens, Node* parent) {
    const auto bytes = NodeBytes(tokens.size());
    auto reservation = ledger->Reserve(ResourceCategory::kMetadata, bytes);
    auto node = std::make_unique<Node>();
    node->parent = parent;
    node->edge = Edge(tokens);
    node->bytes = bytes;
    node->metadata = reservation.Convert();
    return node;
  }
  Tree& Get(const Identity& identity) {
    auto it = trees.find(identity);
    if (it == trees.end())
      throw std::invalid_argument("unregistered compatibility identity");
    return it->second;
  }
  Record& Get(IndexEntryId id) {
    auto it = directory.find(id.value);
    if (it == directory.end())
      throw std::invalid_argument("unknown prefix index entry");
    return it->second->records.at(id.value);
  }
  // All allocation/admission happens before changing an existing edge.
  Node* Place(Tree& tree, std::span<const Token> tokens) {
    auto* node = tree.root.get();
    while (!tokens.empty()) {
      auto child = node->children.find(tokens.front());
      if (child == node->children.end()) {
        auto fresh = MakeNode(tokens, node);
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
      auto split = MakeNode(tokens.first(common), node);
      const auto remainder = std::span<const Token>(old->edge).subspan(common);
      const auto narrowed_bytes = NodeBytes(remainder.size());
      auto narrowed_reservation =
          ledger->Reserve(ResourceCategory::kMetadata, narrowed_bytes);
      ResourceCharge narrowed_charge;
      auto suffix = Edge(remainder);
      narrowed_charge = narrowed_reservation.Convert();
      const auto old_key = suffix.front();
      // Preallocate the map entry, so linking the old subtree cannot fail.
      split->children.emplace(old_key, nullptr);
      Node* result = split.get();  // NOLINT(misc-const-correctness): returned
                                   // for record insertion
      if (common < tokens.size()) {
        auto fresh = MakeNode(tokens.subspan(common), split.get());
        result = fresh.get();
        split->children.emplace(tokens[common], std::move(fresh));
      }
      // Release the former edge before its charge. Shrinking capacity matters
      // when long histories are inserted before their shorter checkpoints.
      // Existing directory pointers remain valid throughout the split.
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
  void Prune(Node* node) {
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
          auto reservation =
              ledger->Reserve(ResourceCategory::kMetadata, bytes);
          std::vector<Token> merged(count);
          if (merged.capacity() != count)
            throw std::logic_error("unexpected prefix edge capacity");
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
  IndexEntryId Publish(Tree& tree, std::span<const Token> tokens,
                       Record record) {
    if (next == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("prefix index identifiers exhausted");
    record.id = {next};
    // Preallocate publication map nodes before touching the token tree. Node
    // handle insertion below cannot allocate after a successful edge split.
    std::map<std::uint64_t, Record> pending;
    pending.emplace(next, std::move(record));
    auto entry = pending.extract(next);
    directory.emplace(next, nullptr);
    Node* node = nullptr;
    try {
      node = Place(tree, tokens);
    } catch (...) {
      directory.erase(next);
      throw;
    }
    node->records.insert(std::move(entry));
    directory.at(next) = node;
    return {next++};
  }
  std::size_t RecordBytes(const Record& r) const {
    auto bytes = sizeof(decltype(directory)::value_type) +
                 sizeof(decltype(Node::records)::value_type);
    bytes = Add(
        bytes, Bytes(r.availability.capacity(), sizeof(ComponentAvailability)));
    if (r.live) {
      bytes = Add(bytes, Bytes(r.live->tokens.capacity(), sizeof(Token)));
      bytes = Add(bytes, r.live->compatibility.capacity());
      bytes = Add(bytes, r.live->input.capacity());
      bytes = Add(bytes, Bytes(r.live->positions.capacity(),
                               sizeof(ComponentPosition)));
    }
    return bytes;
  }
  bool Resident(const Tree& tree, const Record& r) const {
    if (r.live) {
      if (!r.live->available ||
          r.live->positions.size() != tree.descriptors.size())
        return false;
      for (const auto& d : tree.descriptors) {
        auto p =
            std::ranges::find(r.live->positions, d.id, &ComponentPosition::id);
        if (p == r.live->positions.end() ||
            (d.kind == ComponentKind::kPrivateState &&
             p->valid_rows != r.Boundary()))
          return false;
      }
      return true;
    }
    if (r.checkpoint->Components().size() != tree.descriptors.size())
      return false;
    for (const auto& d : tree.descriptors) {
      auto a =
          std::ranges::find(r.availability, d.id, &ComponentAvailability::id);
      auto c = std::ranges::find_if(
          r.checkpoint->Components(),
          [&](const auto& c) { return Same(d, c.descriptor); });
      if (a == r.availability.end() || !a->resident ||
          c == r.checkpoint->Components().end() ||
          (d.kind == ComponentKind::kPrivateState &&
           c->position.valid_rows != r.Boundary()))
        return false;
    }
    return true;
  }
  PrefixCandidate Candidate(const Record& r) const {
    PrefixCandidate result{r.id, r.Boundary(), r.checkpoint, {}, 0};
    if (r.live) {
      result.live = r.live->location;
    } else {
      for (const auto& c : r.checkpoint->Components())
        result.transfer_bytes =
            Add(result.transfer_bytes,
                c.descriptor.kind == ComponentKind::kPrivateState
                    ? c.descriptor.state_bytes
                    : Bytes(c.position.valid_rows, c.descriptor.row_bytes));
    }
    return result;
  }
  template<class Visit>
  void Path(const Tree& tree, std::span<const Token> tokens,
            Visit visit) const {
    auto* node = tree.root.get();
    visit(*node);
    while (!tokens.empty()) {
      auto it = node->children.find(tokens.front());
      if (it == node->children.end())
        return;
      node = it->second.get();
      auto common =
          std::ranges::mismatch(node->edge, tokens).in1 - node->edge.begin();
      if (static_cast<std::size_t>(common) != node->edge.size())
        return;
      tokens = tokens.subspan(node->edge.size());
      visit(*node);
    }
  }
};
PrefixIndex::PrefixIndex(ResourceLedger& ledger) {
  auto reservation = ledger.Reserve(ResourceCategory::kMetadata, sizeof(Impl));
  impl_ = std::make_unique<Impl>(ledger, std::move(reservation));
}
PrefixIndex::~PrefixIndex() = default;
void PrefixIndex::Register(Identity identity,
                           std::span<const ComponentDescriptor> descriptors) {
  if (descriptors.empty())
    throw std::invalid_argument("prefix index has no required components");
  for (std::size_t i = 0; i < descriptors.size(); ++i) {
    const auto& d = descriptors[i];
    if ((d.kind == ComponentKind::kAppendRows &&
         (!d.row_bytes || !d.rows_per_chunk || d.state_bytes)) ||
        (d.kind == ComponentKind::kPrivateState &&
         (!d.state_bytes || d.row_bytes || d.rows_per_chunk)) ||
        (d.kind != ComponentKind::kAppendRows &&
         d.kind != ComponentKind::kPrivateState))
      throw std::invalid_argument("invalid prefix index component geometry");
    for (std::size_t j = 0; j < i; ++j)
      if (d.id == descriptors[j].id)
        throw std::invalid_argument("duplicate prefix index component");
  }
  auto it = impl_->trees.find(identity);
  if (it != impl_->trees.end()) {
    if (descriptors.size() != it->second.descriptors.size() ||
        std::ranges::any_of(descriptors, [&](const auto& d) {
          return std::ranges::none_of(
              it->second.descriptors,
              [&](const auto& other) { return Same(d, other); });
        }))
      throw std::invalid_argument("compatibility inventory changed");
    return;
  }
  const auto bytes =
      Add(sizeof(decltype(impl_->trees)::value_type),
          Add(identity.capacity(),
              Bytes(descriptors.size(), sizeof(ComponentDescriptor))));
  auto reservation = impl_->ledger->Reserve(ResourceCategory::kMetadata, bytes);
  Impl::Tree tree;
  tree.descriptors =
      std::vector<ComponentDescriptor>(descriptors.begin(), descriptors.end());
  if (tree.descriptors.capacity() != descriptors.size())
    throw std::logic_error("unexpected prefix inventory capacity");
  tree.bytes = bytes;
  tree.root = impl_->MakeNode({}, nullptr);
  tree.metadata = reservation.Convert();
  impl_->trees.emplace(std::move(identity), std::move(tree));
}
IndexEntryId PrefixIndex::Insert(
    std::shared_ptr<const Checkpoint> checkpoint,
    std::vector<ComponentAvailability> availability, Rows stable) {
  if (!checkpoint || stable > checkpoint->Boundary())
    throw std::invalid_argument("invalid indexed checkpoint");
  auto& tree = impl_->Get(checkpoint->Compatibility());
  ValidateAvailability(availability, tree.descriptors);
  Impl::Record r;
  r.checkpoint = std::move(checkpoint);
  r.availability = std::move(availability);
  r.stable = stable;
  r.bytes = impl_->RecordBytes(r);
  r.metadata = impl_->Charge(r.bytes);
  auto tokens = r.checkpoint->Tokens();
  return impl_->Publish(tree, tokens, std::move(r));
}
IndexEntryId PrefixIndex::Insert(LiveFrontier frontier) {
  auto& tree = impl_->Get(frontier.compatibility);
  Impl::Record r;
  r.live = std::move(frontier);
  r.bytes = impl_->RecordBytes(r);
  r.metadata = impl_->Charge(r.bytes);
  auto tokens = std::span<const Token>(r.live->tokens);
  return impl_->Publish(tree, tokens, std::move(r));
}
void PrefixIndex::SetAvailability(
    IndexEntryId id, std::vector<ComponentAvailability> availability) {
  auto& r = impl_->Get(id);
  if (!r.checkpoint)
    throw std::invalid_argument("live frontier has no checkpoint availability");
  ValidateAvailability(availability,
                       impl_->Get(r.checkpoint->Compatibility()).descriptors);
  Impl::Record replacement;
  replacement.availability = std::move(availability);
  replacement.bytes = impl_->RecordBytes(replacement);
  auto charge = impl_->Charge(replacement.bytes);
  r.availability.swap(replacement.availability);
  r.bytes = replacement.bytes;
  std::vector<ComponentAvailability>{}.swap(replacement.availability);
  r.metadata = std::move(charge);
}
void PrefixIndex::SetLiveAvailable(IndexEntryId id, bool available) {
  auto& r = impl_->Get(id);
  if (!r.live)
    throw std::invalid_argument("entry is not a live frontier");
  r.live->available = available;
}
void PrefixIndex::Erase(IndexEntryId id) {
  auto it = impl_->directory.find(id.value);
  if (it == impl_->directory.end())
    return;
  auto* node = it->second;
  node->records.erase(id.value);
  impl_->directory.erase(it);
  impl_->Prune(node);
}
PrefixLookup PrefixIndex::Lookup(const PrefixQuery& query) const {
  if (query.stable_prefix_tokens > query.tokens.size() ||
      query.input.TokenCount() != query.tokens.size())
    throw std::invalid_argument("invalid prefix query boundaries");
  PrefixLookup result;
  if (!query.reuse) {
    result.reason = SelectionReason::kDisabled;
    return result;
  }
  auto it = impl_->trees.find(query.compatibility);
  if (it == impl_->trees.end()) {
    result.reason = SelectionReason::kUnknownCompatibility;
    return result;
  }
  const auto& tree = it->second;
  bool fallback = query.stable_prefix_tokens == 0;
  bool input_rejected = false, missing = false, stable_rejected = false;
  impl_->Path(tree, query.tokens, [&](const auto& node) {
    for (const auto& [id, r] : node.records) {
      (void)id;
      if (!std::ranges::equal(r.Input(), query.input.At(r.Boundary()))) {
        input_rejected = true;
        continue;
      }
      if (!impl_->Resident(tree, r)) {
        missing = true;
        continue;
      }
      if (r.checkpoint &&
          (r.Boundary() <= query.stable_prefix_tokens ||
           (r.stable != 0 && r.stable <= query.stable_prefix_tokens)))
        fallback = true;
      result.candidates.push_back(impl_->Candidate(r));
    }
  });
  std::erase_if(result.candidates, [&](const auto& c) {
    const bool rejected = !fallback && c.boundary > query.stable_prefix_tokens;
    stable_rejected |= rejected;
    return rejected;
  });
  std::ranges::sort(result.candidates, [](const auto& a, const auto& b) {
    const auto key = [](const auto& c) {
      return std::tuple{std::numeric_limits<Rows>::max() - c.boundary,
                        !c.live.has_value(),
                        c.live ? c.live->slot.value : c.checkpoint->Id().value,
                        c.live ? c.live->generation : 0, c.entry.value};
    };
    return key(a) < key(b);
  });
  if (result.candidates.empty()) {
    result.reason = stable_rejected  ? SelectionReason::kStablePrefixBoundary
                    : missing        ? SelectionReason::kMissingComponent
                    : input_rejected ? SelectionReason::kInputIdentity
                                     : SelectionReason::kNoCompatibleBoundary;
    return result;
  }
  result.selected = result.candidates.front();
  result.reason = result.selected->live
                      ? SelectionReason::kExactLiveContinuation
                      : SelectionReason::kDeepestCheckpoint;
  return result;
}
Rows PrefixIndex::CachedPrefixTokens(const PrefixQuery& query) const {
  auto copy = query;
  copy.stable_prefix_tokens = 0;
  copy.reuse = true;
  const auto result = Lookup(copy);
  return result.selected ? result.selected->boundary : 0;
}
Rows PrefixIndex::CommonPrefixTokens(const PrefixQuery& query) const {
  if (query.input.TokenCount() != query.tokens.size())
    throw std::invalid_argument("invalid prefix query boundaries");
  auto it = impl_->trees.find(query.compatibility);
  if (it == impl_->trees.end())
    return 0;
  // Follow token agreement once, then inspect only the matching subtree. A
  // stored record's complete identity gates learning, even if its state is
  // unavailable or its boundary is longer than the incoming prompt.
  Rows longest = 0;
  struct Pending {
    const Impl::Node* node;
    Rows depth, common;
    bool diverged;
  };
  std::vector<Pending> pending{{it->second.root.get(), 0, 0, false}};
  while (!pending.empty()) {
    const auto current = pending.back();
    pending.pop_back();
    for (const auto& [id, r] : current.node->records) {
      (void)id;
      const auto input = r.Boundary() <= query.tokens.size()
                             ? query.input.At(r.Boundary())
                             : query.input.Complete();
      if (std::ranges::equal(input, r.Input()))
        longest = std::max(longest, current.common);
    }
    for (const auto& [token, child] : current.node->children) {
      (void)token;
      Rows common = current.common;
      bool diverged = current.diverged;
      if (!diverged) {
        const auto suffix =
            query.tokens.subspan(static_cast<std::size_t>(current.depth));
        const auto matched =
            static_cast<Rows>(std::ranges::mismatch(child->edge, suffix).in1 -
                              child->edge.begin());
        common += matched;
        diverged = matched != child->edge.size();
      }
      // Descendants cannot improve agreement after a branch has diverged.
      if (common > longest)
        pending.push_back({child.get(), current.depth + child->edge.size(),
                           common, diverged});
    }
  }
  return longest;
}
std::size_t PrefixIndex::MetadataBytes() const {
  std::size_t bytes = sizeof(Impl);
  std::vector<const Impl::Node*> pending;
  for (const auto& [identity, tree] : impl_->trees) {
    (void)identity;
    bytes = Add(bytes, tree.bytes);
    pending.push_back(tree.root.get());
  }
  while (!pending.empty()) {
    const auto* node = pending.back();
    pending.pop_back();
    bytes = Add(bytes, node->bytes);
    for (const auto& [id, r] : node->records) {
      (void)id;
      bytes = Add(bytes, r.bytes);
    }
    for (const auto& [token, child] : node->children) {
      (void)token;
      pending.push_back(child.get());
    }
  }
  return bytes;
}
}  // namespace gufo::cache
