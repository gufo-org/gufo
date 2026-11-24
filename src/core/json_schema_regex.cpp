#include "src/core/json_schema_regex.hpp"

#include <gufo/json_schema.h>
#include <unicode/uniset.h>
#include <unicode/unistr.h>
#include <unicode/usetiter.h>

#include <algorithm>
#include <new>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/json_schema_call.hpp"

namespace gufo::sampling {
namespace {
using Id = std::uint32_t;
constexpr Id kDead = JsonSchemaRegex::kDead;

// Reports an ICU set's ranges and multi-codepoint strings.
inline void ReportSet(const icu::UnicodeSet& set, GufoJsonSchemaRangeFn add,
                      GufoJsonSchemaStringFn add_string, void* sink) {
  for (int32_t i = 0; i < set.getRangeCount(); ++i)
    add(sink, static_cast<std::uint32_t>(set.getRangeStart(i)),
        static_cast<std::uint32_t>(set.getRangeEnd(i)));
  if (!set.hasStrings())
    return;
  icu::UnicodeSetIterator strings(set);
  strings.skipToStrings();
  std::vector<std::uint32_t> codepoints;
  while (strings.next()) {
    const auto& string = strings.getString();
    codepoints.clear();
    for (int32_t i = 0; i < string.length(); i = string.moveIndex32(i, 1))
      codepoints.push_back(static_cast<std::uint32_t>(string.char32At(i)));
    add_string(sink, codepoints.data(), codepoints.size());
  }
}

// Resolves Unicode properties and \s with ICU, so their names, aliases and
// Unicode version stay those of the pinned ICU.
int ResolveSet(void* context, const std::uint8_t* pattern, std::size_t len,
               GufoJsonSchemaRangeFn add, GufoJsonSchemaStringFn add_string,
               void* sink) noexcept {
  auto& call = *static_cast<detail::JsonSchemaCall*>(context);
  return call.failure.Guard(GUFO_JSON_SCHEMA_SET_FAILED, [&] {
    UErrorCode status = U_ZERO_ERROR;
    const icu::UnicodeSet set(icu::UnicodeString::fromUTF8(std::string_view(
                                  reinterpret_cast<const char*>(pattern), len)),
                              0, nullptr, status);
    if (status == U_MEMORY_ALLOCATION_ERROR)
      throw std::bad_alloc();
    if (U_FAILURE(status))
      return GUFO_JSON_SCHEMA_SET_REJECTED;
    ReportSet(set, add, add_string, sink);
    return GUFO_JSON_SCHEMA_SET_RESOLVED;
  });
}

}  // namespace

// The Rust compiler owns the tables; this keeps them alive.
struct JsonSchemaRegex::Impl {
  struct Free {
    void operator()(GufoJsonSchemaRegex* regex) const {
      gufo_json_schema_regex_free(regex);
    }
  };
  std::unique_ptr<GufoJsonSchemaRegex, Free> handle;
  GufoJsonSchemaRegexTables tables{};

  [[nodiscard]] Id Next(Id state, Id cls) const {
    return tables.next[static_cast<std::size_t>(state) * tables.classes + cls];
  }
  [[nodiscard]] Id Distance(Id state) const { return tables.distance[state]; }
  void CheckState(Id state) const {
    if (state >= tables.states)
      throw std::out_of_range("JSON Schema regex state out of range");
  }
  // Whether class `cls` has a codepoint in [first, last].
  [[nodiscard]] bool ContainsSome(Id cls, Id first, Id last) const {
    const auto* begin = tables.class_ranges + 2 * tables.class_offsets[cls];
    const auto* end = tables.class_ranges + 2 * tables.class_offsets[cls + 1];
    // The first range whose end reaches `first`.
    std::size_t low = 0, high = (end - begin) / 2;
    while (low < high) {
      const auto mid = (low + high) / 2;
      if (begin[2 * mid + 1] < first)
        low = mid + 1;
      else
        high = mid;
    }
    return begin + 2 * low != end && begin[2 * low] <= last;
  }
};

JsonSchemaRegex::JsonSchemaRegex(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

std::shared_ptr<const JsonSchemaRegex> JsonSchemaRegex::Compile(
    std::span<const std::string> patterns, Id maximum) {
  std::vector<GufoJsonSchemaBytes> entries;
  entries.reserve(patterns.size());
  for (const auto& pattern : patterns)
    entries.push_back({reinterpret_cast<const std::uint8_t*>(pattern.data()),
                       pattern.size()});
  // Own the handle before anything else can throw.
  auto impl = std::make_shared<Impl>();
  detail::JsonSchemaCall call;
  GufoJsonSchemaRegex* handle = nullptr;
  const auto status = gufo_json_schema_regex_compile(
      entries.data(), entries.size(), maximum, ResolveSet,
      detail::JsonSchemaCall::Reason, &call, &handle);
  impl->handle.reset(handle);
  call.Check(status);
  call.Check(gufo_json_schema_regex_tables(handle, &impl->tables));
  return std::shared_ptr<const JsonSchemaRegex>(
      new JsonSchemaRegex(std::move(impl)));
}

bool JsonSchemaRegex::Accepting(Id state) const {
  if (state == kDead)
    return false;
  impl_->CheckState(state);
  return impl_->tables.accepting[state] != 0;
}
Id JsonSchemaRegex::MaximumSuffix() const {
  return impl_->tables.maximum_suffix;
}
Id JsonSchemaRegex::Advance(Id state, Id cp) const {
  if (state == kDead || cp > 0x10ffff)
    return kDead;
  const auto& tables = impl_->tables;
  const auto* starts = tables.lookup_starts;
  const auto index =
      std::upper_bound(starts, starts + tables.lookup_len + 1, cp) - starts - 1;
  const auto cls = tables.lookup_classes[index];
  if (cls == kDead)
    return kDead;
  impl_->CheckState(state);
  return impl_->Next(state, cls);
}
bool JsonSchemaRegex::CanFinish(Id state, Id minimum, Id maximum) const {
  if (state == kDead || minimum > maximum)
    return false;
  impl_->CheckState(state);
  const auto& tables = impl_->tables;
  const auto distance = impl_->Distance(state);
  if (distance == kDead || distance > maximum)
    return false;
  if (distance >= minimum)
    return true;
  // The tables never change; keep their pointers out of the loop.
  const auto* offsets = tables.successor_offsets;
  const auto* successors = tables.successors;
  const auto* distances = tables.distance;
  std::vector<Id> active{state};
  auto anchor = active;
  std::vector<bool> reachable(tables.states);
  Id steps = 0, block = 1, period = 0;
  bool cycle = false;
  while (steps < minimum) {
    std::fill(reachable.begin(), reachable.end(), false);
    for (auto current : active)
      for (auto i = offsets[current]; i < offsets[current + 1]; ++i) {
        const auto successor = successors[i];
        if (distances[successor] != kDead)
          reachable[successor] = true;
      }
    active.clear();
    for (Id next = 0; next < reachable.size(); ++next)
      if (reachable[next])
        active.push_back(next);
    if (active.empty())
      return false;
    ++steps;
    if (!cycle) {
      ++period;
      if (active == anchor) {
        // Brent cycle detection bounds scratch memory even for very large
        // minLength. Whole periods preserve the set of possible DFA states.
        steps += ((minimum - steps) / period) * period;
        cycle = true;
      } else if (period == block) {
        anchor = active;
        block *= 2;
        period = 0;
      }
    }
  }
  return std::ranges::any_of(active, [&](Id current) {
    return distances[current] <= maximum - minimum;
  });
}
bool JsonSchemaRegex::CanAdvance(Id state, Id first, Id last, Id minimum,
                                 Id maximum) const {
  if (state == kDead || first > last)
    return false;
  impl_->CheckState(state);
  for (Id cls = 0; cls < impl_->tables.classes; ++cls) {
    const auto next = impl_->Next(state, cls);
    if (next != kDead && impl_->ContainsSome(cls, first, last) &&
        CanFinish(next, minimum, maximum))
      return true;
  }
  return false;
}

}  // namespace gufo::sampling
