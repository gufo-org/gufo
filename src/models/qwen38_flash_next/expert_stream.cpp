#include "src/models/qwen38_flash_next/expert_stream.hpp"

#include <charconv>
#include <unordered_map>

namespace gufo::models::qwen38_flash_next {
namespace {

void Fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

}  // namespace

std::optional<std::size_t> ParseByteSize(std::string_view text) noexcept {
  auto body = text;
  while (!body.empty() && (body.front() == ' ' || body.front() == '\t')) {
    body.remove_prefix(1);
  }
  while (!body.empty() && (body.back() == ' ' || body.back() == '\t')) {
    body.remove_suffix(1);
  }
  if (body.empty() || body.front() == '-') {
    return std::nullopt;
  }
  std::size_t digits = 0;
  while (digits < body.size() && body[digits] >= '0' && body[digits] <= '9') {
    ++digits;
  }
  if (digits == 0) {
    return std::nullopt;
  }
  std::size_t value = 0;
  const auto parsed = std::from_chars(body.data(), body.data() + digits, value);
  if (parsed.ec != std::errc{} || parsed.ptr != body.data() + digits) {
    return std::nullopt;
  }
  auto unit = body.substr(digits);
  while (!unit.empty() && (unit.front() == ' ' || unit.front() == '\t')) {
    unit.remove_prefix(1);
  }
  while (!unit.empty() && (unit.back() == ' ' || unit.back() == '\t')) {
    unit.remove_suffix(1);
  }
  const auto lower = [](char c) {
    return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
  };
  if (unit.empty() || (unit.size() == 1 && lower(unit[0]) == 'b')) {
    return value;
  }
  // The multiplier letter leads a `KiB`-style suffix and trails a bare one.
  const bool trailing_b = lower(unit.back()) == 'b';
  const bool iec = unit.size() > 1 && trailing_b;
  if (iec) {
    if (unit.size() > 3 || lower(unit[0]) == 'b') {
      return std::nullopt;
    }
    if (unit.size() == 3 && lower(unit[1]) != 'i') {
      return std::nullopt;
    }
  } else if (unit.size() != 1) {
    return std::nullopt;
  }
  std::size_t shift = 0;
  switch (lower(unit[0])) {
    case 'k':
      shift = 10;
      break;
    case 'm':
      shift = 20;
      break;
    case 'g':
      shift = 30;
      break;
    default:
      return std::nullopt;
  }
  const std::size_t multiplier = std::size_t{1} << shift;
  if (value > ~std::size_t{0} / multiplier) {
    return std::nullopt;
  }
  return value * multiplier;
}

bool ExpertStreamSupported(const LayerWeights& layer) noexcept {
  using core::GgmlType;
  const auto gate = layer.ffn_gate_exps.type;
  const auto up = layer.ffn_up_exps.type;
  const auto down = layer.ffn_down_exps.type;
  const bool gate_up = gate == GgmlType::kQ4_K || gate == GgmlType::kQ5_K ||
                       gate == GgmlType::kQ8_0;
  return !layer.ffn_gate_exps.empty() && !layer.ffn_up_exps.empty() &&
         !layer.ffn_down_exps.empty() && gate_up && up == gate &&
         (down == GgmlType::kQ5_1 || down == GgmlType::kQ8_0);
}

std::size_t ExpertBytes(const LayerWeights& layer) noexcept {
  return layer.ffn_gate_exps.ExpertBytes() + layer.ffn_up_exps.ExpertBytes() +
         layer.ffn_down_exps.ExpertBytes();
}

ExpertStreamPlan PlanExpertStreaming(const std::vector<LayerWeights>& layers,
                                     const LayerWeights* mtp,
                                     std::uint32_t num_experts,
                                     std::uint32_t num_experts_used,
                                     std::size_t budget_bytes,
                                     std::string* error) {
  ExpertStreamPlan plan;
  if (budget_bytes == 0) {
    return plan;
  }
  // Group the streamable layers by per-expert byte size; the dominant
  // class gets the cache so the fewest experts stay fully resident.
  std::unordered_map<std::size_t, std::size_t> class_counts;
  std::size_t streamable = 0;
  for (const auto& l : layers) {
    if (ExpertStreamSupported(l)) {
      ++class_counts[ExpertBytes(l)];
      ++streamable;
    }
  }
  if (streamable == 0) {
    Fail(error,
         "expert cache requested but no layer's routed-expert "
         "formats support streaming");
    return plan;
  }
  for (const auto& [bytes, count] : class_counts) {
    if (plan.expert_bytes == 0 || count > class_counts[plan.expert_bytes]) {
      plan.expert_bytes = bytes;
    }
  }
  plan.layers = class_counts[plan.expert_bytes];
  std::size_t resident = 0;
  for (const auto& l : layers) {
    if (ExpertStreamSupported(l) && ExpertBytes(l) != plan.expert_bytes) {
      resident += l.ffn_gate_exps.SizeBytes() + l.ffn_up_exps.SizeBytes() +
                  l.ffn_down_exps.SizeBytes();
    }
  }
  if (mtp != nullptr) {
    if (ExpertStreamSupported(*mtp) && ExpertBytes(*mtp) == plan.expert_bytes) {
      ++plan.layers;
    } else if (!mtp->ffn_gate_exps.empty()) {
      resident += mtp->ffn_gate_exps.SizeBytes() +
                  mtp->ffn_up_exps.SizeBytes() + mtp->ffn_down_exps.SizeBytes();
    }
  }
  plan.resident_expert_bytes = resident;
  const std::size_t per_layer = budget_bytes / plan.layers;
  if (per_layer < plan.expert_bytes) {
    Fail(error, "expert cache of " + std::to_string(budget_bytes) +
                    " bytes leaves less than one expert per streamed layer; "
                    "at least " +
                    std::to_string(plan.expert_bytes * plan.layers) +
                    " bytes are required");
    plan.expert_bytes = 0;
    plan.layers = 0;
    return plan;
  }
  plan.slots_per_layer =
      static_cast<std::uint32_t>(per_layer / plan.expert_bytes);
  plan.num_experts = num_experts;
  if (num_experts != 0 && plan.slots_per_layer >= num_experts) {
    // The budget covers every streamed expert: slabs of this size cost
    // the same device bytes as full residency while adding the host-side
    // id round-trip to every forward. Treat the request as satisfied by
    // the plain resident path.
    Fail(error, "expert cache of " + std::to_string(budget_bytes) +
                    " bytes already covers all " + std::to_string(num_experts) +
                    " experts per streamed layer; run without --expert-cache "
                    "to keep them fully resident");
    plan.expert_bytes = 0;
    plan.layers = 0;
    plan.slots_per_layer = 0;
    return plan;
  }
  if (plan.slots_per_layer < num_experts_used) {
    Fail(error, "expert cache of " + std::to_string(budget_bytes) +
                    " bytes leaves " + std::to_string(plan.slots_per_layer) +
                    " resident experts per streamed layer; one token routes " +
                    std::to_string(num_experts_used) +
                    " experts and their weights must fit together");
    plan.expert_bytes = 0;
    plan.layers = 0;
    return plan;
  }
  return plan;
}

}  // namespace gufo::models::qwen38_flash_next
