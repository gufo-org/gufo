#include "src/models/qwen38_flash_next/kernels/rocm/expert_stream_runtime.hpp"

#include <algorithm>
#include <cstring>

namespace gufo::models::qwen38_flash_next::rocm {

ExpertStreamCache::ExpertStreamCache(const DeviceModel& model,
                                     hipStream_t stream)
    : stream_(stream) {
  const auto build = [&](const DeviceLayer& l) {
    if (l.stream.slots == 0) {
      states_.push_back(nullptr);
      return;
    }
    if (l.stream.slots < model.config().num_experts_used) {
      error_ =
          "expert cache is too small to hold one token's routed experts (" +
          std::to_string(l.stream.slots) + " slots < " +
          std::to_string(model.config().num_experts_used) +
          " routed per token)";
      return;
    }
    auto st = std::make_unique<LayerState>();
    st->slab = l.stream;
    st->slot_of_expert.assign(model.config().num_experts, -1);
    st->entries.assign(l.stream.slots, Entry{});
    st->last_use.assign(l.stream.slots, 0);
    states_.push_back(std::move(st));
  };
  for (const auto& l : model.layers()) {
    build(l);
    if (!error_.empty()) {
      return;
    }
  }
  if (model.has_mtp()) {
    build(model.mtp());
    if (!error_.empty()) {
      return;
    }
  }
  bool any = false;
  std::size_t expert_bytes = 0;
  for (const auto& st : states_) {
    if (st == nullptr) {
      continue;
    }
    any = true;
    expert_bytes =
        std::max(expert_bytes,
                 st->slab.gate_bytes + st->slab.up_bytes + st->slab.down_bytes);
  }
  if (!any) {
    error_ = "expert streaming requested with no streamed layers";
    return;
  }
  ring_stride_ = expert_bytes;
  void* pinned = nullptr;
  if (hipHostMalloc(&pinned, ring_stride_ * kRing, hipHostMallocCoherent) ==
      hipSuccess) {
    ring_ = static_cast<std::uint8_t*>(pinned);
    ring_pinned_ = true;
  } else {
    ring_backing_.resize(ring_stride_ * kRing);
    ring_ = ring_backing_.data();
  }
  for (hipEvent_t& event : ring_free_) {
    if (hipEventCreateWithFlags(&event, hipEventDisableTiming) != hipSuccess) {
      error_ = "expert streaming staging event failed";
      return;
    }
  }
}

ExpertStreamCache::~ExpertStreamCache() {
  for (hipEvent_t event : ring_free_) {
    if (event != nullptr) {
      (void)hipEventDestroy(event);
    }
  }
  if (ring_pinned_ && ring_ != nullptr) {
    (void)hipHostFree(ring_);
  }
}

ExpertStreamCache::LayerState* ExpertStreamCache::State(
    const ExpertStreamLayer& slab) const {
  for (const auto& st : states_) {
    if (st != nullptr && st->slab.gate == slab.gate) {
      return st.get();
    }
  }
  return nullptr;
}

std::uint32_t ExpertStreamCache::Victim(const LayerState& st) const {
  // An empty slot is always free. Otherwise the least-recently-used slot
  // the current PrepareGroup has not assigned (its last_use still trails
  // the group's stamp) is the victim: every slot claimed during this call
  // holds an expert this group reads and must survive until the group's
  // GEMMs retire them with the stream.
  std::uint32_t best = UINT32_MAX;
  std::uint64_t best_use = UINT64_MAX;
  for (std::uint32_t s = 0; s < st.slab.slots; ++s) {
    if (st.entries[s].expert < 0) {
      return s;
    }
    if (st.last_use[s] == stamp_) {
      continue;
    }
    if (st.last_use[s] < best_use) {
      best_use = st.last_use[s];
      best = s;
    }
  }
  return best;
}

bool ExpertStreamCache::Load(LayerState& st, std::uint32_t slot,
                             std::int32_t expert) {
  // The ring entry may still be in flight from an earlier load; the ring
  // usually hides this wait behind the shared-expert GEMMs.
  hipEvent_t& free_event = ring_free_[ring_next_];
  if (hipEventQuery(free_event) == hipErrorNotReady &&
      hipEventSynchronize(free_event) != hipSuccess) {
    error_ = "expert streaming staging wait failed";
    return false;
  }
  std::uint8_t* dst = ring_ + ring_stride_ * ring_next_;
  const auto* gate =
      st.slab.src_gate + st.slab.gate_bytes * static_cast<std::size_t>(expert);
  const auto* up =
      st.slab.src_up + st.slab.up_bytes * static_cast<std::size_t>(expert);
  const auto* down =
      st.slab.src_down + st.slab.down_bytes * static_cast<std::size_t>(expert);
  std::memcpy(dst, gate, st.slab.gate_bytes);
  std::memcpy(dst + st.slab.gate_bytes, up, st.slab.up_bytes);
  std::memcpy(dst + st.slab.gate_bytes + st.slab.up_bytes, down,
              st.slab.down_bytes);
  const bool ok =
      hipMemcpyAsync(
          static_cast<std::uint8_t*>(st.slab.gate) + st.slab.gate_bytes * slot,
          dst, st.slab.gate_bytes, hipMemcpyHostToDevice,
          stream_) == hipSuccess &&
      hipMemcpyAsync(
          static_cast<std::uint8_t*>(st.slab.up) + st.slab.up_bytes * slot,
          dst + st.slab.gate_bytes, st.slab.up_bytes, hipMemcpyHostToDevice,
          stream_) == hipSuccess &&
      hipMemcpyAsync(
          static_cast<std::uint8_t*>(st.slab.down) + st.slab.down_bytes * slot,
          dst + st.slab.gate_bytes + st.slab.up_bytes, st.slab.down_bytes,
          hipMemcpyHostToDevice, stream_) == hipSuccess;
  if (!ok) {
    error_ = "expert streaming upload failed";
    return false;
  }
  (void)hipEventRecord(free_event, stream_);
  ring_next_ = (ring_next_ + 1) % kRing;
  ++stats_.loads;
  stats_.load_bytes +=
      st.slab.gate_bytes + st.slab.up_bytes + st.slab.down_bytes;
  return true;
}

bool ExpertStreamCache::PrepareGroup(const ExpertStreamLayer& slab,
                                     std::int32_t* ids_host,
                                     std::uint32_t ids_offset,
                                     std::uint32_t ids_count,
                                     std::int32_t* ids_device) {
  auto* st = State(slab);
  if (st == nullptr) {
    error_ = "expert streaming prepare on a non-streamed layer";
    return false;
  }
  ++stamp_;
  for (std::uint32_t i = 0; i < ids_count; ++i) {
    std::int32_t& id = ids_host[ids_offset + i];
    if (id < 0) {
      continue;  // RouterTopK's padding marker passes through untouched
    }
    ++stats_.lookups;
    std::int32_t slot = st->slot_of_expert[static_cast<std::size_t>(id)];
    if (slot < 0) {
      const std::uint32_t victim = Victim(*st);
      if (victim == UINT32_MAX) {
        error_ = "expert cache overflowed one routing group";
        return false;
      }
      const std::int32_t old = st->entries[victim].expert;
      if (old >= 0) {
        st->slot_of_expert[static_cast<std::size_t>(old)] = -1;
      }
      if (!Load(*st, victim, id)) {
        return false;
      }
      st->entries[victim].expert = id;
      st->slot_of_expert[static_cast<std::size_t>(id)] =
          static_cast<std::int32_t>(victim);
      slot = static_cast<std::int32_t>(victim);
    }
    st->last_use[static_cast<std::size_t>(slot)] = stamp_;
    id = slot;
  }
  const auto bytes = static_cast<std::size_t>(ids_count) * sizeof(std::int32_t);
  if (hipMemcpyAsync(ids_device + ids_offset, ids_host + ids_offset, bytes,
                     hipMemcpyHostToDevice, stream_) != hipSuccess) {
    error_ = "expert slot id upload failed";
    return false;
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
