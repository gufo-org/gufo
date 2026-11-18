// Exercise the model-neutral expert store on shapes DeepSeek V4.1 does not
// have: a fused gate_up + down layout (two pieces) and a one-piece layout,
// with layer sizes that differ, through the store's own API only.
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/core/hip/expert_stream/store.hpp"

namespace {

using gufo::expert_stream::Layout;
using gufo::expert_stream::Overlap;
using gufo::expert_stream::Piece;
using gufo::expert_stream::Policy;
using gufo::expert_stream::Ranked;
using gufo::expert_stream::Store;

uint8_t Byte(uint32_t layer, uint32_t piece, uint32_t expert, uint64_t index) {
  return static_cast<uint8_t>(layer * 13 + piece * 31 + expert * 7 + index * 3);
}

/* `layers` layers of `experts` experts, each with pieces of `sizes[layer]`
 * bytes, written to two files (even layers to the first, odd to the second)
 * at unaligned offsets. */
struct Model {
  Layout layout;
  std::vector<std::vector<uint64_t>> sizes;  // [layer][piece]
  int fds[2] = {-1, -1};
  bool ok = true;

  Model(uint32_t experts, uint32_t top_k,
        std::vector<std::vector<uint64_t>> piece_sizes)
      : sizes(std::move(piece_sizes)) {
    layout.experts = experts;
    layout.top_k = top_k;
    layout.pieces = static_cast<uint32_t>(sizes[0].size());
    layout.max_batch_tokens = 64;
    for (int& fd : fds) {
      char path[] = "/tmp/gufo-expert-stream-XXXXXX";
      fd = mkstemp(path);
      if (fd < 0) {
        ok = false;
        return;
      }
      (void)unlink(path);
    }
    uint64_t offsets[2] = {96, 96};
    for (uint32_t l = 0; l < sizes.size() && ok; ++l) {
      const uint32_t file = l % 2;
      std::vector<Piece> pieces;
      for (uint32_t p = 0; p < layout.pieces && ok; ++p) {
        const uint64_t size = sizes[l][p];
        std::vector<uint8_t> bytes(size * experts);
        for (uint32_t e = 0; e < experts; ++e)
          for (uint64_t i = 0; i < size; ++i)
            bytes[e * size + i] = Byte(l, p, e, i);
        ok = pwrite(fds[file], bytes.data(), bytes.size(),
                    static_cast<off_t>(offsets[file])) ==
             static_cast<ssize_t>(bytes.size());
        pieces.push_back({fds[file], offsets[file], size});
        offsets[file] += bytes.size() + 8;  // a gap between tensors
      }
      layout.layers.push_back(pieces);
    }
  }
  ~Model() {
    for (int fd : fds)
      if (fd >= 0)
        (void)close(fd);
  }
  uint64_t Slot(uint32_t layer) const {
    uint64_t bytes = 0;
    for (uint64_t size : sizes[layer])
      bytes += size;
    return (bytes + 255) / 256 * 256;
  }
};

/* Every routed expert's pieces read back through the tables equal the
 * file's bytes. */
bool CheckTables(const Model& m, uint32_t layer,
                 const std::vector<int32_t>& ids, const void* const* tables) {
  for (uint32_t p = 0; p < m.layout.pieces; ++p) {
    std::vector<const uint8_t*> rows(m.layout.experts);
    if (!tables[p] ||
        hipMemcpy(rows.data(), tables[p], rows.size() * sizeof(rows[0]),
                  hipMemcpyDeviceToHost) != hipSuccess)
      return false;
    std::vector<uint8_t> bytes(m.sizes[layer][p]);
    for (int32_t e : ids) {
      if (e < 0)
        continue;
      if (hipMemcpy(bytes.data(), rows[e], bytes.size(),
                    hipMemcpyDeviceToHost) != hipSuccess)
        return false;
      for (uint64_t i = 0; i < bytes.size(); ++i)
        if (bytes[i] != Byte(layer, p, static_cast<uint32_t>(e), i))
          return false;
    }
  }
  return true;
}

bool ok = true;
void Check(bool pass, const char* label) {
  if (!pass)
    std::fprintf(stderr, "expert stream check failed: %s\n", label);
  ok = pass && ok;
}

struct Runner {
  const Model& m;
  Store& store;
  int32_t* routes = nullptr;
  const void* tables[gufo::expert_stream::kMaxPieces] = {};

  bool Run(uint32_t layer, const std::vector<int32_t>& ids,
           bool (*overlap)(void*, const Overlap*) = nullptr,
           void* context = nullptr) {
    if (ids.empty() || ids.size() % m.layout.top_k ||
        hipMemcpy(routes, ids.data(), ids.size() * sizeof(int32_t),
                  hipMemcpyHostToDevice) != hipSuccess)
      return false;
    return store.Prepare(layer, routes,
                         static_cast<uint32_t>(ids.size() / m.layout.top_k),
                         tables, overlap, context) &&
           CheckTables(m, layer, ids, tables);
  }
};

void FusedLayout(int32_t* routes) {
  // Five layers of 100 experts, top 8, gate_up and down pieces; odd layers
  // are smaller, as a build mixing block types would be.
  std::vector<std::vector<uint64_t>> sizes;
  for (uint32_t l = 0; l < 5; ++l)
    sizes.push_back(l % 2 ? std::vector<uint64_t>{640, 320}
                          : std::vector<uint64_t>{1280, 640});
  Model m(100, 8, sizes);
  Check(m.ok, "fused fixture");
  if (!m.ok)
    return;
  const uint64_t slot = m.Slot(0);
  Policy policy;
  policy.cache_limit_bytes = 8 * slot;
  policy.log_tag = "expert_stream_test";
  auto store = Store::Create(m.layout, policy);
  Check(store != nullptr, "fused create");
  if (!store)
    return;
  Runner run{m, *store, routes};
  const std::vector<int32_t> first{0, 1, 2, 3, 4, 5, 6, 7};
  Check(run.Run(0, first), "fused cold bytes");
  const auto cold = store->GetStats();
  Check(cold.uses == 8 && cold.hits == 0 && cold.bytes_read == 8 * (1280 + 640),
        "fused cold stats");
  Check(run.Run(0, {7, 6, 5, 4, 3, 2, 1, 0}), "fused warm bytes");
  const auto warm = store->GetStats();
  Check(warm.hits == 8 && warm.bytes_read == cold.bytes_read,
        "fused warm stats");
  // A cached expert's pieces sit back to back in its slot.
  {
    std::vector<const uint8_t*> gate_up(100), down(100);
    Check(hipMemcpy(gate_up.data(), run.tables[0], 100 * sizeof(void*),
                    hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(down.data(), run.tables[1], 100 * sizeof(void*),
                        hipMemcpyDeviceToHost) == hipSuccess &&
              down[3] == gate_up[3] + 1280,
          "fused slot layout");
  }
  // The overlap sees both pieces' tables and the split hit and miss rows.
  {
    Overlap info{};
    const void* seen[2] = {};
    struct Capture {
      Overlap* info;
      const void** seen;
    } capture{&info, seen};
    const auto callback = [](void* context, const Overlap* overlap) {
      auto* c = static_cast<Capture*>(context);
      *c->info = *overlap;
      c->seen[0] = overlap->table[0];
      c->seen[1] = overlap->table[1];
      return true;
    };
    const std::vector<int32_t> mixed{0, 1, 2, 3, 50, 51, 52, -1};
    Check(run.Run(0, mixed, callback, &capture), "fused split bytes");
    std::vector<int32_t> hit(8), miss(8);
    Check(info.rows == 8 && info.miss_rows == 3 && info.hit_ids &&
              info.miss_ids &&
              hipMemcpy(hit.data(), info.hit_ids, 8 * sizeof(int32_t),
                        hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(miss.data(), info.miss_ids, 8 * sizeof(int32_t),
                        hipMemcpyDeviceToHost) == hipSuccess &&
              hit == std::vector<int32_t>{0, 1, 2, 3, -1, -1, -1, -1} &&
              miss == std::vector<int32_t>{-1, -1, -1, -1, 50, 51, 52, -1} &&
              seen[0] == run.tables[0] && seen[1] == run.tables[1],
          "fused split rows");
  }
  // A smaller layer's experts share the larger slots once the cache is full,
  // and every layer reads from its own file.
  for (uint32_t l = 1; l < 5; ++l)
    Check(run.Run(l, {10, 11, 12, 13, 14, 15, 16, 17}), "fused other layers");
  Check(store->GetStats().cache_bytes <= 8 * slot + 2 * m.Slot(1),
        "fused cache bound");
  // Repeated use of one expert eventually makes it resident.
  for (int i = 0; i < 4; ++i)
    Check(run.Run(4, {90, 90, 90, 90, 90, 90, 90, 90}), "fused eviction bytes");
  const auto before = store->GetStats();
  Check(run.Run(4, {90, 90, 90, 90, 90, 90, 90, 90}) &&
            store->GetStats().hits == before.hits + 8,
        "fused eviction admits");
  Check(!run.Run(0, {0, 1, 2, 3, 4, 5, 6, 100}), "fused invalid route");
  Check(!run.Run(5, {0, 1, 2, 3, 4, 5, 6, 7}), "fused invalid layer");
}

void OnePieceLayout(int32_t* routes) {
  // Three layers of 16 experts, top 2, one piece each (a format that stores
  // an expert's tensors back to back); seeded from a ranking.
  Model m(16, 2, {{3000}, {2000}, {3000}});
  Check(m.ok, "one-piece fixture");
  if (!m.ok)
    return;
  Policy policy;
  policy.cache_limit_bytes = 64 * m.Slot(0);
  policy.log_tag = "expert_stream_test";
  auto store = Store::Create(m.layout, policy);
  Check(store != nullptr, "one-piece create");
  if (!store)
    return;
  Check(store->Seed({{1, 5, 10}, {2, 9, 8}}, "test ranking"), "one-piece seed");
  Check(!store->Seed({{3, 0, 1}}, "bad layer"),
        "one-piece seed rejects a layer");
  Check(!store->Seed({{0, 1, 1}, {0, 1, 2}}, "duplicate"),
        "one-piece seed rejects duplicates");
  Runner run{m, *store, routes};
  const auto seeded = store->GetStats();
  Check(seeded.uses == 0 && seeded.bytes_read == 0,
        "one-piece seed stays out of the stats");
  Check(run.Run(1, {5, 5}) && store->GetStats().hits == 2,
        "one-piece seeded hit");
  Check(run.Run(2, {9, 3}), "one-piece bytes");
}

void InvalidLayouts() {
  Policy policy;
  policy.log_tag = "expert_stream_test";
  Model m(16, 2, {{256, 256}});
  Layout too_many = m.layout;
  too_many.pieces = gufo::expert_stream::kMaxPieces + 1;
  Check(!Store::Create(too_many, policy), "rejects too many pieces");
  Layout mismatch = m.layout;
  mismatch.layers[0].pop_back();
  Check(!Store::Create(mismatch, policy), "rejects a layer with fewer pieces");
  Layout top_k = m.layout;
  top_k.top_k = 17;
  Check(!Store::Create(top_k, policy), "rejects top-k above the experts");
  Layout narrow = m.layout;
  narrow.table_experts = 15;
  Check(!Store::Create(narrow, policy),
        "rejects a table narrower than experts");
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
    return 77;
  int32_t* routes = nullptr;
  if (hipMalloc(&routes, 64 * 8 * sizeof(int32_t)) != hipSuccess)
    return 1;
  FusedLayout(routes);
  OnePieceLayout(routes);
  InvalidLayouts();
  (void)hipFree(routes);
  return ok ? 0 : 1;
}
