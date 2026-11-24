// Exercise the model-neutral expert store on shapes DeepSeek V4.1 does not
// have: a fused gate_up + down layout (two pieces) and a one-piece layout,
// with layer sizes that differ, through the store's own API only.
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "src/core/hip/expert_stream/store.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/mem/ledger.hpp"
#include "src/mem/remap_barrier.hpp"

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
  std::filesystem::path paths[2];
  gufo::platform::PlatformFile files[2] = {gufo::platform::kInvalidFile,
                                           gufo::platform::kInvalidFile};
  bool ok = true;

  Model(uint32_t experts, uint32_t top_k,
        std::vector<std::vector<uint64_t>> piece_sizes)
      : sizes(std::move(piece_sizes)) {
    layout.experts = experts;
    layout.top_k = top_k;
    layout.pieces = static_cast<uint32_t>(sizes[0].size());
    layout.max_batch_tokens = 64;
    static int models = 0;
    const std::string stem = "gufo-expert-stream-" +
                             std::to_string(gufo::platform::ProcessId()) + "-" +
                             std::to_string(models++);
    std::ofstream out[2];
    for (int f = 0; f < 2; ++f) {
      paths[f] = std::filesystem::temp_directory_path() /
                 (stem + "-" + std::to_string(f));
      out[f].open(paths[f], std::ios::binary | std::ios::trunc);
      ok = ok && out[f].good();
    }
    uint64_t offsets[2] = {96, 96};
    for (int f = 0; f < 2 && ok; ++f)
      out[f].write(std::string(offsets[f], '\0').data(), offsets[f]);
    for (uint32_t l = 0; l < sizes.size() && ok; ++l) {
      const uint32_t file = l % 2;
      std::vector<Piece> pieces;
      for (uint32_t p = 0; p < layout.pieces && ok; ++p) {
        const uint64_t size = sizes[l][p];
        std::vector<char> bytes(size * experts + 8);  // a gap between tensors
        for (uint32_t e = 0; e < experts; ++e)
          for (uint64_t i = 0; i < size; ++i)
            bytes[e * size + i] = static_cast<char>(Byte(l, p, e, i));
        ok = static_cast<bool>(out[file].write(
            bytes.data(), static_cast<std::streamsize>(bytes.size())));
        pieces.push_back({gufo::platform::kInvalidFile, offsets[file], size});
        offsets[file] += bytes.size();
      }
      layout.layers.push_back(pieces);
    }
    for (int f = 0; f < 2 && ok; ++f) {
      out[f].close();
      files[f] = gufo::platform::OpenReadFile(paths[f]);
      ok = gufo::platform::FileValid(files[f]);
    }
    for (uint32_t l = 0; l < layout.layers.size() && ok; ++l)
      for (Piece& piece : layout.layers[l])
        piece.file = files[l % 2];
  }
  ~Model() {
    for (int f = 0; f < 2; ++f) {
      if (gufo::platform::FileValid(files[f]))
        gufo::platform::CloseFile(files[f]);
      std::error_code ec;
      std::filesystem::remove(paths[f], ec);
    }
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
  hipStream_t stream = nullptr;

  bool Run(uint32_t layer, const std::vector<int32_t>& ids,
           bool (*overlap)(void*, const Overlap*) = nullptr,
           void* context = nullptr) {
    if (ids.empty() || ids.size() % m.layout.top_k ||
        hipMemcpyAsync(routes, ids.data(), ids.size() * sizeof(int32_t),
                       hipMemcpyHostToDevice, stream) != hipSuccess ||
        !store.Prepare(layer, routes,
                       static_cast<uint32_t>(ids.size() / m.layout.top_k),
                       tables, overlap, context, nullptr, stream))
      return false;
    if (!stream)
      return CheckTables(m, layer, ids, tables);
    // Read the tables back on the caller's stream, ordered after Prepare's
    // copies, and through the margin past every routed expert.
    for (uint32_t p = 0; p < m.layout.pieces; ++p) {
      std::vector<const uint8_t*> rows(m.layout.experts);
      if (hipMemcpyAsync(rows.data(), tables[p], rows.size() * sizeof(rows[0]),
                         hipMemcpyDeviceToHost, stream) != hipSuccess ||
          hipStreamSynchronize(stream) != hipSuccess)
        return false;
      std::vector<uint8_t> bytes(m.sizes[layer][p] +
                                 m.layout.read_margin_bytes);
      for (int32_t e : ids) {
        if (e < 0)
          continue;
        const uint64_t size =
            p + 1 == m.layout.pieces ? bytes.size() : m.sizes[layer][p];
        if (hipMemcpyAsync(bytes.data(), rows[e], size, hipMemcpyDeviceToHost,
                           stream) != hipSuccess ||
            hipStreamSynchronize(stream) != hipSuccess)
          return false;
        for (uint64_t i = 0; i < m.sizes[layer][p]; ++i)
          if (bytes[i] != Byte(layer, p, static_cast<uint32_t>(e), i))
            return false;
      }
    }
    return true;
  }
};

// Whether the store keeps its cache in device pages here, as it decides:
// virtual memory management and a device heap that covers host memory.
bool DevicePages() {
  int device = 0, vmm = 0;
  size_t heap_free = 0, heap_total = 0;
  const auto host = gufo::platform::QueryHostMemory();
  return hipGetDevice(&device) == hipSuccess &&
         hipDeviceGetAttribute(
             &vmm, hipDeviceAttributeVirtualMemoryManagementSupported,
             device) == hipSuccess &&
         vmm && hipMemGetInfo(&heap_free, &heap_total) == hipSuccess &&
         gufo::mem::HeapCoversHost(heap_total, host ? host->total_bytes : 0);
}

// A grant large enough that cache_limit_bytes alone bounds the cache.
constexpr uint64_t kAmpleGrant = uint64_t{1} << 40;

// Fit fills the cache; the checks of cold misses, admission and eviction
// start from an empty one that misses refill.
bool FitEmpty(Store& store) {
  if (!store.Fit(kAmpleGrant))
    return false;
  (void)store.Release(UINT64_MAX);
  return store.Restore() > 0 && store.GetStats().cache_bytes == 0;
}

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
  Check(FitEmpty(*store), "fused fit, then empty");
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
  // an expert's tensors back to back), with a ranking.
  Model m(16, 2, {{3000}, {2000}, {3000}});
  Check(m.ok, "one-piece fixture");
  if (!m.ok)
    return;
  Policy policy;
  policy.cache_limit_bytes = 64 * m.Slot(0);
  policy.log_tag = "expert_stream_test";
  policy.ranking_source = "test ranking";
  policy.ranking = {{3, 0, 1}};
  Check(Store::Create(m.layout, policy) == nullptr,
        "one-piece create rejects a ranked layer");
  policy.ranking = {{0, 1, 1}, {0, 1, 2}};
  Check(Store::Create(m.layout, policy) == nullptr,
        "one-piece create rejects a duplicate");
  policy.ranking = {{1, 5, 10}, {2, 9, 8}};
  auto store = Store::Create(m.layout, policy);
  Check(store != nullptr, "one-piece create");
  if (!store)
    return;
  Runner run{m, *store, routes};
  // Until Fit nothing is cached: misses read into the stage.
  Check(run.Run(1, {5, 5}) && store->GetStats().hits == 0 &&
            store->GetStats().cache_bytes == 0,
        "one-piece nothing cached before fit");
  Check(store->Release(1) == 0, "one-piece release needs a grant");
  // A grant takes the prefetch stage first, bounds the cache, and fills the
  // room left from the ranking, outside the statistics.
  const uint64_t stage = uint64_t{3000} * policy.prefetch_stage_rows;
  const auto before = store->GetStats();
  Check(store->Fit(stage + 4 * m.Slot(0)), "one-piece fit");
  Check(store->GetStats().cache_bytes > 0 &&
            store->GetStats().cache_bytes <= 4 * m.Slot(0),
        "one-piece fit bound");
  const auto fitted = store->GetStats();
  Check(fitted.uses == before.uses && fitted.bytes_read == before.bytes_read,
        "one-piece fit reads stay out of the stats");
  Check(run.Run(1, {5, 5}) && store->GetStats().hits == fitted.hits + 2,
        "one-piece fit caches the ranking");
  Check(run.Run(2, {9, 3}), "one-piece bytes");
  Check(store->GetStats().cache_bytes <= 4 * m.Slot(0),
        "one-piece fit bounds growth");
  // Release frees whole chunks (here the only one) and holds the cache
  // there: misses still read correctly and nothing is cached.
  const uint64_t held = store->GetStats().cache_bytes;
  Check(store->Release(1) == held && store->GetStats().cache_bytes == 0 &&
            store->ReleasedBytes() == held,
        "one-piece release frees the chunk");
  for (int pass = 0; pass < 2; ++pass)
    Check(run.Run(1, {5, 7}), "one-piece bytes after release");
  const auto released = store->GetStats();
  Check(released.cache_bytes == 0, "one-piece release holds the cache");
  Check(store->Release(1) == 0, "one-piece release of an empty cache");
  // Restore lets misses refill it, within the grant.
  Check(store->Restore() > 0 && store->Restore() == 0 &&
            store->ReleasedBytes() == 0,
        "one-piece restore raises the limit once");
  // Misses refill both size classes, two experts each, within the grant,
  // so one class maps new pages where the released chunk had its own: the
  // device may still reach the freed pages there until a remap barrier
  // settles.
  const auto settles = gufo::mem::RemapBarrier::Stats().settles;
  Check(run.Run(0, {4, 6}) && run.Run(0, {4, 6}) && run.Run(1, {5, 7}) &&
            run.Run(1, {5, 7}) && store->GetStats().hits == released.hits + 4,
        "one-piece restore regrows");
  Check(!DevicePages() || gufo::mem::RemapBarrier::Stats().settles > settles,
        "one-piece regrowth settles the remap barrier");
  Check(store->GetStats().cache_bytes <= 4 * m.Slot(0),
        "one-piece restore stays within the grant");
}

void ReleaseKeepsHottest(int32_t* routes) {
  // One size class filled in two chunks: the older holds the two experts
  // ranked first, the newer the next two, which then take every route.
  Model m(16, 2, {{3000}, {3000}});
  Check(m.ok, "release-keep fixture");
  if (!m.ok)
    return;
  Policy policy;
  policy.cache_limit_bytes = 64 * m.Slot(0);
  policy.log_tag = "expert_stream_test";
  policy.ranking_source = "test ranking";
  policy.ranking = {{0, 0, 4}, {0, 1, 3}, {0, 2, 2}, {0, 3, 1}};
  auto store = Store::Create(m.layout, policy);
  Check(store != nullptr, "release-keep create");
  if (!store)
    return;
  Runner run{m, *store, routes};
  const uint64_t stage = uint64_t{3000} * policy.prefetch_stage_rows;
  Check(store->Fit(stage + 2 * m.Slot(0)) &&
            store->Fit(stage + 4 * m.Slot(0)) &&
            store->GetStats().cache_bytes == 4 * m.Slot(0),
        "release-keep fit in two chunks");
  for (int i = 0; i < 4; ++i)
    Check(run.Run(0, {2, 3}), "release-keep bytes");
  // Release frees the newer chunk; its experts are the ones in use, so they
  // move into the older chunk and the two ranked first are evicted.
  const uint64_t held = store->GetStats().cache_bytes;
  Check(store->Release(1) == 2 * m.Slot(0) &&
            store->GetStats().cache_bytes == held - 2 * m.Slot(0),
        "release-keep frees the newer chunk");
  const auto released = store->GetStats();
  Check(run.Run(0, {2, 3}) && store->GetStats().hits == released.hits + 2,
        "release keeps the experts in use");
  const auto kept = store->GetStats();
  Check(run.Run(0, {0, 1}) && store->GetStats().hits == kept.hits,
        "release evicts the coldest");
}

void CallerStream(int32_t* routes) {
  // Two pieces, a read margin, and every call on a non-blocking stream: each
  // miss is staged and admitted through copies on that stream.
  Model m(16, 2, {{1000, 700}, {1200, 500}});
  m.layout.read_margin_bytes = 4096;
  Check(m.ok, "caller-stream fixture");
  if (!m.ok)
    return;
  Policy policy;
  policy.cache_limit_bytes = 6 * m.Slot(0);
  policy.log_tag = "expert_stream_test";
  auto store = Store::Create(m.layout, policy);
  Check(store != nullptr, "caller-stream create");
  if (!store)
    return;
  Check(FitEmpty(*store), "caller-stream fit, then empty");
  Runner run{m, *store, routes};
  Check(
      hipStreamCreateWithFlags(&run.stream, hipStreamNonBlocking) == hipSuccess,
      "caller-stream stream");
  for (uint32_t step = 0; step < 24; ++step) {
    // Mostly two experts per layer, the cache holds them; every fourth step
    // a cold one.
    const int32_t a = static_cast<int32_t>(step % 2);
    const int32_t b = step % 4 == 3 ? static_cast<int32_t>(4 + step % 12) : 1;
    Check(run.Run(step % 2, {a, b, b, a == b ? -1 : a}),
          "caller-stream bytes and margin");
  }
  const auto stats = store->GetStats();
  Check(stats.bytes_read > 0 && stats.hits > 0 && stats.hits < stats.uses,
        "caller-stream hits and misses");
  // The prefetch stage: each piece's widest layer, plus its margin.
  const uint64_t stage = uint64_t{1200 + 700} * policy.prefetch_stage_rows +
                         2 * uint64_t{m.layout.read_margin_bytes};
  Check(store->Fit(stage + 2 * m.Slot(0)), "caller-stream refit");
  Check(store->GetStats().cache_bytes <= 2 * m.Slot(0),
        "caller-stream fit counts the margin");
  Check(run.Run(1, {3, 4}), "caller-stream after fit");
  (void)hipStreamDestroy(run.stream);
}

void FailedPrepareCachesNothing(int32_t* routes) {
  // More misses than the demand stage holds, in a batch that admits them
  // all: they are read straight into cache slots. When the call fails (here
  // its overlapped work), none of them may stay cached.
  Model m(100, 8, {{1280, 640}});
  Check(m.ok, "failed-prepare fixture");
  if (!m.ok)
    return;
  Policy policy;
  policy.cache_limit_bytes = 128 * m.Slot(0);
  policy.log_tag = "expert_stream_test";
  auto store = Store::Create(m.layout, policy);
  Check(store != nullptr, "failed-prepare create");
  if (!store)
    return;
  Check(FitEmpty(*store), "failed-prepare fit, then empty");
  Runner run{m, *store, routes};
  std::vector<int32_t> ids(72);
  for (int32_t e = 0; e < 72; ++e)
    ids[e] = e;
  const auto refuse = [](void*, const Overlap*) { return false; };
  Check(!run.Run(0, ids, refuse), "failed-prepare fails");
  const auto failed = store->GetStats();
  Check(run.Run(0, ids) && store->GetStats().hits == failed.hits,
        "failed-prepare caches none of its reads");
  Check(run.Run(0, ids) && store->GetStats().hits == failed.hits + 72,
        "failed-prepare then caches them");
}

bool WaitForPrefetch(const Store& store) {
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!store.PrefetchIdle()) {
    if (std::chrono::steady_clock::now() > until)
      return false;
    std::this_thread::yield();
  }
  return true;
}

void PrefetchWindowWraps(int32_t* routes) {
  // Three layers, so the last and the first share their parity: a prefetch
  // for layer 0 submitted after the last layer's call must not write the
  // stage rows that call's table names. Nothing is cached (no Fit), so the
  // adopted experts stay in the prefetch stage. The table is wider than the
  // experts; its extra entries stay null.
  Model m(16, 2, {{3000}, {2000}, {3000}});
  m.layout.table_experts = 32;
  Check(m.ok, "prefetch-wrap fixture");
  if (!m.ok)
    return;
  Policy policy;
  policy.log_tag = "expert_stream_test";
  auto store = Store::Create(m.layout, policy);
  Check(store != nullptr, "prefetch-wrap create");
  if (!store)
    return;
  Runner run{m, *store, routes};
  Check(run.Run(0, {0, 1}) && run.Run(1, {0, 1}), "prefetch-wrap first layers");
  const int32_t last[] = {4, 5};
  Check(store->Prefetch(2, last, 2) && WaitForPrefetch(*store),
        "prefetch-wrap last layer prefetch");
  const auto before = store->GetStats();
  Check(run.Run(2, {4, 5}) &&
            store->GetStats().prefetch_bytes == before.prefetch_bytes + 6000,
        "prefetch-wrap last layer adopts");
  const void* const table = run.tables[0];
  std::vector<const void*> wide(32, &run);
  Check(hipMemcpy(wide.data(), table, wide.size() * sizeof(void*),
                  hipMemcpyDeviceToHost) == hipSuccess &&
            std::all_of(wide.begin() + 16, wide.end(),
                        [](const void* p) { return p == nullptr; }),
        "prefetch-wrap table entries past the experts are null");
  const int32_t first[] = {6, 7};
  Check(store->Prefetch(0, first, 2) && WaitForPrefetch(*store),
        "prefetch-wrap first layer prefetch");
  Check(CheckTables(m, 2, {4, 5}, &table),
        "prefetch-wrap keeps the last layer's rows");
  const auto wrapped = store->GetStats();
  Check(run.Run(0, {6, 7}) &&
            store->GetStats().prefetch_bytes == wrapped.prefetch_bytes + 6000,
        "prefetch-wrap first layer adopts");
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
  ReleaseKeepsHottest(routes);
  CallerStream(routes);
  FailedPrepareCachesNothing(routes);
  PrefetchWindowWraps(routes);
  InvalidLayouts();
  (void)hipFree(routes);
  return ok ? 0 : 1;
}
