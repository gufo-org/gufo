#include "src/core/npu/bfp_gemm.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "src/core/npu/host_ops.hpp"

// bfpgemm.py's layouts. A matrix pads to Kp (a multiple of 512 inputs) by Np
// (a multiple of 1024 outputs, G groups) and lies in the arena as the eight
// core columns read it: [column c][group][k block of 64][tile of 2304 words],
// tile = the 1x2 8x8-block shuffle of the 64 x 128 slice (Lc = 8 group + c).
//
// Rows are staged in blocks of up to 64 bands of 512 rows, laid out as
// [core row][band][K / 512][128][512] bf16. A band holds rows of one call; a
// block holds either consecutive bands of one call or one band of each of
// several calls of the same shape whose matrices lie at a constant arena
// stride (a batch, such as per-head attention products). A job is one band by
// up to `ng` groups, or one group by up to `nb` bands (the matrix advancing by
// `b_rep` words per band, 0 within one call); its output comes back as
// [column][tile][512][128] bf16, tiles in the job's order.
namespace gufo::npu {
namespace {

constexpr int kBand = 512, kGroup = 1024, kKc = 512, kKb = 64;
constexpr int kMaxRepeat = 64, kMaxWrap = 1023;  // shim BD limits
// kStripWords: one core row's 128 x 512 chunk of A as bfp16 (9 bytes per 8).
constexpr std::size_t kTileWords = 2304, kStripWords = 128 * kKc / 8 * 9 / 4;
constexpr std::size_t kABytes = std::size_t(64) << 20, kCBytes = std::size_t(64)
                                                                 << 20;
constexpr unsigned kArgA = 3, kArgW = 4, kArgC = 5;
constexpr std::size_t kLaneWords = std::size_t(1) << 18;
constexpr int kLanes = 4;
constexpr std::size_t kMaxArenaBytes = std::size_t(512) << 22;  // bfpgemm.py W
constexpr int kScratchIds = 1 << 16;  // ids per scratch region
// Scalars: ng nb kb tiles nb*kc ng*kb b_rep a_base a_row b_base b_col c_base
// c_col t1 t2 (t1, t2: the tiles core columns 2-3 and 4-7 compute and drain:
// all of them, or 1 when those columns have no outputs in the job; they then
// only pass the other tiles' A and B objects and drain one unused tile, since
// DMA sizes cannot be 0).
constexpr int kScalars = 15;
constexpr int kVar[3] = {7, 9, 11};  // the offsets; the rest is the shape

struct Matrix {
  int n{}, k{}, kp{}, groups{};
  std::size_t b_words{};
};

struct BandSrc {
  std::size_t call;
  int r0, nr;  // the call's rows [r0, r0 + nr)
};

struct Block {
  std::vector<BandSrc> bands;
  int kc;
  std::size_t a_words;
};

struct Job {
  Matrix m;  // of the job's first band
  std::size_t block;
  int g0, ng, b0, nb;
  std::size_t b_rep;  // arena words between the matrices of consecutive bands
  std::size_t c_words;
};

struct Dispatch {
  std::vector<Block> blocks;
  std::vector<Job> jobs;
  std::size_t a_end{}, c_end{};
};

// Staging buffers of one thread's dispatches.
struct Lane {
  Buffer a, c;
  Instructions ins;  // the current dispatch's stream
  // Rows last written per band of each staged block slot (word offset), with
  // the block's kc: padding rows only need zeroing where an earlier block
  // wrote.
  std::map<std::size_t, std::pair<int, std::vector<int>>> a_rows;
  bool busy{};
};

struct Region {
  std::size_t begin, end, cursor;  // arena words
  std::vector<Matrix> matrices;
};

Matrix Shape(int n, int k) {
  return Matrix{n, k, (k + kKc - 1) / kKc * kKc, (n + kGroup - 1) / kGroup, 0};
}

std::size_t Words(const Matrix& m) {
  return std::size_t(m.groups) * 8 * std::size_t(m.kp / kKb) * kTileWords;
}

float FromBf16(std::uint16_t h) {
  return std::bit_cast<float>(std::uint32_t(h) << 16);
}

}  // namespace

void BfpGemm::EncodeBlock(const float* values, std::uint8_t* out) {
  host::EncodeBfp16Block(values, out);
}

struct BfpGemm::Impl {
  Program program;
  DispatchLibrary generate;
  Buffer arena;
  // Guards the matrix tables, the arena cursor, the regions and templates.
  std::mutex mutex;
  std::size_t arena_used{};
  std::deque<Matrix> matrices;
  std::deque<Region> regions;
  // Per job shape: the stream at zero offsets and how its words change with
  // the A, W and C offsets (dispatch_generate is linear in them).
  struct Template {
    std::vector<std::uint32_t> words;
    std::vector<Instructions::Patch> patches[3];
  };
  std::map<std::array<std::int32_t, kScalars - 3>, Template> templates;
  Lane lanes[kLanes];
  std::mutex lane_mutex;
  std::condition_variable lane_free;
  bool profile{std::getenv("GUFO_NPU_PROFILE") != nullptr};
  // NPU busy time: the union of in-flight dispatches (profile only).
  std::mutex busy_mutex;
  int inflight{};
  std::chrono::steady_clock::time_point busy_since;
  double busy_us{};
  void Begin() {
    std::lock_guard lock(busy_mutex);
    if (inflight++ == 0)
      busy_since = std::chrono::steady_clock::now();
  }
  void End() {
    std::lock_guard lock(busy_mutex);
    if (--inflight == 0)
      busy_us += std::chrono::duration<double, std::micro>(
                     std::chrono::steady_clock::now() - busy_since)
                     .count();
  }

  // Packs matrix `m` (placed at m.b_words) from row-major f32 rows
  // w[o * ld + i] into the arena.
  // Encodes tile t (of 8 groups kb) of matrix `m` from rows w[o * ld + i].
  void EncodeTile(const Matrix& m, const float* w, std::size_t ld,
                  std::size_t t, std::uint8_t* tile) {
    const int kb = m.kp / kKb;
    const int lr = int(t % std::size_t(kb));
    const int cg = int(t / std::size_t(kb));
    const int col = cg / m.groups, g = cg % m.groups;
    const int lc = 8 * g + col;
    auto* base = reinterpret_cast<std::uint8_t*>(arena.data()) + m.b_words * 4;
    std::uint8_t* out =
        base + ((std::size_t(col) * std::size_t(m.groups) + std::size_t(g)) *
                    std::size_t(kb) +
                std::size_t(lr)) *
                   kTileWords * 4;
    if (lc * 128 >= m.n || lr * kKb >= m.k) {  // all padding
      std::memset(out, 0, kTileWords * 4);
      return;
    }
    std::uint8_t* dst = tile;
    float v[8];
    // [superblock 8][block row 8][b 2][c 8]: 8 consecutive k at one n.
    for (int sb = 0; sb < 8; ++sb)
      for (int br = 0; br < 8; ++br) {
        const int k0 = lr * kKb + br * 8;
        for (int b = 0; b < 2; ++b)
          for (int cc = 0; cc < 8; ++cc, dst += 9) {
            const int nn = lc * 128 + sb * 16 + b * 8 + cc;
            const float* src = w + std::size_t(nn) * ld + std::size_t(k0);
            if (nn < m.n && k0 + 8 <= m.k) {
              EncodeBlock(src, dst);
              continue;
            }
            if (nn >= m.n || k0 >= m.k) {
              std::memset(dst, 0, 9);
              continue;
            }
            for (int r = 0; r < 8; ++r)
              v[r] = (nn < m.n && k0 + r < m.k) ? src[r] : 0.0f;
            EncodeBlock(v, dst);
          }
      }
    std::memcpy(out, tile, kTileWords * 4);
  }

  static std::size_t Tiles(const Matrix& m) {
    return std::size_t(8) * std::size_t(m.groups) * std::size_t(m.kp / kKb);
  }

  // Packs matrix `m` (placed at m.b_words) from row-major f32 rows
  // w[o * ld + i] into the arena.
  void Encode(const Matrix& m, const float* w, std::size_t ld) {
    host::ParallelFor(Tiles(m), 4, [&](std::size_t lo, std::size_t hi) {
      std::vector<std::uint8_t> tile(kTileWords * 4);
      for (std::size_t t = lo; t < hi; ++t)
        EncodeTile(m, w, ld, t, tile.data());
    });
    arena.ToDevice(m.b_words * 4, Words(m) * 4);
  }

  int Upload(int n, int k, const float* w, std::size_t ld) {
    if (n <= 0 || k <= 0)
      throw Error("BfpGemm: empty matrix");
    Matrix m = Shape(n, k);
    if (m.kp / kKb > kMaxWrap)
      throw Error("BfpGemm: k is too large");
    int id;
    {
      std::lock_guard lock(mutex);
      const std::size_t limit =
          regions.empty() ? arena.size() / 4 : regions.back().begin;
      if (arena_used + Words(m) > limit)
        throw Error("BfpGemm: the weight arena is full");
      m.b_words = arena_used;
      arena_used += Words(m);
      matrices.push_back(m);
      id = int(matrices.size()) - 1;
    }
    Encode(m, w, ld);
    return id;
  }

  // Under `mutex`.
  Matrix Resolve(int id) const {
    if (id >= 0)
      return matrices.at(std::size_t(id));
    const int r = (-id - 1) / kScratchIds, i = (-id - 1) % kScratchIds;
    return regions.at(std::size_t(r)).matrices.at(std::size_t(i));
  }

  // Appends `block` and its jobs to the plan. A job that no longer fits
  // the C buffer starts a new dispatch, which stages the block again.
  static void Add(std::vector<Dispatch>& plan, const Block& block,
                  const std::vector<Job>& jobs) {
    const std::size_t a_words =
        block.bands.size() * 4 * std::size_t(block.kc) * kStripWords;
    if (a_words * 4 > kABytes)
      throw Error("BfpGemm: one block exceeds the staging buffers");
    bool staged = false;
    for (Job j : jobs) {
      const std::size_t words =
          std::size_t(j.ng) * std::size_t(j.nb) * 8 * std::size_t(kBand) * 64;
      if (words * 4 > kCBytes)
        throw Error("BfpGemm: one job exceeds the staging buffers");
      Dispatch* d = &plan.back();
      const bool fits =
          (staged ? d->a_end : d->a_end + a_words) * 4 <= kABytes &&
          (d->c_end + words) * 4 <= kCBytes;
      if (!fits) {
        if (!d->jobs.empty()) {
          plan.emplace_back();
          d = &plan.back();
        }
        staged = false;
      }
      if (!staged) {
        d->blocks.push_back(block);
        d->blocks.back().a_words = d->a_end;
        d->a_end += a_words;
        staged = true;
      }
      j.block = d->blocks.size() - 1;
      j.c_words = d->c_end;
      d->c_end += words;
      d->jobs.push_back(j);
    }
  }

  std::vector<Dispatch> Plan(std::span<const Call> calls,
                             const std::vector<Matrix>& resolved) const {
    std::vector<Dispatch> plan(1);
    for (std::size_t i = 0; i < calls.size();) {
      const Matrix& m = resolved[i];
      const int kc = m.kp / kKc, kb = m.kp / kKb;
      // A batch: single-band calls of one output group and the same padded
      // k, matrices at a constant positive stride.
      std::size_t end = i + 1;
      if (calls[i].rows <= kBand && m.groups == 1) {
        const std::size_t batch_max = std::min<std::size_t>(
            kMaxRepeat,
            kABytes / (std::size_t(4) * std::size_t(kc) * kStripWords * 4));
        while (end < calls.size() && end - i < batch_max &&
               std::size_t(kc) * (end - i + 1) <= std::size_t(kMaxWrap)) {
          const Matrix& e = resolved[end];
          if (calls[end].rows > kBand || calls[end].rows <= 0 || e.kp != m.kp ||
              e.groups != 1)
            break;
          const std::size_t stride = resolved[i + 1].b_words - m.b_words;
          if (resolved[i + 1].b_words <= m.b_words ||
              e.b_words != m.b_words + stride * (end - i))
            break;
          ++end;
        }
      }
      if (end - i > 1) {
        Block block{{}, kc, 0};
        for (std::size_t c = i; c < end; ++c)
          block.bands.push_back({c, 0, calls[c].rows});
        Job job{
            m, 0, 0, 1, 0, int(end - i), resolved[i + 1].b_words - m.b_words,
            0};
        Add(plan, block, {job});
        i = end;
        continue;
      }
      const int bands_total = (calls[i].rows + kBand - 1) / kBand;
      // Bands per block: the shim's repeat limit and the A staging buffer.
      const int bands_max = std::max(
          1, std::min<int>(kMaxRepeat,
                           int(kABytes / (std::size_t(4) * std::size_t(kc) *
                                          kStripWords * 4))));
      for (int bb = 0; bb < bands_total; bb += bands_max) {
        const int bands = std::min(bands_max, bands_total - bb);
        Block block{{}, kc, 0};
        for (int b = 0; b < bands; ++b) {
          const int r0 = (bb + b) * kBand;
          block.bands.push_back({i, r0, std::min(kBand, calls[i].rows - r0)});
        }
        // One band by many groups, or one group by many bands: whichever
        // takes fewer jobs.
        std::vector<Job> jobs;
        if (bands == 1 || m.groups < bands) {
          const int ng_max = std::max(1, std::min(kMaxRepeat, kMaxWrap / kb));
          for (int b = 0; b < bands; ++b)
            for (int g0 = 0; g0 < m.groups; g0 += ng_max)
              jobs.push_back(
                  Job{m, 0, g0, std::min(ng_max, m.groups - g0), b, 1, 0, 0});
        } else {
          for (int g = 0; g < m.groups; ++g)
            jobs.push_back(Job{m, 0, g, 1, 0, bands, 0, 0});
        }
        Add(plan, block, jobs);
      }
      ++i;
    }
    return plan;
  }

  std::vector<std::uint32_t> Generate(const std::int32_t* s) const {
    try {
      const auto w =
          generate.Generate(std::span<const std::int32_t>(s, kScalars));
      if (w.size() > 4)
        return {w.begin(), w.end()};
    } catch (const Error&) {
    }
    std::string list;
    for (int i = 0; i < kScalars; ++i)
      list += (i ? " " : "") + std::to_string(s[i]);
    throw Error("BfpGemm: instruction generation failed for " + list);
  }

  // Under `mutex`: the template of one job's shape, checked against a direct
  // generation at its first offsets.
  const Template& TemplateFor(const std::int32_t* s) {
    std::array<std::int32_t, kScalars - 3> key{};
    for (int i = 0, j = 0; i < kScalars; ++i)
      if (i != kVar[0] && i != kVar[1] && i != kVar[2])
        key[std::size_t(j++)] = s[i];
    if (auto it = templates.find(key); it != templates.end())
      return it->second;
    std::int32_t z[kScalars];
    std::copy(s, s + kScalars, z);
    for (int v : kVar)
      z[v] = 0;
    Template t;
    t.words = Generate(z);
    for (int i = 0; i < 3; ++i) {
      std::int32_t one[kScalars];
      std::copy(z, z + kScalars, one);
      one[kVar[i]] = 1;
      t.patches[i] = DiffInstructions(t.words, Generate(one));
    }
    std::vector<std::uint32_t> check = t.words;
    for (int i = 0; i < 3; ++i)
      for (const auto& p : t.patches[i])
        check[p.index] += std::uint32_t(s[kVar[i]]) * p.delta;
    if (check != Generate(s))
      throw Error("BfpGemm: instruction streams are not linear in offsets");
    return templates.emplace(key, std::move(t)).first->second;
  }

  // Writes the concatenated stream of a dispatch's jobs into the lane.
  void Stream(Lane& lane, const std::vector<std::int32_t>& scalars) {
    std::vector<std::uint32_t> txn(4, 0);
    std::uint32_t ops = 0;
    {
      std::lock_guard lock(mutex);
      for (std::size_t i = 0; i < scalars.size(); i += kScalars) {
        const std::int32_t* s = scalars.data() + i;
        const Template& t = TemplateFor(s);
        const std::size_t at = txn.size();
        txn.insert(txn.end(), t.words.begin() + 4, t.words.end());
        for (int v = 0; v < 3; ++v)
          for (const auto& p : t.patches[v])
            txn[at + p.index - 4] += std::uint32_t(s[kVar[v]]) * p.delta;
        if (ops == 0)
          txn[0] = t.words[0], txn[1] = t.words[1];
        ops += t.words[2];
      }
    }
    txn[2] = ops;
    txn[3] = std::uint32_t(txn.size() * 4);
    if (txn.size() > lane.ins.capacity())
      lane.ins = program.AllocateInstructions(txn.size() * 2);
    lane.ins.Assign(txn);
  }

  // Stages the blocks' rows as bfp16 in the order the cores read them: per
  // core row r (128 rows), band and 512-wide K chunk, 32 sub-tile objects
  // [k block 8][32-row sub-tile 4] of 2304 bytes, each [row pair 2][column
  // block 8][half 2][row 8] groups of 8 k values.
  void Pack(Lane& lane, const Dispatch& d, std::span<const Call> calls,
            const std::vector<Matrix>& resolved) {
    struct Item {
      const Block* block;
      int band, i0, i1;
    };
    std::vector<Item> items;
    for (const Block& b : d.blocks) {
      std::vector<int> rows;
      for (const BandSrc& s : b.bands)
        rows.push_back(s.nr);
      auto it = lane.a_rows.find(b.a_words);
      const bool known = it != lane.a_rows.end() && it->second.first == b.kc &&
                         it->second.second.size() == rows.size();
      for (std::size_t band = 0; band < rows.size(); ++band) {
        const int end =
            known ? std::max(rows[band], it->second.second[band]) : kBand;
        for (int i0 = 0; i0 < end; i0 += 32)
          items.push_back({&b, int(band), i0, std::min(end, i0 + 32)});
      }
      lane.a_rows[b.a_words] = {b.kc, std::move(rows)};
    }
    auto* a8 = reinterpret_cast<std::uint8_t*>(lane.a.data());
    host::ParallelFor(items.size(), 1, [&](std::size_t lo, std::size_t hi) {
      std::vector<float> row;
      for (std::size_t t = lo; t < hi; ++t) {
        const Item& it = items[t];
        const Block& b = *it.block;
        const BandSrc& src = b.bands[std::size_t(it.band)];
        const Call& call = calls[src.call];
        const std::size_t k = std::size_t(resolved[src.call].k);
        const std::size_t bands = b.bands.size(), kc = std::size_t(b.kc);
        row.assign(kc * kKc, 0.0f);
        for (int i = it.i0; i < it.i1; ++i) {
          if (i < src.nr) {
            const std::size_t at = std::size_t(src.r0 + i) * k;
            if (call.x16)
              host::Bf16ToFloat(call.x16 + at, row.data(), k);
            else
              std::copy(call.x + at, call.x + at + k, row.data());
          } else {
            std::fill(row.begin(), row.end(), 0.0f);
          }
          const std::size_t r = std::size_t(i) / 128, ri = std::size_t(i) % 128;
          const std::size_t sub = ri / 32, rs = ri % 32;
          const std::size_t rp = rs / 16, half = (rs % 16) / 8, rr = rs % 8;
          for (std::size_t q = 0; q < kc; ++q) {
            std::uint8_t* chunk =
                a8 +
                (b.a_words +
                 ((r * bands + std::size_t(it.band)) * kc + q) * kStripWords) *
                    4;
            for (std::size_t kb = 0; kb < 8; ++kb) {
              std::uint8_t* obj = chunk + (kb * 4 + sub) * 2304;
              const float* v = row.data() + q * kKc + kb * kKb;
              for (std::size_t cb = 0; cb < 8; ++cb)
                host::EncodeBfp16Block(
                    v + cb * 8, obj + ((rp * 8 + cb) * 2 + half) * 72 + rr * 9);
            }
          }
        }
      }
    });
  }

  // Copies the jobs' output tiles into their calls' rows: one work item per
  // (job, column, tile) and block of 64 rows. The buffer is uncached, so each
  // item copies its rows out once before converting.
  void Unpack(Lane& lane, const Dispatch& d, std::span<const Call> calls,
              const std::vector<Matrix>& resolved) {
    struct Item {
      const Job* job;
      int col, tile, r0, r1;
    };
    std::vector<Item> items;
    for (const Job& j : d.jobs) {
      const Block& b = d.blocks[j.block];
      for (int col = 0; col < 8; ++col)
        for (int t = 0; t < j.ng * j.nb; ++t) {
          const int g = j.g0 + (j.nb == 1 ? t : 0);
          const int band = j.b0 + (j.nb == 1 ? 0 : t);
          const BandSrc& src = b.bands[std::size_t(band)];
          if ((g * 8 + col) * 128 >= resolved[src.call].n)
            continue;
          const int rows = src.nr;
          for (int r0 = 0; r0 < rows; r0 += 64)
            items.push_back({&j, col, t, r0, std::min(rows, r0 + 64)});
        }
    }
    const auto* base = reinterpret_cast<const std::uint16_t*>(lane.c.data());
    host::ParallelFor(items.size(), 1, [&](std::size_t lo, std::size_t hi) {
      std::vector<std::uint16_t> tile(std::size_t(64) * 128);
      for (std::size_t t = lo; t < hi; ++t) {
        const Item& it = items[t];
        const Job& j = *it.job;
        const Block& b = d.blocks[j.block];
        const int g = j.g0 + (j.nb == 1 ? it.tile : 0);
        const int band = j.b0 + (j.nb == 1 ? 0 : it.tile);
        const BandSrc& src = b.bands[std::size_t(band)];
        const Call& call = calls[src.call];
        const int n = resolved[src.call].n, n0 = (g * 8 + it.col) * 128;
        const int w = std::min(128, n - n0);
        const std::uint16_t* p =
            base + j.c_words * 2 +
            ((std::size_t(it.col) * std::size_t(j.ng * j.nb) +
              std::size_t(it.tile)) *
                 kBand +
             std::size_t(it.r0)) *
                128;
        const std::size_t rows = std::size_t(it.r1 - it.r0);
        std::memcpy(tile.data(), p, rows * 128 * 2);
        if (call.epilogue) {
          (*call.epilogue)(src.r0 + it.r0, int(rows), n0, w, tile.data());
          continue;
        }
        for (std::size_t i = 0; i < rows; ++i)
          host::Bf16ToFloat(
              tile.data() + i * 128,
              call.y + std::size_t(src.r0 + it.r0 + int(i)) * std::size_t(n) +
                  std::size_t(n0),
              std::size_t(w));
      }
    });
  }

  void Run(std::span<const Call> calls) {
    std::vector<Matrix> resolved;
    {
      std::lock_guard lock(mutex);
      for (const Call& call : calls)
        resolved.push_back(Resolve(call.id));
    }
    const std::vector<Dispatch> plan = Plan(calls, resolved);
    Lane* lane;
    {
      std::unique_lock lock(lane_mutex);
      lane_free.wait(lock, [&] {
        return std::any_of(std::begin(lanes), std::end(lanes),
                           [](const Lane& l) { return !l.busy; });
      });
      lane = std::find_if(std::begin(lanes), std::end(lanes),
                          [](const Lane& l) { return !l.busy; });
      lane->busy = true;
    }
    struct Release {
      Impl* impl;
      Lane* lane;
      ~Release() {
        {
          std::lock_guard lock(impl->lane_mutex);
          lane->busy = false;
        }
        impl->lane_free.notify_all();
      }
    } release{this, lane};

    using Clock = std::chrono::steady_clock;
    auto us = [](Clock::time_point t) {
      return std::chrono::duration<double, std::micro>(Clock::now() - t)
          .count();
    };
    for (const Dispatch& d : plan) {
      if (d.jobs.empty())
        continue;
      std::vector<std::int32_t> scalars;
      for (const Job& j : d.jobs) {
        const Block& b = d.blocks[j.block];
        const int kc = j.m.kp / kKc, kb = j.m.kp / kKb;
        // Columns past the job's widest output range idle (no C drained).
        int width = 0;
        for (int t = 0; t < j.nb; ++t)
          width =
              std::max(width, resolved[b.bands[std::size_t(j.b0 + t)].call].n);
        const int cols =
            j.ng > 1 ? 8
                     : std::clamp((width - j.g0 * kGroup + 127) / 128, 1, 8);
        const int tiles = j.ng * j.nb;
        const std::int32_t s[kScalars] = {
            j.ng,
            j.nb,
            kb,
            j.ng * j.nb,
            j.nb * kc,
            j.ng * kb,
            std::int32_t(j.b_rep),
            std::int32_t(b.a_words +
                         std::size_t(j.b0) * std::size_t(kc) * kStripWords),
            std::int32_t(b.bands.size() * std::size_t(kc) * kStripWords),
            std::int32_t(j.m.b_words +
                         std::size_t(j.g0) * std::size_t(kb) * kTileWords),
            std::int32_t(std::size_t(j.m.groups) * std::size_t(kb) *
                         kTileWords),
            std::int32_t(j.c_words),
            j.ng * j.nb * kBand * 128 / 2,
            cols > 2 ? tiles : 1,
            cols > 4 ? tiles : 1};
        scalars.insert(scalars.end(), s, s + kScalars);
      }
      auto t = Clock::now();
      Pack(*lane, d, calls, resolved);
      Stream(*lane, scalars);
      lane->a.ToDevice(0, d.a_end * 4);
      const double t_pack = us(t);
      t = Clock::now();
      if (profile)
        Begin();
      program.Start(lane->ins, {&lane->a, &arena, &lane->c}).Wait();
      if (profile)
        End();
      const double t_npu = us(t);
      t = Clock::now();
      lane->c.FromDevice(0, d.c_end * 4);
      Unpack(*lane, d, calls, resolved);
      if (profile)
        std::fprintf(stderr,
                     "bfpgemm: %zu jobs, pack %.1f npu %.1f unpack %.1f us\n",
                     d.jobs.size(), t_pack, t_npu, us(t));
    }
  }
};

BfpGemm BfpGemm::Open(Device& device, const std::filesystem::path& dir,
                      std::size_t arena_bytes) {
  BfpGemm g;
  g.impl_ = std::make_unique<Impl>();
  auto& m = *g.impl_;
  m.program = Program::Load(device, dir / "bfpgemm.xclbin");
  m.generate = DispatchLibrary::Open(dir / "bfpgemm_dispatch.so", kScalars);
  for (Lane& lane : m.lanes) {
    lane.a = m.program.Allocate(kABytes, kArgA);
    lane.c = m.program.Allocate(kCBytes, kArgC);
    lane.ins = m.program.AllocateInstructions(kLaneWords);
  }
  if (arena_bytes > kMaxArenaBytes)
    throw Error("BfpGemm: the weight arena exceeds the design's 2 GiB");
  m.arena = m.program.Allocate(arena_bytes, kArgW);
  return g;
}

BfpGemm::BfpGemm() = default;
BfpGemm::~BfpGemm() = default;
BfpGemm::BfpGemm(BfpGemm&&) noexcept = default;
BfpGemm& BfpGemm::operator=(BfpGemm&&) noexcept = default;

int BfpGemm::Upload(int n, int k, std::span<const float> weights) {
  if (weights.size() < std::size_t(n) * std::size_t(k))
    throw Error("BfpGemm: too few weights");
  return impl_->Upload(n, k, weights.data(), std::size_t(k));
}

int BfpGemm::UploadBf16(int n, int k, std::span<const std::uint16_t> weights) {
  if (weights.size() < std::size_t(n) * std::size_t(k))
    throw Error("BfpGemm: too few weights");
  std::vector<float> rows(std::size_t(n) * std::size_t(k));
  host::ParallelFor(std::size_t(n), 16, [&](std::size_t lo, std::size_t hi) {
    host::Bf16ToFloat(weights.data() + lo * std::size_t(k),
                      rows.data() + lo * std::size_t(k),
                      (hi - lo) * std::size_t(k));
  });
  return impl_->Upload(n, k, rows.data(), std::size_t(k));
}

BfpGemm::Scratch BfpGemm::NewScratch(std::size_t bytes) {
  auto& m = *impl_;
  std::lock_guard lock(m.mutex);
  const std::size_t words = (bytes + 3) / 4;
  const std::size_t end =
      m.regions.empty() ? m.arena.size() / 4 : m.regions.back().begin;
  if (end < m.arena_used + words)
    throw Error("BfpGemm: no arena left for a scratch region");
  m.regions.push_back(Region{end - words, end, end - words, {}});
  Scratch s;
  s.owner_ = this;
  s.index_ = int(m.regions.size()) - 1;
  return s;
}

int BfpGemm::Scratch::Upload(int n, int k, const float* weights,
                             std::size_t row_stride, std::size_t col_stride) {
  auto& m = *owner_->impl_;
  if (n <= 0 || k <= 0)
    throw Error("BfpGemm: empty matrix");
  Matrix mat = Shape(n, k);
  int id;
  {
    std::lock_guard lock(m.mutex);
    Region& r = m.regions.at(std::size_t(index_));
    if (r.cursor + Words(mat) > r.end ||
        r.matrices.size() >= std::size_t(kScratchIds))
      throw Error("BfpGemm: the scratch region is full (" +
                  std::to_string(r.matrices.size()) + " matrices, " +
                  std::to_string((r.cursor - r.begin) * 4 >> 20) + " of " +
                  std::to_string((r.end - r.begin) * 4 >> 20) + " MiB used, " +
                  std::to_string(n) + " x " + std::to_string(k) + " asked)");
    mat.b_words = r.cursor;
    r.cursor += Words(mat);
    r.matrices.push_back(mat);
    id = -(1 + index_ * kScratchIds + int(r.matrices.size()) - 1);
  }
  if (col_stride == 1) {
    m.Encode(mat, weights, row_stride);
  } else {
    // A transposed view: gather it into rows first.
    std::vector<float> rows(std::size_t(n) * std::size_t(k));
    host::ParallelFor(std::size_t(k), 64, [&](std::size_t lo, std::size_t hi) {
      for (std::size_t i = lo; i < hi; ++i)
        for (std::size_t o = 0; o < std::size_t(n); ++o)
          rows[o * std::size_t(k) + i] =
              weights[o * row_stride + i * col_stride];
    });
    m.Encode(mat, rows.data(), std::size_t(k));
  }
  return id;
}

std::vector<int> BfpGemm::Scratch::UploadMany(std::span<const Spec> specs) {
  auto& m = *owner_->impl_;
  std::vector<Matrix> mats;
  std::vector<int> ids;
  std::size_t first = 0, last = 0;
  {
    std::lock_guard lock(m.mutex);
    Region& r = m.regions.at(std::size_t(index_));
    first = r.cursor;
    for (const Spec& sp : specs) {
      if (sp.n <= 0 || sp.k <= 0)
        throw Error("BfpGemm: empty matrix");
      Matrix mat = Shape(sp.n, sp.k);
      if (r.cursor + Words(mat) > r.end ||
          r.matrices.size() >= std::size_t(kScratchIds))
        throw Error("BfpGemm: the scratch region is full (" +
                    std::to_string((r.cursor - r.begin) * 4 >> 20) + " of " +
                    std::to_string((r.end - r.begin) * 4 >> 20) + " MiB)");
      mat.b_words = r.cursor;
      r.cursor += Words(mat);
      r.matrices.push_back(mat);
      ids.push_back(-(1 + index_ * kScratchIds + int(r.matrices.size()) - 1));
      mats.push_back(mat);
    }
    last = r.cursor;
  }
  // Transposed views are gathered into rows first.
  std::vector<std::vector<float>> rows(specs.size());
  std::vector<std::size_t> transposed;
  for (std::size_t i = 0; i < specs.size(); ++i)
    if (specs[i].col_stride != 1 && specs[i].same_as < 0)
      transposed.push_back(i);
  host::ParallelFor(transposed.size(), 1, [&](std::size_t lo, std::size_t hi) {
    for (std::size_t t = lo; t < hi; ++t) {
      const Spec& sp = specs[transposed[t]];
      auto& out = rows[transposed[t]];
      out.resize(std::size_t(sp.n) * std::size_t(sp.k));
      for (std::size_t i = 0; i < std::size_t(sp.k); ++i)
        for (std::size_t o = 0; o < std::size_t(sp.n); ++o)
          out[o * std::size_t(sp.k) + i] =
              sp.weights[o * sp.row_stride + i * sp.col_stride];
    }
  });
  std::vector<std::size_t> start(specs.size() + 1, 0);
  for (std::size_t i = 0; i < specs.size(); ++i)
    start[i + 1] = start[i] + (specs[i].same_as < 0 ? Impl::Tiles(mats[i]) : 0);
  host::ParallelFor(start.back(), 8, [&](std::size_t lo, std::size_t hi) {
    std::vector<std::uint8_t> tile(kTileWords * 4);
    std::size_t i = std::size_t(
        std::upper_bound(start.begin(), start.end(), lo) - start.begin() - 1);
    for (std::size_t t = lo; t < hi; ++t) {
      while (t >= start[i + 1])
        ++i;
      const Spec& sp = specs[i];
      const bool tr = sp.col_stride != 1;
      m.EncodeTile(mats[i], tr ? rows[i].data() : sp.weights,
                   tr ? std::size_t(sp.k) : sp.row_stride, t - start[i],
                   tile.data());
    }
  });
  auto* base = reinterpret_cast<std::uint8_t*>(m.arena.data());
  for (std::size_t i = 0; i < specs.size(); ++i) {
    const int src = specs[i].same_as;
    if (src < 0)
      continue;
    if (std::size_t(src) >= i ||
        Words(mats[std::size_t(src)]) != Words(mats[i]))
      throw Error("BfpGemm: same_as must name an earlier matrix of one shape");
    std::memcpy(base + mats[i].b_words * 4,
                base + mats[std::size_t(src)].b_words * 4, Words(mats[i]) * 4);
  }
  if (last > first)
    m.arena.ToDevice(first * 4, (last - first) * 4);
  return ids;
}

int BfpGemm::Scratch::Duplicate(int id) {
  auto& m = *owner_->impl_;
  Matrix mat;
  int copy;
  {
    std::lock_guard lock(m.mutex);
    const Matrix src = m.Resolve(id);
    Region& r = m.regions.at(std::size_t(index_));
    if (r.cursor + Words(src) > r.end ||
        r.matrices.size() >= std::size_t(kScratchIds))
      throw Error("BfpGemm: the scratch region is full");
    mat = src;
    mat.b_words = r.cursor;
    r.cursor += Words(mat);
    r.matrices.push_back(mat);
    copy = -(1 + index_ * kScratchIds + int(r.matrices.size()) - 1);
    auto* base = reinterpret_cast<std::uint8_t*>(m.arena.data());
    std::memcpy(base + mat.b_words * 4, base + src.b_words * 4, Words(mat) * 4);
  }
  m.arena.ToDevice(mat.b_words * 4, Words(mat) * 4);
  return copy;
}

void BfpGemm::Scratch::Reset() {
  auto& m = *owner_->impl_;
  std::lock_guard lock(m.mutex);
  Region& r = m.regions.at(std::size_t(index_));
  r.cursor = r.begin;
  r.matrices.clear();
}

void BfpGemm::Run(std::span<const Call> calls) {
  impl_->Run(calls);
}

double BfpGemm::TakeBusyMs() {
  std::lock_guard lock(impl_->busy_mutex);
  const double ms = impl_->busy_us / 1e3;
  impl_->busy_us = 0;
  return ms;
}

}  // namespace gufo::npu
