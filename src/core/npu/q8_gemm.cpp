#include "src/core/npu/q8_gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// q8gemm.py's layouts. A matrix splits into groups of at most 32768 codes
// (K). A group's K splits into kp slices of at most 2048 codes, one per set
// of 16 / kp consecutive shim streams, and its outputs into `count` equal
// parts of nt tiles of 8 outputs per core: one job computes the whole group
// for up to 8 G rows (G chunks <= 4 for the cores' A buffer, G tiles <= 32 for
// their accumulators), the cores of each stream set the partial sums of their
// K slice, which the host adds. The arena holds [stream][part][chunk of 512
// codes][tile][parity] records; the A staging one copy of the rows per stream
// (one shared copy when kp = 1). The jobs of one Run concatenate their
// instruction streams into one dispatch.
namespace gufo::npu {
namespace {

constexpr int kStreams = 16, kCores = 32, kRec = 4352, kCodes = 4096;
constexpr int kMaxA = 8, kMaxTiles = 32, kMaxParts = 64, kMaxChunks = 4;
// nt <= 16 leaves G = 2 (16 rows) for every shape.
constexpr int kMaxPartTiles = kMaxTiles / 2;
constexpr std::size_t kRecWords = kRec / 4, kOutWords = 128;
constexpr std::size_t kABytes = std::size_t(16) << 20, kCBytes = std::size_t(16)
                                                                 << 20;
constexpr unsigned kArgA = 3, kArgW = 4, kArgC = 5;

struct Part {
  int k0, k;               // the group's codes
  int kp, kc, nt, count;   // K slices, chunks per slice, tiles, output parts
  std::size_t b_words;     // offset in the arena
  std::vector<float> top;  // [slice][output]: its largest block scale / 32767
  [[nodiscard]] int Sets() const { return kStreams / kp; }  // streams per slice
  [[nodiscard]] int Span() const { return 2 * Sets() * nt * 8; }  // per part
  [[nodiscard]] int Records() const { return 2 * kc * nt; }  // per stream/part
  [[nodiscard]] std::size_t Bytes() const {
    return std::size_t(kStreams) * count * Records() * kRec;
  }
};

struct Matrix {
  int n, k;
  std::vector<Part> parts;
};

struct Job {
  const Matrix* m;
  const Part* p;
  std::size_t call;
  int r0, nr, g;
  int p0, parts;  // output parts [p0, p0 + parts) of the group
  std::size_t a_words, c_words;
  std::vector<float> scales;  // [slice][row]: the rows' int16 scales
  [[nodiscard]] int Copies() const { return p->kp == 1 ? 1 : kStreams; }
  [[nodiscard]] std::size_t AWords() const {
    return std::size_t(Copies()) * 2 * g * p->kc * kRecWords;
  }
  // One row: each output record holds row 0 of 8 tiles.
  [[nodiscard]] bool Compact() const { return nr == 1; }
  [[nodiscard]] int Gt() const {  // output records per core and part
    return Compact() ? (p->nt + 7) / 8 : g * p->nt;
  }
  [[nodiscard]] std::size_t CWords() const {
    return std::size_t(8) * Gt() * 4 * kOutWords * parts;
  }
};

}  // namespace

struct Q8Gemm::Impl {
  Program program;
  DispatchLibrary generate;
  Buffer a, c, arena;
  std::size_t arena_used{};
  std::vector<Matrix> matrices;
  std::map<std::vector<int>, Instructions> streams;  // (id, rows)... -> stream
  std::mutex mutex;
  std::vector<std::int16_t> q;
  std::vector<std::int32_t> out;

  int Upload(int n, int k, std::span<const std::int8_t> codes,
             std::span<const float> scales) {
    if (n <= 0 || k <= 0 || k % 32)
      throw Error("Q8Gemm: k must be a positive multiple of 32");
    if (codes.size() < std::size_t(n) * k ||
        scales.size() < std::size_t(n) * (k / 32))
      throw Error("Q8Gemm: codes or scales too short");
    Matrix m{n, k, {}};
    const int group_codes = kStreams * kMaxChunks * 512;
    for (int k0 = 0; k0 < k; k0 += group_codes) {
      Part p;
      p.k0 = k0, p.k = std::min(group_codes, k - k0);
      const int chunks = (p.k + 511) / 512;
      p.kp = 1;
      while ((chunks + p.kp - 1) / p.kp > kMaxChunks)
        p.kp *= 2;
      p.kc = (chunks + p.kp - 1) / p.kp;
      // Equal parts of at most kMaxPartTiles tiles per core, least padding.
      const int per_tile = 2 * p.Sets() * 8;
      const int tiles = (n + per_tile - 1) / per_tile;
      p.count = (tiles + kMaxPartTiles - 1) / kMaxPartTiles;
      p.nt = (tiles + p.count - 1) / p.count;
      for (int cnt = p.count + 1; cnt <= std::min(kMaxParts, tiles); ++cnt) {
        const int t = (tiles + cnt - 1) / cnt;
        if (cnt * t < p.count * p.nt)
          p.count = cnt, p.nt = t;
      }
      if (p.count > kMaxParts)
        throw Error("Q8Gemm: too many outputs");
      const std::size_t bytes = p.Bytes();
      if (arena_used + bytes > arena.size())
        throw Error("Q8Gemm: weight arena full");
      p.b_words = arena_used / 4;
      auto* w = reinterpret_cast<std::uint8_t*>(arena.data()) + arena_used;
      std::memset(w, 0, bytes);
      p.top.assign(std::size_t(p.kp) * n, 0.0f);
      const int span = p.Span();
      for (int slice = 0; slice < p.kp; ++slice) {
        const int s0 = p.k0 + slice * p.kc * 512;
        const int blocks =
            std::max(0, std::min(p.kc * 512, k0 + p.k - s0)) / 32;
        for (int j = 0; j < n; ++j) {
          const float* s = scales.data() + std::size_t(j) * (k / 32) + s0 / 32;
          float top = 0;
          for (int b = 0; b < blocks; ++b)
            top = std::max(top, s[b]);
          if (!(top > 0))
            top = 1;
          p.top[std::size_t(slice) * n + j] = top / 32767.0f;
          // output j: part j / span, then the slice's core jl / (8 nt), tile
          // (jl / 8) % nt, column jl % 8
          const int part = j / span, jl = j % span;
          const int core = jl / (8 * p.nt), t = (jl / 8) % p.nt, nn = jl % 8;
          const int stream = slice * p.Sets() + core / 2, parity = core % 2;
          const std::int8_t* row = codes.data() + std::size_t(j) * k + s0;
          std::uint8_t* pw =
              w + (std::size_t(stream) * p.count + part) * p.Records() * kRec;
          for (int kc = 0; kc < p.kc; ++kc) {
            std::uint8_t* rec =
                pw + ((std::size_t(kc) * p.nt + t) * 2 + parity) * kRec;
            auto* u = reinterpret_cast<std::int16_t*>(rec + kCodes);
            for (int blk = 0; blk < 16; ++blk) {
              const int b = kc * 16 + blk;
              if (b >= blocks)
                break;
              u[blk * 8 + nn] =
                  std::int16_t(std::lround(s[b] / top * 32767.0f));
              for (int kk = 0; kk < 32; ++kk)
                rec[blk * 256 + kk * 8 + nn] = std::uint8_t(row[b * 32 + kk]);
            }
          }
        }
      }
      arena.ToDevice(arena_used, bytes);
      arena_used += (bytes + 4095) / 4096 * 4096;
      m.parts.push_back(std::move(p));
    }
    matrices.push_back(std::move(m));
    return int(matrices.size()) - 1;
  }

  // Jobs of the calls in dispatches: each dispatch's A rows and outputs fit
  // the staging buffers (a large batch takes several dispatches).
  std::vector<std::vector<Job>> Plan(std::span<const Call> calls) {
    std::vector<std::vector<Job>> groups(1);
    std::size_t a_words = 0, c_words = 0;
    for (std::size_t i = 0; i < calls.size(); ++i) {
      const Matrix& m = matrices.at(std::size_t(calls[i].id));
      for (const Part& p : m.parts) {
        const int G =
            std::max(1, std::min(kMaxA / (2 * p.kc), kMaxTiles / p.nt));
        for (int r0 = 0; r0 < calls[i].rows; r0 += 8 * G) {
          const int nr = std::min(8 * G, calls[i].rows - r0), g = (nr + 7) / 8;
          Job one{&m, &p, i, r0, nr, g, 0, 1, 0, 0, {}};
          const std::size_t aw = one.AWords(), cw = one.CWords();
          if (aw * 4 > a.size() || cw * 4 > c.size())
            throw Error("Q8Gemm: one job exceeds the staging buffers");
          // As many parts per job as the output staging holds.
          const int per =
              int(std::min<std::size_t>(p.count, c.size() / 4 / cw));
          for (int p0 = 0; p0 < p.count; p0 += per) {
            Job j = one;
            j.p0 = p0, j.parts = std::min(per, p.count - p0);
            if ((a_words + aw) * 4 > a.size() ||
                (c_words + j.CWords()) * 4 > c.size()) {
              groups.emplace_back();
              a_words = c_words = 0;
            }
            j.a_words = a_words, j.c_words = c_words;
            a_words += aw;
            c_words += j.CWords();
            groups.back().push_back(std::move(j));
          }
        }
      }
    }
    return groups;
  }

  const Instructions& Stream(const std::vector<Job>& jobs,
                             const std::vector<int>& key) {
    if (auto it = streams.find(key); it != streams.end())
      return it->second;
    // Streams concatenate: one header [info, info, ops, bytes], then ops.
    std::vector<std::uint32_t> txn(4, 0);
    std::uint32_t ops = 0;
    for (const Job& j : jobs) {
      const Part& p = *j.p;
      const int ga = 2 * j.g * p.kc;
      const std::int32_t s[16] = {
          j.g,
          p.kc,
          p.nt,
          ga,
          j.Gt(),
          2 * p.nt,
          p.Records(),
          std::int32_t(j.a_words),
          std::int32_t(p.b_words + std::size_t(j.p0) * p.Records() * kRecWords),
          std::int32_t(j.c_words),
          j.parts,
          j.parts * p.Records(),
          j.parts * j.Gt(),
          p.count * p.Records(),
          j.Copies() == 1 ? 0 : std::int32_t(ga * kRecWords),
          j.Compact() ? 1 : 0};
      std::span<const std::uint32_t> words;
      try {
        words = generate.Generate(s);
      } catch (const Error& e) {
        std::string shape;
        for (const std::int32_t v : s)
          shape += ' ' + std::to_string(v);
        throw Error(std::string(e.what()) + " (n " + std::to_string(j.m->n) +
                    ", k " + std::to_string(j.m->k) + ", rows " +
                    std::to_string(j.nr) + "; scalars" + shape + ")");
      }
      if (words.size() <= 4)
        throw Error("Q8Gemm: instruction generation failed");
      if (ops == 0)
        txn[0] = words[0], txn[1] = words[1];
      ops += words[2];
      txn.insert(txn.end(), words.begin() + 4, words.end());
    }
    txn[2] = ops;
    txn[3] = std::uint32_t(txn.size() * 4);
    Instructions ins = program.AllocateInstructions(txn.size());
    ins.Assign(txn);
    return streams.emplace(key, std::move(ins)).first->second;
  }

  // A records per stream copy [group][chunk][half]: [blk 8][k4 8][mm 8][k 4]
  // of the stream's K slice.
  void Pack(Job& j, const float* x) {
    const Part& p = *j.p;
    const int width = p.kc * 512, ga = 2 * j.g * p.kc;
    j.scales.assign(std::size_t(p.kp) * j.g * 8, 0.0f);
    auto* base = reinterpret_cast<std::uint8_t*>(a.data()) + j.a_words * 4;
    for (int slice = 0; slice < p.kp; ++slice) {
      const int s0 = p.k0 + slice * width;
      const int kn = std::max(0, std::min(width, p.k0 + p.k - s0));
      q.assign(std::size_t(j.g) * 8 * width, 0);
      float* sc = j.scales.data() + std::size_t(slice) * j.g * 8;
      for (int i = 0; i < j.nr; ++i) {
        const float* xi = x + std::size_t(j.r0 + i) * j.m->k + s0;
        float top = 0;
        for (int c0 = 0; c0 < kn; ++c0)
          top = std::max(top, std::fabs(xi[c0]));
        const float inv = top > 0 ? 32767.0f / top : 0.0f;
        sc[i] = top > 0 ? top / 32767.0f : 1.0f;
        std::int16_t* qi = &q[std::size_t(i) * width];
        for (int c0 = 0; c0 < kn; ++c0) {  // round half away from zero
          const float t = xi[c0] * inv;
          qi[c0] = std::int16_t(t + (t >= 0 ? 0.5f : -0.5f));
        }
      }
      const int first = j.Copies() == 1 ? 0 : slice * p.Sets();
      auto* dst0 = base + std::size_t(first) * ga * kRec;
      for (int gi = 0; gi < j.g; ++gi)
        for (int kc = 0; kc < p.kc; ++kc)
          for (int h = 0; h < 2; ++h) {
            auto* dst = reinterpret_cast<std::int16_t*>(
                dst0 + ((std::size_t(gi) * p.kc + kc) * 2 + h) * kRec);
            for (int blk = 0; blk < 8; ++blk)
              for (int k4 = 0; k4 < 8; ++k4)
                for (int mm = 0; mm < 8; ++mm)
                  std::memcpy(dst + blk * 256 + k4 * 32 + mm * 4,
                              &q[(std::size_t(gi) * 8 + mm) * width + kc * 512 +
                                 h * 256 + blk * 32 + k4 * 4],
                              8);
          }
      if (j.Copies() > 1)
        for (int s = 1; s < p.Sets(); ++s)
          std::memcpy(dst0 + std::size_t(s) * ga * kRec, dst0,
                      std::size_t(ga) * kRec);
    }
  }

  // Output column col, part i at (col * parts + i) * gt * 512 words: records
  // [tile][group][core row jj][hi 64 | lo 64] (the acc64 as floor(acc / 2^24)
  // and acc mod 2^32). Each slice's partial sums add into y.
  void Unpack(const Job& j, float* y) const {
    const Part& p = *j.p;
    const std::size_t gt = std::size_t(j.Gt());
    const int span = p.Span(), n = j.m->n;
    for (int part = j.p0; part < j.p0 + j.parts; ++part)
      for (int col = 0; col < 8; ++col)
        for (int jj = 0; jj < 4; ++jj) {
          const std::int32_t* base =
              out.data() + j.c_words +
              (std::size_t(col) * j.parts + (part - j.p0)) * gt * 4 * kOutWords;
          const int core = 4 * col + jj, stream = core / 2;
          const int slice = stream / p.Sets(),
                    local = core - slice * 2 * p.Sets();
          const float* tops = p.top.data() + std::size_t(slice) * n;
          const float* rss = j.scales.data() + std::size_t(slice) * j.g * 8;
          for (int t = 0; t < p.nt; ++t) {
            const int out0 = part * span + (local * p.nt + t) * 8;
            if (out0 >= n)
              continue;
            const int outs = std::min(8, n - out0);
            if (j.Compact()) {  // row 0 of tile t: record t / 8, slot t % 8
              const std::int32_t* w =
                  base + (std::size_t(t / 8) * 4 + jj) * kOutWords +
                  (t % 8) * 8;
              const std::int32_t *hi = w, *lo = w + 64;
              float* dst = y + std::size_t(j.r0) * n + out0;
              const float rs = rss[0];
              const float* top = tops + out0;
              for (int nn = 0; nn < outs; ++nn)
                dst[nn] +=
                    (float(hi[nn]) * 16777216.0f + float(lo[nn] & 0xFFFFFF)) *
                    (rs * top[nn]);
              continue;
            }
            for (int gi = 0; gi < j.g; ++gi) {
              const std::int32_t* w =
                  base + (std::size_t(t) * j.g + gi) * 4 * kOutWords +
                  std::size_t(jj) * kOutWords;
              for (int mm = 0; mm < 8; ++mm) {
                const int row = gi * 8 + mm;
                if (row >= j.nr)
                  break;
                float* dst = y + std::size_t(j.r0 + row) * n + out0;
                const float rs = rss[row];
                const std::int32_t *hi = w + mm * 8, *lo = w + 64 + mm * 8;
                const float* top = tops + out0;
                for (int nn = 0; nn < outs; ++nn)
                  dst[nn] +=
                      (float(hi[nn]) * 16777216.0f + float(lo[nn] & 0xFFFFFF)) *
                      (rs * top[nn]);
              }
            }
          }
        }
  }

  void Run(std::span<const Call> calls) {
    std::lock_guard lock(mutex);
    std::vector<std::vector<Job>> groups = Plan(calls);
    for (const Call& call : calls)
      std::fill_n(call.y,
                  std::size_t(call.rows) * matrices[std::size_t(call.id)].n,
                  0.0f);
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
      auto& jobs = groups[gi];
      std::vector<int> key{int(gi)};
      for (const Call& call : calls)
        key.push_back(call.id), key.push_back(call.rows);
      const Instructions& ins = Stream(jobs, key);
      std::size_t a_end = 0, c_end = 0;
      for (Job& j : jobs) {
        Pack(j, calls[j.call].x);
        a_end = j.a_words + j.AWords();
        c_end = j.c_words + j.CWords();
      }
      a.ToDevice(0, a_end * 4);
      static const bool profile = std::getenv("GUFO_NPU_PROFILE") != nullptr;
      const auto t0 = std::chrono::steady_clock::now();
      program.Start(ins, {&a, &arena, &c}).Wait();
      if (profile)
        std::fprintf(stderr, "q8gemm: %zu jobs, npu %.1f us\n", jobs.size(),
                     std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
      c.FromDevice(0, c_end * 4);
      // The buffer is uncached: copy it out once, then unpack.
      out.resize(c_end);
      std::memcpy(out.data(), c.data(), c_end * 4);
      for (const Job& j : jobs)
        Unpack(j, calls[j.call].y);
    }
  }
};

Q8Gemm Q8Gemm::Open(Device& device, const std::filesystem::path& dir,
                    std::size_t arena_bytes) {
  if (arena_bytes > kMaxArenaBytes)
    throw Error("Q8Gemm: the weight arena holds at most 4 GiB");
  Q8Gemm g;
  g.impl_ = std::make_unique<Impl>();
  auto& m = *g.impl_;
  m.program = Program::Load(device, dir / "q8gemm.xclbin");
  m.generate = DispatchLibrary::Open(dir / "q8gemm_dispatch.so", 16);
  m.a = m.program.Allocate(kABytes, kArgA);
  m.c = m.program.Allocate(kCBytes, kArgC);
  m.arena = m.program.Allocate(arena_bytes, kArgW);
  return g;
}

Q8Gemm::Q8Gemm() = default;
Q8Gemm::~Q8Gemm() = default;
Q8Gemm::Q8Gemm(Q8Gemm&&) noexcept = default;
Q8Gemm& Q8Gemm::operator=(Q8Gemm&&) noexcept = default;

int Q8Gemm::Upload(int n, int k, std::span<const std::int8_t> codes,
                   std::span<const float> scales) {
  std::lock_guard lock(impl_->mutex);
  return impl_->Upload(n, k, codes, scales);
}

void Q8Gemm::Run(std::span<const Call> calls) {
  impl_->Run(calls);
}

}  // namespace gufo::npu
