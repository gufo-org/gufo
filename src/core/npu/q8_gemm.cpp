#include "src/core/npu/q8_gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// q8gemm.py's layouts. A matrix splits into parts of at most 2048 codes (K)
// by 4096 outputs (N), each laid out in the arena as its 16 shim streams read
// them: [stream][chunk of 512 codes][tile of 8 outputs][parity]. A product
// over any rows becomes jobs of up to 8 G rows per part (G chunks <= 4 for
// the cores' A buffer, G tiles <= 32 for their accumulators); the jobs of one
// Run concatenate their instruction streams into one dispatch.
namespace gufo::npu {
namespace {

constexpr int kStreams = 16, kCores = 32, kRec = 4352, kCodes = 4096;
constexpr int kMaxA = 8, kMaxTiles = 32;
constexpr int kMaxK = 2048, kMaxN = kCores * 8 * kMaxTiles / 2;
constexpr std::size_t kRecWords = kRec / 4, kOutWords = 128;
constexpr std::size_t kABytes = std::size_t(16) << 20, kCBytes = std::size_t(16)
                                                                 << 20;
constexpr unsigned kArgA = 3, kArgW = 4, kArgC = 5;

struct Part {
  int k0, kp, n0, np, kc, nt;
  std::size_t b_words;     // offset in the arena
  std::vector<float> top;  // per output: its largest block scale / 32767
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
  std::size_t a_words, c_words;
  std::vector<float> scales;  // per row: its int16 scale
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
    for (int k0 = 0; k0 < k; k0 += kMaxK)
      for (int n0 = 0; n0 < n; n0 += kMaxN) {
        Part p;
        p.k0 = k0, p.kp = std::min(kMaxK, k - k0), p.n0 = n0,
        p.np = std::min(kMaxN, n - n0);
        p.kc = (p.kp + 511) / 512,
        p.nt = (p.np + kCores * 8 - 1) / (kCores * 8);
        const std::size_t bytes =
            std::size_t(kStreams) * p.kc * p.nt * 2 * kRec;
        if (arena_used + bytes > arena.size())
          throw Error("Q8Gemm: weight arena full");
        p.b_words = arena_used / 4;
        auto* w = reinterpret_cast<std::uint8_t*>(arena.data()) + arena_used;
        std::memset(w, 0, bytes);
        const int blocks = p.kp / 32;
        p.top.assign(std::size_t(p.np), 0.0f);
        for (int j = 0; j < p.np; ++j) {
          const float* s =
              scales.data() + std::size_t(n0 + j) * (k / 32) + k0 / 32;
          float top = 0;
          for (int b = 0; b < blocks; ++b)
            top = std::max(top, s[b]);
          if (!(top > 0))
            top = 1;
          p.top[std::size_t(j)] = top / 32767.0f;
          // output j: core j / (8 nt), tile (j / 8) % nt, column j % 8
          const int core = j / (8 * p.nt), t = (j / 8) % p.nt, nn = j % 8;
          const int stream = core / 2, parity = core % 2;
          const std::int8_t* row = codes.data() + std::size_t(n0 + j) * k + k0;
          for (int kc = 0; kc < p.kc; ++kc) {
            std::uint8_t* rec =
                w + ((std::size_t(stream) * p.kc + kc) * p.nt * 2 +
                     std::size_t(t) * 2 + parity) *
                        kRec;
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
          const std::size_t aw = std::size_t(g) * p.kc * 2 * kRecWords,
                            cw = std::size_t(8) * g * p.nt * 4 * kOutWords;
          if (aw * 4 > a.size() || cw * 4 > c.size())
            throw Error("Q8Gemm: one job exceeds the staging buffers");
          if ((a_words + aw) * 4 > a.size() || (c_words + cw) * 4 > c.size()) {
            groups.emplace_back();
            a_words = c_words = 0;
          }
          groups.back().push_back(
              Job{&m, &p, i, r0, nr, g, a_words, c_words, {}});
          a_words += aw;
          c_words += cw;
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
      const std::int32_t s[10] = {j.g,
                                  p.kc,
                                  p.nt,
                                  2 * j.g * p.kc,
                                  j.g * p.nt,
                                  2 * p.nt,
                                  2 * p.kc * p.nt,
                                  std::int32_t(j.a_words),
                                  std::int32_t(p.b_words),
                                  std::int32_t(j.c_words)};
      const auto words = generate.Generate(s);
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

  void Pack(Job& j, const float* x) {
    const Part& p = *j.p;
    const int width = p.kc * 512;
    q.assign(std::size_t(j.g) * 8 * width, 0);
    j.scales.assign(std::size_t(j.g) * 8, 0.0f);
    for (int i = 0; i < j.nr; ++i) {
      const float* xi = x + std::size_t(j.r0 + i) * j.m->k + p.k0;
      float top = 0;
      for (int c0 = 0; c0 < p.kp; ++c0)
        top = std::max(top, std::fabs(xi[c0]));
      const float inv = top > 0 ? 32767.0f / top : 0.0f;
      j.scales[std::size_t(i)] = top > 0 ? top / 32767.0f : 1.0f;
      std::int16_t* qi = &q[std::size_t(i) * width];
      for (int c0 = 0; c0 < p.kp; ++c0) {  // round half away from zero
        const float t = xi[c0] * inv;
        qi[c0] = std::int16_t(t + (t >= 0 ? 0.5f : -0.5f));
      }
    }
    // A records [group][chunk][half]: [blk 8][k4 8][mm 8][k 4]
    auto* base = reinterpret_cast<std::uint8_t*>(a.data()) + j.a_words * 4;
    for (int gi = 0; gi < j.g; ++gi)
      for (int kc = 0; kc < p.kc; ++kc)
        for (int h = 0; h < 2; ++h) {
          auto* dst = reinterpret_cast<std::int16_t*>(
              base + ((std::size_t(gi) * p.kc + kc) * 2 + h) * kRec);
          for (int blk = 0; blk < 8; ++blk)
            for (int k4 = 0; k4 < 8; ++k4)
              for (int mm = 0; mm < 8; ++mm)
                std::memcpy(dst + blk * 256 + k4 * 32 + mm * 4,
                            &q[(std::size_t(gi) * 8 + mm) * width + kc * 512 +
                               h * 256 + blk * 32 + k4 * 4],
                            8);
        }
  }

  // Output column col at col * gt * 512 words: records [tile][group][core row
  // jj][hi 64 | lo 64] (the acc64 as floor(acc / 2^24) and acc mod 2^32).
  void Unpack(const Job& j, float* y) const {
    const Part& p = *j.p;
    const std::size_t gt = std::size_t(j.g) * p.nt;
    const std::int32_t* base = out.data() + j.c_words;
    for (int col = 0; col < 8; ++col)
      for (int jj = 0; jj < 4; ++jj) {
        const int core = 4 * col + jj;
        for (int t = 0; t < p.nt; ++t) {
          const int out0 = (core * p.nt + t) * 8;
          if (out0 >= p.np)
            continue;
          const int outs = std::min(8, p.np - out0);
          for (int gi = 0; gi < j.g; ++gi) {
            const std::int32_t* w =
                base +
                (std::size_t(col) * gt + (std::size_t(t) * j.g + gi)) * 4 *
                    kOutWords +
                std::size_t(jj) * kOutWords;
            for (int mm = 0; mm < 8; ++mm) {
              const int row = gi * 8 + mm;
              if (row >= j.nr)
                break;
              float* dst = y + std::size_t(j.r0 + row) * j.m->n + p.n0 + out0;
              const float rs = j.scales[std::size_t(row)];
              const std::int32_t *hi = w + mm * 8, *lo = w + 64 + mm * 8;
              const float* top = p.top.data() + out0;
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
        a_end = j.a_words + std::size_t(j.g) * j.p->kc * 2 * kRecWords;
        c_end = j.c_words + std::size_t(8) * j.g * j.p->nt * 4 * kOutWords;
      }
      a.ToDevice(0, a_end * 4);
      program.Start(ins, {&a, &arena, &c}).Wait();
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
  Q8Gemm g;
  g.impl_ = std::make_unique<Impl>();
  auto& m = *g.impl_;
  m.program = Program::Load(device, dir / "q8gemm.xclbin");
  m.generate = DispatchLibrary::Open(dir / "q8gemm_dispatch.so", 10);
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
