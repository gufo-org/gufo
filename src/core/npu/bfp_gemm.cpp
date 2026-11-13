#include "src/core/npu/bfp_gemm.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "src/core/npu/host_ops.hpp"

// bfpgemm.py's layouts. A matrix pads to Kp (a multiple of 512 inputs) by Np
// (a multiple of 1024 outputs, G groups) and lies in the arena as the eight
// core columns read it: [column c][group][k block of 64][tile of 2304 words],
// tile = the 1x2 8x8-block shuffle of the 64 x 128 slice (Lc = 8 group + c).
// A product becomes jobs of one 512-row band by up to `ng` groups; a band's
// rows are staged once per Run as [core row][K / 512][128][512] bf16, and a
// job's output comes back as [column][group][512][128] bf16.
namespace gufo::npu {
namespace {

constexpr int kBand = 512, kGroup = 1024, kKc = 512, kKb = 64;
constexpr std::size_t kTileWords = 2304;
constexpr std::size_t kABytes = std::size_t(64) << 20, kCBytes = std::size_t(64)
                                                                 << 20;
constexpr unsigned kArgA = 3, kArgW = 4, kArgC = 5;
constexpr std::size_t kMaxStreams = 512;

std::uint16_t ToBf16(float f) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(f);
  if ((u & 0x7f800000u) == 0x7f800000u)
    return std::uint16_t(u >> 16);
  u += 0x7fffu + ((u >> 16) & 1u);
  return std::uint16_t(u >> 16);
}

float FromBf16(std::uint16_t h) {
  return std::bit_cast<float>(std::uint32_t(h) << 16);
}

struct Matrix {
  int n, k, kp, groups;
  std::size_t b_words;
};

struct Band {
  std::size_t call;
  int r0, nr;
  std::size_t a_words;
};

struct Job {
  const Matrix* m;
  std::size_t band;  // index in its dispatch's bands
  int g0, ng;
  std::size_t c_words;
};

struct Dispatch {
  std::vector<Band> bands;
  std::vector<Job> jobs;
  std::size_t a_end{}, c_end{};
};

}  // namespace

void BfpGemm::EncodeBlock(const float* values, std::uint8_t* out) {
  int e = 0;
  for (int i = 0; i < 8; ++i)
    e = std::max(e,
                 int((std::bit_cast<std::uint32_t>(values[i]) >> 23) & 0xff));
  const float inv = std::ldexp(1.0f, 127 + 6 - e);  // 64 / 2^(e - 127)
  out[0] = std::uint8_t(e);
  for (int i = 0; i < 8; ++i) {
    const float t = std::nearbyint(values[i] * inv);
    out[1 + i] = std::uint8_t(std::int8_t(std::clamp(t, -128.0f, 127.0f)));
  }
}

struct BfpGemm::Impl {
  Program program;
  DispatchLibrary generate;
  Buffer a, c, arena;
  std::size_t arena_used{};
  std::vector<Matrix> matrices;
  std::map<std::vector<std::int32_t>, Instructions> streams;
  std::mutex mutex;
  std::vector<std::uint16_t> out;

  template<typename Get>
  int Upload(int n, int k, Get get) {
    if (n <= 0 || k <= 0)
      throw Error("BfpGemm: empty matrix");
    Matrix m{n, k, (k + kKc - 1) / kKc * kKc, (n + kGroup - 1) / kGroup,
             arena_used};
    const int kb = m.kp / kKb;
    const std::size_t words = std::size_t(m.groups) * 8 * kb * kTileWords;
    if ((arena_used + words) * 4 > arena.size())
      throw Error("BfpGemm: the weight arena is full");
    auto* base = reinterpret_cast<std::uint8_t*>(arena.data()) + m.b_words * 4;
    host::ParallelFor(
        std::size_t(8) * m.groups, 1, [&](std::size_t lo, std::size_t hi) {
          std::vector<std::uint8_t> tile(kTileWords * 4);
          float v[8];
          for (std::size_t t = lo; t < hi; ++t) {
            const int col = int(t) / m.groups, g = int(t) % m.groups;
            const int lc = 8 * g + col;
            for (int lr = 0; lr < kb; ++lr) {
              std::uint8_t* dst = tile.data();
              // [superblock 8][block row 8][b 2][c 8]: 8 consecutive k at one
              // n.
              for (int sb = 0; sb < 8; ++sb)
                for (int br = 0; br < 8; ++br)
                  for (int b = 0; b < 2; ++b)
                    for (int cc = 0; cc < 8; ++cc, dst += 9) {
                      const int nn = lc * 128 + sb * 16 + b * 8 + cc;
                      const int k0 = lr * kKb + br * 8;
                      for (int r = 0; r < 8; ++r)
                        v[r] = (nn < n && k0 + r < k) ? get(nn, k0 + r) : 0.0f;
                      EncodeBlock(v, dst);
                    }
              std::memcpy(base + ((std::size_t(col) * m.groups + g) * kb + lr) *
                                     kTileWords * 4,
                          tile.data(), tile.size());
            }
          }
        });
    arena.ToDevice(m.b_words * 4, words * 4);
    arena_used += words;
    matrices.push_back(m);
    return int(matrices.size()) - 1;
  }

  std::vector<Dispatch> Plan(std::span<const Call> calls) {
    std::vector<Dispatch> plan(1);
    for (std::size_t i = 0; i < calls.size(); ++i) {
      const Matrix& m = matrices.at(std::size_t(calls[i].id));
      const int kc = m.kp / kKc, kb = m.kp / kKb;
      const std::size_t a_words = std::size_t(kc) * 4 * 128 * kKc / 2;
      const int ng_max = std::max(1, std::min(64, 1023 / kb));
      for (int r0 = 0; r0 < calls[i].rows; r0 += kBand) {
        const int nr = std::min(kBand, calls[i].rows - r0);
        for (int g0 = 0; g0 < m.groups; g0 += ng_max) {
          const int ng = std::min(ng_max, m.groups - g0);
          const std::size_t c_words = std::size_t(ng) * 8 * kBand * 128 / 2;
          Dispatch* d = &plan.back();
          const bool new_band = d->bands.empty() || d->bands.back().call != i ||
                                d->bands.back().r0 != r0;
          if ((new_band ? d->a_end + a_words : d->a_end) * 4 > a.size() ||
              (d->c_end + c_words) * 4 > c.size()) {
            if (d->jobs.empty())
              throw Error("BfpGemm: one job exceeds the staging buffers");
            plan.emplace_back();
            d = &plan.back();
          }
          if (d->bands.empty() || d->bands.back().call != i ||
              d->bands.back().r0 != r0) {
            d->bands.push_back(Band{i, r0, nr, d->a_end});
            d->a_end += a_words;
          }
          d->jobs.push_back(Job{&m, d->bands.size() - 1, g0, ng, d->c_end});
          d->c_end += c_words;
        }
      }
    }
    return plan;
  }

  const Instructions& Stream(const std::vector<std::int32_t>& scalars) {
    if (auto it = streams.find(scalars); it != streams.end())
      return it->second;
    if (streams.size() >= kMaxStreams)
      streams.clear();
    std::vector<std::uint32_t> txn(4, 0);
    std::uint32_t ops = 0;
    for (std::size_t i = 0; i < scalars.size(); i += 10) {
      const auto words = generate.Generate(
          std::span<const std::int32_t>(scalars.data() + i, 10));
      if (words.size() <= 4)
        throw Error("BfpGemm: instruction generation failed");
      if (ops == 0)
        txn[0] = words[0], txn[1] = words[1];
      ops += words[2];
      txn.insert(txn.end(), words.begin() + 4, words.end());
    }
    txn[2] = ops;
    txn[3] = std::uint32_t(txn.size() * 4);
    Instructions ins = program.AllocateInstructions(txn.size());
    ins.Assign(txn);
    return streams.emplace(scalars, std::move(ins)).first->second;
  }

  void PackBand(const Band& band, const Matrix& m, const float* x) {
    const int kc = m.kp / kKc;
    auto* base = reinterpret_cast<std::uint16_t*>(a.data()) + band.a_words * 2;
    host::ParallelFor(kBand, 16, [&](std::size_t b, std::size_t e) {
      std::vector<std::uint16_t> row(std::size_t(m.kp), 0);
      for (std::size_t i = b; i < e; ++i) {
        const std::size_t r = i / 128, ii = i % 128;
        if (int(i) < band.nr) {
          const float* xi = x + std::size_t(band.r0 + int(i)) * m.k;
          for (int kk = 0; kk < m.k; ++kk)
            row[std::size_t(kk)] = ToBf16(xi[kk]);
        } else {
          std::fill(row.begin(), row.end(), std::uint16_t(0));
        }
        for (int q = 0; q < kc; ++q)
          std::memcpy(base + ((r * kc + q) * 128 + ii) * kKc,
                      row.data() + std::size_t(q) * kKc, kKc * 2);
      }
    });
  }

  // Copies one job's output tiles into its call's rows.
  void Unpack(const Job& j, const Band& b, const Call& call,
              const std::uint16_t* out_words) {
    const int n = j.m->n;
    host::ParallelFor(
        std::size_t(8) * j.ng, 1, [&](std::size_t lo, std::size_t hi) {
          for (std::size_t t = lo; t < hi; ++t) {
            const int col = int(t) / j.ng, g = int(t) % j.ng;
            const int n0 = ((j.g0 + g) * 8 + col) * 128;
            if (n0 >= n)
              continue;
            const int w = std::min(128, n - n0);
            const std::uint16_t* src =
                out_words + j.c_words * 2 +
                (std::size_t(col) * j.ng + g) * kBand * 128;
            for (int i = 0; i < b.nr; ++i) {
              float* y = call.y + std::size_t(b.r0 + i) * n + n0;
              const std::uint16_t* sr = src + std::size_t(i) * 128;
              for (int q = 0; q < w; ++q)
                y[q] = FromBf16(sr[q]);
            }
          }
        });
  }

  void Run(std::span<const Call> calls) {
    std::lock_guard lock(mutex);
    for (const Dispatch& d : Plan(calls)) {
      std::vector<std::int32_t> scalars;
      for (const Job& j : d.jobs) {
        const Band& b = d.bands[j.band];
        const int kc = j.m->kp / kKc, kb = j.m->kp / kKb;
        const std::int32_t a_row = kc * 128 * kKc / 2;
        const std::int32_t b_col = j.m->groups * kb * std::int32_t(kTileWords);
        const std::int32_t c_col = j.ng * kBand * 128 / 2;
        const std::int32_t s[10] = {
            j.ng,
            kb,
            kc,
            j.ng * kb,
            std::int32_t(b.a_words),
            a_row,
            std::int32_t(j.m->b_words + std::size_t(j.g0) * kb * kTileWords),
            b_col,
            std::int32_t(j.c_words),
            c_col};
        scalars.insert(scalars.end(), s, s + 10);
      }
      for (const Band& b : d.bands) {
        const Call& call = calls[b.call];
        PackBand(b, matrices[std::size_t(call.id)], call.x);
      }
      const Instructions& ins = Stream(scalars);
      a.ToDevice(0, d.a_end * 4);
      program.Start(ins, {&a, &arena, &c}).Wait();
      c.FromDevice(0, d.c_end * 4);
      // The buffer is uncached: copy it out once, then unpack.
      out.resize(d.c_end * 2);
      const auto* src = reinterpret_cast<const std::uint16_t*>(c.data());
      host::ParallelFor(out.size(), std::size_t(1) << 18,
                        [&](std::size_t lo, std::size_t hi) {
                          std::memcpy(out.data() + lo, src + lo, (hi - lo) * 2);
                        });
      for (const Job& j : d.jobs)
        Unpack(j, d.bands[j.band], calls[d.bands[j.band].call], out.data());
    }
  }
};

BfpGemm BfpGemm::Open(Device& device, const std::filesystem::path& dir,
                      std::size_t arena_bytes) {
  BfpGemm g;
  g.impl_ = std::make_unique<Impl>();
  auto& m = *g.impl_;
  m.program = Program::Load(device, dir / "bfpgemm.xclbin");
  m.generate = DispatchLibrary::Open(dir / "bfpgemm_dispatch.so", 10);
  m.a = m.program.Allocate(kABytes, kArgA);
  m.c = m.program.Allocate(kCBytes, kArgC);
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
  std::lock_guard lock(impl_->mutex);
  return impl_->Upload(
      n, k, [&](int nn, int kk) { return weights[std::size_t(nn) * k + kk]; });
}

int BfpGemm::UploadBf16(int n, int k, std::span<const std::uint16_t> weights) {
  if (weights.size() < std::size_t(n) * std::size_t(k))
    throw Error("BfpGemm: too few weights");
  std::lock_guard lock(impl_->mutex);
  return impl_->Upload(n, k, [&](int nn, int kk) {
    return FromBf16(weights[std::size_t(nn) * k + kk]);
  });
}

void BfpGemm::Run(std::span<const Call> calls) {
  impl_->Run(calls);
}

}  // namespace gufo::npu
