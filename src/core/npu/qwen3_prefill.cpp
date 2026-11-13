#include "src/core/npu/qwen3_prefill.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "src/core/npu/host_ops.hpp"

// qwen3pre.py's layouts (kernels/qwen3pre/plan.py; offsets from
// qwen3pre_<M>_plan.txt). B holds each GEMM's [K][N] weight in phase order,
// column tile (128 outputs) by K tile (64 inputs), each tile the 1x2 8x8-block
// shuffle of bfp16ebs8 blocks (8 consecutive inputs of one output). A token
// record is [xn | x | o | pad] bf16 (REC elements); the input goes into x.
// The attention runs in two passes of 4 KV heads: pass p's query head j is
// query head 2 (j / 4 + 4 p) + j % 4 for j % 4 < 2 and a zero row otherwise;
// its gates are zero (sigmoid(0) = 1/2, so o_proj doubles), and o_proj reads
// pass 1's heads before pass 0's.
namespace gufo::npu {
namespace {

constexpr int kRecordH = 2048;
constexpr int kKTile = 64, kNTile = 128;
constexpr std::size_t kBlockBytes = 9, kTileBlocks = 1024;
constexpr unsigned kArgA = 3, kArgB = 4, kArgC = 5;
constexpr int kKOffset = 9216, kVOffset = 11264;  // within a K/V piece

float FromBf16(std::uint16_t h) {
  return std::bit_cast<float>(std::uint32_t(h) << 16);
}

struct Gemm {
  int layer, pass, K, N;
  std::string name;
  std::size_t b;  // offset in bfp16 blocks
};

struct Variant {
  int M{};
  std::size_t rec{}, qkv_row{}, piece{}, b_size{}, c_size{}, qs{};
  std::vector<Gemm> gemms;
  std::vector<std::tuple<std::size_t, int, std::string>> ablocks;
  std::map<std::pair<int, int>, std::size_t> pieces;
  Program program;
  Instructions ins;
  Buffer a, c;
};

Variant ReadPlan(const std::filesystem::path& file) {
  std::ifstream in(file);
  if (!in)
    throw Error("Qwen3Prefill: no plan " + file.string());
  Variant v;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string key;
    s >> key;
    if (key == "M")
      s >> v.M;
    else if (key == "rec")
      s >> v.rec;
    else if (key == "qkv_row")
      s >> v.qkv_row;
    else if (key == "piece_size")
      s >> v.piece;
    else if (key == "b_size")
      s >> v.b_size;
    else if (key == "c_size")
      s >> v.c_size;
    else if (key == "qs")
      s >> v.qs;
    else if (key == "gemm") {
      Gemm g;
      s >> g.layer >> g.name >> g.pass >> g.K >> g.N >> g.b;
      v.gemms.push_back(g);
    } else if (key == "ablock") {
      std::size_t off;
      int l;
      std::string name;
      s >> off >> l >> name;
      v.ablocks.emplace_back(off, l, name);
    } else if (key == "piece") {
      int l, p;
      std::size_t off;
      s >> l >> p >> off;
      v.pieces[{l, p}] = off;
    }
  }
  if (v.M <= 0 || v.gemms.empty())
    throw Error("Qwen3Prefill: bad plan " + file.string());
  return v;
}

// A 128 x 512 A block whose first L1 objects hold vec linearly, 2048 values
// each (object i: rows 32 i .. 32 i + 31 of columns 0..63, blocked
// [m/r][k/s][r][s]); plan.ablock_put.
void PutABlock(std::uint16_t* block, const std::vector<std::uint16_t>& vec) {
  std::fill_n(block, 128 * 512, std::uint16_t(0));
  for (std::size_t p = 0; p < vec.size(); ++p) {
    const std::size_t obj = p / 2048, q = p % 2048;
    const std::size_t row = obj * 32 + (q / 512) * 8 + (q / 8) % 8;
    const std::size_t col = ((q / 64) % 8) * 8 + q % 8;
    block[row * 512 + col] = vec[p];
  }
}

}  // namespace

struct Qwen3Prefill::Impl {
  std::vector<Variant> variants;  // by increasing M
  Buffer b;                       // weights, shared by every variant
  double theta{1e6};
  Variant* last{};  // the variant of the last Run

  // Packs one GEMM: col(n, out) writes column n's K inputs (zeros allowed).
  template<typename Column>
  void Pack(const Gemm& g, Column col) {
    auto* base = reinterpret_cast<std::uint8_t*>(b.data()) + g.b * kBlockBytes;
    const int kt = g.K / kKTile;
    host::ParallelFor(
        std::size_t(g.N / kNTile), 1, [&](std::size_t lo, std::size_t hi) {
          std::vector<float> cols(std::size_t(kNTile) * std::size_t(g.K));
          std::vector<float> blk(8 * kTileBlocks);
          for (std::size_t lc = lo; lc < hi; ++lc) {
            for (int c = 0; c < kNTile; ++c)
              col(int(lc) * kNTile + c, cols.data() + std::size_t(c) * g.K);
            for (int lr = 0; lr < kt; ++lr) {
              float* d = blk.data();
              // [superblock 8][block row 8][b 2][c 8] blocks of 8 consecutive k
              for (int sb = 0; sb < 8; ++sb)
                for (int br = 0; br < 8; ++br)
                  for (int bb = 0; bb < 2; ++bb)
                    for (int cc = 0; cc < 8; ++cc, d += 8)
                      std::memcpy(d,
                                  cols.data() +
                                      std::size_t(sb * 16 + bb * 8 + cc) * g.K +
                                      std::size_t(lr * kKTile + br * 8),
                                  8 * sizeof(float));
              host::EncodeBfp16Blocks(
                  blk.data(), kTileBlocks,
                  base + (lc * std::size_t(kt) + std::size_t(lr)) *
                             kTileBlocks * kBlockBytes,
                  kBlockBytes);
            }
          }
        });
    b.ToDevice(g.b * kBlockBytes,
               std::size_t(g.K) * std::size_t(g.N) / 8 * kBlockBytes);
  }

  void WriteRope(Variant& v) {
    auto* c = reinterpret_cast<std::uint16_t*>(v.c.data());
    for (int r = 0; r < v.M; ++r) {
      std::uint16_t* t = c + v.qs + std::size_t(r) * v.qkv_row + 4096;
      for (int i = 0; i < 64; ++i) {
        const double a = double(r) * std::pow(theta, -2.0 * i / kHeadDim);
        t[i] = host::FloatToBf16(float(std::cos(a)));
        t[64 + i] = host::FloatToBf16(float(std::sin(a)));
      }
    }
    v.c.ToDevice(v.qs * 2, std::size_t(v.M) * v.qkv_row * 2);
  }
};

Qwen3Prefill::Qwen3Prefill() = default;
Qwen3Prefill::~Qwen3Prefill() = default;
Qwen3Prefill::Qwen3Prefill(Qwen3Prefill&&) noexcept = default;
Qwen3Prefill& Qwen3Prefill::operator=(Qwen3Prefill&&) noexcept = default;

Qwen3Prefill Qwen3Prefill::Open(Device& device,
                                const std::filesystem::path& dir) {
  Qwen3Prefill p;
  p.impl_ = std::make_unique<Impl>();
  auto& m = *p.impl_;
  for (int M : {128, 256, 512, 1024, 2048}) {
    const auto stem = dir / ("qwen3pre_" + std::to_string(M));
    std::error_code ec;
    if (!std::filesystem::is_regular_file(stem.string() + ".xclbin", ec))
      continue;
    Variant v = ReadPlan(stem.string() + "_plan.txt");
    v.program = Program::Load(device, stem.string() + ".xclbin");
    v.ins = v.program.LoadInstructions(stem.string() + ".bin");
    v.a = v.program.Allocate(std::size_t(v.M) * v.rec * 2, kArgA);
    v.c = v.program.Allocate(v.c_size * 2, kArgC);
    if (!m.variants.empty() && (v.b_size != m.variants.front().b_size))
      throw Error("Qwen3Prefill: variants disagree on the weight layout");
    if (m.variants.empty())
      m.b = v.program.Allocate(v.b_size * kBlockBytes, kArgB);
    // A blocks: q/k norms come with the layers; the residual phases' weight
    // is 1 (plain residual), the first phase's zero.
    auto* c = reinterpret_cast<std::uint16_t*>(v.c.data());
    const std::vector<std::uint16_t> ones(kHidden, host::FloatToBf16(1.0f));
    for (const auto& [off, l, name] : v.ablocks)
      if (name == "post" || name == "ffn_post")
        PutABlock(c + off, ones);
      else
        PutABlock(c + off, {});
    v.c.ToDevice();
    m.WriteRope(v);
    m.variants.push_back(std::move(v));
  }
  if (m.variants.empty())
    throw Error("Qwen3Prefill: no qwen3pre program in " + dir.string());
  return p;
}

int Qwen3Prefill::max_rows() const noexcept {
  return impl_->variants.back().M;
}

void Qwen3Prefill::SetRopeTheta(double theta) {
  impl_->theta = theta;
  for (Variant& v : impl_->variants)
    impl_->WriteRope(v);
}

void Qwen3Prefill::SetLayer(int l, const Layer& w) {
  auto& m = *impl_;
  const Variant& v0 = m.variants.front();
  const std::size_t H = kHidden;
  for (const Gemm& g : v0.gemms) {
    if (g.layer != l)
      continue;
    if (g.name == "qkv") {
      m.Pack(g, [&](int n, float* out) {
        const std::uint16_t* row = nullptr;
        if (n < 2048) {
          const int j = n / kHeadDim, part = j / 4, hl = j % 4;
          if (hl < 2) {
            const int q = 2 * (part + 4 * g.pass) + hl;
            row = w.q + (std::size_t(q) * kHeadDim + n % kHeadDim) * H;
          }
        } else if (n < 3072) {
          const std::uint16_t* src = n < 2560 ? w.k : w.v;
          const int j = (n - (n < 2560 ? 2048 : 2560)) / kHeadDim;
          row =
              src + (std::size_t(j + 4 * g.pass) * kHeadDim + n % kHeadDim) * H;
        }
        if (!row) {
          std::fill_n(out, H, 0.0f);
          return;
        }
        for (std::size_t i = 0; i < H; ++i)
          out[i] = FromBf16(row[i]) * w.input_norm[i];
      });
    } else if (g.name == "attn_out") {
      m.Pack(g, [&](int n, float* out) {
        const std::uint16_t* row = w.o + std::size_t(n) * H;  // o_proj row n
        for (int ps = 0; ps < 2; ++ps)
          for (int j = 0; j < 16; ++j) {
            float* d = out + (1 - ps) * 2048 + j * kHeadDim;  // pass 1 first
            const int part = j / 4, hl = j % 4;
            if (hl >= 2) {
              std::fill_n(d, kHeadDim, 0.0f);
              continue;
            }
            const int q = 2 * (part + 4 * ps) + hl;
            for (int e = 0; e < kHeadDim; ++e)
              d[e] = 2.0f * FromBf16(row[q * kHeadDim + e]);
          }
      });
    } else if (g.name == "gate_up") {
      m.Pack(g, [&](int n, float* out) {
        const int jb = n / 128, within = n % 128;
        const std::uint16_t* row =
            within < 64 ? w.gate + std::size_t(64 * jb + within) * H
                        : w.up + std::size_t(64 * jb + within - 64) * H;
        for (std::size_t i = 0; i < H; ++i)
          out[i] = FromBf16(row[i]) * w.post_norm[i];
      });
    } else if (g.name == "down") {
      m.Pack(g, [&](int n, float* out) {
        const std::uint16_t* row = w.down + std::size_t(n) * kFfn;
        for (int i = 0; i < kFfn; ++i)
          out[i] = FromBf16(row[i]);
      });
    } else {
      throw Error("Qwen3Prefill: unknown GEMM " + g.name);
    }
  }
  // q/k norm block: [q norm * log2(e) / sqrt(128) | k norm].
  std::vector<std::uint16_t> qkn(2 * kHeadDim);
  for (int i = 0; i < kHeadDim; ++i) {
    qkn[std::size_t(i)] = host::FloatToBf16(
        w.q_norm[i] * float(1.4426950408889634 / std::sqrt(double(kHeadDim))));
    qkn[std::size_t(kHeadDim + i)] = host::FloatToBf16(w.k_norm[i]);
  }
  for (Variant& v : m.variants) {
    auto* c = reinterpret_cast<std::uint16_t*>(v.c.data());
    for (const auto& [off, al, name] : v.ablocks)
      if (al == l && name == "qkn") {
        PutABlock(c + off, qkn);
        v.c.ToDevice(off * 2, 128 * 512 * 2);
      }
  }
}

void Qwen3Prefill::Run(const float* x, int n) {
  auto& m = *impl_;
  Variant* v = nullptr;
  for (Variant& cand : m.variants)
    if (cand.M >= n) {
      v = &cand;
      break;
    }
  if (!v || n <= 0)
    throw Error("Qwen3Prefill: " + std::to_string(n) + " rows exceed " +
                std::to_string(max_rows()));
  auto* a = reinterpret_cast<std::uint16_t*>(v->a.data());
  host::ParallelFor(std::size_t(v->M), 16, [&](std::size_t lo, std::size_t hi) {
    for (std::size_t r = lo; r < hi; ++r) {
      std::uint16_t* rec = a + r * v->rec;
      std::fill_n(rec, v->rec, std::uint16_t(0));
      if (int(r) < n)
        host::FloatToBf16(x + r * kRecordH, rec + kRecordH, kRecordH);
    }
  });
  v->a.ToDevice();
  v->program.Start(v->ins, {&v->a, &m.b, &v->c}).Wait();
  m.last = v;
}

void Qwen3Prefill::Kv(int l, int n, float* k, float* v) const {
  Variant* var = impl_->last;
  if (!var || n > var->M)
    throw Error("Qwen3Prefill: no run holds these positions");
  // Piece of 16 tokens and core g (KV head 4 pass + g): K at 9216, V at 11264,
  // each [key block 2][dim block 16][8 keys][8 dims].
  const std::size_t blocks = (std::size_t(n) + 15) / 16;
  for (int ps = 0; ps < 2; ++ps) {
    const std::size_t off = var->pieces.at({l, ps});
    var->c.FromDevice(off * 2, blocks * 4 * var->piece * 2);
  }
  const auto* c = reinterpret_cast<const std::uint16_t*>(var->c.data());
  host::ParallelFor(std::size_t(n), 16, [&](std::size_t lo, std::size_t hi) {
    for (std::size_t t = lo; t < hi; ++t) {
      const std::size_t blk = t / 16, slot = t % 16;
      for (int h = 0; h < kKvHeads; ++h) {
        const int ps = h / 4, g = h % 4;
        const std::uint16_t* piece = c + var->pieces.at({l, ps}) +
                                     (blk * 4 + std::size_t(g)) * var->piece;
        float* kd = k + (t * kKvHeads + std::size_t(h)) * kHeadDim;
        float* vd = v + (t * kKvHeads + std::size_t(h)) * kHeadDim;
        for (int d = 0; d < kHeadDim; ++d) {
          const std::size_t at = ((slot >> 3) * 16 + std::size_t(d >> 3)) * 64 +
                                 (slot & 7) * 8 + std::size_t(d & 7);
          kd[d] = FromBf16(piece[kKOffset + at]);
          vd[d] = FromBf16(piece[kVOffset + at]);
        }
      }
    }
  });
}

}  // namespace gufo::npu
