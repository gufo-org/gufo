#include "src/core/npu/qwen3_decoder.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "src/core/npu/host_ops.hpp"

// qwen3dec.py's layouts (kernels/qwen3dec/layout.py; offsets from
// qwen3dec_layout.txt). Weight records are REC bytes per stream s, record r
// and core j of the stream's pair: [stream][record][core], the cores of
// stream s being j = 2 (s % 2) and 2 (s % 2) + 1 of column s / 2.
namespace gufo::npu {
namespace {

constexpr int kH = 2048, kI = 6144, kRec = 4 * kH + 4 * (kH / 32) * 2;
constexpr int kQkvRecs = 32, kOutRecs = 16, kGuRecs = 96, kDownRecs = 48;
constexpr unsigned kArgS = 3, kArgW = 4;

std::uint16_t ToBf16(float f) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return std::uint16_t(u >> 16);
}
float FromBf16(std::uint16_t h) {
  return std::bit_cast<float>(std::uint32_t(h) << 16);
}

// Row `l` of core (c, j) for a vector of `chunks` x 8 rows per core.
int Row(int c, int j, int l, int chunks) {
  return ((c * chunks + l / 8) * 4 + j) * 8 + l % 8;
}

// Q8_0 of a bf16 row's blocks [b0, b0 + nb): codes and bf16 scales.
void QuantizeRow(const std::uint16_t* row, int b0, int nb, std::int8_t* codes,
                 std::uint16_t* scales) {
  for (int b = 0; b < nb; ++b) {
    const std::uint16_t* x = row + 32 * (b0 + b);
    float amax = 0;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::fabs(FromBf16(x[i])));
    const std::uint16_t d16 = ToBf16(amax / 127.0f);
    const float d = FromBf16(d16), inv = d > 0 ? 1.0f / d : 0.0f;
    scales[b] = d16;
    for (int i = 0; i < 32; ++i)
      codes[32 * b + i] = std::int8_t(
          std::clamp(std::nearbyint(FromBf16(x[i]) * inv), -127.0f, 127.0f));
  }
}

// tile4 (Dot4's order): four rows of nb blocks; rows may be null (zeros).
// Writes nb * 128 code bytes, then the scales [block][row].
void Tile4(const std::uint16_t* const rows[4], int b0, int nb,
           std::int8_t* out) {
  std::vector<std::int8_t> c(std::size_t(4) * 32 * nb, 0);
  std::vector<std::uint16_t> s(std::size_t(4) * nb, 0);
  for (int r = 0; r < 4; ++r)
    if (rows[r])
      QuantizeRow(rows[r], b0, nb, c.data() + std::size_t(r) * 32 * nb,
                  s.data() + std::size_t(r) * nb);
  for (int b = 0; b < nb; ++b)
    for (int t = 0; t < 2; ++t)
      for (int k = 0; k < 8; ++k)
        for (int half = 0; half < 2; ++half)
          for (int r = 0; r < 4; ++r)
            out[(((b * 2 + t) * 8 + k) * 2 + half) * 4 + r] =
                c[std::size_t(r) * 32 * nb + 32 * b + 16 * t + 8 * half + k];
  auto* sc = reinterpret_cast<std::uint16_t*>(out + 128 * nb);
  for (int b = 0; b < nb; ++b)
    for (int r = 0; r < 4; ++r)
      sc[b * 4 + r] = s[std::size_t(r) * nb + b];
}

}  // namespace

struct Qwen3Decoder::Impl {
  Program program;
  DispatchLibrary generate;
  Buffer s, w;
  Instructions ins;
  std::map<std::string, std::size_t> lay;
  int head_rows{}, head_chunks{};
  std::vector<float> logits;
  std::vector<std::uint32_t> next;
  int next_pos{-1};
  Run pending;

  std::size_t At(const std::string& key) const {
    auto it = lay.find(key);
    if (it == lay.end())
      throw Error("qwen3dec layout lacks " + key);
    return it->second;
  }
  std::int8_t* S(std::size_t at) {
    return reinterpret_cast<std::int8_t*>(s.data()) + at;
  }
  std::int8_t* Wb(std::size_t at) {
    return reinterpret_cast<std::int8_t*>(w.data()) + at;
  }

  // Records `records` per core for every stream from build(c, j, r, out).
  template<typename Build>
  void Pack(std::size_t base, int records, Build build) {
    host::ParallelFor(
        std::size_t(16) * records, 4, [&](std::size_t lo, std::size_t hi) {
          for (std::size_t i = lo; i < hi; ++i) {
            const int st = int(i) / records, r = int(i) % records, c = st / 2;
            for (int p = 0; p < 2; ++p) {
              const int j = 2 * (st % 2) + p;
              std::int8_t* rec =
                  Wb(base + ((std::size_t(st) * records + r) * 2 + p) * kRec);
              std::memset(rec, 0, kRec);
              build(c, j, r, rec);
            }
          }
        });
    w.ToDevice(base, std::size_t(16) * records * 2 * kRec);
  }

  void WriteBf16(std::size_t at, const float* v, int n) {
    auto* p = reinterpret_cast<std::uint16_t*>(S(at));
    for (int i = 0; i < n; ++i)
      p[i] = ToBf16(v[i]);
    s.ToDevice(at, std::size_t(n) * 2);
  }

  std::size_t Block(int b) const {
    return At("blocks") + std::size_t(b) * At("block_bytes");
  }

  std::vector<std::uint32_t> Generate(int pos) const {
    // The KV stream's records in four tasks of 1..64 records each.
    const int recs = std::max(4, (pos + 15) / 16);
    if (recs > 4 * 64)
      throw Error("Qwen3Decoder: position beyond the KV cache");
    std::int32_t sc[6] = {pos, 0, 0, 0, 0, head_chunks};
    for (int i = 0; i < 4; ++i)
      sc[1 + i] = std::int32_t(recs / 4 + (i < recs % 4 ? 1 : 0));
    const auto words = generate.Generate(sc);
    return {words.begin(), words.end()};
  }
};

Qwen3Decoder Qwen3Decoder::Open(Device& device,
                                const std::filesystem::path& dir) {
  Qwen3Decoder d;
  d.impl_ = std::make_unique<Impl>();
  auto& m = *d.impl_;
  std::ifstream in(dir / "qwen3dec_layout.txt");
  std::string key;
  std::size_t value;
  while (in >> key >> value)
    m.lay[key] = value;
  if (m.lay.empty())
    throw Error("Qwen3Decoder: no layout in " + dir.string());
  m.program = Program::Load(device, dir / "qwen3dec.xclbin");
  m.generate = DispatchLibrary::Open(dir / "qwen3dec_dispatch.so", 6);
  m.s = m.program.Allocate(m.At("s_bytes"), kArgS);
  m.w = m.program.Allocate(m.At("w_bytes"), kArgW);
  std::memset(m.s.data(), 0, m.s.size());
  m.s.ToDevice();
  m.ins = m.program.AllocateInstructions(std::size_t(1) << 20);
  return d;
}

Qwen3Decoder::Qwen3Decoder() = default;
Qwen3Decoder::~Qwen3Decoder() {
  if (impl_ && impl_->pending)
    impl_->pending.Wait();
}
Qwen3Decoder::Qwen3Decoder(Qwen3Decoder&&) noexcept = default;
Qwen3Decoder& Qwen3Decoder::operator=(Qwen3Decoder&&) noexcept = default;

int Qwen3Decoder::capacity() const noexcept {
  return int(impl_->lay.at("ctx"));
}

void Qwen3Decoder::SetLayer(int l, const Layer& L) {
  auto& m = *impl_;
  const std::string p = "w." + std::to_string(l) + ".";
  // q/k/v rows: [q 2048 | k 1024 | v 1024], 128 per core.
  auto qkv_row = [&](int r) -> const std::uint16_t* {
    if (r < 2048)
      return L.q + std::size_t(r) * kH;
    if (r < 3072)
      return L.k + std::size_t(r - 2048) * kH;
    return L.v + std::size_t(r - 3072) * kH;
  };
  m.Pack(m.At(p + "qkv"), kQkvRecs, [&](int c, int j, int r, std::int8_t* rec) {
    const std::uint16_t* rows[4];
    for (int i = 0; i < 4; ++i)
      rows[i] = qkv_row(Row(c, j, 4 * r + i, 16));
    Tile4(rows, 0, kH / 32, rec);
  });
  m.Pack(m.At(p + "out"), kOutRecs, [&](int c, int j, int r, std::int8_t* rec) {
    const std::uint16_t* rows[4];
    for (int i = 0; i < 4; ++i)
      rows[i] = L.o + std::size_t(Row(c, j, 4 * r + i, 8)) * kH;
    Tile4(rows, 0, kH / 32, rec);
  });
  m.Pack(
      m.At(p + "gate_up"), kGuRecs, [&](int c, int j, int r, std::int8_t* rec) {
        int ch[2];
        for (int x = 0; x < 2; ++x) {
          const int e = 2 * r + x;
          ch[x] = ((c * 12 + e / 16) * 4 + j) * 16 + e % 16;
        }
        const std::uint16_t* rows[4] = {
            L.gate + std::size_t(ch[0]) * kH, L.gate + std::size_t(ch[1]) * kH,
            L.up + std::size_t(ch[0]) * kH, L.up + std::size_t(ch[1]) * kH};
        Tile4(rows, 0, kH / 32, rec);
      });
  // Down: per set of 16 rows, four 4-row groups of 192 blocks of I; a record
  // holds 64 blocks of one group.
  m.Pack(m.At(p + "down"), kDownRecs,
         [&](int c, int j, int r, std::int8_t* rec) {
           const int set = r / 12, u = 64 * (r % 12), g = u / 192, kb = u % 192;
           const std::uint16_t* rows[4];
           for (int i = 0; i < 4; ++i)
             rows[i] =
                 L.down + std::size_t(Row(c, j, 16 * set + 4 * g + i, 8)) * kI;
           Tile4(rows, kb, 64, rec);
         });
  // Norm weights: block 2l (attention input) and 2l + 1 (FFN input) hold
  // their pre-norm weight at byte 20480; the q/k norms follow the q/k/v rows.
  m.WriteBf16(m.Block(2 * l) + 20480, L.input_norm, kH);
  m.WriteBf16(m.Block(2 * l + 1) + 20480, L.post_attention_norm, kH);
  const std::size_t q = m.At("qkv." + std::to_string(l)) + 4096 * 4;
  m.WriteBf16(q, L.q_norm, 128);
  m.WriteBf16(q + 256, L.k_norm, 128);
}

void Qwen3Decoder::SetHead(const std::uint16_t* weights, int rows,
                           const float* norm) {
  auto& m = *impl_;
  if (rows <= 0 || rows > kMaxHeadRows)
    throw Error("Qwen3Decoder: unsupported head rows");
  m.head_rows = rows;
  m.head_chunks = (rows + 255) / 256;
  const int chunks = m.head_chunks, records = chunks * 2;
  m.Pack(m.At("w.head"), records, [&](int c, int j, int r, std::int8_t* rec) {
    const std::uint16_t* rs[4];
    for (int i = 0; i < 4; ++i) {
      const int row = Row(c, j, 4 * r + i, chunks);
      rs[i] = row < rows ? weights + std::size_t(row) * kH : nullptr;
    }
    Tile4(rs, 0, kH / 32, rec);
  });
  m.WriteBf16(m.Block(2 * kLayers) + 20480, norm, kH);
  m.logits.assign(std::size_t(rows), 0.0f);
}

void Qwen3Decoder::SetRopeTheta(double theta) {
  auto& m = *impl_;
  const int ctx = capacity();
  const std::size_t at = m.At("rope");
  for (int pos = 0; pos <= ctx; ++pos) {
    auto* p = reinterpret_cast<float*>(m.S(at + std::size_t(pos) * 512));
    for (int i = 0; i < 64; ++i) {
      const double a = pos * std::pow(theta, -double(2 * i) / 128.0);
      p[i] = float(std::cos(a));
      p[64 + i] = float(std::sin(a));
    }
  }
  m.s.ToDevice(at, std::size_t(ctx + 1) * 512);
}

void Qwen3Decoder::WriteKv(int layer, int pos, int count, const float* k,
                           std::size_t k_stride, const float* v,
                           std::size_t v_stride) {
  auto& m = *impl_;
  const std::size_t ctx = std::size_t(capacity());
  if (pos < 0 || count < 0 || std::size_t(pos + count) > ctx)
    throw Error("Qwen3Decoder: KV entries beyond the cache");
  const std::size_t base = m.At("kv." + std::to_string(layer));
  // Column c's first two cores (the attention cores of KV head c) each keep
  // [ctx][k bf16 128 | v bf16 128].
  host::ParallelFor(
      std::size_t(kKvHeads) * 2, 1, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t cj = lo; cj < hi; ++cj) {
          const std::size_t c = cj / 2, j = cj % 2;
          const std::size_t at =
              base + ((c * 4 + j) * ctx + std::size_t(pos)) * 512;
          for (int r = 0; r < count; ++r) {
            auto* p = reinterpret_cast<std::uint16_t*>(
                m.S(at + std::size_t(r) * 512));
            const float* kr = k + std::size_t(r) * k_stride + c * 128;
            const float* vr = v + std::size_t(r) * v_stride + c * 128;
            for (int i = 0; i < 128; ++i) {
              p[i] = ToBf16(kr[i]);
              p[128 + i] = ToBf16(vr[i]);
            }
          }
          m.s.ToDevice(at, std::size_t(count) * 512);
        }
      });
}

void Qwen3Decoder::Start(const float* row, int pos) {
  auto& m = *impl_;
  if (m.pending)
    m.pending.Wait();
  if (pos < 0 || pos >= capacity())
    throw Error("Qwen3Decoder: position beyond the KV cache");
  const std::size_t at = m.At("input");
  std::memcpy(m.S(at), row, std::size_t(kH) * 4);
  m.s.ToDevice(at, std::size_t(kH) * 4);
  if (m.next_pos == pos)
    m.ins.Assign(m.next);
  else
    m.ins.Assign(m.Generate(pos));
  m.pending = m.program.Start(m.ins, {&m.s, &m.w});
  // The next position's stream, on the host while the NPU runs.
  if (pos + 1 < capacity()) {
    m.next = m.Generate(pos + 1);
    m.next_pos = pos + 1;
  }
}

void Qwen3Decoder::Wait() {
  auto& m = *impl_;
  if (!m.pending)
    return;
  m.pending.Wait();
  m.pending = Run();
  const std::size_t at = m.At("logits");
  m.s.FromDevice(at, std::size_t(m.head_rows) * 4);
  std::memcpy(m.logits.data(), m.S(at), std::size_t(m.head_rows) * 4);
}

std::span<const float> Qwen3Decoder::logits() {
  return impl_->logits;
}

void Qwen3Decoder::Hidden(float* out) {
  auto& m = *impl_;
  const std::size_t at = m.Block(2 * kLayers);
  m.s.FromDevice(at, std::size_t(2) * kH * 4);
  std::vector<float> ho(std::size_t(2) * kH);
  std::memcpy(ho.data(), m.S(at), ho.size() * 4);
  for (int i = 0; i < kH; ++i)
    out[i] = ho[std::size_t(i)] + ho[std::size_t(kH) + std::size_t(i)];
}

}  // namespace gufo::npu
