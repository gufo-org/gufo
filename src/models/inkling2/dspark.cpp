#include "src/models/inkling2/dspark.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string_view>

#include "src/core/gguf_reader.hpp"
#include "src/models/inkling2/hip/dspark.hpp"
#include "src/models/inkling2/hip/fast.hpp"
#include "src/models/inkling2/hip/kernels.hpp"

namespace gufo::models::inkling2 {
namespace fs = fast;
namespace kk = hip;
namespace {

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string("inkling2 dspark: ") + what + ": " +
                             hipGetErrorString(e));
}

using core::GgmlType;
using core::GgufReader;
using core::GgufTensorInfo;

constexpr int kE = dsp::kEmbd;

}  // namespace

struct Drafter::Impl {
  std::unique_ptr<GgufReader> reader;
  Target target{};
  std::vector<void*> allocs;
  float theta = 1e6f, eps = 1e-6f;

  struct Layer {
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm, *k_ctx_norm, *ctx_norm;
    const void *wq, *wk, *wv, *wo, *wg, *wu, *wd, *wk_ctx, *wv_ctx;
  };
  std::array<Layer, 3> layer{};
  const float *mask = nullptr, *out_norm = nullptr, *markov_norm = nullptr,
              *bj = nullptr;
  const uint16_t *w1 = nullptr, *wj = nullptr;
  float alpha[4][4]{};

  // scratch
  float *x, *h, *q, *k, *v, *y, *g, *u, *logits, *state, *joint, *ws, *wsa,
      *mix, *d_conf;
  // Low-rank Markov correction of the proposal head: pt = (W_U W_o)^T
  // [256][kHeadRows] F16, cb = W_U b_o, base = W_U h_i [8][kHeadRows].
  _Float16* pt = nullptr;
  float *cb = nullptr, *base = nullptr;
  // Proposal head: the target LM head's rows of the kHeadRows tokens with the
  // largest markov_w1 norms (the Loom package's choice), as a Q8_0 copy.
  static constexpr int kHeadRows = 65536;
  void* head = nullptr;
  int* head_ids = nullptr;
  _Float16 *h16, *o16, *a16, *mix16;
  float* ikv;  // injection K | V [n][2048]
  int *d_anchor, *d_pos, *d_prev, *d_tokens;
  // pinned host staging: [anchor, pos] in, [tokens 8 | confidences 8] out
  int* h_in = nullptr;
  float* h_out = nullptr;
  // work items of n = 1 .. 8 rows (one item each) for injections
  int* small_items = nullptr;
  int *it_n, *it_e, *it_s, *it_c;  // GEMM work items (n rows)
  int* blk_items;                  // the 8-row block: one item
  static constexpr int kMaxInject = 2048;

  template<class T>
  T* Alloc(std::size_t n) {
    void* p = nullptr;
    Check(hipMalloc(&p, n * sizeof(T)), "scratch");
    allocs.push_back(p);
    return static_cast<T*>(p);
  }
  ~Impl() {
    for (void* p : allocs)
      (void)hipFree(p);
    if (h_in)
      (void)hipHostFree(h_in);
    if (h_out)
      (void)hipHostFree(h_out);
  }

  const GgufTensorInfo& Need(std::string_view name, GgmlType type,
                             std::uint64_t elements) const {
    const GgufTensorInfo* t = reader->FindTensor(name);
    if (t == nullptr)
      throw std::runtime_error("missing tensor " + std::string(name));
    if (t->type != type || t->ElementCount() != elements)
      throw std::runtime_error("unexpected tensor " + std::string(name));
    return *t;
  }
  const void* Upload(const GgufTensorInfo& t, std::size_t* total) {
    void* p = Alloc<char>(t.size_bytes);
    Check(hipMemcpy(p, t.data, t.size_bytes, hipMemcpyHostToDevice),
          "weight upload");
    *total += t.size_bytes;
    return p;
  }
  const float* F32(std::string_view n, std::uint64_t e, std::size_t* tot) {
    return static_cast<const float*>(Upload(Need(n, GgmlType::kF32, e), tot));
  }
  const void* Q8(std::string_view n, std::uint64_t e, std::size_t* tot) {
    return Upload(Need(n, GgmlType::kQ8_0, e), tot);
  }
  const uint16_t* BF16(std::string_view n, std::uint64_t e, std::size_t* tot) {
    return static_cast<const uint16_t*>(
        Upload(Need(n, GgmlType::kBF16, e), tot));
  }

  fs::GemmItems Items(int* n_items, int* e, int* s, int* c, int rows) {
    fs::GemmItems it{};
    it.nitems = n_items;
    it.item_e = e;
    it.item_start = s;
    it.item_n = c;
    it.max_items = (rows + 63) / 64;
    it.tile = 64;
    return it;
  }
  void Gemm(hipStream_t s, const void* w, int m, int k, const void* a,
            float* out, const fs::GemmItems& it) {
    fs::GemmDense(s, w, m, k, a, out, it);
  }
};

std::unique_ptr<Drafter> Drafter::Open(const std::string& path,
                                       const Target& target,
                                       std::string* error) {
  std::unique_ptr<Drafter> d(new Drafter());
  try {
    auto impl = std::make_unique<Impl>();
    std::string err;
    impl->reader = GgufReader::OpenFile(path, &err);
    if (!impl->reader)
      throw std::runtime_error(err);
    const GgufReader& r = *impl->reader;
    if (r.GetMetadataString("general.architecture").value_or("") != "dflash")
      throw std::runtime_error("not a DFlash/DSpark GGUF");
    auto u32 = [&](const char* k) { return r.GetMetadataUint32(k); };
    if (u32("dflash.block_count") != 3u ||
        u32("dflash.embedding_length") != 2304u ||
        u32("dflash.feed_forward_length") != 9216u ||
        u32("dflash.attention.head_count") != 24u ||
        u32("dflash.attention.head_count_kv") != 8u ||
        u32("dflash.block_size") != 8u ||
        u32("dflash.dspark.markov_rank") != 256u ||
        u32("dflash.attention.sliding_window") != 1024u)
      throw std::runtime_error("unsupported DSpark hyperparameters");
    {
      const auto* m = r.FindMetadata("dflash.target_layers");
      const auto* v =
          m ? std::get_if<std::vector<std::uint64_t>>(&m->value) : nullptr;
      const auto* vi =
          m ? std::get_if<std::vector<std::int64_t>>(&m->value) : nullptr;
      std::vector<std::int64_t> ids;
      if (v)
        ids.assign(v->begin(), v->end());
      else if (vi)
        ids = *vi;
      if (ids != std::vector<std::int64_t>{5, 23, 41, 48})
        throw std::runtime_error("unsupported DSpark target layers");
    }
    impl->theta = r.GetMetadataFloat32("dflash.rope.freq_base").value_or(1e6f);
    impl->eps = r.GetMetadataFloat32("dflash.attention.layer_norm_rms_epsilon")
                    .value_or(1e-6f);
    impl->target = target;
    std::size_t tot = 0;
    Impl& I = *impl;
    for (int l = 0; l < 3; ++l) {
      const std::string b = "blk." + std::to_string(l) + ".";
      auto& L = I.layer[l];
      L.attn_norm = I.F32(b + "attn_norm.weight", kE, &tot);
      L.ffn_norm = I.F32(b + "ffn_norm.weight", kE, &tot);
      L.q_norm = I.F32(b + "attn_q_norm.weight", 128, &tot);
      L.k_norm = I.F32(b + "attn_k_norm.weight", 128, &tot);
      L.k_ctx_norm = I.F32(b + "attn_k_ctx_norm.weight", 128, &tot);
      L.ctx_norm = I.F32(b + "dflare_norm.weight", kE, &tot);
      L.wq = I.Q8(b + "attn_q.weight", 2304ull * 3072, &tot);
      L.wk = I.Q8(b + "attn_k.weight", 2304ull * 1024, &tot);
      L.wv = I.Q8(b + "attn_v.weight", 2304ull * 1024, &tot);
      L.wo = I.Q8(b + "attn_output.weight", 3072ull * 2304, &tot);
      L.wg = I.Q8(b + "ffn_gate.weight", 2304ull * 9216, &tot);
      L.wu = I.Q8(b + "ffn_up.weight", 2304ull * 9216, &tot);
      L.wd = I.Q8(b + "ffn_down.weight", 9216ull * 2304, &tot);
      L.wk_ctx = I.Q8(b + "attn_k_ctx.weight", 2304ull * 1024, &tot);
      L.wv_ctx = I.Q8(b + "attn_v_ctx.weight", 2304ull * 1024, &tot);
    }
    I.mask = I.F32("mask_embd.weight", kE, &tot);
    I.out_norm = I.F32("output_norm.weight", kE, &tot);
    I.markov_norm = I.F32("dspark_norm.weight", kE, &tot);
    I.bj = I.F32("joint_proj.bias", 768, &tot);
    I.wj = I.BF16("joint_proj.weight", 768ull * 5120, &tot);
    I.w1 = I.BF16("markov_w1.weight", 256ull * 201024, &tot);
    {
      const auto& fuse = I.Need("dspark_fuse.weight", GgmlType::kF32, 16);
      const float* f = static_cast<const float*>(fuse.data);
      for (int j = 0; j < 4; ++j) {  // softmax of each row (as the converter)
        float m = -INFINITY, z = 0.f;
        for (int i = 0; i < 4; ++i)
          m = std::max(m, f[j * 4 + i]);
        for (int i = 0; i < 4; ++i)
          z += std::exp(f[j * 4 + i] - m);
        for (int i = 0; i < 4; ++i)
          I.alpha[j][i] = std::exp(f[j * 4 + i] - m) / z;
      }
    }
    d->weight_bytes_ = tot;
    constexpr int B = dsp::kBlock;
    I.x = I.Alloc<float>(B * kE);
    I.h = I.Alloc<float>(B * kE);
    I.h16 = I.Alloc<_Float16>(B * kE);
    I.q = I.Alloc<float>(B * 3072);
    I.k = I.Alloc<float>(B * 1024);
    I.v = I.Alloc<float>(B * 1024);
    I.o16 = I.Alloc<_Float16>(B * 3072);
    I.y = I.Alloc<float>(B * kE);
    I.g = I.Alloc<float>(B * 9216);
    I.u = I.Alloc<float>(B * 9216);
    I.a16 = I.Alloc<_Float16>(B * 9216);
    I.logits = I.Alloc<float>(static_cast<std::size_t>(target.vocab));
    I.state = I.Alloc<float>(256);
    I.joint = I.Alloc<float>(1024);
    I.ws = I.Alloc<float>(3 * 128 + 1);
    I.d_conf = I.Alloc<float>(8);
    {
      // the kHeadRows text tokens with the largest markov_w1 rows
      const auto& w1t =
          I.Need("markov_w1.weight", GgmlType::kBF16, 256ull * 201024);
      const auto* w1h = static_cast<const std::uint16_t*>(w1t.data);
      const int V = target.vocab;
      std::vector<std::pair<float, int>> norm(static_cast<std::size_t>(V));
      for (int t = 0; t < V; ++t) {
        float s = 0.f;
        for (int c = 0; c < 256; ++c) {
          std::uint32_t u =
              static_cast<std::uint32_t>(w1h[(std::size_t)t * 256 + c]) << 16;
          float f;
          std::memcpy(&f, &u, 4);
          s += f * f;
        }
        norm[static_cast<std::size_t>(t)] = {s, t};
      }
      const int keep = std::min(Impl::kHeadRows, V);
      std::partial_sort(norm.begin(), norm.begin() + keep, norm.end(),
                        [](const auto& a, const auto& b) {
                          return a.first > b.first ||
                                 (a.first == b.first && a.second < b.second);
                        });
      std::vector<int> ids(static_cast<std::size_t>(keep));
      for (int i = 0; i < keep; ++i)
        ids[static_cast<std::size_t>(i)] =
            norm[static_cast<std::size_t>(i)].second;
      std::sort(ids.begin(), ids.end());
      I.head_ids = I.Alloc<int>(ids.size());
      Check(hipMemcpy(I.head_ids, ids.data(), ids.size() * 4,
                      hipMemcpyHostToDevice),
            "head ids");
      I.head = I.Alloc<char>(ids.size() * (2304 / 32) * 34);
      dsp::GatherQ8Rows(nullptr, target.output, I.head_ids,
                        static_cast<int>(ids.size()), I.head);
      Check(hipDeviceSynchronize(), "head copy");
      tot += ids.size() * (2304 / 32) * 34;

      // (W_U W_o)^T and W_U b_o in one GEMM: the head against the 256
      // columns of W_o and b_o as 257 activation rows
      const int R = static_cast<int>(ids.size());
      const auto& wot =
          I.Need("dspark_hidden_out.weight", GgmlType::kBF16, 256ull * kE);
      const auto& bot = I.Need("dspark_hidden_out.bias", GgmlType::kF32, kE);
      const auto* woh = static_cast<const std::uint16_t*>(wot.data);
      const auto* boh = static_cast<const float*>(bot.data);
      std::vector<float> xt(static_cast<std::size_t>(257) * kE);
      for (int c = 0; c < kE; ++c) {
        for (int j = 0; j < 256; ++j) {
          const std::uint32_t u =
              static_cast<std::uint32_t>(woh[static_cast<std::size_t>(c) * 256 +
                                             static_cast<std::size_t>(j)])
              << 16;
          std::memcpy(&xt[static_cast<std::size_t>(j) * kE +
                          static_cast<std::size_t>(c)],
                      &u, 4);
        }
        xt[static_cast<std::size_t>(256) * kE + static_cast<std::size_t>(c)] =
            boh[c];
      }
      float *dx = nullptr, *dout = nullptr;
      Check(hipMalloc(reinterpret_cast<void**>(&dx), xt.size() * 4), "lr in");
      Check(hipMalloc(reinterpret_cast<void**>(&dout),
                      static_cast<std::size_t>(257) * R * 4),
            "lr out");
      Check(hipMemcpy(dx, xt.data(), xt.size() * 4, hipMemcpyHostToDevice),
            "lr in");
      kk::Gemm(nullptr, kk::kTQ8_0, I.head, R, kE, dx, kE, dout, R, 257);
      I.pt = I.Alloc<_Float16>(static_cast<std::size_t>(256) * R);
      I.cb = I.Alloc<float>(static_cast<std::size_t>(R));
      I.base = I.Alloc<float>(static_cast<std::size_t>(dsp::kBlock) * R);
      dsp::ToHalf(nullptr, dout, static_cast<std::size_t>(256) * R, I.pt);
      Check(hipMemcpy(I.cb, dout + static_cast<std::size_t>(256) * R,
                      static_cast<std::size_t>(R) * 4, hipMemcpyDeviceToDevice),
            "lr bias");
      Check(hipDeviceSynchronize(), "lr tables");
      (void)hipFree(dx);
      (void)hipFree(dout);
      tot += static_cast<std::size_t>(256) * R * 2 +
             static_cast<std::size_t>(R) * 4;
    }
    I.wsa = I.Alloc<float>(static_cast<std::size_t>(dsp::kHeads) *
                           dsp::kAttnSplits * B * (128 + 2));
    I.mix = I.Alloc<float>(static_cast<std::size_t>(Impl::kMaxInject) * 4 * kE);
    I.mix16 =
        I.Alloc<_Float16>(static_cast<std::size_t>(Impl::kMaxInject) * 3 * kE);
    I.ikv = I.Alloc<float>(static_cast<std::size_t>(Impl::kMaxInject) * 2048);
    I.d_anchor = I.Alloc<int>(1);
    I.d_pos = I.Alloc<int>(1);
    I.d_prev = I.Alloc<int>(1);
    I.d_tokens = I.Alloc<int>(8);
    I.it_n = I.Alloc<int>(1);
    I.it_e = I.Alloc<int>(Impl::kMaxInject / 64);
    I.it_s = I.Alloc<int>(Impl::kMaxInject / 64);
    I.it_c = I.Alloc<int>(Impl::kMaxInject / 64);
    I.blk_items = I.Alloc<int>(4);
    Check(hipHostMalloc(reinterpret_cast<void**>(&I.h_in), 4 * sizeof(int)),
          "pinned");
    Check(hipHostMalloc(reinterpret_cast<void**>(&I.h_out), 16 * sizeof(float)),
          "pinned");
    I.small_items = I.Alloc<int>(8 * 4);
    {
      int host[8 * 4];
      for (int r = 0; r < 8; ++r) {
        host[r * 4] = 1;
        host[r * 4 + 1] = 0;
        host[r * 4 + 2] = 0;
        host[r * 4 + 3] = r + 1;
      }
      Check(hipMemcpy(I.small_items, host, sizeof host, hipMemcpyHostToDevice),
            "items");
    }
    {
      const int host[4] = {1, 0, 0, B};
      Check(hipMemcpy(I.blk_items, host, sizeof host, hipMemcpyHostToDevice),
            "items");
      std::vector<int> zeros(Impl::kMaxInject / 64, 0);
      Check(hipMemcpy(I.it_e, zeros.data(), zeros.size() * 4,
                      hipMemcpyHostToDevice),
            "items");
    }
    d->impl_ = std::move(impl);
  } catch (const std::exception& e) {
    if (error)
      *error = std::string("DSpark draft ") + path + ": " + e.what();
    return nullptr;
  }
  return d;
}

Drafter::~Drafter() = default;

void Drafter::Inject(hipStream_t s, State& st, const float* feats,
                     std::size_t pos0, int n) {
  Impl& I = *impl_;
  if (n <= 0)
    return;
  if (pos0 > st.valid || pos0 < st.first)
    st.first = pos0;  // a gap: the context before pos0 is missing
  for (int done = 0; done < n; done += Impl::kMaxInject) {
    const int m = std::min(n - done, Impl::kMaxInject);
    const float* f = feats + static_cast<std::size_t>(done) * kFeatureFloats;
    const std::size_t p0 = pos0 + static_cast<std::size_t>(done);
    // work items of m rows (tile 64): preset for the few rows of a decode
    // step, uploaded for a prefill chunk
    fs::GemmItems it{};
    if (m <= 8) {
      int* b = I.small_items + (m - 1) * 4;
      it = I.Items(b, b + 1, b + 2, b + 3, m);
    } else {
      const int ni = (m + 63) / 64;
      std::vector<int> host(1 + 2 * ni);
      host[0] = ni;
      for (int i = 0; i < ni; ++i) {
        host[1 + i] = i * 64;
        host[1 + ni + i] = std::min(64, m - i * 64);
      }
      Check(hipMemcpyAsync(I.it_n, host.data(), 4, hipMemcpyHostToDevice, s),
            "items");
      Check(hipMemcpyAsync(I.it_s, host.data() + 1, ni * 4,
                           hipMemcpyHostToDevice, s),
            "items");
      Check(hipMemcpyAsync(I.it_c, host.data() + 1 + ni, ni * 4,
                           hipMemcpyHostToDevice, s),
            "items");
      // the pageable copies above are staged: wait before `host` goes away
      Check(hipStreamSynchronize(s), "items");
      it = I.Items(I.it_n, I.it_e, I.it_s, I.it_c, m);
    }
    dsp::MixArgs ma{};
    ma.feat = f;
    std::memcpy(ma.alpha, I.alpha, sizeof ma.alpha);
    for (int l = 0; l < 3; ++l)
      ma.w[l] = I.layer[l].ctx_norm;
    ma.w[3] = I.markov_norm;
    ma.out = I.mix;
    ma.out16 = I.mix16;
    ma.n = m;
    ma.eps = I.eps;
    dsp::Mix(s, ma);
    for (int l = 0; l < 3; ++l) {
      const _Float16* a = I.mix16 + static_cast<std::size_t>(l) * m * kE;
      float* kraw = I.ikv;
      float* vraw = I.ikv + static_cast<std::size_t>(m) * 1024;
      I.Gemm(s, I.layer[l].wk_ctx, 1024, kE, a, kraw, it);
      I.Gemm(s, I.layer[l].wv_ctx, 1024, kE, a, vraw, it);
      dsp::KvStoreArgs ka{};
      ka.k = kraw;
      ka.v = vraw;
      ka.ldk = ka.ldv = 1024;
      ka.knorm = I.layer[l].k_ctx_norm;
      ka.kc = static_cast<_Float16*>(st.k[l]);
      ka.vc = static_cast<_Float16*>(st.v[l]);
      ka.cap = static_cast<int>(st.cap[l]);
      ka.pos0 = static_cast<int>(p0);
      ka.n = m;
      ka.theta = I.theta;
      ka.eps = I.eps;
      dsp::KvStore(s, ka);
    }
    // the Markov context of the last row
    Check(hipMemcpyAsync(st.f,
                         I.mix + (static_cast<std::size_t>(m - 1) * 4 + 3) * kE,
                         kE * 4, hipMemcpyDeviceToDevice, s),
          "markov context");
    st.f_pos = p0 + static_cast<std::size_t>(m) - 1;
  }
  st.valid = pos0 + static_cast<std::size_t>(n);
}

int Drafter::Propose(hipStream_t s, State& st, std::int32_t anchor,
                     std::size_t pos, int n, std::int32_t* out,
                     float* confidence, const More& more,
                     const Coupled* coupled) {
  Impl& I = *impl_;
  if (n <= 0)
    return 0;
  n = std::min(n, kMaxDraft);
  constexpr int B = dsp::kBlock;
  I.h_in[0] = anchor;
  I.h_in[1] = static_cast<int>(pos);
  Check(hipMemcpyAsync(I.d_anchor, I.h_in, 4, hipMemcpyHostToDevice, s),
        "anchor");
  Check(hipMemcpyAsync(I.d_pos, I.h_in + 1, 4, hipMemcpyHostToDevice, s),
        "pos");
  Check(hipMemcpyAsync(I.d_prev, I.h_in, 4, hipMemcpyHostToDevice, s), "prev");
  dsp::Embed(s, I.target.token_embd, I.target.token_embd_norm, I.d_anchor,
             I.mask, I.layer[0].attn_norm, I.x, I.h, I.h16, I.eps);
  for (int l = 0; l < 3; ++l) {
    const auto& L = I.layer[l];
    dsp::Gemv8(s, L.wq, 3072, kE, I.h16, I.q);
    dsp::Gemv8(s, L.wk, 1024, kE, I.h16, I.k);
    dsp::Gemv8(s, L.wv, 1024, kE, I.h16, I.v);
    dsp::QkNormRope(s, I.q, I.k, L.q_norm, L.k_norm, I.d_pos, I.theta, I.eps);
    dsp::AttnArgs aa{};
    aa.q = I.q;
    aa.k = I.k;
    aa.v = I.v;
    aa.kc = static_cast<const _Float16*>(st.k[l]);
    aa.vc = static_cast<const _Float16*>(st.v[l]);
    aa.cap = static_cast<int>(st.cap[l]);
    aa.pos = I.d_pos;
    aa.window = WindowLayer(l) ? static_cast<int>(kWindow) : 0;
    aa.first = static_cast<int>(st.first);
    aa.ws = I.wsa;
    aa.out16 = I.o16;
    dsp::Attn(s, aa);
    dsp::Gemv8(s, L.wo, kE, 3072, I.o16, I.y);
    dsp::AddNorm(s, I.x, I.y, L.ffn_norm, I.h, I.h16, B, I.eps);
    dsp::Gemv8(s, L.wg, 9216, kE, I.h16, I.g);
    dsp::Gemv8(s, L.wu, 9216, kE, I.h16, I.u);
    dsp::Swiglu(s, I.g, I.u, I.a16, B * 9216);
    dsp::Gemv8(s, L.wd, kE, 9216, I.a16, I.y);
    dsp::AddNorm(s, I.x, I.y, l < 2 ? I.layer[l + 1].attn_norm : I.out_norm,
                 I.h, I.h16, B, I.eps);
  }
  // W_U h_i of every block row, one pass over the proposal head
  dsp::Gemv8(s, I.head, Impl::kHeadRows, kE, I.h16, I.base);
  Check(hipMemsetAsync(I.state, 0, 256 * 4, s), "rnn state");
  dsp::MarkovArgs ma{};
  ma.w1 = I.w1;
  ma.wj = I.wj;
  ma.bj = I.bj;
  ma.h = I.h;
  ma.f = st.f;
  ma.state = I.state;
  ma.joint = I.joint;
  ma.prev = I.d_prev;
  int done = 0;
  for (int i = 1; i <= n; ++i) {
    ma.row = i;
    dsp::MarkovStep(s, ma);
    dsp::LrLogits(s, I.pt, I.cb,
                  I.base + static_cast<std::size_t>(i) * Impl::kHeadRows,
                  I.joint + 3 * dsp::kRank, I.logits, Impl::kHeadRows);
    if (coupled != nullptr)
      dsp::CoupledSample(s, I.logits, Impl::kHeadRows, I.head_ids,
                         I.target.softcap, coupled->temperature, coupled->min_p,
                         coupled->u[i - 1], I.ws, I.d_tokens, i - 1, I.d_prev,
                         I.d_conf);
    else
      dsp::Argmax(s, I.logits, Impl::kHeadRows, I.head_ids, I.ws, I.d_tokens,
                  i - 1, I.d_prev, I.d_conf);
    done = i;
    if (i < n && more) {  // read this step's probability to decide on the next
      Check(hipMemcpyAsync(I.h_out + 8 + (i - 1), I.d_conf + (i - 1), 4,
                           hipMemcpyDeviceToHost, s),
            "confidence");
      Check(hipStreamSynchronize(s), "propose step");
      if (!more(i, I.h_out + 8))
        break;
    }
  }
  Check(hipMemcpyAsync(I.h_out, I.d_tokens, static_cast<std::size_t>(done) * 4,
                       hipMemcpyDeviceToHost, s),
        "drafts");
  Check(
      hipMemcpyAsync(I.h_out + 8, I.d_conf, static_cast<std::size_t>(done) * 4,
                     hipMemcpyDeviceToHost, s),
      "confidence");
  Check(hipStreamSynchronize(s), "propose sync");
  std::memcpy(out, I.h_out, static_cast<std::size_t>(done) * 4);
  if (confidence != nullptr)
    std::memcpy(confidence, I.h_out + 8, static_cast<std::size_t>(done) * 4);
  return done;
}

}  // namespace gufo::models::inkling2
