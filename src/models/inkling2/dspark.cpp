#include "src/models/inkling2/dspark.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/platform/prefault.hpp>
#include <stdexcept>
#include <string_view>

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
  // [256][kHeadRows] int8, cb = W_U b_o, base = W_U h_i [8][kHeadRows].
  std::int8_t* pt = nullptr;  // int8, one scale per row (ps)
  float* ps = nullptr;
  float *cb = nullptr, *base = nullptr;
  // Proposal head: the target LM head's rows of the kHeadRows tokens with the
  // largest markov_w1 norms (the Loom package's choice), as a Q8_0 copy.
  static constexpr int kHeadRows = 65536;
  void* head = nullptr;
  int* head_ids = nullptr;
  _Float16 *h16, *o16, *a16, *mix16;
  void* qa = nullptr;
  float* qs = nullptr;
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
  /// Streams the drafter's weights to the device without faulting the
  /// mapping (see DeviceWeights::Upload in engine.cpp).
  std::unique_ptr<::gufo::hip::WeightUpload> upload;
  ~Impl() {
    if (upload) {
      std::string ignored;
      (void)upload->Finish(&ignored);  // queued copies target `allocs`
    }
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
    std::string error;
    if (!upload)
      upload =
          ::gufo::hip::WeightUpload::Create(reader->GetMappedRegions(), &error);
    if (!upload || !upload->CopyMapped(t.data, t.size_bytes, p, &error))
      throw std::runtime_error("weight upload: " + error);
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

  fs::GemmItems Items(int* n_items, int* e, int* s, int* c, int rows,
                      int tile = 64) {
    fs::GemmItems it{};
    it.nitems = n_items;
    it.item_e = e;
    it.item_start = s;
    it.item_n = c;
    it.max_items = (rows + tile - 1) / tile;
    it.tile = tile;
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
    // llama.cpp's layout (#32218): per-layer ctx_mix (already softmaxed),
    // ctx_norm, markov_w1 over the tokenizer rows, and markov_w2 plus bias,
    // the Markov output folded through the BF16 unembedding.
    auto u32 = [&](const char* k) { return r.GetMetadataUint32(k); };
    if (u32("dflash.block_count") != 3u ||
        u32("dflash.embedding_length") != 2304u ||
        u32("dflash.feed_forward_length") != 9216u ||
        u32("dflash.attention.head_count") != 24u ||
        u32("dflash.attention.head_count_kv") != 8u ||
        u32("dflash.block_size") != 8u ||
        // llama.cpp's n_swa: the block's 1024 previous positions plus 1
        u32("dflash.attention.sliding_window") != 1025u)
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
      L.ctx_norm = I.F32(b + "ctx_norm.weight", kE, &tot);
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
    {
      // one mask row per draft slot, all copies of the checkpoint's one row
      constexpr int kSlots = dsp::kBlock - 1;
      const auto& m = I.Need("mask_embd.weight", GgmlType::kF32,
                             static_cast<std::uint64_t>(kE) * kSlots);
      const auto* row = static_cast<const char*>(m.data);
      for (int i = 1; i < kSlots; ++i)
        if (std::memcmp(row, row + static_cast<std::size_t>(i) * kE * 4,
                        static_cast<std::size_t>(kE) * 4) != 0)
          throw std::runtime_error("per-slot DSpark mask embeddings");
      void* p = I.Alloc<float>(kE);
      Check(hipMemcpy(p, row, static_cast<std::size_t>(kE) * 4,
                      hipMemcpyHostToDevice),
            "weight upload");
      tot += static_cast<std::size_t>(kE) * 4;
      I.mask = static_cast<const float*>(p);
    }
    I.out_norm = I.F32("output_norm.weight", kE, &tot);
    I.markov_norm = I.F32("markov_ctx_norm.weight", kE, &tot);
    I.bj = I.F32("joint_proj.bias", 768, &tot);
    I.wj = I.BF16("joint_proj.weight", 768ull * 5120, &tot);
    // markov_w1: one row per tokenizer token
    const auto w1_rows = static_cast<std::uint64_t>(target.vocab);
    I.w1 = I.BF16("markov_w1.weight", 256ull * w1_rows, &tot);
    // The kernels below read these weights.
    if (I.upload) {
      std::string upload_error;
      const bool uploaded = I.upload->Finish(&upload_error);
      I.upload.reset();
      if (!uploaded)
        throw std::runtime_error("weight upload: " + upload_error);
    }
    // the converter's softmax of each fuse row: layers 0-2, then the head
    for (int j = 0; j < 4; ++j) {
      const std::string n = j < 3
                                ? "blk." + std::to_string(j) + ".ctx_mix.weight"
                                : std::string("markov_ctx_mix.weight");
      std::memcpy(I.alpha[j], I.Need(n, GgmlType::kF32, 4).data,
                  sizeof(I.alpha[j]));
    }
    d->weight_bytes_ = tot;
    constexpr int B = dsp::kBlock;
    constexpr int S = kMaxSessions;  // ProposeBatch: a slice per session
    I.x = I.Alloc<float>(S * B * kE);
    I.h = I.Alloc<float>(S * B * kE);
    I.h16 = I.Alloc<_Float16>(S * B * kE);
    I.q = I.Alloc<float>(S * B * 3072);
    I.k = I.Alloc<float>(S * B * 1024);
    I.v = I.Alloc<float>(S * B * 1024);
    I.o16 = I.Alloc<_Float16>(S * B * 3072);
    I.y = I.Alloc<float>(S * B * kE);
    I.g = I.Alloc<float>(S * B * 9216);
    I.u = I.Alloc<float>(S * B * 9216);
    I.a16 = I.Alloc<_Float16>(S * B * 9216);
    I.qa = I.Alloc<char>(static_cast<std::size_t>(S) * B * 9216);
    I.qs = I.Alloc<float>(static_cast<std::size_t>(S) * B * (9216 / 32));
    I.logits = I.Alloc<float>(static_cast<std::size_t>(S) * Impl::kHeadRows);
    I.state = I.Alloc<float>(S * 256);
    I.joint = I.Alloc<float>(S * 1024);
    I.ws = I.Alloc<float>(S * (3 * 128 + 1));
    I.d_conf = I.Alloc<float>(S * 8);
    {
      // the kHeadRows text tokens with the largest markov_w1 rows (computed on
      // the device from the already-uploaded I.w1 so the host mapping is not
      // scanned)
      const int V = target.vocab;
      float* d_norms = nullptr;
      Check(hipMalloc(reinterpret_cast<void**>(&d_norms),
                      static_cast<std::size_t>(V) * sizeof(float)),
            "w1 norms");
      dsp::RowNormSqBf16(nullptr, I.w1, V, d_norms);
      std::vector<float> h_norms(static_cast<std::size_t>(V));
      Check(hipMemcpy(h_norms.data(), d_norms,
                      static_cast<std::size_t>(V) * sizeof(float),
                      hipMemcpyDeviceToHost),
            "w1 norms");
      (void)hipFree(d_norms);
      std::vector<std::pair<float, int>> norm(static_cast<std::size_t>(V));
      for (int t = 0; t < V; ++t)
        norm[static_cast<std::size_t>(t)] = {
            h_norms[static_cast<std::size_t>(t)], t};
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
      tot += ids.size() * (2304 / 32) * 34;

      // (W_U W_o)^T and W_U b_o as [256][R] int8 + [R] bias: gather the
      // selected BF16 rows in row-major order and transpose+quantize on device.
      const int R = static_cast<int>(ids.size());
      const auto& w2t = I.Need("markov_w2.weight", GgmlType::kBF16,
                               256ull * static_cast<std::uint64_t>(V));
      const auto& b2t = I.Need("markov_w2.bias", GgmlType::kF32,
                               static_cast<std::uint64_t>(V));
      const auto* w2h = static_cast<const std::uint16_t*>(w2t.data);
      const auto* b2h = static_cast<const float*>(b2t.data);
      std::vector<std::uint16_t> w2_sel(static_cast<std::size_t>(R) * 256);
      std::vector<float> cb_sel(static_cast<std::size_t>(R));
      for (int i = 0; i < R; ++i) {
        const auto t =
            static_cast<std::size_t>(ids[static_cast<std::size_t>(i)]);
        std::memcpy(&w2_sel[static_cast<std::size_t>(i) * 256], w2h + t * 256,
                    256 * sizeof(std::uint16_t));
        cb_sel[static_cast<std::size_t>(i)] = b2h[t];
      }
      std::uint16_t* dw2 = nullptr;
      Check(hipMalloc(reinterpret_cast<void**>(&dw2),
                      w2_sel.size() * sizeof(std::uint16_t)),
            "lr out");
      Check(hipMemcpy(dw2, w2_sel.data(), w2_sel.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice),
            "lr tables");
      I.pt = I.Alloc<std::int8_t>(static_cast<std::size_t>(256) * R);
      I.ps = I.Alloc<float>(static_cast<std::size_t>(R));
      I.cb = I.Alloc<float>(static_cast<std::size_t>(R));
      I.base = I.Alloc<float>(static_cast<std::size_t>(kMaxSessions) *
                              dsp::kBlock * R);
      dsp::QuantPtBf16(nullptr, dw2, R, I.pt, I.ps);
      Check(hipMemcpy(I.cb, cb_sel.data(),
                      static_cast<std::size_t>(R) * sizeof(float),
                      hipMemcpyHostToDevice),
            "lr bias");
      Check(hipDeviceSynchronize(), "lr tables");
      (void)hipFree(dw2);
      tot += static_cast<std::size_t>(256) * R +
             static_cast<std::size_t>(R) * 4 + static_cast<std::size_t>(R) * 4;
    }
    I.wsa = I.Alloc<float>(static_cast<std::size_t>(dsp::kHeads) *
                           dsp::kAttnSplits * B * (128 + 2));
    I.mix = I.Alloc<float>(static_cast<std::size_t>(Impl::kMaxInject) * 4 * kE);
    I.mix16 =
        I.Alloc<_Float16>(static_cast<std::size_t>(Impl::kMaxInject) * 3 * kE);
    I.ikv = I.Alloc<float>(static_cast<std::size_t>(Impl::kMaxInject) * 2048);
    I.d_anchor = I.Alloc<int>(kMaxSessions);
    I.d_pos = I.Alloc<int>(kMaxSessions);
    I.d_prev = I.Alloc<int>(kMaxSessions);
    I.d_tokens = I.Alloc<int>(kMaxSessions * 8);
    I.it_n = I.Alloc<int>(1);
    I.it_e = I.Alloc<int>(Impl::kMaxInject / 64);
    I.it_s = I.Alloc<int>(Impl::kMaxInject / 64);
    I.it_c = I.Alloc<int>(Impl::kMaxInject / 64);
    I.blk_items = I.Alloc<int>(4);
    Check(hipHostMalloc(reinterpret_cast<void**>(&I.h_in),
                        2 * kMaxSessions * sizeof(int)),
          "pinned");
    Check(hipHostMalloc(reinterpret_cast<void**>(&I.h_out),
                        16 * kMaxSessions * sizeof(float)),
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
    // The host reads (mask row, alpha, markov tables) are done: drop the
    // mapping's pages from the process.
    for (const auto& region : I.reader->GetMappedRegions())
      ::gufo::platform::ReleaseMappedPages(region.data, region.size);
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
                     std::size_t pos0, int n,
                     std::span<const InjectCapture> captures) {
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
      it = I.Items(b, b + 1, b + 2, b + 3, m, 64);
    } else {
      const int tile = m >= 256 ? 128 : 64;
      const int ni = (m + tile - 1) / tile;
      // Pageable stack buffer: hipMemcpyAsync stages it before returning, so
      // back-to-back ubatches never overwrite each other's item lists.
      int items[1 + 2 * (Impl::kMaxInject / 64)];
      items[0] = ni;
      for (int i = 0; i < ni; ++i) {
        items[1 + i] = i * tile;
        items[1 + ni + i] = std::min(tile, m - i * tile);
      }
      Check(hipMemcpyAsync(I.it_n, items, 4, hipMemcpyHostToDevice, s),
            "items");
      Check(hipMemcpyAsync(I.it_s, items + 1, ni * 4, hipMemcpyHostToDevice, s),
            "items");
      Check(hipMemcpyAsync(I.it_c, items + 1 + ni, ni * 4,
                           hipMemcpyHostToDevice, s),
            "items");
      it = I.Items(I.it_n, I.it_e, I.it_s, I.it_c, m, tile);
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
    for (const auto& cap : captures) {
      if (cap.dst != nullptr && cap.row >= done && cap.row < done + m)
        Check(
            hipMemcpyAsync(
                cap.dst,
                I.mix + (static_cast<std::size_t>(cap.row - done) * 4 + 3) * kE,
                kE * 4, hipMemcpyDeviceToDevice, s),
            "markov checkpoint");
    }
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

void Drafter::InjectRows(hipStream_t s, std::span<const Rows> rows) {
  Impl& I = *impl_;
  int m = 0;
  for (const auto& r : rows)
    m += std::max(r.n, 0);
  if (m == 0)
    return;
  if (m > kMaxRows) {
    for (const auto& r : rows)
      Inject(s, *r.st, r.feats, r.pos0, r.n);
    return;
  }
  // mix of each request's rows into rows [off, off + n) of the step;
  // F16 layers `m` rows apart, the GEMV operand
  int off = 0;
  for (const auto& r : rows) {
    if (r.n <= 0)
      continue;
    dsp::MixArgs ma{};
    ma.feat = r.feats;
    std::memcpy(ma.alpha, I.alpha, sizeof ma.alpha);
    for (int l = 0; l < 3; ++l)
      ma.w[l] = I.layer[l].ctx_norm;
    ma.w[3] = I.markov_norm;
    ma.out = I.mix + static_cast<std::size_t>(off) * 4 * kE;
    ma.out16 = I.mix16 + static_cast<std::size_t>(off) * kE;
    ma.n = r.n;
    ma.ld16 = m;
    ma.eps = I.eps;
    dsp::Mix(s, ma);
    off += r.n;
  }
  float* kraw = I.ikv;
  float* vraw = I.ikv + static_cast<std::size_t>(m) * 1024;
  for (int l = 0; l < 3; ++l) {
    const _Float16* a = I.mix16 + static_cast<std::size_t>(l) * m * kE;
    {
      const void* w[2] = {I.layer[l].wk_ctx, I.layer[l].wv_ctx};
      const int mm[2] = {1024, 1024};
      float* y[2] = {kraw, vraw};
      dsp::Gemv8N(s, 2, w, mm, kE, a, y, m, I.qa, I.qs);
    }
    off = 0;
    for (const auto& r : rows) {
      if (r.n <= 0)
        continue;
      State& st = *r.st;
      dsp::KvStoreArgs ka{};
      ka.k = kraw + static_cast<std::size_t>(off) * 1024;
      ka.v = vraw + static_cast<std::size_t>(off) * 1024;
      ka.ldk = ka.ldv = 1024;
      ka.knorm = I.layer[l].k_ctx_norm;
      ka.kc = static_cast<_Float16*>(st.k[l]);
      ka.vc = static_cast<_Float16*>(st.v[l]);
      ka.cap = static_cast<int>(st.cap[l]);
      ka.pos0 = static_cast<int>(r.pos0);
      ka.n = r.n;
      ka.theta = I.theta;
      ka.eps = I.eps;
      dsp::KvStore(s, ka);
      off += r.n;
    }
  }
  off = 0;
  for (const auto& r : rows) {
    if (r.n <= 0)
      continue;
    State& st = *r.st;
    if (r.pos0 > st.valid || r.pos0 < st.first)
      st.first = r.pos0;  // a gap: the context before pos0 is missing
    // the Markov context of the request's last row
    Check(hipMemcpyAsync(
              st.f,
              I.mix + (static_cast<std::size_t>(off + r.n - 1) * 4 + 3) * kE,
              kE * 4, hipMemcpyDeviceToDevice, s),
          "markov context");
    st.f_pos = r.pos0 + static_cast<std::size_t>(r.n) - 1;
    st.valid = r.pos0 + static_cast<std::size_t>(r.n);
    off += r.n;
  }
}

int Drafter::Propose(hipStream_t s, State& st, std::int32_t anchor,
                     std::size_t pos, int n, std::int32_t* out,
                     float* confidence, const More& more,
                     const Coupled* coupled) {
  Request r{.st = &st,
            .anchor = anchor,
            .pos = pos,
            .n = n,
            .out = out,
            .confidence = confidence,
            .more = more ? &more : nullptr,
            .coupled = coupled};
  ProposeBatch(s, std::span<Request>(&r, 1));
  return r.done;
}

void Drafter::ProposeBatch(hipStream_t s, std::span<Request> req) {
  Impl& I = *impl_;
  const int S = static_cast<int>(req.size());
  if (S < 1 || S > kMaxSessions)
    throw std::runtime_error("inkling2 dspark: bad proposal batch");
  constexpr int B = dsp::kBlock;
  constexpr std::size_t R = Impl::kHeadRows;
  int nmax = 0;
  for (int g = 0; g < S; ++g) {
    req[g].n = std::clamp(req[g].n, 0, kMaxDraft);
    req[g].done = 0;
    nmax = std::max(nmax, req[g].n);
    I.h_in[g] = req[g].anchor;
    I.h_in[kMaxSessions + g] = static_cast<int>(req[g].pos);
  }
  if (nmax == 0)
    return;
  Check(hipMemcpyAsync(I.d_anchor, I.h_in, 4 * S, hipMemcpyHostToDevice, s),
        "anchor");
  Check(hipMemcpyAsync(I.d_pos, I.h_in + kMaxSessions, 4 * S,
                       hipMemcpyHostToDevice, s),
        "pos");
  Check(hipMemcpyAsync(I.d_prev, I.h_in, 4 * S, hipMemcpyHostToDevice, s),
        "prev");
  // The block rows of session g are rows [8g, 8g + 8) of every buffer; the
  // weight GEMVs read each weight row once for all sessions (their blocks run
  // next to each other), everything else runs per session.
  for (int g = 0; g < S; ++g)
    dsp::Embed(s, I.target.token_embd, I.target.token_embd_norm, I.d_anchor + g,
               I.mask, I.layer[0].attn_norm,
               I.x + static_cast<std::size_t>(g) * B * kE,
               I.h + static_cast<std::size_t>(g) * B * kE,
               I.h16 + static_cast<std::size_t>(g) * B * kE, I.eps);
  const int nrows = B * S;
  for (int l = 0; l < 3; ++l) {
    const auto& L = I.layer[l];
    {
      const void* w[3] = {L.wq, L.wk, L.wv};
      const int m[3] = {3072, 1024, 1024};
      float* y[3] = {I.q, I.k, I.v};
      dsp::Gemv8N(s, 3, w, m, kE, I.h16, y, nrows, I.qa, I.qs);
    }
    for (int g = 0; g < S; ++g) {
      State& st = *req[g].st;
      float* q = I.q + static_cast<std::size_t>(g) * B * 3072;
      float* k = I.k + static_cast<std::size_t>(g) * B * 1024;
      float* v = I.v + static_cast<std::size_t>(g) * B * 1024;
      dsp::QkNormRope(s, q, k, L.q_norm, L.k_norm, I.d_pos + g, I.theta, I.eps);
      dsp::AttnArgs aa{};
      aa.q = q;
      aa.k = k;
      aa.v = v;
      aa.kc = static_cast<const _Float16*>(st.k[l]);
      aa.vc = static_cast<const _Float16*>(st.v[l]);
      aa.cap = static_cast<int>(st.cap[l]);
      aa.pos = I.d_pos + g;
      aa.window = WindowLayer(l) ? static_cast<int>(kWindow) : 0;
      aa.first = static_cast<int>(st.first);
      aa.ws = I.wsa;
      aa.out16 = I.o16 + static_cast<std::size_t>(g) * B * 3072;
      const int pos_i = static_cast<int>(req[g].pos);
      const int lo =
          std::max(aa.first, aa.window ? std::max(0, pos_i - aa.window) : 0);
      const int nkeys = std::max(0, pos_i - lo) + B;
      aa.nsplits = std::clamp((nkeys + 127) / 128, 2, dsp::kAttnSplits);
      dsp::Attn(s, aa);
    }
    dsp::Gemv8(s, L.wo, kE, 3072, I.o16, I.y, nrows, I.qa, I.qs);
    dsp::AddNorm(s, I.x, I.y, L.ffn_norm, I.h, I.h16, nrows, I.eps);
    {
      const void* w[2] = {L.wg, L.wu};
      const int m[2] = {9216, 9216};
      float* y[2] = {I.g, I.u};
      dsp::Gemv8N(s, 2, w, m, kE, I.h16, y, nrows, I.qa, I.qs);
    }
    dsp::Swiglu(s, I.g, I.u, I.a16, nrows * 9216);
    dsp::Gemv8(s, L.wd, kE, 9216, I.a16, I.y, nrows, I.qa, I.qs);
    dsp::AddNorm(s, I.x, I.y, l < 2 ? I.layer[l + 1].attn_norm : I.out_norm,
                 I.h, I.h16, nrows, I.eps);
  }
  // W_U h_i of every block row, one pass over the proposal head
  dsp::Gemv8(s, I.head, static_cast<int>(R), kE, I.h16, I.base, nrows, I.qa,
             I.qs);
  Check(hipMemsetAsync(I.state, 0, static_cast<std::size_t>(S) * 256 * 4, s),
        "rnn state");
  std::array<bool, kMaxSessions> live{};
  for (int g = 0; g < S; ++g)
    live[g] = req[g].n > 0;
  for (int i = 1; i <= nmax; ++i) {
    bool any_more = false;
    dsp::LrBatch lrb{};
    lrb.count = 0;
    for (int g = 0; g < S; ++g) {
      if (!live[g])
        continue;
      Request& rq = req[g];
      dsp::MarkovArgs ma{};
      ma.w1 = I.w1;
      ma.wj = I.wj;
      ma.bj = I.bj;
      ma.h = I.h + static_cast<std::size_t>(g) * B * kE;
      ma.f = rq.st->f;
      ma.state = I.state + g * 256;
      ma.joint = I.joint + g * 1024;
      ma.prev = I.d_prev + g;
      ma.row = i;
      dsp::MarkovStep(s, ma);
      lrb.base[lrb.count] = I.base + (static_cast<std::size_t>(g) * B + i) * R;
      lrb.lat[lrb.count] = ma.joint + 3 * dsp::kRank;
      lrb.out[lrb.count] = I.logits + static_cast<std::size_t>(g) * R;
      ++lrb.count;
    }
    if (lrb.count > 0)
      dsp::LrLogits8Batch(s, I.pt, I.ps, I.cb, lrb, static_cast<int>(R));
    for (int g = 0; g < S; ++g) {
      if (!live[g])
        continue;
      Request& rq = req[g];
      float* logits = I.logits + static_cast<std::size_t>(g) * R;
      float* ws = I.ws + g * (3 * 128 + 1);
      if (rq.coupled != nullptr)
        dsp::CoupledSample(
            s, logits, static_cast<int>(R), I.head_ids, I.target.softcap,
            rq.coupled->temperature, rq.coupled->min_p, rq.coupled->u[i - 1],
            ws, I.d_tokens + g * 8, i - 1, I.d_prev + g, I.d_conf + g * 8);
      else
        dsp::Argmax(s, logits, static_cast<int>(R), I.head_ids, ws,
                    I.d_tokens + g * 8, i - 1, I.d_prev + g, I.d_conf + g * 8);
      rq.done = i;
      if (i >= rq.n)
        live[g] = false;
      else if (rq.more != nullptr)
        any_more = true;
    }
    if (any_more) {  // read this step's probabilities to decide on the next
      for (int g = 0; g < S; ++g)
        if (live[g] && req[g].more != nullptr)
          Check(hipMemcpyAsync(I.h_out + 16 * g + 8, I.d_conf + g * 8,
                               static_cast<std::size_t>(i) * 4,
                               hipMemcpyDeviceToHost, s),
                "confidence");
      Check(hipStreamSynchronize(s), "propose step");
      for (int g = 0; g < S; ++g)
        if (live[g] && req[g].more != nullptr &&
            !(*req[g].more)(i, I.h_out + 16 * g + 8))
          live[g] = false;
    }
    if (std::none_of(live.begin(), live.begin() + S, [](bool b) { return b; }))
      break;
  }
  for (int g = 0; g < S; ++g) {
    const std::size_t d = static_cast<std::size_t>(req[g].done);
    Check(hipMemcpyAsync(I.h_out + 16 * g, I.d_tokens + g * 8, d * 4,
                         hipMemcpyDeviceToHost, s),
          "drafts");
    Check(hipMemcpyAsync(I.h_out + 16 * g + 8, I.d_conf + g * 8, d * 4,
                         hipMemcpyDeviceToHost, s),
          "confidence");
  }
  Check(hipStreamSynchronize(s), "propose sync");
  for (int g = 0; g < S; ++g) {
    const std::size_t d = static_cast<std::size_t>(req[g].done);
    std::memcpy(req[g].out, I.h_out + 16 * g, d * 4);
    if (req[g].confidence != nullptr)
      std::memcpy(req[g].confidence, I.h_out + 16 * g + 8, d * 4);
  }
}

}  // namespace gufo::models::inkling2
