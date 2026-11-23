#include "src/models/inkling2/engine.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/experimental/state_memory.hpp>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "src/models/inkling2/hip/fast.hpp"
#include "src/models/inkling2/hip/kernels.hpp"

namespace gufo::models::inkling2 {
namespace {

namespace kk = hip;
namespace fs = fast;

constexpr int kEmbd = 2304;
constexpr int kQ = 3072;   // 24 heads x 128
constexpr int kKV = 1024;  // 8 heads x 128
constexpr int kRel = 384;  // 24 heads x 16
constexpr int kConvChannels = 1024 + 1024 + 2304 + 2304;
constexpr int kConvOffK = 0, kConvOffV = 1024, kConvOffAttn = 2048,
              kConvOffFfn = 2048 + 2304;
constexpr int kLogitChunk = 64;

}  // namespace

std::shared_mutex& CaptureGate() {
  static std::shared_mutex m;
  return m;
}

namespace {

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string("inkling2: ") + what + ": " +
                             hipGetErrorString(e));
}

void* AllocDevice(std::size_t bytes, const char* what) {
  void* p = nullptr;
  hipError_t e = hipMalloc(&p, bytes);
  if (e == hipErrorOutOfMemory) {
    (void)hipGetLastError();
    std::fprintf(stderr,
                 "inkling2: hipMalloc of %s (%zu bytes) ran out of device "
                 "memory; falling back to managed memory\n",
                 what, bytes);
    e = hipMallocManaged(&p, bytes, hipMemAttachGlobal);
    if (e == hipSuccess)
      (void)hipMemAdvise(p, bytes, hipMemAdviseSetCoarseGrain, 0);
    (void)hipGetLastError();
  }
  if (e != hipSuccess || p == nullptr)
    throw std::runtime_error(std::string("inkling2: cannot allocate ") + what +
                             " (" + std::to_string(bytes) +
                             " bytes): " + hipGetErrorString(e));
  return p;
}

/// Device copies of every tensor of the GGUF, found again from the host pointer
/// of a bound TensorRef.
class DeviceWeights {
public:
  ~DeviceWeights() {
    for (auto& r : regions_)
      (void)hipFree(r.dev);
  }
  void Upload(const core::GgufReader& reader, const Inkling2Weights& weights,
              std::size_t* total) {
    std::size_t sum = 0;
    for (const auto& v : weights.widened) {  // F32 copies of BF16 small tensors
                                             // (not in the file mapping)
      const std::size_t bytes = v->size() * sizeof(float);
      void* dev = AllocDevice(bytes, "widened weights");
      Check(hipMemcpy(dev, v->data(), bytes, hipMemcpyHostToDevice),
            "weight upload");
      regions_.push_back(
          {reinterpret_cast<std::uintptr_t>(v->data()), bytes, dev});
      sum += bytes;
    }
    for (const auto& t : reader.GetTensors()) {
      if (t.size_bytes == 0)
        continue;
      void* dev = AllocDevice(t.size_bytes, std::string(t.name).c_str());
      constexpr std::size_t kChunk = std::size_t{64} << 20;
      for (std::size_t off = 0; off < t.size_bytes; off += kChunk) {
        const std::size_t n = std::min(kChunk, t.size_bytes - off);
        Check(hipMemcpy(static_cast<char*>(dev) + off,
                        static_cast<const char*>(t.data) + off, n,
                        hipMemcpyHostToDevice),
              "weight upload");
      }
      regions_.push_back(
          {reinterpret_cast<std::uintptr_t>(t.data), t.size_bytes, dev});
      sum += t.size_bytes;
    }
    std::sort(regions_.begin(), regions_.end(),
              [](const Region& a, const Region& b) { return a.base < b.base; });
    *total = sum;
  }
  [[nodiscard]] const void* Ptr(const TensorRef& t) const {
    const auto a = reinterpret_cast<std::uintptr_t>(t.data);
    auto it = std::upper_bound(
        regions_.begin(), regions_.end(), a,
        [](std::uintptr_t v, const Region& r) { return v < r.base; });
    if (it == regions_.begin())
      throw std::runtime_error("inkling2: tensor not uploaded");
    --it;
    if (a + t.size_bytes > it->base + it->size)
      throw std::runtime_error("inkling2: tensor not uploaded");
    return static_cast<const char*>(it->dev) + (a - it->base);
  }

private:
  struct Region {
    std::uintptr_t base;
    std::size_t size;
    void* dev;
  };
  std::vector<Region> regions_;
};

struct In {
  const float* p;
  int ld;
};

}  // namespace

struct Engine::Impl {
  Engine* eng = nullptr;
  const Inkling2Config* cfg = nullptr;
  const Inkling2Weights* w = nullptr;
  hipStream_t s = nullptr;
  DeviceWeights dw;
  std::vector<void*> scratch_allocs;
  std::array<float*, 24> msgq{};
  float* msgq_all = nullptr;

  int U = 512;
  int T = 0;     // tokens of the running ubatch
  int pos0 = 0;  // position of the first token of the running ubatch

  // activations (all [U][dim] F32 unless noted)
  float *cur, *prelude, *bankout, *tau;
  float* blocks[6];
  float* bank[4];
  float *mixed, *ctl[3], *in[3], *msgs, *gbuf, *gate64, *gainb, *forgetb;
  float *Qb, *Kb, *Vb, *Kc, *Vc, *Rb, *Gb, *Sb, *Ob, *AOp, *AO, *h, *hn, *mixf,
      *rlog, *wts;
  float *dynK, *dynV, *gu, *act, *ye, *moe;
  int* ids_d;
  int* tok_d;
  int* off_d;
  int* sorted_d;
  float* logits_d;
  int nblocks = 0;

  // ---- decode (T = 1) state: separate small buffers, a device slot table and
  // HIP graphs ----
  struct LayerFast {
    bool proj = false;   // q, gate, rel, k, v are Q8_0 with K = 2304
    bool oproj = false;  // attn_output Q8_0, K = 3072
    bool tiny = false;   // sink / conv projections are F32 / F16 / BF16
    int moe =
        -1;  // fast MoE format (fs::kFmtQ8 / kFmtNvfp4) or -1 (generic kernels)
    void* bt = nullptr;     // f16 [1024][16] relative bias table
    bool tiny_pad = false;  // F32 copies (padded to 64 rows) of sink / conv
                            // projections exist
    float *sink_f = nullptr, *kconv_f = nullptr, *vconv_f = nullptr;
  };
  std::array<LayerFast, 24> lf{};
  bool head_fast = false;
  hipStream_t ds = nullptr;
  // prefill scratch (f16 activations, expert grouping, dense work items)
  void *p_iq16, *p_ik16, *p_iv16, *p_ob16, *p_h16, *p_act16, *p_qn16;
  float* p_rn;
  fs::MoeWs mws{};
  int *dn_nitems, *dn_ie, *dn_is, *dn_in;
  std::vector<int> dn_host;  // kept alive for the async copy
  int moe_max_items = 0;
  // F16 copies of the F32 loop-controller and router weights for the decode
  // GEMVs (half the bytes; INKLING2_CTL_F32=1 keeps F32). Prefill keeps using
  // the F32 originals.
  bool ctl_half = true;
  float* d_moe_part = nullptr;
  std::map<const void*, void*> half_of;
  const void* HW(const TensorRef& t) const {
    if (!ctl_half)
      return dw.Ptr(t);
    return half_of.at(t.data);
  }
  /// Whether every value of an F32 tensor is a BF16 value (the checkpoint's own
  /// dtype).
  static bool Bf16Exact(const TensorRef& t) {
    if (static_cast<int>(t.type) != kk::kTF32)
      return false;
    const auto* src = static_cast<const std::uint32_t*>(t.data);
    for (std::size_t i = 0; i < t.size_bytes / 4; ++i)
      if ((src[i] & 0xFFFFu) != 0)
        return false;
    return true;
  }
  /// BF16 copy of a BF16-exact F32 tensor: the same values in half the bytes.
  void MakeHalf(const TensorRef& t) {
    if (t.size_bytes == 0 || half_of.count(t.data))
      return;
    const std::size_t n = t.size_bytes / 4;
    std::vector<std::uint16_t> h(n);
    const auto* src = static_cast<const std::uint32_t*>(t.data);
    for (std::size_t i = 0; i < n; ++i)
      h[i] = static_cast<std::uint16_t>(src[i] >> 16);
    void* d = AllocDevice(n * 2, "f16 weights");
    scratch_allocs.push_back(d);
    Check(hipMemcpy(d, h.data(), n * 2, hipMemcpyHostToDevice), "f16 weights");
    half_of[t.data] = d;
  }
  std::map<int, hipGraphExec_t> graphs;
  std::set<int> failed_graphs;  // keys whose capture failed: those steps run
                                // eagerly, no retry
  float *d_cur, *d_prelude, *d_bankout, *d_msgs, *d_mixed, *d_ctl, *d_in,
      *d_g64, *d_proj, *d_sink, *d_dyn, *d_ob, *d_ao, *d_h, *d_mixf, *d_hn,
      *d_moe, *d_rlog, *d_wts8, *d_act8, *d_ws_o, *d_ws_m, *d_ws_l;
  static constexpr int kMaxB = 4;  // sessions per decode batch
  static constexpr int kWideTileTokens =
      256;  // ubatches at least this long use 128-token GEMM tiles
  // token strides (floats) of the decode buffers
  static constexpr int kSIn = 3 * kEmbd;   // d_in, d_mixed
  static constexpr int kSCtl = 4 * kEmbd;  // d_ctl, d_msgs
  static constexpr int kSProj = 8576;      // d_proj
  static constexpr std::size_t kMaxSplit = 256;
  int nb = 1;  // tokens of the decode step being built
  float* d_blk[16];
  int nblk_alloc = 0;
  int* d_ids8;
  int* d_step;  // {token, position}
  unsigned* d_counter;
  fs::SlotEntry* d_tbl;
  const float* d_states[16];
  int d_ns = 0;
  // ---- M2 hook: layered prefill (spill/restore of the transient activations
  // around a yield) ----
  std::span<const InputRows>
      input_rows;  // M5 hook: set by Forward (under the engine mutex)
  struct YieldRun {
    const PrefillYield* spec = nullptr;
    std::unique_lock<std::mutex>* lock = nullptr;
    std::size_t total =
        0;  // visit units of the whole Forward call (48 per ubatch)
    std::size_t done = 0;
    std::size_t k = 1;  // next group boundary index
  };
  // Live between two visits of a ubatch: cur, prelude, bankout, blocks[6],
  // bank[4], tau.
  float* spill_buf[13] = {};
  float* spill_tau = nullptr;
  float** LiveBufs(float** out) {
    int n = 0;
    out[n++] = cur;
    out[n++] = prelude;
    out[n++] = bankout;
    for (auto* b : blocks)
      out[n++] = b;
    for (auto* b : bank)
      out[n++] = b;
    return out;
  }
  void Spill() {
    float* live[13];
    LiveBufs(live);
    const std::size_t rows = static_cast<std::size_t>(T) * kEmbd * 4;
    for (int i = 0; i < 13; ++i)
      Check(hipMemcpyAsync(spill_buf[i], live[i], rows, hipMemcpyDeviceToDevice,
                           s),
            "spill");
    Check(hipMemcpyAsync(spill_tau, tau, static_cast<std::size_t>(T) * 4,
                         hipMemcpyDeviceToDevice, s),
          "spill");
  }
  void Unspill() {
    float* live[13];
    LiveBufs(live);
    const std::size_t rows = static_cast<std::size_t>(T) * kEmbd * 4;
    for (int i = 0; i < 13; ++i)
      Check(hipMemcpyAsync(live[i], spill_buf[i], rows, hipMemcpyDeviceToDevice,
                           s),
            "unspill");
    Check(hipMemcpyAsync(tau, spill_tau, static_cast<std::size_t>(T) * 4,
                         hipMemcpyDeviceToDevice, s),
          "unspill");
  }
  /// Called after every visit unit; yields at the group boundaries. `mid`:
  /// transient activations are live (inside a ubatch), so they are spilled
  /// around the yield.
  void MaybeYield(YieldRun* y, bool mid) {
    if (y == nullptr || y->spec == nullptr)
      return;
    ++y->done;
    const std::size_t g = y->spec->groups;
    if (y->done >= y->total || y->k >= g)
      return;
    if (y->done < (y->k * y->total + g - 1) / g)
      return;
    while (y->k < g && (y->k * y->total + g - 1) / g <= y->done)
      ++y->k;
    const int saved_T = T, saved_pos0 = pos0, saved_nblocks = nblocks;
    if (mid)
      Spill();
    // The decode graphs run on another (non-blocking) stream and share scratch
    // (expert routing, logits) with this stream: drain it before the lock is
    // released.
    Check(hipStreamSynchronize(s), "yield sync");
    y->lock->unlock();  // the decode round of other sessions takes the engine
                        // mutex
    try {
      y->spec->yield();
    } catch (...) {
      y->lock->lock();
      throw;
    }
    y->lock->lock();
    T = saved_T;
    pos0 = saved_pos0;
    nblocks = saved_nblocks;
    if (mid) {
      Unspill();
      // The dense-GEMM work items are per ubatch and shared: another session's
      // Forward replaced them.
      UploadDenseItems();
    }
  }
  // ---- end M2 hook ----

  float* NewBuf(std::size_t floats) {
    auto* p = static_cast<float*>(AllocDevice(floats * 4, "scratch"));
    scratch_allocs.push_back(p);
    return p;
  }
  void* NewRaw(std::size_t bytes) {
    void* p = AllocDevice(bytes, "scratch");
    scratch_allocs.push_back(p);
    return p;
  }
  int* NewInts(std::size_t n) {
    auto* p = static_cast<int*>(AllocDevice(n * 4, "scratch"));
    scratch_allocs.push_back(p);
    return p;
  }

  ~Impl() {
    for (auto& g : graphs)
      (void)hipGraphExecDestroy(g.second);
    if (ds)
      (void)hipStreamDestroy(ds);
    for (void* p : scratch_allocs)
      (void)hipFree(p);
  }

  void Init(Engine* e) {
    eng = e;
    cfg = &e->weights_.config;
    w = &e->weights_;
    U = static_cast<int>(e->ubatch_);
    const std::size_t u = static_cast<std::size_t>(U);
    cur = NewBuf(u * kEmbd);
    prelude = NewBuf(u * kEmbd);
    bankout = NewBuf(u * kEmbd);
    tau = NewBuf(u);
    for (auto& b : spill_buf)
      b = NewBuf(u * kEmbd);  // M2 hook: layered prefill spill area
    spill_tau = NewBuf(u);
    for (auto& b : blocks)
      b = NewBuf(u * kEmbd);
    for (auto& b : bank)
      b = NewBuf(u * kEmbd);
    mixed = NewBuf(u * 3 * kEmbd);
    for (auto& b : ctl)
      b = NewBuf(u * kEmbd);
    for (auto& b : in)
      b = NewBuf(u * kEmbd);
    msgs = NewBuf(u * 4 * kEmbd);
    gbuf = NewBuf(u * 2 * kEmbd);
    gate64 = NewBuf(u * 64 * 3);
    gainb = NewBuf(u * kEmbd);
    forgetb = NewBuf(u * kEmbd);
    Qb = NewBuf(u * kQ);
    Kb = NewBuf(u * kKV);
    Vb = NewBuf(u * kKV);
    Kc = NewBuf(u * kKV);
    Vc = NewBuf(u * kKV);
    Rb = NewBuf(u * kRel);
    Gb = NewBuf(u * kQ);
    Sb = NewBuf(u * 24);
    Ob = NewBuf(u * kQ);
    AOp = NewBuf(u * kEmbd);
    AO = NewBuf(u * kEmbd);
    h = NewBuf(u * kEmbd);
    hn = NewBuf(u * kEmbd);
    mixf = NewBuf(u * kEmbd);
    rlog = NewBuf(u * 128);
    wts = NewBuf(u * 8);
    dynK = NewBuf(u * 32);
    dynV = NewBuf(u * 32);
    gu = NewBuf(u * 8 * 2 * 1152);
    act = NewBuf(u * 8 * 1152);
    ye = NewBuf(u * 8 * kEmbd);
    moe = NewBuf(u * kEmbd);
    ids_d = NewInts(u * 8);
    tok_d = NewInts(u);
    off_d = NewInts(129);
    sorted_d = NewInts(u * 8);
    logits_d =
        NewBuf(static_cast<std::size_t>(kLogitChunk) * cfg->n_vocab_text);
    // message queries of the core layers, [4][2304] each, one core layer after
    // the other (a decode pass mixes all of them in one launch)
    msgq_all = NewBuf(static_cast<std::size_t>(cfg->CoreLen()) * 4 * kEmbd);
    for (std::uint32_t il = cfg->core_start; il < cfg->core_end; ++il) {
      float* q =
          msgq_all + static_cast<std::size_t>(il - cfg->core_start) * 4 * kEmbd;
      msgq[il] = q;
      for (int st = 0; st < kStreams; ++st)
        Check(hipMemcpy(q + st * kEmbd, dw.Ptr(w->layers[il].msg_query[st]),
                        kEmbd * 4, hipMemcpyDeviceToDevice),
              "msg query copy");
    }
    InitDecode();
  }

  static bool IsType(const TensorRef& t, int ty) {
    return static_cast<int>(t.type) == ty;
  }
  static bool SmallType(const TensorRef& t) {
    const int ty = static_cast<int>(t.type);
    return ty == kk::kTF32 || ty == kk::kTF16 || ty == kk::kTBF16;
  }

  void InitDecode() {
    {
      const std::size_t u = static_cast<std::size_t>(U);
      p_iq16 = NewRaw(u * kEmbd * 2);
      p_ik16 = NewRaw(u * kEmbd * 2);
      p_iv16 = NewRaw(u * kEmbd * 2);
      p_ob16 = NewRaw(u * kQ * 2);
      p_h16 = NewRaw(u * kEmbd * 2);
      p_act16 = NewRaw(u * 8 * 1152 * 2);
      p_qn16 = NewRaw(u * kQ * 2);
      p_rn = NewBuf(u * kRel);
      moe_max_items = static_cast<int>(u * 8 / 32 + 128 + 8);
      mws.cnt = NewInts(128);
      mws.goff = NewInts(129);
      mws.fill = NewInts(128);
      mws.nitems = NewInts(1);
      mws.item_e = NewInts(moe_max_items);
      mws.item_start = NewInts(moe_max_items);
      mws.item_n = NewInts(moe_max_items);
      mws.postok = NewInts(u * 8);
      mws.pair = NewInts(u * 8);
      mws.posw = NewBuf(u * 8);
      dn_nitems = NewInts(1);
      dn_ie = NewInts(u / 64 + 2);
      dn_is = NewInts(u / 64 + 2);
      dn_in = NewInts(u / 64 + 2);
    }
    // Decode reads the router and loop-controller weights as BF16 when that
    // loses nothing (checkpoints store them in BF16; the GGUF widens them to
    // F32), otherwise as F32.
    for (std::uint32_t il = 0; il < 24 && ctl_half; ++il) {
      const LayerWeights& L = w->layers[il];
      if (!Bf16Exact(L.ffn_gate_inp))
        ctl_half = false;
      for (const auto& c : L.loop)
        if ((c.down.size_bytes && !Bf16Exact(c.down)) ||
            (c.gain.size_bytes && !Bf16Exact(c.gain)) ||
            (c.forget.size_bytes && !Bf16Exact(c.forget)))
          ctl_half = false;
    }
    if (ctl_half) {
      for (std::uint32_t il = 0; il < 24; ++il) {
        const LayerWeights& L = w->layers[il];
        MakeHalf(L.ffn_gate_inp);
        for (const auto& c : L.loop) {
          MakeHalf(c.down);
          MakeHalf(c.gain);
          MakeHalf(c.forget);
        }
      }
    }
    Check(hipStreamCreateWithFlags(&ds, hipStreamNonBlocking), "decode stream");
    // Decode buffers are kMaxB tokens wide ([token][...]); a step of nb tokens
    // uses the first nb rows.
    constexpr std::size_t B = kMaxB;
    d_cur = NewBuf(B * kEmbd);
    d_prelude = NewBuf(B * kEmbd);
    d_bankout = NewBuf(B * kEmbd);
    d_msgs = NewBuf(B * kSCtl * cfg->CoreLen());
    d_mixed = NewBuf(B * kSIn);
    d_ctl = NewBuf(B * kSCtl);
    d_in = NewBuf(B * kSIn);
    d_g64 = NewBuf(B * 4 * 64);
    d_proj = NewBuf(B * kSProj);
    d_sink = NewBuf(B * 24);
    d_dyn = NewBuf(B * 64);
    d_ob = NewBuf(B * kQ);
    d_ao = NewBuf(B * kEmbd);
    d_h = NewBuf(B * kEmbd);
    d_mixf = NewBuf(B * kEmbd);
    d_hn = NewBuf(B * kEmbd);
    d_moe = NewBuf(B * kEmbd);
    d_rlog = NewBuf(B * 128);
    d_wts8 = NewBuf(B * 8);
    d_act8 = NewBuf(B * 8 * 1152);
    d_ws_o = NewBuf(B * kMaxSplit * 8 * 12 * 128);
    d_ws_m = NewBuf(B * kMaxSplit * 8 * 12);
    d_ws_l = NewBuf(B * kMaxSplit * 8 * 12);
    for (auto& b : d_blk)
      b = NewBuf(B * kEmbd);
    d_moe_part = NewBuf(B * 8 * kEmbd);
    d_ids8 = NewInts(B * 8);
    d_step = NewInts(2 * B);  // tokens [B], positions [B]
    d_counter = reinterpret_cast<unsigned*>(NewInts(B));
    Check(hipMemset(d_counter, 0, B * 4), "counter");
    d_tbl = static_cast<fs::SlotEntry*>(
        AllocDevice(B * 48 * sizeof(fs::SlotEntry), "slot table"));
    scratch_allocs.push_back(d_tbl);
    const int ff = static_cast<int>(cfg->n_ff_expert);
    for (std::uint32_t il = 0; il < 24; ++il) {
      const LayerWeights& L = w->layers[il];
      LayerFast& f = lf[il];
      f.proj = IsType(L.attn_q, kk::kTQ8_0) && IsType(L.attn_k, kk::kTQ8_0) &&
               IsType(L.attn_v, kk::kTQ8_0) &&
               IsType(L.attn_gate, kk::kTQ8_0) &&
               IsType(L.attn_rel, kk::kTQ8_0);
      f.oproj = IsType(L.attn_output, kk::kTQ8_0);
      f.tiny = SmallType(L.attn_sink_proj) &&
               (!L.is_global || (SmallType(L.attn_k_conv_proj) &&
                                 SmallType(L.attn_v_conv_proj)));
      const int tg = static_cast<int>(L.ffn_gate_up_exps.type),
                td = static_cast<int>(L.ffn_down_exps.type);
      if (tg == td && kEmbd == 2304 && ff == 1152) {
        if (tg == kk::kTQ8_0)
          f.moe = fs::kFmtQ8;
        if (tg == kk::kTNVFP4)
          f.moe = fs::kFmtNvfp4;
      }
      // relative bias table: F32 [1024][16] -> f16
      std::vector<__half> bt(1024 * 16);
      const float* src = static_cast<const float*>(L.attn_rel_bias.data);
      for (int i = 0; i < 1024 * 16; ++i)
        bt[i] = __float2half(src[i]);
      f.bt = AllocDevice(bt.size() * 2, "bias table");
      scratch_allocs.push_back(f.bt);
      Check(hipMemcpy(f.bt, bt.data(), bt.size() * 2, hipMemcpyHostToDevice),
            "bias table");
    }
    for (std::uint32_t il = 0; il < 24; ++il) {
      const LayerWeights& L = w->layers[il];
      LayerFast& f = lf[il];
      if (!f.tiny)
        continue;
      auto make = [&](const TensorRef& t, int rows) {
        std::vector<float> host(64 * 2304, 0.f);
        const int ty = static_cast<int>(t.type);
        for (int i = 0; i < rows * 2304; ++i) {
          if (ty == kk::kTF32)
            host[i] = static_cast<const float*>(t.data)[i];
          else if (ty == kk::kTF16)
            host[i] = __half2float(static_cast<const __half*>(t.data)[i]);
          else {
            const std::uint32_t bits =
                static_cast<std::uint32_t>(
                    static_cast<const std::uint16_t*>(t.data)[i])
                << 16;
            std::memcpy(&host[i], &bits, 4);
          }
        }
        auto* d =
            static_cast<float*>(AllocDevice(host.size() * 4, "padded weights"));
        scratch_allocs.push_back(d);
        Check(hipMemcpy(d, host.data(), host.size() * 4, hipMemcpyHostToDevice),
              "padded weights");
        return d;
      };
      f.sink_f = make(L.attn_sink_proj, 24);
      if (L.is_global) {
        f.kconv_f = make(L.attn_k_conv_proj, 32);
        f.vconv_f = make(L.attn_v_conv_proj, 32);
      }
      f.tiny_pad = true;
    }
    head_fast = IsType(w->output, kk::kTQ8_0);
    // Capture the short-context graphs now (single-threaded, before any cache
    // thread exists).
    if (BatchCapable()) {
      for (nb = 1; nb <= kMaxB; ++nb)
        (void)CaptureGraph(2048, 2048 * 8 + nb);
      nb = 1;
    }
  }

  // ---- helpers ----
  const void* D(const TensorRef& t) const { return dw.Ptr(t); }
  const float* DF(const TensorRef& t) const {
    return static_cast<const float*>(dw.Ptr(t));
  }

  void G(const TensorRef& wt, const float* x, int ldx, float* out, int ldo,
         int n_tokens, int rows = 0) {
    kk::Gemm(s, static_cast<int>(wt.type), D(wt),
             rows ? rows : static_cast<int>(wt.ne[1]),
             static_cast<int>(wt.ne[0]), x, ldx, out, ldo, n_tokens);
  }

  /// Attention-residual mix of the ubatch's states with `nq` queries into out
  /// [T][nq][2304].
  void FMix(const kk::MixStates& st, const float* query, int nq, float* out) {
    kk::Mix(s, st, query, nq, kEmbd, T, out, cfg->rms_eps);
  }

  kk::MixStates States(const float* last, int extra_blocks_from = 0) {
    kk::MixStates st{};
    for (int i = extra_blocks_from; i < nblocks; ++i)
      st.ptr[st.n++] = blocks[i];
    st.ptr[st.n++] = last;
    return st;
  }

  // ---- one layer visit ----
  void Layer(Session& ss, int ip, int slot_idx,
             const float* msg /*[T][4][2304] or null*/) {
    const LayerWeights& L = w->layers[ip];
    const LayerFast& F = lf[ip];
    const bool is_swa = !L.is_global;
    Session::Slot& slot = ss.slots_[slot_idx];
    // The producers of the projection inputs also write their F16 copies.
    void* const in16[3] = {p_iq16, p_ik16, p_iv16};
    bool have16 = false;

    // A. attention inputs
    In inp[3] = {{cur, kEmbd}, {cur, kEmbd}, {cur, kEmbd}};
    bool normed =
        false;  // the fused mix already wrote the normalised inputs into in[]
    if (L.attn_res_query) {
      const int nq = is_swa ? 1 : 3;
      if (!msg) {
        fs::MixJob mj{};
        kk::MixStates st = States(cur);
        for (int i = 0; i < st.n; ++i)
          mj.st[mj.n++] = st.ptr[i];
        mj.query = DF(L.attn_res_query);
        mj.nq = nq;
        mj.ntok = T;
        for (int j = 0; j < nq; ++j) {
          mj.outq[j] = in[j];
          mj.outh[j] = F.proj ? in16[j] : nullptr;
          mj.nw[j] = is_swa ? DF(L.attn_norm)
                            : DF(j == 0   ? L.attn_q_inp_norm
                                 : j == 1 ? L.attn_k_inp_norm
                                          : L.attn_v_inp_norm);
        }
        fs::Mix(s, mj, cfg->rms_eps);
        normed = true;
        have16 = F.proj;
      } else {
        FMix(States(cur), DF(L.attn_res_query), nq, mixed);
        for (int j = 0; j < 3; ++j)
          inp[j] = {mixed + (is_swa ? 0 : j) * kEmbd, nq * kEmbd};
      }
    }
    if (msg) {
      fs::CtlPJobs cj{};
      cj.n = 3;
      for (int j = 0; j < 3; ++j) {
        const LoopController& c = L.loop[kStreamQ + j];
        cj.j[j] = {
            inp[j].p,        inp[j].ld,
            msg + j * kEmbd, 4 * kEmbd,
            DF(c.down),      DF(c.gain),
            DF(c.forget),    gate64 + static_cast<std::size_t>(j) * U * 64,
            ctl[j]};
      }
      fs::CtlPrefill(s, cj, T, cfg->rms_eps);
      for (int j = 0; j < 3; ++j)
        inp[j] = {ctl[j], kEmbd};
    }
    const float* iq;
    const float* ik;
    const float* iv;
    if (normed) {
      iq = in[0];
      ik = is_swa ? in[0] : in[1];
      iv = is_swa ? in[0] : in[2];
    } else if (is_swa) {
      const float* nw = DF(L.attn_norm);
      if (msg) {
        for (int j = 0; j < 3; ++j)
          kk::RmsNorm(s, inp[j].p, inp[j].ld, nw, in[j], kEmbd, kEmbd, T,
                      cfg->rms_eps, F.proj ? in16[j] : nullptr);
        have16 = F.proj;
        iq = in[0];
        ik = in[1];
        iv = in[2];
      } else {
        kk::RmsNorm(s, inp[0].p, inp[0].ld, nw, in[0], kEmbd, kEmbd, T,
                    cfg->rms_eps, F.proj ? in16[0] : nullptr);
        have16 = F.proj;
        iq = ik = iv = in[0];
      }
    } else {
      kk::RmsNorm(s, inp[0].p, inp[0].ld, DF(L.attn_q_inp_norm), in[0], kEmbd,
                  kEmbd, T, cfg->rms_eps, F.proj ? in16[0] : nullptr);
      kk::RmsNorm(s, inp[1].p, inp[1].ld, DF(L.attn_k_inp_norm), in[1], kEmbd,
                  kEmbd, T, cfg->rms_eps, F.proj ? in16[1] : nullptr);
      kk::RmsNorm(s, inp[2].p, inp[2].ld, DF(L.attn_v_inp_norm), in[2], kEmbd,
                  kEmbd, T, cfg->rms_eps, F.proj ? in16[2] : nullptr);
      have16 = F.proj;
      iq = in[0];
      ik = in[1];
      iv = in[2];
    }

    // B. projections
    const bool fp = F.proj;
    if (fp) {
      const std::size_t n = static_cast<std::size_t>(T) * kEmbd;
      if (!have16) {
        fs::CvtF16(s, iq, p_iq16, n);
        if (ik != iq)
          fs::CvtF16(s, ik, p_ik16, n);
        if (iv != iq && iv != ik)
          fs::CvtF16(s, iv, p_iv16, n);
      }
      const void* a_q = p_iq16;
      const void* a_k = ik != iq ? p_ik16 : p_iq16;
      const void* a_v = iv == iq ? p_iq16 : (iv == ik ? a_k : p_iv16);
      auto mm = [&](const TensorRef& wt, const void* a, float* out) {
        fs::GemmDense(s, D(wt), static_cast<int>(wt.ne[1]),
                      static_cast<int>(wt.ne[0]), a, out, DenseItems());
      };
      mm(L.attn_q, a_q, Qb);
      mm(L.attn_k, a_k, Kb);
      mm(L.attn_v, a_v, Vb);
      mm(L.attn_rel, a_q, Rb);
      mm(L.attn_gate, a_q, Gb);
    } else {
      G(L.attn_q, iq, kEmbd, Qb, kQ, T);
      G(L.attn_k, ik, kEmbd, Kb, kKV, T);
      G(L.attn_v, iv, kEmbd, Vb, kKV, T);
      G(L.attn_rel, iq, kEmbd, Rb, kRel, T);
      G(L.attn_gate, iq, kEmbd, Gb, kQ, T);
    }
    const float* dk = nullptr;
    const float* dv = nullptr;
    if (F.tiny_pad) {
      fs::SkinnyPrefill(s, iq, kEmbd, F.sink_f, 24, Sb, 24, T);
      if (!is_swa) {
        fs::SkinnyPrefill(s, ik, kEmbd, F.kconv_f, 32, dynK, 32, T);
        fs::SkinnyPrefill(s, iv, kEmbd, F.vconv_f, 32, dynV, 32, T);
        dk = dynK;
        dv = dynV;
      }
    } else {
      G(L.attn_sink_proj, iq, kEmbd, Sb, 24, T);
      if (!is_swa) {
        G(L.attn_k_conv_proj, ik, kEmbd, dynK, 32, T);
        G(L.attn_v_conv_proj, iv, kEmbd, dynV, 32, T);
        dk = dynK;
        dv = dynV;
      }
    }
    const float conv_scale =
        1.0f / std::sqrt(static_cast<float>((is_swa ? 1 : 2) * 4 + 1));
    float* cs = slot.conv;
    kk::Conv(s, Kb, cs + kConvOffK * 3, DF(L.attn_k_conv), dk, Kc, T, kKV, 128,
             conv_scale);
    kk::Conv(s, Vb, cs + kConvOffV * 3, DF(L.attn_v_conv), dv, Vc, T, kKV, 128,
             conv_scale);
    kk::ConvStateUpdate(s, Kb, cs + kConvOffK * 3, T, kKV);
    kk::ConvStateUpdate(s, Vb, cs + kConvOffV * 3, T, kKV);

    // C/D. clips, KV write, attention
    {
      fs::PrefillAttn pa{};
      pa.n = T;
      pa.pos0 = pos0;
      pa.cap = static_cast<int>(slot.rows);
      pa.swa = is_swa ? 1 : 0;
      pa.k = Kc;
      pa.v = Vc;
      pa.q = Qb;
      pa.r = Rb;
      pa.qscale = DF(L.attn_q_clip);
      pa.rclip = DF(L.attn_rel_clip);
      pa.kc = slot.k;
      pa.vc = slot.v;
      pa.qn = p_qn16;
      pa.rn = p_rn;
      pa.bt = F.bt;
      pa.sink = Sb;
      pa.gate = Gb;
      pa.out = Ob;
      pa.out16 = F.oproj ? p_ob16 : nullptr;
      pa.qk_thr = cfg->qk_clip;
      pa.r_thr = cfg->rel_clip;
      pa.eps = cfg->rms_eps;
      pa.tau_floor = static_cast<float>(cfg->log_scale_floor);
      pa.tau_alpha = cfg->log_scale_alpha;
      fs::AttnPrefill(s, pa);
    }
    if (F.oproj) {
      fs::GemmDense(s, D(L.attn_output), kEmbd, kQ, p_ob16, AOp, DenseItems());
    } else {
      G(L.attn_output, Ob, kQ, AOp, kEmbd, T);
    }
    // E. residual / block bookkeeping
    if (ip % cfg->attn_res_block != 0) {
      fs::ConvAdd(s, AOp, cs + kConvOffAttn * 3, DF(L.attn_post_conv), cur, T,
                  kEmbd, 1.0f / std::sqrt(5.0f));
    } else {
      kk::Conv(s, AOp, cs + kConvOffAttn * 3, DF(L.attn_post_conv), nullptr, AO,
               T, kEmbd, 128, 1.0f / std::sqrt(5.0f));
      kk::ConvStateUpdate(s, AOp, cs + kConvOffAttn * 3, T, kEmbd);
      // The partial sum so far becomes a finished block; the next one starts at
      // the attention output. Nothing before this point writes `cur`, so
      // swapping buffers replaces two copies.
      std::swap(cur, blocks[nblocks++]);
      std::swap(cur, AO);
    }
    // ffn_inp = cur

    // F. MoE input
    const float* hin = mixf;
    bool h_done = false;
    if (!msg) {
      fs::MixJob mj{};
      kk::MixStates st = States(cur);
      for (int i = 0; i < st.n; ++i)
        mj.st[mj.n++] = st.ptr[i];
      mj.query = DF(L.ffn_res_query);
      mj.nq = 1;
      mj.ntok = T;
      mj.outq[0] = h;
      mj.outh[0] = F.moe >= 0 ? p_h16 : nullptr;
      mj.nw[0] = DF(L.ffn_norm);
      fs::Mix(s, mj, cfg->rms_eps);
      h_done = true;
    } else {
      FMix(States(cur), DF(L.ffn_res_query), 1, mixf);
    }
    if (msg) {
      const LoopController& c = L.loop[kStreamFfn];
      fs::CtlPJobs cj{};
      cj.n = 1;
      cj.j[0] = {mixf,         kEmbd,      msg + 3 * kEmbd,
                 4 * kEmbd,    DF(c.down), DF(c.gain),
                 DF(c.forget), gate64,     ctl[0]};
      fs::CtlPrefill(s, cj, T, cfg->rms_eps);
      hin = ctl[0];
    }
    if (!h_done)
      kk::RmsNorm(s, hin, kEmbd, DF(L.ffn_norm), h, kEmbd, kEmbd, T,
                  cfg->rms_eps, F.moe >= 0 ? p_h16 : nullptr);

    // G. MoE
    if (Moe(L, lf[ip])) {
      fs::MoeTail(s, ye, cs + kConvOffFfn * 3, DF(L.ffn_post_conv), cur, T,
                  kEmbd, 1.0f / std::sqrt(5.0f));
    } else {
      kk::Conv(s, moe, cs + kConvOffFfn * 3, DF(L.ffn_post_conv), nullptr, AO,
               T, kEmbd, 128, 1.0f / std::sqrt(5.0f));
      kk::ConvStateUpdate(s, moe, cs + kConvOffFfn * 3, T, kEmbd);
      kk::Add(s, cur, AO, cur, T * kEmbd);
    }
  }

  /// Work items of a dense GEMM over the T tokens of the running ubatch (tiles
  /// of 64 tokens). Tokens per dense GEMM item: long ubatches use the 128-token
  /// kernel.
  int DenseTile() const { return T >= kWideTileTokens ? 128 : 64; }
  fs::GemmItems DenseItems() {
    const int tile = DenseTile();
    return {dn_nitems, dn_ie, dn_is, dn_in, (T + tile - 1) / tile, tile};
  }
  void UploadDenseItems() {
    const int tile = DenseTile();
    const int nt = (T + tile - 1) / tile;
    dn_host.assign(static_cast<std::size_t>(1 + 3 * (U / 64 + 2)), 0);
    const int stride = U / 64 + 2;
    dn_host[0] = nt;
    for (int i = 0; i < nt; ++i) {
      dn_host[1 + i] = 0;
      dn_host[1 + stride + i] = i * tile;
      dn_host[1 + 2 * stride + i] = std::min(tile, T - i * tile);
    }
    Check(
        hipMemcpyAsync(dn_nitems, dn_host.data(), 4, hipMemcpyHostToDevice, s),
        "items");
    Check(hipMemcpyAsync(dn_ie, dn_host.data() + 1, stride * 4,
                         hipMemcpyHostToDevice, s),
          "items");
    Check(hipMemcpyAsync(dn_is, dn_host.data() + 1 + stride, stride * 4,
                         hipMemcpyHostToDevice, s),
          "items");
    Check(hipMemcpyAsync(dn_in, dn_host.data() + 1 + 2 * stride, stride * 4,
                         hipMemcpyHostToDevice, s),
          "items");
  }

  /// The routed experts of the ubatch. True: the fast kernels left the per-slot
  /// rows in `ye` (the caller fuses their sum into the tail, MoeTail); false:
  /// the sum is in `moe`.
  bool Moe(const LayerWeights& L, const LayerFast& F) {
    const int E = static_cast<int>(cfg->n_expert),
              K = static_cast<int>(cfg->n_expert_used);
    const int nff = static_cast<int>(cfg->n_ff_expert);
    fs::RouterPrefill(s, DF(L.ffn_gate_inp), h, rlog, T);
    if (K == 8 && E == 128)
      fs::TopkPrefill(s, rlog, DF(L.exp_probs_b), ids_d, wts, T,
                      cfg->expert_weights_scale);
    else
      kk::RouterTopK(s, rlog, DF(L.exp_probs_b), ids_d, wts, T, E, K,
                     cfg->expert_weights_scale);
    const int P = T * K;
    if (F.moe >= 0) {
      const int tile = T >= kWideTileTokens ? 128 : T >= 1024 ? 64 : 32;
      const int max_items = (P + tile - 1) / tile + E;
      fs::MoeRoute(s, ids_d, wts, P, tile, mws);  // h's F16 copy is in p_h16
      fs::MoeGuPrefill(s, F.moe, tile, D(L.ffn_gate_up_exps),
                       L.ffn_gate_up_scale ? DF(L.ffn_gate_up_scale) : nullptr,
                       p_h16, p_act16, mws, max_items, cfg->situ_beta,
                       cfg->situ_linear_beta, nff);
      // per-(token, slot) outputs go to `ye`, then are summed in slot order (no
      // f32 atomics: two identical prefills must give identical bits, which the
      // cache and seeded sampling rely on)
      fs::MoeDownPrefill(s, F.moe, tile, D(L.ffn_down_exps),
                         L.ffn_down_scale ? DF(L.ffn_down_scale) : nullptr,
                         p_act16, ye, mws, max_items, kEmbd);
      return true;
    }
    fs::RouteBuild(s, ids_d, P, E, off_d, sorted_d);
    kk::MoeGemm(s, static_cast<int>(L.ffn_gate_up_exps.type),
                D(L.ffn_gate_up_exps), 2 * nff, kEmbd, E,
                L.ffn_gate_up_scale ? DF(L.ffn_gate_up_scale) : nullptr, h,
                kEmbd, K, sorted_d, off_d, gu, 2 * nff, P);
    kk::Situ(s, gu, act, nff, P, cfg->situ_beta, cfg->situ_linear_beta);
    kk::MoeGemm(s, static_cast<int>(L.ffn_down_exps.type), D(L.ffn_down_exps),
                kEmbd, nff, E,
                L.ffn_down_scale ? DF(L.ffn_down_scale) : nullptr, act, nff, 0,
                sorted_d, off_d, ye, kEmbd, P);
    kk::MoeCombine(s, ye, wts, moe, kEmbd, T, K);
    return false;
  }

  // =============================================================================================
  // Decode (T = 1): the whole 48-visit step as one HIP graph per
  // attention-split bucket. All position / token / session state is read from
  // device memory (d_step, d_tbl).
  // =============================================================================================
  const float* DS(const TensorRef& t) const {
    return static_cast<const float*>(dw.Ptr(t));
  }

  void GenericG(const TensorRef& wt, const float* x, float* out, int rows = 0) {
    kk::Gemm(ds, static_cast<int>(wt.type), D(wt),
             rows ? rows : static_cast<int>(wt.ne[1]),
             static_cast<int>(wt.ne[0]), x, static_cast<int>(wt.ne[0]), out,
             rows ? rows : static_cast<int>(wt.ne[1]), 1);
  }

  // the post-conv + residual of the last op is deferred so that the next mix
  // can absorb it
  struct Pending {
    bool on = false;
    fs::ConvResidArgs a{};
  } pend;
  void FlushPending() {
    if (!pend.on)
      return;
    fs::ConvResid(ds, pend.a, nb);
    pend.on = false;
  }

  void DMix(const float* last, const float* query, int nq, float* out,
            const float* const* nw = nullptr, int ostride = 0) {
    fs::MixJob mj{};
    mj.ntok = nb;
    mj.ostride = ostride;
    for (int i = 0; i < d_ns; ++i)
      mj.st[mj.n++] = d_states[i];
    mj.st[mj.n++] = last;
    mj.query = query;
    mj.out = out;
    mj.nq = nq;
    if (nw)
      for (int q = 0; q < nq; ++q)
        mj.nw[q] = nw[q];
    if (pend.on) {
      if (nq == 1) {
        mj.has_cr = 1;
        mj.cr = pend.a;
        pend.on = false;
      } else {
        FlushPending();
      }
    }
    fs::Mix(ds, mj, cfg->rms_eps);
  }

  float* NewBlock() {
    if (nblk_alloc >= 16)
      throw std::runtime_error("inkling2: too many blocks");
    return d_blk[nblk_alloc++];
  }

  void DLayer(int ip, int slot_idx, const float* msg, int bucket_ctx) {
    const LayerWeights& L = w->layers[ip];
    const LayerFast& F = lf[ip];
    const bool swa = !L.is_global;
    float* cur = d_cur;
    const fs::SlotEntry* slot = d_tbl + slot_idx;
    const int* pos = d_step + kMaxB;

    // A. attention inputs
    const float* inp[3] = {cur, cur, cur};
    const bool three = msg || !swa;  // separate q / k / v inputs
    bool normed =
        false;  // the mix already wrote the normalised inputs into d_in
    if (L.attn_res_query) {
      const int nq = swa ? 1 : 3;
      if (!msg) {
        const float* nwp[3];
        for (int j = 0; j < 3; ++j)
          nwp[j] = swa ? DS(L.attn_norm)
                       : DS(j == 0   ? L.attn_q_inp_norm
                            : j == 1 ? L.attn_k_inp_norm
                                     : L.attn_v_inp_norm);
        DMix(cur, DS(L.attn_res_query), nq, d_in, nwp, kSIn);
        normed = true;
      } else {
        DMix(cur, DS(L.attn_res_query), nq, d_mixed, nullptr, kSIn);
        for (int j = 0; j < 3; ++j)
          inp[j] = d_mixed + (swa ? 0 : j) * kEmbd;
      }
    } else {
      FlushPending();
    }
    if (msg) {
      fs::CtlJobs cj{};
      cj.n = 3;
      for (int j = 0; j < 3; ++j) {
        const LoopController& c = L.loop[kStreamQ + j];
        cj.x[j] = inp[j];
        cj.msg[j] = msg + j * kEmbd;
        cj.down[j] = HW(c.down);
        cj.gain[j] = HW(c.gain);
        cj.forget[j] = HW(c.forget);
        cj.half = ctl_half;
        cj.g64[j] = d_g64 + j * 64;
        cj.out[j] = d_ctl + j * kEmbd;
        cj.xs[j] = kSIn;
        cj.ms[j] = kSCtl * static_cast<int>(cfg->CoreLen());  // all core layers
        cj.gs[j] = 4 * 64;
        cj.os[j] = kSCtl;
        inp[j] = cj.out[j];
      }
      fs::CtlGate(ds, cj, cfg->rms_eps, nb);
      fs::CtlApply(ds, cj, nb);
    }
    if (!normed) {
      fs::NormJobs nj{};
      nj.n = three ? 3 : 1;
      for (int j = 0; j < nj.n; ++j) {
        nj.x[j] = inp[j];
        nj.out[j] = d_in + j * kEmbd;
        nj.xs[j] = msg ? kSCtl : kEmbd;
        nj.os[j] = kSIn;
        nj.w[j] = swa ? DS(L.attn_norm)
                      : DS(j == 0   ? L.attn_q_inp_norm
                           : j == 1 ? L.attn_k_inp_norm
                                    : L.attn_v_inp_norm);
      }
      fs::Rms(ds, nj, cfg->rms_eps, nb);
    }
    const float* iq = d_in;
    const float* ik = three ? d_in + kEmbd : d_in;
    const float* iv = three ? d_in + 2 * kEmbd : d_in;

    // B. projections: Q [3072] | gate [3072] | rel [384] | K [1024] | V [1024]
    float* Qb = d_proj;
    float* Gb = d_proj + 3072;
    float* Rb = d_proj + 6144;
    float* Kb = d_proj + 6528;
    float* Vb = d_proj + 7552;
    const bool fuse_tiny =
        F.tiny &&
        F.proj;  // sink / dynamic-tap rows ride in the projection launch
    if (F.proj) {
      fs::ProjSpec ps{};
      const TensorRef* ws[5] = {&L.attn_q, &L.attn_gate, &L.attn_rel, &L.attn_k,
                                &L.attn_v};
      const int in_idx[5] = {0, 0, 0, three ? 1 : 0, three ? 2 : 0};
      ps.n = 5;
      for (int i = 0; i < 5; ++i) {
        ps.w[i] = D(*ws[i]);
        ps.m[i] = static_cast<int>(ws[i]->ne[1]);
        ps.in[i] = in_idx[i];
      }
      if (fuse_tiny) {
        auto add = [&](const TensorRef& t, int in, float* y, int rows, int ys) {
          ps.tw[ps.tn] = D(t);
          ps.ttype[ps.tn] = static_cast<int>(t.type);
          ps.trows[ps.tn] = rows;
          ps.tin[ps.tn] = in;
          ps.ty[ps.tn] = y;
          ps.tys[ps.tn] = ys;
          ++ps.tn;
        };
        add(L.attn_sink_proj, 0, d_sink, 24, 24);
        if (!swa) {
          add(L.attn_k_conv_proj, three ? 1 : 0, d_dyn, 32, 64);
          add(L.attn_v_conv_proj, three ? 2 : 0, d_dyn + 32, 32, 64);
        }
      }
      fs::ProjMulti(ds, ps, d_in, three ? 3 : 1, d_proj, nb, kSIn, kSProj);
    } else {
      GenericG(L.attn_q, iq, Qb);
      GenericG(L.attn_gate, iq, Gb);
      GenericG(L.attn_rel, iq, Rb);
      GenericG(L.attn_k, ik, Kb);
      GenericG(L.attn_v, iv, Vb);
    }
    float* dynK = d_dyn;
    float* dynV = d_dyn + 32;
    if (fuse_tiny) {
      // computed inside the projection launch
    } else if (F.tiny) {
      fs::TinyJobs tj{};
      tj.n = 0;
      auto add = [&](const TensorRef& t, const float* x, float* y, int rows) {
        tj.w[tj.n] = D(t);
        tj.type[tj.n] = static_cast<int>(t.type);
        tj.x[tj.n] = x;
        tj.y[tj.n] = y;
        tj.rows[tj.n] = rows;
        tj.xs[tj.n] = kSIn;
        tj.ys[tj.n] = (y == d_sink) ? 24 : 64;
        ++tj.n;
      };
      add(L.attn_sink_proj, iq, d_sink, 24);
      if (!swa) {
        add(L.attn_k_conv_proj, ik, dynK, 32);
        add(L.attn_v_conv_proj, iv, dynV, 32);
      }
      fs::Tiny(ds, tj, nb);
    } else {
      GenericG(L.attn_sink_proj, iq, d_sink);
      if (!swa) {
        GenericG(L.attn_k_conv_proj, ik, dynK);
        GenericG(L.attn_v_conv_proj, iv, dynV);
      }
    }

    // C. conv + clip + KV write
    {
      fs::KvPrepArgs ka{};
      ka.k = Kb;
      ka.v = Vb;
      ka.dk = swa ? nullptr : dynK;
      ka.dv = swa ? nullptr : dynV;
      ka.wk = DS(L.attn_k_conv);
      ka.wv = DS(L.attn_v_conv);
      ka.slot = slot;
      ka.conv_off_k = kConvOffK;
      ka.conv_off_v = kConvOffV;
      ka.pos = pos;
      ka.scale = 1.0f / std::sqrt(static_cast<float>((swa ? 1 : 2) * 4 + 1));
      ka.qk_thr = cfg->qk_clip;
      ka.eps = cfg->rms_eps;
      ka.ld = kSProj;
      ka.ldd = 64;
      fs::KvPrep(ds, ka, nb);
    }

    // D. attention + gate
    {
      fs::AttnArgs aa{};
      aa.slot = slot;
      aa.pos = pos;
      aa.q = Qb;
      aa.r = Rb;
      aa.qscale = DS(L.attn_q_clip);
      aa.rclip = DS(L.attn_rel_clip);
      aa.bt = F.bt;
      aa.sink = d_sink;
      aa.gate = Gb;
      aa.out = d_ob;
      aa.ws_o = d_ws_o;
      aa.ws_m = d_ws_m;
      aa.ws_l = d_ws_l;
      aa.swa = swa ? 1 : 0;
      if (swa) {
        aa.chunk = 128;
        aa.nsplit = 8;
      } else {
        int chunk = std::clamp(bucket_ctx / 32, 128, 2048);
        chunk = std::max(chunk, (bucket_ctx / 256 + 63) / 64 * 64);
        aa.chunk = chunk;
        aa.nsplit = (bucket_ctx + chunk - 1) / chunk;
      }
      aa.qk_thr = cfg->qk_clip;
      aa.r_thr = cfg->rel_clip;
      aa.tau_floor = static_cast<float>(cfg->log_scale_floor);
      aa.tau_alpha = cfg->log_scale_alpha;
      aa.ldp = kSProj;
      aa.ws_tok = static_cast<int>(kMaxSplit * 8 * 12);
      fs::AttnDecode(ds, aa, nb);
    }
    if (F.oproj)
      fs::GemvQ8(ds, kQ, D(L.attn_output), d_ob, d_ao, kEmbd, 0.f, nb);
    else
      GenericG(L.attn_output, d_ob, d_ao);
    {
      fs::ConvResidArgs ca{};
      ca.x = d_ao;
      ca.w = DS(L.attn_post_conv);
      ca.slot = slot;
      ca.conv_off = kConvOffAttn;
      ca.scale = 1.0f / std::sqrt(5.0f);
      ca.cur = cur;
      ca.blk = nullptr;
      if (ip % static_cast<int>(cfg->attn_res_block) == 0) {
        ca.blk = NewBlock();
        d_states[d_ns++] = ca.blk;
      }
      pend.on = true;
      pend.a = ca;
    }

    // F. MoE input
    if (!msg) {
      const float* nwp[1] = {DS(L.ffn_norm)};
      DMix(cur, DS(L.ffn_res_query), 1, d_h, nwp);
    } else {
      DMix(cur, DS(L.ffn_res_query), 1, d_mixf);
      fs::CtlJobs cj{};
      cj.n = 1;
      const LoopController& c = L.loop[kStreamFfn];
      cj.x[0] = d_mixf;
      cj.msg[0] = msg + 3 * kEmbd;
      cj.down[0] = HW(c.down);
      cj.gain[0] = HW(c.gain);
      cj.forget[0] = HW(c.forget);
      cj.half = ctl_half;
      cj.g64[0] = d_g64;
      cj.out[0] = d_ctl;
      cj.xs[0] = kEmbd;
      cj.ms[0] = kSCtl * static_cast<int>(cfg->CoreLen());
      cj.gs[0] = 4 * 64;
      cj.os[0] = kSCtl;
      fs::CtlGate(ds, cj, cfg->rms_eps, nb);
      fs::CtlApply(ds, cj, nb);
      fs::NormJobs nj{};
      nj.n = 1;
      nj.x[0] = d_ctl;
      nj.w[0] = DS(L.ffn_norm);
      nj.out[0] = d_h;
      nj.xs[0] = kSCtl;
      nj.os[0] = kEmbd;
      fs::Rms(ds, nj, cfg->rms_eps, nb);
    }

    // G. MoE
    const int E = static_cast<int>(cfg->n_expert);
    const int nff = static_cast<int>(cfg->n_ff_expert);
    fs::Router(ds, HW(L.ffn_gate_inp), DS(L.exp_probs_b), d_h, d_rlog,
               d_counter, d_ids8, d_wts8, cfg->expert_weights_scale, nb,
               ctl_half);
    if (F.moe >= 0) {
      fs::MoeGu(ds, F.moe, D(L.ffn_gate_up_exps),
                L.ffn_gate_up_scale ? DS(L.ffn_gate_up_scale) : nullptr, d_h,
                d_ids8, d_act8, nff, cfg->situ_beta, cfg->situ_linear_beta, nb);
      fs::MoeDownE(ds, F.moe, D(L.ffn_down_exps),
                   L.ffn_down_scale ? DS(L.ffn_down_scale) : nullptr, d_act8,
                   d_ids8, d_wts8, d_moe_part, kEmbd, nb);
    } else {
      fs::RouteBuild(ds, d_ids8, 8, E, off_d, sorted_d);
      kk::MoeGemm(ds, static_cast<int>(L.ffn_gate_up_exps.type),
                  D(L.ffn_gate_up_exps), 2 * nff, kEmbd, E,
                  L.ffn_gate_up_scale ? DS(L.ffn_gate_up_scale) : nullptr, d_h,
                  kEmbd, 8, sorted_d, off_d, gu, 2 * nff, 8);
      kk::Situ(ds, gu, act, nff, 8, cfg->situ_beta, cfg->situ_linear_beta);
      kk::MoeGemm(ds, static_cast<int>(L.ffn_down_exps.type),
                  D(L.ffn_down_exps), kEmbd, nff, E,
                  L.ffn_down_scale ? DS(L.ffn_down_scale) : nullptr, act, nff,
                  0, sorted_d, off_d, ye, kEmbd, 8);
      kk::MoeCombine(ds, ye, d_wts8, d_moe, kEmbd, 1, 8);
    }
    {
      fs::ConvResidArgs ca{};
      ca.x = F.moe >= 0 ? d_moe_part : d_moe;
      ca.nsum = F.moe >= 0 ? 8 : 0;
      ca.w = DS(L.ffn_post_conv);
      ca.slot = slot;
      ca.conv_off = kConvOffFfn;
      ca.scale = 1.0f / std::sqrt(5.0f);
      ca.cur = cur;
      ca.blk = nullptr;
      pend.on = true;
      pend.a = ca;
    }
  }

  /// Enqueues the whole decode step on `ds` (captured or run directly).
  void BuildDecode(int bucket_ctx) {
    float* cur = d_cur;
    pend.on = false;
    kk::EmbedRows(ds, static_cast<int>(w->token_embd.type), D(w->token_embd),
                  kEmbd, d_step, nb, cur, kEmbd);
    kk::RmsNorm(ds, cur, kEmbd, DS(w->token_embd_norm), cur, kEmbd, kEmbd, nb,
                cfg->rms_eps);
    nblk_alloc = 0;
    d_ns = 0;
    for (int ip = 0; ip < static_cast<int>(cfg->core_start); ++ip)
      DLayer(ip, ip, nullptr, bucket_ctx);
    FlushPending();
    Check(hipMemcpyAsync(d_prelude, cur, nb * kEmbd * 4,
                         hipMemcpyDeviceToDevice, ds),
          "prelude");
    const int prelude_n = d_ns;
    const int core_len = static_cast<int>(cfg->CoreLen());
    const float* bank[16];
    int bank_n = 0;
    for (int pass = 0; pass < static_cast<int>(cfg->n_passes); ++pass) {
      if (pass > 0) {
        FlushPending();
        bank_n = 0;
        for (int i = prelude_n + 1; i < d_ns; ++i)
          bank[bank_n++] = d_states[i];
        Check(hipMemcpyAsync(d_bankout, cur, nb * kEmbd * 4,
                             hipMemcpyDeviceToDevice, ds),
              "bankout");
        Check(hipMemcpyAsync(cur, d_prelude, nb * kEmbd * 4,
                             hipMemcpyDeviceToDevice, ds),
              "restore");
        d_ns = prelude_n;
      }
      for (int ip = static_cast<int>(cfg->core_start);
           ip < static_cast<int>(cfg->core_end); ++ip) {
        if (pass == 0) {
          DLayer(ip, ip, nullptr, bucket_ctx);
        } else {
          if (ip == static_cast<int>(cfg->core_start)) {
            // The messages of every core layer of this pass come from the same
            // bank (the previous pass), so one launch mixes all of them:
            // [token][core layer][4][2304].
            fs::MixJob mj{};
            mj.ntok = nb;
            for (int i = 0; i < bank_n; ++i)
              mj.st[mj.n++] = bank[i];
            mj.st[mj.n++] = d_bankout;
            mj.query = msgq_all;
            mj.out = d_msgs;
            mj.nq = 4 * core_len;
            fs::Mix(ds, mj, cfg->rms_eps);
          }
          const int slot = static_cast<int>(cfg->n_phys) +
                           (pass - 1) * core_len +
                           (ip - static_cast<int>(cfg->core_start));
          DLayer(
              ip, slot,
              d_msgs + static_cast<std::size_t>(ip - cfg->core_start) * kSCtl,
              bucket_ctx);
        }
      }
    }
    for (int ip = static_cast<int>(cfg->core_end);
         ip < static_cast<int>(cfg->n_phys); ++ip)
      DLayer(ip, ip, nullptr, bucket_ctx);
    DMix(cur, DS(w->output_res_query), 1, d_mixf);
    kk::RmsNorm(ds, d_mixf, kEmbd, DS(w->output_norm), d_hn, kEmbd, kEmbd, nb,
                cfg->rms_eps);
    const int V = static_cast<int>(cfg->n_vocab_text);
    if (head_fast) {
      fs::GemvQ8(ds, kEmbd, D(w->output), d_hn, logits_d, V, cfg->logit_softcap,
                 nb);
    } else {
      kk::Gemm(ds, static_cast<int>(w->output.type), D(w->output), V, kEmbd,
               d_hn, kEmbd, logits_d, V, 1);
      kk::Softcap(ds, logits_d, V, cfg->logit_softcap);
    }
  }

  static int BucketCtx(int pos) {
    int c = 2048;
    while (c < pos + 1)
      c *= 2;
    return c;
  }

  /// Captures the decode step of the current `nb` and `ctx` into a graph.
  /// Returns graphs.end() when the capture was invalidated: the cache adapter
  /// issues hipMemcpy on the legacy stream from other threads without the
  /// engine mutex, and the runtime can refuse such a copy while a capture is
  /// open even in thread-local mode.
  std::map<int, hipGraphExec_t>::iterator CaptureGraph(int ctx, int key) {
    // Exclusive: no cache or vision thread may issue a device call while the
    // capture is open.
    const std::unique_lock<std::shared_mutex> gate(CaptureGate());
    if (hipStreamBeginCapture(ds, hipStreamCaptureModeThreadLocal) !=
        hipSuccess) {
      (void)hipGetLastError();
      if (failed_graphs.insert(key).second)
        std::fprintf(
            stderr,
            "inkling2: decode graph capture could not start for context bucket "
            "%d, width %d; those steps run without a graph\n",
            ctx, key % 8);
      return graphs.end();
    }
    bool ok = true;
    try {
      BuildDecode(ctx);
    } catch (const std::exception&) {
      ok = false;
    }
    hipGraph_t g = nullptr;
    if (hipStreamEndCapture(ds, &g) != hipSuccess || g == nullptr)
      ok = false;
    hipGraphExec_t ex = nullptr;
    if (ok && hipGraphInstantiateWithFlags(&ex, g, 0) != hipSuccess)
      ok = false;
    if (g)
      (void)hipGraphDestroy(g);
    (void)hipGetLastError();
    if (!ok) {
      if (failed_graphs.insert(key).second)
        std::fprintf(stderr,
                     "inkling2: decode graph capture failed for context bucket "
                     "%d, width %d; those steps run without a graph\n",
                     ctx, key % 8);
      return graphs.end();
    }
    return graphs.emplace(key, ex).first;
  }

  /// All 48 visits can run through the batched fast kernels (needed for nb >
  /// 1).
  [[nodiscard]] bool BatchCapable() const {
    if (!head_fast)
      return false;
    for (const auto& f : lf)
      if (!f.proj || !f.oproj || !f.tiny || f.moe < 0)
        return false;
    return true;
  }

  void Decode(Session& ss, std::int32_t token, std::vector<float>* out) {
    Session* sp = &ss;
    std::vector<float>* op = out;
    DecodeN(&sp, &token, &op, 1);
  }

  /// One decode step of `n` (<= kMaxB) distinct sessions, one token each, as
  /// one graph launch: the weights are read once for all tokens.
  void DecodeN(Session* const* sess, const std::int32_t* tokens,
               std::vector<float>* const* outs, int n) {
    nb = n;
    int maxpos = 0;
    std::int32_t step[2 * kMaxB] = {};
    for (int b = 0; b < n; ++b) {
      const int pos = static_cast<int>(sess[b]->pos_);
      maxpos = std::max(maxpos, pos);
      step[b] = tokens[b];
      step[kMaxB + b] = pos;
    }
    const int ctx = BucketCtx(maxpos);
    // M2: the cache adapter writes state on the default stream; `ds` is
    // non-blocking, so order it.
    Check(hipStreamSynchronize(s), "pre-decode sync");
    Check(hipMemcpyAsync(d_step, step, sizeof step, hipMemcpyHostToDevice, ds),
          "step");
    for (int b = 0; b < n; ++b)
      Check(hipMemcpyAsync(d_tbl + b * 48, sess[b]->tbl_,
                           48 * sizeof(fs::SlotEntry), hipMemcpyDeviceToDevice,
                           ds),
            "slot table");
    const int key = ctx * 8 + n;
    auto it = graphs.find(key);
    if (it == graphs.end() && failed_graphs.count(key) == 0)
      it = CaptureGraph(ctx, key);
    if (it != graphs.end()) {
      Check(hipGraphLaunch(it->second, ds), "graph launch");
    } else {
      BuildDecode(ctx);  // capture was invalidated (see CaptureGraph): run this
                         // step eagerly; a failed key is not retried
    }
    const std::size_t V = cfg->n_vocab_text;
    for (int b = 0; b < n; ++b) {
      outs[b]->resize(V);
      Check(hipMemcpyAsync(outs[b]->data(), logits_d + b * V, V * 4,
                           hipMemcpyDeviceToHost, ds),
            "logits copy");
    }
    Check(hipStreamSynchronize(ds), "decode sync");
    for (int b = 0; b < n; ++b)
      sess[b]->pos_ += 1;
    nb = 1;
  }

  void Tap(int visit, const char* what) {
    if (!eng->tap_)
      return;
    std::vector<float> row(kEmbd);
    Check(hipStreamSynchronize(s), "tap sync");
    Check(hipMemcpy(row.data(), cur, kEmbd * 4, hipMemcpyDeviceToHost),
          "tap copy");
    eng->tap_(visit, what, row);
  }

  // ---- one ubatch ----
  void Ubatch(Session& ss, std::span<const std::int32_t> tok, int first_idx,
              bool last_ubatch, const LogitsSink& sink,
              YieldRun* yr = nullptr) {
    T = static_cast<int>(tok.size());
    pos0 = static_cast<int>(ss.pos_);
    Check(hipMemcpyAsync(tok_d, tok.data(), tok.size() * 4,
                         hipMemcpyHostToDevice, s),
          "token copy");
    kk::EmbedRows(s, static_cast<int>(w->token_embd.type), D(w->token_embd),
                  kEmbd, tok_d, T, cur, kEmbd);
    kk::RmsNorm(s, cur, kEmbd, DF(w->token_embd_norm), cur, kEmbd, kEmbd, T,
                cfg->rms_eps);
    for (const InputRows& r :
         input_rows) {  // M5 hook: image rows replace the (normed) embeddings
      const std::size_t lo = std::max<std::size_t>(r.first, first_idx);
      const std::size_t hi = std::min<std::size_t>(
          r.first + r.count, first_idx + static_cast<std::size_t>(T));
      if (lo >= hi)
        continue;
      Check(hipMemcpyAsync(cur + (lo - first_idx) * kEmbd,
                           r.device + (lo - r.first) * kEmbd,
                           (hi - lo) * kEmbd * sizeof(float),
                           hipMemcpyDeviceToDevice, s),
            "input rows copy");
    }
    kk::LogScale(s, tau, pos0, T, static_cast<float>(cfg->log_scale_floor),
                 cfg->log_scale_alpha);
    nblocks = 0;
    UploadDenseItems();
    int visit = 0;
    for (int ip = 0; ip < static_cast<int>(cfg->core_start); ++ip) {
      Layer(ss, ip, ip, nullptr);
      Tap(visit++, "l_out");
      MaybeYield(yr, true);  // M2 hook
    }
    kk::Copy(s, cur, prelude, T * kEmbd);
    const int prelude_n = nblocks;
    const int core_len = static_cast<int>(cfg->CoreLen());
    for (int pass = 0; pass < static_cast<int>(cfg->n_passes); ++pass) {
      int bank_n = 0;
      if (pass > 0) {
        // The previous pass's core blocks and end state become the message bank
        // (buffer swaps).
        for (int i = prelude_n + 1; i < nblocks; ++i)
          std::swap(blocks[i], bank[bank_n++]);
        std::swap(cur, bankout);
        kk::Copy(s, prelude, cur, T * kEmbd);
        nblocks = prelude_n;
      }
      for (int ip = static_cast<int>(cfg->core_start);
           ip < static_cast<int>(cfg->core_end); ++ip) {
        if (pass == 0) {
          Layer(ss, ip, ip, nullptr);
        } else {
          kk::MixStates st{};
          for (int i = 0; i < bank_n; ++i)
            st.ptr[st.n++] = bank[i];
          st.ptr[st.n++] = bankout;
          FMix(st, msgq[ip], 4, msgs);
          const int slot = static_cast<int>(cfg->n_phys) +
                           (pass - 1) * core_len +
                           (ip - static_cast<int>(cfg->core_start));
          Layer(ss, ip, slot, msgs);
        }
        Tap(visit++, "l_out");
        MaybeYield(yr, true);  // M2 hook
      }
    }
    for (int ip = static_cast<int>(cfg->core_end);
         ip < static_cast<int>(cfg->n_phys); ++ip) {
      Layer(ss, ip, ip, nullptr);
      Tap(visit++, "l_out");
      if (visit < 48)
        MaybeYield(yr,
                   true);  // M2 hook (the last visit is followed by the head)
    }
    ss.pos_ += T;
    if (!last_ubatch && !sink)
      return;

    // output: attention-residual mix, norm, lm head, soft cap, crop
    FMix(States(cur), DF(w->output_res_query), 1, mixf);
    kk::RmsNorm(s, mixf, kEmbd, DF(w->output_norm), hn, kEmbd, kEmbd, T,
                cfg->rms_eps);
    const int V = static_cast<int>(cfg->n_vocab_text);
    auto chunk = [&](int a, int n, std::vector<float>* host) {
      G(w->output, hn + static_cast<std::size_t>(a) * kEmbd, kEmbd, logits_d, V,
        n, V);
      kk::Softcap(s, logits_d, static_cast<long>(n) * V, cfg->logit_softcap);
      host->resize(static_cast<std::size_t>(n) * V);
      Check(hipMemcpyAsync(host->data(), logits_d, host->size() * 4,
                           hipMemcpyDeviceToHost, s),
            "logits copy");
      Check(hipStreamSynchronize(s), "logits sync");
    };
    std::vector<float> host;
    if (sink) {
      for (int a = 0; a < T; a += kLogitChunk) {
        const int n = std::min(kLogitChunk, T - a);
        chunk(a, n, &host);
        sink(static_cast<std::size_t>(first_idx + a), n, host.data());
        if (last_ubatch && a + n == T)
          ss.logits_.assign(host.end() - V, host.end());
      }
    } else {
      chunk(T - 1, 1, &host);
      ss.logits_ = std::move(host);
    }
  }
};

// ---------------------------------------------------------------------------------------------
Session::~Session() = default;

std::shared_ptr<mem::StateAccount> Session::account() const {
  return state_ ? state_->account() : nullptr;
}

void Session::EnsureRows(std::size_t end) {
  std::string error;
  for (const auto& region : regions_) {
    if (!region->Ensure(end + kKvMarginRows, &error))
      throw std::runtime_error("inkling2: cannot back K/V rows: " + error);
  }
}

void Session::BackRows(std::size_t first, std::size_t end) {
  EnsureRows(end);
  if (regions_.empty())
    return;
  // Appends keep the rows before `first`; pages a checkpoint shares are copied
  // only where the write lands in them. One barrier for every region.
  std::string error;
  for (const auto& region : regions_)
    if (!region->PrepareWrite(first, end, first, &error))
      throw std::runtime_error("inkling2: cannot prepare K/V rows: " + error);
  if (!regions_.front()->SettleRemaps(&error))
    throw std::runtime_error("inkling2: cannot settle K/V pages: " + error);
}

void Session::PrepareRestore(std::size_t rows) {
  std::string error;
  for (const auto& region : regions_)
    (void)region->Restore({}, rows);
  if (!regions_.empty() && !regions_.front()->SettleRemaps(&error))
    throw std::runtime_error("inkling2: cannot settle K/V pages: " + error);
  EnsureRows(rows);
}

Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::Create(
    std::shared_ptr<const core::GgufReader> reader, std::uint32_t max_context,
    std::string* error) {
  try {
    auto e = std::unique_ptr<Engine>(new Engine());
    std::string why;
    auto bound = Inkling2Weights::Bind(*reader, &why);
    if (!bound) {
      if (error)
        *error = "inkling2: " + why;
      return nullptr;
    }
    e->weights_ = std::move(*bound);
    e->reader_ = std::move(reader);
    e->max_context_ = max_context;
    int dev = 0;
    Check(hipGetDevice(&dev), "device");
    hipDeviceProp_t prop{};
    Check(hipGetDeviceProperties(&prop, dev), "device properties");
    if (prop.warpSize != 32)
      throw std::runtime_error("inkling2: wave32 device required");
    e->impl_ = std::make_unique<Impl>();
    e->impl_->s = nullptr;
    e->impl_->dw.Upload(*e->reader_, e->weights_, &e->weight_bytes_);
    e->impl_->Init(e.get());
    return e;
  } catch (const std::exception& ex) {
    if (error)
      *error = ex.what();
    return nullptr;
  }
}

Engine::StateLayout Engine::DeclareState() const {
  StateLayout layout;
  auto& state = layout.state;
  const std::size_t kv_row = static_cast<std::size_t>(kKV) * 2;  // f16
  const std::size_t ring = 1024 + ubatch_;
  for (int i = 0; i < 48; ++i) {
    const int phys = i < 24 ? i : 6 + (i - 24) % 12;
    if (weights_.config.global_layer[phys]) {
      // Global visits keep every position: rows backed as they are written.
      layout.k[i] = state.Context({.row_bytes = kv_row, .rows = max_context_});
      layout.v[i] = state.Context({.row_bytes = kv_row, .rows = max_context_});
    } else {
      // Window visits keep a ring of window + one ubatch rows.
      layout.k[i] = state.Fixed(kv_row * ring);
      layout.v[i] = state.Fixed(kv_row * ring);
    }
    layout.conv[i] =
        state.Fixed(static_cast<std::size_t>(kConvChannels) * 3 * 4);
  }
  layout.table = state.Fixed(48 * sizeof(fs::SlotEntry));
  return layout;
}

std::size_t Engine::session_bytes() const {
  return DeclareState().state.Bytes(mem::StateBytes::kSize);
}

std::size_t Engine::demand_mapped_bytes() const {
  if (!mem::DemandMapped())
    return 0;
  return DeclareState().state.ContextBytes(mem::StateBytes::kSize);
}

std::unique_ptr<Session> Engine::NewSession() {
  std::unique_ptr<Session> s(new Session());
  const StateLayout layout = DeclareState();
  std::string error;
  {
    std::shared_lock<std::shared_mutex> gate(CaptureGate());
    s->state_ = mem::SessionState::Create(layout.state, &error);
  }
  if (!s->state_)
    throw std::runtime_error("inkling2: session state: " + error);
  const auto& state = *s->state_;
  s->bytes_ = state.allocated_bytes();
  const std::size_t ring = 1024 + ubatch_;
  fs::SlotEntry host[48];
  for (int i = 0; i < 48; ++i) {
    const int phys = i < 24 ? i : 6 + (i - 24) % 12;
    auto& sl = s->slots_[i];
    sl.global = weights_.config.global_layer[phys];
    sl.rows = sl.global ? max_context_ : ring;
    sl.k = state.data(layout.k[i]);
    sl.v = state.data(layout.v[i]);
    sl.conv = state.data<float>(layout.conv[i]);
    if (sl.global) {
      s->regions_.push_back(state.region(layout.k[i]));
      s->regions_.push_back(state.region(layout.v[i]));
    }
    host[i] = {sl.k, sl.v, sl.conv, static_cast<int>(sl.rows), 0};
  }
  s->tbl_ = state.data(layout.table);
  {
    const std::shared_lock<std::shared_mutex> gate(CaptureGate());
    Check(hipMemcpy(s->tbl_, host, sizeof host, hipMemcpyHostToDevice),
          "slot table");
  }
  Reset(*s);
  return s;
}

void Engine::Reset(Session& s) {
  {
    const std::shared_lock<std::shared_mutex> gate(CaptureGate());
    for (auto& sl : s.slots_)
      Check(hipMemset(sl.conv, 0, kConvChannels * 3 * 4), "conv reset");
  }
  s.pos_ = 0;
  s.logits_.clear();
}

bool Engine::batch_capable() const {
  return impl_->BatchCapable();
}

void Engine::DecodeBatch(std::span<Session* const> sessions,
                         std::span<const std::int32_t> tokens) {
  if (sessions.size() != tokens.size() ||
      sessions.size() > static_cast<std::size_t>(Impl::kMaxB))
    throw std::runtime_error("inkling2: bad decode batch");
  const std::lock_guard lock(mutex_);
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Session& s = *sessions[i];
    for (std::size_t j = 0; j < i; ++j)
      if (sessions[j] == &s)
        throw std::runtime_error("inkling2: duplicate session in decode batch");
    if (s.pos_ + 1 > max_context_)
      throw std::runtime_error("inkling2: context exhausted");
    if (tokens[i] < 0 ||
        static_cast<std::uint32_t>(tokens[i]) >= weights_.config.n_vocab_text)
      throw std::runtime_error("inkling2: token id out of range");
  }
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Session& s = *sessions[i];
    if (s.hooks_ != nullptr)
      s.hooks_->BeforeWrite(s.pos_, s.pos_ + 1);  // M2 hook
    s.BackRows(s.pos_, s.pos_ + 1);
  }
  if (sessions.size() >= 2 && batch_capable()) {
    std::vector<float>* outs[Impl::kMaxB];
    for (std::size_t i = 0; i < sessions.size(); ++i)
      outs[i] = &sessions[i]->logits_;
    impl_->DecodeN(sessions.data(), tokens.data(), outs,
                   static_cast<int>(sessions.size()));
    return;
  }
  for (std::size_t i = 0; i < sessions.size(); ++i)
    impl_->Decode(*sessions[i], tokens[i], &sessions[i]->logits_);
}

void Engine::Synchronize() {
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  Check(hipStreamSynchronize(impl_->s), "synchronize");
}

void Engine::Forward(Session& s, std::span<const std::int32_t> tokens,
                     const LogitsSink& sink, const PrefillYield* yield,
                     std::span<const InputRows> rows) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (s.pos_ + tokens.size() > max_context_)
    throw std::runtime_error("inkling2: context exhausted");
  for (auto t : tokens)
    if (t < 0 || static_cast<std::uint32_t>(t) >= weights_.config.n_vocab_text)
      throw std::runtime_error("inkling2: token id out of range");
  Impl::YieldRun run;  // M2 hook: layered prefill
  Impl::YieldRun* yr = nullptr;
  if (yield != nullptr && yield->groups >= 2 && yield->yield) {
    const std::size_t ubatches = (tokens.size() + ubatch_ - 1) / ubatch_;
    run.spec = yield;
    run.lock = &lock;
    run.total = ubatches * 48;
    yr = &run;
  }
  for (const auto& r : rows)
    if (r.device == nullptr || r.first + r.count > tokens.size())
      throw std::runtime_error("inkling2: input rows outside the batch");
  struct RowsScope {  // the yield callback may run other sessions: rows belong
                      // to this call only
    Impl* i;
    std::span<const InputRows> saved;
    ~RowsScope() { i->input_rows = saved; }
  } rows_scope{impl_.get(), impl_->input_rows};
  impl_->input_rows = rows;
  if (tokens.size() == 1 && rows.empty()) {
    if (s.hooks_ != nullptr)
      s.hooks_->BeforeWrite(s.pos_, s.pos_ + 1);  // M2 hook
    s.BackRows(s.pos_, s.pos_ + 1);
    impl_->Decode(s, tokens[0], &s.logits_);
    if (sink)
      sink(0, 1, s.logits_.data());
    return;
  }
  std::size_t done = 0;
  while (done < tokens.size()) {
    const std::size_t n = std::min<std::size_t>(ubatch_, tokens.size() - done);
    if (s.hooks_ != nullptr)
      s.hooks_->BeforeWrite(s.pos_, s.pos_ + n);  // M2 hook
    s.BackRows(s.pos_, s.pos_ + n);
    impl_->Ubatch(s, tokens.subspan(done, n), static_cast<int>(done),
                  done + n == tokens.size(), sink, yr);
    done += n;
    if (yr != nullptr && done < tokens.size())
      impl_->MaybeYield(yr, false);  // between ubatches
  }
  Check(hipStreamSynchronize(impl_->s), "forward sync");
}

}  // namespace gufo::models::inkling2
