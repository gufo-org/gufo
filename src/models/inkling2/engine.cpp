#include "src/models/inkling2/engine.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <gufo/ext/experimental/state_memory.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <gufo/ext/platform/prefault.hpp>
#include <limits>
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

/// Device memory only: weights and scratch are read by every step, and a
/// managed fallback would put them in host RAM on Windows (separate from the
/// GPU's carve-out) without saying so.
void* AllocDevice(std::size_t bytes, const char* what) {
  void* p = nullptr;
  const hipError_t e = hipMalloc(&p, std::max<std::size_t>(bytes, 1));
  if (e != hipSuccess || p == nullptr) {
    (void)hipGetLastError();
    throw std::runtime_error(
        std::string("inkling2: cannot allocate ") + what + " (" +
        std::to_string(bytes) +
        " bytes of device memory): " + hipGetErrorString(e));
  }
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
  /// Reads the file through hip::WeightUpload, which streams it to the device
  /// without faulting the mapping: a copy through the mapping keeps the whole
  /// file resident in the process (on Windows in its working set), which the
  /// RAM prompt cache, sized from free host memory, then lacks.
  void Upload(const core::GgufReader& reader, const Inkling2Weights& weights,
              std::size_t* total) {
    std::size_t sum = 0;
    for (const auto& v : weights.widened) {  // F32 copies of BF16 small tensors
                                             // (not in the file mapping)
      const std::size_t bytes = v->size() * sizeof(float);
      void* dev = AllocDevice(bytes, "widened weights");
      regions_.push_back(
          {reinterpret_cast<std::uintptr_t>(v->data()), bytes, dev});
      Check(hipMemcpy(dev, v->data(), bytes, hipMemcpyHostToDevice),
            "weight upload");
      sum += bytes;
    }
    std::string error;
    auto upload =
        ::gufo::hip::WeightUpload::Create(reader.GetMappedRegions(), &error);
    if (!upload)
      throw std::runtime_error("inkling2: weight upload: " + error);
    bool ok = true;
    for (const auto& t : reader.GetTensors()) {
      if (t.size_bytes == 0)
        continue;
      void* dev = nullptr;
      try {
        dev = AllocDevice(t.size_bytes, std::string(t.name).c_str());
      } catch (...) {
        (void)upload->Finish(&error);  // queued copies still target regions_
        throw;
      }
      regions_.push_back(
          {reinterpret_cast<std::uintptr_t>(t.data), t.size_bytes, dev});
      sum += t.size_bytes;
      if (!upload->CopyMapped(t.data, t.size_bytes, dev, &error)) {
        ok = false;
        break;
      }
    }
    // Finish waits for every queued copy, also after a failed one.
    if (!upload->Finish(&error) || !ok)
      throw std::runtime_error("inkling2: weight upload: " + error);
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
  // Prefill side stream for projections that run beside larger ones.
  hipStream_t ps = nullptr;
  hipEvent_t ev_fork = nullptr, ev_join = nullptr, ev_bank = nullptr;
  // Cancellable passes: the last two visits' ends (see MaybeYield).
  hipEvent_t ev_visit[2] = {};
  // Decode steps: dense weights are read on a side stream while the
  // latency-bound mixes, controllers, router and attention run (the DRAM is
  // mostly idle then), so their GEMVs find them in the last-level cache: the
  // next visit's q projection (and its loop controllers) beside the expert
  // input and router, the gate projection beside the attention inputs, the
  // output projection and the router beside the attention. The expert GEMVs
  // between a prefetch and its use stream with DLC loads, which do not
  // allocate in that cache. Eight blocks (four loads in flight per thread)
  // reach the DRAM rate; more take CUs and bandwidth from the main chain.
  // Only where Engine::Create measured it to pay: on Windows/TheRock 10.0 the
  // side stream's cross-queue waits in the replayed graph cost more than the
  // prefetch saves (decode 39.4 tok/s with it, 46.0 without at C = 1).
  static constexpr int kPrefetchBlocks = 8;
  bool decode_prefetch = false;
  hipStream_t dps = nullptr;
  hipEvent_t ev_pf = nullptr, ev_pfj = nullptr;
  bool pf_open = false;
  void Prefetch(const fs::TouchList& l) {
    if (!decode_prefetch || l.n == 0)
      return;
    Check(hipEventRecord(ev_pf, ds), "prefetch fork");
    Check(hipStreamWaitEvent(dps, ev_pf, 0), "prefetch fork");
    fs::TouchN(dps, l, kPrefetchBlocks);
    pf_open = true;
  }
  void PrefetchJoin() {
    if (!pf_open)
      return;
    Check(hipEventRecord(ev_pfj, dps), "prefetch join");
    Check(hipStreamWaitEvent(ds, ev_pfj, 0), "prefetch join");
    pf_open = false;
  }
  hipEvent_t ev_msg[2] = {};
  // DSpark drafter (Engine::AttachDrafter) and the target features it reads:
  // the inputs of visit slots 5, 29 and 47 and the final normed output, [row]
  // [4][2304] (decode steps: d_feat; prefill ubatches: p_feat).
  std::unique_ptr<Drafter> dsp;
  float* d_feat = nullptr;
  float* p_feat = nullptr;
  static int FeatureIndex(int slot_idx) {
    return slot_idx == 5 ? 0 : slot_idx == 29 ? 1 : slot_idx == 47 ? 2 : -1;
  }
  struct UbatchCheckpoint {
    std::size_t index = 0;
    std::size_t position = 0;
    int local_t = 0;  // tokens from ubatch start (1..T)
    float* conv = nullptr;
    float* draft_f = nullptr;
    std::vector<float> logits;
  };
  std::vector<UbatchCheckpoint>* ubatch_cps = nullptr;
  // Interior checkpoint state for Engine::kMaxPassCheckpoints boundaries,
  // allocated once (convolution histories at Init, drafter features at
  // AttachDrafter). A prefill run inside another's yield finds it busy and
  // allocates its own.
  static constexpr std::size_t kCpConvFloats =
      static_cast<std::size_t>(48) * kConvChannels * 3;
  float* cp_conv = nullptr;
  float* cp_draft_f = nullptr;
  bool cp_busy = false;

  void CaptureFeature(hipStream_t st, const float* src, int rows, int f,
                      float* dst) {
    Check(hipMemcpy2DAsync(dst + static_cast<std::size_t>(f) * kEmbd,
                           Drafter::kFeatureFloats * 4, src, kEmbd * 4,
                           kEmbd * 4, static_cast<std::size_t>(rows),
                           hipMemcpyDeviceToDevice, st),
          "feature copy");
  }
  const float* msg_inv = nullptr;  // [4][T] inverse RMS of the visit's msgs
  DeviceWeights dw;
  std::vector<void*> scratch_allocs;
  std::array<float*, 24> msgq{};
  float* msgq_all = nullptr;

  int U = 512;
  int T = 0;     // tokens of the running ubatch
  int pos0 = 0;  // position of the first token of the running ubatch

  // activations (all [U][dim] F32 unless noted)
  float *cur, *prelude, *bankout;
  float* blocks[6];
  float* bank[4];
  float *mixed, *ctl[3], *in[3], *msgs, *msgs_next, *minv[2], *xinv, *gate64;
  float *Qb, *Kb, *Vb, *Rb, *Gb, *Sb, *Ob, *AOp, *AO, *h, *hn, *mixf, *rlog,
      *wts;
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
  /// k-interleaved BF16 copy of a controller gain/forget matrix [2304][64]
  /// (prefill apply + norm).
  std::map<const void*, void*> k4_of;
  const void* HW4(const TensorRef& t) const { return k4_of.at(t.data); }
  void MakeK4(const TensorRef& t) {
    if (t.size_bytes == 0 || k4_of.count(t.data))
      return;
    void* d = AllocDevice(t.size_bytes / 2, "ctl k4 weights");
    scratch_allocs.push_back(d);
    fs::CtlWeightsK4(s, DF(t), d, static_cast<int>(t.ne[1]));
    k4_of[t.data] = d;
  }
  /// BF16 copy of a BF16-exact F32 tensor on the device: the same values in
  /// half the bytes, without reading back from the host mapping.
  void MakeHalf(const TensorRef& t) {
    if (t.size_bytes == 0 || half_of.count(t.data))
      return;
    const std::size_t n = t.size_bytes / 4;
    void* d = AllocDevice(n * 2, "f16 weights");
    scratch_allocs.push_back(d);
    fs::F32ToBf16(s, DF(t), d, n);
    half_of[t.data] = d;
  }
  std::map<int, hipGraphExec_t> graphs;
  // Verification steps (one session, consecutive positions): keyed like
  // `graphs`; the conv kernels read in-step history and capture their inputs.
  std::map<int, hipGraphExec_t> vgraphs;
  bool verify_mode = false;
  // all rows of the verification are one session's (else several sessions'
  // rows, each run through the per-row attention kernel)
  bool verify_one_session = true;
  float* d_vcap = nullptr;  // [48][kVerifyRows][6656] conv inputs of a step
  // device argmax and GPU linear sampler workspaces
  float* d_amax_ws = nullptr;
  int* d_amax = nullptr;
  int* h_amax = nullptr;    // pinned
  float* h_rows = nullptr;  // pinned [kMaxB][vocab]: a verification's logits
  float* d_ls_temp = nullptr;
  float* d_ls_minp = nullptr;
  double* d_ls_u = nullptr;
  double* d_ls_ws = nullptr;
  int* d_ls_out = nullptr;
  float* h_ls_temp = nullptr;  // pinned
  float* h_ls_minp = nullptr;  // pinned
  double* h_ls_u = nullptr;    // pinned
  int* h_ls_out = nullptr;     // pinned
  /// The verification capture of slot `slot_idx` (null outside verification).
  float* VCap(int slot_idx) const {
    return verify_mode ? d_vcap + static_cast<std::size_t>(slot_idx) *
                                      kConvChannels * fs::kVerifyRows
                       : nullptr;
  }
  /// Each verification row's index within its session's rows (d_step's third
  /// plane; null outside verification).
  const int* VRows() const {
    return verify_mode ? d_step + 2 * kMaxB : nullptr;
  }
  std::set<int> failed_graphs;  // keys whose capture failed: those steps run
                                // eagerly, no retry
  float *d_cur, *d_prelude, *d_bankout, *d_msgs, *d_mixed, *d_ctl, *d_in,
      *d_g64, *d_proj, *d_sink, *d_dyn, *d_ob, *d_ao, *d_h, *d_mixf, *d_hn,
      *d_moe, *d_rlog, *d_wts8, *d_act8, *d_ws_o, *d_ws_m, *d_ws_l;
  // rows per decode step: sessions of a batch (up to kMaxDecodeBatch) or one
  // session's verification rows (up to kMaxVerifyRows)
  static constexpr int kMaxB = 64;
  // rows a weight GEMV or attention launch takes; wider steps run as
  // consecutive launches of at most this many rows (same per-row arithmetic)
  static constexpr int kChunk = 8;
  // steps of at least this many rows run the dense Q8 GEMVs through int8
  // WMMA (same outputs; slower for fewer rows, whose tiles are half empty)
  static constexpr int kMmaRows = kChunk + 1;
  // The LM head and the three-input (global-layer) projections take the int8
  // WMMA path from a full chunk: at 8 rows the head's per-row GEMV runs 4 to 7
  // waves per SIMD and streams about 150 GB/s, WMMA about 195 GB/s (the
  // one-input projections and the output matrix stay faster per row)
  static constexpr int kMmaHeadRows = kChunk;
  // verification steps from this many rows group the MoE pairs by expert
  // (consecutive rows of a session share most of their experts; several
  // sessions' rows share fewer)
  static constexpr int kGroupedVerifyRows = 4;
  static constexpr int kGroupedBatchVerifyRows = 6;
  static constexpr int kMsgInvTokens = 1024;
  // A visit of a longer ubatch also checks for cancellation between its
  // attention and expert halves: a pass then stops as soon as one of this
  // width does.
  static constexpr int kMidVisitCancelTokens = 2048;
  // From this ubatch size the loop controllers' apply also normalises the rows
  // (fewer bytes moved); below it the separate kernels have more blocks
  // (measured: pp64 -1%, pp256 equal, pp512 +0.7%).
  static constexpr int kFusedCtlNormTokens = 512;
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
  struct Visit {
    int ip;    // physical layer
    bool msg;  // a later pass's core visit (loop controllers on its inputs)
  };
  std::vector<Visit> d_visits;  // the decode step's visits (BuildDecode)
  int d_visit = 0;              // index of the visit DLayer builds
  int* d_ids8;
  fs::MoeGroups d_groups{};  // wide steps: pairs grouped by expert
  void* d_qa = nullptr;      // wide steps: quantized GEMV inputs
  float* d_qs = nullptr;
  int* d_step;  // {token, position, row within its session}
  unsigned* d_counter;
  unsigned* d_ccnt;     // fused controller apply + norm: [kMaxB][4]
  unsigned* d_moe_cnt;  // fused expert gate/up + down: [2][kMaxB * 8]
  fs::SlotEntry* d_tbl;
  const float* d_states[16];
  int d_ns = 0;
  // ---- layered prefill (spill/restore of the transient activations
  // around a yield) ----
  std::span<const InputRows>
      input_rows;  // set by Forward (under the engine mutex)
  struct YieldRun {
    const PrefillYield* spec = nullptr;
    std::unique_lock<std::mutex>* lock = nullptr;
    std::size_t total =
        0;  // visit units of the whole Forward call (48 per ubatch)
    std::size_t done = 0;
    std::size_t k = 1;       // next group boundary index
    std::size_t checks = 0;  // cancellation points passed
  };
  // The cancellable pass that is running (Ubatch), for the cancellation point
  // inside a visit.
  YieldRun* run = nullptr;
  /// A cancellation point. Kernels run far behind their launches (a pass is
  /// queued in a few milliseconds): the launches stay one point ahead of the
  /// device, so a cancellation stops the pass within about two points. The
  /// device stays busy; the arithmetic is unchanged.
  void CancelPoint(YieldRun* y) {
    if (y == nullptr || y->spec == nullptr || !y->spec->cancelled)
      return;
    ++y->checks;
    Check(hipEventRecord(ev_visit[y->checks & 1], s), "visit event");
    if (y->checks >= 2)
      Check(hipEventSynchronize(ev_visit[(y->checks - 1) & 1]), "visit wait");
    bool stop = true;
    try {
      stop = y->spec->cancelled();
    } catch (...) {
    }
    if (stop) {
      // No queued work of this pass may outlive it: the next Forward reuses
      // the scratch (both streams).
      Check(hipStreamSynchronize(s), "cancel sync");
      Check(hipStreamSynchronize(ps), "cancel sync");
      throw PrefillCancelled();
    }
  }
  // Live between two visits of a ubatch: cur, prelude, bankout, blocks[6],
  // bank[4]. Only a yield to another session spills them, so the area exists
  // from the first one (Engine::ReserveYieldSpill makes it at load).
  float* spill_buf[13] = {};
  void EnsureSpill() {
    if (spill_buf[0] != nullptr)
      return;
    const std::shared_lock<std::shared_mutex> gate(CaptureGate());
    for (auto& b : spill_buf)
      b = NewBuf(static_cast<std::size_t>(U) * kEmbd);
  }
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
    EnsureSpill();
    float* live[13];
    LiveBufs(live);
    const std::size_t rows = static_cast<std::size_t>(T) * kEmbd * 4;
    for (int i = 0; i < 13; ++i)
      Check(hipMemcpyAsync(spill_buf[i], live[i], rows, hipMemcpyDeviceToDevice,
                           s),
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
  }
  /// Called after every visit unit; yields at the group boundaries. `mid`:
  /// transient activations are live (inside a ubatch), so they are spilled
  /// around the yield.
  /// Returns true when it yielded (other sessions may then have used every
  /// scratch buffer that is not spilled).
  bool MaybeYield(YieldRun* y, bool mid) {
    if (y == nullptr || y->spec == nullptr)
      return false;
    ++y->done;
    CancelPoint(y);
    const std::size_t g = y->spec->groups;
    if (!y->spec->yield || g < 2 || y->done >= y->total || y->k >= g)
      return false;
    if (y->done < (y->k * y->total + g - 1) / g)
      return false;
    while (y->k < g && (y->k * y->total + g - 1) / g <= y->done)
      ++y->k;
    const int saved_T = T, saved_pos0 = pos0, saved_nblocks = nblocks;
    auto* const saved_cps = ubatch_cps;
    ubatch_cps = nullptr;
    if (mid)
      Spill();
    // The decode graphs run on another (non-blocking) stream and share scratch
    // (expert routing, logits) with this stream: drain it before the lock is
    // released.
    Check(hipStreamSynchronize(s), "yield sync");
    Check(hipStreamSynchronize(ps), "yield sync");
    y->lock->unlock();  // the decode round of other sessions takes the engine
                        // mutex
    try {
      y->spec->yield();
    } catch (...) {
      y->lock->lock();
      ubatch_cps = saved_cps;
      throw;
    }
    y->lock->lock();
    T = saved_T;
    pos0 = saved_pos0;
    nblocks = saved_nblocks;
    ubatch_cps = saved_cps;
    run = y;
    if (mid) {
      Unspill();
      // The dense-GEMM work items are per ubatch and shared: another session's
      // Forward replaced them.
      UploadDenseItems();
    }
    return true;
  }
  // ---- end layered prefill ----

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

  /// Destroys every captured decode and verification graph; later steps
  /// capture them again.
  void DropGraphs() {
    for (auto& g : graphs)
      (void)hipGraphExecDestroy(g.second);
    graphs.clear();
    for (auto& g : vgraphs)
      (void)hipGraphExecDestroy(g.second);
    vgraphs.clear();
    failed_graphs.clear();
  }

  ~Impl() {
    for (auto& g : graphs)
      (void)hipGraphExecDestroy(g.second);
    for (auto& g : vgraphs)
      (void)hipGraphExecDestroy(g.second);
    if (ds)
      (void)hipStreamDestroy(ds);
    if (ps)
      (void)hipStreamDestroy(ps);
    if (dps)
      (void)hipStreamDestroy(dps);
    for (hipEvent_t e : {ev_pf, ev_pfj})
      if (e)
        (void)hipEventDestroy(e);
    if (h_amax)
      (void)hipHostFree(h_amax);
    if (h_rows)
      (void)hipHostFree(h_rows);
    if (h_ls_temp)
      (void)hipHostFree(h_ls_temp);
    if (h_ls_minp)
      (void)hipHostFree(h_ls_minp);
    if (h_ls_u)
      (void)hipHostFree(h_ls_u);
    if (h_ls_out)
      (void)hipHostFree(h_ls_out);
    if (ev_fork)
      (void)hipEventDestroy(ev_fork);
    if (ev_join)
      (void)hipEventDestroy(ev_join);
    for (hipEvent_t e :
         {ev_bank, ev_msg[0], ev_msg[1], ev_visit[0], ev_visit[1]})
      if (e)
        (void)hipEventDestroy(e);
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
    cp_conv = NewBuf(Engine::kMaxPassCheckpoints * kCpConvFloats);
    prelude = NewBuf(u * kEmbd);
    bankout = NewBuf(u * kEmbd);
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
    msgs_next = NewBuf(u * 4 * kEmbd);
    minv[0] = NewBuf(u * 4);
    minv[1] = NewBuf(u * 4);
    xinv = NewBuf(u * 3);
    gate64 = NewBuf(u * 64 * 3);
    Qb = NewBuf(u * kQ);
    Kb = NewBuf(u * kKV);
    Vb = NewBuf(u * kKV);
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
    ye = NewBuf(u * 8 * kEmbd);
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
    // F32), otherwise as F32. Checked on the device so the host mapping is not
    // faulted back in.
    {
      unsigned* d_non_exact = reinterpret_cast<unsigned*>(NewInts(1));
      Check(hipMemsetAsync(d_non_exact, 0, sizeof(unsigned), s), "bf16 flag");
      auto check_t = [&](const TensorRef& t) {
        if (t.size_bytes == 0)
          return;
        if (static_cast<int>(t.type) != kk::kTF32) {
          ctl_half = false;
          return;
        }
        fs::CheckBf16Exact(s, DF(t), t.size_bytes / 4, d_non_exact);
      };
      for (std::uint32_t il = 0; il < 24 && ctl_half; ++il) {
        const LayerWeights& L = w->layers[il];
        check_t(L.ffn_gate_inp);
        for (const auto& c : L.loop) {
          check_t(c.down);
          check_t(c.gain);
          check_t(c.forget);
        }
      }
      if (ctl_half) {
        unsigned non_exact = 0;
        Check(hipMemcpyAsync(&non_exact, d_non_exact, sizeof(unsigned),
                             hipMemcpyDeviceToHost, s),
              "bf16 flag");
        Check(hipStreamSynchronize(s), "bf16 flag");
        if (non_exact != 0)
          ctl_half = false;
      }
    }
    if (ctl_half) {
      for (std::uint32_t il = 0; il < 24; ++il) {
        const LayerWeights& L = w->layers[il];
        MakeHalf(L.ffn_gate_inp);
        for (const auto& c : L.loop) {
          MakeHalf(c.down);
          MakeHalf(c.gain);
          MakeHalf(c.forget);
          MakeK4(c.gain);
          MakeK4(c.forget);
        }
      }
    }
    Check(hipStreamCreateWithFlags(&ds, hipStreamNonBlocking), "decode stream");
    Check(hipStreamCreateWithFlags(&ps, hipStreamNonBlocking), "side stream");
    Check(hipEventCreateWithFlags(&ev_fork, hipEventDisableTiming), "event");
    Check(hipStreamCreateWithFlags(&dps, hipStreamNonBlocking), "prefetch");
    Check(hipEventCreateWithFlags(&ev_pf, hipEventDisableTiming), "event");
    Check(hipEventCreateWithFlags(&ev_pfj, hipEventDisableTiming), "event");
    Check(hipEventCreateWithFlags(&ev_join, hipEventDisableTiming), "event");
    for (hipEvent_t* e :
         {&ev_bank, &ev_msg[0], &ev_msg[1], &ev_visit[0], &ev_visit[1]})
      Check(hipEventCreateWithFlags(e, hipEventDisableTiming), "event");
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
    // attention split workspaces: one launch's rows (kChunk)
    d_ws_o = NewBuf(kChunk * kMaxSplit * 8 * 12 * 128);
    d_ws_m = NewBuf(kChunk * kMaxSplit * 8 * 12);
    d_ws_l = NewBuf(kChunk * kMaxSplit * 8 * 12);
    for (auto& b : d_blk)
      b = NewBuf(B * kEmbd);
    d_moe_part = NewBuf(B * 8 * kEmbd);
    d_ids8 = NewInts(B * 8);
    d_qa = NewRaw(B * 3 * fs::kQaBytesPerVector);
    d_qs = NewBuf(B * 3 * fs::kQsFloatsPerVector);
    d_groups.sorted = NewInts(B * 8);
    d_groups.ue = NewInts(128);
    d_groups.ub = NewInts(128);
    d_groups.un = NewInts(128);
    d_groups.nd = NewInts(1);
    d_groups.qx = NewRaw(B * 8 * fs::kMoeQuantBytes);
    d_step = NewInts(3 * B);  // tokens [B], positions [B], row in session [B]
    d_vcap =
        NewBuf(static_cast<std::size_t>(48) * kConvChannels * fs::kVerifyRows);
    d_feat = NewBuf(B * Drafter::kFeatureFloats);
    d_amax_ws = NewBuf(B * 64 * 2);
    d_amax = NewInts(B);
    Check(hipHostMalloc(reinterpret_cast<void**>(&h_amax), B * sizeof(int)),
          "pinned");
    Check(hipHostMalloc(reinterpret_cast<void**>(&h_rows),
                        B * cfg->n_vocab_text * sizeof(float)),
          "pinned");
    d_ls_temp = NewBuf(B);
    d_ls_minp = NewBuf(B);
    d_ls_u = static_cast<double*>(AllocDevice(B * sizeof(double), "ls u"));
    scratch_allocs.push_back(d_ls_u);
    d_ls_ws =
        static_cast<double*>(AllocDevice(B * 129 * sizeof(double), "ls ws"));
    scratch_allocs.push_back(d_ls_ws);
    d_ls_out = NewInts(B);
    Check(
        hipHostMalloc(reinterpret_cast<void**>(&h_ls_temp), B * sizeof(float)),
        "pinned");
    Check(
        hipHostMalloc(reinterpret_cast<void**>(&h_ls_minp), B * sizeof(float)),
        "pinned");
    Check(hipHostMalloc(reinterpret_cast<void**>(&h_ls_u), B * sizeof(double)),
          "pinned");
    Check(hipHostMalloc(reinterpret_cast<void**>(&h_ls_out), B * sizeof(int)),
          "pinned");
    d_counter = reinterpret_cast<unsigned*>(NewInts(B));
    Check(hipMemset(d_counter, 0, B * 4), "counter");
    d_ccnt = reinterpret_cast<unsigned*>(NewInts(B * 4));
    Check(hipMemset(d_ccnt, 0, B * 4 * 4), "controller counters");
    d_moe_cnt = reinterpret_cast<unsigned*>(NewInts(2 * B * 8));
    Check(hipMemset(d_moe_cnt, 0, 2 * B * 8 * 4), "expert counters");
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
      // relative bias table: F32 [1024][16] -> f16 on device
      f.bt = AllocDevice(1024 * 16 * 2, "bias table");
      scratch_allocs.push_back(f.bt);
      fs::CvtF16(s, DF(L.attn_rel_bias), f.bt, 1024 * 16);
    }
    for (std::uint32_t il = 0; il < 24; ++il) {
      const LayerWeights& L = w->layers[il];
      LayerFast& f = lf[il];
      if (!f.tiny)
        continue;
      auto make = [&](const TensorRef& t, int rows) {
        auto* d = static_cast<float*>(
            AllocDevice(64 * 2304 * sizeof(float), "padded weights"));
        scratch_allocs.push_back(d);
        fs::PadRowsToF32(s, D(t), static_cast<int>(t.type), rows, 64, 2304, d);
        return d;
      };
      f.sink_f = make(L.attn_sink_proj, 24);
      if (L.is_global) {
        f.kconv_f = make(L.attn_k_conv_proj, 32);
        f.vconv_f = make(L.attn_v_conv_proj, 32);
      }
      f.tiny_pad = true;
    }
    Check(hipStreamSynchronize(s), "init decode weights");
    head_fast = IsType(w->output, kk::kTQ8_0);
    // The generic expert kernels stage F32 gate/up and activation rows: a whole
    // ubatch of pairs when a layer's prefill takes them, else only a decode
    // step's eight.
    const bool generic_moe =
        std::ranges::any_of(lf, [](const LayerFast& f) { return f.moe < 0; });
    const std::size_t pairs = generic_moe ? static_cast<std::size_t>(U) * 8 : 8;
    gu = NewBuf(pairs * 2 * static_cast<std::size_t>(ff));
    act = NewBuf(pairs * static_cast<std::size_t>(ff));
    moe = NewBuf((generic_moe ? static_cast<std::size_t>(U) : 1) * kEmbd);
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
  void FMix(const kk::MixStates& st, const float* query, int nq, float* out,
            float* inv_out = nullptr) {
    kk::Mix(s, st, query, nq, kEmbd, T, out, cfg->rms_eps, inv_out);
  }

  kk::MixStates States(const float* last, int extra_blocks_from = 0) {
    kk::MixStates st{};
    for (int i = extra_blocks_from; i < nblocks; ++i)
      st.ptr[st.n++] = blocks[i];
    st.ptr[st.n++] = last;
    return st;
  }

  /// Sink and (global layers) dynamic conv-tap projections from the padded
  /// F32 copies.
  void SkinnyProj(hipStream_t st, const LayerWeights& L, const LayerFast& F,
                  bool is_swa, const float* iq, const float* ik,
                  const float* iv) {
    (void)L;
    fs::SkinnyPrefill(st, iq, kEmbd, F.sink_f, 24, Sb, 24, T);
    if (!is_swa) {
      fs::SkinnyPrefill(st, ik, kEmbd, F.kconv_f, 32, dynK, 32, T);
      fs::SkinnyPrefill(st, iv, kEmbd, F.vconv_f, 32, dynV, 32, T);
    }
  }

  // ---- one layer visit ----
  /// `side`: work for the side stream, enqueued beside the MoE gate/up GEMM.
  void Layer(Session& ss, int ip, int slot_idx,
             const float* msg /*[T][4][2304] or null*/,
             const std::function<void()>& side = {}) {
    const LayerWeights& L = w->layers[ip];
    const LayerFast& F = lf[ip];
    const bool is_swa = !L.is_global;
    Session::Slot& slot = ss.slots_[slot_idx];
    if (dsp && FeatureIndex(slot_idx) >= 0)
      CaptureFeature(s, cur, T, FeatureIndex(slot_idx), p_feat);
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
        // The controllers that follow read the mixes' inverse RMS (summed as
        // their gate prologue would) instead of re-reading the rows.
        FMix(States(cur), DF(L.attn_res_query), nq, mixed,
             msg ? xinv : nullptr);
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
            inp[j].p,
            inp[j].ld,
            msg + j * kEmbd,
            4 * kEmbd,
            HW(c.down),
            HW(c.gain),
            HW(c.forget),
            gate64 + static_cast<std::size_t>(j) * U * 64,
            ctl[j],
            msg_inv ? msg_inv + static_cast<std::size_t>(j) * T : nullptr,
            xinv + static_cast<std::size_t>(is_swa ? 0 : j) * T,
            ctl_half ? 1 : 0};
      }
      // The input norms run inside the controllers' apply (same values).
      if (ctl_half && F.proj && T >= kFusedCtlNormTokens) {
        for (int j = 0; j < 3; ++j) {
          cj.j[j].nw = DF(is_swa   ? L.attn_norm
                          : j == 0 ? L.attn_q_inp_norm
                          : j == 1 ? L.attn_k_inp_norm
                                   : L.attn_v_inp_norm);
          // Only the sink projection reads an F32 window-layer input (q's).
          cj.j[j].nout = is_swa && j ? nullptr : in[j];
          cj.j[j].nout16 = in16[j];
          const LoopController& c = L.loop[kStreamQ + j];
          cj.j[j].gain4 = HW4(c.gain);
          cj.j[j].forget4 = HW4(c.forget);
        }
        normed = true;
        have16 = true;
      }
      fs::CtlPrefill(s, cj, T, cfg->rms_eps);
      for (int j = 0; j < 3; ++j)
        inp[j] = {ctl[j], kEmbd};
    }
    const float* iq;
    const float* ik;
    const float* iv;
    if (normed && msg) {
      iq = in[0];
      ik = in[1];
      iv = in[2];
    } else if (normed) {
      iq = in[0];
      ik = is_swa ? in[0] : in[1];
      iv = is_swa ? in[0] : in[2];
    } else if (is_swa) {
      const float* nw = DF(L.attn_norm);
      if (msg) {
        // Only the sink projection reads an F32 window-layer input (q's).
        for (int j = 0; j < 3; ++j)
          kk::RmsNorm(s, inp[j].p, inp[j].ld, nw, F.proj && j ? nullptr : in[j],
                      kEmbd, kEmbd, T, cfg->rms_eps,
                      F.proj ? in16[j] : nullptr);
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
      auto mm = [&](hipStream_t st, const TensorRef& wt, const void* a,
                    float* out) {
        fs::GemmDense(st, D(wt), static_cast<int>(wt.ne[1]),
                      static_cast<int>(wt.ne[0]), a, out, DenseItems());
      };
      // q and gate fill the GPU; k, v, rel and the skinny projections have
      // few blocks each and run beside them on the side stream.
      Check(hipEventRecord(ev_fork, s), "fork");
      Check(hipStreamWaitEvent(ps, ev_fork, 0), "fork");
      mm(ps, L.attn_k, a_k, Kb);
      mm(ps, L.attn_v, a_v, Vb);
      mm(ps, L.attn_rel, a_q, Rb);
      if (F.tiny_pad)
        SkinnyProj(ps, L, F, is_swa, iq, ik, iv);
      Check(hipEventRecord(ev_join, ps), "join");
      mm(s, L.attn_q, a_q, Qb);
      mm(s, L.attn_gate, a_q, Gb);
      Check(hipStreamWaitEvent(s, ev_join, 0), "join");
    } else {
      G(L.attn_q, iq, kEmbd, Qb, kQ, T);
      G(L.attn_k, ik, kEmbd, Kb, kKV, T);
      G(L.attn_v, iv, kEmbd, Vb, kKV, T);
      G(L.attn_rel, iq, kEmbd, Rb, kRel, T);
      G(L.attn_gate, iq, kEmbd, Gb, kQ, T);
    }
    const float* dk = is_swa ? nullptr : dynK;
    const float* dv = is_swa ? nullptr : dynV;
    if (F.tiny_pad) {
      if (!fp)
        SkinnyProj(s, L, F, is_swa, iq, ik, iv);
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

    // C/D. clips, KV write, attention
    {
      fs::PrefillAttn pa{};
      pa.n = T;
      pa.pos0 = pos0;
      pa.cap = static_cast<int>(slot.rows);
      pa.swa = is_swa ? 1 : 0;
      // The K/V short convolutions run inside the cache write; the histories
      // advance after it.
      pa.kraw = Kb;
      pa.vraw = Vb;
      pa.kst = cs + kConvOffK * 3;
      pa.vst = cs + kConvOffV * 3;
      pa.kw = DF(L.attn_k_conv);
      pa.vw = DF(L.attn_v_conv);
      pa.kdyn = dk;
      pa.vdyn = dv;
      pa.conv_scale = conv_scale;
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
      if (ubatch_cps != nullptr) {
        for (const auto& cp : *ubatch_cps) {
          if (cp.conv == nullptr)
            continue;
          float* dst =
              cp.conv + static_cast<std::size_t>(slot_idx) * kConvChannels * 3;
          kk::ConvStateCapture(s, Kb, cs + kConvOffK * 3, dst + kConvOffK * 3,
                               cp.local_t, kKV);
          kk::ConvStateCapture(s, Vb, cs + kConvOffV * 3, dst + kConvOffV * 3,
                               cp.local_t, kKV);
        }
      }
      kk::ConvStateUpdate(s, Kb, cs + kConvOffK * 3, T, kKV);
      kk::ConvStateUpdate(s, Vb, cs + kConvOffV * 3, T, kKV);
    }
    if (F.oproj) {
      fs::GemmDense(s, D(L.attn_output), kEmbd, kQ, p_ob16, AOp, DenseItems());
    } else {
      G(L.attn_output, Ob, kQ, AOp, kEmbd, T);
    }
    // E. residual / block bookkeeping
    if (ubatch_cps != nullptr) {
      for (const auto& cp : *ubatch_cps) {
        if (cp.conv == nullptr)
          continue;
        float* dst =
            cp.conv + static_cast<std::size_t>(slot_idx) * kConvChannels * 3;
        kk::ConvStateCapture(s, AOp, cs + kConvOffAttn * 3,
                             dst + kConvOffAttn * 3, cp.local_t, kEmbd);
      }
    }
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
    if (T > kMidVisitCancelTokens)
      CancelPoint(run);

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
      FMix(States(cur), DF(L.ffn_res_query), 1, mixf, msg ? xinv : nullptr);
    }
    if (msg) {
      const LoopController& c = L.loop[kStreamFfn];
      fs::CtlPJobs cj{};
      cj.n = 1;
      cj.j[0] = {mixf,
                 kEmbd,
                 msg + 3 * kEmbd,
                 4 * kEmbd,
                 HW(c.down),
                 HW(c.gain),
                 HW(c.forget),
                 gate64,
                 ctl[0],
                 msg_inv ? msg_inv + 3 * static_cast<std::size_t>(T) : nullptr,
                 xinv,
                 ctl_half ? 1 : 0};
      if (ctl_half && F.moe >= 0 && T >= kFusedCtlNormTokens) {
        cj.j[0].nw = DF(L.ffn_norm);
        cj.j[0].nout = h;
        cj.j[0].nout16 = p_h16;
        cj.j[0].gain4 = HW4(c.gain);
        cj.j[0].forget4 = HW4(c.forget);
        h_done = true;
      }
      fs::CtlPrefill(s, cj, T, cfg->rms_eps);
      hin = ctl[0];
    }
    if (!h_done)
      kk::RmsNorm(s, hin, kEmbd, DF(L.ffn_norm), h, kEmbd, kEmbd, T,
                  cfg->rms_eps, F.moe >= 0 ? p_h16 : nullptr);

    // G. MoE
    if (Moe(L, lf[ip], side)) {
      if (ubatch_cps != nullptr) {
        for (const auto& cp : *ubatch_cps) {
          if (cp.conv == nullptr)
            continue;
          float* dst =
              cp.conv + static_cast<std::size_t>(slot_idx) * kConvChannels * 3;
          fs::MoeTailStateCapture(s, ye, cs + kConvOffFfn * 3,
                                  dst + kConvOffFfn * 3, cp.local_t, kEmbd);
        }
      }
      fs::MoeTail(s, ye, cs + kConvOffFfn * 3, DF(L.ffn_post_conv), cur, T,
                  kEmbd, 1.0f / std::sqrt(5.0f));
    } else {
      if (ubatch_cps != nullptr) {
        for (const auto& cp : *ubatch_cps) {
          if (cp.conv == nullptr)
            continue;
          float* dst =
              cp.conv + static_cast<std::size_t>(slot_idx) * kConvChannels * 3;
          kk::ConvStateCapture(s, moe, cs + kConvOffFfn * 3,
                               dst + kConvOffFfn * 3, cp.local_t, kEmbd);
        }
      }
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
  bool Moe(const LayerWeights& L, const LayerFast& F,
           const std::function<void()>& side) {
    const int E = static_cast<int>(cfg->n_expert),
              K = static_cast<int>(cfg->n_expert_used);
    const int nff = static_cast<int>(cfg->n_ff_expert);
    fs::RouterPrefill(s, HW(L.ffn_gate_inp), h, rlog, T, ctl_half);
    if (K == 8 && E == 128)
      fs::TopkPrefill(s, rlog, DF(L.exp_probs_b), ids_d, wts, T,
                      cfg->expert_weights_scale);
    else
      kk::RouterTopK(s, rlog, DF(L.exp_probs_b), ids_d, wts, T, E, K,
                     cfg->expert_weights_scale);
    const int P = T * K;
    if (F.moe >= 0) {
      const int tile = T >= 512 ? 128 : 32;
      const int max_items = (P + tile - 1) / tile + E;
      fs::MoeRoute(s, ids_d, wts, P, tile, mws);  // h's F16 copy is in p_h16
      if (side)
        side();
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

    // The visit before this one read q (and the controllers) beside its
    // expert input; the first one reads q now.
    const int vi = d_visit++;
    if (F.proj) {
      fs::TouchList tl;
      tl.Add(vi == 0 ? D(L.attn_q) : D(L.attn_gate),
             vi == 0 ? L.attn_q.size_bytes : L.attn_gate.size_bytes);
      Prefetch(tl);
    }
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
    // The input of this visit, now that the mix applied the previous visit's
    // deferred residual.
    if (dsp && FeatureIndex(slot_idx) >= 0)
      CaptureFeature(ds, cur, nb, FeatureIndex(slot_idx), d_feat);
    fs::CtlJobs cj{};
    if (msg) {
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
      if (msg)  // the controllers' outputs, normed in the same launch
        fs::CtlApplyRms(ds, cj, nj, d_ccnt, cfg->rms_eps, nb);
      else
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
      const int ni = three ? 3 : 1;
      if (nb >= (three ? kChunk : kMmaRows)) {
        // one weight read for all rows, int8 WMMA
        fs::QuantRows(ds, kEmbd, d_in, ni, kSIn, nb, d_qa, d_qs);
        if (!fs::ProjMma(ds, ps, d_qa, d_qs, ni, d_proj, nb, kSIn, kSProj,
                         d_in))
          throw std::runtime_error("inkling2: projection layout");
      } else
        for (int c0 = 0; c0 < nb; c0 += kChunk) {
          fs::ProjSpec pc = ps;
          for (int i = 0; i < pc.tn; ++i)
            pc.ty[i] += static_cast<std::size_t>(c0) * pc.tys[i];
          fs::ProjMulti(ds, pc, d_in + static_cast<std::size_t>(c0) * kSIn,
                        three ? 3 : 1,
                        d_proj + static_cast<std::size_t>(c0) * kSProj,
                        std::min(kChunk, nb - c0), kSIn, kSProj);
        }
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

    if (F.oproj) {  // the output projection and the router, read during the
                    // attention
      fs::TouchList tl;
      tl.Add(D(L.attn_output), L.attn_output.size_bytes);
      if (ctl_half)
        tl.Add(HW(L.ffn_gate_inp), L.ffn_gate_inp.size_bytes / 2);
      Prefetch(tl);
    }
    // C. conv + clip + KV write (in the attention launch for distinct
    // sessions' rows)
    fs::KvPrepArgs ka{};
    {
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
      ka.cap = VCap(slot_idx);
      ka.ri = VRows();
      if (verify_mode)
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
      if (!verify_mode) {
        fs::KvAttnDecode(ds, ka, aa, nb);
      } else {
        for (int c0 = 0; c0 < nb; c0 += kChunk) {
          const int cn = std::min(kChunk, nb - c0);
          const std::size_t o = static_cast<std::size_t>(c0);
          fs::AttnArgs ac = aa;
          ac.slot = aa.slot + o * 48;
          ac.pos = aa.pos + o;
          ac.q = aa.q + o * kSProj;
          ac.r = aa.r + o * kSProj;
          ac.gate = aa.gate + o * kSProj;
          ac.sink = aa.sink + o * 24;
          ac.out = aa.out + o * kQ;
          if (verify_one_session && !swa && cn > 1)
            fs::AttnDecodeRows(ds, ac, cn);
          else
            fs::AttnDecode(ds, ac, cn);
        }
      }
    }
    if (F.oproj)
      if (nb >= kMmaRows) {
        fs::QuantRows(ds, kQ, d_ob, 1, kQ, nb, d_qa, d_qs);
        fs::GemvMma(ds, kQ, D(L.attn_output), d_qa, d_qs, d_ao, kEmbd, 0.f, nb);
      } else
        for (int c0 = 0; c0 < nb; c0 += kChunk)
          fs::GemvQ8(ds, kQ, D(L.attn_output),
                     d_ob + static_cast<std::size_t>(c0) * kQ,
                     d_ao + static_cast<std::size_t>(c0) * kEmbd, kEmbd, 0.f,
                     std::min(kChunk, nb - c0));
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
      ca.cap = VCap(slot_idx);
      ca.ri = VRows();
      if (ip % static_cast<int>(cfg->attn_res_block) == 0) {
        ca.blk = NewBlock();
        d_states[d_ns++] = ca.blk;
      }
      pend.on = true;
      pend.a = ca;
    }

    // The next visit's q projection and attention-input controllers, read
    // while this visit's expert input and router run.
    if (vi + 1 < static_cast<int>(d_visits.size()) &&
        lf[d_visits[vi + 1].ip].proj) {
      const LayerWeights& N = w->layers[d_visits[vi + 1].ip];
      fs::TouchList tl;
      tl.Add(D(N.attn_q), N.attn_q.size_bytes);
      if (ctl_half && d_visits[vi + 1].msg)
        for (int j = 0; j < 3; ++j) {
          const LoopController& c = N.loop[kStreamQ + j];
          tl.Add(HW(c.down), c.down.size_bytes / 2);
          tl.Add(HW(c.gain), c.gain.size_bytes / 2);
          tl.Add(HW(c.forget), c.forget.size_bytes / 2);
        }
      Prefetch(tl);
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
      fs::NormJobs nj{};
      nj.n = 1;
      nj.x[0] = d_ctl;
      nj.w[0] = DS(L.ffn_norm);
      nj.out[0] = d_h;
      nj.xs[0] = kSCtl;
      nj.os[0] = kEmbd;
      fs::CtlApplyRms(ds, cj, nj, d_ccnt, cfg->rms_eps, nb);
    }

    // G. MoE
    const int E = static_cast<int>(cfg->n_expert);
    const int nff = static_cast<int>(cfg->n_ff_expert);
    fs::Router(ds, HW(L.ffn_gate_inp), DS(L.exp_probs_b), d_h, d_rlog,
               d_counter, d_ids8, d_wts8, cfg->expert_weights_scale, nb,
               ctl_half);
    // Many rows, a full chunk, or four or more consecutive rows of one session
    // (which share most experts; six of several sessions): each used expert's
    // weights read once for all its pairs. Fewer rows, and other sessions'
    // decode rows (which share fewer experts), run per pair.
    if (F.moe >= 0 &&
        (nb >= kChunk ||
         (verify_mode &&
          nb >= (verify_one_session ? kGroupedVerifyRows
                                    : kGroupedBatchVerifyRows)))) {
      const int slots = std::min(E, 8 * nb);
      fs::MoeRouteGroup(ds, d_ids8, 8 * nb, d_groups);
      fs::MoeQuantGrouped(ds, F.moe, d_h, nb, kEmbd, d_groups);
      fs::MoeGuGrouped(ds, F.moe, D(L.ffn_gate_up_exps),
                       L.ffn_gate_up_scale ? DS(L.ffn_gate_up_scale) : nullptr,
                       d_groups, slots, d_act8, nff, cfg->situ_beta,
                       cfg->situ_linear_beta);
      fs::MoeQuantGrouped(ds, F.moe, d_act8, 8 * nb, nff, d_groups);
      fs::MoeDownGrouped(ds, F.moe, D(L.ffn_down_exps),
                         L.ffn_down_scale ? DS(L.ffn_down_scale) : nullptr,
                         d_groups, slots, d_wts8, d_moe_part, kEmbd);
    } else if (F.moe >= 0 &&
               fs::MoeGuDown(
                   ds, F.moe, D(L.ffn_gate_up_exps),
                   L.ffn_gate_up_scale ? DS(L.ffn_gate_up_scale) : nullptr,
                   D(L.ffn_down_exps),
                   L.ffn_down_scale ? DS(L.ffn_down_scale) : nullptr, d_h,
                   d_ids8, d_wts8, d_act8, d_moe_part, d_moe_cnt, nff, kEmbd,
                   cfg->situ_beta, cfg->situ_linear_beta, nb)) {
      // gate/up and down in one launch
    } else if (F.moe >= 0) {
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
      ca.cap = VCap(slot_idx);
      ca.ri = VRows();
      pend.on = true;
      pend.a = ca;
    }
  }

  /// Enqueues the whole decode step on `ds` (captured or run directly).
  void BuildDecode(int bucket_ctx) {
    float* cur = d_cur;
    pend.on = false;
    // the visits in order (DLayer prefetches the next one's weights)
    d_visits.clear();
    d_visit = 0;
    const int cs = static_cast<int>(cfg->core_start),
              ce = static_cast<int>(cfg->core_end);
    for (int ip = 0; ip < cs; ++ip)
      d_visits.push_back({ip, false});
    for (int pass = 0; pass < static_cast<int>(cfg->n_passes); ++pass)
      for (int ip = cs; ip < ce; ++ip)
        d_visits.push_back({ip, pass > 0});
    for (int ip = ce; ip < static_cast<int>(cfg->n_phys); ++ip)
      d_visits.push_back({ip, false});
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
    if (dsp)
      CaptureFeature(ds, d_hn, nb, 3, d_feat);
    PrefetchJoin();
    const int V = static_cast<int>(cfg->n_vocab_text);
    if (head_fast) {
      if (nb >= kMmaHeadRows) {
        fs::QuantRows(ds, kEmbd, d_hn, 1, kEmbd, nb, d_qa, d_qs);
        fs::GemvMma(ds, kEmbd, D(w->output), d_qa, d_qs, logits_d, V,
                    cfg->logit_softcap, nb);
      } else
        for (int c0 = 0; c0 < nb; c0 += kChunk)
          fs::GemvQ8(ds, kEmbd, D(w->output),
                     d_hn + static_cast<std::size_t>(c0) * kEmbd,
                     logits_d + static_cast<std::size_t>(c0) * V, V,
                     cfg->logit_softcap, std::min(kChunk, nb - c0));
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
    auto& map = verify_mode ? vgraphs : graphs;
    // Exclusive: no cache or vision thread may issue a device call while the
    // capture is open.
    const std::unique_lock<std::shared_mutex> gate(CaptureGate());
    if (hipStreamBeginCapture(ds, hipStreamCaptureModeThreadLocal) !=
        hipSuccess) {
      (void)hipGetLastError();
      if (failed_graphs.insert(verify_mode ? -key : key).second)
        std::fprintf(
            stderr,
            "inkling2: decode graph capture could not start for context bucket "
            "%d, width %d; those steps run without a graph\n",
            ctx, key % 128);
      return map.end();
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
      if (failed_graphs.insert(verify_mode ? -key : key).second)
        std::fprintf(stderr,
                     "inkling2: decode graph capture failed for context bucket "
                     "%d, width %d; those steps run without a graph\n",
                     ctx, key % 128);
      return map.end();
    }
    return map.emplace(key, ex).first;
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

  void LaunchLinearSample(std::span<const Engine::LinearSample> linear, int n) {
    if (linear.empty())
      return;
    for (int r = 0; r < n; ++r) {
      h_ls_temp[r] = linear[r].temperature;
      h_ls_minp[r] = linear[r].min_p;
      h_ls_u[r] = linear[r].u;
    }
    Check(hipMemcpyAsync(d_ls_temp, h_ls_temp,
                         static_cast<std::size_t>(n) * sizeof(float),
                         hipMemcpyHostToDevice, ds),
          "ls temp");
    Check(hipMemcpyAsync(d_ls_minp, h_ls_minp,
                         static_cast<std::size_t>(n) * sizeof(float),
                         hipMemcpyHostToDevice, ds),
          "ls minp");
    Check(hipMemcpyAsync(d_ls_u, h_ls_u,
                         static_cast<std::size_t>(n) * sizeof(double),
                         hipMemcpyHostToDevice, ds),
          "ls u");
    fs::LinearSampleArgs la{};
    la.logits = logits_d;
    la.rows = n;
    la.vocab = static_cast<int>(cfg->n_vocab_text);
    la.temp = d_ls_temp;
    la.min_p = d_ls_minp;
    la.u = d_ls_u;
    la.ws = d_ls_ws;
    la.out = d_ls_out;
    fs::RowsLinearSample(ds, la);
    Check(hipMemcpyAsync(h_ls_out, d_ls_out,
                         static_cast<std::size_t>(n) * sizeof(int),
                         hipMemcpyDeviceToHost, ds),
          "ls out");
  }

  /// One decode step of `n` (<= kMaxB) distinct sessions, one token each, as
  /// one graph launch: the weights are read once for all tokens.
  void DecodeN(Session* const* sess, const std::int32_t* tokens,
               std::vector<float>* const* outs, int n,
               std::int32_t* step_argmax = nullptr,
               const bool* pinned = nullptr,
               std::span<const Engine::LinearSample> linear = {},
               std::int32_t* step_sampled = nullptr) {
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
    const int key = ctx * 128 + n + (decode_prefetch ? (1 << 29) : 0);
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
      if (outs[b] == nullptr) {  // device logits (Engine::Step)
        Check(hipMemcpyAsync(sess[b]->dlogits_, logits_d + b * V, V * 4,
                             hipMemcpyDeviceToDevice, ds),
              "logits copy");
        sess[b]->logits_stale_ = true;
        if (pinned != nullptr && pinned[b])
          Check(hipMemcpyAsync(h_rows + b * V, logits_d + b * V, V * 4,
                               hipMemcpyDeviceToHost, ds),
                "logits copy");
        continue;
      }
      outs[b]->resize(V);
      Check(hipMemcpyAsync(outs[b]->data(), logits_d + b * V, V * 4,
                           hipMemcpyDeviceToHost, ds),
            "logits copy");
      sess[b]->logits_stale_ = false;
    }
    if (step_argmax != nullptr) {
      fs::RowsArgmax(ds, logits_d, n, static_cast<int>(V), d_amax_ws, d_amax);
      Check(hipMemcpyAsync(h_amax, d_amax, static_cast<std::size_t>(n) * 4,
                           hipMemcpyDeviceToHost, ds),
            "argmax copy");
    }
    if (!linear.empty() && step_sampled != nullptr)
      LaunchLinearSample(linear, n);
    if (dsp) {
      Drafter::Rows rows[kMaxB];
      for (int b = 0; b < n; ++b)
        rows[b] = {.st = &sess[b]->draft_,
                   .feats = d_feat + static_cast<std::size_t>(b) *
                                         Drafter::kFeatureFloats,
                   .pos0 = sess[b]->pos_,
                   .n = 1};
      dsp->InjectRows(ds, std::span<const Drafter::Rows>(rows, n));
    }
    Check(hipStreamSynchronize(ds), "decode sync");
    if (step_argmax != nullptr)
      std::memcpy(step_argmax, h_amax, static_cast<std::size_t>(n) * 4);
    if (!linear.empty() && step_sampled != nullptr)
      std::memcpy(step_sampled, h_ls_out, static_cast<std::size_t>(n) * 4);
    for (int b = 0; b < n; ++b)
      sess[b]->pos_ += 1;
    nb = 1;
  }

  /// One verification step: for each of `ng` distinct sessions, counts[g]
  /// tokens at its next positions (rows of a session consecutive, sessions in
  /// order; at most kMaxB rows in all), every row with the arithmetic of a
  /// one-token decode at its position (the caller keeps each session's rows
  /// inside one attention bucket). No session advances; CommitVerify keeps a
  /// prefix of a session's rows.
  void VerifyN(Session* const* sess, const int* counts, int ng,
               const std::int32_t* tokens, std::span<const float>* out,
               std::int32_t* argmax,
               std::span<const Engine::LinearSample> linear = {},
               std::int32_t* sampled = nullptr) {
    int n = 0;
    std::int32_t step[3 * kMaxB] = {};
    int maxpos = 0;
    for (int g = 0; g < ng; ++g) {
      const int pos = static_cast<int>(sess[g]->pos_);
      if (BucketCtx(pos + counts[g] - 1) != BucketCtx(pos))
        throw std::runtime_error("inkling2: verification crosses a bucket");
      for (int j = 0; j < counts[g]; ++j, ++n) {
        step[n] = tokens[n];
        step[kMaxB + n] = pos + j;
        step[2 * kMaxB + n] = j;
      }
      maxpos = std::max(maxpos, pos + counts[g] - 1);
    }
    nb = n;
    const int ctx = BucketCtx(maxpos);
    Check(hipStreamSynchronize(s), "pre-verify sync");
    Check(hipMemcpyAsync(d_step, step, sizeof step, hipMemcpyHostToDevice, ds),
          "step");
    for (int g = 0, b = 0; g < ng; ++g)
      for (int j = 0; j < counts[g]; ++j, ++b)
        Check(hipMemcpyAsync(d_tbl + b * 48, sess[g]->tbl_,
                             48 * sizeof(fs::SlotEntry),
                             hipMemcpyDeviceToDevice, ds),
              "slot table");
    verify_mode = true;
    verify_one_session = ng == 1;
    struct Off {
      bool* f;
      bool* one;
      ~Off() {
        *f = false;
        *one = true;
      }
    } off{&verify_mode, &verify_one_session};
    // Several sessions take the per-row attention kernel: their own graphs.
    const int key = ctx * 128 + n + (ng > 1 ? (1 << 28) : 0) +
                    (decode_prefetch ? (1 << 29) : 0);
    const int fkey = -key;  // failed verification captures
    auto it = vgraphs.find(key);
    if (it == vgraphs.end() && failed_graphs.count(fkey) == 0) {
      it = CaptureGraph(ctx, key);
      if (it == vgraphs.end())
        failed_graphs.insert(fkey);
    }
    if (it != vgraphs.end())
      Check(hipGraphLaunch(it->second, ds), "graph launch");
    else
      BuildDecode(ctx);
    const std::size_t V = cfg->n_vocab_text;
    if (out != nullptr) {
      Check(
          hipMemcpyAsync(h_rows, logits_d, static_cast<std::size_t>(n) * V * 4,
                         hipMemcpyDeviceToHost, ds),
          "logits copy");
      *out = {h_rows, static_cast<std::size_t>(n) * V};
    }
    if (argmax != nullptr) {
      fs::RowsArgmax(ds, logits_d, n, static_cast<int>(V), d_amax_ws, d_amax);
      Check(hipMemcpyAsync(h_amax, d_amax, static_cast<std::size_t>(n) * 4,
                           hipMemcpyDeviceToHost, ds),
            "argmax copy");
    }
    if (!linear.empty() && sampled != nullptr)
      LaunchLinearSample(linear, n);
    Check(hipStreamSynchronize(ds), "verify sync");
    if (argmax != nullptr)
      std::memcpy(argmax, h_amax, static_cast<std::size_t>(n) * 4);
    if (!linear.empty() && sampled != nullptr)
      std::memcpy(sampled, h_ls_out, static_cast<std::size_t>(n) * 4);
    nb = 1;
  }

  /// Keeps the first `keep` rows of the session's verification rows, which
  /// start at step row `row0`.
  /// Keeps the first keep[g] rows of session g's verification rows, which
  /// start at step row row0[g].
  void CommitVerify(Session* const* ss, const int* keep, const int* row0,
                    int ng) {
    Drafter::Rows rows[kMaxB];
    for (int g = 0; g < ng; ++g) {
      fs::CommitVerifyConv(ds, static_cast<const fs::SlotEntry*>(ss[g]->tbl_),
                           d_vcap, keep[g], row0[g]);
      rows[g] = {.st = &ss[g]->draft_,
                 .feats = d_feat + static_cast<std::size_t>(row0[g]) *
                                       Drafter::kFeatureFloats,
                 .pos0 = ss[g]->pos_,
                 .n = keep[g]};
    }
    if (dsp)
      dsp->InjectRows(ds, std::span<const Drafter::Rows>(rows, ng));
    Check(hipStreamSynchronize(ds), "commit sync");
    for (int g = 0; g < ng; ++g)
      ss[g]->pos_ += static_cast<std::size_t>(keep[g]);
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
    run = yr;
    Check(hipMemcpyAsync(tok_d, tok.data(), tok.size() * 4,
                         hipMemcpyHostToDevice, s),
          "token copy");
    kk::EmbedRows(s, static_cast<int>(w->token_embd.type), D(w->token_embd),
                  kEmbd, tok_d, T, cur, kEmbd);
    kk::RmsNorm(s, cur, kEmbd, DF(w->token_embd_norm), cur, kEmbd, kEmbd, T,
                cfg->rms_eps);
    for (const InputRows& r :
         input_rows) {  // image rows replace the (normed) embeddings
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
    nblocks = 0;
    UploadDenseItems();
    int visit = 0;
    for (int ip = 0; ip < static_cast<int>(cfg->core_start); ++ip) {
      Layer(ss, ip, ip, nullptr);
      Tap(visit++, "l_out");
      MaybeYield(yr, true);
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
      // A layer's messages depend only on the bank: the next layer's are
      // mixed on the side stream while this layer runs (two buffers).
      float* mbuf[2] = {msgs, msgs_next};
      auto launch_msgs = [&](int il) {
        const int b = (il - static_cast<int>(cfg->core_start)) & 1;
        Check(hipEventRecord(ev_bank, s), "msgs");
        Check(hipStreamWaitEvent(ps, ev_bank, 0), "msgs");
        kk::MixStates st{};
        for (int i = 0; i < bank_n; ++i)
          st.ptr[st.n++] = bank[i];
        st.ptr[st.n++] = bankout;
        // Long ubatches also take the messages' inverse RMS from the mix
        // (the controllers' gates then skip that half of their prologue); on
        // short ones the longer side-stream chain would delay the next layer.
        kk::Mix(ps, st, msgq[il], 4, kEmbd, T, mbuf[b], cfg->rms_eps,
                T >= kMsgInvTokens ? minv[b] : nullptr);
        Check(hipEventRecord(ev_msg[b], ps), "msgs");
      };
      if (pass > 0)
        launch_msgs(static_cast<int>(cfg->core_start));
      for (int ip = static_cast<int>(cfg->core_start);
           ip < static_cast<int>(cfg->core_end); ++ip) {
        const int b = (ip - static_cast<int>(cfg->core_start)) & 1;
        const bool more = ip + 1 < static_cast<int>(cfg->core_end);
        if (pass == 0) {
          Layer(ss, ip, ip, nullptr);
        } else {
          Check(hipStreamWaitEvent(s, ev_msg[b], 0), "msgs");
          // The other buffer was read by the previous layer, which is queued
          // on `s` before this point.
          const int slot = static_cast<int>(cfg->n_phys) +
                           (pass - 1) * core_len +
                           (ip - static_cast<int>(cfg->core_start));
          msg_inv = T >= kMsgInvTokens ? minv[b] : nullptr;
          Layer(ss, ip, slot, mbuf[b], [&] {
            if (more)
              launch_msgs(ip + 1);
          });
        }
        Tap(visit++, "l_out");
        // A yield lets other sessions reuse the message buffers.
        if (MaybeYield(yr, true) && pass > 0 && more)
          launch_msgs(ip + 1);
      }
    }
    for (int ip = static_cast<int>(cfg->core_end);
         ip < static_cast<int>(cfg->n_phys); ++ip) {
      Layer(ss, ip, ip, nullptr);
      Tap(visit++, "l_out");
      if (visit < 48)
        MaybeYield(yr,
                   true);  // (the last visit is followed by the head)
    }
    ss.pos_ += T;
    const bool has_cps = ubatch_cps != nullptr && !ubatch_cps->empty();
    bool has_interior = false, has_at_end = false;
    if (has_cps) {
      for (const auto& cp : *ubatch_cps) {
        if (cp.local_t < T)
          has_interior = true;
        else
          has_at_end = true;
      }
    }
    if (!last_ubatch && !sink && !dsp && !has_cps)
      return;

    // output: attention-residual mix, norm, lm head, soft cap, crop
    const bool need_all = dsp || sink || has_interior;
    const int out_n = need_all ? T : 1;
    const std::size_t out_off =
        need_all ? 0 : static_cast<std::size_t>(T - 1) * kEmbd;
    {
      kk::MixStates st = States(cur);
      for (int i = 0; i < st.n; ++i)
        st.ptr[i] += out_off;
      kk::Mix(s, st, DF(w->output_res_query), 1, kEmbd, out_n, mixf,
              cfg->rms_eps, nullptr);
    }
    kk::RmsNorm(s, mixf, kEmbd, DF(w->output_norm), hn, kEmbd, kEmbd, out_n,
                cfg->rms_eps);
    if (dsp) {
      CaptureFeature(s, hn, T, 3, p_feat);
      std::vector<Drafter::InjectCapture> dcaps;
      if (has_interior) {
        for (const auto& cp : *ubatch_cps)
          if (cp.draft_f != nullptr)
            dcaps.push_back({cp.local_t - 1, cp.draft_f});
      }
      dsp->Inject(s, ss.draft_, p_feat, static_cast<std::size_t>(pos0), T,
                  dcaps);
      if (!last_ubatch && !sink && !has_cps)
        return;
    }
    const int V = static_cast<int>(cfg->n_vocab_text);
    auto chunk = [&](const float* src_hn, int n, std::vector<float>* host) {
      G(w->output, src_hn, kEmbd, logits_d, V, n, V);
      kk::Softcap(s, logits_d, static_cast<std::int64_t>(n) * V,
                  cfg->logit_softcap);
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
        chunk(hn + static_cast<std::size_t>(a) * kEmbd, n, &host);
        sink(static_cast<std::size_t>(first_idx + a), n, host.data());
        if (has_interior) {
          for (auto& cp : *ubatch_cps) {
            const int r = cp.local_t - 1;
            if (cp.local_t < T && r >= a && r < a + n)
              cp.logits.assign(
                  host.begin() + static_cast<std::ptrdiff_t>(r - a) * V,
                  host.begin() + static_cast<std::ptrdiff_t>(r - a + 1) * V);
          }
        }
        if ((last_ubatch || has_at_end) && a + n == T)
          ss.logits_.assign(host.end() - V, host.end());
        ss.logits_stale_ = false;
      }
    } else if (has_interior) {
      // Pack every interior checkpoint's final hidden row (and the ubatch end
      // row when needed) so the LM head weights are read once.
      std::vector<UbatchCheckpoint*> interior_ptrs;
      int n_out = 0;
      for (auto& cp : *ubatch_cps) {
        if (cp.local_t < T) {
          Check(hipMemcpyAsync(
                    mixf + static_cast<std::size_t>(n_out) * kEmbd,
                    hn + static_cast<std::size_t>(cp.local_t - 1) * kEmbd,
                    kEmbd * sizeof(float), hipMemcpyDeviceToDevice, s),
                "cp hn copy");
          interior_ptrs.push_back(&cp);
          ++n_out;
        }
      }
      const bool need_end = last_ubatch || has_at_end;
      if (need_end) {
        Check(hipMemcpyAsync(mixf + static_cast<std::size_t>(n_out) * kEmbd,
                             hn + static_cast<std::size_t>(T - 1) * kEmbd,
                             kEmbd * sizeof(float), hipMemcpyDeviceToDevice, s),
              "cp end hn copy");
        ++n_out;
      }
      chunk(mixf, n_out, &host);
      for (std::size_t i = 0; i < interior_ptrs.size(); ++i)
        interior_ptrs[i]->logits.assign(
            host.begin() + static_cast<std::ptrdiff_t>(i) * V,
            host.begin() + static_cast<std::ptrdiff_t>(i + 1) * V);
      if (need_end) {
        ss.logits_.assign(host.end() - V, host.end());
        ss.logits_stale_ = false;
      }
    } else {
      chunk(hn + static_cast<std::size_t>(need_all ? T - 1 : 0) * kEmbd, 1,
            &host);
      ss.logits_ = std::move(host);
      ss.logits_stale_ = false;
    }
  }
};

// ---------------------------------------------------------------------------------------------
Session::~Session() = default;

const std::vector<float>& Session::logits() const {
  if (logits_stale_) {
    // The step that left them synchronized its stream.
    const std::shared_lock<std::shared_mutex> gate(CaptureGate());
    logits_.resize(vocab_);
    Check(
        hipMemcpy(logits_.data(), dlogits_, vocab_ * 4, hipMemcpyDeviceToHost),
        "logits copy");
    logits_stale_ = false;
  }
  return logits_;
}

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
    // Init read the small tensors it converts (router, loop controllers, bias
    // tables) through the mapping; nothing reads it again. Drop those pages
    // from the process (a later read would refault them from the file).
    for (const auto& region : e->reader_->GetMappedRegions())
      platform::ReleaseMappedPages(region.data, region.size);
    e->CalibrateDecodePrefetch();
    return e;
  } catch (const std::exception& ex) {
    if (error)
      *error = ex.what();
    return nullptr;
  }
}

void Engine::CalibrateDecodePrefetch() {
  Impl& I = *impl_;
  // Replayed one-session decode steps of a scratch session, as in serving:
  // the prefetch's cost is its cross-stream waits inside the graph. Alternate
  // the layouts and keep each one's fastest step; the first round captures
  // the graphs and is not timed. The layouts' logits are identical (the
  // prefetch only warms the cache), so the choice changes only speed.
  auto session = NewSession();
  double best[2] = {std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()};
  std::int32_t token = 0;
  for (int round = 0; round < 4; ++round) {
    for (int layout = 0; layout < 2; ++layout) {
      I.decode_prefetch = layout == 1;
      std::int32_t argmax = 0;
      const auto start = std::chrono::steady_clock::now();
      Step(*session, token, &argmax);
      const std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - start;
      token = argmax;
      if (round > 0)
        best[layout] = std::min(best[layout], elapsed.count());
    }
  }
  // A tie keeps the plain layout. Drop only the losing layout's graph so the
  // winning short-context graph stays ready.
  I.decode_prefetch = best[1] < best[0];
  const int lose_key = 2048 * 128 + 1 + (I.decode_prefetch ? 0 : (1 << 29));
  if (auto it = I.graphs.find(lose_key); it != I.graphs.end()) {
    (void)hipGraphExecDestroy(it->second);
    I.graphs.erase(it);
  }
  std::fprintf(stderr,
               "inkling2: decode weight prefetch %s (step %.2f ms with it, "
               "%.2f ms without)\n",
               I.decode_prefetch ? "on" : "off", best[1] * 1e3, best[0] * 1e3);
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
  layout.logits =
      state.Fixed(static_cast<std::size_t>(weights_.config.n_vocab_text) * 4);
  if (impl_ != nullptr && impl_->dsp) {
    for (int l = 0; l < 3; ++l) {
      if (Drafter::WindowLayer(l)) {
        layout.dk[l] = state.Fixed(kv_row * ring);
        layout.dv[l] = state.Fixed(kv_row * ring);
      } else {
        layout.dk[l] =
            state.Context({.row_bytes = kv_row, .rows = max_context_});
        layout.dv[l] =
            state.Context({.row_bytes = kv_row, .rows = max_context_});
      }
    }
    layout.df = state.Fixed(2304 * 4);
  }
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
  s->dlogits_ = state.data<float>(layout.logits);
  s->vocab_ = weights_.config.n_vocab_text;
  if (impl_->dsp) {
    for (int l = 0; l < 3; ++l) {
      s->draft_.k[l] = state.data(layout.dk[l]);
      s->draft_.v[l] = state.data(layout.dv[l]);
      s->draft_.cap[l] = Drafter::WindowLayer(l) ? ring : max_context_;
      if (!Drafter::WindowLayer(l)) {
        s->regions_.push_back(state.region(layout.dk[l]));
        s->regions_.push_back(state.region(layout.dv[l]));
      }
    }
    s->draft_.f = state.data<float>(layout.df);
  }
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
  s.logits_stale_ = false;
  s.draft_.valid = 0;
  s.draft_.first = 0;
  s.draft_.f_pos = static_cast<std::size_t>(-1);
}

bool Engine::batch_capable() const {
  return impl_->BatchCapable();
}

void Engine::DecodeBatch(std::span<Session* const> sessions,
                         std::span<const std::int32_t> tokens) {
  if (sessions.size() != tokens.size() || sessions.size() > kMaxDecodeBatch)
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
      s.hooks_->BeforeWrite(s.pos_, s.pos_ + 1);
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

void Engine::DecodeBatchRows(std::span<Session* const> sessions,
                             std::span<const std::int32_t> tokens,
                             std::span<const bool> host,
                             std::span<const float>* rows, std::int32_t* argmax,
                             std::span<const LinearSample> linear,
                             std::int32_t* sampled) {
  if (!batch_capable() || sessions.empty() ||
      sessions.size() != tokens.size() || host.size() != sessions.size() ||
      sessions.size() > kMaxDecodeBatch ||
      (!linear.empty() && linear.size() != sessions.size()))
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
      s.hooks_->BeforeWrite(s.pos_, s.pos_ + 1);
    s.BackRows(s.pos_, s.pos_ + 1);
  }
  std::vector<float>* outs[Impl::kMaxB] = {};
  bool pinned[Impl::kMaxB] = {};
  for (std::size_t i = 0; i < sessions.size(); ++i)
    pinned[i] = host[i];
  impl_->DecodeN(sessions.data(), tokens.data(), outs,
                 static_cast<int>(sessions.size()), argmax, pinned, linear,
                 sampled);
  if (rows != nullptr)
    *rows = {impl_->h_rows, sessions.size() * weights_.config.n_vocab_text};
}

std::size_t Engine::VerifyRows(const Session& s) const {
  if (!batch_capable())
    return 1;
  const auto bucket = [](std::size_t pos) {
    std::size_t c = 2048;
    while (c < pos + 1)
      c *= 2;
    return c;
  };
  std::size_t n = 1;
  while (n < kMaxVerifyRows && s.pos_ + n < max_context_ &&
         bucket(s.pos_ + n) == bucket(s.pos_))
    ++n;
  return n;
}

void Engine::Verify(Session& s, std::span<const std::int32_t> tokens,
                    std::span<const float>* logits, std::int32_t* argmax,
                    std::span<const LinearSample> linear,
                    std::int32_t* sampled) {
  Session* sp = &s;
  const std::size_t n = tokens.size();
  VerifyBatch(std::span<Session* const>(&sp, 1),
              std::span<const std::size_t>(&n, 1), tokens, logits, argmax,
              linear, sampled);
}

void Engine::VerifyBatch(std::span<Session* const> sessions,
                         std::span<const std::size_t> counts,
                         std::span<const std::int32_t> tokens,
                         std::span<const float>* logits, std::int32_t* argmax,
                         std::span<const LinearSample> linear,
                         std::int32_t* sampled) {
  const std::lock_guard lock(mutex_);
  std::size_t total = 0;
  if (sessions.empty() || sessions.size() != counts.size())
    throw std::runtime_error("inkling2: bad verification batch");
  for (std::size_t g = 0; g < sessions.size(); ++g) {
    for (std::size_t h = 0; h < g; ++h)
      if (sessions[h] == sessions[g])
        throw std::runtime_error(
            "inkling2: duplicate session in verification batch");
    if (counts[g] == 0 || counts[g] > VerifyRows(*sessions[g]))
      throw std::runtime_error("inkling2: bad verification width");
    total += counts[g];
  }
  if (total != tokens.size() ||
      total > (sessions.size() == 1 ? kMaxVerifyRows : kMaxBatchVerifyRows) ||
      (!linear.empty() && linear.size() != total))
    throw std::runtime_error("inkling2: bad verification width");
  for (auto t : tokens)
    if (t < 0 || static_cast<std::uint32_t>(t) >= weights_.config.n_vocab_text)
      throw std::runtime_error("inkling2: token id out of range");
  int cnt[kMaxVerifyRows] = {};
  std::size_t row0 = 0;
  for (std::size_t g = 0; g < sessions.size(); ++g) {
    Session& s = *sessions[g];
    if (s.hooks_ != nullptr)
      s.hooks_->BeforeWrite(s.pos_, s.pos_ + counts[g]);
    s.BackRows(s.pos_, s.pos_ + counts[g]);
    cnt[g] = static_cast<int>(counts[g]);
    s.verify_rows_ = counts[g];
    s.verify_row0_ = row0;
    row0 += counts[g];
  }
  impl_->VerifyN(sessions.data(), cnt, static_cast<int>(sessions.size()),
                 tokens.data(), logits, argmax, linear, sampled);
}

void Engine::ReadVerifyRows(std::size_t first, std::size_t count,
                            std::span<const float>* out) {
  const std::lock_guard lock(mutex_);
  const std::size_t V = weights_.config.n_vocab_text;
  if (count == 0 || first + count > kMaxVerifyRows)
    throw std::runtime_error("inkling2: bad verify row read");
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  Check(hipMemcpyAsync(impl_->h_rows + first * V, impl_->logits_d + first * V,
                       count * V * sizeof(float), hipMemcpyDeviceToHost,
                       impl_->ds),
        "verify row read");
  Check(hipStreamSynchronize(impl_->ds), "verify row read");
  *out = {impl_->h_rows + first * V, count * V};
}

void Engine::CommitVerify(Session& s, std::size_t keep,
                          std::span<const float> logits) {
  Session* sp = &s;
  CommitVerifyBatch(std::span<Session* const>(&sp, 1),
                    std::span<const std::size_t>(&keep, 1),
                    std::span<const std::span<const float>>(&logits, 1));
}

void Engine::CommitVerifyBatch(std::span<Session* const> sessions,
                               std::span<const std::size_t> keeps,
                               std::span<const std::span<const float>> logits) {
  const std::lock_guard lock(mutex_);
  const std::size_t V = weights_.config.n_vocab_text;
  if (sessions.size() != keeps.size() || sessions.size() != logits.size() ||
      sessions.size() > kMaxVerifyRows)
    throw std::runtime_error("inkling2: bad verification commit");
  int keep_i[kMaxVerifyRows], row0_i[kMaxVerifyRows];
  for (std::size_t g = 0; g < sessions.size(); ++g) {
    Session& s = *sessions[g];
    const std::size_t keep = keeps[g];
    if (keep == 0 || keep > s.verify_rows_ ||
        (!logits[g].empty() && logits[g].size() < keep * V))
      throw std::runtime_error("inkling2: bad verification commit");
    for (std::size_t h = 0; h < g; ++h)
      if (sessions[h] == &s)
        throw std::runtime_error("inkling2: bad verification commit");
  }
  for (std::size_t g = 0; g < sessions.size(); ++g) {
    Session& s = *sessions[g];
    const std::size_t keep = keeps[g];
    const auto& lg = logits[g];
    if (lg.empty()) {
      Check(hipMemcpyAsync(s.dlogits_,
                           impl_->logits_d + (s.verify_row0_ + keep - 1) * V,
                           V * 4, hipMemcpyDeviceToDevice, impl_->ds),
            "logits copy");
      s.logits_stale_ = true;
    } else {
      s.logits_stale_ = false;
      s.logits_.assign(lg.begin() + static_cast<std::ptrdiff_t>((keep - 1) * V),
                       lg.begin() + static_cast<std::ptrdiff_t>(keep * V));
    }
    keep_i[g] = static_cast<int>(keep);
    row0_i[g] = static_cast<int>(s.verify_row0_);
  }
  impl_->CommitVerify(sessions.data(), keep_i, row0_i,
                      static_cast<int>(sessions.size()));
  for (auto* s : sessions) {
    s->verify_rows_ = 0;
    s->verify_row0_ = 0;
  }
}

bool Engine::AttachDrafter(const std::string& path, std::string* error) {
  const std::lock_guard lock(mutex_);
  Impl& I = *impl_;
  const auto& w = weights_;
  if (static_cast<int>(w.token_embd.type) != kk::kTQ8_0 || !I.head_fast) {
    if (error)
      *error = "DSpark needs the Q8_0 token embedding and LM head";
    return false;
  }
  Drafter::Target t{};
  t.token_embd = I.D(w.token_embd);
  t.token_embd_norm = I.DF(w.token_embd_norm);
  t.output = I.D(w.output);
  t.vocab = static_cast<int>(w.config.n_vocab_text);
  t.eps = w.config.rms_eps;
  t.softcap = w.config.logit_softcap;
  auto d = Drafter::Open(path, t, error);
  if (!d)
    return false;
  {
    const std::shared_lock<std::shared_mutex> gate(CaptureGate());
    if (I.p_feat == nullptr)
      I.p_feat =
          I.NewBuf(static_cast<std::size_t>(ubatch_) * Drafter::kFeatureFloats);
    if (I.cp_draft_f == nullptr)
      I.cp_draft_f = I.NewBuf(kMaxPassCheckpoints * kEmbd);
  }
  // the decode graphs gain the feature copies: capture them again
  I.DropGraphs();
  I.dsp = std::move(d);
  weight_bytes_ += I.dsp->weight_bytes();
  return true;
}

bool Engine::has_drafter() const {
  return impl_->dsp != nullptr;
}

bool Engine::DraftReady(const Session& s) const {
  return impl_->dsp != nullptr && Drafter::Ready(s.draft_, s.pos_);
}

int Engine::Propose(Session& s, std::int32_t anchor, int n, std::int32_t* out,
                    float* confidence, const Drafter::More& more,
                    const Drafter::Coupled* coupled) {
  const std::lock_guard lock(mutex_);
  if (!DraftReady(s))
    throw std::runtime_error("inkling2: no draft context at this position");
  if (anchor < 0 ||
      static_cast<std::uint32_t>(anchor) >= weights_.config.n_vocab_text)
    throw std::runtime_error("inkling2: token id out of range");
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  return impl_->dsp->Propose(impl_->ds, s.draft_, anchor, s.pos_, n, out,
                             confidence, more, coupled);
}

void Engine::ProposeBatch(std::span<ProposeRequest> requests) {
  const std::lock_guard lock(mutex_);
  if (requests.size() > static_cast<std::size_t>(Drafter::kMaxSessions))
    throw std::runtime_error("inkling2: proposal batch too large");
  std::array<Drafter::Request, Drafter::kMaxSessions> r{};
  for (std::size_t g = 0; g < requests.size(); ++g) {
    ProposeRequest& q = requests[g];
    if (!DraftReady(*q.session))
      throw std::runtime_error("inkling2: no draft context at this position");
    if (q.anchor < 0 ||
        static_cast<std::uint32_t>(q.anchor) >= weights_.config.n_vocab_text)
      throw std::runtime_error("inkling2: token id out of range");
    for (std::size_t h = 0; h < g; ++h)
      if (requests[h].session == q.session)
        throw std::runtime_error("inkling2: duplicate session in proposals");
    r[g] = {.st = &q.session->draft_,
            .anchor = q.anchor,
            .pos = q.session->pos_,
            .n = q.n,
            .out = q.out,
            .confidence = q.confidence,
            .more = q.more ? &q.more : nullptr,
            .coupled = q.coupled};
  }
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  impl_->dsp->ProposeBatch(
      impl_->ds, std::span<Drafter::Request>(r.data(), requests.size()));
  for (std::size_t g = 0; g < requests.size(); ++g)
    requests[g].done = r[g].done;
}

void Engine::Step(Session& s, std::int32_t token, std::int32_t* argmax) {
  const std::lock_guard lock(mutex_);
  if (s.pos_ + 1 > max_context_)
    throw std::runtime_error("inkling2: context exhausted");
  if (token < 0 ||
      static_cast<std::uint32_t>(token) >= weights_.config.n_vocab_text)
    throw std::runtime_error("inkling2: token id out of range");
  if (s.hooks_ != nullptr)
    s.hooks_->BeforeWrite(s.pos_, s.pos_ + 1);
  s.BackRows(s.pos_, s.pos_ + 1);
  Session* sp = &s;
  std::vector<float>* none = nullptr;
  impl_->DecodeN(&sp, &token, &none, 1, argmax);
}

void Engine::Decode(Session& s, std::int32_t token) {
  const std::lock_guard lock(mutex_);
  if (s.pos_ + 1 > max_context_)
    throw std::runtime_error("inkling2: context exhausted");
  if (token < 0 ||
      static_cast<std::uint32_t>(token) >= weights_.config.n_vocab_text)
    throw std::runtime_error("inkling2: token id out of range");
  if (s.hooks_ != nullptr)
    s.hooks_->BeforeWrite(s.pos_, s.pos_ + 1);
  s.BackRows(s.pos_, s.pos_ + 1);
  impl_->Decode(s, token, &s.logits_);
}

void Engine::ReserveYieldSpill() {
  const std::lock_guard<std::mutex> lock(mutex_);
  impl_->EnsureSpill();
}

void Engine::Synchronize() {
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  Check(hipStreamSynchronize(impl_->s), "synchronize");
}

bool Engine::CanCaptureInPass(std::size_t offset, std::size_t consumed,
                              std::size_t boundary) const {
  if (consumed == 0 || boundary <= offset || boundary >= offset + consumed)
    return false;
  const std::size_t u_idx = (boundary - offset - 1) / ubatch_;
  const std::size_t u_start = offset + u_idx * ubatch_;
  const std::size_t u_end = std::min(offset + consumed, u_start + ubatch_);
  if (boundary == u_end)
    return true;
  // Every sliding-window ring holds `ring_rows()` rows; rows needed at
  // `boundary` start at `FirstChunkRow` (256-row chunks covering 1024 window
  // rows). They stay intact at `u_end` iff `u_end - first_chunk <= ring_rows`.
  constexpr std::size_t kWin = 1024, kChunk = 256;
  const std::size_t first_chunk =
      boundary <= kWin ? 0 : ((boundary - kWin) / kChunk) * kChunk;
  return u_end - first_chunk <= ring_rows();
}

std::size_t Engine::PrefillCheckpointScratchBytes() const {
  return static_cast<std::size_t>(48) * kConvChannels * 3 * sizeof(float) +
         (has_drafter() ? static_cast<std::size_t>(kEmbd) * sizeof(float) : 0);
}

void Engine::Forward(Session& s, std::span<const std::int32_t> tokens,
                     const LogitsSink& sink, const PrefillYield* yield,
                     std::span<const InputRows> rows,
                     std::span<const std::size_t> checkpoints,
                     const PrefillCheckpointConsumer& on_checkpoint) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (s.pos_ + tokens.size() > max_context_)
    throw std::runtime_error("inkling2: context exhausted");
  for (auto t : tokens)
    if (t < 0 || static_cast<std::uint32_t>(t) >= weights_.config.n_vocab_text)
      throw std::runtime_error("inkling2: token id out of range");
  if (!checkpoints.empty()) {
    if (!on_checkpoint || checkpoints.size() > kMaxPassCheckpoints)
      throw std::invalid_argument("inkling2: invalid in-pass checkpoints");
    const std::size_t start_pos = s.pos_;
    for (std::size_t i = 0; i < checkpoints.size(); ++i) {
      if ((i > 0 && checkpoints[i] <= checkpoints[i - 1]) ||
          !CanCaptureInPass(start_pos, tokens.size(), checkpoints[i]))
        throw std::invalid_argument("inkling2: unsupported in-pass boundary");
    }
  }
  Impl::YieldRun run;
  Impl::YieldRun* yr = nullptr;
  if (yield != nullptr &&
      ((yield->groups >= 2 && yield->yield) || yield->cancelled)) {
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
    std::vector<Impl::UbatchCheckpoint>* saved_cps;
    ~RowsScope() {
      i->input_rows = saved;
      i->ubatch_cps = saved_cps;
    }
  } rows_scope{impl_.get(), impl_->input_rows, impl_->ubatch_cps};
  impl_->input_rows = rows;
  std::size_t done = 0;
  std::size_t next_cp = 0;
  while (done < tokens.size()) {
    const std::size_t n = std::min<std::size_t>(ubatch_, tokens.size() - done);
    const std::size_t u_start = s.pos_;
    const std::size_t u_end = u_start + n;
    std::vector<Impl::UbatchCheckpoint> u_cps;
    std::size_t n_interior = 0;
    while (next_cp < checkpoints.size() && checkpoints[next_cp] <= u_end) {
      const std::size_t pos = checkpoints[next_cp];
      Impl::UbatchCheckpoint item{};
      item.index = next_cp;
      item.position = pos;
      item.local_t = static_cast<int>(pos - u_start);
      u_cps.push_back(std::move(item));
      if (pos < u_end)
        ++n_interior;
      ++next_cp;
    }
    struct CpDevScope {  // the interior checkpoints' device state
      Impl* impl = nullptr;
      bool shared = false;  // impl's preallocated state, else owned below
      float* conv = nullptr;
      float* draft_f = nullptr;
      ~CpDevScope() {
        if (shared) {
          impl->cp_busy = false;
        } else if (conv != nullptr || draft_f != nullptr) {
          const std::shared_lock<std::shared_mutex> gate(CaptureGate());
          if (conv != nullptr)
            (void)hipFree(conv);
          if (draft_f != nullptr)
            (void)hipFree(draft_f);
        }
      }
    } cp_dev{.impl = impl_.get()};
    constexpr std::size_t kConvFloats = Impl::kCpConvFloats;
    if (n_interior > 0) {
      float* conv = impl_->cp_conv;
      float* draft_f = impl_->dsp ? impl_->cp_draft_f : nullptr;
      if (impl_->cp_busy) {  // a prefill inside another's yield
        const std::shared_lock<std::shared_mutex> gate(CaptureGate());
        Check(hipMalloc(reinterpret_cast<void**>(&cp_dev.conv),
                        n_interior * kConvFloats * sizeof(float)),
              "cp conv alloc");
        if (impl_->dsp)
          Check(hipMalloc(reinterpret_cast<void**>(&cp_dev.draft_f),
                          n_interior * kEmbd * sizeof(float)),
                "cp draft_f alloc");
        conv = cp_dev.conv;
        draft_f = cp_dev.draft_f;
      } else {
        impl_->cp_busy = true;
        cp_dev.shared = true;
      }
      std::size_t slot = 0;
      for (auto& cp : u_cps) {
        if (cp.position < u_end) {
          cp.conv = conv + slot * kConvFloats;
          if (draft_f != nullptr)
            cp.draft_f = draft_f + slot * kEmbd;
          ++slot;
        }
      }
    }
    if (s.hooks_ != nullptr)
      s.hooks_->BeforeWrite(s.pos_, s.pos_ + n);
    s.BackRows(s.pos_, s.pos_ + n);
    // Record drafter first position after gap-restart check for this ubatch.
    const std::size_t draft_first =
        impl_->dsp ? ((u_start > s.draft_.valid || u_start < s.draft_.first)
                          ? u_start
                          : s.draft_.first)
                   : 0;
    impl_->ubatch_cps = u_cps.empty() ? nullptr : &u_cps;
    impl_->Ubatch(s, tokens.subspan(done, n), static_cast<int>(done),
                  done + n == tokens.size(), sink, yr);
    impl_->ubatch_cps = nullptr;
    done += n;
    if (!u_cps.empty()) {
      Check(hipStreamSynchronize(impl_->s), "checkpoint sync");
      for (const auto& cp : u_cps) {
        PrefillCheckpoint pub{};
        pub.position = cp.position;
        pub.valid_end = u_end;
        pub.conv = cp.conv;
        pub.logits = cp.logits;
        pub.draft_f = cp.draft_f;
        if (impl_->dsp) {
          pub.draft_first = draft_first;
          pub.draft_valid = cp.position;
          pub.draft_f_pos = cp.position - 1;
        }
        on_checkpoint(cp.index, pub);
      }
    }
    if (yr != nullptr && done < tokens.size())
      impl_->MaybeYield(yr, false);  // between ubatches
  }
  Check(hipStreamSynchronize(impl_->s), "forward sync");
}

}  // namespace gufo::models::inkling2
