#include "src/core/npu/host_ops.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define GUFO_NPU_HOST_AVX2 1
#endif

namespace gufo::npu::host {
namespace {

#ifdef GUFO_NPU_HOST_AVX2
// e^x for 8 lanes: range reduction by ln 2 and a degree-6 polynomial
// (relative error ~2e-7 over the float range; clamps at +-88).
inline __m256 Exp8(__m256 x) {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)),
                    _mm256_set1_ps(88.3f));
  const __m256 n =
      _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 p = _mm256_set1_ps(1.9875691500e-4f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
  p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r),
                      _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
  const __m256i e = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

// act(x) for 8 lanes; erf GELU has no vector form here (returns false).
inline bool Activate8(Activation a, __m256 x, __m256* out) {
  const __m256 one = _mm256_set1_ps(1.0f);
  switch (a) {
    case Activation::kNone:
      *out = x;
      return true;
    case Activation::kSilu:
      *out = _mm256_div_ps(
          x, _mm256_add_ps(one, Exp8(_mm256_sub_ps(_mm256_setzero_ps(), x))));
      return true;
    case Activation::kGeluTanh: {
      // 0.5 x (1 + tanh(u)) = x / (1 + e^(-2u)), u = k (x + 0.044715 x^3)
      const __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
      const __m256 u =
          _mm256_mul_ps(_mm256_set1_ps(-2.0f * 0.7978845608028654f),
                        _mm256_fmadd_ps(_mm256_set1_ps(0.044715f), x3, x));
      *out = _mm256_div_ps(x, _mm256_add_ps(one, Exp8(u)));
      return true;
    }
    case Activation::kGeluErf: {
      // 0.5 x (1 + erf(x / sqrt 2)), erf by Abramowitz-Stegun 7.1.26
      // (absolute error below 1.5e-7).
      const __m256 z = _mm256_mul_ps(x, _mm256_set1_ps(0.7071067811865476f));
      const __m256 sign = _mm256_and_ps(z, _mm256_set1_ps(-0.0f));
      const __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), z);
      const __m256 t = _mm256_div_ps(
          one, _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), a, one));
      __m256 p = _mm256_set1_ps(1.061405429f);
      p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-1.453152027f));
      p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.421413741f));
      p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.284496736f));
      p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.254829592f));
      p = _mm256_mul_ps(p, t);
      const __m256 e =
          Exp8(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(a, a)));
      const __m256 erf =
          _mm256_or_ps(_mm256_fnmadd_ps(p, e, one), sign);  // 1 - p e^(-a^2)
      *out = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x),
                           _mm256_add_ps(one, erf));
      return true;
    }
  }
  return false;
}
#endif

// A fixed pool. Each ParallelFor publishes a job with its own counters;
// workers take chunks from it and the caller helps until all are done. A
// worker late for one job only ever sees that job's exhausted counter.
class Pool {
public:
  explicit Pool(unsigned n) {
    n = std::max(1u, n);
    for (unsigned i = 1; i < n; ++i)
      workers_.emplace_back([this] { Work(); });
  }
  ~Pool() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : workers_)
      t.join();
  }

  std::size_t size() const { return workers_.size() + 1; }

  void Run(std::size_t chunks, const std::function<void(std::size_t)>& fn) {
    std::lock_guard serial(run_mutex_);
    auto job = std::make_shared<Job>();
    job->fn = &fn;
    job->chunks = chunks;
    // Wake only as many workers as there are chunks besides the caller's:
    // every woken worker takes the mutex, and a crowd of them delays the
    // job's end.
    const std::size_t helpers = std::min(chunks - 1, workers_.size());
    {
      std::lock_guard lock(mutex_);
      job_ = job;
      ++generation_;
      wanted_ = helpers;
    }
    for (std::size_t i = 0; i < helpers; ++i)
      wake_.notify_one();
    Help(*job);
    std::unique_lock lock(mutex_);
    finished_.wait(lock, [&] { return job->done.load() == chunks; });
    job_.reset();
    wanted_ = 0;
  }

  static thread_local bool inside;

private:
  struct Job {
    const std::function<void(std::size_t)>* fn{};
    std::size_t chunks{};
    std::atomic<std::size_t> next{0}, done{0};
  };

  void Help(Job& job) {
    inside = true;
    for (;;) {
      const std::size_t i = job.next.fetch_add(1);
      if (i >= job.chunks)
        break;
      (*job.fn)(i);
      if (job.done.fetch_add(1) + 1 == job.chunks) {
        std::lock_guard lock(mutex_);
        finished_.notify_all();
      }
    }
    inside = false;
  }

  void Work() {
    std::uint64_t seen = 0;
    for (;;) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] {
          return stop_ || (generation_ != seen && wanted_ > 0);
        });
        if (stop_)
          return;
        seen = generation_;
        --wanted_;
        job = job_;
      }
      Help(*job);
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_, run_mutex_;
  std::condition_variable wake_, finished_;
  std::shared_ptr<Job> job_;
  std::uint64_t generation_{};
  std::size_t wanted_{};  // workers the current job still wakes
  bool stop_{};
};

thread_local bool Pool::inside = false;

// The pool of the calling thread: a share of the hardware threads under a
// SharePool scope, else all of them.
thread_local Pool* current = nullptr;

Pool& FullPool() {
  static Pool pool(std::thread::hardware_concurrency());
  return pool;
}

Pool& ThePool() {
  return current ? *current : FullPool();
}

}  // namespace

SharePool::SharePool(unsigned index, unsigned count) : previous_(current) {
  static std::mutex mutex;
  static std::map<std::pair<unsigned, unsigned>, std::unique_ptr<Pool>> pools;
  std::lock_guard lock(mutex);
  auto& pool = pools[{index, count}];
  if (!pool)
    pool = std::make_unique<Pool>(std::max(
        1u, std::thread::hardware_concurrency() / std::max(1u, count)));
  current = pool.get();
}

SharePool::~SharePool() {
  current = static_cast<Pool*>(previous_);
}

std::size_t Threads() {
  return ThePool().size();
}

namespace {

std::size_t Chunks(const Pool& pool, std::size_t n, std::size_t grain) {
  grain = std::max<std::size_t>(grain, 1);
  return std::min((n + grain - 1) / grain, pool.size() * 4);
}

}  // namespace

bool RunsInline(std::size_t n, std::size_t grain) {
  return Pool::inside || Chunks(ThePool(), n, grain) <= 1;
}

void ParallelForPool(std::size_t n, std::size_t grain,
                     const std::function<void(std::size_t, std::size_t)>& fn) {
  if (n == 0)
    return;
  Pool& pool = ThePool();
  const std::size_t chunks = Chunks(pool, n, grain);
  if (chunks <= 1 || Pool::inside) {
    fn(0, n);
    return;
  }
  const std::size_t step = (n + chunks - 1) / chunks;
  const std::function<void(std::size_t)> body = [&](std::size_t c) {
    const std::size_t b = c * step, e = std::min(n, b + step);
    if (b < e)
      fn(b, e);
  };
  pool.Run(chunks, body);
}

float Bf16ToFloat(std::uint16_t h) {
  return std::bit_cast<float>(std::uint32_t(h) << 16);
}

std::uint16_t FloatToBf16(float f) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(f);
  if ((u & 0x7f800000u) == 0x7f800000u)
    return std::uint16_t((u >> 16) | ((u & 0xffff) ? 0x40 : 0));
  u += 0x7fffu + ((u >> 16) & 1u);
  return std::uint16_t(u >> 16);
}

void RoundBf16(float* x, std::size_t n) {
  std::size_t i = 0;
#ifdef GUFO_NPU_HOST_AVX2
  // Round to nearest even; finite inputs (activations).
  const __m256i bias = _mm256_set1_epi32(0x7fff), one = _mm256_set1_epi32(1),
                mask = _mm256_set1_epi32(int(0xffff0000u));
  for (; i + 8 <= n; i += 8) {
    const __m256i u = _mm256_castps_si256(_mm256_loadu_ps(x + i));
    const __m256i r =
        _mm256_add_epi32(_mm256_add_epi32(u, bias),
                         _mm256_and_si256(_mm256_srli_epi32(u, 16), one));
    _mm256_storeu_ps(x + i, _mm256_castsi256_ps(_mm256_and_si256(r, mask)));
  }
#endif
  for (; i < n; ++i)
    x[i] = Bf16ToFloat(FloatToBf16(x[i]));
}

void Bf16ToFloat(const std::uint16_t* src, float* dst, std::size_t n) {
  std::size_t i = 0;
#ifdef GUFO_NPU_HOST_AVX2
  for (; i + 8 <= n; i += 8) {
    const __m128i h =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(_mm256_slli_epi32(
                                  _mm256_cvtepu16_epi32(h), 16)));
  }
#endif
  for (; i < n; ++i)
    dst[i] = Bf16ToFloat(src[i]);
}

void FloatToBf16(const float* src, std::uint16_t* dst, std::size_t n) {
  std::size_t i = 0;
#ifdef GUFO_NPU_HOST_AVX2
  // Round to nearest even; finite inputs (activations).
  const __m256i bias = _mm256_set1_epi32(0x7fff), one = _mm256_set1_epi32(1);
  for (; i + 16 <= n; i += 16) {
    __m256i a = _mm256_castps_si256(_mm256_loadu_ps(src + i));
    __m256i b = _mm256_castps_si256(_mm256_loadu_ps(src + i + 8));
    a = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_add_epi32(a, bias),
                         _mm256_and_si256(_mm256_srli_epi32(a, 16), one)),
        16);
    b = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_add_epi32(b, bias),
                         _mm256_and_si256(_mm256_srli_epi32(b, 16), one)),
        16);
    // packus works per 128-bit lane: fix the order with a permute.
    const __m256i packed =
        _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xd8);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), packed);
  }
#endif
  for (; i < n; ++i)
    dst[i] = FloatToBf16(src[i]);
}

void ExpInPlace(float* x, std::size_t n, float shift, float* sum) {
  std::size_t i = 0;
  float total = 0;
#ifdef GUFO_NPU_HOST_AVX2
  const __m256 sh = _mm256_set1_ps(shift);
  __m256 acc = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    const __m256 e = Exp8(_mm256_sub_ps(_mm256_loadu_ps(x + i), sh));
    _mm256_storeu_ps(x + i, e);
    acc = _mm256_add_ps(acc, e);
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, acc);
  for (float l : lanes)
    total += l;
#endif
  for (; i < n; ++i) {
    x[i] = std::exp(x[i] - shift);
    total += x[i];
  }
  if (sum)
    *sum = total;
}

void SoftmaxRowBf16(std::uint16_t* x, std::size_t n, std::size_t lo,
                    std::size_t hi, float* tmp) {
  const std::size_t m = hi - lo;
  Bf16ToFloat(x + lo, tmp, m);
  float top = -INFINITY;
  for (std::size_t t = 0; t < m; ++t)
    top = std::max(top, tmp[t]);
  float sum = 0;
  ExpInPlace(tmp, m, top, &sum);
  const float inv = 1.0f / sum;
  for (std::size_t t = 0; t < m; ++t)
    tmp[t] *= inv;
  std::fill(x, x + lo, std::uint16_t(0));
  FloatToBf16(tmp, x + lo, m);
  std::fill(x + hi, x + n, std::uint16_t(0));
}

namespace {

#ifdef GUFO_NPU_HOST_AVX2
// One bfp16ebs8 block from 8 lanes: the shared exponent is the largest, and
// mantissas round to nearest even. Blocks below 2^-121 encode as zero.
inline void EncodeBlock8(__m256 x, std::uint8_t* out) {
  __m256i ex = _mm256_and_si256(_mm256_srli_epi32(_mm256_castps_si256(x), 23),
                                _mm256_set1_epi32(0xff));
  ex = _mm256_max_epi32(ex, _mm256_permute2x128_si256(ex, ex, 1));
  ex = _mm256_max_epi32(ex, _mm256_shuffle_epi32(ex, 0x4e));
  ex = _mm256_max_epi32(ex, _mm256_shuffle_epi32(ex, 0xb1));
  const int e = _mm256_cvtsi256_si32(ex);
  if (e < 6) {
    std::memset(out, 0, 9);
    out[0] = std::uint8_t(e);
    return;
  }
  const __m256 inv = _mm256_castsi256_ps(_mm256_set1_epi32((260 - e) << 23));
  __m256 y = _mm256_round_ps(_mm256_mul_ps(x, inv),
                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  y = _mm256_min_ps(_mm256_max_ps(y, _mm256_set1_ps(-128.0f)),
                    _mm256_set1_ps(127.0f));
  // int32 -> int8: pack to 16 bits, then 8 bits; lane order fixed by the
  // permute.
  const __m256i q = _mm256_cvtps_epi32(y);
  const __m128i lo = _mm256_castsi256_si128(q),
                hi = _mm256_extracti128_si256(q, 1);
  const __m128i w = _mm_packs_epi32(lo, hi);
  const __m128i b = _mm_packs_epi16(w, w);
  out[0] = std::uint8_t(e);
  std::uint64_t bytes = std::uint64_t(_mm_cvtsi128_si64(b));
  std::memcpy(out + 1, &bytes, 8);
}
#endif

}  // namespace

void EncodeBfp16Blocks(const float* values, std::size_t blocks,
                       std::uint8_t* out, std::size_t stride,
                       std::size_t src_stride) {
#ifdef GUFO_NPU_HOST_AVX2
  for (std::size_t i = 0; i < blocks; ++i)
    EncodeBlock8(_mm256_loadu_ps(values + i * src_stride), out + i * stride);
#else
  for (std::size_t i = 0; i < blocks; ++i)
    EncodeBfp16Block(values + i * src_stride, out + i * stride);
#endif
}

void EncodeBfp16BlocksBf16(const std::uint16_t* values, std::size_t blocks,
                           std::uint8_t* out, std::size_t stride) {
#ifdef GUFO_NPU_HOST_AVX2
  for (std::size_t i = 0; i < blocks; ++i) {
    const __m128i h =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(values + 8 * i));
    EncodeBlock8(
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16)),
        out + i * stride);
  }
#else
  float v[8];
  for (std::size_t i = 0; i < blocks; ++i) {
    Bf16ToFloat(values + 8 * i, v, 8);
    EncodeBfp16Block(v, out + i * stride);
  }
#endif
}

void EncodeBfp16Block(const float* values, std::uint8_t* out) {
  int e = 0;
#ifdef GUFO_NPU_HOST_AVX2
  const __m256 x = _mm256_loadu_ps(values);
  __m256i ex = _mm256_and_si256(_mm256_srli_epi32(_mm256_castps_si256(x), 23),
                                _mm256_set1_epi32(0xff));
  ex = _mm256_max_epi32(ex, _mm256_permute2x128_si256(ex, ex, 1));
  ex = _mm256_max_epi32(ex, _mm256_shuffle_epi32(ex, 0x4e));
  ex = _mm256_max_epi32(ex, _mm256_shuffle_epi32(ex, 0xb1));
  e = _mm256_cvtsi256_si32(ex);
  if (e >= 6) {
    // 64 / 2^(e - 127) = 2^(133 - e), a normal float for e >= 6.
    const __m256 inv = _mm256_castsi256_ps(_mm256_set1_epi32((260 - e) << 23));
    __m256 y = _mm256_round_ps(_mm256_mul_ps(x, inv),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    y = _mm256_min_ps(_mm256_max_ps(y, _mm256_set1_ps(-128.0f)),
                      _mm256_set1_ps(127.0f));
    alignas(32) std::int32_t q[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(q), _mm256_cvtps_epi32(y));
    out[0] = std::uint8_t(e);
    for (int i = 0; i < 8; ++i)
      out[1 + i] = std::uint8_t(std::int8_t(q[i]));
    return;
  }
#else
  for (int i = 0; i < 8; ++i)
    e = std::max(e,
                 int((std::bit_cast<std::uint32_t>(values[i]) >> 23) & 0xff));
#endif
  out[0] = std::uint8_t(e);
  if (e < 6) {  // below 2^-121: encodes as zero
    std::fill(out + 1, out + 9, std::uint8_t(0));
    return;
  }
  const float inv = std::ldexp(1.0f, 127 + 6 - e);
  for (int i = 0; i < 8; ++i) {
    const float t = std::nearbyint(values[i] * inv);
    out[1 + i] = std::uint8_t(std::int8_t(std::clamp(t, -128.0f, 127.0f)));
  }
}

float Dot(const float* a, const float* b, std::size_t n) {
  std::size_t i = 0;
  float s = 0;
#ifdef GUFO_NPU_HOST_AVX2
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  for (; i + 16 <= n; i += 16) {
    acc0 =
        _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                           _mm256_loadu_ps(b + i + 8), acc1);
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  __m128 h =
      _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
  h = _mm_add_ps(h, _mm_movehl_ps(h, h));
  h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
  s = _mm_cvtss_f32(h);
#endif
  for (; i < n; ++i)
    s += a[i] * b[i];
  return s;
}

void Axpy(float a, const float* x, float* y, std::size_t n) {
  std::size_t i = 0;
#ifdef GUFO_NPU_HOST_AVX2
  const __m256 va = _mm256_set1_ps(a);
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i),
                                            _mm256_loadu_ps(y + i)));
#endif
  for (; i < n; ++i)
    y[i] += a * x[i];
}

void RmsNormRows(const float* x, float* y, std::size_t rows, std::size_t dim,
                 const float* weight, float eps, float offset) {
  ParallelFor(rows,
              std::max<std::size_t>(1, 16384 / std::max<std::size_t>(dim, 1)),
              [&](std::size_t b, std::size_t e) {
                for (std::size_t r = b; r < e; ++r) {
                  const float* xr = x + r * dim;
                  float* yr = y + r * dim;
                  const float ss = Dot(xr, xr, dim);
                  const float inv = 1.0f / std::sqrt(ss / float(dim) + eps);
                  if (weight)
                    for (std::size_t i = 0; i < dim; ++i)
                      yr[i] = xr[i] * inv * (weight[i] + offset);
                  else
                    for (std::size_t i = 0; i < dim; ++i)
                      yr[i] = xr[i] * inv;
                }
              });
}

void RmsNormRowsBf16(const float* x, std::uint16_t* y, std::size_t rows,
                     std::size_t dim, const float* weight, float eps) {
  ParallelFor(rows,
              std::max<std::size_t>(1, 16384 / std::max<std::size_t>(dim, 1)),
              [&](std::size_t b, std::size_t e) {
                std::vector<float> row(dim);
                for (std::size_t r = b; r < e; ++r) {
                  const float* xr = x + r * dim;
                  const float inv =
                      1.0f / std::sqrt(Dot(xr, xr, dim) / float(dim) + eps);
                  for (std::size_t i = 0; i < dim; ++i)
                    row[i] = xr[i] * inv * (weight ? weight[i] : 1.0f);
                  FloatToBf16(row.data(), y + r * dim, dim);
                }
              });
}

void AddNormRows(float* state, const float* x, std::size_t rows,
                 std::size_t dim, const float* weight, float eps, float scale) {
  ParallelFor(rows,
              std::max<std::size_t>(1, 16384 / std::max<std::size_t>(dim, 1)),
              [&](std::size_t b, std::size_t e) {
                for (std::size_t r = b; r < e; ++r) {
                  const float* xr = x + r * dim;
                  float* sr = state + r * dim;
                  const float inv =
                      1.0f / std::sqrt(Dot(xr, xr, dim) / float(dim) + eps);
                  for (std::size_t i = 0; i < dim; ++i)
                    sr[i] = (sr[i] + xr[i] * inv * weight[i]) * scale;
                }
              });
}

void LayerNormRows(const float* x, float* y, std::size_t rows, std::size_t dim,
                   const float* weight, const float* bias, float eps) {
  ParallelFor(rows,
              std::max<std::size_t>(1, 16384 / std::max<std::size_t>(dim, 1)),
              [&](std::size_t b, std::size_t e) {
                for (std::size_t r = b; r < e; ++r) {
                  const float* xr = x + r * dim;
                  float* yr = y + r * dim;
                  double mean = 0;
                  for (std::size_t i = 0; i < dim; ++i)
                    mean += xr[i];
                  mean /= double(dim);
                  double var = 0;
                  for (std::size_t i = 0; i < dim; ++i)
                    var += (xr[i] - mean) * (xr[i] - mean);
                  var /= double(dim);
                  const float inv = float(1.0 / std::sqrt(var + eps));
                  const float m = float(mean);
                  for (std::size_t i = 0; i < dim; ++i) {
                    float v = (xr[i] - m) * inv;
                    if (weight)
                      v *= weight[i];
                    if (bias)
                      v += bias[i];
                    yr[i] = v;
                  }
                }
              });
}

void RopeNeox(float* x, std::size_t rows, std::size_t row_stride,
              std::size_t heads, std::size_t head_dim, std::size_t rotary,
              const std::int32_t* positions, float theta) {
  const std::size_t half = rotary / 2;
  std::vector<double> inv(half);
  for (std::size_t i = 0; i < half; ++i)
    inv[i] = std::pow(double(theta), -double(2 * i) / double(rotary));
  ParallelFor(rows, 8, [&](std::size_t b, std::size_t e) {
    std::vector<float> c(half), s(half);
    for (std::size_t r = b; r < e; ++r) {
      for (std::size_t i = 0; i < half; ++i) {
        const double a = double(positions[r]) * inv[i];
        c[i] = float(std::cos(a));
        s[i] = float(std::sin(a));
      }
      RopeNeoxRow(x + r * row_stride, heads, head_dim, rotary, c.data(),
                  s.data());
    }
  });
}

void RopeNeoxRow(float* x, std::size_t heads, std::size_t head_dim,
                 std::size_t rotary, const float* cos, const float* sin) {
  const std::size_t half = rotary / 2;
  for (std::size_t h = 0; h < heads; ++h) {
    float* p = x + h * head_dim;
    for (std::size_t i = 0; i < half; ++i) {
      const float x0 = p[i], x1 = p[i + half];
      p[i] = x0 * cos[i] - x1 * sin[i];
      p[i + half] = x1 * cos[i] + x0 * sin[i];
    }
  }
}

float Activate(Activation a, float x) {
  switch (a) {
    case Activation::kNone:
      return x;
    case Activation::kSilu:
      return x / (1.0f + std::exp(-x));
    case Activation::kGeluTanh: {
      const float k = 0.7978845608028654f;
      return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
    }
    case Activation::kGeluErf:
      return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
  }
  return x;
}

void ActivateRows(float* x, std::size_t rows, std::size_t dim,
                  const float* bias, Activation a) {
  ParallelFor(rows,
              std::max<std::size_t>(1, 8192 / std::max<std::size_t>(dim, 1)),
              [&](std::size_t b, std::size_t e) {
                for (std::size_t r = b; r < e; ++r) {
                  float* p = x + r * dim;
                  std::size_t i = 0;
#ifdef GUFO_NPU_HOST_AVX2
                  __m256 v;
                  for (; i + 8 <= dim; i += 8) {
                    __m256 in = _mm256_loadu_ps(p + i);
                    if (bias)
                      in = _mm256_add_ps(in, _mm256_loadu_ps(bias + i));
                    if (!Activate8(a, in, &v))
                      break;
                    _mm256_storeu_ps(p + i, v);
                  }
#endif
                  for (; i < dim; ++i)
                    p[i] = Activate(a, bias ? p[i] + bias[i] : p[i]);
                }
              });
}

void GatedRows(const float* gate, std::size_t gate_stride, const float* up,
               std::size_t up_stride, float* out, std::size_t out_stride,
               std::size_t rows, std::size_t dim, Activation a) {
  ParallelFor(
      rows, std::max<std::size_t>(1, 8192 / std::max<std::size_t>(dim, 1)),
      [&](std::size_t b, std::size_t e) {
        for (std::size_t r = b; r < e; ++r) {
          const float* g = gate + r * gate_stride;
          const float* u = up + r * up_stride;
          float* o = out + r * out_stride;
          std::size_t i = 0;
#ifdef GUFO_NPU_HOST_AVX2
          __m256 v;
          for (; i + 8 <= dim; i += 8) {
            if (!Activate8(a, _mm256_loadu_ps(g + i), &v))
              break;
            _mm256_storeu_ps(o + i, _mm256_mul_ps(v, _mm256_loadu_ps(u + i)));
          }
#endif
          for (; i < dim; ++i)
            o[i] = Activate(a, g[i]) * u[i];
        }
      });
}

void Attention(const AttentionArgs& p, const KeyRange& keys, bool skip_empty) {
  const std::size_t group = p.heads / p.kv_heads;
  // Work items: (head, block of 4 query rows) so a key row serves 4 dots.
  constexpr std::size_t kBlock = 4;
  const std::size_t blocks = (p.rows + kBlock - 1) / kBlock;
  ParallelFor(p.heads * blocks, 1, [&](std::size_t b, std::size_t e) {
    std::vector<float> scores;
    std::vector<float> acc(p.dim_v);
    for (std::size_t item = b; item < e; ++item) {
      const std::size_t h = item / blocks, blk = item % blocks;
      const std::size_t kh = h / group;
      const std::size_t r0 = blk * kBlock, r1 = std::min(p.rows, r0 + kBlock);
      for (std::size_t r = r0; r < r1; ++r) {
        std::size_t kb = 0, ke = 0;
        keys(r, &kb, &ke);
        float* o = p.out + r * p.out_stride + h * p.dim_v;
        if (ke <= kb) {
          if (!skip_empty)
            std::fill_n(o, p.dim_v, 0.0f);
          continue;
        }
        const float* q = p.q + r * p.q_stride + h * p.dim;
        scores.resize(ke - kb);
        float top = -INFINITY;
        for (std::size_t t = kb; t < ke; ++t) {
          float s = Dot(q, p.k + t * p.k_stride + kh * p.dim, p.dim) * p.scale;
          if (p.softcap > 0)
            s = p.softcap * std::tanh(s / p.softcap);
          scores[t - kb] = s;
          top = std::max(top, s);
        }
        float sum = 0;
        ExpInPlace(scores.data(), scores.size(), top, &sum);
        const float inv = 1.0f / sum;
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (std::size_t t = kb; t < ke; ++t)
          Axpy(scores[t - kb] * inv, p.v + t * p.v_stride + kh * p.dim_v,
               acc.data(), p.dim_v);
        std::copy(acc.begin(), acc.end(), o);
      }
    }
  });
}

}  // namespace gufo::npu::host
