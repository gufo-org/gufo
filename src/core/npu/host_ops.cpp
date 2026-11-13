#include "src/core/npu/host_ops.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define GUFO_NPU_HOST_AVX2 1
#endif

namespace gufo::npu::host {
namespace {

// A fixed pool: the caller publishes a job and a generation, workers take
// chunks from an atomic cursor, and the caller helps until all are done.
class Pool {
public:
  Pool() {
    const unsigned n = std::max(1u, std::thread::hardware_concurrency());
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
    {
      std::lock_guard lock(mutex_);
      fn_ = &fn;
      chunks_ = chunks;
      next_.store(0);
      done_.store(0);
      ++generation_;
    }
    wake_.notify_all();
    Help();
    std::unique_lock lock(mutex_);
    finished_.wait(lock, [&] { return done_.load() == chunks_; });
    fn_ = nullptr;
  }

  static thread_local bool inside;

private:
  void Help() {
    inside = true;
    for (;;) {
      const std::size_t i = next_.fetch_add(1);
      if (i >= chunks_)
        break;
      (*fn_)(i);
      if (done_.fetch_add(1) + 1 == chunks_) {
        std::lock_guard lock(mutex_);
        finished_.notify_all();
      }
    }
    inside = false;
  }

  void Work() {
    std::uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_)
          return;
        seen = generation_;
        if (!fn_)
          continue;
      }
      Help();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_, run_mutex_;
  std::condition_variable wake_, finished_;
  const std::function<void(std::size_t)>* fn_{};
  std::size_t chunks_{};
  std::atomic<std::size_t> next_{0}, done_{0};
  std::uint64_t generation_{};
  bool stop_{};
};

thread_local bool Pool::inside = false;

Pool& ThePool() {
  static Pool pool;
  return pool;
}

}  // namespace

std::size_t Threads() {
  return ThePool().size();
}

void ParallelFor(std::size_t n, std::size_t grain,
                 const std::function<void(std::size_t, std::size_t)>& fn) {
  if (n == 0)
    return;
  grain = std::max<std::size_t>(grain, 1);
  Pool& pool = ThePool();
  const std::size_t max_chunks = (n + grain - 1) / grain;
  const std::size_t chunks = std::min(max_chunks, pool.size() * 4);
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
  for (std::size_t i = 0; i < n; ++i)
    x[i] = Bf16ToFloat(FloatToBf16(x[i]));
}

void Bf16ToFloat(const std::uint16_t* src, float* dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    dst[i] = Bf16ToFloat(src[i]);
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

void RopeNeox(float* x, std::size_t rows, std::size_t heads,
              std::size_t head_dim, std::size_t rotary,
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
      for (std::size_t h = 0; h < heads; ++h) {
        float* p = x + (r * heads + h) * head_dim;
        for (std::size_t i = 0; i < half; ++i) {
          const float x0 = p[i], x1 = p[i + half];
          p[i] = x0 * c[i] - x1 * s[i];
          p[i + half] = x1 * c[i] + x0 * s[i];
        }
      }
    }
  });
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
                  for (std::size_t i = 0; i < dim; ++i)
                    p[i] = Activate(a, bias ? p[i] + bias[i] : p[i]);
                }
              });
}

void GatedRows(const float* gate, std::size_t gate_stride, const float* up,
               std::size_t up_stride, float* out, std::size_t out_stride,
               std::size_t rows, std::size_t dim, Activation a) {
  ParallelFor(rows,
              std::max<std::size_t>(1, 8192 / std::max<std::size_t>(dim, 1)),
              [&](std::size_t b, std::size_t e) {
                for (std::size_t r = b; r < e; ++r) {
                  const float* g = gate + r * gate_stride;
                  const float* u = up + r * up_stride;
                  float* o = out + r * out_stride;
                  for (std::size_t i = 0; i < dim; ++i)
                    o[i] = Activate(a, g[i]) * u[i];
                }
              });
}

void Attention(const AttentionArgs& p, const KeyRange& keys) {
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
        for (float& s : scores) {
          s = std::exp(s - top);
          sum += s;
        }
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
