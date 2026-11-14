#include "src/models/qwen3_asr/npu/split_q8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace gufo::models::qwen3_asr::npu {
namespace host = gufo::npu::host;

Lane::Lane() : thread_([this] { Loop(); }) {}

Lane::~Lane() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  thread_.join();
}

void Lane::Post(std::function<void()> task) {
  {
    std::lock_guard lock(mutex_);
    task_ = std::move(task);
    busy_ = true;
    error_ = nullptr;
  }
  wake_.notify_one();
}

void Lane::Wait() {
  std::unique_lock lock(mutex_);
  done_.wait(lock, [&] { return !busy_; });
  if (error_)
    std::rethrow_exception(std::exchange(error_, nullptr));
}

void Lane::Loop() {
  std::unique_lock lock(mutex_);
  for (;;) {
    wake_.wait(lock, [&] { return stop_ || (busy_ && task_); });
    if (stop_)
      return;
    auto task = std::move(task_);
    task_ = nullptr;
    lock.unlock();
    std::exception_ptr error;
    try {
      task();
    } catch (...) {
      error = std::current_exception();
    }
    lock.lock();
    error_ = error;
    busy_ = false;
    done_.notify_all();
  }
}

namespace {

void Quantize(const std::uint16_t* bf16, std::size_t rows, int k,
              std::int8_t* codes, float* scales) {
  const std::size_t blocks = rows * std::size_t(k) / 32;
  host::ParallelFor(blocks, 1024, [&](std::size_t b, std::size_t e) {
    for (std::size_t blk = b; blk < e; ++blk) {
      float v[32], amax = 0;
      for (int i = 0; i < 32; ++i) {
        v[i] = host::Bf16ToFloat(bf16[blk * 32 + std::size_t(i)]);
        amax = std::max(amax, std::fabs(v[i]));
      }
      const float d = amax / 127.0f, inv = d > 0 ? 1.0f / d : 0.0f;
      scales[blk] = d;
      for (int i = 0; i < 32; ++i)
        codes[blk * 32 + std::size_t(i)] =
            std::int8_t(std::lround(std::clamp(v[i] * inv, -127.0f, 127.0f)));
    }
  });
}

// sum over 32-code blocks of (int16 x . int8 w) * scale, for one row.
float RowDot(const std::int16_t* x, const std::int8_t* w, const float* s,
             int k) {
#if defined(__AVX2__) && defined(__FMA__)
  __m256 acc = _mm256_setzero_ps();
  for (int b = 0; b < k / 32; ++b) {
    const __m256i c = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(w + std::size_t(b) * 32));
    const __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(c));
    const __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(c, 1));
    const __m256i x0 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(x + std::size_t(b) * 32));
    const __m256i x1 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(x + std::size_t(b) * 32 + 16));
    const __m256i d =
        _mm256_add_epi32(_mm256_madd_epi16(lo, x0), _mm256_madd_epi16(hi, x1));
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(s[b]), acc);
  }
  __m128 h =
      _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  h = _mm_add_ps(h, _mm_movehl_ps(h, h));
  h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
  return _mm_cvtss_f32(h);
#else
  float acc = 0;
  for (int b = 0; b < k / 32; ++b) {
    std::int32_t d = 0;
    for (int i = 0; i < 32; ++i)
      d += std::int32_t(x[b * 32 + i]) * w[b * 32 + i];
    acc += float(d) * s[b];
  }
  return acc;
#endif
}

}  // namespace

SplitQ8::SplitQ8(gufo::npu::Q8Gemm& gemm, const std::uint16_t* bf16, int n,
                 int k, int npu_rows)
    : gemm_(&gemm), n_(n), k_(k), npu_rows_(std::clamp(npu_rows, 0, n)) {
  if (k % 32)
    throw std::runtime_error("Qwen3-ASR NPU: Q8_0 needs k % 32 == 0");
  if (npu_rows_ > 0) {
    std::vector<std::int8_t> c(std::size_t(npu_rows_) * k);
    std::vector<float> s(c.size() / 32);
    Quantize(bf16, std::size_t(npu_rows_), k, c.data(), s.data());
    npu_id_ = gemm.Upload(npu_rows_, k, c, s);
  }
  const std::size_t cpu_rows = std::size_t(n - npu_rows_);
  codes_.resize(cpu_rows * std::size_t(k));
  scales_.resize(codes_.size() / 32);
  if (cpu_rows)
    Quantize(bf16 + std::size_t(npu_rows_) * k, cpu_rows, k, codes_.data(),
             scales_.data());
}

void SplitQ8::Run(Lane& lane, const float* x, float* y) const {
  if (npu_rows_ > 0)
    lane.Post([this, x, y] { gemm_->Run(npu_id_, x, 1, y); });
  const int cpu_rows = n_ - npu_rows_;
  if (cpu_rows > 0) {
    float top = 0;
    for (int i = 0; i < k_; ++i)
      top = std::max(top, std::fabs(x[i]));
    const float inv = top > 0 ? 32767.0f / top : 0.0f;
    const float scale = top > 0 ? top / 32767.0f : 1.0f;
    std::vector<std::int16_t> q(static_cast<std::size_t>(k_));
    for (int i = 0; i < k_; ++i) {
      const float t = x[i] * inv;
      q[std::size_t(i)] = std::int16_t(t + (t >= 0 ? 0.5f : -0.5f));
    }
    const std::size_t blocks = std::size_t(k_) / 32;
    host::ParallelFor(
        std::size_t(cpu_rows), 32, [&](std::size_t b, std::size_t e) {
          for (std::size_t r = b; r < e; ++r)
            y[std::size_t(npu_rows_) + r] =
                RowDot(q.data(), codes_.data() + r * std::size_t(k_),
                       scales_.data() + r * blocks, k_) *
                scale;
        });
  }
  if (npu_rows_ > 0)
    lane.Wait();
}

int NpuRows(int n, int k) {
  // Measured on Strix Halo: an NPU dispatch costs about 95 us plus 17.5 us
  // per MB of Q8_0 weights; the CPU about 25 us plus 10 us per MB.
  const double mb = double(n) * k * (1.0 + 4.0 / 32.0) / 1e6;
  const double share = (10.0 * mb - 70.0) / (27.5 * mb);
  if (share <= 0.0)
    return 0;
  return std::min(n, int(std::lround(share * n / 8.0)) * 8);
}

}  // namespace gufo::models::qwen3_asr::npu
