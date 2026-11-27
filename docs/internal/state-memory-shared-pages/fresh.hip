// Is memory mapped at a never-used address visible to the device at once,
// also while another process keeps the GPU busy? K + 1 handles hold distinct
// constants (written through a home mapping that is unmapped again). Each
// batch maps all of them, then reads word 0 of every page right away with a
// kernel and with hipMemcpy, then unmaps them.
//
//   fresh <reps> <case>...
//   case 0: remap at the same address, alternating two handle sets (the
//           known stale-translation case, as a control)
//   case 1: map at a fresh reservation per batch (never mapped before)
//   case 2: map at the next never-used slots of one large reservation, right
//           after the previous slots were used (a region that grows)
//   case 3: as 1 with new handles (hipMemCreate per batch, released after):
//           a kernel writes each page through the new mapping at once, then
//           the reads check what it wrote (new memory reads zero anyway)
//   case 4: as 2 with new handles, written and checked as in 3
// A batch counts as correct when every page reads its handle's constant.
// Wrong values are split into zeros and other values; batches still wrong
// after 100 ms of polling with the kernel are counted as never.
#include <algorithm>
#include <cstdint>

#include "common.hpp"
static const size_t P = 2u << 20, W = P / 4;
static const int K = 8;
__global__ void Fill(uint32_t* d, size_t words, uint32_t v) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x)
    d[i] = v;
}
__global__ void ReadK(const uint32_t* d, size_t stride_words, int k,
                      uint32_t* out) {
  if (threadIdx.x < (unsigned)k)
    out[threadIdx.x] = d[threadIdx.x * stride_words];
}
static Vmm v;
static uint32_t* dout;
static hipStream_t s;
static void* reserve(size_t n) {
  hipDeviceptr_t p{};
  C(hipMemAddressReserve(&p, n * P, P, nullptr, 0));
  return p;
}
static void spin_us(double us) {
  const double t = now_us();
  while (now_us() - t < us) {
  }
}
static uint32_t value(int set, int k) { return 0x1000u * (set + 1) + k; }
static void read_kernel(void* x, uint32_t* o) {
  ReadK<<<1, 64, 0, s>>>((const uint32_t*)x, W, K + 1, dout);
  C(hipStreamSynchronize(s));
  C(hipMemcpy(o, dout, 4 * (K + 1), hipMemcpyDeviceToHost));
}
static void read_copy(void* x, uint32_t* o) {
  for (int k = 0; k <= K; ++k)
    C(hipMemcpy(&o[k], (char*)x + k * P, 4, hipMemcpyDeviceToHost));
}
struct Tally {
  int kernel_ok = 0, copy_ok = 0, never = 0, zeros = 0, others = 0;
  double wait_max = 0;
};
// Checks that every page of the batch at x reads `want`.
static void check(void* x, const uint32_t* want, Tally& t) {
  uint32_t o[K + 1];
  auto count = [&](const uint32_t* r) {
    int ok = 0;
    for (int k = 0; k <= K; ++k) {
      if (r[k] == want[k])
        ++ok;
      else if (r[k] == 0)
        ++t.zeros;
      else
        ++t.others;
    }
    return ok == K + 1;
  };
  read_kernel(x, o);
  const bool kn = count(o);
  read_copy(x, o);
  const bool cn = count(o);
  t.kernel_ok += kn;
  t.copy_ok += cn;
  bool done = kn;
  const double t0 = now_us();
  while (!done && now_us() - t0 < 100000) {
    spin_us(100);
    read_kernel(x, o);
    done = true;
    for (int k = 0; k <= K; ++k)
      done = done && o[k] == want[k];
  }
  if (!done)
    ++t.never;
  else if (!kn)
    t.wait_max = std::max(t.wait_max, now_us() - t0);
}
__global__ void WriteK(uint32_t* d, size_t stride_words, int k,
                       uint32_t base) {
  if (threadIdx.x < (unsigned)k)
    d[threadIdx.x * stride_words] = base + threadIdx.x;
}
static void check_set(void* x, int set, Tally& t) {
  uint32_t want[K + 1];
  for (int k = 0; k <= K; ++k)
    want[k] = value(set, k);
  check(x, want, t);
}
// New handles at x: written by a kernel at once, then checked.
static void fresh_handles(void* x, int r, Tally& t) {
  hipMemGenericAllocationHandle_t n[K + 1];
  for (int k = 0; k <= K; ++k) {
    C(hipMemCreate(&n[k], P, &v.prop, 0));
    C(hipMemMap((char*)x + k * P, P, 0, n[k], 0));
  }
  C(hipMemSetAccess(x, (K + 1) * P, &v.acc, 1));
  const uint32_t base = 0xA0000u + 16u * r;
  WriteK<<<1, 64, 0, s>>>((uint32_t*)x, W, K + 1, base);
  C(hipStreamSynchronize(s));
  uint32_t want[K + 1];
  for (int k = 0; k <= K; ++k)
    want[k] = base + k;
  check(x, want, t);
  C(hipMemUnmap(x, (K + 1) * P));
  for (int k = 0; k <= K; ++k)
    C(hipMemRelease(n[k]));
}
static void map_batch(void* x, hipMemGenericAllocationHandle_t* h) {
  for (int k = 0; k <= K; ++k)
    C(hipMemMap((char*)x + k * P, P, 0, h[k], 0));
  C(hipMemSetAccess(x, (K + 1) * P, &v.acc, 1));
}
int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 60;
  std::vector<int> cases;
  for (int i = 2; i < argc; ++i)
    cases.push_back(atoi(argv[i]));
  if (cases.empty())
    cases = {0, 1, 2};
  C(hipMalloc(&dout, 4096));
  C(hipStreamCreate(&s));
  hipMemGenericAllocationHandle_t h[2][K + 1];
  for (int j = 0; j < 2; ++j)
    for (int k = 0; k <= K; ++k) {
      C(hipMemCreate(&h[j][k], P, &v.prop, 0));
      void* home = reserve(1);  // kept reserved: never reused
      C(hipMemMap(home, P, 0, h[j][k], 0));
      C(hipMemSetAccess(home, P, &v.acc, 1));
      Fill<<<64, 256, 0, s>>>((uint32_t*)home, W, value(j, k));
      C(hipStreamSynchronize(s));
      C(hipMemUnmap(home, P));
    }
  for (int c : cases) {
    Tally t;
    if (c == 0) {
      void* x = reserve(K + 1);
      map_batch(x, h[0]);
      uint32_t o[K + 1];
      read_kernel(x, o);  // the translations are in use
      int set = 0;
      for (int r = 0; r < reps; ++r) {
        const int n = 1 - set;
        C(hipMemUnmap(x, (K + 1) * P));
        map_batch(x, h[n]);
        check_set(x, n, t);
        set = n;
      }
      C(hipMemUnmap(x, (K + 1) * P));
    } else if (c == 1) {
      for (int r = 0; r < reps; ++r) {
        void* x = reserve(K + 1);  // kept reserved: never reused
        map_batch(x, h[r % 2]);
        check_set(x, r % 2, t);
        C(hipMemUnmap(x, (K + 1) * P));
      }
    } else if (c == 2) {
      char* base = (char*)reserve((size_t)(reps + 1) * (K + 1));
      for (int r = 0; r < reps; ++r) {
        void* x = base + (size_t)r * (K + 1) * P;
        map_batch(x, h[r % 2]);
        check_set(x, r % 2, t);
        C(hipMemUnmap(x, (K + 1) * P));
      }
    } else if (c == 3) {
      for (int r = 0; r < reps; ++r)
        fresh_handles(reserve(K + 1), r, t);  // kept reserved: never reused
    } else if (c == 4) {
      char* base = (char*)reserve((size_t)(reps + 1) * (K + 1));
      for (int r = 0; r < reps; ++r)
        fresh_handles(base + (size_t)r * (K + 1) * P, r, t);
    }
    printf(
        "case %d: kernel all correct %3d/%d, copy all correct %3d/%d; wrong "
        "reads: %d zero, %d other; still wrong after 100 ms %d (max wait of "
        "the others %.0f us)\n",
        c, t.kernel_ok, reps, t.copy_ok, reps, t.zeros, t.others, t.never,
        t.wait_max);
    fflush(stdout);
  }
  return 0;
}
