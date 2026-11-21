// Which unmap/map sequences reach the handle they name? Every handle starts
// with its own id pattern, written through a first mapping at a "home" address
// unless a case says otherwise. A case then remaps and writes marker patterns
// through the address under test with a kernel. At the end every handle is read
// through a witness mapping at a fresh address (kernel read), so the report
// says which handle each write really reached.
#include <sched.h>
#include <time.h>

#include <cstdint>
#include <string>

#include "common.hpp"
static const size_t P = 2u << 20, W = P / 4;
__global__ void Fill(uint32_t* d, size_t words, uint32_t v) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x)
    d[i] = v;
}
__global__ void Probe(const uint32_t* d, size_t words, uint32_t* out) {
  // out[0] = word 0, out[1] = number of words different from word 0
  uint32_t n = 0;
  const uint32_t v = d[0];
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x)
    n += d[i] != v;
  if (blockIdx.x == 0 && threadIdx.x == 0)
    out[0] = v;
  if (n)
    atomicAdd(out + 1, n);
}
static Vmm v;
static uint32_t* dout;
static void* reserve(size_t n = 1) {
  hipDeviceptr_t p{};
  C(hipMemAddressReserve(&p, n * P, P, nullptr, 0));
  return p;
}
static hipMemGenericAllocationHandle_t make() {
  hipMemGenericAllocationHandle_t h;
  C(hipMemCreate(&h, P, &v.prop, 0));
  return h;
}
static void map(void* a, hipMemGenericAllocationHandle_t h) {
  C(hipMemMap(a, P, 0, h, 0));
  C(hipMemSetAccess(a, P, &v.acc, 1));
}
static void unmap(void* a) {
  C(hipMemUnmap(a, P));
}
static void fill(void* a, uint32_t val) {
  Fill<<<64, 256>>>((uint32_t*)a, W, val);
  C(hipDeviceSynchronize());
}
static std::string see(void* a) {
  C(hipMemset(dout, 0, 8));
  Probe<<<64, 256>>>((const uint32_t*)a, W, dout);
  C(hipDeviceSynchronize());
  uint32_t o[2];
  C(hipMemcpy(o, dout, 8, hipMemcpyDeviceToHost));
  char b[64];
  snprintf(b, sizeof b, "%x%s", o[0], o[1] ? "(mixed)" : "");
  return b;
}
static std::string witness(hipMemGenericAllocationHandle_t h) {
  void* w = reserve();
  map(w, h);
  auto s = see(w);
  return s;
}
int main(int argc, char** argv) {
  const int c = argc > 1 ? atoi(argv[1]) : 0;
  C(hipMalloc(&dout, 8));
  if (c == 1) {  // T1 replace a region page whose old handle stays parked
    void* X = reserve();
    void* park = reserve();
    auto old = make(), neu = make();
    map(X, old);
    fill(X, 0x01d);
    map(park, old);
    unmap(X);
    map(X, neu);
    fill(X, 0x2e3);
    printf(
        "T1 replace: X sees %s (want 2e3), park sees %s (want 1d), old=%s "
        "new=%s\n",
        see(X).c_str(), see(park).c_str(), witness(old).c_str(),
        witness(neu).c_str());
  } else if (c == 2) {  // T3 restore: region page own -> shared handle that is
                        // parked
    void* A = reserve();
    void* X = reserve();
    void* park = reserve();
    auto sh = make(), own = make();
    map(A, sh);
    fill(A, 0x5a);
    map(park, sh);
    map(X, own);
    fill(X, 0x0e);
    unmap(X);
    map(X, sh);
    printf("T3 restore: X sees %s (want 5a), own=%s (want e) shared=%s\n",
           see(X).c_str(), witness(own).c_str(), witness(sh).c_str());
    fill(A, 0x5b);
    printf("   after writer appends via A: X sees %s park sees %s (want 5b)\n",
           see(X).c_str(), see(park).c_str());
  } else if (c == 3) {  // T4 park slot reuse
    void* A = reserve(2);
    void* park = reserve();
    auto h0 = make(), h1 = make();
    map(A, h0);
    fill(A, 0x10);
    map((char*)A + P, h1);
    fill((char*)A + P, 0x11);
    map(park, h0);
    printf("T4 park: first %s (want 10)", see(park).c_str());
    unmap(park);
    map(park, h1);
    printf(" reuse %s (want 11) h0=%s h1=%s\n", see(park).c_str(),
           witness(h0).c_str(), witness(h1).c_str());
  } else if (c ==
             4) {  // Va: scratch F0 -> unmap -> F1, F0 not mapped elsewhere
    void* S = reserve();
    auto f0 = make(), f1 = make();
    map(S, f0);
    fill(S, 0xa0);
    unmap(S);
    map(S, f1);
    fill(S, 0xa1);
    printf(
        "Va plain reuse: S sees %s (want a1) f0=%s (want a0) f1=%s (want a1)\n",
        see(S).c_str(), witness(f0).c_str(), witness(f1).c_str());
  } else if (c ==
             5) {  // Vb: scratch F0 -> unmap -> F0 mapped at B -> S gets F1
    void* S = reserve();
    void* B = reserve(2);
    auto f0 = make(), f1 = make();
    map(S, f0);
    fill(S, 0xb0);
    unmap(S);
    map(B, f0);
    map(S, f1);
    fill(S, 0xb1);
    printf(
        "Vb moved handle: S sees %s (want b1) B sees %s (want b0) f0=%s "
        "f1=%s\n",
        see(S).c_str(), see(B).c_str(), witness(f0).c_str(),
        witness(f1).c_str());
  } else if (c == 6) {  // Vc: handle first mapped at its home B, then aliased
                        // at S, S reused for another aliased handle
    void* S = reserve();
    void* B = reserve(2);
    auto f0 = make(), f1 = make();
    map(B, f0);
    fill(B, 0xc0);
    map((char*)B + P, f1);
    fill((char*)B + P, 0xc1);
    map(S, f0);
    auto s0 = see(S);
    unmap(S);
    map(S, f1);
    auto s1 = see(S);
    fill(S, 0xcc);
    printf(
        "Vc alias reuse: S saw %s then %s (want c0 c1); after write via S: B0 "
        "%s (want c0) B1 %s (want cc)\n",
        s0.c_str(), s1.c_str(), see(B).c_str(), see((char*)B + P).c_str());
  } else if (c == 7) {  // Vd: region page replaced with a fresh handle, then
                        // replaced again (two rewinds)
    void* X = reserve();
    void* park = reserve(2);
    auto h0 = make(), h1 = make(), h2 = make();
    map(X, h0);
    fill(X, 0xd0);
    map(park, h0);
    unmap(X);
    map(X, h1);
    fill(X, 0xd1);
    map((char*)park + P, h1);
    unmap(X);
    map(X, h2);
    fill(X, 0xd2);
    printf(
        "Vd twice: X %s (want d2) park0 %s (d0) park1 %s (d1) h0=%s h1=%s "
        "h2=%s\n",
        see(X).c_str(), see(park).c_str(), see((char*)park + P).c_str(),
        witness(h0).c_str(), witness(h1).c_str(), witness(h2).c_str());
  } else if (c == 8) {  // Ve: handle released to a pool, reused at another
                        // address, old address remapped (recycling)
    void* X = reserve();
    void* Y = reserve();
    auto h0 = make(), h1 = make();
    map(X, h0);
    fill(X, 0xe0);
    unmap(X);
    map(Y, h0);
    fill(Y, 0xe5);
    map(X, h1);
    fill(X, 0xe1);
    printf("Ve recycle: X %s (want e1) Y %s (want e5) h0=%s h1=%s\n",
           see(X).c_str(), see(Y).c_str(), witness(h0).c_str(),
           witness(h1).c_str());
  } else if (c == 9) {  // Vf: as Vb but a device sync after every map/unmap
    void* S = reserve();
    void* B = reserve(2);
    auto f0 = make(), f1 = make();
    map(S, f0);
    fill(S, 0xb0);
    unmap(S);
    C(hipDeviceSynchronize());
    map(B, f0);
    C(hipDeviceSynchronize());
    map(S, f1);
    C(hipDeviceSynchronize());
    fill(S, 0xb1);
    printf(
        "Vf moved handle, synced: S sees %s (want b1) B sees %s (want b0) "
        "f0=%s f1=%s\n",
        see(S).c_str(), see(B).c_str(), witness(f0).c_str(),
        witness(f1).c_str());
  } else if (c == 10) {  // Vg: as Ve/Vb but the handle's second address is
                         // mapped BEFORE the first is unmapped
    void* S = reserve();
    void* B = reserve(2);
    auto f0 = make(), f1 = make();
    map(S, f0);
    fill(S, 0xb0);
    map(B, f0);
    unmap(S);
    map(S, f1);
    fill(S, 0xb1);
    printf(
        "Vg alias then unmap first: S sees %s (want b1) B sees %s (want b0) "
        "f0=%s f1=%s\n",
        see(S).c_str(), see(B).c_str(), witness(f0).c_str(),
        witness(f1).c_str());
  } else if (c >= 11 && c <= 16) {
    void* S = reserve();
    auto f0 = make(), f1 = make();
    void* home = reserve();
    map(home, f0);
    fill(home, 0xa0);
    unmap(home);  // f0 gets its pattern elsewhere
    uint32_t* big = nullptr;
    const size_t big_words = (4ull << 30) / 4;
    if (c == 12)
      C(hipMalloc(&big, big_words * 4));
    map(S, f0);
    if (c != 11) {
      (void)see(S);
    }  // 11: S never touched before the remap
    unmap(S);
    if (c == 13) {
      timespec ts{1, 0};
      nanosleep(&ts, nullptr);
    }
    if (c == 15) {
      C(hipDeviceSynchronize());
      void* other = reserve();
      auto fo = make();
      map(other, fo);
      fill(other, 0x77);
    }
    map(S, f1);
    if (c == 12) {
      Fill<<<4096, 256>>>(big, big_words, 1);
      C(hipDeviceSynchronize());
    }
    if (c == 14) {
      timespec ts{1, 0};
      nanosleep(&ts, nullptr);
    }
    if (c == 16) {
      C(hipMemSetAccess(S, P, &v.acc, 1));
      C(hipMemSetAccess(S, P, &v.acc, 1));
    }
    fill(S, 0xa1);
    const char* names[] = {"S untouched before remap", "4 GiB fill after remap",
                           "1 s sleep after unmap",    "1 s sleep after remap",
                           "other map+kernel between", "SetAccess twice"};
    printf("E%d %s: S sees %s f0=%s (want a0) f1=%s (want a1)\n", c,
           names[c - 11], see(S).c_str(), witness(f0).c_str(),
           witness(f1).c_str());
  } else if (c >= 20 && c <= 29) {
    // c=20: sleep argv[2] microseconds after the remap. c=21: sleep before the
    // unmap (after the touch). c=22: the touch, the write and the checks on
    // separate streams. c=23: unmap twice. c=24: free + reserve the address
    // again. c=25: hipMemSetAccess none then read-write. c=26: touch with
    // hipMemcpy instead of a kernel.
    const long us = argc > 2 ? atol(argv[2]) : 0;
    void* S = reserve();
    auto f0 = make(), f1 = make();
    void* home = reserve();
    map(home, f0);
    fill(home, 0xa0);
    unmap(home);
    hipStream_t s1 = nullptr, s2 = nullptr;
    C(hipStreamCreate(&s1));
    C(hipStreamCreate(&s2));
    map(S, f0);
    if (c == 22) {
      Fill<<<1, 1, 0, s1>>>((uint32_t*)S + 7, 1, 0xa0);
      C(hipStreamSynchronize(s1));
    } else if (c == 26) {
      uint32_t x;
      C(hipMemcpy(&x, S, 4, hipMemcpyDeviceToHost));
    } else
      (void)see(S);
    if (c == 21) {
      timespec ts{us / 1000000, (us % 1000000) * 1000};
      nanosleep(&ts, nullptr);
    }
    unmap(S);
    if (c == 23) {
      hipError_t e = hipMemUnmap(S, P);
      printf("second unmap: %s\n", hipGetErrorString(e));
      (void)hipGetLastError();
    }
    if (c == 24) {
      C(hipMemAddressFree(S, P));
      void* S2 = reserve();
      printf("re-reserved %s\n", S2 == S ? "same" : "other");
      S = S2;
    }
    map(S, f1);
    if (c == 25) {
      hipMemAccessDesc none = v.acc;
      none.flags = hipMemAccessFlagsProtNone;
      hipError_t e = hipMemSetAccess(S, P, &none, 1);
      printf("prot none: %s\n", hipGetErrorString(e));
      (void)hipGetLastError();
      C(hipMemSetAccess(S, P, &v.acc, 1));
    }
    if (c == 20) {
      timespec ts{us / 1000000, (us % 1000000) * 1000};
      nanosleep(&ts, nullptr);
    }
    if (c == 22) {
      Fill<<<64, 256, 0, s2>>>((uint32_t*)S, W, 0xa1);
      C(hipStreamSynchronize(s2));
    } else
      fill(S, 0xa1);
    printf("E%d us=%ld: f0=%s (want a0) f1=%s (want a1)\n", c, us,
           witness(f0).c_str(), witness(f1).c_str());
  } else if (c >= 30 && c <= 39) {
    // Barriers after the remap, none of which waits: 30 dummy map+unmap, 31
    // twice, 32 dummy map only, 33 unmap+map S again with the same handle, 34
    // dummy map+unmap BEFORE mapping the new handle, 35 as 30 but the dummy
    // handle is touched by a kernel first, 36 hipMemRelease of an unused
    // handle, 37 busy spin 150 us (no sleep), 38 sched_yield x 200.
    void* S = reserve();
    void* DV = reserve();
    auto f0 = make(), f1 = make(), d = make();
    void* home = reserve();
    map(home, f0);
    fill(home, 0xa0);
    unmap(home);
    map(S, f0);
    (void)see(S);
    unmap(S);
    if (c == 34) {
      map(DV, d);
      unmap(DV);
    }
    map(S, f1);
    if (c == 30 || c == 31) {
      map(DV, d);
      unmap(DV);
    }
    if (c == 31) {
      map(DV, d);
      unmap(DV);
    }
    if (c == 32) {
      map(DV, d);
    }
    if (c == 33) {
      unmap(S);
      map(S, f1);
    }
    if (c == 35) {
      map(DV, d);
      fill(DV, 5);
      unmap(DV);
    }
    if (c == 36) {
      auto x = make();
      C(hipMemRelease(x));
    }
    if (c == 37) {
      const double t = now_us();
      while (now_us() - t < 150) {
      }
    }
    if (c == 38) {
      for (int i = 0; i < 200; ++i)
        sched_yield();
    }
    fill(S, 0xa1);
    printf("B%d: f0=%s (want a0) f1=%s (want a1)\n", c, witness(f0).c_str(),
           witness(f1).c_str());
  } else if (c == 40 || c == 41) {
    // The wide-pass scratch pattern (mem::ScratchArena): map, use, unmap AND
    // release the handle, then map a new handle at the same address and use it.
    // 41: device idle 400 us before the use.
    void* X = reserve();
    int ok = 0, old = 0;
    const int reps = 50;
    for (int r = 0; r < reps; ++r) {
      auto h0 = make();
      map(X, h0);
      fill(X, 0x100 + r);
      unmap(X);
      C(hipMemRelease(h0));
      auto h1 = make();
      map(X, h1);
      if (c == 41) {
        C(hipDeviceSynchronize());
        const double t = now_us();
        while (now_us() - t < 400) {
        }
      }
      fill(X, 0x200 + r);
      // h1 through a witness: holds the new value only if the write reached it.
      void* w = reserve();
      map(w, h1);
      C(hipMemset(dout, 0, 8));
      Probe<<<64, 256>>>((const uint32_t*)w, W, dout);
      C(hipDeviceSynchronize());
      uint32_t o[2];
      C(hipMemcpy(o, dout, 8, hipMemcpyDeviceToHost));
      ok += o[0] == 0x200u + r;
      old += o[0] != 0x200u + r;
      unmap(w);
      unmap(X);
      C(hipMemRelease(h1));
      {
        C(hipDeviceSynchronize());
        const double t = now_us();
        while (now_us() - t < 500) {
        }
      }  // clean slate for the next repetition
    }
    printf(
        "scratch release pattern%s: write reached the new handle %d of %d; "
        "went to the released page %d\n",
        c == 41 ? ", idle first" : "", ok, reps, old);
  } else if (c == 42 || c == 43 || c == 44) {
    // As 40, with the new handle made BEFORE the touch, so that nothing slow
    // sits between the last use and the next. 42: unmap + release of the old
    // handle. 43: unmap only (old kept). 44: unmap only, then release of a
    // third handle that was mapped and unmapped earlier.
    void* X = reserve();
    void* Z = reserve();
    int ok = 0;
    const int reps = 50;
    for (int r = 0; r < reps; ++r) {
      auto h0 = make(), h1 = make(), h2 = make();
      map(Z, h2);
      fill(Z, 7);
      unmap(Z);
      {
        C(hipDeviceSynchronize());
        const double t = now_us();
        while (now_us() - t < 500) {
        }
      }
      map(X, h0);
      fill(X, 0x100 + r);
      unmap(X);
      if (c == 42)
        C(hipMemRelease(h0));
      if (c == 44)
        C(hipMemRelease(h2));
      map(X, h1);
      fill(X, 0x200 + r);
      void* w = reserve();
      map(w, h1);
      C(hipMemset(dout, 0, 8));
      Probe<<<64, 256>>>((const uint32_t*)w, W, dout);
      C(hipDeviceSynchronize());
      uint32_t o[2];
      C(hipMemcpy(o, dout, 8, hipMemcpyDeviceToHost));
      ok += o[0] == 0x200u + r;
      unmap(w);
      unmap(X);
      {
        C(hipDeviceSynchronize());
        const double t = now_us();
        while (now_us() - t < 500) {
        }
      }
    }
    const char* names[] = {"unmap + release old handle", "unmap only",
                           "unmap, release another used handle"};
    printf("no gap, %s: write reached the new handle %d of %d\n", names[c - 42],
           ok, reps);
  }
  return 0;
}
