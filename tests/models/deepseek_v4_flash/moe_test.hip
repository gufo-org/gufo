#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_q2_down.hip.hpp"

namespace {

void check(hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(e));
    std::exit(1);
  }
}
template<class T>
struct Device {
  T* p{};
  size_t n;
  explicit Device(size_t count) : n(count) {
    check(hipMalloc(&p, n * sizeof(T)));
  }
  ~Device() { (void)hipFree(p); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void copy(const std::vector<T>& v) {
    check(hipMemcpy(p, v.data(), n * sizeof(T), hipMemcpyHostToDevice));
  }
};
size_t shared(unsigned mt) {
  return ds4_rocm_q2_down_wide_shmem(mt, 16, 16, 16, 4);
}
template<unsigned MT>
void launch(__half* out, const char* w, const __half* mid,
            const unsigned* counts, const unsigned* offsets,
            const unsigned* pairs, const unsigned* experts,
            const unsigned* tiles, unsigned nt, unsigned ne, unsigned cols,
            unsigned dim) {
  moe_down_q2K_hotlist_wmma_wide_kernel<MT, 16, 16, 16, 4, true, true>
      <<<dim3(cols / 64, nt), 32 * MT, shared(MT)>>>(
          nullptr, out, w, nullptr, mid, counts, offsets, pairs, experts, ne,
          dim, cols, uint64_t(cols) * (dim / 256) * 84,
          uint64_t(dim / 256) * 84, 0, tiles);
}
void run(unsigned dim, std::mt19937& rng) {
  constexpr unsigned cols = 64;
  const unsigned nb = dim / 256;
  std::vector<unsigned> counts{0, 1, 63, 64, 65, 127, 128, 129, 191, 192, 193};
  const unsigned ne = counts.size(),
                 np = std::accumulate(counts.begin(), counts.end(), 0u);
  std::vector<hip_block_q2_K> w(uint64_t(ne) * cols * nb);
  std::vector<__half> mid(uint64_t(np) * dim), a(uint64_t(np) * cols),
      b(a.size());
  std::vector<unsigned> offsets(ne), pairs(np), experts(ne);
  std::iota(experts.rbegin(), experts.rend(), 0);
  for (auto& block : w) {
    for (auto& v : block.scales)
      v = rng();
    for (auto& v : block.qs)
      v = rng();
    const __half d = __float2half(float(1 + rng() % 7) * 0.0002F),
                 m = __float2half(float(1 + rng() % 7) * 0.0001F);
    std::memcpy(&block.d, &d, 2);
    std::memcpy(&block.dmin, &m, 2);
  }
  for (auto& v : mid)
    v = __float2half(float(int(rng() % 1024) - 512) * 0.002F);
  std::vector<unsigned> assigned;
  for (unsigned e = 0; e < ne; e++)
    for (unsigned i = 0; i < counts[e]; i++)
      assigned.push_back(e);
  std::shuffle(assigned.begin(), assigned.end(), rng);
  for (unsigned i = 1; i < ne; i++)
    offsets[i] = offsets[i - 1] + counts[i - 1];
  auto next = offsets;
  for (unsigned i = 0; i < np; i++)
    pairs[next[assigned[i]]++] = i;
  std::vector<unsigned> map4, map8;
  ds4_rocm_q2_down_tile_map(map4, counts.data(), experts.data(), ne, 4, false);
  const unsigned wide = ds4_rocm_q2_down_tile_map(map8, counts.data(),
                                                  experts.data(), ne, 4, true);
  // Check ownership separately from numerical equality: duplicate identical
  // writes could otherwise conceal an overlapping tile or a GPU data race.
  for (unsigned mt : {2u, 4u})
    for (bool paired : {false, true}) {
      if (mt == 2 && paired)
        continue;
      std::vector<unsigned> map, visits(np);
      const unsigned split = ds4_rocm_q2_down_tile_map(
          map, counts.data(), experts.data(), ne, mt, paired);
      for (unsigned tile = 0; tile < map.size(); tile++) {
        const unsigned e = experts[map[tile] >> 16],
                       rows = 16 * mt * (tile < split ? 2 : 1);
        const unsigned first = (map[tile] & 65535) * rows;
        for (unsigned row = first; row < std::min(first + rows, counts[e]);
             row++)
          visits[pairs[offsets[e] + row]]++;
      }
      if (std::any_of(visits.begin(), visits.end(),
                      [](unsigned n) { return n != 1; })) {
        std::fprintf(stderr, "Q2 down tile ownership failed\n");
        std::exit(2);
      }
    }
  Device<hip_block_q2_K> dw(w.size());
  Device<__half> dm(mid.size()), da(a.size()), db(b.size());
  Device<unsigned> dc(ne), dof(ne), dp(np), de(ne), dt4(map4.size()),
      dt8(map8.size());
  dw.copy(w);
  dm.copy(mid);
  dc.copy(counts);
  dof.copy(offsets);
  dp.copy(pairs);
  de.copy(experts);
  dt4.copy(map4);
  dt8.copy(map8);
  const auto old = [&] {
    launch<4>(da.p, reinterpret_cast<const char*>(dw.p), dm.p, dc.p, dof.p,
              dp.p, de.p, dt4.p, map4.size(), ne, cols, dim);
  };
  const auto opt = [&] {
    if (wide)
      launch<8>(db.p, reinterpret_cast<const char*>(dw.p), dm.p, dc.p, dof.p,
                dp.p, de.p, dt8.p, wide, ne, cols, dim);
    if (wide < map8.size())
      launch<4>(db.p, reinterpret_cast<const char*>(dw.p), dm.p, dc.p, dof.p,
                dp.p, de.p, dt8.p + wide, map8.size() - wide, ne, cols, dim);
  };
  check(hipMemset(da.p, 0xff, a.size() * 2));
  check(hipMemset(db.p, 0xff, b.size() * 2));
  old();
  opt();
  check(hipGetLastError());
  check(hipMemcpy(a.data(), da.p, a.size() * 2, hipMemcpyDeviceToHost));
  check(hipMemcpy(b.data(), db.p, b.size() * 2, hipMemcpyDeviceToHost));
  if (std::memcmp(a.data(), b.data(), a.size() * 2)) {
    std::fprintf(stderr, "Q2 down differs dim=%u\n", dim);
    std::exit(2);
  }
  for (unsigned pair : {0u, np / 2, np - 1})
    for (unsigned row : {0u, cols / 2, cols - 1}) {
      double sum = 0;
      for (unsigned k = 0; k < dim; k++) {
        const auto& block =
            w[(uint64_t(assigned[pair]) * cols + row) * nb + k / 256];
        const unsigned j = k % 256, g = j / 16, within = g & 7;
        const unsigned qi = (g >> 3) * 32 + (within & 1) * 16 + j % 16,
                       shift = (within >> 1) * 2;
        __half hd, hm;
        std::memcpy(&hd, &block.d, 2);
        std::memcpy(&hm, &block.dmin, 2);
        const float ds = __half2float(hd) * (block.scales[g] & 15),
                    mn = __half2float(hm) * (block.scales[g] >> 4);
        const float weight = __half2float(__float2half(
            std::fma(ds, float((block.qs[qi] >> shift) & 3), -mn)));
        sum += weight * double(__half2float(mid[uint64_t(pair) * dim + k]));
      }
      const float observed = __half2float(a[uint64_t(pair) * cols + row]);
      if (!std::isfinite(observed) ||
          std::abs(observed - sum) > 0.002 + std::abs(sum) * 0.001) {
        std::fprintf(stderr, "CPU FAIL got=%g expected=%g\n", observed, sum);
        std::exit(3);
      }
    }
  std::printf(
      "Q2 down dim=%u: mixed row tiles exact, independent CPU samples pass\n",
      dim);
}
}  // namespace

void test_q2_down() {
  std::mt19937 rng(731);
  for (unsigned dim : {256u, 2048u})
    run(dim, rng);
}
