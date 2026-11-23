#include "vision_image.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

namespace gufo::models::inkling2::vision {

static int AlignUp(int v) {
  return (v + kAlign - 1) / kAlign * kAlign;
}

ImagePlan PlanImage(int w, int h) {
  ImagePlan p;
  p.src_w = w;
  p.src_h = h;
  if (int64_t(AlignUp(w)) * AlignUp(h) > kMaxPixels) {
    // mtmd img_tool::calc_size_preserved_ratio with align 40, max_pixels, no
    // min
    auto round_f = [](float x) { return int(std::round(x / kAlign)) * kAlign; };
    auto floor_f = [](float x) { return int(std::floor(x / kAlign)) * kAlign; };
    int wb = std::max(kAlign, round_f(float(w)));
    int hb = std::max(kAlign, round_f(float(h)));
    if (int64_t(hb) * wb > kMaxPixels) {
      const float beta = std::sqrt(float(h) * float(w) / float(kMaxPixels));
      hb = std::max(kAlign, floor_f(float(h) / beta));
      wb = std::max(kAlign, floor_f(float(w) / beta));
    }
    p.src_w = wb;
    p.src_h = hb;
    p.downscaled = true;
  }
  p.pad_w = AlignUp(p.src_w);
  p.pad_h = AlignUp(p.src_h);
  p.grid_w = p.pad_w / kPatch;
  p.grid_h = p.pad_h / kPatch;
  p.tok_w = p.pad_w / kAlign;
  p.tok_h = p.pad_h / kAlign;
  p.n_tokens = p.tok_w * p.tok_h;
  return p;
}

namespace {
struct Taps {
  std::vector<int> first, count;
  std::vector<float> w;
  int maxn = 0;
};
Taps MakeTaps(int src, int dst) {
  Taps t;
  const float scale = float(src) / dst, support = std::max(1.0f, scale);
  t.maxn = int(std::ceil(support * 2)) + 1;
  t.first.resize(dst);
  t.count.resize(dst);
  t.w.assign(size_t(dst) * t.maxn, 0.f);
  for (int i = 0; i < dst; i++) {
    const float c = (i + 0.5f) * scale;
    int lo = std::max(0, int(std::floor(c - support + 0.5f)));
    int hi = std::min(src, int(std::floor(c + support + 0.5f)));
    float sum = 0;
    for (int j = lo; j < hi; j++) {
      float wv = std::max(0.f, 1.f - std::fabs((j + 0.5f - c) / support));
      t.w[size_t(i) * t.maxn + (j - lo)] = wv;
      sum += wv;
    }
    if (sum <= 0) {
      lo = std::min(src - 1, int(c));
      hi = lo + 1;
      t.w[size_t(i) * t.maxn] = 1;
      sum = 1;
    }
    for (int j = lo; j < hi; j++)
      t.w[size_t(i) * t.maxn + (j - lo)] /= sum;
    t.first[i] = lo;
    t.count[i] = hi - lo;
  }
  return t;
}
}  // namespace

// Separable triangle-filter resize of `img` to dw x dh (box-like when
// shrinking).
static void Resize(const ImageU8& img, int dw, int dh, ImageU8* out) {
  Taps tx = MakeTaps(img.w, dw), ty = MakeTaps(img.h, dh);
  std::vector<float> tmp(size_t(dw) * img.h * 3);
  for (int y = 0; y < img.h; y++)
    for (int x = 0; x < dw; x++)
      for (int c = 0; c < 3; c++) {
        float a = 0;
        for (int k = 0; k < tx.count[x]; k++)
          a += tx.w[size_t(x) * tx.maxn + k] *
               img.rgb[(size_t(y) * img.w + tx.first[x] + k) * 3 + c];
        tmp[(size_t(y) * dw + x) * 3 + c] = a;
      }
  out->w = dw;
  out->h = dh;
  out->rgb.resize(size_t(dw) * dh * 3);
  for (int y = 0; y < dh; y++)
    for (int x = 0; x < dw; x++)
      for (int c = 0; c < 3; c++) {
        float a = 0;
        for (int k = 0; k < ty.count[y]; k++)
          a += ty.w[size_t(y) * ty.maxn + k] *
               tmp[(size_t(ty.first[y] + k) * dw + x) * 3 + c];
        out->rgb[(size_t(y) * dw + x) * 3 + c] =
            uint8_t(std::min(255.f, std::max(0.f, std::round(a))));
      }
}

const ImageU8& ApplyPlan(const ImageU8& img, const ImagePlan& plan,
                         ImageU8* scratch) {
  if (!plan.downscaled)
    return img;
  Resize(img, plan.src_w, plan.src_h, scratch);
  return *scratch;
}

bool ShrinkToFit(ImageU8* img, int max_side) {
  const int longest = std::max(img->w, img->h);
  if (max_side < 1 || longest <= max_side)
    return false;
  const float k = float(max_side) / float(longest);
  const int dw = std::max(1, int(std::lround(img->w * k))),
            dh = std::max(1, int(std::lround(img->h * k)));
  ImageU8 out;
  Resize(*img, dw, dh, &out);
  *img = std::move(out);
  return true;
}

}  // namespace gufo::models::inkling2::vision
