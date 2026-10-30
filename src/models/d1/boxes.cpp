#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "src/models/d1/model.hpp"
namespace gufo::models::d1 {
namespace {
// LiquidAI/d1-Omni-2B calibration.json, revision
// 431f1d667de5afcc9b0c58b2d9c170a7b38b358b. GGUF omits typed-output tables.
using Knot = std::pair<double, double>;
constexpr std::array<Knot, 10> kFoundBox = {{
    {0.00023668845211989844, 0.0},
    {0.0016392320222247323, 0.0056179775280898875},
    {0.008931349509895959, 0.02247191011235955},
    {0.04187172912907894, 0.10989010989010989},
    {0.2251482912219025, 0.45555555555555555},
    {0.6116028519378982, 0.7415730337078652},
    {0.825754906103156, 0.967032967032967},
    {0.9298707563271752, 0.9944444444444445},
    {0.970087437490838, 0.9945054945054945},
    {0.9928894672080476, 1.0},
}};
constexpr std::array<Knot, 8> kBox = {{
    {0.0052734375, 0.0},
    {0.19986979166666666, 0.1},
    {0.6123046875, 0.26666666666666666},
    {0.8121419270833333, 0.5166666666666667},
    {0.9237640881147541, 0.5901639344262295},
    {0.9784261688931297, 0.7251908396946565},
    {0.996875, 0.7636363636363637},
    {1.0, 0.936923076923077},
}};
constexpr std::array<Knot, 7> kFoundBoxes = {{
    {0.0006402823943090133, 0.0},
    {0.019446603905956206, 0.009708737864077669},
    {0.09626234011992091, 0.12745098039215685},
    {0.3871771655163214, 0.3627450980392157},
    {0.7680638221964662, 0.8415841584158416},
    {0.9589245048812044, 0.9801980198019802},
    {0.991101552510481, 1.0},
}};
constexpr std::array<Knot, 9> kBoxes = {{
    {0.0045166015625, 0.0},
    {0.0833740234375, 0.125},
    {0.3898555871212121, 0.18181818181818182},
    {0.5975341796875, 0.25},
    {0.76531982421875, 0.625},
    {0.9145951704545454, 0.6363636363636364},
    {0.9719669117647058, 0.7058823529411765},
    {0.9945845170454546, 0.8636363636363636},
    {0.9997829861111112, 0.9401709401709402},
}};
double Through(double value, std::span<const Knot> knots) {
  auto at =
      std::lower_bound(knots.begin(), knots.end(), value,
                       [](const Knot& k, double v) { return k.first < v; });
  if (at == knots.begin())
    return at->second;
  if (at == knots.end())
    return knots.back().second;
  const auto [a, pa] = *(at - 1);
  const auto [b, pb] = *at;
  return pa + (value - a) * (pb - pa) / (b - a);
}
using Rectangle = std::array<double, 4>;
std::pair<double, double> Ordered(double a, double b) {
  if (a < b)
    return {a, b};
  const double middle = std::clamp((a + b) / 2, 0.0005, 0.9995);
  return {middle - 0.0005, middle + 0.0005};
}
Rectangle Rect(Rectangle edges) {
  auto [top, bottom] = Ordered(edges[0], edges[1]);
  auto [left, right] = Ordered(edges[2], edges[3]);
  return {top, bottom, left, right};
}
json::Value JsonRect(const Rectangle& r) {
  auto value = json::Value::object();
  const char* names[] = {"top", "bottom", "left", "right"};
  for (int i = 0; i < 4; ++i)
    value[names[i]] = r[i];
  return value;
}
double Area(const Rectangle& b) {
  return (b[1] - b[0]) * (b[3] - b[2]);
}
double IoU(const Rectangle& a, const Rectangle& b) {
  const double overlap =
      std::max(0.0, std::min(a[1], b[1]) - std::max(a[0], b[0])) *
      std::max(0.0, std::min(a[3], b[3]) - std::max(a[2], b[2]));
  return overlap / (Area(a) + Area(b) - overlap);
}
double Radical(unsigned i, unsigned base) {
  double value = 0, scale = 1;
  while (i) {
    scale /= base;
    value += (i % base) * scale;
    i /= base;
  }
  return value;
}
struct BoxRead {
  Rectangle rectangle;
  double confidence;
};
BoxRead ReadEdges(std::span<const std::vector<double>> edges) {
  std::array<std::vector<double>, 4> cdfs;
  Rectangle median{};
  for (int edge = 0; edge < 4; ++edge) {
    if (edges[edge].size() != 1000)
      throw std::invalid_argument("box edges need 1000 bins");
    const auto& row = edges[edge];
    const double total = std::accumulate(row.begin(), row.end(), 0.0);
    if (!std::isfinite(total) || total <= 0)
      throw std::invalid_argument("invalid box distribution");
    double mass = 0;
    for (double p : row) {
      if (!std::isfinite(p) || p < 0)
        throw std::invalid_argument("invalid box probability");
      mass += p / total;
      cdfs[edge].push_back(mass);
    }
    median[edge] =
        std::min<std::size_t>(
            999, std::lower_bound(cdfs[edge].begin(), cdfs[edge].end(), 0.5) -
                     cdfs[edge].begin()) *
        0.001;
  }
  const auto rectangle = Rect(median);
  int hits = 0;
  const unsigned bases[] = {2, 3, 5, 7};
  for (unsigned i = 1; i <= 512; ++i) {
    Rectangle sample{};
    for (int d = 0; d < 4; ++d)
      sample[d] = std::min<std::size_t>(
                      999, std::lower_bound(cdfs[d].begin(), cdfs[d].end(),
                                            Radical(i, bases[d])) -
                               cdfs[d].begin()) *
                  0.001;
    hits += IoU(Rect(sample), rectangle) >= 0.5;
  }
  return {rectangle, hits / 512.0};
}
}  // namespace
json::Value ReadBoxDistributions(
    const Question& q, std::span<const std::vector<double>> probabilities) {
  if (!q.IsBox() || q.blocks < 1 || q.blocks > 16 ||
      (q.type == "box" && q.blocks != 1) ||
      probabilities.size() != q.blocks * 5)
    throw std::invalid_argument("invalid box slots");
  auto result = json::Value::object();
  result["type"] = q.type;
  auto presence = json::Value::array();
  struct Found {
    Rectangle rectangle;
    double found, confidence;
  };
  std::vector<Found> found;
  double below = 1;
  for (unsigned k = 0; k < q.blocks; ++k) {
    const auto& p = probabilities[k * 5];
    if (p.size() != 2 || !std::isfinite(p[0]) || p[0] < 0 || p[0] > 1)
      throw std::invalid_argument("invalid box presence distribution");
    const double present =
        Through(p[0], q.type == "box" ? std::span<const Knot>(kFoundBox)
                                      : std::span<const Knot>(kFoundBoxes));
    below = std::min(below, present);
    presence.push_back(below);
    if (below >= 0.5) {
      auto box = ReadEdges(probabilities.subspan(k * 5 + 1, 4));
      found.push_back(
          {box.rectangle, below,
           Through(box.confidence, q.type == "box"
                                       ? std::span<const Knot>(kBox)
                                       : std::span<const Knot>(kBoxes))});
    }
  }
  if (q.type == "box") {
    result["box"] =
        found.empty() ? json::Value() : JsonRect(found[0].rectangle);
    result["found"] = presence.items()[0];
    result["confidence"] = found.empty() ? 0.0 : found[0].confidence;
    return result;
  }
  std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) {
    return a.rectangle[2] < b.rectangle[2] ||
           (a.rectangle[2] == b.rectangle[2] &&
            a.rectangle[0] < b.rectangle[0]);
  });
  auto boxes = json::Value::array();
  for (const auto& box : found) {
    auto item = json::Value::object();
    item["box"] = JsonRect(box.rectangle);
    item["found"] = box.found;
    item["confidence"] = box.confidence;
    boxes.push_back(std::move(item));
  }
  result["boxes"] = boxes;
  result["count"] = found.size();
  result["at_least"] = !found.empty() && found.size() == q.blocks;
  result["presence"] = presence;
  return result;
}
json::Value ReadBoxes(const Question& q, std::span<const float> logits,
                      std::size_t vocabulary) {
  if (!vocabulary || logits.size() != q.Slots() * vocabulary)
    throw std::invalid_argument("invalid box logit shape");
  std::vector<std::vector<double>> p;
  for (std::size_t i = 0; i < q.Slots(); ++i)
    p.push_back(Probabilities(i % 5 ? q.edges : q.readout,
                              logits.subspan(i * vocabulary, vocabulary), 1.0));
  return ReadBoxDistributions(q, p);
}
}  // namespace gufo::models::d1
