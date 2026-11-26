#include "src/core/metrics.hpp"

#include <gufo/metrics.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/rust/callback.hpp"

namespace gufo::core::metrics {
namespace {

const std::uint8_t* Bytes(std::string_view text) {
  return reinterpret_cast<const std::uint8_t*>(text.data());
}

void Check(GufoMetricsStatus status, std::string_view what,
           std::string_view name) {
  switch (status) {
    case GUFO_METRICS_OK:
      return;
    case GUFO_METRICS_INVALID:
      throw std::invalid_argument(std::string(what) +
                                  " is invalid: " + std::string(name));
    case GUFO_METRICS_DUPLICATE:
      throw std::invalid_argument(
          std::string(what) + " is already registered: " + std::string(name));
    case GUFO_METRICS_CALLBACK:
    case GUFO_METRICS_INTERNAL:
      break;
  }
  throw std::logic_error("Rust metrics registry failed internally");
}

GufoMetricsKind ToC(detail::Kind kind) {
  switch (kind) {
    case detail::Kind::kCounterU64:
      return GUFO_METRICS_COUNTER_U64;
    case detail::Kind::kCounterF64:
      return GUFO_METRICS_COUNTER_F64;
    case detail::Kind::kGaugeU64:
      return GUFO_METRICS_GAUGE_U64;
    case detail::Kind::kGaugeF64:
      return GUFO_METRICS_GAUGE_F64;
    case detail::Kind::kGaugeI64:
      return GUFO_METRICS_GAUGE_I64;
    case detail::Kind::kHistogram:
      return GUFO_METRICS_HISTOGRAM;
  }
  throw std::logic_error("unknown metric kind");
}

}  // namespace

namespace detail {

const GufoMetricsFamily* AddFamily(const GufoMetricsRegistry* registry,
                                   Kind kind, std::string_view name,
                                   std::string_view help,
                                   std::span<const double> bounds) {
  const GufoMetricsFamily* family = nullptr;
  Check(gufo_metrics_family_new(registry, ToC(kind), Bytes(name), name.size(),
                                Bytes(help), help.size(), bounds.data(),
                                bounds.size(), &family),
        "metric family", name);
  return family;
}

const GufoMetricsCell* AddSeries(const GufoMetricsFamily* family,
                                 std::string_view family_name,
                                 std::span<const Label> labels) {
  std::vector<GufoMetricsLabel> raw;
  raw.reserve(labels.size());
  for (const Label& label : labels) {
    raw.push_back({Bytes(label.name), label.name.size(), Bytes(label.value),
                   label.value.size()});
  }
  const GufoMetricsCell* cell = nullptr;
  const GufoMetricsStatus status =
      gufo_metrics_series_new(family, raw.data(), raw.size(), &cell);
  if (status != GUFO_METRICS_OK) {
    std::string described(family_name);
    described += '{';
    for (const Label& label : labels) {
      if (described.back() != '{')
        described += ',';
      described.append(label.name).append("=\"").append(label.value) += '"';
    }
    described += '}';
    Check(status, "metric series", described);
  }
  return cell;
}

void SetVisible(const GufoMetricsFamily* family, bool visible) noexcept {
  gufo_metrics_family_set_visible(family, visible);
}

}  // namespace detail

void Counter::Add(std::uint64_t delta) const noexcept {
  gufo_metrics_u64_add(cell_, delta);
}
void Counter::Set(std::uint64_t value) const noexcept {
  gufo_metrics_u64_set(cell_, value);
}
void FloatCounter::Add(double delta) const noexcept {
  gufo_metrics_f64_add(cell_, delta);
}
void MaxCounter::Observe(std::uint64_t value) const noexcept {
  gufo_metrics_u64_max(cell_, value);
}
void UintGauge::Set(std::uint64_t value) const noexcept {
  gufo_metrics_u64_set(cell_, value);
}
void Gauge::Set(double value) const noexcept {
  gufo_metrics_f64_set(cell_, value);
}
void Gauge::Add(double delta) const noexcept {
  gufo_metrics_f64_add(cell_, delta);
}
void IntGauge::Add(std::int64_t delta) const noexcept {
  gufo_metrics_i64_add(cell_, delta);
}
void IntGauge::Set(std::int64_t value) const noexcept {
  gufo_metrics_i64_set(cell_, value);
}
void Histogram::Observe(double value) const noexcept {
  gufo_metrics_histogram_observe(cell_, value);
}

Registry::Registry() : handle_(gufo_metrics_registry_new()) {
  if (!handle_)
    throw std::logic_error("Rust metrics registry failed to allocate");
}

Registry::~Registry() = default;

void Registry::Free::operator()(GufoMetricsRegistry* registry) const noexcept {
  gufo_metrics_registry_free(registry);
}

std::string Registry::Render() const {
  struct Sink {
    std::string text;
    ffi::CallbackFailure failure;
  } sink;
  const GufoMetricsStatus status = gufo_metrics_render(
      handle_.get(),
      [](void* context, const std::uint8_t* bytes,
         std::size_t len) noexcept -> int {
        auto& sink = *static_cast<Sink*>(context);
        return sink.failure.Guard(1, [&] {
          sink.text.append(reinterpret_cast<const char*>(bytes), len);
          return 0;
        });
      },
      &sink);
  sink.failure.Rethrow();
  Check(status, "metrics render", {});
  return std::move(sink.text);
}

}  // namespace gufo::core::metrics
