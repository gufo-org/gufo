#ifndef GUFO_CORE_METRICS_HPP_
#define GUFO_CORE_METRICS_HPP_

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>

struct GufoMetricsCell;
struct GufoMetricsFamily;
struct GufoMetricsRegistry;

namespace gufo::core::metrics {

// A metrics registry in Rust, rendered in the Prometheus text format. Register
// every family once, at startup, then update the returned metrics from any
// thread: an update is one lock-free call that neither locks nor allocates. A
// metric is a pointer into its Registry, so the Registry must outlive every
// thread that updates one; a process-wide Registry should never be destroyed.
//
// A default-constructed metric ignores updates.

// A label as registered: a fixed name and value.
struct Label {
  std::string_view name;
  std::string_view value;
};

namespace detail {
const GufoMetricsFamily* AddFamily(const GufoMetricsRegistry* registry,
                                   int kind, std::string_view name,
                                   std::string_view help,
                                   std::span<const double> bounds);
const GufoMetricsCell* AddSeries(const GufoMetricsFamily* family,
                                 std::string_view family_name,
                                 std::span<const Label> labels);
void SetVisible(const GufoMetricsFamily* family, bool visible) noexcept;
}  // namespace detail

// A count that only grows. Set mirrors a total another component keeps.
class Counter {
public:
  Counter() = default;
  void Add(std::uint64_t delta) const noexcept;
  void Set(std::uint64_t value) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr int kKind = 0;
  explicit Counter(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

// A fractional count that only grows, such as seconds.
class FloatCounter {
public:
  FloatCounter() = default;
  void Add(double delta) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr int kKind = 1;
  explicit FloatCounter(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

// The largest value observed, reported as a counter.
class MaxCounter {
public:
  MaxCounter() = default;
  void Observe(std::uint64_t value) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr int kKind = 0;
  explicit MaxCounter(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

// A non-negative integer level, such as bytes.
class UintGauge {
public:
  UintGauge() = default;
  void Set(std::uint64_t value) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr int kKind = 2;
  explicit UintGauge(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

// A fractional level, such as a ratio or a rate.
class Gauge {
public:
  Gauge() = default;
  void Set(double value) const noexcept;
  void Add(double delta) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr int kKind = 3;
  explicit Gauge(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

// A signed integer level that several owners adjust by deltas.
class IntGauge {
public:
  IntGauge() = default;
  void Add(std::int64_t delta) const noexcept;
  void Set(std::int64_t value) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr int kKind = 4;
  explicit IntGauge(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

// Observations counted into fixed buckets, with their sum.
class Histogram {
public:
  Histogram() = default;
  void Observe(double value) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr int kKind = 5;
  explicit Histogram(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

// Shows or hides a family without naming its metric type.
class Visibility {
public:
  // A hidden family renders nothing until shown again.
  void Set(bool visible) const noexcept {
    detail::SetVisible(family_, visible);
  }

private:
  template<class>
  friend class Family;
  explicit Visibility(const GufoMetricsFamily* family) : family_(family) {}
  const GufoMetricsFamily* family_;
};

// One named metric whose series differ by label values.
template<class Metric>
class Family {
public:
  // Adds the series with these labels; throws std::invalid_argument on a
  // repeated label set or an unusable label.
  Metric Add(std::initializer_list<Label> labels = {}) const;

  [[nodiscard]] Visibility visibility() const { return Visibility(family_); }

private:
  friend class Registry;
  Family(const GufoMetricsFamily* family, std::string_view name)
      : family_(family), name_(name) {}
  const GufoMetricsFamily* family_;
  std::string name_;
};

class Registry {
public:
  Registry();
  ~Registry();
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;

  // Registers a family after the earlier ones; the text lists families in
  // this order. Throws std::invalid_argument for a name the text format
  // cannot carry, a name already registered, or histogram bounds that are not
  // finite and strictly increasing (and for bounds on any other kind).
  template<class Metric>
  Family<Metric> AddFamily(std::string_view name, std::string_view help,
                           std::span<const double> bounds = {}) {
    return Family<Metric>(
        detail::AddFamily(handle_.get(), Metric::kKind, name, help, bounds),
        name);
  }

  // A family with one unlabeled series.
  template<class Metric>
  Metric Add(std::string_view name, std::string_view help,
             std::span<const double> bounds = {}) {
    return AddFamily<Metric>(name, help, bounds).Add();
  }

  // The text exposition of every visible family that has a series.
  [[nodiscard]] std::string Render() const;

private:
  struct Free {
    void operator()(GufoMetricsRegistry* registry) const noexcept;
  };
  std::unique_ptr<GufoMetricsRegistry, Free> handle_;
};

template<class Metric>
Metric Family<Metric>::Add(std::initializer_list<Label> labels) const {
  return Metric(
      detail::AddSeries(family_, name_, {labels.begin(), labels.size()}));
}

}  // namespace gufo::core::metrics

#endif  // GUFO_CORE_METRICS_HPP_
