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

struct Label {
  std::string_view name;
  std::string_view value;
};

namespace detail {
enum class Kind {
  kCounterU64,
  kCounterF64,
  kGaugeU64,
  kGaugeF64,
  kGaugeI64,
  kHistogram,
};

const GufoMetricsFamily* AddFamily(const GufoMetricsRegistry* registry,
                                   Kind kind, std::string_view name,
                                   std::string_view help,
                                   std::span<const double> bounds);
const GufoMetricsCell* AddSeries(const GufoMetricsFamily* family,
                                 std::string_view family_name,
                                 std::span<const Label> labels);
void SetVisible(const GufoMetricsFamily* family, bool visible) noexcept;
}  // namespace detail

class Counter {
public:
  Counter() = default;
  void Add(std::uint64_t delta) const noexcept;
  void Set(std::uint64_t value) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr detail::Kind kKind = detail::Kind::kCounterU64;
  explicit Counter(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

class FloatCounter {
public:
  FloatCounter() = default;
  void Add(double delta) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr detail::Kind kKind = detail::Kind::kCounterF64;
  explicit FloatCounter(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

class MaxCounter {
public:
  MaxCounter() = default;
  void Observe(std::uint64_t value) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr detail::Kind kKind = detail::Kind::kCounterU64;
  explicit MaxCounter(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

class UintGauge {
public:
  UintGauge() = default;
  void Set(std::uint64_t value) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr detail::Kind kKind = detail::Kind::kGaugeU64;
  explicit UintGauge(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

class Gauge {
public:
  Gauge() = default;
  void Set(double value) const noexcept;
  void Add(double delta) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr detail::Kind kKind = detail::Kind::kGaugeF64;
  explicit Gauge(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

class IntGauge {
public:
  IntGauge() = default;
  void Add(std::int64_t delta) const noexcept;
  void Set(std::int64_t value) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr detail::Kind kKind = detail::Kind::kGaugeI64;
  explicit IntGauge(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

class Histogram {
public:
  Histogram() = default;
  void Observe(double value) const noexcept;

private:
  template<class>
  friend class Family;
  friend class Registry;
  static constexpr detail::Kind kKind = detail::Kind::kHistogram;
  explicit Histogram(const GufoMetricsCell* cell) : cell_(cell) {}
  const GufoMetricsCell* cell_ = nullptr;
};

class Visibility {
public:
  void Set(bool visible) const noexcept {
    detail::SetVisible(family_, visible);
  }

private:
  template<class>
  friend class Family;
  explicit Visibility(const GufoMetricsFamily* family) : family_(family) {}
  const GufoMetricsFamily* family_;
};

template<class Metric>
class Family {
public:
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

  template<class Metric>
  Family<Metric> AddFamily(std::string_view name, std::string_view help,
                           std::span<const double> bounds = {}) {
    return Family<Metric>(
        detail::AddFamily(handle_.get(), Metric::kKind, name, help, bounds),
        name);
  }

  template<class Metric>
  Metric Add(std::string_view name, std::string_view help,
             std::span<const double> bounds = {}) {
    return AddFamily<Metric>(name, help, bounds).Add();
  }

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
