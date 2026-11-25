#ifndef GUFO_METRICS_H_
#define GUFO_METRICS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust metrics registry, rendered in the Prometheus
   text format. src/core/metrics.hpp wraps it for C++.

   A family and a cell (one series' value) are borrowed from the registry that
   made them and stay valid until gufo_metrics_registry_free; free the registry
   only after every thread stopped using it. A registry renders its families in
   registration order. Registering and rendering lock; the update functions
   never lock or allocate. An update function applied to a cell of another
   kind (u64 on a float cell, observe on a counter) is ignored; the C++
   wrapper cannot make one. */
typedef struct GufoMetricsRegistry GufoMetricsRegistry;
typedef struct GufoMetricsFamily GufoMetricsFamily;
typedef struct GufoMetricsCell GufoMetricsCell;

/* What a family holds. Every kind but GUFO_METRICS_HISTOGRAM takes the
   updates named in its row; an update on a cell of another kind does
   nothing.
     COUNTER_U64, GAUGE_U64   gufo_metrics_u64_add, _set, _max
     COUNTER_F64, GAUGE_F64   gufo_metrics_f64_add, _set
     GAUGE_I64                gufo_metrics_i64_add, _set
     HISTOGRAM                gufo_metrics_histogram_observe */
typedef enum GufoMetricsKind {
  GUFO_METRICS_COUNTER_U64 = 0,
  GUFO_METRICS_COUNTER_F64 = 1,
  GUFO_METRICS_GAUGE_U64 = 2,
  GUFO_METRICS_GAUGE_F64 = 3,
  GUFO_METRICS_GAUGE_I64 = 4,
  GUFO_METRICS_HISTOGRAM = 5,
} GufoMetricsKind;

typedef enum GufoMetricsStatus {
  GUFO_METRICS_OK = 0,
  /* A metric or label name the text format cannot carry, a repeated or
     reserved label name ("le" on a histogram, a "__" prefix), non-UTF-8
     text, or histogram bounds that are empty, not finite, or not strictly
     increasing (and any bounds on another kind). */
  GUFO_METRICS_INVALID = 1,
  /* The family name, or the label set within the family, is registered. */
  GUFO_METRICS_DUPLICATE = 2,
  /* The render callback returned nonzero. */
  GUFO_METRICS_CALLBACK = 3,
  /* A caller bug (null pointer) or a registry bug. */
  GUFO_METRICS_INTERNAL = 4,
} GufoMetricsStatus;

typedef struct GufoMetricsLabel {
  const uint8_t* name;
  size_t name_len;
  const uint8_t* value;
  size_t value_len;
} GufoMetricsLabel;

/* Null on failure. */
GufoMetricsRegistry* gufo_metrics_registry_new(void);
void gufo_metrics_registry_free(GufoMetricsRegistry* registry);

/* Registers a family after the earlier ones. bounds are a histogram's upper
   bounds (the +Inf bucket is implicit); pass none for other kinds. help is
   escaped on output. */
GufoMetricsStatus gufo_metrics_family_new(
    const GufoMetricsRegistry* registry, GufoMetricsKind kind,
    const uint8_t* name, size_t name_len, const uint8_t* help, size_t help_len,
    const double* bounds, size_t bounds_len, const GufoMetricsFamily** out);

/* Adds a series with fixed labels to *family; no labels is one unlabeled
   series. Label values are escaped on output. */
GufoMetricsStatus gufo_metrics_series_new(const GufoMetricsFamily* family,
                                          const GufoMetricsLabel* labels,
                                          size_t labels_len,
                                          const GufoMetricsCell** out);

/* A hidden family, and a family with no series, renders nothing. Families
   start visible. */
void gufo_metrics_family_set_visible(const GufoMetricsFamily* family,
                                     bool visible);

/* Receives the whole exposition once, as bytes valid during the call; returns
   nonzero to fail the render with GUFO_METRICS_CALLBACK. */
typedef int (*GufoMetricsWriteFn)(void* context, const uint8_t* bytes,
                                  size_t len);
GufoMetricsStatus gufo_metrics_render(const GufoMetricsRegistry* registry,
                                      GufoMetricsWriteFn write, void* context);

/* Lock-free updates with relaxed ordering. max keeps the larger value. */
void gufo_metrics_u64_add(const GufoMetricsCell* cell, uint64_t delta);
void gufo_metrics_u64_set(const GufoMetricsCell* cell, uint64_t value);
void gufo_metrics_u64_max(const GufoMetricsCell* cell, uint64_t value);
void gufo_metrics_f64_add(const GufoMetricsCell* cell, double delta);
void gufo_metrics_f64_set(const GufoMetricsCell* cell, double value);
void gufo_metrics_i64_add(const GufoMetricsCell* cell, int64_t delta);
void gufo_metrics_i64_set(const GufoMetricsCell* cell, int64_t value);
void gufo_metrics_histogram_observe(const GufoMetricsCell* cell, double value);

#ifdef __cplusplus
}
#endif

#endif
