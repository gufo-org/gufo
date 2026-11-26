#ifndef GUFO_METRICS_H_
#define GUFO_METRICS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Families and cells are borrowed from their registry and stay valid until
   gufo_metrics_registry_free; free the registry only after every thread has
   stopped using it. */
typedef struct GufoMetricsRegistry GufoMetricsRegistry;
typedef struct GufoMetricsFamily GufoMetricsFamily;
typedef struct GufoMetricsCell GufoMetricsCell;

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
  GUFO_METRICS_INVALID = 1,
  GUFO_METRICS_DUPLICATE = 2,
  GUFO_METRICS_CALLBACK = 3,
  GUFO_METRICS_INTERNAL = 4,
} GufoMetricsStatus;

typedef struct GufoMetricsLabel {
  const uint8_t* name;
  size_t name_len;
  const uint8_t* value;
  size_t value_len;
} GufoMetricsLabel;

GufoMetricsRegistry* gufo_metrics_registry_new(void);
void gufo_metrics_registry_free(GufoMetricsRegistry* registry);

GufoMetricsStatus gufo_metrics_family_new(
    const GufoMetricsRegistry* registry, GufoMetricsKind kind,
    const uint8_t* name, size_t name_len, const uint8_t* help, size_t help_len,
    const double* bounds, size_t bounds_len, const GufoMetricsFamily** out);

GufoMetricsStatus gufo_metrics_series_new(const GufoMetricsFamily* family,
                                          const GufoMetricsLabel* labels,
                                          size_t labels_len,
                                          const GufoMetricsCell** out);

void gufo_metrics_family_set_visible(const GufoMetricsFamily* family,
                                     bool visible);

/* Receives the whole exposition once, as bytes valid during the call; returns
   nonzero to fail the render with GUFO_METRICS_CALLBACK. */
typedef int (*GufoMetricsWriteFn)(void* context, const uint8_t* bytes,
                                  size_t len);
GufoMetricsStatus gufo_metrics_render(const GufoMetricsRegistry* registry,
                                      GufoMetricsWriteFn write, void* context);

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
