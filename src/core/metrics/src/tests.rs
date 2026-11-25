use std::ffi::c_void;
use std::sync::Arc;

use crate::{Error, Kind, Registry};

fn series(registry: &Registry, kind: Kind, name: &str, help: &str) -> Arc<crate::Series> {
    registry
        .add_family(kind, name, help, &[])
        .unwrap()
        .add_series(&[])
        .unwrap()
}

#[test]
fn renders_in_registration_order_with_help_and_type() {
    let registry = Registry::new();
    series(
        &registry,
        Kind::GaugeI64,
        "b_gauge",
        "second registered first",
    )
    .value
    .set_i64(-4);
    let counter = series(&registry, Kind::CounterU64, "a_total", "Counts things");
    counter.value.add_u64(2);
    counter.value.add_u64(3);
    assert_eq!(
        registry.render(),
        "# HELP b_gauge second registered first\n# TYPE b_gauge gauge\nb_gauge -4\n\
         # HELP a_total Counts things\n# TYPE a_total counter\na_total 5\n"
    );
}

#[test]
fn escapes_help_and_label_values() {
    let registry = Registry::new();
    let family = registry
        .add_family(
            Kind::GaugeU64,
            "m",
            "back\\slash and\nnewline \"quoted\"",
            &[],
        )
        .unwrap();
    family
        .add_series(&[("path", "a\\b\"c\nd"), ("kind", "x")])
        .unwrap()
        .value
        .set_u64(1);
    assert_eq!(
        registry.render(),
        "# HELP m back\\\\slash and\\nnewline \"quoted\"\n# TYPE m gauge\n\
         m{path=\"a\\\\b\\\"c\\nd\",kind=\"x\"} 1\n"
    );
}

#[test]
fn u64_max_set_and_f64_add_and_set() {
    let registry = Registry::new();
    let high = series(&registry, Kind::CounterU64, "high", "h");
    for value in [5, 9, 7] {
        high.value.max_u64(value);
    }
    let seconds = series(&registry, Kind::CounterF64, "seconds", "s");
    seconds.value.add_f64(0.1);
    seconds.value.add_f64(0.2);
    let ratio = series(&registry, Kind::GaugeF64, "ratio", "r");
    ratio.value.set_f64(1234.5678);
    ratio.value.add_f64(0.0);
    let text = registry.render();
    assert!(text.contains("\nhigh 9\n"), "{text}");
    assert!(text.contains("\nseconds 0.30000000000000004\n"), "{text}");
    assert!(text.contains("\nratio 1234.5678\n"), "{text}");
}

#[test]
fn an_update_on_a_cell_of_another_kind_is_ignored() {
    let registry = Registry::new();
    let cell = series(&registry, Kind::CounterU64, "c", "c");
    cell.value.add_f64(1.5);
    cell.value.set_i64(9);
    cell.value.observe(1.0);
    assert!(registry.render().ends_with("\nc 0\n"));
}

#[test]
fn non_finite_observations_are_counted_but_never_summed() {
    let registry = Registry::new();
    let cell = registry
        .add_family(Kind::Histogram, "h", "h", &[1.0])
        .unwrap()
        .add_series(&[])
        .unwrap();
    for value in [0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        cell.value.observe(value);
    }
    assert_eq!(
        registry.render(),
        "# HELP h h\n# TYPE h histogram\n\
         h_bucket{le=\"1\"} 2\n\
         h_bucket{le=\"+Inf\"} 4\n\
         h_sum 0.5\n\
         h_count 4\n"
    );
}

#[test]
fn histogram_buckets_are_cumulative() {
    let registry = Registry::new();
    let histogram = registry
        .add_family(
            Kind::Histogram,
            "latency_seconds",
            "Latency",
            &[0.1, 0.5, 1.0],
        )
        .unwrap();
    let ok = histogram.add_series(&[("status", "ok")]).unwrap();
    for value in [0.05, 0.1, 0.3, 0.5, 2.0, f64::NAN] {
        ok.value.observe(value);
    }
    let empty = histogram.add_series(&[("status", "error")]).unwrap();
    empty.value.observe(0.7);
    assert_eq!(
        registry.render(),
        "# HELP latency_seconds Latency\n# TYPE latency_seconds histogram\n\
         latency_seconds_bucket{status=\"ok\",le=\"0.1\"} 2\n\
         latency_seconds_bucket{status=\"ok\",le=\"0.5\"} 4\n\
         latency_seconds_bucket{status=\"ok\",le=\"1\"} 4\n\
         latency_seconds_bucket{status=\"ok\",le=\"+Inf\"} 6\n\
         latency_seconds_sum{status=\"ok\"} 2.95\n\
         latency_seconds_count{status=\"ok\"} 6\n\
         latency_seconds_bucket{status=\"error\",le=\"0.1\"} 0\n\
         latency_seconds_bucket{status=\"error\",le=\"0.5\"} 0\n\
         latency_seconds_bucket{status=\"error\",le=\"1\"} 1\n\
         latency_seconds_bucket{status=\"error\",le=\"+Inf\"} 1\n\
         latency_seconds_sum{status=\"error\"} 0.7\n\
         latency_seconds_count{status=\"error\"} 1\n"
    );
}

#[test]
fn unlabeled_histogram_has_only_the_le_label() {
    let registry = Registry::new();
    let cell = registry
        .add_family(Kind::Histogram, "h", "h", &[1.0])
        .unwrap()
        .add_series(&[])
        .unwrap();
    cell.value.observe(0.5);
    cell.value.observe(1.5);
    assert_eq!(
        registry.render(),
        "# HELP h h\n# TYPE h histogram\nh_bucket{le=\"1\"} 1\nh_bucket{le=\"+Inf\"} 2\n\
         h_sum 2\nh_count 2\n"
    );
}

#[test]
fn concurrent_updates_lose_nothing() {
    const THREADS: u64 = 8;
    const PER_THREAD: u64 = 20_000;
    let registry = Registry::new();
    let count = series(&registry, Kind::CounterU64, "count", "c");
    let seconds = series(&registry, Kind::CounterF64, "seconds", "s");
    let signed = series(&registry, Kind::GaugeI64, "signed", "i");
    let high = series(&registry, Kind::CounterU64, "high", "h");
    let histogram = registry
        .add_family(Kind::Histogram, "h", "h", &[10_000.0])
        .unwrap()
        .add_series(&[])
        .unwrap();
    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let (count, seconds, signed, high, histogram) =
                (&count, &seconds, &signed, &high, &histogram);
            scope.spawn(move || {
                for step in 0..PER_THREAD {
                    count.value.add_u64(1);
                    seconds.value.add_f64(0.5);
                    signed.value.add_i64(if thread % 2 == 0 { 2 } else { -1 });
                    high.value.max_u64(thread * PER_THREAD + step);
                    histogram.value.observe(1.0);
                }
            });
        }
    });
    let total = THREADS * PER_THREAD;
    let text = registry.render();
    assert!(text.contains(&format!("\ncount {total}\n")), "{text}");
    assert!(
        text.contains(&format!("\nseconds {}\n", total / 2)),
        "{text}"
    );
    let signed_total = (THREADS / 2) * PER_THREAD * 2 - (THREADS / 2) * PER_THREAD;
    assert!(
        text.contains(&format!("\nsigned {signed_total}\n")),
        "{text}"
    );
    assert!(text.contains(&format!("\nhigh {}\n", total - 1)), "{text}");
    assert!(
        text.contains(&format!("h_bucket{{le=\"+Inf\"}} {total}\nh_sum {total}\n")),
        "{text}"
    );
}

#[test]
fn duplicate_names_and_label_sets_are_rejected() {
    let registry = Registry::new();
    let family = registry.add_family(Kind::CounterU64, "m", "", &[]).unwrap();
    assert_eq!(
        registry.add_family(Kind::GaugeU64, "m", "", &[]).err(),
        Some(Error::Duplicate)
    );
    family.add_series(&[("a", "1")]).unwrap();
    assert_eq!(
        family.add_series(&[("a", "1")]).err(),
        Some(Error::Duplicate)
    );
    family.add_series(&[("a", "2")]).unwrap();
}

#[test]
fn label_order_does_not_make_a_series_distinct() {
    let registry = Registry::new();
    let family = registry.add_family(Kind::CounterU64, "m", "", &[]).unwrap();
    family.add_series(&[("a", "1"), ("b", "2")]).unwrap();
    assert_eq!(
        family.add_series(&[("b", "2"), ("a", "1")]).err(),
        Some(Error::Duplicate)
    );
    family.add_series(&[("b", "1"), ("a", "2")]).unwrap();
    assert!(
        registry
            .render()
            .contains("m{a=\"1\",b=\"2\"} 0\nm{b=\"1\",a=\"2\"} 0\n")
    );
}

#[test]
fn a_series_must_use_the_label_names_of_the_first() {
    let registry = Registry::new();
    let family = registry.add_family(Kind::CounterU64, "m", "", &[]).unwrap();
    family.add_series(&[("a", "1")]).unwrap();
    for labels in [&[][..], &[("b", "1")], &[("a", "2"), ("b", "1")]] {
        assert_eq!(family.add_series(labels).err(), Some(Error::Invalid));
    }
    family.add_series(&[("a", "2")]).unwrap();
}

#[test]
fn histogram_sample_names_are_reserved() {
    let registry = Registry::new();
    registry
        .add_family(Kind::Histogram, "h_seconds", "", &[1.0])
        .unwrap();
    for name in [
        "h_seconds",
        "h_seconds_bucket",
        "h_seconds_sum",
        "h_seconds_count",
    ] {
        assert_eq!(
            registry.add_family(Kind::CounterU64, name, "", &[]).err(),
            Some(Error::Duplicate),
            "{name}"
        );
    }
    registry
        .add_family(Kind::CounterU64, "h_seconds_total", "", &[])
        .unwrap();

    registry
        .add_family(Kind::CounterU64, "g_count", "", &[])
        .unwrap();
    assert_eq!(
        registry.add_family(Kind::Histogram, "g", "", &[1.0]).err(),
        Some(Error::Duplicate)
    );
}

#[test]
fn invalid_names_labels_and_bounds_are_rejected() {
    let registry = Registry::new();
    for name in ["", "1m", "a-b", "a b", "é"] {
        assert_eq!(
            registry.add_family(Kind::GaugeU64, name, "", &[]).err(),
            Some(Error::Invalid)
        );
    }
    for bounds in [
        &[][..],
        &[1.0, 1.0],
        &[2.0, 1.0],
        &[f64::INFINITY],
        &[f64::NAN],
    ] {
        assert_eq!(
            registry.add_family(Kind::Histogram, "h", "", bounds).err(),
            Some(Error::Invalid)
        );
    }
    assert_eq!(
        registry.add_family(Kind::GaugeU64, "g", "", &[1.0]).err(),
        Some(Error::Invalid)
    );
    let counter = registry.add_family(Kind::CounterU64, "c", "", &[]).unwrap();
    for labels in [
        &[("1a", "x")][..],
        &[("a:b", "x")],
        &[("__x", "x")],
        &[("a", "1"), ("a", "2")],
    ] {
        assert_eq!(counter.add_series(labels).err(), Some(Error::Invalid));
    }
    let histogram = registry
        .add_family(Kind::Histogram, "h", "", &[1.0])
        .unwrap();
    assert_eq!(
        histogram.add_series(&[("le", "1")]).err(),
        Some(Error::Invalid)
    );
    counter.add_series(&[("le", "1")]).unwrap();
    assert_eq!(
        registry.render(),
        "# HELP c \n# TYPE c counter\nc{le=\"1\"} 0\n"
    );
}

#[test]
fn hidden_and_empty_families_render_nothing() {
    let registry = Registry::new();
    let hidden = registry
        .add_family(Kind::GaugeU64, "hidden", "h", &[])
        .unwrap();
    hidden.add_series(&[]).unwrap().value.set_u64(1);
    registry
        .add_family(Kind::GaugeU64, "empty", "e", &[])
        .unwrap();
    series(&registry, Kind::GaugeU64, "shown", "s");
    hidden.set_visible(false);
    assert_eq!(
        registry.render(),
        "# HELP shown s\n# TYPE shown gauge\nshown 0\n"
    );
    hidden.set_visible(true);
    assert!(registry.render().starts_with("# HELP hidden h\n"));
}

#[allow(unsafe_code)]
mod c_interface {
    use super::*;
    use crate::ffi::*;

    extern "C" fn collect(context: *mut c_void, bytes: *const u8, len: usize) -> i32 {
        // SAFETY: the test passes a `String` and render gives `len` bytes.
        let out = unsafe { &mut *context.cast::<String>() };
        // SAFETY: render hands a valid buffer for the call.
        out.push_str(
            std::str::from_utf8(unsafe { std::slice::from_raw_parts(bytes, len) }).unwrap(),
        );
        0
    }

    extern "C" fn refuse(_: *mut c_void, _: *const u8, _: usize) -> i32 {
        1
    }

    #[test]
    fn registers_updates_and_renders() {
        let registry = gufo_metrics_registry_new();
        let mut family: *const c_void = std::ptr::null();
        let mut cell: *const c_void = std::ptr::null();
        let label = GufoMetricsLabel {
            name: b"k".as_ptr(),
            name_len: 1,
            value: b"v".as_ptr(),
            value_len: 1,
        };
        // SAFETY: every pointer is live and every length matches its buffer.
        unsafe {
            assert_eq!(
                gufo_metrics_family_new(
                    registry,
                    0,
                    b"n".as_ptr(),
                    1,
                    b"h".as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    &mut family
                ),
                0
            );
            assert_eq!(
                gufo_metrics_family_new(
                    registry,
                    0,
                    b"n".as_ptr(),
                    1,
                    b"h".as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    &mut family
                ),
                2
            );
            assert_eq!(
                gufo_metrics_family_new(
                    registry,
                    9,
                    b"x".as_ptr(),
                    1,
                    b"h".as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    &mut family
                ),
                1
            );
            assert_eq!(gufo_metrics_series_new(family, &label, 1, &mut cell), 0);
            gufo_metrics_u64_add(cell, 41);
            gufo_metrics_u64_add(cell, 1);
            gufo_metrics_f64_add(cell, 1.0);
            gufo_metrics_u64_add(std::ptr::null(), 1);
            let mut text = String::new();
            assert_eq!(
                gufo_metrics_render(registry, Some(collect), (&raw mut text).cast()),
                0
            );
            assert_eq!(text, "# HELP n h\n# TYPE n counter\nn{k=\"v\"} 42\n");
            assert_eq!(
                gufo_metrics_render(registry, Some(refuse), std::ptr::null_mut()),
                3
            );
            assert_eq!(gufo_metrics_render(registry, None, std::ptr::null_mut()), 4);
            gufo_metrics_family_set_visible(family, false);
            let mut hidden = String::new();
            assert_eq!(
                gufo_metrics_render(registry, Some(collect), (&raw mut hidden).cast()),
                0
            );
            assert_eq!(hidden, "");
            gufo_metrics_registry_free(registry);
            gufo_metrics_registry_free(std::ptr::null_mut());
        }
    }

    fn new_family(registry: *const c_void, name: &[u8]) -> (u32, *const c_void) {
        let mut family: *const c_void = std::ptr::null();
        // SAFETY: the registry is live, the name buffer matches its length,
        // and `family` is writable.
        let status = unsafe {
            gufo_metrics_family_new(
                registry,
                0,
                name.as_ptr(),
                name.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut family,
            )
        };
        (status, family)
    }

    #[test]
    fn rejects_a_null_registry_family_or_output() {
        let mut out: *const c_void = std::ptr::null();
        // SAFETY: null handles are the case under test; `out` is writable.
        unsafe {
            assert_eq!(
                gufo_metrics_family_new(
                    std::ptr::null(),
                    0,
                    b"n".as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    &mut out
                ),
                4
            );
            assert_eq!(
                gufo_metrics_series_new(std::ptr::null(), std::ptr::null(), 0, &mut out),
                4
            );
            assert_eq!(
                gufo_metrics_render(std::ptr::null(), Some(collect), std::ptr::null_mut()),
                4
            );
            gufo_metrics_family_set_visible(std::ptr::null(), false);
            gufo_metrics_histogram_observe(std::ptr::null(), 1.0);
            gufo_metrics_i64_set(std::ptr::null(), 1);
        }
        let registry = gufo_metrics_registry_new();
        // SAFETY: the registry is live and freed once; a null `out` is the
        // case under test.
        unsafe {
            assert_eq!(
                gufo_metrics_family_new(
                    registry,
                    0,
                    b"n".as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut()
                ),
                4
            );
            gufo_metrics_registry_free(registry);
        }
    }

    #[test]
    fn rejects_kinds_beyond_the_last() {
        let registry = gufo_metrics_registry_new();
        for kind_value in [6, 7, u32::MAX] {
            let mut family: *const c_void = std::ptr::null();
            // SAFETY: the registry is live and `family` is writable.
            let status = unsafe {
                gufo_metrics_family_new(
                    registry,
                    kind_value,
                    b"n".as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    &mut family,
                )
            };
            assert_eq!(status, 1, "kind {kind_value}");
        }
        // SAFETY: the registry is live and freed once.
        unsafe { gufo_metrics_registry_free(registry) };
    }

    #[test]
    fn rejects_invalid_utf8_names() {
        let registry = gufo_metrics_registry_new();
        assert_eq!(new_family(registry, &[0xff, 0xfe]).0, 1);
        let (status, family) = new_family(registry, b"n");
        assert_eq!(status, 0);
        let mut cell: *const c_void = std::ptr::null();
        let label = GufoMetricsLabel {
            name: [0xffu8].as_ptr(),
            name_len: 1,
            value: b"v".as_ptr(),
            value_len: 1,
        };
        // SAFETY: the family is live, the label buffers match their lengths,
        // and `cell` is writable.
        unsafe {
            assert_eq!(gufo_metrics_series_new(family, &label, 1, &mut cell), 1);
            gufo_metrics_registry_free(registry);
        }
    }

    #[test]
    fn accepts_null_pointers_with_zero_lengths() {
        let registry = gufo_metrics_registry_new();
        let mut family: *const c_void = std::ptr::null();
        let mut cell: *const c_void = std::ptr::null();
        let empty_text = GufoMetricsLabel {
            name: b"k".as_ptr(),
            name_len: 1,
            value: std::ptr::null(),
            value_len: 0,
        };
        // SAFETY: the registry and family are live, every nonzero length
        // matches its buffer, and the outputs are writable.
        unsafe {
            assert_eq!(
                gufo_metrics_family_new(
                    registry,
                    0,
                    b"n".as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    &mut family
                ),
                0
            );
            assert_eq!(
                gufo_metrics_series_new(family, std::ptr::null(), 0, &mut cell),
                0
            );
            assert_eq!(
                gufo_metrics_series_new(family, &empty_text, 1, &mut cell),
                1
            );
            let mut nonnull_len_without_data = GufoMetricsLabel {
                name: std::ptr::null(),
                name_len: 1,
                value: std::ptr::null(),
                value_len: 0,
            };
            assert_eq!(
                gufo_metrics_series_new(family, &raw mut nonnull_len_without_data, 1, &mut cell),
                1
            );
            gufo_metrics_registry_free(registry);
        }
    }

    #[test]
    fn a_failing_write_callback_reports_callback() {
        let registry = gufo_metrics_registry_new();
        // SAFETY: the registry is live and freed once; `refuse` ignores its
        // arguments.
        unsafe {
            assert_eq!(
                gufo_metrics_render(registry, Some(refuse), std::ptr::null_mut()),
                3
            );
            gufo_metrics_registry_free(registry);
        }
    }
}
