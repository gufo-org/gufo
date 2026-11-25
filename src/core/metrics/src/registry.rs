use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::format;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    CounterU64,
    CounterF64,
    GaugeU64,
    GaugeF64,
    GaugeI64,
    Histogram,
}

impl Kind {
    fn type_name(self) -> &'static str {
        match self {
            Kind::CounterU64 | Kind::CounterF64 => "counter",
            Kind::GaugeU64 | Kind::GaugeF64 | Kind::GaugeI64 => "gauge",
            Kind::Histogram => "histogram",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Invalid,
    Duplicate,
}

pub struct Histogram {
    bounds: Box<[f64]>,
    buckets: Box<[AtomicU64]>,
    sum: AtomicF64,
}

impl Histogram {
    fn new(bounds: &[f64]) -> Self {
        Self {
            bounds: bounds.into(),
            buckets: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum: AtomicF64::new(0.0),
        }
    }

    pub fn observe(&self, value: f64) {
        let bucket = if value.is_nan() {
            self.bounds.len()
        } else {
            self.bounds.partition_point(|&bound| bound < value)
        };
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        if value.is_finite() {
            self.sum.add(value);
        }
    }
}

pub enum Value {
    U64(AtomicU64),
    F64(AtomicF64),
    I64(AtomicI64),
    Histogram(Histogram),
}

pub struct AtomicF64(AtomicU64);

impl AtomicF64 {
    fn new(value: f64) -> Self {
        Self(AtomicU64::new(value.to_bits()))
    }

    fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    fn store(&self, value: f64) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }

    fn add(&self, delta: f64) {
        let mut current = self.0.load(Ordering::Relaxed);
        while let Err(seen) = self.0.compare_exchange_weak(
            current,
            (f64::from_bits(current) + delta).to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            current = seen;
        }
    }
}

impl Value {
    fn new(kind: Kind, bounds: &[f64]) -> Self {
        match kind {
            Kind::CounterU64 | Kind::GaugeU64 => Value::U64(AtomicU64::new(0)),
            Kind::CounterF64 | Kind::GaugeF64 => Value::F64(AtomicF64::new(0.0)),
            Kind::GaugeI64 => Value::I64(AtomicI64::new(0)),
            Kind::Histogram => Value::Histogram(Histogram::new(bounds)),
        }
    }

    pub fn add_u64(&self, delta: u64) {
        if let Value::U64(cell) = self {
            cell.fetch_add(delta, Ordering::Relaxed);
        }
    }

    pub fn set_u64(&self, value: u64) {
        if let Value::U64(cell) = self {
            cell.store(value, Ordering::Relaxed);
        }
    }

    pub fn max_u64(&self, value: u64) {
        if let Value::U64(cell) = self {
            cell.fetch_max(value, Ordering::Relaxed);
        }
    }

    pub fn add_f64(&self, delta: f64) {
        if let Value::F64(cell) = self {
            cell.add(delta);
        }
    }

    pub fn set_f64(&self, value: f64) {
        if let Value::F64(cell) = self {
            cell.store(value);
        }
    }

    pub fn add_i64(&self, delta: i64) {
        if let Value::I64(cell) = self {
            cell.fetch_add(delta, Ordering::Relaxed);
        }
    }

    pub fn set_i64(&self, value: i64) {
        if let Value::I64(cell) = self {
            cell.store(value, Ordering::Relaxed);
        }
    }

    pub fn observe(&self, value: f64) {
        if let Value::Histogram(histogram) = self {
            histogram.observe(value);
        }
    }
}

pub struct Series {
    label_text: String,
    sorted_labels: Vec<(String, String)>,
    pub value: Value,
}

pub struct Family {
    name: String,
    help: String,
    kind: Kind,
    bounds: Box<[f64]>,
    visible: AtomicBool,
    series: Mutex<Vec<Arc<Series>>>,
}

impl Family {
    pub fn set_visible(&self, visible: bool) {
        self.visible.store(visible, Ordering::Relaxed);
    }

    pub fn add_series(&self, labels: &[(&str, &str)]) -> Result<Arc<Series>, Error> {
        let mut sorted_labels: Vec<(String, String)> = Vec::with_capacity(labels.len());
        let mut label_text = String::new();
        for (name, value) in labels {
            let reserved =
                name.starts_with("__") || (self.kind == Kind::Histogram && *name == "le");
            if !valid_name(name, false) || reserved {
                return Err(Error::Invalid);
            }
            sorted_labels.push((name.to_string(), value.to_string()));
            if !label_text.is_empty() {
                label_text.push(',');
            }
            label_text.push_str(name);
            label_text.push_str("=\"");
            escape(value, true, &mut label_text);
            label_text.push('"');
        }
        sorted_labels.sort();
        if sorted_labels.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(Error::Invalid);
        }
        let mut series = lock(&self.series);
        if let Some(first) = series.first() {
            let names = first.sorted_labels.iter().map(|(name, _)| name);
            if !names.eq(sorted_labels.iter().map(|(name, _)| name)) {
                return Err(Error::Invalid);
            }
        }
        if series
            .iter()
            .any(|existing| existing.sorted_labels == sorted_labels)
        {
            return Err(Error::Duplicate);
        }
        let added = Arc::new(Series {
            label_text,
            sorted_labels,
            value: Value::new(self.kind, &self.bounds),
        });
        series.push(Arc::clone(&added));
        Ok(added)
    }
}

pub struct Registry {
    families: Mutex<Vec<Arc<Family>>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn valid_name(name: &str, allow_colon: bool) -> bool {
    let mut chars = name.chars();
    let head = |c: char| c.is_ascii_alphabetic() || c == '_' || (allow_colon && c == ':');
    chars.next().is_some_and(head) && chars.all(|c| head(c) || c.is_ascii_digit())
}

fn escape(text: &str, quote: bool, out: &mut String) {
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '"' if quote => out.push_str("\\\""),
            c => out.push(c),
        }
    }
}

impl Registry {
    pub fn new() -> Self {
        Self {
            families: Mutex::new(Vec::new()),
        }
    }

    pub fn add_family(
        &self,
        kind: Kind,
        name: &str,
        help: &str,
        bounds: &[f64],
    ) -> Result<Arc<Family>, Error> {
        let sorted = bounds.windows(2).all(|pair| pair[0] < pair[1]);
        let finite = bounds.iter().all(|bound| bound.is_finite());
        let bounds_fit = if kind == Kind::Histogram {
            !bounds.is_empty() && sorted && finite
        } else {
            bounds.is_empty()
        };
        if !valid_name(name, true) || !bounds_fit {
            return Err(Error::Invalid);
        }
        let mut families = lock(&self.families);
        let taken = |family: &Family| {
            let derived = |histogram: bool, name: &str, candidate: &str| {
                name == candidate
                    || (histogram
                        && ["_bucket", "_sum", "_count"]
                            .iter()
                            .any(|suffix| candidate.strip_suffix(suffix) == Some(name)))
            };
            derived(family.kind == Kind::Histogram, &family.name, name)
                || derived(kind == Kind::Histogram, name, &family.name)
        };
        if families.iter().any(|family| taken(family)) {
            return Err(Error::Duplicate);
        }
        let family = Arc::new(Family {
            name: name.to_string(),
            help: help.to_string(),
            kind,
            bounds: bounds.into(),
            visible: AtomicBool::new(true),
            series: Mutex::new(Vec::new()),
        });
        families.push(Arc::clone(&family));
        Ok(family)
    }

    pub fn render(&self) -> String {
        let families: Vec<Arc<Family>> = lock(&self.families).clone();
        let mut out = String::new();
        for family in families {
            if !family.visible.load(Ordering::Relaxed) {
                continue;
            }
            let series: Vec<Arc<Series>> = lock(&family.series).clone();
            if series.is_empty() {
                continue;
            }
            out.push_str("# HELP ");
            out.push_str(&family.name);
            out.push(' ');
            escape(&family.help, false, &mut out);
            out.push_str("\n# TYPE ");
            out.push_str(&family.name);
            out.push(' ');
            out.push_str(family.kind.type_name());
            out.push('\n');
            for series in &series {
                render_series(&family.name, series, &mut out);
            }
        }
        out
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

fn sample_line(name: &str, suffix: &str, labels: &str, value: &str, out: &mut String) {
    out.push_str(name);
    out.push_str(suffix);
    if !labels.is_empty() {
        out.push('{');
        out.push_str(labels);
        out.push('}');
    }
    out.push(' ');
    out.push_str(value);
    out.push('\n');
}

fn render_series(name: &str, series: &Series, out: &mut String) {
    let labels = series.label_text.as_str();
    match &series.value {
        Value::U64(cell) => sample_line(
            name,
            "",
            labels,
            &cell.load(Ordering::Relaxed).to_string(),
            out,
        ),
        Value::F64(cell) => sample_line(name, "", labels, &format::printf_17g(cell.load()), out),
        Value::I64(cell) => sample_line(
            name,
            "",
            labels,
            &cell.load(Ordering::Relaxed).to_string(),
            out,
        ),
        Value::Histogram(histogram) => {
            let mut cumulative = 0u64;
            let bounds = histogram
                .bounds
                .iter()
                .map(|&bound| format::shortest_round_trip(bound))
                .chain(std::iter::once("+Inf".to_string()));
            for (bound, bucket) in bounds.zip(histogram.buckets.iter()) {
                cumulative += bucket.load(Ordering::Relaxed);
                let mut with_le = String::from(labels);
                if !with_le.is_empty() {
                    with_le.push(',');
                }
                let _ = write!(with_le, "le=\"{bound}\"");
                sample_line(name, "_bucket", &with_le, &cumulative.to_string(), out);
            }
            let sum = format::shortest_round_trip(histogram.sum.load());
            sample_line(name, "_sum", labels, &sum, out);
            sample_line(name, "_count", labels, &cumulative.to_string(), out);
        }
    }
}
