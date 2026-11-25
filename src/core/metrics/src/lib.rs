//! A metrics registry behind opaque C handles, rendered in the Prometheus
//! text format. See `include/gufo/metrics.h` and `src/core/metrics.hpp`.

#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod ffi;
mod format;
mod registry;

pub use registry::{Error, Family, Histogram, Kind, Registry, Series, Value};

#[cfg(test)]
mod tests;
