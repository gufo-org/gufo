#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod ffi;
mod format;
mod registry;

pub use registry::{Error, Family, Histogram, Kind, Registry, Series, Value};

#[cfg(test)]
mod tests;
