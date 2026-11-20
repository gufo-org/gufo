//! How compilation fails.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// The schema is invalid or unsupported: the reason, without the
    /// "JSON Schema: " prefix.
    Rejected(String),
    /// The schema is valid but no value satisfies it.
    Empty(String),
    /// An allocation sized by the input failed.
    NoMemory,
    /// A caller callback failed; it holds the cause.
    Stopped,
}

pub type Result<T> = std::result::Result<T, Failure>;

pub fn invalid<T>(reason: impl Into<String>) -> Result<T> {
    Err(Failure::Rejected(reason.into()))
}

pub fn empty<T>(reason: impl Into<String>) -> Result<T> {
    Err(Failure::Empty(reason.into()))
}
