//! How a decode ends early: a client rejection (400-class) or an internal
//! failure (500-class).

/// A rejected request, as the C++ `Error` (src/cli/serve/openai_chat.cpp) and
/// `Err` (src/cli/serve/http_server.cpp) responses spell it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub status: i32,
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// A check failed with this message; the caller names the status and
    /// code, as a C++ helper's `*error = ...; return false` did. The message
    /// may be empty.
    Invalid(String),
    Rejected(Rejection),
    /// A C++ helper rejected the request; the host kept its response.
    HostRejected,
    NoMemory,
    /// A callback failed; the host recorded why.
    Stopped,
    /// A caller or decoder bug.
    Internal,
}

pub type Outcome<T> = Result<T, Failure>;

impl Failure {
    /// Turns `Invalid` into a rejection with `status` and `code`.
    pub fn with(self, status: i32, code: &'static str) -> Self {
        match self {
            Self::Invalid(message) => Self::Rejected(Rejection {
                status,
                code,
                message,
            }),
            other => other,
        }
    }
}

pub fn invalid<T>(message: impl Into<String>) -> Outcome<T> {
    Err(Failure::Invalid(message.into()))
}

/// A 400 rejection, as `Error(400, "Bad Request", message, code)` makes it.
pub fn bad_request(code: &'static str, message: impl Into<String>) -> Failure {
    Failure::Rejected(Rejection {
        status: 400,
        code,
        message: message.into(),
    })
}

/// `InvalidCompatibilityRequest(message)`, whose `const char*` message ends
/// at the first NUL.
pub fn compatibility(message: impl Into<String>) -> Failure {
    let mut message = message.into();
    if let Some(end) = message.find('\0') {
        message.truncate(end);
    }
    bad_request("invalid_request", message)
}

impl From<std::collections::TryReserveError> for Failure {
    fn from(_: std::collections::TryReserveError) -> Self {
        Self::NoMemory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_messages_end_at_nul() {
        assert_eq!(
            compatibility("unsupported member: a\0b"),
            bad_request("invalid_request", "unsupported member: a")
        );
        assert_eq!(
            Failure::Invalid("x".into()).with(404, "c"),
            Failure::Rejected(Rejection {
                status: 404,
                code: "c",
                message: "x".into()
            })
        );
        assert_eq!(Failure::Stopped.with(400, "c"), Failure::Stopped);
    }
}
