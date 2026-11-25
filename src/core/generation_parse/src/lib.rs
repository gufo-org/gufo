//! Parses generated text into reasoning, content and tool calls, streamed
//! piece by piece or whole, as the OpenAI-compatible and Messages endpoints
//! report them. Every marker comes from the host (C++), which also keeps the
//! Qwen call scanner, the registered dialect scanners and the tool grammar.

#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod abi;
pub mod args;
pub mod config;
pub mod events;
pub mod framing;
pub mod markers;
pub mod markup;
pub mod parse;
pub mod quote;
pub mod reasoning;
pub mod stream;
pub mod text;

#[cfg(test)]
mod tests;

pub use abi::*;
pub use config::{Config, Error, Host, Initial, Markup, Mode, Reasoning, Tool, ToolChoice};
pub use events::{Event, Sink, Span};
pub use stream::Stream;
