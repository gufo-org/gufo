//! Tool arguments: their JSON text, the schema checks that type Qwen's
//! untyped values, and Python literal recovery.

pub mod json_out;
pub mod python_literals;
pub mod schema;

/// One argument as the API reports it: a string, or JSON text
/// (`ChatMessage::ToolArgument`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Argument {
    pub name: String,
    pub value: String,
    pub is_string: bool,
}

/// A parsed call. `partial_arguments` is the append-only JSON prefix of a
/// call still being written; absent on complete calls.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Call {
    pub name: String,
    pub arguments: Vec<Argument>,
    pub partial_arguments: Option<String>,
}
