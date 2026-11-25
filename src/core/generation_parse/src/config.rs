//! What one request parses with: every marker comes from the host, which
//! also scans calls for the dialects it owns and checks grammars.

use gufo_json::{Limits, Value, parse_value};

/// What the prompt left open when generation started
/// (`ext::InitialOutputState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Initial {
    Auto,
    Reasoning,
    Content,
    ContentAfterReasoning,
}

/// Which parser reads the response: `Plain` for unconstrained output,
/// `Structured` under a response format, and `ToolOnly` under the tool
/// grammar without a response format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Plain,
    Structured,
    ToolOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
}

/// Client envelope tags (`<invoke name=` and its parameters and closers),
/// removed only when they name a declared tool or echo an accepted call.
#[derive(Debug, Clone, Default)]
pub struct Envelope {
    pub heads: Vec<Vec<u8>>,
    pub parameters: Vec<Vec<u8>>,
    pub closers: Vec<Vec<u8>>,
}

/// The call markup of the request's tool dialect.
#[derive(Debug, Clone, Default)]
pub struct Markup {
    /// Openers that start a call; nothing else does.
    pub markers: Vec<Vec<u8>>,
    /// Complete native call headers which end a constrained reasoning phase.
    pub reasoning_markers: Vec<Vec<u8>>,
    pub closers: Vec<Vec<u8>>,
    /// The host scans calls opened by `qwen_open` itself.
    pub qwen_calls: bool,
    /// A native grammar frames arguments before a value completes, so partial
    /// calls stream.
    pub native: bool,
    /// A registered dialect scans the other markers.
    pub dialect: bool,
    /// Whitespace after an explicit end of reasoning that is framing.
    pub reasoning_separator: Vec<u8>,
    /// The Qwen call opener (`QwenToolSyntax::kCallOpen`).
    pub qwen_open: Vec<u8>,
    pub envelope: Envelope,
}

/// Reasoning delimiters (`ext::ReasoningMarkup`).
#[derive(Debug, Clone, Default)]
pub struct Reasoning {
    pub open: Vec<u8>,
    pub close: Vec<u8>,
}

/// A declared tool and its parameter schema, parsed once.
#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    /// The schema, or `None` when its text is not JSON.
    pub schema: Option<Value>,
}

impl Tool {
    pub fn new(name: String, parameters_json: &[u8]) -> Result<Self, Error> {
        let schema = match parse_value(parameters_json, Limits::NONE) {
            Ok(value) => Some(value),
            Err(gufo_json::ValueError::NoMemory) => return Err(Error::NoMemory),
            Err(_) => None,
        };
        Ok(Self { name, schema })
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub initial: Initial,
    pub mode: Mode,
    pub choice: ToolChoice,
    /// Report the source byte ranges of final content.
    pub track_spans: bool,
    /// Streamed pieces are only buffered: tool deltas come from the final
    /// parse, not from a re-parse per piece.
    pub buffered: bool,
    pub reasoning: Reasoning,
    pub markup: Markup,
    pub tools: Vec<Tool>,
}

impl Config {
    /// Tool calls are recognized at all.
    pub fn recognize_tools(&self) -> bool {
        !self.tools.is_empty() && self.choice != ToolChoice::None
    }
}

/// Why `tool_choice: required` was not met: no declared call was produced,
/// or a constrained turn did not complete one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsatisfied {
    NotProduced,
    NotCompleted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    ToolChoiceUnsatisfied(Unsatisfied),
    /// A streamed tool call did not complete as the final parse read it.
    UnsafeStream,
    /// An event sink or host callback asked to stop.
    Stopped,
    NoMemory,
    Internal,
}

impl From<std::collections::TryReserveError> for Error {
    fn from(_: std::collections::TryReserveError) -> Self {
        Self::NoMemory
    }
}

/// One parameter of a Qwen call as written (`QwenToolCallParameter`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QwenParameter {
    pub name: Vec<u8>,
    pub value: Vec<u8>,
    pub framed: bool,
    pub progress: usize,
}

/// One Qwen call as written (`QwenToolCallSyntax`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QwenCall {
    pub name: Vec<u8>,
    pub parameters: Vec<QwenParameter>,
    pub unfinished: Option<QwenParameter>,
    pub native: bool,
    pub json_payload: Vec<u8>,
    pub canonical_header: bool,
    pub complete: bool,
}

/// One scan step (`QwenToolCallScan`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QwenScan {
    pub matched: bool,
    pub progress: usize,
    pub end: usize,
    pub call: Option<QwenCall>,
}

/// An argument a dialect scanner read (`ext::ScannedToolArgument`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScannedArgument {
    pub name: Vec<u8>,
    pub value: Vec<u8>,
    pub is_string: bool,
}

/// A call a dialect scanner read (`ext::ScannedToolCall`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScannedCall {
    pub name: Vec<u8>,
    pub arguments: Vec<ScannedArgument>,
    pub unfinished: Option<ScannedArgument>,
    pub valid: bool,
    pub complete: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DialectScan {
    pub calls: Vec<ScannedCall>,
    /// [begin, end) of each complete call or call block.
    pub spans: Vec<(usize, usize)>,
}

/// What stays in C++: the Qwen scanner paired with its renderer, the
/// registered dialect scanners, and the tool grammar.
pub trait Host {
    fn qwen_scan(&mut self, text: &[u8]) -> Result<QwenScan, Error>;
    fn dialect_scan(
        &mut self,
        text: &[u8],
        markers: &[&[u8]],
        partial: bool,
    ) -> Result<DialectScan, Error>;
    /// The tool at `tool`'s grammar (strict off) accepts `arguments` whole;
    /// false also when the grammar does not compile.
    fn grammar_accepts(&mut self, tool: usize, arguments: &[u8]) -> Result<bool, Error>;
}
