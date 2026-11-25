//! What the parser reports, and where content came from.

/// Bytes `[begin, begin + size)` of the decoded response
/// (`TextSourceSpan`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Span {
    pub begin: usize,
    pub size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event<'a> {
    /// Streamed reasoning; `spans` locate it in the response when known.
    ReasoningDelta {
        text: &'a [u8],
        spans: &'a [Span],
    },
    /// Streamed content.
    ContentDelta {
        text: &'a [u8],
        spans: &'a [Span],
    },
    /// The newly parsed suffix of call `index`'s arguments. `begin` opens the
    /// call and carries its `name`; `complete` closes it.
    ToolDelta {
        index: usize,
        name: &'a [u8],
        arguments: &'a [u8],
        begin: bool,
        complete: bool,
    },
    /// The final parse, after any deltas: reasoning, content, each call,
    /// then the end.
    FinalReasoning {
        text: &'a [u8],
    },
    FinalContent {
        text: &'a [u8],
        spans: &'a [Span],
    },
    FinalCall {
        index: usize,
        name: &'a [u8],
        arguments: &'a [u8],
    },
    FinalEnd {
        hide_tool_markup: bool,
    },
}

/// Receives events; false stops the parse.
pub trait Sink {
    fn event(&mut self, event: Event<'_>) -> bool;
}

impl<F: FnMut(Event<'_>) -> bool> Sink for F {
    fn event(&mut self, event: Event<'_>) -> bool {
        self(event)
    }
}
