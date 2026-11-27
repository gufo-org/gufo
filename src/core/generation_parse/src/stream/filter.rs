//! The streamed text filter (`StreamingTextFilter`): separates reasoning
//! from content as pieces arrive, holds back bytes that may still open a
//! reasoning close, a call marker or client framing, and switches to tool
//! mode at the first call marker.

use super::utf8::Decoder;
use crate::config::{Config, Error, Initial, Mode};
use crate::events::{Event, Sink, Span};
use crate::framing::{content_before, hold};
use crate::markers::{earliest, held_prefix, is_prefix, reasoning_tool};
use crate::parse::append_spans;
use crate::quote::QuoteTracker;
use crate::reasoning::after_separator;
use crate::text::{NPOS, find, is_space, offset_in, trim};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Initial,
    Thinking,
    Content,
}

/// The bytes streamed so far, for the final tail.
#[derive(Debug, Default)]
struct Emitted {
    reasoning: usize,
    content: usize,
}

impl Emitted {
    fn mapped(
        &mut self,
        sink: &mut dyn Sink,
        piece: &[u8],
        reasoning: bool,
        spans: &[Span],
    ) -> Result<(), Error> {
        let event = if reasoning {
            Event::ReasoningDelta { text: piece, spans }
        } else {
            Event::ContentDelta { text: piece, spans }
        };
        if !sink.event(event) {
            return Err(Error::Stopped);
        }
        if reasoning {
            self.reasoning += piece.len();
        }
        Ok(())
    }

    /// `offset` locates `piece` in the response, when known.
    fn emit(
        &mut self,
        sink: &mut dyn Sink,
        piece: &[u8],
        reasoning: bool,
        offset: Option<usize>,
    ) -> Result<(), Error> {
        match offset {
            None => self.mapped(sink, piece, reasoning, &[]),
            Some(begin) => self.mapped(
                sink,
                piece,
                reasoning,
                &[Span {
                    begin,
                    size: piece.len(),
                }],
            ),
        }
    }
}

fn append(buffer: &mut Vec<u8>, bytes: &[u8]) -> Result<(), Error> {
    buffer.try_reserve(bytes.len())?;
    buffer.extend_from_slice(bytes);
    Ok(())
}

#[derive(Debug)]
pub struct Filter {
    decoder: Decoder,
    piece: Vec<u8>,
    raw: Vec<u8>,
    pending: Vec<u8>,
    hidden: Vec<u8>,
    state: State,
    tool_mode: bool,
    raw_content: bool,
    tool_only: bool,
    recognize_tools: bool,
    structured_started: bool,
    quotes: QuoteTracker,
    emitted: Emitted,
    trim_reasoning_separator: bool,
    reasoning_start: bool,
}

impl Filter {
    pub fn new(config: &Config) -> Self {
        let mut filter = Self {
            decoder: Decoder::default(),
            piece: Vec::new(),
            raw: Vec::new(),
            pending: Vec::new(),
            hidden: Vec::new(),
            state: State::Initial,
            tool_mode: false,
            raw_content: config.mode != Mode::Plain,
            tool_only: config.mode == Mode::ToolOnly,
            recognize_tools: config.recognize_tools(),
            structured_started: false,
            quotes: QuoteTracker::default(),
            emitted: Emitted::default(),
            trim_reasoning_separator: false,
            reasoning_start: false,
        };
        match config.initial {
            Initial::Reasoning => {
                filter.state = State::Thinking;
                filter.reasoning_start = true;
            }
            Initial::Content | Initial::ContentAfterReasoning => {
                filter.state = State::Content;
                // The prompt closed reasoning: the model writes the separator.
                filter.trim_reasoning_separator = config.initial == Initial::ContentAfterReasoning;
            }
            Initial::Auto => {}
        }
        filter
    }

    /// The decoded response so far.
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    pub fn in_tools(&self) -> bool {
        self.tool_mode
    }

    /// `pending` always holds a suffix of what the model generated, so the
    /// context of a marker inside it is decided against the whole response.
    fn pending_offset(&self) -> usize {
        self.raw.len() - self.pending.len()
    }

    pub fn push(
        &mut self,
        bytes: &[u8],
        last: bool,
        config: &Config,
        sink: &mut dyn Sink,
    ) -> Result<(), Error> {
        let mut piece = std::mem::take(&mut self.piece);
        piece.clear();
        let result = self.push_decoded(bytes, last, config, sink, &mut piece);
        self.piece = piece;
        result
    }

    fn push_decoded(
        &mut self,
        bytes: &[u8],
        last: bool,
        config: &Config,
        sink: &mut dyn Sink,
        piece: &mut Vec<u8>,
    ) -> Result<(), Error> {
        self.decoder.push(bytes, last, piece)?;
        append(&mut self.raw, piece)?;
        self.quotes.append(piece);
        if self.tool_mode {
            return append(&mut self.hidden, piece);
        }
        append(&mut self.pending, piece)?;
        let open = &config.reasoning.open;
        if self.reasoning_start {
            // A prompt-opened reasoning phase may still emit its opening tag.
            // Match the buffered parser without exposing a split opener as
            // thought.
            if !last && open.starts_with(&self.pending) && self.pending.len() < open.len() {
                return Ok(());
            }
            if self.pending.starts_with(open) {
                self.pending.drain(..open.len());
            }
            self.reasoning_start = false;
        }
        // Content can open a later thought (`Reasoning::repeats`), which then
        // runs through the thinking state again.
        while self.process(config, sink)? {}
        Ok(())
    }

    /// One pass over the states; true when content opened a later thought
    /// and the rest of `pending` is reasoning to process.
    fn process(&mut self, config: &Config, sink: &mut dyn Sink) -> Result<bool, Error> {
        let markup = &config.markup;
        let open = &config.reasoning.open;
        let close = &config.reasoning.close;
        if self.state == State::Initial {
            let skipped = self
                .pending
                .iter()
                .take_while(|&&byte| is_space(byte))
                .count();
            let view = &self.pending[skipped..];
            if view.is_empty() {
                return Ok(false);
            }
            if !open.is_empty() && open.starts_with(view) {
                if view == open.as_slice() {
                    self.state = State::Thinking;
                    self.pending.clear();
                }
                return Ok(false);
            }
            if !open.is_empty() && view.starts_with(open) {
                // A buffered result arrives as one piece: the opening tag
                // together with the reasoning after it.
                self.state = State::Thinking;
                self.pending.drain(..skipped + open.len());
            } else {
                self.state = State::Content;
            }
        }

        if self.state == State::Thinking {
            let offset = self.pending_offset();
            let think_found = find(&self.raw, close, offset);
            let end_pos = if think_found == NPOS {
                NPOS
            } else {
                think_found - offset
            };
            let thinking_markers = if self.raw_content {
                &markup.reasoning_markers
            } else {
                &markup.markers
            };
            let found = if !self.recognize_tools {
                NPOS
            } else if self.raw_content {
                reasoning_tool(&self.raw, thinking_markers, &self.quotes, offset)
            } else {
                earliest(&self.raw, thinking_markers, &self.quotes, offset)
            };
            let marker = if found == NPOS { NPOS } else { found - offset };
            if marker < end_pos {
                if marker > 0 {
                    self.emitted
                        .emit(sink, &self.pending[..marker], true, None)?;
                }
                self.hidden.clear();
                append(&mut self.hidden, &self.pending[marker..])?;
                self.pending.clear();
                self.tool_mode = true;
                return Ok(false);
            }
            if end_pos != NPOS {
                if end_pos > 0 {
                    self.emitted
                        .emit(sink, &self.pending[..end_pos], true, None)?;
                }
                self.pending.drain(..end_pos + close.len());
                self.state = State::Content;
                self.quotes.reset(&self.raw, offset + end_pos + close.len());
                self.trim_reasoning_separator = true;
            } else {
                let mut held = if self.recognize_tools {
                    held_prefix(&self.raw, self.raw.len(), thinking_markers, &self.quotes)
                } else {
                    0
                };
                held = held.min(self.pending.len());
                for len in (1..=self.pending.len().min(close.len().saturating_sub(1))).rev() {
                    if close.starts_with(&self.pending[self.pending.len() - len..]) {
                        held = held.max(len);
                        break;
                    }
                }
                let ready = self.pending.len() - held;
                if ready > 0 {
                    self.emitted
                        .emit(sink, &self.pending[..ready], true, None)?;
                }
                self.pending.drain(..ready);
                return Ok(false);
            }
        }

        if self.state == State::Content {
            if self.trim_reasoning_separator {
                let content = after_separator(&self.pending, &markup.reasoning_separator).len();
                if content == 0 {
                    self.pending.clear();
                    return Ok(false);
                }
                self.pending.drain(..self.pending.len() - content);
                self.trim_reasoning_separator = false;
            }
            if self.raw_content {
                return self.structured_content(config, sink);
            }
            let offset = self.pending_offset();
            let found = if self.recognize_tools {
                earliest(&self.raw, &markup.markers, &self.quotes, offset)
            } else {
                NPOS
            };
            let marker = if found == NPOS { NPOS } else { found - offset };
            let repeats = config.reasoning.repeats && !open.is_empty();
            let thought = if repeats {
                find(&self.pending, open, 0)
            } else {
                NPOS
            };
            if thought != NPOS && (marker == NPOS || thought < marker) {
                // A later thought: the content before it, then reasoning.
                let prose = if self.recognize_tools {
                    content_before(
                        &self.raw,
                        offset,
                        offset + thought,
                        &markup.closers,
                        &markup.envelope,
                        &self.quotes,
                        &config.tools,
                        false,
                    )
                } else {
                    &self.pending[..thought]
                };
                if !prose.is_empty() {
                    self.emitted
                        .emit(sink, prose, false, Some(offset_in(&self.raw, prose)))?;
                }
                self.pending.drain(..thought + open.len());
                self.state = State::Thinking;
                return Ok(true);
            }
            if marker != NPOS {
                let prose = if self.recognize_tools {
                    content_before(
                        &self.raw,
                        offset,
                        found,
                        &markup.closers,
                        &markup.envelope,
                        &self.quotes,
                        &config.tools,
                        false,
                    )
                } else {
                    &self.pending[..marker]
                };
                if !prose.is_empty() {
                    self.emitted
                        .emit(sink, prose, false, Some(offset_in(&self.raw, prose)))?;
                }
                self.hidden.clear();
                append(&mut self.hidden, &self.pending[marker..])?;
                self.pending.clear();
                self.tool_mode = true;
                return Ok(false);
            }
            let held = if self.recognize_tools {
                held_prefix(&self.raw, self.raw.len(), &markup.markers, &self.quotes)
                    .max(hold(
                        &self.raw,
                        self.raw.len(),
                        offset,
                        &markup.closers,
                        &markup.envelope,
                        &self.quotes,
                        &config.tools,
                    ))
                    .min(self.pending.len())
            } else {
                0
            };
            // Hold an unfinished opener of a later thought.
            let mut held = held;
            if repeats {
                for len in (1..=self.pending.len().min(open.len() - 1)).rev() {
                    if open.starts_with(&self.pending[self.pending.len() - len..]) {
                        held = held.max(len);
                        break;
                    }
                }
            }
            let ready = self.pending.len() - held;
            if ready > 0 {
                self.emitted
                    .emit(sink, &self.pending[..ready], false, Some(offset))?;
            }
            self.pending.drain(..ready);
        }
        Ok(false)
    }

    /// Content under a response format or the tool grammar; true when it
    /// opened a later thought (`Reasoning::repeats`).
    fn structured_content(&mut self, config: &Config, sink: &mut dyn Sink) -> Result<bool, Error> {
        let markup = &config.markup;
        let markers = &markup.markers;
        if self.tool_only {
            let offset = self.pending_offset();
            let found = earliest(&self.raw, markers, &self.quotes, offset);
            let start = if found == NPOS { NPOS } else { found - offset };
            let open = &config.reasoning.open;
            let repeats = config.reasoning.repeats && !open.is_empty();
            let thought = if repeats {
                find(&self.pending, open, 0)
            } else {
                NPOS
            };
            if thought != NPOS && (start == NPOS || thought < start) {
                let prose = content_before(
                    &self.raw,
                    offset,
                    offset + thought,
                    &markup.closers,
                    &markup.envelope,
                    &self.quotes,
                    &config.tools,
                    false,
                );
                if !prose.is_empty() {
                    self.emitted
                        .emit(sink, prose, false, Some(offset_in(&self.raw, prose)))?;
                }
                self.emitted.content += prose.len();
                self.pending.drain(..thought + open.len());
                self.state = State::Thinking;
                return Ok(true);
            }
            if start != NPOS {
                let prose = content_before(
                    &self.raw,
                    offset,
                    found,
                    &markup.closers,
                    &markup.envelope,
                    &self.quotes,
                    &config.tools,
                    false,
                );
                if !prose.is_empty() {
                    self.emitted
                        .emit(sink, prose, false, Some(offset_in(&self.raw, prose)))?;
                }
                self.emitted.content += prose.len();
                self.hidden.clear();
                append(&mut self.hidden, &self.pending[start..])?;
                self.pending.clear();
                self.tool_mode = true;
                return Ok(false);
            }
            let held = held_prefix(&self.raw, self.raw.len(), markers, &self.quotes)
                .max(hold(
                    &self.raw,
                    self.raw.len(),
                    offset,
                    &markup.closers,
                    &markup.envelope,
                    &self.quotes,
                    &config.tools,
                ))
                .min(self.pending.len());
            let mut held = held;
            if repeats {
                for len in (1..=self.pending.len().min(open.len() - 1)).rev() {
                    if open.starts_with(&self.pending[self.pending.len() - len..]) {
                        held = held.max(len);
                        break;
                    }
                }
            }
            let ready = self.pending.len() - held;
            if ready > 0 {
                let text = content_before(
                    &self.raw,
                    offset,
                    offset + ready,
                    &markup.closers,
                    &markup.envelope,
                    &self.quotes,
                    &config.tools,
                    false,
                );
                if !text.is_empty() {
                    self.emitted
                        .emit(sink, text, false, Some(offset_in(&self.raw, text)))?;
                }
                self.emitted.content += text.len();
            }
            self.pending.drain(..ready);
            return Ok(false);
        }
        if !self.structured_started {
            let content = trim(&self.pending);
            // Positions in `content` are read against the response's quote
            // state, as the C++ filter read them.
            if content.is_empty()
                || (is_prefix(content, markers) && earliest(content, markers, &self.quotes, 0) != 0)
            {
                return Ok(false);
            }
            self.structured_started = true;
            if earliest(content, markers, &self.quotes, 0) == 0 {
                self.tool_mode = true;
                self.hidden = std::mem::take(&mut self.pending);
                return Ok(false);
            }
        }
        let offset = self.pending_offset();
        let text = std::mem::take(&mut self.pending);
        if text.is_empty() {
            return Ok(false);
        }
        self.emitted.emit(sink, &text, false, Some(offset))?;
        Ok(false)
    }

    /// Streams the tail the final parse decided: `content`, `reasoning` and
    /// `spans` are its result.
    pub fn finish(
        &mut self,
        hide_tool_markup: bool,
        content: &[u8],
        reasoning: &[u8],
        content_spans: &[Span],
        config: &Config,
        sink: &mut dyn Sink,
    ) -> Result<(), Error> {
        let emit_content = |emitted: &mut Emitted, sink: &mut dyn Sink, offset: usize| {
            let mut spans = Vec::new();
            append_spans(&mut spans, content_spans, offset, content.len() - offset);
            emitted.mapped(sink, &content[offset..], false, &spans)
        };
        if self.tool_mode && self.state == State::Thinking {
            let tail = &reasoning[self.emitted.reasoning.min(reasoning.len())..];
            if !tail.is_empty() {
                self.emitted.emit(sink, tail, true, None)?;
            }
            if !content.is_empty() {
                emit_content(&mut self.emitted, sink, 0)?;
            }
            return Ok(());
        }
        if self.tool_only && self.tool_mode && hide_tool_markup {
            // The prefix was streamed before the first call. Preserve prose
            // between or after complete calls without exposing their markup
            // or partial calls.
            let offset = self.emitted.content.min(content.len());
            if offset < content.len() {
                emit_content(&mut self.emitted, sink, offset)?;
            }
            return Ok(());
        }
        if self.raw_content && !self.structured_started && hide_tool_markup {
            self.pending.clear();
            return Ok(());
        }
        if !self.tool_mode {
            if !self.pending.is_empty() {
                // With tools declared, the closing tags the model repeats
                // after a call are markup; a client must never store them as
                // assistant prose (#383).
                let offset = self.pending_offset();
                let markup = &config.markup;
                let text = if self.recognize_tools && self.state == State::Content {
                    content_before(
                        &self.raw,
                        offset,
                        self.raw.len(),
                        &markup.closers,
                        &markup.envelope,
                        &self.quotes,
                        &config.tools,
                        false,
                    )
                } else {
                    &self.pending
                };
                let emitted =
                    self.emitted
                        .emit(sink, text, self.state == State::Thinking, Some(offset));
                self.pending.clear();
                emitted?;
            }
            return Ok(());
        }
        if !hide_tool_markup && !self.hidden.is_empty() {
            let offset = self.raw.len() - self.hidden.len();
            return self.emitted.emit(sink, &self.hidden, false, Some(offset));
        }
        Ok(())
    }
}
