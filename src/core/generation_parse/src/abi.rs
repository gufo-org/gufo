//! The C interface (include/gufo/generation_parse.h): config decoding, the
//! host callbacks, and the opaque streaming handle.

use crate::config::{
    Config, DialectScan, Envelope, Error, Host, Initial, Markup, Mode, QwenCall, QwenParameter,
    QwenScan, Reasoning, ScannedArgument, ScannedCall, Tool, ToolChoice, Unsatisfied,
};
use crate::events::{Event, Sink, Span};
use crate::parse::{emit_final, parse};
use crate::quote::QuoteTracker;
use crate::stream::Stream;
use crate::stream::utf8::Decoder;
use gufo_ffi::guard;
use std::ffi::{c_int, c_void};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoGenerationBytes {
    pub data: *const u8,
    pub len: usize,
}

impl GufoGenerationBytes {
    const EMPTY: Self = Self {
        data: std::ptr::null(),
        len: 0,
    };

    fn of(bytes: &[u8]) -> Self {
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    /// # Safety
    /// `data` points to `len` readable bytes for `'a`, or `len` is 0.
    #[allow(unsafe_code)]
    unsafe fn get<'a>(self) -> Result<&'a [u8], Error> {
        // SAFETY: guaranteed by the caller.
        unsafe { gufo_ffi::input(self.data, self.len) }.ok_or(Error::Internal)
    }

    /// # Safety
    /// As for `get`.
    #[allow(unsafe_code)]
    unsafe fn to_vec(self) -> Result<Vec<u8>, Error> {
        // SAFETY: guaranteed by the caller.
        crate::parse::copy(unsafe { self.get() }?)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoGenerationTool {
    pub name: GufoGenerationBytes,
    pub parameters_json: GufoGenerationBytes,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoGenerationRange {
    pub begin: usize,
    pub end: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoQwenParameter {
    pub name: GufoGenerationBytes,
    pub value: GufoGenerationBytes,
    pub framed: u8,
    pub progress: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoQwenScan {
    pub matched: u8,
    pub has_call: u8,
    pub native: u8,
    pub canonical_header: u8,
    pub complete: u8,
    pub has_unfinished: u8,
    pub progress: usize,
    pub end: usize,
    pub name: GufoGenerationBytes,
    pub json_payload: GufoGenerationBytes,
    pub parameters: *const GufoQwenParameter,
    pub parameter_count: usize,
    pub unfinished: GufoQwenParameter,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoScannedArgument {
    pub name: GufoGenerationBytes,
    pub value: GufoGenerationBytes,
    pub is_string: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoScannedCall {
    pub name: GufoGenerationBytes,
    pub arguments: *const GufoScannedArgument,
    pub argument_count: usize,
    pub has_unfinished: u8,
    pub valid: u8,
    pub complete: u8,
    pub unfinished: GufoScannedArgument,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoDialectScan {
    pub calls: *const GufoScannedCall,
    pub call_count: usize,
    pub spans: *const GufoGenerationRange,
    pub span_count: usize,
}

pub type GufoGenerationQwenScanFn = unsafe extern "C" fn(
    context: *mut c_void,
    text: *const u8,
    len: usize,
    scan: *mut GufoQwenScan,
) -> c_int;
pub type GufoGenerationDialectScanFn = unsafe extern "C" fn(
    context: *mut c_void,
    text: *const u8,
    len: usize,
    markers: *const GufoGenerationBytes,
    marker_count: usize,
    partial: c_int,
    scan: *mut GufoDialectScan,
) -> c_int;
pub type GufoGenerationGrammarFn = unsafe extern "C" fn(
    context: *mut c_void,
    tool: usize,
    arguments: *const u8,
    len: usize,
    accepted: *mut c_int,
) -> c_int;

pub const GUFO_GENERATION_QWEN_CALLS: u32 = 1;
pub const GUFO_GENERATION_NATIVE: u32 = 2;
pub const GUFO_GENERATION_DIALECT: u32 = 4;
pub const GUFO_GENERATION_TRACK_SPANS: u32 = 8;
pub const GUFO_GENERATION_BUFFERED: u32 = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoGenerationConfig {
    pub initial: u32,
    pub mode: u32,
    pub tool_choice: u32,
    pub flags: u32,
    pub reasoning_open: GufoGenerationBytes,
    pub reasoning_close: GufoGenerationBytes,
    pub reasoning_separator: GufoGenerationBytes,
    pub qwen_open: GufoGenerationBytes,
    pub markers: *const GufoGenerationBytes,
    pub marker_count: usize,
    pub reasoning_markers: *const GufoGenerationBytes,
    pub reasoning_marker_count: usize,
    pub closers: *const GufoGenerationBytes,
    pub closer_count: usize,
    pub envelope_heads: *const GufoGenerationBytes,
    pub envelope_head_count: usize,
    pub envelope_parameters: *const GufoGenerationBytes,
    pub envelope_parameter_count: usize,
    pub envelope_closers: *const GufoGenerationBytes,
    pub envelope_closer_count: usize,
    pub tools: *const GufoGenerationTool,
    pub tool_count: usize,
    pub qwen_scan: Option<GufoGenerationQwenScanFn>,
    pub dialect_scan: Option<GufoGenerationDialectScanFn>,
    pub grammar_accepts: Option<GufoGenerationGrammarFn>,
    pub context: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoGenerationStatus {
    Complete = 0,
    ToolChoiceUnsatisfied = 1,
    UnsafeStream = 2,
    Stopped = 3,
    NoMemory = 4,
    Internal = 5,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoGenerationEventKind {
    ReasoningDelta = 0,
    ContentDelta = 1,
    ToolDelta = 2,
    FinalReasoning = 3,
    FinalContent = 4,
    FinalCall = 5,
    FinalEnd = 6,
    Unsatisfied = 7,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoGenerationEvent {
    pub kind: GufoGenerationEventKind,
    pub index: usize,
    pub text: GufoGenerationBytes,
    pub name: GufoGenerationBytes,
    pub arguments: GufoGenerationBytes,
    pub spans: *const Span,
    pub span_count: usize,
    pub begin: u8,
    pub complete: u8,
    pub hide_tool_markup: u8,
}

/// Returns 0 to continue, nonzero to stop.
pub type GufoGenerationEventFn =
    unsafe extern "C" fn(context: *mut c_void, event: *const GufoGenerationEvent) -> c_int;

/// # Safety
/// `list` points to `count` byte strings, each readable during the call.
#[allow(unsafe_code)]
unsafe fn strings(list: *const GufoGenerationBytes, count: usize) -> Result<Vec<Vec<u8>>, Error> {
    // SAFETY: guaranteed by the caller.
    let list = unsafe { gufo_ffi::input(list, count) }.ok_or(Error::Internal)?;
    let mut out = Vec::new();
    out.try_reserve(list.len())?;
    for bytes in list {
        // SAFETY: each entry is readable, as guaranteed by the caller.
        out.push(unsafe { bytes.to_vec() }?);
    }
    Ok(out)
}

/// # Safety
/// Every pointer in `raw` is valid as the header documents.
#[allow(unsafe_code)]
unsafe fn decode(raw: &GufoGenerationConfig) -> Result<Config, Error> {
    let initial = match raw.initial {
        0 => Initial::Auto,
        1 => Initial::Reasoning,
        2 => Initial::Content,
        3 => Initial::ContentAfterReasoning,
        _ => return Err(Error::Internal),
    };
    let mode = match raw.mode {
        0 => Mode::Plain,
        1 => Mode::Structured,
        2 => Mode::ToolOnly,
        _ => return Err(Error::Internal),
    };
    let choice = match raw.tool_choice {
        0 => ToolChoice::Auto,
        1 => ToolChoice::None,
        2 => ToolChoice::Required,
        _ => return Err(Error::Internal),
    };
    let known = GUFO_GENERATION_QWEN_CALLS
        | GUFO_GENERATION_NATIVE
        | GUFO_GENERATION_DIALECT
        | GUFO_GENERATION_TRACK_SPANS
        | GUFO_GENERATION_BUFFERED;
    if raw.flags & !known != 0 {
        return Err(Error::Internal);
    }
    // SAFETY: the caller guarantees each list and string.
    let (markup, reasoning, tool_list) = unsafe {
        (
            Markup {
                markers: strings(raw.markers, raw.marker_count)?,
                reasoning_markers: strings(raw.reasoning_markers, raw.reasoning_marker_count)?,
                closers: strings(raw.closers, raw.closer_count)?,
                qwen_calls: raw.flags & GUFO_GENERATION_QWEN_CALLS != 0,
                native: raw.flags & GUFO_GENERATION_NATIVE != 0,
                dialect: raw.flags & GUFO_GENERATION_DIALECT != 0,
                reasoning_separator: raw.reasoning_separator.to_vec()?,
                qwen_open: raw.qwen_open.to_vec()?,
                envelope: Envelope {
                    heads: strings(raw.envelope_heads, raw.envelope_head_count)?,
                    parameters: strings(raw.envelope_parameters, raw.envelope_parameter_count)?,
                    closers: strings(raw.envelope_closers, raw.envelope_closer_count)?,
                },
            },
            Reasoning {
                open: raw.reasoning_open.to_vec()?,
                close: raw.reasoning_close.to_vec()?,
            },
            gufo_ffi::input(raw.tools, raw.tool_count).ok_or(Error::Internal)?,
        )
    };
    let mut tools = Vec::new();
    tools.try_reserve(tool_list.len())?;
    for tool in tool_list {
        // SAFETY: the caller guarantees each tool's strings.
        let (name, parameters) = unsafe { (tool.name.get()?, tool.parameters_json.get()?) };
        tools.push(Tool::new(crate::text::utf8(name)?, parameters)?);
    }
    Ok(Config {
        initial,
        mode,
        choice,
        track_spans: raw.flags & GUFO_GENERATION_TRACK_SPANS != 0,
        buffered: raw.flags & GUFO_GENERATION_BUFFERED != 0,
        reasoning,
        markup,
        tools,
    })
}

/// The C++ callbacks of one config.
struct CHost {
    qwen_scan: Option<GufoGenerationQwenScanFn>,
    dialect_scan: Option<GufoGenerationDialectScanFn>,
    grammar_accepts: Option<GufoGenerationGrammarFn>,
    context: *mut c_void,
}

impl CHost {
    fn new(raw: &GufoGenerationConfig) -> Self {
        Self {
            qwen_scan: raw.qwen_scan,
            dialect_scan: raw.dialect_scan,
            grammar_accepts: raw.grammar_accepts,
            context: raw.context,
        }
    }
}

/// # Safety
/// As for `GufoGenerationBytes::get`, for every byte string of `raw`.
#[allow(unsafe_code)]
unsafe fn qwen_parameter(raw: &GufoQwenParameter) -> Result<QwenParameter, Error> {
    // SAFETY: guaranteed by the caller.
    unsafe {
        Ok(QwenParameter {
            name: raw.name.to_vec()?,
            value: raw.value.to_vec()?,
            framed: raw.framed != 0,
            progress: raw.progress,
        })
    }
}

/// # Safety
/// As for `GufoGenerationBytes::get`.
#[allow(unsafe_code)]
unsafe fn scanned_argument(raw: &GufoScannedArgument) -> Result<ScannedArgument, Error> {
    // SAFETY: guaranteed by the caller.
    unsafe {
        Ok(ScannedArgument {
            name: raw.name.to_vec()?,
            value: raw.value.to_vec()?,
            is_string: raw.is_string != 0,
        })
    }
}

#[allow(unsafe_code)]
impl Host for CHost {
    fn qwen_scan(&mut self, text: &[u8]) -> Result<QwenScan, Error> {
        let scan_fn = self.qwen_scan.ok_or(Error::Internal)?;
        let mut raw = GufoQwenScan {
            matched: 0,
            has_call: 0,
            native: 0,
            canonical_header: 0,
            complete: 0,
            has_unfinished: 0,
            progress: 0,
            end: 0,
            name: GufoGenerationBytes::EMPTY,
            json_payload: GufoGenerationBytes::EMPTY,
            parameters: std::ptr::null(),
            parameter_count: 0,
            unfinished: GufoQwenParameter {
                name: GufoGenerationBytes::EMPTY,
                value: GufoGenerationBytes::EMPTY,
                framed: 0,
                progress: 0,
            },
        };
        // SAFETY: the config allows calling `qwen_scan` with `context`; `text`
        // and `raw` live through the call.
        if unsafe { scan_fn(self.context, text.as_ptr(), text.len(), &mut raw) } != 0 {
            return Err(Error::Stopped);
        }
        // SAFETY: the callback leaves every pointer of `raw` valid until its
        // next call, and nothing here calls it again.
        unsafe {
            let call = if raw.has_call != 0 {
                let parameters =
                    gufo_ffi::input(raw.parameters, raw.parameter_count).ok_or(Error::Internal)?;
                let mut list = Vec::new();
                list.try_reserve(parameters.len())?;
                for parameter in parameters {
                    list.push(qwen_parameter(parameter)?);
                }
                Some(QwenCall {
                    name: raw.name.to_vec()?,
                    parameters: list,
                    unfinished: if raw.has_unfinished != 0 {
                        Some(qwen_parameter(&raw.unfinished)?)
                    } else {
                        None
                    },
                    native: raw.native != 0,
                    json_payload: raw.json_payload.to_vec()?,
                    canonical_header: raw.canonical_header != 0,
                    complete: raw.complete != 0,
                })
            } else {
                None
            };
            Ok(QwenScan {
                matched: raw.matched != 0,
                progress: raw.progress,
                end: raw.end,
                call,
            })
        }
    }

    fn dialect_scan(
        &mut self,
        text: &[u8],
        markers: &[&[u8]],
        partial: bool,
    ) -> Result<DialectScan, Error> {
        let scan_fn = self.dialect_scan.ok_or(Error::Internal)?;
        let mut list = Vec::new();
        list.try_reserve(markers.len())?;
        list.extend(markers.iter().map(|marker| GufoGenerationBytes::of(marker)));
        let mut raw = GufoDialectScan {
            calls: std::ptr::null(),
            call_count: 0,
            spans: std::ptr::null(),
            span_count: 0,
        };
        // SAFETY: the config allows calling `dialect_scan` with `context`;
        // the text, markers and `raw` live through the call.
        let failed = unsafe {
            scan_fn(
                self.context,
                text.as_ptr(),
                text.len(),
                list.as_ptr(),
                list.len(),
                c_int::from(partial),
                &mut raw,
            )
        };
        if failed != 0 {
            return Err(Error::Stopped);
        }
        // SAFETY: the callback leaves every pointer of `raw` valid until its
        // next call, and nothing here calls it again.
        unsafe {
            let calls = gufo_ffi::input(raw.calls, raw.call_count).ok_or(Error::Internal)?;
            let spans = gufo_ffi::input(raw.spans, raw.span_count).ok_or(Error::Internal)?;
            let mut scan = DialectScan::default();
            scan.calls.try_reserve(calls.len())?;
            for call in calls {
                let arguments =
                    gufo_ffi::input(call.arguments, call.argument_count).ok_or(Error::Internal)?;
                let mut converted = Vec::new();
                converted.try_reserve(arguments.len())?;
                for argument in arguments {
                    converted.push(scanned_argument(argument)?);
                }
                scan.calls.push(ScannedCall {
                    name: call.name.to_vec()?,
                    arguments: converted,
                    unfinished: if call.has_unfinished != 0 {
                        Some(scanned_argument(&call.unfinished)?)
                    } else {
                        None
                    },
                    valid: call.valid != 0,
                    complete: call.complete != 0,
                });
            }
            scan.spans.try_reserve(spans.len())?;
            for span in spans {
                if span.begin > span.end || span.end > text.len() {
                    return Err(Error::Internal);
                }
                scan.spans.push((span.begin, span.end));
            }
            Ok(scan)
        }
    }

    fn grammar_accepts(&mut self, tool: usize, arguments: &[u8]) -> Result<bool, Error> {
        let accepts = self.grammar_accepts.ok_or(Error::Internal)?;
        let mut accepted: c_int = 0;
        // SAFETY: the config allows calling `grammar_accepts` with
        // `context`; `arguments` and `accepted` live through the call.
        if unsafe {
            accepts(
                self.context,
                tool,
                arguments.as_ptr(),
                arguments.len(),
                &mut accepted,
            )
        } != 0
        {
            return Err(Error::Stopped);
        }
        Ok(accepted != 0)
    }
}

/// Forwards events to a C callback.
struct CSink {
    event: GufoGenerationEventFn,
    context: *mut c_void,
}

impl Sink for CSink {
    #[allow(unsafe_code)]
    fn event(&mut self, event: Event<'_>) -> bool {
        let mut raw = GufoGenerationEvent {
            kind: GufoGenerationEventKind::FinalEnd,
            index: 0,
            text: GufoGenerationBytes::EMPTY,
            name: GufoGenerationBytes::EMPTY,
            arguments: GufoGenerationBytes::EMPTY,
            spans: std::ptr::null(),
            span_count: 0,
            begin: 0,
            complete: 0,
            hide_tool_markup: 0,
        };
        let spans_of = |raw: &mut GufoGenerationEvent, spans: &[Span]| {
            raw.spans = spans.as_ptr();
            raw.span_count = spans.len();
        };
        match event {
            Event::ReasoningDelta { text, spans } => {
                raw.kind = GufoGenerationEventKind::ReasoningDelta;
                raw.text = GufoGenerationBytes::of(text);
                spans_of(&mut raw, spans);
            }
            Event::ContentDelta { text, spans } => {
                raw.kind = GufoGenerationEventKind::ContentDelta;
                raw.text = GufoGenerationBytes::of(text);
                spans_of(&mut raw, spans);
            }
            Event::ToolDelta {
                index,
                name,
                arguments,
                begin,
                complete,
            } => {
                raw.kind = GufoGenerationEventKind::ToolDelta;
                raw.index = index;
                raw.name = GufoGenerationBytes::of(name);
                raw.arguments = GufoGenerationBytes::of(arguments);
                raw.begin = u8::from(begin);
                raw.complete = u8::from(complete);
            }
            Event::FinalReasoning { text } => {
                raw.kind = GufoGenerationEventKind::FinalReasoning;
                raw.text = GufoGenerationBytes::of(text);
            }
            Event::FinalContent { text, spans } => {
                raw.kind = GufoGenerationEventKind::FinalContent;
                raw.text = GufoGenerationBytes::of(text);
                spans_of(&mut raw, spans);
            }
            Event::FinalCall {
                index,
                name,
                arguments,
            } => {
                raw.kind = GufoGenerationEventKind::FinalCall;
                raw.index = index;
                raw.name = GufoGenerationBytes::of(name);
                raw.arguments = GufoGenerationBytes::of(arguments);
            }
            Event::FinalEnd { hide_tool_markup } => {
                raw.kind = GufoGenerationEventKind::FinalEnd;
                raw.hide_tool_markup = u8::from(hide_tool_markup);
            }
        }
        // SAFETY: the caller allows calling `event` with `context`; `raw`
        // and the bytes it points to live through the call.
        (unsafe { (self.event)(self.context, &raw) }) == 0
    }
}

fn status(result: Result<(), Error>, sink: &mut CSink) -> GufoGenerationStatus {
    match result {
        Ok(()) => GufoGenerationStatus::Complete,
        Err(Error::ToolChoiceUnsatisfied(reason)) => {
            report_unsatisfied(sink, reason);
            GufoGenerationStatus::ToolChoiceUnsatisfied
        }
        Err(Error::UnsafeStream) => GufoGenerationStatus::UnsafeStream,
        Err(Error::Stopped) => GufoGenerationStatus::Stopped,
        Err(Error::NoMemory) => GufoGenerationStatus::NoMemory,
        Err(Error::Internal) => GufoGenerationStatus::Internal,
    }
}

/// Says which requirement failed: `complete` is set when a constrained turn
/// did not complete a call, clear when no call was produced.
#[allow(unsafe_code)]
fn report_unsatisfied(sink: &mut CSink, reason: Unsatisfied) {
    let raw = GufoGenerationEvent {
        kind: GufoGenerationEventKind::Unsatisfied,
        index: 0,
        text: GufoGenerationBytes::EMPTY,
        name: GufoGenerationBytes::EMPTY,
        arguments: GufoGenerationBytes::EMPTY,
        spans: std::ptr::null(),
        span_count: 0,
        begin: 0,
        complete: u8::from(reason == Unsatisfied::NotCompleted),
        hide_tool_markup: 0,
    };
    // SAFETY: as in `CSink::event`; the status follows regardless.
    let _ = unsafe { (sink.event)(sink.context, &raw) };
}

/// One streamed generation behind the C interface.
pub struct GufoGeneration {
    stream: Stream,
    host: CHost,
}

/// Moves `value` to the heap like `Box::new`, but returns null when memory
/// runs out instead of aborting the process.
fn try_box<T>(value: T) -> *mut T {
    let mut slot = Vec::new();
    if slot.try_reserve_exact(1).is_err() {
        return std::ptr::null_mut();
    }
    slot.push(value);
    // One element fills the reserved capacity, so no reallocation happens,
    // and `[T; 1]` has the layout of `T`: `Box::<T>::from_raw` frees it.
    Box::<[T; 1]>::try_from(slot.into_boxed_slice())
        .map_or(std::ptr::null_mut(), |boxed| Box::into_raw(boxed).cast())
}

/// Makes a streaming handle for `config` in `*handle`, or stores null and
/// fails.
///
/// # Safety
/// `config` is valid as the header documents, during this call; its
/// callbacks and context stay callable until the handle is freed. `handle`
/// is writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_generation_new(
    config: *const GufoGenerationConfig,
    handle: *mut *mut GufoGeneration,
) -> GufoGenerationStatus {
    // SAFETY: the caller passes a writable pointer; `as_mut` rejects null.
    let Some(handle) = (unsafe { handle.as_mut() }) else {
        return GufoGenerationStatus::Internal;
    };
    *handle = std::ptr::null_mut();
    guard(GufoGenerationStatus::Internal, || {
        // SAFETY: the caller passes a valid config; `as_ref` rejects null.
        let Some(raw) = (unsafe { config.as_ref() }) else {
            return GufoGenerationStatus::Internal;
        };
        // SAFETY: the config's pointers are valid during this call.
        let config = match unsafe { decode(raw) } {
            Ok(config) => config,
            Err(Error::NoMemory) => return GufoGenerationStatus::NoMemory,
            Err(_) => return GufoGenerationStatus::Internal,
        };
        let boxed = try_box(GufoGeneration {
            stream: Stream::new(config),
            host: CHost::new(raw),
        });
        if boxed.is_null() {
            return GufoGenerationStatus::NoMemory;
        }
        *handle = boxed;
        GufoGenerationStatus::Complete
    })
}

/// # Safety
/// `handle` is null or comes from `gufo_generation_new` and is not used
/// afterwards.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_generation_free(handle: *mut GufoGeneration) {
    guard((), || {
        if !handle.is_null() {
            // SAFETY: the caller passes a handle from `gufo_generation_new`
            // once, allocated with the layout of `GufoGeneration`.
            drop(unsafe { Box::from_raw(handle) });
        }
    });
}

/// Reads `len` generated bytes and reports streamed deltas through `event`.
/// After any status but `Complete`, the handle accepts no more input.
///
/// # Safety
/// `handle` comes from `gufo_generation_new`; `bytes` points to `len`
/// readable bytes or `len` is 0; `event` may be called with `context` until
/// this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_generation_feed(
    handle: *mut GufoGeneration,
    bytes: *const u8,
    len: usize,
    event: Option<GufoGenerationEventFn>,
    context: *mut c_void,
) -> GufoGenerationStatus {
    // SAFETY: the caller passes a live handle; `as_mut` rejects null.
    let (Some(generation), Some(event)) = (unsafe { handle.as_mut() }, event) else {
        return GufoGenerationStatus::Internal;
    };
    guard(GufoGenerationStatus::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { gufo_ffi::input(bytes, len) }) else {
            return GufoGenerationStatus::Internal;
        };
        let mut sink = CSink { event, context };
        let result = generation
            .stream
            .feed(bytes, &mut generation.host, &mut sink);
        status(result, &mut sink)
    })
}

/// Ends the response: streams the last tool and text deltas, then the final
/// events. `enforce_required` fails an unmet `tool_choice: required`;
/// `interrupted` (a stop before the model's own end) lets unfinished
/// streamed calls lapse.
///
/// # Safety
/// As for `gufo_generation_feed`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_generation_finish(
    handle: *mut GufoGeneration,
    enforce_required: c_int,
    interrupted: c_int,
    event: Option<GufoGenerationEventFn>,
    context: *mut c_void,
) -> GufoGenerationStatus {
    // SAFETY: the caller passes a live handle; `as_mut` rejects null.
    let (Some(generation), Some(event)) = (unsafe { handle.as_mut() }, event) else {
        return GufoGenerationStatus::Internal;
    };
    guard(GufoGenerationStatus::Internal, || {
        let mut sink = CSink { event, context };
        let result = generation.stream.finish(
            enforce_required != 0,
            interrupted != 0,
            &mut generation.host,
            &mut sink,
        );
        status(result, &mut sink)
    })
}

/// Parses a whole response of `len` bytes and reports the final events.
///
/// # Safety
/// `config` is valid as for `gufo_generation_new`, and `bytes` and `event`
/// as for `gufo_generation_feed`, during this call.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_generation_parse(
    config: *const GufoGenerationConfig,
    bytes: *const u8,
    len: usize,
    enforce_required: c_int,
    event: Option<GufoGenerationEventFn>,
    context: *mut c_void,
) -> GufoGenerationStatus {
    let Some(event) = event else {
        return GufoGenerationStatus::Internal;
    };
    guard(GufoGenerationStatus::Internal, || {
        // SAFETY: the caller passes a valid config; `as_ref` rejects null.
        let Some(raw) = (unsafe { config.as_ref() }) else {
            return GufoGenerationStatus::Internal;
        };
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { gufo_ffi::input(bytes, len) }) else {
            return GufoGenerationStatus::Internal;
        };
        let mut sink = CSink { event, context };
        let result = (|| {
            // SAFETY: the config's pointers are valid during this call.
            let config = unsafe { decode(raw) }?;
            let mut host = CHost::new(raw);
            let mut decoded = Vec::new();
            Decoder::default().push(bytes, true, &mut decoded)?;
            let parsed = parse(
                &decoded,
                &config,
                enforce_required != 0,
                false,
                config.track_spans,
                &mut QuoteTracker::default(),
                &mut host,
            )?;
            emit_final(&parsed, &mut sink)
        })();
        status(result, &mut sink)
    })
}
