//! The C++ side of a decode: the `Host` trait the decoders call, and its
//! implementation over the C callbacks of `GufoChatHost`.

use crate::error::{Failure, Outcome};
use std::ffi::{c_int, c_void};

/// `GufoChatOp`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Model = 0,
    MaxTokens = 1,
    Stream = 2,
    IncludeUsage = 3,
    CachePrompt = 10,
    CachePersist = 11,
    ReturnProgress = 12,
    AddVisionId = 13,
    ReasoningEnabled = 14,
    ReasoningEffort = 15,
    PreserveThinking = 16,
    BudgetTokens = 17,
    ToolChoice = 18,
    ForcedToolName = 19,
    ParallelToolCalls = 20,
    ConstrainedTools = 21,
    NamespaceKey = 22,
    NamespaceValue = 23,
    Tool = 30,
    ToolDescription = 31,
    ToolParameters = 32,
    ToolDefinition = 33,
    SelectTool = 34,
    Message = 40,
    Content = 41,
    Name = 42,
    Thought = 43,
    ToolCallId = 44,
    Image = 45,
    ImageLow = 46,
    Call = 47,
    CallName = 48,
    Argument = 49,
    ArgumentValue = 50,
    Part = 51,
}

/// `GufoChatHostField`: a C++ helper and the value it reads.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostField {
    ResponseFormat = 0,
    Stop = 1,
    Sampling = 2,
    Logprobs = 3,
    ResponsesOptions = 4,
    TextFormat = 5,
    MessagesOptions = 6,
    StopSequences = 7,
}

/// `GufoChatRole`: `tokenization::ChatRole`.
pub const ROLE_SYSTEM: i32 = 0;
pub const ROLE_DEVELOPER: i32 = 1;
pub const ROLE_USER: i32 = 2;
pub const ROLE_ASSISTANT: i32 = 3;
pub const ROLE_TOOL: i32 = 4;

/// `GufoChatEffort`: `ReasoningEffort`.
pub const EFFORT_MINIMAL: u8 = 0;
pub const EFFORT_LOW: u8 = 1;
pub const EFFORT_MEDIUM: u8 = 2;
pub const EFFORT_HIGH: u8 = 3;
pub const EFFORT_XHIGH: u8 = 4;
pub const EFFORT_MAX: u8 = 5;

/// `GufoChatToolChoice`: `ChatRequest::ToolChoice`.
pub const TOOL_CHOICE_AUTO: i32 = 0;
pub const TOOL_CHOICE_NONE: i32 = 1;
pub const TOOL_CHOICE_REQUIRED: i32 = 2;

pub const HOST_OK: c_int = 0;
pub const HOST_REJECTED: c_int = 1;

/// What a decoder asks of C++. Helper failures with a message come back as
/// `Failure::Invalid`, which the caller turns into its rejection.
pub trait Host {
    fn serves_model(&mut self, name: &str) -> Outcome<bool>;
    fn read_image(&mut self, url: &[u8]) -> Outcome<u32>;
    fn compile_schema(&mut self, json: &[u8]) -> Outcome<()>;
    /// Runs the helper for `field` on `json`, or on an absent value.
    fn field(&mut self, field: HostField, argument: i32, json: Option<&[u8]>) -> Outcome<()>;
    fn emit(&mut self, op: Op, index: u32, integer: i64, text: &[u8]) -> Outcome<()>;
}

#[repr(C)]
pub struct GufoChatMessage {
    pub state: *mut c_void,
    pub set: Option<unsafe extern "C" fn(state: *mut c_void, data: *const u8, len: usize)>,
}

#[repr(C)]
pub struct GufoChatEvent {
    pub op: Op,
    pub index: u32,
    pub integer: i64,
    pub text: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct GufoChatRejection {
    pub status: i32,
    pub code: *const u8,
    pub code_len: usize,
    pub message: *const u8,
    pub message_len: usize,
}

#[repr(C)]
pub struct GufoChatHost {
    pub serves_model:
        Option<unsafe extern "C" fn(context: *mut c_void, name: *const u8, len: usize) -> c_int>,
    pub read_image: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            url: *const u8,
            len: usize,
            image_id: *mut u32,
            error: *const GufoChatMessage,
        ) -> c_int,
    >,
    pub compile_schema: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            json: *const u8,
            len: usize,
            error: *const GufoChatMessage,
        ) -> c_int,
    >,
    pub host_field: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            field: HostField,
            argument: i32,
            json: *const u8,
            len: usize,
        ) -> c_int,
    >,
    pub emit:
        Option<unsafe extern "C" fn(context: *mut c_void, event: *const GufoChatEvent) -> c_int>,
    pub reject:
        Option<unsafe extern "C" fn(context: *mut c_void, rejection: *const GufoChatRejection)>,
}

/// An empty buffer's pointer may dangle; C++ gets null instead.
fn pointer(bytes: &[u8]) -> *const u8 {
    if bytes.is_empty() {
        std::ptr::null()
    } else {
        bytes.as_ptr()
    }
}

/// Collects a helper's failure message.
#[allow(unsafe_code)]
unsafe extern "C" fn set_message(state: *mut c_void, data: *const u8, len: usize) {
    gufo_ffi::guard((), || {
        // SAFETY: `state` is the `Vec<u8>` a `Callbacks` method passed with
        // this function, alive for the callback that calls it.
        let out = unsafe { &mut *state.cast::<Option<Vec<u8>>>() };
        // SAFETY: the caller passes `len` readable bytes.
        let bytes = unsafe { gufo_ffi::input(data, len) };
        *out = match bytes {
            Some(bytes) => {
                let mut copy = Vec::new();
                copy.try_reserve_exact(bytes.len()).ok().map(|()| {
                    copy.extend_from_slice(bytes);
                    copy
                })
            }
            None => None,
        };
    })
}

/// The callbacks, each verified present, with the context they take.
pub struct Callbacks {
    context: *mut c_void,
    serves_model: unsafe extern "C" fn(*mut c_void, *const u8, usize) -> c_int,
    read_image: unsafe extern "C" fn(
        *mut c_void,
        *const u8,
        usize,
        *mut u32,
        *const GufoChatMessage,
    ) -> c_int,
    compile_schema:
        unsafe extern "C" fn(*mut c_void, *const u8, usize, *const GufoChatMessage) -> c_int,
    host_field: unsafe extern "C" fn(*mut c_void, HostField, i32, *const u8, usize) -> c_int,
    emit: unsafe extern "C" fn(*mut c_void, *const GufoChatEvent) -> c_int,
    reject: unsafe extern "C" fn(*mut c_void, *const GufoChatRejection),
}

impl Callbacks {
    /// `None` when a callback is missing.
    pub fn new(host: &GufoChatHost, context: *mut c_void) -> Option<Self> {
        Some(Self {
            context,
            serves_model: host.serves_model?,
            read_image: host.read_image?,
            compile_schema: host.compile_schema?,
            host_field: host.host_field?,
            emit: host.emit?,
            reject: host.reject?,
        })
    }

    #[allow(unsafe_code)]
    pub fn reject(&mut self, status: i32, code: &str, message: &str) {
        let rejection = GufoChatRejection {
            status,
            code: pointer(code.as_bytes()),
            code_len: code.len(),
            message: pointer(message.as_bytes()),
            message_len: message.len(),
        };
        // SAFETY: the caller of `gufo_chat_decode` allows calling its
        // callbacks with `context` until it returns; `rejection` and its
        // bytes outlive the call.
        unsafe { (self.reject)(self.context, &rejection) };
    }

    /// The message a failed helper set; `Failure::Invalid` carries it.
    fn helper_failure(message: Option<Vec<u8>>) -> Failure {
        match message.map(String::from_utf8) {
            Some(Ok(text)) => Failure::Invalid(text),
            // C++ messages are bytes; keep them as the rejection carries them.
            Some(Err(error)) => {
                Failure::Invalid(String::from_utf8_lossy(error.as_bytes()).into_owned())
            }
            None => Failure::NoMemory,
        }
    }
}

impl Host for Callbacks {
    #[allow(unsafe_code)]
    fn serves_model(&mut self, name: &str) -> Outcome<bool> {
        // SAFETY: as in `reject`; `name` outlives the call.
        match unsafe { (self.serves_model)(self.context, pointer(name.as_bytes()), name.len()) } {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Failure::Stopped),
        }
    }

    #[allow(unsafe_code)]
    fn read_image(&mut self, url: &[u8]) -> Outcome<u32> {
        let mut message: Option<Vec<u8>> = Some(Vec::new());
        let sink = GufoChatMessage {
            state: (&raw mut message).cast(),
            set: Some(set_message),
        };
        let mut id = 0u32;
        // SAFETY: as in `reject`; `url`, `id` and `sink` (with the message it
        // writes) outlive the call.
        match unsafe { (self.read_image)(self.context, pointer(url), url.len(), &mut id, &sink) } {
            0 => Ok(id),
            1 => Err(Self::helper_failure(message)),
            _ => Err(Failure::Stopped),
        }
    }

    #[allow(unsafe_code)]
    fn compile_schema(&mut self, json: &[u8]) -> Outcome<()> {
        let mut message: Option<Vec<u8>> = Some(Vec::new());
        let sink = GufoChatMessage {
            state: (&raw mut message).cast(),
            set: Some(set_message),
        };
        // SAFETY: as in `read_image`.
        match unsafe { (self.compile_schema)(self.context, pointer(json), json.len(), &sink) } {
            0 => Ok(()),
            1 => Err(Self::helper_failure(message)),
            _ => Err(Failure::Stopped),
        }
    }

    #[allow(unsafe_code)]
    fn field(&mut self, field: HostField, argument: i32, json: Option<&[u8]>) -> Outcome<()> {
        let json = json.unwrap_or(&[]);
        // SAFETY: as in `reject`; `json` outlives the call.
        match unsafe { (self.host_field)(self.context, field, argument, pointer(json), json.len()) }
        {
            HOST_OK => Ok(()),
            HOST_REJECTED => Err(Failure::HostRejected),
            _ => Err(Failure::Stopped),
        }
    }

    #[allow(unsafe_code)]
    fn emit(&mut self, op: Op, index: u32, integer: i64, text: &[u8]) -> Outcome<()> {
        let event = GufoChatEvent {
            op,
            index,
            integer,
            text: pointer(text),
            len: text.len(),
        };
        // SAFETY: as in `reject`; `event` and `text` outlive the call.
        match unsafe { (self.emit)(self.context, &event) } {
            0 => Ok(()),
            _ => Err(Failure::Stopped),
        }
    }
}
