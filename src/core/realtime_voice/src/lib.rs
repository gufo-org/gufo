//! C ABI for the Realtime voice conversation session protocol.
#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod session;

use std::ffi::c_void;

use gufo_ffi::{guard, input};
use session::{
    ClientDecision, FunctionCall, Host, ResponseOutcome, Session, Settings, Transcription,
    VadSettings,
};

/// Opaque Realtime voice session handle.
pub struct GufoRealtimeVoice {
    inner: Session,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoRealtimeVoiceStatus {
    Ok = 0,
    Closed = 1,
    Internal = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoRealtimeVoiceDecision {
    Queue = 0,
    Cancel = 1,
    VadAudio = 2,
    Handled = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoRealtimeVoiceBytes {
    pub data: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoRealtimeVoiceSettings {
    pub instructions: GufoRealtimeVoiceBytes,
    pub audio_output: bool,
    pub tools: *const GufoRealtimeVoiceBytes,
    pub tools_len: usize,
    pub max_output_tokens: usize,
    pub reasoning: bool,
    pub voice: GufoRealtimeVoiceBytes,
    pub response_instructions: GufoRealtimeVoiceBytes,
    pub call_tools: bool,
    pub out_of_band: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoRealtimeVoiceVadSettings {
    pub threshold: f64,
    pub prefix_padding_ms: i32,
    pub silence_duration_ms: i32,
    pub create_response: bool,
    pub interrupt_response: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoRealtimeVoiceFunctionCall {
    pub name: GufoRealtimeVoiceBytes,
    pub arguments: GufoRealtimeVoiceBytes,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoRealtimeVoiceResponseOutcome {
    pub ok: bool,
    pub cancelled: bool,
    pub incomplete: bool,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub calls: *const GufoRealtimeVoiceFunctionCall,
    pub calls_len: usize,
    pub failure: GufoRealtimeVoiceBytes,
}

pub type GufoRealtimeVoiceSendFn =
    unsafe extern "C" fn(context: *mut c_void, data: *const u8, len: usize) -> bool;
pub type GufoRealtimeVoiceBytesFn =
    unsafe extern "C" fn(context: *mut c_void, data: *const u8, len: usize);
pub type GufoRealtimeVoiceErrorSetFn =
    unsafe extern "C" fn(error_ctx: *mut c_void, data: *const u8, len: usize);
pub type GufoRealtimeVoiceActiveResponseFn = unsafe extern "C" fn(
    context: *mut c_void,
    settings: *const GufoRealtimeVoiceSettings,
    response_id: *const u8,
    response_id_len: usize,
);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GufoRealtimeVoiceHost {
    pub context: *mut c_void,
    pub send_event: Option<GufoRealtimeVoiceSendFn>,
    pub configure: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            settings: *const GufoRealtimeVoiceSettings,
            set_error: GufoRealtimeVoiceErrorSetFn,
            error_ctx: *mut c_void,
        ) -> bool,
    >,
    pub on_vad_settings: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            has_vad: bool,
            vad: *const GufoRealtimeVoiceVadSettings,
        ),
    >,
    pub append_audio: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            pcm: *const u8,
            len: usize,
            record: bool,
            set_error: GufoRealtimeVoiceErrorSetFn,
            error_ctx: *mut c_void,
        ) -> bool,
    >,
    pub commit_audio: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            committed: *mut bool,
            set_error: GufoRealtimeVoiceErrorSetFn,
            error_ctx: *mut c_void,
        ) -> bool,
    >,
    pub clear_audio: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            set_error: GufoRealtimeVoiceErrorSetFn,
            error_ctx: *mut c_void,
        ) -> bool,
    >,
    pub truncate_reply: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            samples: usize,
            set_error: GufoRealtimeVoiceErrorSetFn,
            error_ctx: *mut c_void,
        ) -> bool,
    >,
    pub add_user_text: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            text: *const u8,
            len: usize,
            set_error: GufoRealtimeVoiceErrorSetFn,
            error_ctx: *mut c_void,
        ) -> bool,
    >,
    pub add_tool_outputs: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            outputs: *const GufoRealtimeVoiceBytes,
            count: usize,
            set_error: GufoRealtimeVoiceErrorSetFn,
            error_ctx: *mut c_void,
        ) -> bool,
    >,
    pub serves_model:
        Option<unsafe extern "C" fn(context: *mut c_void, name: *const u8, len: usize) -> bool>,
    /// Transcribes the recorded audio of the user turn committed last;
    /// `language` is empty when the model detects it. The host calls
    /// `on_transcript` once with the whole transcript.
    pub transcribe_turn: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            language: GufoRealtimeVoiceBytes,
            prompt: GufoRealtimeVoiceBytes,
            on_transcript: GufoRealtimeVoiceBytesFn,
            transcript_ctx: *mut c_void,
            set_error: GufoRealtimeVoiceErrorSetFn,
            error_ctx: *mut c_void,
        ) -> bool,
    >,
    pub delete_user_turn: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            set_error: GufoRealtimeVoiceErrorSetFn,
            error_ctx: *mut c_void,
        ) -> bool,
    >,
    pub supports_response_options: Option<unsafe extern "C" fn(context: *mut c_void) -> bool>,
}

fn as_c_bytes(bytes: &[u8]) -> GufoRealtimeVoiceBytes {
    GufoRealtimeVoiceBytes {
        data: if bytes.is_empty() {
            std::ptr::null()
        } else {
            bytes.as_ptr()
        },
        len: bytes.len(),
    }
}

fn with_c_settings<R>(settings: &Settings, f: impl FnOnce(&GufoRealtimeVoiceSettings) -> R) -> R {
    let tools: Vec<GufoRealtimeVoiceBytes> = settings
        .tools
        .iter()
        .map(|t| as_c_bytes(t.as_bytes()))
        .collect();
    let c_settings = GufoRealtimeVoiceSettings {
        instructions: as_c_bytes(settings.instructions.as_bytes()),
        audio_output: settings.audio_output,
        tools: if tools.is_empty() {
            std::ptr::null()
        } else {
            tools.as_ptr()
        },
        tools_len: tools.len(),
        max_output_tokens: settings.max_output_tokens,
        reasoning: settings.reasoning,
        voice: as_c_bytes(settings.voice.as_bytes()),
        response_instructions: as_c_bytes(settings.response_instructions.as_bytes()),
        call_tools: settings.call_tools,
        out_of_band: settings.out_of_band,
    };
    f(&c_settings)
}

#[allow(unsafe_code)]
unsafe extern "C" fn set_error_string(error_ctx: *mut c_void, data: *const u8, len: usize) {
    guard((), || {
        // SAFETY: `error_ctx` is a `&mut String` pointer created by `CHost`
        // below, and the host callback provides `len` readable bytes (or
        // len == 0).
        let Some(dst) = (unsafe { (error_ctx as *mut String).as_mut() }) else {
            return;
        };
        // SAFETY: caller guarantees `len` readable bytes or `len == 0`.
        let Some(bytes) = (unsafe { input(data, len) }) else {
            return;
        };
        *dst = String::from_utf8_lossy(bytes).into_owned();
    });
}

struct CHost {
    raw: GufoRealtimeVoiceHost,
}

impl CHost {
    /// # Safety
    /// `ptr` must be null or point to a valid `GufoRealtimeVoiceHost` whose
    /// callbacks and `context` remain valid for the duration of the call.
    #[allow(unsafe_code)]
    unsafe fn from_ptr(ptr: *const GufoRealtimeVoiceHost) -> Option<Self> {
        // SAFETY: caller upholds the contract above; `as_ref` handles null.
        let raw = unsafe { ptr.as_ref() }?;
        if raw.send_event.is_none()
            || raw.configure.is_none()
            || raw.on_vad_settings.is_none()
            || raw.append_audio.is_none()
            || raw.commit_audio.is_none()
            || raw.clear_audio.is_none()
            || raw.truncate_reply.is_none()
            || raw.add_user_text.is_none()
            || raw.add_tool_outputs.is_none()
            || raw.serves_model.is_none()
            || raw.transcribe_turn.is_none()
            || raw.delete_user_turn.is_none()
            || raw.supports_response_options.is_none()
        {
            return None;
        }
        Some(Self { raw: *raw })
    }
}

impl Host for CHost {
    #[allow(unsafe_code)]
    fn send_event(&mut self, json: &str) -> bool {
        let cb = self.raw.send_event.expect("validated send_event");
        let b = as_c_bytes(json.as_bytes());
        // SAFETY: caller guarantees `send_event` and `context` are valid for the call.
        unsafe { cb(self.raw.context, b.data, b.len) }
    }

    #[allow(unsafe_code)]
    fn configure(&mut self, settings: &Settings) -> Result<(), String> {
        let cb = self.raw.configure.expect("validated configure");
        let mut err = String::new();
        let ok = with_c_settings(settings, |c_settings| {
            // SAFETY: caller guarantees `configure` and `context` are valid for the call.
            unsafe {
                cb(
                    self.raw.context,
                    c_settings,
                    set_error_string,
                    (&raw mut err).cast::<c_void>(),
                )
            }
        });
        if ok { Ok(()) } else { Err(err) }
    }

    #[allow(unsafe_code)]
    fn on_vad_settings(&mut self, vad: Option<VadSettings>) {
        let cb = self.raw.on_vad_settings.expect("validated on_vad_settings");
        if let Some(v) = vad {
            let c_vad = GufoRealtimeVoiceVadSettings {
                threshold: v.threshold,
                prefix_padding_ms: v.prefix_padding_ms,
                silence_duration_ms: v.silence_duration_ms,
                create_response: v.create_response,
                interrupt_response: v.interrupt_response,
            };
            // SAFETY: caller guarantees `on_vad_settings` and `context` are valid.
            unsafe { cb(self.raw.context, true, &c_vad) };
        } else {
            // SAFETY: caller guarantees `on_vad_settings` and `context` are valid.
            unsafe { cb(self.raw.context, false, std::ptr::null()) };
        }
    }

    #[allow(unsafe_code)]
    fn append_audio(&mut self, pcm: &[u8], record: bool) -> Result<(), String> {
        let cb = self.raw.append_audio.expect("validated append_audio");
        let mut err = String::new();
        let b = as_c_bytes(pcm);
        // SAFETY: caller guarantees `append_audio` and `context` are valid.
        let ok = unsafe {
            cb(
                self.raw.context,
                b.data,
                b.len,
                record,
                set_error_string,
                (&raw mut err).cast::<c_void>(),
            )
        };
        if ok { Ok(()) } else { Err(err) }
    }

    #[allow(unsafe_code)]
    fn commit_audio(&mut self) -> Result<bool, String> {
        let cb = self.raw.commit_audio.expect("validated commit_audio");
        let mut committed = false;
        let mut err = String::new();
        // SAFETY: caller guarantees `commit_audio` and `context` are valid.
        let ok = unsafe {
            cb(
                self.raw.context,
                &raw mut committed,
                set_error_string,
                (&raw mut err).cast::<c_void>(),
            )
        };
        if ok { Ok(committed) } else { Err(err) }
    }

    #[allow(unsafe_code)]
    fn clear_audio(&mut self) -> Result<(), String> {
        let cb = self.raw.clear_audio.expect("validated clear_audio");
        let mut err = String::new();
        // SAFETY: caller guarantees `clear_audio` and `context` are valid.
        let ok = unsafe {
            cb(
                self.raw.context,
                set_error_string,
                (&raw mut err).cast::<c_void>(),
            )
        };
        if ok { Ok(()) } else { Err(err) }
    }

    #[allow(unsafe_code)]
    fn truncate_reply(&mut self, samples: usize) -> Result<(), String> {
        let cb = self.raw.truncate_reply.expect("validated truncate_reply");
        let mut err = String::new();
        // SAFETY: caller guarantees `truncate_reply` and `context` are valid.
        let ok = unsafe {
            cb(
                self.raw.context,
                samples,
                set_error_string,
                (&raw mut err).cast::<c_void>(),
            )
        };
        if ok { Ok(()) } else { Err(err) }
    }

    #[allow(unsafe_code)]
    fn add_user_text(&mut self, text: &str) -> Result<(), String> {
        let cb = self.raw.add_user_text.expect("validated add_user_text");
        let mut err = String::new();
        let b = as_c_bytes(text.as_bytes());
        // SAFETY: caller guarantees `add_user_text` and `context` are valid.
        let ok = unsafe {
            cb(
                self.raw.context,
                b.data,
                b.len,
                set_error_string,
                (&raw mut err).cast::<c_void>(),
            )
        };
        if ok { Ok(()) } else { Err(err) }
    }

    #[allow(unsafe_code)]
    fn add_tool_outputs(&mut self, outputs: &[String]) -> Result<(), String> {
        let cb = self
            .raw
            .add_tool_outputs
            .expect("validated add_tool_outputs");
        let mut err = String::new();
        let c_outputs: Vec<GufoRealtimeVoiceBytes> =
            outputs.iter().map(|o| as_c_bytes(o.as_bytes())).collect();
        let ptr = if c_outputs.is_empty() {
            std::ptr::null()
        } else {
            c_outputs.as_ptr()
        };
        // SAFETY: caller guarantees `add_tool_outputs` and `context` are valid.
        let ok = unsafe {
            cb(
                self.raw.context,
                ptr,
                c_outputs.len(),
                set_error_string,
                (&raw mut err).cast::<c_void>(),
            )
        };
        if ok { Ok(()) } else { Err(err) }
    }

    #[allow(unsafe_code)]
    fn serves_model(&mut self, name: &str) -> bool {
        let cb = self.raw.serves_model.expect("validated serves_model");
        let b = as_c_bytes(name.as_bytes());
        // SAFETY: caller guarantees `serves_model` and `context` are valid.
        unsafe { cb(self.raw.context, b.data, b.len) }
    }

    #[allow(unsafe_code)]
    fn transcribe_turn(&mut self, transcription: &Transcription) -> Result<String, String> {
        let cb = self.raw.transcribe_turn.expect("validated transcribe_turn");
        let mut transcript = String::new();
        let mut err = String::new();
        // SAFETY: caller guarantees `transcribe_turn` and `context` are valid;
        // `transcript` and `err` outlive the call.
        let ok = unsafe {
            cb(
                self.raw.context,
                as_c_bytes(transcription.language.as_bytes()),
                as_c_bytes(transcription.prompt.as_bytes()),
                set_error_string,
                (&raw mut transcript).cast::<c_void>(),
                set_error_string,
                (&raw mut err).cast::<c_void>(),
            )
        };
        if ok { Ok(transcript) } else { Err(err) }
    }

    #[allow(unsafe_code)]
    fn supports_response_options(&mut self) -> bool {
        let cb = self
            .raw
            .supports_response_options
            .expect("validated supports_response_options");
        // SAFETY: caller guarantees `supports_response_options` and `context` are valid.
        unsafe { cb(self.raw.context) }
    }

    #[allow(unsafe_code)]
    fn delete_user_turn(&mut self) -> Result<(), String> {
        let cb = self
            .raw
            .delete_user_turn
            .expect("validated delete_user_turn");
        let mut err = String::new();
        // SAFETY: caller guarantees `delete_user_turn` and `context` are valid.
        let ok = unsafe {
            cb(
                self.raw.context,
                set_error_string,
                (&raw mut err).cast::<c_void>(),
            )
        };
        if ok { Ok(()) } else { Err(err) }
    }
}

/// Creates a new Realtime voice session handle.
///
/// # Safety
/// `id` and `model` must point to `id_len` and `model_len` readable UTF-8 bytes
/// (or be null when length is 0).
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_new(
    id: *const u8,
    id_len: usize,
    model: *const u8,
    model_len: usize,
) -> *mut GufoRealtimeVoice {
    guard(std::ptr::null_mut(), || {
        // SAFETY: caller guarantees readable byte slices.
        let Some(id_bytes) = (unsafe { input(id, id_len) }) else {
            return std::ptr::null_mut();
        };
        // SAFETY: caller guarantees readable byte slices.
        let Some(model_bytes) = (unsafe { input(model, model_len) }) else {
            return std::ptr::null_mut();
        };
        let id = String::from_utf8_lossy(id_bytes).into_owned();
        let model = String::from_utf8_lossy(model_bytes).into_owned();
        Box::into_raw(Box::new(GufoRealtimeVoice {
            inner: Session::new(id, model),
        }))
    })
}

/// Frees a Realtime voice session handle. Null is a no-op.
///
/// # Safety
/// `session` must be null or a pointer returned by `gufo_realtime_voice_new`
/// that has not yet been freed and is no longer referenced by any thread.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_free(session: *mut GufoRealtimeVoice) {
    guard((), || {
        if !session.is_null() {
            // SAFETY: caller transfers ownership of a live handle.
            drop(unsafe { Box::from_raw(session) });
        }
    });
}

/// Formats and sends a `session_unavailable` server error event.
///
/// # Safety
/// `message` points to `message_len` readable bytes (or null at length 0).
/// `send_event` is valid to call with `send_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_session_unavailable(
    message: *const u8,
    message_len: usize,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        let Some(send_event) = send_event else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(msg_bytes) = (unsafe { input(message, message_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let msg = String::from_utf8_lossy(msg_bytes);
        let json = Session::session_unavailable(&msg);
        let b = as_c_bytes(json.as_bytes());
        // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
        if unsafe { send_event(send_ctx, b.data, b.len) } {
            GufoRealtimeVoiceStatus::Ok
        } else {
            GufoRealtimeVoiceStatus::Closed
        }
    })
}

/// Sends the initial `session.created` event.
///
/// # Safety
/// `session` must be a live handle; `send_event` must be valid with `send_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_session_created(
    session: *const GufoRealtimeVoice,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live for this call.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let Some(send_event) = send_event else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let json = session.inner.session_created();
        let b = as_c_bytes(json.as_bytes());
        // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
        if unsafe { send_event(send_ctx, b.data, b.len) } {
            GufoRealtimeVoiceStatus::Ok
        } else {
            GufoRealtimeVoiceStatus::Closed
        }
    })
}

/// Formats and sends a protocol `error` event.
///
/// # Safety
/// `session` must be live; byte pointers must be valid for their lengths;
/// `send_event` must be valid with `send_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_error(
    session: *const GufoRealtimeVoice,
    message: *const u8,
    message_len: usize,
    client_event: *const u8,
    client_event_len: usize,
    code: *const u8,
    code_len: usize,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let Some(send_event) = send_event else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let Some(msg_bytes) = (unsafe { input(message, message_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let Some(client_bytes) = (unsafe { input(client_event, client_event_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let Some(code_bytes) = (unsafe { input(code, code_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let msg = String::from_utf8_lossy(msg_bytes);
        let client_ev = String::from_utf8_lossy(client_bytes);
        let code_str = String::from_utf8_lossy(code_bytes);
        let json = session.inner.error_event(&msg, &client_ev, &code_str);
        let b = as_c_bytes(json.as_bytes());
        // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
        if unsafe { send_event(send_ctx, b.data, b.len) } {
            GufoRealtimeVoiceStatus::Ok
        } else {
            GufoRealtimeVoiceStatus::Closed
        }
    })
}

/// Classifies an incoming client WebSocket text frame on the reader thread.
///
/// # Safety
/// `session` must be live; `raw` must point to `raw_len` readable bytes;
/// `decision` must be writable; callbacks must be valid with their contexts.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_classify_client_event(
    session: *const GufoRealtimeVoice,
    raw: *const u8,
    raw_len: usize,
    vad_active: bool,
    responding: bool,
    decision: *mut GufoRealtimeVoiceDecision,
    on_pcm: Option<GufoRealtimeVoiceBytesFn>,
    pcm_ctx: *mut c_void,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `decision` is writable.
        let Some(decision) = (unsafe { decision.as_mut() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let Some(send_event) = send_event else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `raw` has `raw_len` readable bytes.
        let Some(raw_bytes) = (unsafe { input(raw, raw_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        match session
            .inner
            .classify_client_event(raw_bytes, vad_active, responding)
        {
            ClientDecision::Queue => {
                *decision = GufoRealtimeVoiceDecision::Queue;
                GufoRealtimeVoiceStatus::Ok
            }
            ClientDecision::Cancel => {
                *decision = GufoRealtimeVoiceDecision::Cancel;
                GufoRealtimeVoiceStatus::Ok
            }
            ClientDecision::VadAudio(pcm) => {
                let Some(on_pcm) = on_pcm else {
                    return GufoRealtimeVoiceStatus::Internal;
                };
                let b = as_c_bytes(&pcm);
                // SAFETY: caller guarantees `on_pcm` and `pcm_ctx` are valid.
                unsafe { on_pcm(pcm_ctx, b.data, b.len) };
                *decision = GufoRealtimeVoiceDecision::VadAudio;
                GufoRealtimeVoiceStatus::Ok
            }
            ClientDecision::Error(json) => {
                *decision = GufoRealtimeVoiceDecision::Handled;
                let b = as_c_bytes(json.as_bytes());
                // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
                if unsafe { send_event(send_ctx, b.data, b.len) } {
                    GufoRealtimeVoiceStatus::Ok
                } else {
                    GufoRealtimeVoiceStatus::Closed
                }
            }
        }
    })
}

/// Emits `input_audio_buffer.speech_started` and returns the allocated `item_id`.
///
/// # Safety
/// `session` must be live; callbacks must be valid with their contexts.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_vad_speech_started(
    session: *const GufoRealtimeVoice,
    audio_start_ms: i64,
    on_item_id: Option<GufoRealtimeVoiceBytesFn>,
    item_ctx: *mut c_void,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let (Some(on_item_id), Some(send_event)) = (on_item_id, send_event) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let (item_id, json) = session.inner.vad_speech_started(audio_start_ms);
        let id_b = as_c_bytes(item_id.as_bytes());
        // SAFETY: caller guarantees `on_item_id` and `item_ctx` are valid.
        unsafe { on_item_id(item_ctx, id_b.data, id_b.len) };
        let ev_b = as_c_bytes(json.as_bytes());
        // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
        if unsafe { send_event(send_ctx, ev_b.data, ev_b.len) } {
            GufoRealtimeVoiceStatus::Ok
        } else {
            GufoRealtimeVoiceStatus::Closed
        }
    })
}

/// Emits `input_audio_buffer.speech_stopped`.
///
/// # Safety
/// `session` must be live; `item_id` must point to `item_id_len` readable bytes;
/// `send_event` must be valid with `send_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_vad_speech_stopped(
    session: *const GufoRealtimeVoice,
    item_id: *const u8,
    item_id_len: usize,
    audio_end_ms: i64,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let Some(send_event) = send_event else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(id_bytes) = (unsafe { input(item_id, item_id_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let item_str = String::from_utf8_lossy(id_bytes);
        let json = session.inner.vad_speech_stopped(&item_str, audio_end_ms);
        let ev_b = as_c_bytes(json.as_bytes());
        // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
        if unsafe { send_event(send_ctx, ev_b.data, ev_b.len) } {
            GufoRealtimeVoiceStatus::Ok
        } else {
            GufoRealtimeVoiceStatus::Closed
        }
    })
}

/// Handles a queued client event on the worker thread.
///
/// # Safety
/// `session` must be live; `raw` must point to `raw_len` readable bytes;
/// `host` must point to a valid host table; `start_response` must be writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_handle_client_event(
    session: *const GufoRealtimeVoice,
    raw: *const u8,
    raw_len: usize,
    audio_open: bool,
    host: *const GufoRealtimeVoiceHost,
    start_response: *mut bool,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `start_response` is writable.
        let Some(start_response) = (unsafe { start_response.as_mut() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        *start_response = false;
        // SAFETY: caller guarantees `host` points to a valid host table.
        let Some(mut c_host) = (unsafe { CHost::from_ptr(host) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `raw` has `raw_len` readable bytes.
        let Some(raw_bytes) = (unsafe { input(raw, raw_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        match session
            .inner
            .handle_client_event(raw_bytes, audio_open, &mut c_host)
        {
            Ok(respond) => {
                *start_response = respond;
                GufoRealtimeVoiceStatus::Ok
            }
            Err(()) => GufoRealtimeVoiceStatus::Closed,
        }
    })
}

/// Appends VAD-sliced PCM16 audio on the worker thread.
///
/// # Safety
/// `session` must be live; `pcm` must point to `pcm_len` readable bytes;
/// `host` must point to a valid host table.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_handle_vad_audio(
    session: *const GufoRealtimeVoice,
    pcm: *const u8,
    pcm_len: usize,
    audio_open: bool,
    host: *const GufoRealtimeVoiceHost,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `host` points to a valid host table.
        let Some(mut c_host) = (unsafe { CHost::from_ptr(host) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `pcm` has `pcm_len` readable bytes.
        let Some(pcm_bytes) = (unsafe { input(pcm, pcm_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        match session
            .inner
            .handle_vad_audio(pcm_bytes, audio_open, &mut c_host)
        {
            Ok(()) => GufoRealtimeVoiceStatus::Ok,
            Err(()) => GufoRealtimeVoiceStatus::Closed,
        }
    })
}

/// Clears the audio buffer after a too-short VAD burst on the worker thread.
///
/// # Safety
/// `session` must be live; `host` must point to a valid host table.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_handle_vad_clear(
    session: *const GufoRealtimeVoice,
    host: *const GufoRealtimeVoiceHost,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `host` points to a valid host table.
        let Some(mut c_host) = (unsafe { CHost::from_ptr(host) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        match session.inner.handle_vad_clear(&mut c_host) {
            Ok(()) => GufoRealtimeVoiceStatus::Ok,
            Err(()) => GufoRealtimeVoiceStatus::Closed,
        }
    })
}

/// Commits a VAD turn on the worker thread and optionally starts a response.
///
/// # Safety
/// `session` must be live; `item_id` must point to `item_id_len` readable bytes;
/// `host` must point to a valid host table; `start_response` must be writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_handle_vad_commit(
    session: *const GufoRealtimeVoice,
    item_id: *const u8,
    item_id_len: usize,
    respond: bool,
    interrupt_active: bool,
    host: *const GufoRealtimeVoiceHost,
    start_response: *mut bool,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `start_response` is writable.
        let Some(start_response) = (unsafe { start_response.as_mut() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        *start_response = false;
        // SAFETY: caller guarantees `host` points to a valid host table.
        let Some(mut c_host) = (unsafe { CHost::from_ptr(host) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `item_id` has `item_id_len` readable bytes.
        let Some(id_bytes) = (unsafe { input(item_id, item_id_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let id_str = String::from_utf8_lossy(id_bytes);
        match session
            .inner
            .handle_vad_commit(&id_str, respond, interrupt_active, &mut c_host)
        {
            Ok(start) => {
                *start_response = start;
                GufoRealtimeVoiceStatus::Ok
            }
            Err(()) => GufoRealtimeVoiceStatus::Closed,
        }
    })
}

/// Sends `response.created` and passes the active response's settings and
/// `response_id` to `on_active`.
///
/// # Safety
/// `session` must be live; `on_active` and `send_event` must be valid with
/// their contexts.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_response_start(
    session: *const GufoRealtimeVoice,
    on_active: Option<GufoRealtimeVoiceActiveResponseFn>,
    active_ctx: *mut c_void,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let (Some(on_active), Some(send_event)) = (on_active, send_event) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let mut send = |json: &str| {
            let b = as_c_bytes(json.as_bytes());
            // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
            unsafe { send_event(send_ctx, b.data, b.len) }
        };
        let Ok((settings, response_id)) = session.inner.response_start(&mut send) else {
            return GufoRealtimeVoiceStatus::Closed;
        };
        with_c_settings(&settings, |c_settings| {
            let id_b = as_c_bytes(response_id.as_bytes());
            // SAFETY: caller guarantees `on_active` and `active_ctx` are valid.
            unsafe { on_active(active_ctx, c_settings, id_b.data, id_b.len) };
        });
        GufoRealtimeVoiceStatus::Ok
    })
}

/// Emits a reasoning delta for the active response.
///
/// # Safety
/// `session` must be live; `text` must point to `text_len` readable bytes;
/// `send_event` must be valid with `send_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_response_reasoning(
    session: *const GufoRealtimeVoice,
    text: *const u8,
    text_len: usize,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let Some(send_event) = send_event else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(bytes) = (unsafe { input(text, text_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let s = String::from_utf8_lossy(bytes);
        let mut send = |json: &str| {
            let b = as_c_bytes(json.as_bytes());
            // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
            unsafe { send_event(send_ctx, b.data, b.len) }
        };
        match session.inner.response_reasoning(&s, &mut send) {
            Ok(()) => GufoRealtimeVoiceStatus::Ok,
            Err(()) => GufoRealtimeVoiceStatus::Closed,
        }
    })
}

/// Emits a text or audio-transcript delta for the active response.
///
/// # Safety
/// `session` must be live; `text` must point to `text_len` readable bytes;
/// `send_event` must be valid with `send_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_response_text(
    session: *const GufoRealtimeVoice,
    text: *const u8,
    text_len: usize,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let Some(send_event) = send_event else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(bytes) = (unsafe { input(text, text_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let s = String::from_utf8_lossy(bytes);
        let mut send = |json: &str| {
            let b = as_c_bytes(json.as_bytes());
            // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
            unsafe { send_event(send_ctx, b.data, b.len) }
        };
        match session.inner.response_text(&s, &mut send) {
            Ok(()) => GufoRealtimeVoiceStatus::Ok,
            Err(()) => GufoRealtimeVoiceStatus::Closed,
        }
    })
}

/// Emits an audio delta for the active response.
///
/// # Safety
/// `session` must be live; `pcm` must point to `pcm_len` readable bytes;
/// `send_event` must be valid with `send_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_response_audio(
    session: *const GufoRealtimeVoice,
    pcm: *const u8,
    pcm_len: usize,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let Some(send_event) = send_event else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(bytes) = (unsafe { input(pcm, pcm_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let mut send = |json: &str| {
            let b = as_c_bytes(json.as_bytes());
            // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
            unsafe { send_event(send_ctx, b.data, b.len) }
        };
        match session.inner.response_audio(bytes, &mut send) {
            Ok(()) => GufoRealtimeVoiceStatus::Ok,
            Err(()) => GufoRealtimeVoiceStatus::Closed,
        }
    })
}

/// Finishes the active response, emitting item/part `.done` events, function
/// calls, and `response.done`.
///
/// # Safety
/// `session` must be live; `outcome` and its slices must be readable;
/// `send_event` must be valid with `send_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_realtime_voice_response_finish(
    session: *const GufoRealtimeVoice,
    outcome: *const GufoRealtimeVoiceResponseOutcome,
    send_event: Option<GufoRealtimeVoiceSendFn>,
    send_ctx: *mut c_void,
) -> GufoRealtimeVoiceStatus {
    guard(GufoRealtimeVoiceStatus::Internal, || {
        // SAFETY: caller guarantees `session` is live.
        let Some(session) = (unsafe { session.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `outcome` is readable.
        let Some(c_outcome) = (unsafe { outcome.as_ref() }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let Some(send_event) = send_event else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `failure` bytes are readable.
        let Some(failure_bytes) = (unsafe { input(c_outcome.failure.data, c_outcome.failure.len) })
        else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        // SAFETY: caller guarantees `calls` points to `calls_len` readable entries.
        let Some(slice) = (unsafe { input(c_outcome.calls, c_outcome.calls_len) }) else {
            return GufoRealtimeVoiceStatus::Internal;
        };
        let mut calls = Vec::new();
        if calls.try_reserve_exact(slice.len()).is_err() {
            return GufoRealtimeVoiceStatus::Internal;
        }
        for call in slice {
            // SAFETY: caller guarantees readable byte slices.
            let Some(name_b) = (unsafe { input(call.name.data, call.name.len) }) else {
                return GufoRealtimeVoiceStatus::Internal;
            };
            // SAFETY: caller guarantees readable byte slices.
            let Some(args_b) = (unsafe { input(call.arguments.data, call.arguments.len) }) else {
                return GufoRealtimeVoiceStatus::Internal;
            };
            calls.push(FunctionCall {
                name: String::from_utf8_lossy(name_b).into_owned(),
                arguments: String::from_utf8_lossy(args_b).into_owned(),
            });
        }
        let outcome = ResponseOutcome {
            ok: c_outcome.ok,
            cancelled: c_outcome.cancelled,
            incomplete: c_outcome.incomplete,
            input_tokens: c_outcome.input_tokens,
            output_tokens: c_outcome.output_tokens,
            calls,
            failure: String::from_utf8_lossy(failure_bytes).into_owned(),
        };
        let mut send = |json: &str| {
            let b = as_c_bytes(json.as_bytes());
            // SAFETY: caller guarantees `send_event` and `send_ctx` are valid.
            unsafe { send_event(send_ctx, b.data, b.len) }
        };
        match session.inner.response_finish(&outcome, &mut send) {
            Ok(()) => GufoRealtimeVoiceStatus::Ok,
            Err(()) => GufoRealtimeVoiceStatus::Closed,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(unsafe_code)]
    unsafe extern "C" fn collect_event(ctx: *mut c_void, data: *const u8, len: usize) -> bool {
        // SAFETY: test passes a `&mut Vec<String>` pointer as `ctx`.
        let events = unsafe { &mut *ctx.cast::<Vec<String>>() };
        // SAFETY: callee passes `len` readable UTF-8 bytes.
        let bytes = unsafe { input(data, len) }.unwrap();
        events.push(String::from_utf8_lossy(bytes).into_owned());
        true
    }

    #[test]
    #[allow(unsafe_code)]
    fn c_abi_lifecycle_and_session_created() {
        let id = b"test";
        let model = b"voice";
        // SAFETY: valid byte slices.
        let handle =
            unsafe { gufo_realtime_voice_new(id.as_ptr(), id.len(), model.as_ptr(), model.len()) };
        assert!(!handle.is_null());
        let mut events = Vec::<String>::new();
        // SAFETY: handle is live and `events` lives for the call.
        let status = unsafe {
            gufo_realtime_voice_session_created(
                handle,
                Some(collect_event),
                (&raw mut events).cast::<c_void>(),
            )
        };
        assert_eq!(status, GufoRealtimeVoiceStatus::Ok);
        assert_eq!(events.len(), 1);
        assert!(events[0].contains(r#""type":"session.created""#));
        assert!(events[0].contains(r#""id":"sess_test""#));
        // SAFETY: transfers ownership back to Rust.
        unsafe { gufo_realtime_voice_free(handle) };
        // SAFETY: null is a no-op.
        unsafe { gufo_realtime_voice_free(std::ptr::null_mut()) };
    }
}
