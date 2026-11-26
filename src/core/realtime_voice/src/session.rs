//! Realtime voice conversation session protocol state machine.
//!
//! Handles OpenAI Realtime client events (`session.update`,
//! `input_audio_buffer.*`, `conversation.item.*`, `response.create`,
//! `response.cancel`) and formats server events (`session.created` through
//! `response.done`). Audio buffers, VAD, ASR, TTS and generation stay in C++.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use gufo_json::{Limits, Number, Numbers, Object, Value, ValueError, dump, parse_value};

pub const SAMPLE_RATE: u32 = 24_000;
/// One user turn holds at most two minutes of audio.
pub const MAX_TURN_SAMPLES: usize = 120 * SAMPLE_RATE as usize;
const MAX_TURN_BYTES: usize = MAX_TURN_SAMPLES * 2;
const MIN_COMMIT_SAMPLES: usize = SAMPLE_RATE as usize / 10;
const MAX_INSTRUCTIONS_BYTES: usize = 65_536;
const MAX_TOOLS: usize = 64;
const MAX_OUTPUT_TOKENS: usize = 4_096;
const MAX_PENDING_OUTPUTS: usize = 64;
const MAX_TRANSCRIPTION_PROMPT_BYTES: usize = 16_384;
const MAX_NAME_BYTES: usize = 64;
const CLIENT_EVENT_LIMITS: Limits = Limits {
    bytes: 16 << 20,
    values: 65_536,
};

const INITIAL_CONFIG: &str = r#"{"type":"realtime","object":"realtime.session","output_modalities":["audio"],"instructions":"","tools":[],"tool_choice":"auto","max_output_tokens":"inf","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":null,"turn_detection":null,"noise_reduction":null},"output":{"format":{"type":"audio/pcm","rate":24000}}}}"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    pub instructions: String,
    pub audio_output: bool,
    pub tools: Vec<String>,
    pub max_output_tokens: usize,
    pub reasoning: bool,
    /// `audio.output.voice`; empty: the conversation's own voice.
    pub voice: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            instructions: String::new(),
            audio_output: true,
            tools: Vec::new(),
            max_output_tokens: MAX_OUTPUT_TOKENS,
            reasoning: false,
            voice: String::new(),
        }
    }
}

/// `audio.input.transcription`: each committed user turn is transcribed by
/// the served model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Transcription {
    /// Empty: the model detects the language.
    pub language: String,
    pub prompt: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadSettings {
    pub threshold: f64,
    pub prefix_padding_ms: i32,
    pub silence_duration_ms: i32,
    pub create_response: bool,
    pub interrupt_response: bool,
}

impl Default for VadSettings {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
            create_response: true,
            interrupt_response: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseOutcome {
    pub ok: bool,
    pub cancelled: bool,
    pub incomplete: bool,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub calls: Vec<FunctionCall>,
    pub failure: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientDecision {
    Queue,
    Cancel,
    VadAudio(Vec<u8>),
    Error(String),
}

pub trait Host {
    fn send_event(&mut self, json: &str) -> bool;
    fn configure(&mut self, settings: &Settings) -> Result<(), String>;
    fn on_vad_settings(&mut self, vad: Option<VadSettings>);
    /// `record`: the host keeps the turn's audio for `transcribe_turn`.
    fn append_audio(&mut self, pcm: &[u8], record: bool) -> Result<(), String>;
    fn commit_audio(&mut self) -> Result<bool, String>;
    fn clear_audio(&mut self) -> Result<(), String>;
    fn truncate_reply(&mut self, samples: usize) -> Result<(), String>;
    fn add_user_text(&mut self, text: &str) -> Result<(), String>;
    fn add_tool_outputs(&mut self, outputs: &[String]) -> Result<(), String>;
    /// Whether `name` names the served model (`audio.input.transcription.model`).
    fn serves_model(&mut self, name: &str) -> bool;
    /// Transcribes the recorded audio of the user turn committed last.
    fn transcribe_turn(&mut self, transcription: &Transcription) -> Result<String, String>;
    /// Removes the latest user turn, which nothing has followed.
    fn delete_user_turn(&mut self) -> Result<(), String>;
}

struct Shared {
    id: String,
    sequence: AtomicUsize,
    items: AtomicUsize,
}

impl Shared {
    fn next_event_id(&self) -> String {
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        format!("event_{}_{}", self.id, seq)
    }

    fn next_item_id(&self) -> String {
        let n = self.items.fetch_add(1, Ordering::Relaxed) + 1;
        format!("item_{}_{}", self.id, n)
    }

    fn format_event(&self, mut event: Object) -> String {
        set_member(&mut event, "event_id", Value::String(self.next_event_id()));
        dump_json(&Value::Object(event))
    }

    fn simple_event(&self, kind: &str, key: Option<(&str, Value)>) -> String {
        let mut event = Object::new();
        push_member(&mut event, "type", Value::String(kind.to_string()));
        if let Some((k, v)) = key {
            push_member(&mut event, k, v);
        }
        self.format_event(event)
    }

    fn error_event(&self, message: &str, client_event: &str, code: &str) -> String {
        let mut error = Object::new();
        push_member(
            &mut error,
            "type",
            Value::String("invalid_request_error".to_string()),
        );
        push_member(&mut error, "code", Value::String(code.to_string()));
        push_member(&mut error, "message", Value::String(message.to_string()));
        if !client_event.is_empty() {
            push_member(
                &mut error,
                "event_id",
                Value::String(client_event.to_string()),
            );
        }
        self.simple_event("error", Some(("error", Value::Object(error))))
    }
}

struct ActiveResponse {
    response_id: String,
    settings: Settings,
    response: Object,
    item: String,
    transcript: String,
    reasoning: String,
    scratch: String,
    /// The deletable user message the reply follows; deletable again when
    /// the reply adds no item.
    deletable: Option<(String, String)>,
}

struct State {
    config: Value,
    settings: Settings,
    responses: usize,
    turn_samples: usize,
    previous_item: String,
    last_reply_item: String,
    outputs: Vec<String>,
    active_response: Option<ActiveResponse>,
    transcription: Option<Transcription>,
    /// The latest item when it is a user message, with the item before it:
    /// the only item `conversation.item.delete` removes.
    deletable: Option<(String, String)>,
}

pub struct Session {
    shared: Shared,
    state: Mutex<State>,
}

impl Session {
    pub fn new(id: String, model: String) -> Self {
        let mut config = parse_value(INITIAL_CONFIG.as_bytes(), Limits::NONE)
            .expect("static initial config is valid JSON");
        if let Value::Object(obj) = &mut config {
            set_member(obj, "id", Value::String(format!("sess_{id}")));
            set_member(obj, "model", Value::String(model));
        }
        Self {
            shared: Shared {
                id,
                sequence: AtomicUsize::new(0),
                items: AtomicUsize::new(0),
            },
            state: Mutex::new(State {
                config,
                settings: Settings::default(),
                responses: 0,
                turn_samples: 0,
                previous_item: String::new(),
                last_reply_item: String::new(),
                outputs: Vec::new(),
                active_response: None,
                transcription: None,
                deletable: None,
            }),
        }
    }

    pub fn session_unavailable(message: &str) -> String {
        let mut error = Object::new();
        push_member(
            &mut error,
            "type",
            Value::String("server_error".to_string()),
        );
        push_member(
            &mut error,
            "code",
            Value::String("session_unavailable".to_string()),
        );
        push_member(&mut error, "message", Value::String(message.to_string()));
        let mut event = Object::new();
        push_member(&mut event, "type", Value::String("error".to_string()));
        push_member(&mut event, "error", Value::Object(error));
        dump_json(&Value::Object(event))
    }

    pub fn session_created(&self) -> String {
        let config = self.state.lock().expect("session lock").config.clone();
        self.shared
            .simple_event("session.created", Some(("session", config)))
    }

    pub fn error_event(&self, message: &str, client_event: &str, code: &str) -> String {
        self.shared.error_event(message, client_event, code)
    }

    /// Reader-side classification: validates JSON framing, handles
    /// `response.cancel` immediately, and decodes `input_audio_buffer.append`
    /// PCM16 when server VAD is active.
    pub fn classify_client_event(
        &self,
        raw: &[u8],
        vad_active: bool,
        responding: bool,
    ) -> ClientDecision {
        let message = match parse_value(raw, CLIENT_EVENT_LIMITS) {
            Ok(v) => v,
            Err(e) => {
                return ClientDecision::Error(self.shared.error_event(
                    &format_json_error(e),
                    "",
                    "invalid_request",
                ));
            }
        };
        let Some(obj) = message.as_object() else {
            return ClientDecision::Error(self.shared.error_event(
                "event must be an object",
                "",
                "invalid_request",
            ));
        };
        let client_event = member_str(obj, "event_id");
        let kind = member_str(obj, "type");
        if kind.starts_with('_') {
            return ClientDecision::Error(self.shared.error_event(
                &format!("unsupported realtime conversation event: {kind}"),
                client_event,
                "invalid_request",
            ));
        }
        if kind == "input_audio_buffer.append" && vad_active {
            if let Err(msg) = check_members(obj, &["type", "event_id", "audio"], "event") {
                return ClientDecision::Error(self.shared.error_event(
                    &msg,
                    client_event,
                    "invalid_request",
                ));
            }
            return match decode_audio_pcm(obj.get("audio")) {
                Ok(pcm) => ClientDecision::VadAudio(pcm),
                Err(msg) => ClientDecision::Error(self.shared.error_event(
                    &msg,
                    client_event,
                    "invalid_request",
                )),
            };
        }
        if kind == "response.cancel" {
            if let Err(msg) = check_members(obj, &["type", "event_id", "response_id"], "event") {
                return ClientDecision::Error(self.shared.error_event(
                    &msg,
                    client_event,
                    "invalid_request",
                ));
            }
            if !responding {
                return ClientDecision::Error(self.shared.error_event(
                    "no response is in progress",
                    client_event,
                    "response_cancel_not_active",
                ));
            }
            return ClientDecision::Cancel;
        }
        ClientDecision::Queue
    }

    pub fn vad_speech_started(&self, audio_start_ms: i64) -> (String, String) {
        let item_id = self.shared.next_item_id();
        let mut out = Object::new();
        push_member(&mut out, "item_id", Value::String(item_id.clone()));
        push_member(
            &mut out,
            "type",
            Value::String("input_audio_buffer.speech_started".to_string()),
        );
        push_member(&mut out, "audio_start_ms", num_i64(audio_start_ms));
        (item_id, self.shared.format_event(out))
    }

    pub fn vad_speech_stopped(&self, item_id: &str, audio_end_ms: i64) -> String {
        let mut out = Object::new();
        push_member(&mut out, "item_id", Value::String(item_id.to_string()));
        push_member(
            &mut out,
            "type",
            Value::String("input_audio_buffer.speech_stopped".to_string()),
        );
        push_member(&mut out, "audio_end_ms", num_i64(audio_end_ms));
        self.shared.format_event(out)
    }

    /// Worker-side handler for a queued client event. Returns `Ok(start_response)`
    /// or `Err(())` when `host.send_event` reports the connection closed.
    pub fn handle_client_event(
        &self,
        raw: &[u8],
        audio_open: bool,
        host: &mut dyn Host,
    ) -> Result<bool, ()> {
        let mut state = self.state.lock().expect("session lock");
        let message = match parse_value(raw, CLIENT_EVENT_LIMITS) {
            Ok(v) => v,
            Err(e) => {
                let err = self
                    .shared
                    .error_event(&format_json_error(e), "", "invalid_request");
                return if host.send_event(&err) {
                    Ok(false)
                } else {
                    Err(())
                };
            }
        };
        let client_event = message
            .as_object()
            .map(|o| member_str(o, "event_id").to_string())
            .unwrap_or_default();
        match self.dispatch_client_event(&mut state, &message, audio_open, host) {
            Ok(start_response) => Ok(start_response),
            Err(DispatchError::Closed) => Err(()),
            Err(DispatchError::Protocol(msg)) => {
                let err = self
                    .shared
                    .error_event(&msg, &client_event, "invalid_request");
                if host.send_event(&err) {
                    Ok(false)
                } else {
                    Err(())
                }
            }
        }
    }

    pub fn handle_vad_audio(
        &self,
        pcm: &[u8],
        audio_open: bool,
        host: &mut dyn Host,
    ) -> Result<(), ()> {
        let mut state = self.state.lock().expect("session lock");
        match self.append_pcm(&mut state, pcm, audio_open, host) {
            Ok(()) => Ok(()),
            Err(DispatchError::Closed) => Err(()),
            Err(DispatchError::Protocol(msg)) => {
                let err = self.shared.error_event(&msg, "", "invalid_request");
                if host.send_event(&err) {
                    Ok(())
                } else {
                    Err(())
                }
            }
        }
    }

    pub fn handle_vad_clear(&self, host: &mut dyn Host) -> Result<(), ()> {
        let mut state = self.state.lock().expect("session lock");
        if let Err(msg) = host.clear_audio() {
            let err = self.shared.error_event(&msg, "", "invalid_request");
            return if host.send_event(&err) {
                Ok(())
            } else {
                Err(())
            };
        }
        state.turn_samples = 0;
        let ev = self.shared.simple_event("input_audio_buffer.cleared", None);
        if host.send_event(&ev) {
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn handle_vad_commit(
        &self,
        item_id: &str,
        respond: bool,
        interrupt_active: bool,
        host: &mut dyn Host,
    ) -> Result<bool, ()> {
        let mut state = self.state.lock().expect("session lock");
        match self.commit_turn(&mut state, item_id, host) {
            Ok(false) => Ok(false),
            Ok(true) => {
                if respond && !interrupt_active {
                    let settings = state.settings.clone();
                    self.begin_response(&mut state, settings);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            Err(DispatchError::Closed) => Err(()),
            Err(DispatchError::Protocol(msg)) => {
                let err = self.shared.error_event(&msg, "", "invalid_request");
                if host.send_event(&err) {
                    Ok(false)
                } else {
                    Err(())
                }
            }
        }
    }

    pub fn response_start(
        &self,
        send: &mut dyn FnMut(&str) -> bool,
    ) -> Result<(Settings, String), ()> {
        let state = self.state.lock().expect("session lock");
        let Some(active) = state.active_response.as_ref() else {
            return Err(());
        };
        let ev = self.shared.simple_event(
            "response.created",
            Some(("response", Value::Object(active.response.clone()))),
        );
        if !send(&ev) {
            return Err(());
        }
        Ok((active.settings.clone(), active.response_id.clone()))
    }

    pub fn response_reasoning(
        &self,
        text: &str,
        send: &mut dyn FnMut(&str) -> bool,
    ) -> Result<(), ()> {
        let mut state = self.state.lock().expect("session lock");
        let Some(active) = state.active_response.as_mut() else {
            return Err(());
        };
        active.reasoning.push_str(text);
        open_response_item(&self.shared, active, send)?;
        send_delta(
            &self.shared,
            active,
            "response.reasoning_text.delta",
            text,
            send,
        )
    }

    pub fn response_text(&self, text: &str, send: &mut dyn FnMut(&str) -> bool) -> Result<(), ()> {
        let mut state = self.state.lock().expect("session lock");
        let Some(active) = state.active_response.as_mut() else {
            return Err(());
        };
        active.transcript.push_str(text);
        open_response_item(&self.shared, active, send)?;
        let kind = if active.settings.audio_output {
            "response.output_audio_transcript.delta"
        } else {
            "response.output_text.delta"
        };
        send_delta(&self.shared, active, kind, text, send)
    }

    pub fn response_audio(&self, pcm: &[u8], send: &mut dyn FnMut(&str) -> bool) -> Result<(), ()> {
        let mut state = self.state.lock().expect("session lock");
        let Some(active) = state.active_response.as_mut() else {
            return Err(());
        };
        open_response_item(&self.shared, active, send)?;
        send_audio_delta(&self.shared, active, pcm, send)
    }

    pub fn response_finish(
        &self,
        outcome: &ResponseOutcome,
        send: &mut dyn FnMut(&str) -> bool,
    ) -> Result<(), ()> {
        let mut state = self.state.lock().expect("session lock");
        let Some(mut active) = state.active_response.take() else {
            return Err(());
        };
        let status = if !outcome.ok {
            "failed"
        } else if outcome.cancelled {
            "cancelled"
        } else if outcome.incomplete {
            "incomplete"
        } else {
            "completed"
        };
        let mut output: Vec<Value> = Vec::new();
        if !active.item.is_empty() {
            let kind = if active.settings.audio_output {
                "audio"
            } else {
                "text"
            };
            if active.settings.audio_output {
                let mut done = Object::new();
                push_member(
                    &mut done,
                    "type",
                    Value::String("response.output_audio.done".to_string()),
                );
                push_member(
                    &mut done,
                    "response_id",
                    Value::String(active.response_id.clone()),
                );
                push_member(&mut done, "item_id", Value::String(active.item.clone()));
                push_member(&mut done, "output_index", num_i64(0));
                push_member(&mut done, "content_index", num_i64(0));
                if !send(&self.shared.format_event(done)) {
                    return Err(());
                }
            }
            let mut text_done = Object::new();
            push_member(
                &mut text_done,
                "type",
                Value::String(
                    if active.settings.audio_output {
                        "response.output_audio_transcript.done"
                    } else {
                        "response.output_text.done"
                    }
                    .to_string(),
                ),
            );
            push_member(
                &mut text_done,
                "response_id",
                Value::String(active.response_id.clone()),
            );
            push_member(
                &mut text_done,
                "item_id",
                Value::String(active.item.clone()),
            );
            push_member(&mut text_done, "output_index", num_i64(0));
            push_member(&mut text_done, "content_index", num_i64(0));
            push_member(
                &mut text_done,
                if active.settings.audio_output {
                    "transcript"
                } else {
                    "text"
                },
                Value::String(active.transcript.clone()),
            );
            if !send(&self.shared.format_event(text_done)) {
                return Err(());
            }

            let mut part = Object::new();
            push_member(&mut part, "type", Value::String(kind.to_string()));
            push_member(
                &mut part,
                if active.settings.audio_output {
                    "transcript"
                } else {
                    "text"
                },
                Value::String(active.transcript.clone()),
            );
            let part_val = Value::Object(part);

            let mut part_done = Object::new();
            push_member(
                &mut part_done,
                "type",
                Value::String("response.content_part.done".to_string()),
            );
            push_member(
                &mut part_done,
                "response_id",
                Value::String(active.response_id.clone()),
            );
            push_member(
                &mut part_done,
                "item_id",
                Value::String(active.item.clone()),
            );
            push_member(&mut part_done, "output_index", num_i64(0));
            push_member(&mut part_done, "content_index", num_i64(0));
            push_member(&mut part_done, "part", part_val.clone());
            if !send(&self.shared.format_event(part_done)) {
                return Err(());
            }

            let mut message = Object::new();
            push_member(
                &mut message,
                "object",
                Value::String("realtime.item".to_string()),
            );
            push_member(&mut message, "type", Value::String("message".to_string()));
            push_member(&mut message, "role", Value::String("assistant".to_string()));
            push_member(&mut message, "id", Value::String(active.item.clone()));
            push_member(
                &mut message,
                "status",
                Value::String(
                    if status == "completed" {
                        "completed"
                    } else {
                        "incomplete"
                    }
                    .to_string(),
                ),
            );
            push_member(&mut message, "content", Value::Array(vec![part_val]));
            let message_val = Value::Object(message);

            let mut item_done = Object::new();
            push_member(
                &mut item_done,
                "type",
                Value::String("response.output_item.done".to_string()),
            );
            push_member(
                &mut item_done,
                "response_id",
                Value::String(active.response_id.clone()),
            );
            push_member(&mut item_done, "output_index", num_i64(0));
            push_member(&mut item_done, "item", message_val.clone());
            if !send(&self.shared.format_event(item_done)) {
                return Err(());
            }
            state.previous_item = active.item.clone();
            state.last_reply_item = active.item.clone();
            output.push(message_val);
        }

        for call in &outcome.calls {
            let call_item = self.shared.next_item_id();
            let index = output.len() as i64;
            let call_id = format!("call_{call_item}");
            let mut entry = Object::new();
            push_member(&mut entry, "id", Value::String(call_item.clone()));
            push_member(
                &mut entry,
                "object",
                Value::String("realtime.item".to_string()),
            );
            push_member(
                &mut entry,
                "type",
                Value::String("function_call".to_string()),
            );
            push_member(
                &mut entry,
                "status",
                Value::String("in_progress".to_string()),
            );
            push_member(&mut entry, "call_id", Value::String(call_id.clone()));
            push_member(&mut entry, "name", Value::String(call.name.clone()));
            push_member(&mut entry, "arguments", Value::String(String::new()));

            let mut added = Object::new();
            push_member(
                &mut added,
                "type",
                Value::String("response.output_item.added".to_string()),
            );
            push_member(
                &mut added,
                "response_id",
                Value::String(active.response_id.clone()),
            );
            push_member(&mut added, "output_index", num_i64(index));
            push_member(&mut added, "item", Value::Object(entry.clone()));
            if !send(&self.shared.format_event(added)) {
                return Err(());
            }

            let mut arguments = Object::new();
            push_member(
                &mut arguments,
                "type",
                Value::String("response.function_call_arguments.done".to_string()),
            );
            push_member(
                &mut arguments,
                "response_id",
                Value::String(active.response_id.clone()),
            );
            push_member(&mut arguments, "item_id", Value::String(call_item.clone()));
            push_member(&mut arguments, "output_index", num_i64(index));
            push_member(&mut arguments, "call_id", Value::String(call_id));
            push_member(&mut arguments, "name", Value::String(call.name.clone()));
            push_member(
                &mut arguments,
                "arguments",
                Value::String(call.arguments.clone()),
            );
            if !send(&self.shared.format_event(arguments)) {
                return Err(());
            }

            set_member(&mut entry, "status", Value::String("completed".to_string()));
            set_member(
                &mut entry,
                "arguments",
                Value::String(call.arguments.clone()),
            );
            let entry_val = Value::Object(entry);
            let mut done = Object::new();
            push_member(
                &mut done,
                "type",
                Value::String("response.output_item.done".to_string()),
            );
            push_member(
                &mut done,
                "response_id",
                Value::String(active.response_id.clone()),
            );
            push_member(&mut done, "output_index", num_i64(index));
            push_member(&mut done, "item", entry_val.clone());
            if !send(&self.shared.format_event(done)) {
                return Err(());
            }
            state.previous_item = call_item;
            output.push(entry_val);
        }

        set_member(
            &mut active.response,
            "status",
            Value::String(status.to_string()),
        );
        set_member(&mut active.response, "output", Value::Array(output));
        if !outcome.ok {
            let mut err = Object::new();
            push_member(&mut err, "type", Value::String("server_error".to_string()));
            push_member(&mut err, "message", Value::String(outcome.failure.clone()));
            let mut details = Object::new();
            push_member(&mut details, "type", Value::String("failed".to_string()));
            push_member(&mut details, "error", Value::Object(err));
            set_member(
                &mut active.response,
                "status_details",
                Value::Object(details),
            );
        } else {
            let mut usage = Object::new();
            push_member(&mut usage, "input_tokens", num_i64(outcome.input_tokens));
            push_member(&mut usage, "output_tokens", num_i64(outcome.output_tokens));
            push_member(
                &mut usage,
                "total_tokens",
                num_i64(outcome.input_tokens.saturating_add(outcome.output_tokens)),
            );
            set_member(&mut active.response, "usage", Value::Object(usage));
            if !active.reasoning.is_empty() {
                set_member(
                    &mut active.response,
                    "reasoning",
                    Value::String(active.reasoning),
                );
            }
        }
        if active.item.is_empty() && outcome.calls.is_empty() {
            state.deletable = active.deletable;
        }
        let ev = self.shared.simple_event(
            "response.done",
            Some(("response", Value::Object(active.response))),
        );
        if send(&ev) { Ok(()) } else { Err(()) }
    }

    fn dispatch_client_event(
        &self,
        state: &mut State,
        message: &Value,
        audio_open: bool,
        host: &mut dyn Host,
    ) -> Result<bool, DispatchError> {
        let obj = message
            .as_object()
            .ok_or_else(|| DispatchError::Protocol("event must be an object".to_string()))?;
        let kind = member_str(obj, "type");
        match kind {
            "session.update" => {
                check_members(obj, &["type", "event_id", "session"], "event")?;
                let session_val = obj.get("session").ok_or_else(|| {
                    DispatchError::Protocol("session.update requires session".to_string())
                })?;
                let session_obj = check_object_members(
                    session_val,
                    &[
                        "type",
                        "instructions",
                        "output_modalities",
                        "tools",
                        "tool_choice",
                        "max_output_tokens",
                        "audio",
                        "reasoning",
                    ],
                    "session",
                )?;
                let mut next = state.config.clone();
                let mut settings = state.settings.clone();
                let mut vad: Option<VadSettings> = None;
                let mut vad_changed = false;
                let mut transcription = state.transcription.clone();
                for (name, value) in session_obj.members() {
                    match name.as_str() {
                        "type" => {
                            if value.as_str() != Some("realtime") {
                                return Err(DispatchError::Protocol(
                                    "conversation sessions have type realtime".to_string(),
                                ));
                            }
                        }
                        "instructions" => {
                            let Some(s) = value.as_str() else {
                                return Err(DispatchError::Protocol(
                                    "instructions must be a string of at most 64 KiB".to_string(),
                                ));
                            };
                            if s.len() > MAX_INSTRUCTIONS_BYTES {
                                return Err(DispatchError::Protocol(
                                    "instructions must be a string of at most 64 KiB".to_string(),
                                ));
                            }
                            settings.instructions = s.to_string();
                        }
                        "output_modalities" => {
                            settings.audio_output = parse_audio_modality(value)?;
                        }
                        "tools" => {
                            let Some(items) = value.as_array() else {
                                return Err(DispatchError::Protocol(
                                    "tools must be an array of at most 64".to_string(),
                                ));
                            };
                            if items.len() > MAX_TOOLS {
                                return Err(DispatchError::Protocol(
                                    "tools must be an array of at most 64".to_string(),
                                ));
                            }
                            settings.tools.clear();
                            for tool in items {
                                settings.tools.push(parse_tool(tool)?);
                            }
                        }
                        "tool_choice" => {
                            if value.as_str() != Some("auto") {
                                return Err(DispatchError::Protocol(
                                    "tool_choice must be auto".to_string(),
                                ));
                            }
                        }
                        "max_output_tokens" => {
                            settings.max_output_tokens = parse_output_tokens(value)?;
                        }
                        "reasoning" => {
                            let reasoning_obj =
                                check_object_members(value, &["effort"], "reasoning")?;
                            let effort = member_str(reasoning_obj, "effort");
                            if !matches!(effort, "none" | "low" | "medium" | "high") {
                                return Err(DispatchError::Protocol(
                                    "reasoning.effort must be none, low, medium or high"
                                        .to_string(),
                                ));
                            }
                            settings.reasoning = effort != "none";
                        }
                        "audio" => {
                            let audio_obj =
                                check_object_members(value, &["input", "output"], "audio")?;
                            if let Some(input) = audio_obj.get("input") {
                                let input_obj = check_object_members(
                                    input,
                                    &[
                                        "format",
                                        "transcription",
                                        "turn_detection",
                                        "noise_reduction",
                                    ],
                                    "audio.input",
                                )?;
                                if let Some(format) = input_obj.get("format") {
                                    check_audio_format(format)?;
                                }
                                if let Some(opt) = input_obj.get("noise_reduction")
                                    && !is_null(opt)
                                {
                                    return Err(DispatchError::Protocol(
                                        "audio.input.noise_reduction must be null".to_string(),
                                    ));
                                }
                                if let Some(value) = input_obj.get("transcription") {
                                    transcription = parse_transcription(value, host)?;
                                    set_audio_member(&mut next, "input", "transcription", value);
                                }
                                if let Some(detection) = input_obj.get("turn_detection") {
                                    vad = parse_turn_detection(detection)?;
                                    vad_changed = true;
                                }
                            }
                            if let Some(output) = audio_obj.get("output") {
                                let output_obj = check_object_members(
                                    output,
                                    &["format", "voice"],
                                    "audio.output",
                                )?;
                                if let Some(format) = output_obj.get("format") {
                                    check_audio_format(format)?;
                                }
                                if let Some(voice) = output_obj.get("voice") {
                                    settings.voice = parse_voice(voice)?;
                                    set_audio_member(&mut next, "output", "voice", voice);
                                }
                            }
                            continue;
                        }
                        _ => {}
                    }
                    if let Value::Object(next_obj) = &mut next {
                        set_member(next_obj, name, value.clone());
                    }
                }
                if settings.reasoning && settings.audio_output {
                    return Err(DispatchError::Protocol(
                        "reasoning applies to text output_modalities".to_string(),
                    ));
                }
                if audio_open {
                    return Err(DispatchError::Protocol(
                        "commit or clear the audio buffer before session.update".to_string(),
                    ));
                }
                host.configure(&settings).map_err(DispatchError::Protocol)?;
                state.settings = settings;
                state.transcription = transcription;
                if vad_changed {
                    if let Value::Object(next_obj) = &mut next
                        && let Some(Value::Object(audio_obj)) = next_obj.get_mut("audio")
                        && let Some(Value::Object(input_obj)) = audio_obj.get_mut("input")
                    {
                        set_member(input_obj, "turn_detection", turn_detection_json(vad));
                    }
                    host.on_vad_settings(vad);
                }
                state.config = next;
                let ev = self
                    .shared
                    .simple_event("session.updated", Some(("session", state.config.clone())));
                if !host.send_event(&ev) {
                    return Err(DispatchError::Closed);
                }
                Ok(false)
            }
            "conversation.item.truncate" => {
                check_members(
                    obj,
                    &[
                        "type",
                        "event_id",
                        "item_id",
                        "content_index",
                        "audio_end_ms",
                    ],
                    "event",
                )?;
                let item_id = member_str(obj, "item_id");
                if item_id.is_empty() || item_id != state.last_reply_item {
                    return Err(DispatchError::Protocol(
                        "only the latest assistant message can be truncated".to_string(),
                    ));
                }
                let Some(end) = obj.get("audio_end_ms").and_then(as_f64) else {
                    return Err(DispatchError::Protocol(
                        "audio_end_ms must be a number".to_string(),
                    ));
                };
                if end < 0.0 {
                    return Err(DispatchError::Protocol(
                        "audio_end_ms must be a number".to_string(),
                    ));
                }
                let samples = (end * f64::from(SAMPLE_RATE) / 1000.0) as usize;
                host.truncate_reply(samples)
                    .map_err(DispatchError::Protocol)?;
                let mut event = Object::new();
                push_member(
                    &mut event,
                    "type",
                    Value::String("conversation.item.truncated".to_string()),
                );
                push_member(
                    &mut event,
                    "item_id",
                    Value::String(state.last_reply_item.clone()),
                );
                push_member(&mut event, "content_index", num_i64(0));
                push_member(&mut event, "audio_end_ms", num_i64(end as i64));
                if !host.send_event(&self.shared.format_event(event)) {
                    return Err(DispatchError::Closed);
                }
                Ok(false)
            }
            "conversation.item.delete" => {
                check_members(obj, &["type", "event_id", "item_id"], "event")?;
                let item_id = member_str(obj, "item_id");
                let Some((latest, before)) = state.deletable.clone() else {
                    return Err(DispatchError::Protocol(
                        "only the latest item, a user message, can be deleted".to_string(),
                    ));
                };
                if item_id.is_empty() || item_id != latest {
                    return Err(DispatchError::Protocol(
                        "only the latest item, a user message, can be deleted".to_string(),
                    ));
                }
                if audio_open {
                    return Err(DispatchError::Protocol(
                        "commit or clear the audio buffer before deleting an item".to_string(),
                    ));
                }
                host.delete_user_turn().map_err(DispatchError::Protocol)?;
                state.deletable = None;
                state.previous_item = before;
                let mut event = Object::new();
                push_member(
                    &mut event,
                    "type",
                    Value::String("conversation.item.deleted".to_string()),
                );
                push_member(&mut event, "item_id", Value::String(latest));
                if !host.send_event(&self.shared.format_event(event)) {
                    return Err(DispatchError::Closed);
                }
                Ok(false)
            }
            "input_audio_buffer.append" => {
                check_members(obj, &["type", "event_id", "audio"], "event")?;
                let pcm = decode_audio_pcm(obj.get("audio"))?;
                self.append_pcm(state, &pcm, audio_open, host)?;
                Ok(false)
            }
            "input_audio_buffer.clear" => {
                check_members(obj, &["type", "event_id"], "event")?;
                host.clear_audio().map_err(DispatchError::Protocol)?;
                state.turn_samples = 0;
                let ev = self.shared.simple_event("input_audio_buffer.cleared", None);
                if !host.send_event(&ev) {
                    return Err(DispatchError::Closed);
                }
                Ok(false)
            }
            "input_audio_buffer.commit" => {
                check_members(obj, &["type", "event_id"], "event")?;
                if state.turn_samples < MIN_COMMIT_SAMPLES {
                    return Err(DispatchError::Protocol(
                        "commit requires at least 100 ms of audio".to_string(),
                    ));
                }
                let item = self.shared.next_item_id();
                if !self.commit_turn(state, &item, host)? {
                    return Err(DispatchError::Protocol(
                        "the audio buffer is empty".to_string(),
                    ));
                }
                Ok(false)
            }
            "conversation.item.create" => {
                check_members(
                    obj,
                    &["type", "event_id", "previous_item_id", "item"],
                    "event",
                )?;
                let item_val = obj.get("item").ok_or_else(|| {
                    DispatchError::Protocol("conversation.item.create requires item".to_string())
                })?;
                if let Some(previous) = obj.get("previous_item_id")
                    && !is_null(previous)
                    && previous.as_str() != Some(state.previous_item.as_str())
                {
                    return Err(DispatchError::Protocol(
                        "items can only be appended".to_string(),
                    ));
                }
                if audio_open {
                    return Err(DispatchError::Protocol(
                        "commit or clear the audio buffer before adding an item".to_string(),
                    ));
                }
                let item_obj = item_val
                    .as_object()
                    .ok_or_else(|| DispatchError::Protocol("item must be an object".to_string()))?;
                let item_kind = member_str(item_obj, "type");
                let mut echo = item_obj.clone();
                if let Some(id) = item_obj.get("id")
                    && !id
                        .as_str()
                        .is_some_and(|id| !id.is_empty() && id.len() <= MAX_NAME_BYTES)
                {
                    return Err(DispatchError::Protocol(
                        "item.id must be a non-empty string of at most 64 bytes".to_string(),
                    ));
                }
                if item_obj.get("id").is_none() {
                    set_member(&mut echo, "id", Value::String(self.shared.next_item_id()));
                }
                set_member(
                    &mut echo,
                    "object",
                    Value::String("realtime.item".to_string()),
                );
                set_member(&mut echo, "status", Value::String("completed".to_string()));
                if item_kind == "function_call_output" {
                    check_members(item_obj, &["type", "id", "call_id", "output"], "item")?;
                    let call_id = member_str(item_obj, "call_id");
                    let Some(output) = item_obj.get("output").and_then(Value::as_str) else {
                        return Err(DispatchError::Protocol(
                            "function_call_output needs call_id and a string output".to_string(),
                        ));
                    };
                    if call_id.is_empty() {
                        return Err(DispatchError::Protocol(
                            "function_call_output needs call_id and a string output".to_string(),
                        ));
                    }
                    if state.outputs.len() >= MAX_PENDING_OUTPUTS {
                        return Err(DispatchError::Protocol(
                            "too many pending function_call_output items".to_string(),
                        ));
                    }
                    state.outputs.push(output.to_string());
                    self.emit_item_created(state, Value::Object(echo), host)?;
                    return Ok(false);
                }
                if item_kind != "message" || member_str(item_obj, "role") != "user" {
                    return Err(DispatchError::Protocol(
                        "items are user messages or function_call_output".to_string(),
                    ));
                }
                check_members(item_obj, &["type", "id", "role", "content"], "item")?;
                let Some(content) = item_obj.get("content").and_then(Value::as_array) else {
                    return Err(DispatchError::Protocol(
                        "message content must be a list".to_string(),
                    ));
                };
                if content.is_empty() {
                    return Err(DispatchError::Protocol(
                        "message content must be a list".to_string(),
                    ));
                }
                flush_outputs(state, host)?;
                let mut text = String::new();
                let mut spoken: Option<Vec<u8>> = None;
                for part in content {
                    let part_obj =
                        check_object_members(part, &["type", "text", "audio"], "content part")?;
                    match member_str(part_obj, "type") {
                        "input_text" => text.push_str(member_str(part_obj, "text")),
                        "input_audio" if spoken.is_none() => {
                            spoken = Some(decode_audio_pcm(part_obj.get("audio"))?);
                        }
                        _ => {
                            return Err(DispatchError::Protocol(
                                "content parts are input_text, or one input_audio".to_string(),
                            ));
                        }
                    }
                }
                if spoken.is_some() && !text.is_empty() {
                    return Err(DispatchError::Protocol(
                        "a message is text or speech, not both".to_string(),
                    ));
                }
                if let Some(pcm) = spoken {
                    let res = host
                        .append_audio(&pcm, false)
                        .and_then(|()| host.commit_audio());
                    match res {
                        Ok(true) => {}
                        Ok(false) => {
                            let _ = host.clear_audio();
                            return Err(DispatchError::Protocol("empty input_audio".to_string()));
                        }
                        Err(e) => {
                            let _ = host.clear_audio();
                            return Err(DispatchError::Protocol(if e.is_empty() {
                                "empty input_audio".to_string()
                            } else {
                                e
                            }));
                        }
                    }
                } else {
                    host.add_user_text(&text).map_err(DispatchError::Protocol)?;
                }
                self.emit_user_item(state, Value::Object(echo), host)?;
                Ok(false)
            }
            "response.create" => {
                check_members(obj, &["type", "event_id", "response"], "event")?;
                let mut settings = state.settings.clone();
                if let Some(response_val) = obj.get("response") {
                    let resp_obj = check_object_members(
                        response_val,
                        &["output_modalities", "max_output_tokens"],
                        "response",
                    )?;
                    if let Some(modalities) = resp_obj.get("output_modalities") {
                        settings.audio_output = parse_audio_modality(modalities)?;
                    }
                    if let Some(tokens) = resp_obj.get("max_output_tokens") {
                        settings.max_output_tokens = parse_output_tokens(tokens)?;
                    }
                    if settings.reasoning && settings.audio_output {
                        return Err(DispatchError::Protocol(
                            "reasoning applies to text output_modalities".to_string(),
                        ));
                    }
                }
                flush_outputs(state, host)?;
                self.begin_response(state, settings);
                Ok(true)
            }
            _ => Err(DispatchError::Protocol(format!(
                "unsupported realtime conversation event: {kind}"
            ))),
        }
    }

    fn append_pcm(
        &self,
        state: &mut State,
        pcm: &[u8],
        audio_open: bool,
        host: &mut dyn Host,
    ) -> Result<(), DispatchError> {
        let samples = pcm.len() / 2;
        if state.turn_samples.saturating_add(samples) > MAX_TURN_SAMPLES {
            return Err(DispatchError::Protocol(
                "a user turn holds at most 120 s of audio".to_string(),
            ));
        }
        if !audio_open {
            flush_outputs(state, host)?;
        }
        host.append_audio(pcm, state.transcription.is_some())
            .map_err(DispatchError::Protocol)?;
        state.turn_samples += samples;
        Ok(())
    }

    fn commit_turn(
        &self,
        state: &mut State,
        item: &str,
        host: &mut dyn Host,
    ) -> Result<bool, DispatchError> {
        if !host.commit_audio().map_err(DispatchError::Protocol)? {
            return Ok(false);
        }
        state.turn_samples = 0;
        let mut committed = Object::new();
        push_member(
            &mut committed,
            "type",
            Value::String("input_audio_buffer.committed".to_string()),
        );
        push_member(&mut committed, "item_id", Value::String(item.to_string()));
        push_member(
            &mut committed,
            "previous_item_id",
            previous_item_value(&state.previous_item),
        );
        if !host.send_event(&self.shared.format_event(committed)) {
            return Err(DispatchError::Closed);
        }
        let mut part = Object::new();
        push_member(&mut part, "type", Value::String("input_audio".to_string()));
        push_member(&mut part, "transcript", Value::Null);
        let mut created = Object::new();
        push_member(&mut created, "type", Value::String("message".to_string()));
        push_member(
            &mut created,
            "object",
            Value::String("realtime.item".to_string()),
        );
        push_member(&mut created, "role", Value::String("user".to_string()));
        push_member(
            &mut created,
            "status",
            Value::String("completed".to_string()),
        );
        push_member(
            &mut created,
            "content",
            Value::Array(vec![Value::Object(part)]),
        );
        push_member(&mut created, "id", Value::String(item.to_string()));
        self.emit_user_item(state, Value::Object(created), host)?;
        if let Some(transcription) = state.transcription.clone() {
            self.transcribe(item, &transcription, host)?;
        }
        Ok(true)
    }

    /// Sends the transcript of the committed turn `item` (one delta with the
    /// whole text, then `completed`), or `failed`.
    fn transcribe(
        &self,
        item: &str,
        transcription: &Transcription,
        host: &mut dyn Host,
    ) -> Result<(), DispatchError> {
        let result = host.transcribe_turn(transcription);
        let event = |kind: &str| {
            let mut event = Object::new();
            push_member(&mut event, "type", Value::String(kind.to_string()));
            push_member(&mut event, "item_id", Value::String(item.to_string()));
            push_member(&mut event, "content_index", num_i64(0));
            event
        };
        let mut events = Vec::new();
        match result {
            Ok(text) => {
                if !text.is_empty() {
                    let mut delta = event("conversation.item.input_audio_transcription.delta");
                    push_member(&mut delta, "delta", Value::String(text.clone()));
                    events.push(delta);
                }
                let mut done = event("conversation.item.input_audio_transcription.completed");
                push_member(&mut done, "transcript", Value::String(text));
                events.push(done);
            }
            Err(message) => {
                let mut error = Object::new();
                push_member(
                    &mut error,
                    "type",
                    Value::String("server_error".to_string()),
                );
                push_member(
                    &mut error,
                    "code",
                    Value::String("transcription_failed".to_string()),
                );
                push_member(&mut error, "message", Value::String(message));
                let mut failed = event("conversation.item.input_audio_transcription.failed");
                push_member(&mut failed, "error", Value::Object(error));
                events.push(failed);
            }
        }
        for event in events {
            if !host.send_event(&self.shared.format_event(event)) {
                return Err(DispatchError::Closed);
            }
        }
        Ok(())
    }

    /// Emits a user message item, which stays deletable until another item
    /// or a reply follows it.
    fn emit_user_item(
        &self,
        state: &mut State,
        item: Value,
        host: &mut dyn Host,
    ) -> Result<(), DispatchError> {
        let before = state.previous_item.clone();
        self.emit_item_created(state, item, host)?;
        state.deletable = Some((state.previous_item.clone(), before));
        Ok(())
    }

    fn emit_item_created(
        &self,
        state: &mut State,
        item: Value,
        host: &mut dyn Host,
    ) -> Result<(), DispatchError> {
        let mut event = Object::new();
        push_member(
            &mut event,
            "type",
            Value::String("conversation.item.created".to_string()),
        );
        push_member(
            &mut event,
            "previous_item_id",
            previous_item_value(&state.previous_item),
        );
        if let Some(id) = item.as_object().map(|o| member_str(o, "id")) {
            state.previous_item = id.to_string();
        }
        state.deletable = None;
        push_member(&mut event, "item", item);
        if !host.send_event(&self.shared.format_event(event)) {
            return Err(DispatchError::Closed);
        }
        Ok(())
    }

    fn begin_response(&self, state: &mut State, settings: Settings) {
        state.responses += 1;
        let response_id = format!("resp_{}_{}", self.shared.id, state.responses);
        state.last_reply_item.clear();
        let deletable = state.deletable.take();
        let mut response = Object::new();
        push_member(&mut response, "id", Value::String(response_id.clone()));
        push_member(
            &mut response,
            "object",
            Value::String("realtime.response".to_string()),
        );
        push_member(
            &mut response,
            "status",
            Value::String("in_progress".to_string()),
        );
        push_member(&mut response, "output", Value::Array(Vec::new()));
        push_member(
            &mut response,
            "output_modalities",
            Value::Array(vec![Value::String(
                if settings.audio_output {
                    "audio"
                } else {
                    "text"
                }
                .to_string(),
            )]),
        );
        state.active_response = Some(ActiveResponse {
            response_id,
            settings,
            response,
            item: String::new(),
            transcript: String::new(),
            reasoning: String::new(),
            scratch: String::new(),
            deletable,
        });
    }
}

enum DispatchError {
    Protocol(String),
    Closed,
}

impl From<String> for DispatchError {
    fn from(msg: String) -> Self {
        Self::Protocol(msg)
    }
}

fn flush_outputs(state: &mut State, host: &mut dyn Host) -> Result<(), DispatchError> {
    if state.outputs.is_empty() {
        return Ok(());
    }
    let outputs = std::mem::take(&mut state.outputs);
    host.add_tool_outputs(&outputs)
        .map_err(DispatchError::Protocol)
}

fn open_response_item(
    shared: &Shared,
    active: &mut ActiveResponse,
    send: &mut dyn FnMut(&str) -> bool,
) -> Result<(), ()> {
    if !active.item.is_empty() {
        return Ok(());
    }
    let item_id = shared.next_item_id();
    active.item = item_id.clone();

    let mut item = Object::new();
    push_member(
        &mut item,
        "object",
        Value::String("realtime.item".to_string()),
    );
    push_member(&mut item, "type", Value::String("message".to_string()));
    push_member(&mut item, "role", Value::String("assistant".to_string()));
    push_member(
        &mut item,
        "status",
        Value::String("in_progress".to_string()),
    );
    push_member(&mut item, "content", Value::Array(Vec::new()));
    push_member(&mut item, "id", Value::String(item_id.clone()));

    let mut added = Object::new();
    push_member(
        &mut added,
        "type",
        Value::String("response.output_item.added".to_string()),
    );
    push_member(
        &mut added,
        "response_id",
        Value::String(active.response_id.clone()),
    );
    push_member(&mut added, "output_index", num_i64(0));
    push_member(&mut added, "item", Value::Object(item));
    if !send(&shared.format_event(added)) {
        return Err(());
    }

    let mut part_obj = Object::new();
    push_member(
        &mut part_obj,
        "type",
        Value::String(
            if active.settings.audio_output {
                "audio"
            } else {
                "text"
            }
            .to_string(),
        ),
    );
    push_member(
        &mut part_obj,
        if active.settings.audio_output {
            "transcript"
        } else {
            "text"
        },
        Value::String(String::new()),
    );
    let mut part = Object::new();
    push_member(
        &mut part,
        "type",
        Value::String("response.content_part.added".to_string()),
    );
    push_member(
        &mut part,
        "response_id",
        Value::String(active.response_id.clone()),
    );
    push_member(&mut part, "item_id", Value::String(item_id));
    push_member(&mut part, "output_index", num_i64(0));
    push_member(&mut part, "content_index", num_i64(0));
    push_member(&mut part, "part", Value::Object(part_obj));
    if !send(&shared.format_event(part)) {
        return Err(());
    }
    Ok(())
}

fn push_event_id(buf: &mut String, id: &str, seq: usize) {
    use std::fmt::Write;
    buf.push_str(r#""event_"#);
    push_json_escaped(buf, id);
    let _ = write!(buf, "_{seq}\"");
}

fn send_delta(
    shared: &Shared,
    active: &mut ActiveResponse,
    kind: &str,
    delta: &str,
    send: &mut dyn FnMut(&str) -> bool,
) -> Result<(), ()> {
    let seq = shared.sequence.fetch_add(1, Ordering::Relaxed) + 1;
    let buf = &mut active.scratch;
    buf.clear();
    buf.push_str(r#"{"type":""#);
    buf.push_str(kind);
    buf.push_str(r#"","response_id":"#);
    push_json_str(buf, &active.response_id);
    buf.push_str(r#","item_id":"#);
    push_json_str(buf, &active.item);
    buf.push_str(r#","output_index":0,"content_index":0,"delta":"#);
    push_json_str(buf, delta);
    buf.push_str(r#","event_id":"#);
    push_event_id(buf, &shared.id, seq);
    buf.push('}');
    if send(buf) { Ok(()) } else { Err(()) }
}

fn send_audio_delta(
    shared: &Shared,
    active: &mut ActiveResponse,
    pcm: &[u8],
    send: &mut dyn FnMut(&str) -> bool,
) -> Result<(), ()> {
    let seq = shared.sequence.fetch_add(1, Ordering::Relaxed) + 1;
    let encoded_len = gufo_base64::encoded_len(pcm.len()).ok_or(())?;
    let buf = &mut active.scratch;
    buf.clear();
    buf.try_reserve(
        encoded_len + active.response_id.len() + active.item.len() + shared.id.len() + 128,
    )
    .map_err(|_| ())?;
    buf.push_str(r#"{"type":"response.output_audio.delta","response_id":"#);
    push_json_str(buf, &active.response_id);
    buf.push_str(r#","item_id":"#);
    push_json_str(buf, &active.item);
    buf.push_str(r#","output_index":0,"content_index":0,"delta":""#);
    // Encode base64 directly into the JSON buffer without intermediate copies.
    // Base64 output is pure ASCII (`A-Za-z0-9+/=`), requiring no JSON escaping.
    let mut raw = std::mem::take(buf).into_bytes();
    let start = raw.len();
    raw.resize(start + encoded_len, 0);
    gufo_base64::encode(pcm, &mut raw[start..]);
    *buf = String::from_utf8(raw).expect("JSON prefix and base64 are valid UTF-8");
    buf.push_str(r#"","event_id":"#);
    push_event_id(buf, &shared.id, seq);
    buf.push('}');
    if send(buf) { Ok(()) } else { Err(()) }
}

fn push_json_str(out: &mut String, text: &str) {
    out.push('"');
    push_json_escaped(out, text);
    out.push('"');
}

fn push_json_escaped(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r#"\\"#),
            '\x08' => out.push_str(r#"\b"#),
            '\x0c' => out.push_str(r#"\f"#),
            '\n' => out.push_str(r#"\n"#),
            '\r' => out.push_str(r#"\r"#),
            '\t' => out.push_str(r#"\t"#),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

fn previous_item_value(previous: &str) -> Value {
    if previous.is_empty() {
        Value::Null
    } else {
        Value::String(previous.to_string())
    }
}

fn dump_json(value: &Value) -> String {
    dump(value, Numbers::AsParsed).expect("JSON serialization out of memory")
}

fn format_json_error(err: ValueError) -> String {
    match err {
        ValueError::Rejected(reason) => format!("JSON parse error: {reason}"),
        ValueError::TooLarge => "JSON input too large".to_string(),
        ValueError::NoMemory => "out of memory".to_string(),
        ValueError::Internal => "internal JSON parser error".to_string(),
    }
}

fn is_null(value: &Value) -> bool {
    matches!(value, Value::Null)
}

fn as_f64(value: &Value) -> Option<f64> {
    value.as_number().map(Number::value)
}

fn member_str<'a>(obj: &'a Object, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

fn push_member(obj: &mut Object, key: &str, value: Value) {
    let _ = obj.push(key.to_string(), value);
}

fn set_member(obj: &mut Object, key: &str, value: Value) {
    if let Some(slot) = obj.get_mut(key) {
        *slot = value;
    } else {
        let _ = obj.push(key.to_string(), value);
    }
}

fn num_i64(v: i64) -> Value {
    Value::Number(Number::new(v as f64).expect("i64 is finite"))
}

fn num_f64(v: f64) -> Value {
    Value::Number(Number::new(v).expect("number is finite"))
}

fn check_members(obj: &Object, allowed: &[&str], what: &str) -> Result<(), String> {
    for (name, _) in obj.members() {
        if !allowed.contains(&name.as_str()) {
            return Err(format!("unsupported {what} field: {name}"));
        }
    }
    Ok(())
}

fn check_object_members<'a>(
    value: &'a Value,
    allowed: &[&str],
    what: &str,
) -> Result<&'a Object, String> {
    let Some(obj) = value.as_object() else {
        return Err(format!("{what} must be an object"));
    };
    check_members(obj, allowed, what)?;
    Ok(obj)
}

fn decode_audio_pcm(audio: Option<&Value>) -> Result<Vec<u8>, String> {
    let err = || "audio must be base64 PCM16 (mono, 24 kHz)".to_string();
    let Some(text) = audio.and_then(Value::as_str) else {
        return Err(err());
    };
    let bytes = text.as_bytes();
    if !bytes.len().is_multiple_of(4) || bytes.len() / 4 > MAX_TURN_BYTES.div_ceil(3) {
        return Err(err());
    }
    let cap = gufo_base64::decoded_capacity(bytes.len());
    let mut out = Vec::new();
    out.try_reserve_exact(cap).map_err(|_| err())?;
    out.resize(cap, 0);
    let written = gufo_base64::decode(bytes, true, MAX_TURN_BYTES, &mut out).map_err(|_| err())?;
    out.truncate(written);
    if !out.len().is_multiple_of(2) {
        return Err(err());
    }
    Ok(out)
}

#[cfg(test)]
fn encode_base64(bytes: &[u8]) -> String {
    let Some(len) = gufo_base64::encoded_len(bytes.len()) else {
        return String::new();
    };
    let mut out = Vec::new();
    if out.try_reserve_exact(len).is_err() {
        return String::new();
    }
    out.resize(len, 0);
    gufo_base64::encode(bytes, &mut out);
    String::from_utf8(out).expect("base64 is valid UTF-8")
}

fn parse_output_tokens(value: &Value) -> Result<usize, String> {
    if value.as_str() == Some("inf") {
        return Ok(MAX_OUTPUT_TOKENS);
    }
    if let Some(n) = as_f64(value)
        && (1.0..=MAX_OUTPUT_TOKENS as f64).contains(&n)
        && n == n.floor()
    {
        return Ok(n as usize);
    }
    Err("max_output_tokens must be an integer in [1, 4096] or \"inf\"".to_string())
}

/// `["audio"]` and `["audio", "text"]` (audio replies stream their
/// transcript) speak; `["text"]` writes.
fn parse_audio_modality(value: &Value) -> Result<bool, String> {
    let modes: Option<Vec<&str>> = value
        .as_array()
        .and_then(|items| items.iter().map(Value::as_str).collect());
    match modes.as_deref() {
        Some(["audio"] | ["audio", "text"] | ["text", "audio"]) => Ok(true),
        Some(["text"]) => Ok(false),
        _ => Err(
            "output_modalities must be [\"audio\"], [\"text\"] or [\"audio\", \"text\"]"
                .to_string(),
        ),
    }
}

fn parse_transcription(
    value: &Value,
    host: &mut dyn Host,
) -> Result<Option<Transcription>, String> {
    if is_null(value) {
        return Ok(None);
    }
    let obj = check_object_members(value, &["model", "language", "prompt"], "transcription")?;
    let mut transcription = Transcription::default();
    for (name, field) in obj.members() {
        if is_null(field) {
            continue;
        }
        let Some(text) = field.as_str() else {
            return Err(format!("transcription.{name} must be a string"));
        };
        match name.as_str() {
            "model" => {
                if !host.serves_model(text) {
                    return Err(format!(
                        "transcription.model must name the served model, not {text}"
                    ));
                }
            }
            "language" => {
                if text.len() > MAX_NAME_BYTES {
                    return Err("transcription.language holds at most 64 bytes".to_string());
                }
                transcription.language = text.to_string();
            }
            _ => {
                if text.len() > MAX_TRANSCRIPTION_PROMPT_BYTES {
                    return Err("transcription.prompt holds at most 16 KiB".to_string());
                }
                transcription.prompt = text.to_string();
            }
        }
    }
    Ok(Some(transcription))
}

fn parse_voice(value: &Value) -> Result<String, String> {
    if is_null(value) {
        return Ok(String::new());
    }
    match value.as_str() {
        Some(voice) if voice.len() <= MAX_NAME_BYTES => Ok(voice.to_string()),
        _ => Err("audio.output.voice must be a string of at most 64 bytes".to_string()),
    }
}

/// Sets `audio.<direction>.<key>` of a session configuration.
fn set_audio_member(config: &mut Value, direction: &str, key: &str, value: &Value) {
    if let Value::Object(config) = config
        && let Some(Value::Object(audio)) = config.get_mut("audio")
        && let Some(Value::Object(side)) = audio.get_mut(direction)
    {
        set_member(side, key, value.clone());
    }
}

fn parse_tool(tool: &Value) -> Result<String, String> {
    let obj = check_object_members(tool, &["type", "name", "description", "parameters"], "tool")?;
    if member_str(obj, "type") != "function" || member_str(obj, "name").is_empty() {
        return Err("tools must be named functions".to_string());
    }
    let mut function = Object::new();
    for (key, value) in obj.members() {
        if key != "type" {
            push_member(&mut function, key, value.clone());
        }
    }
    Ok(dump_json(&Value::Object(function)))
}

fn check_audio_format(format: &Value) -> Result<(), String> {
    let obj = check_object_members(format, &["type", "rate"], "audio format")?;
    let bad_rate = obj
        .get("rate")
        .is_some_and(|r| as_f64(r) != Some(f64::from(SAMPLE_RATE)));
    if member_str(obj, "type") != "audio/pcm" || bad_rate {
        return Err("audio is mono PCM16 at 24000 Hz".to_string());
    }
    Ok(())
}

fn parse_turn_detection(value: &Value) -> Result<Option<VadSettings>, String> {
    if is_null(value) {
        return Ok(None);
    }
    let Some(obj) = value.as_object() else {
        return Err("turn_detection must be null or an object".to_string());
    };
    let mut settings = VadSettings::default();
    for (name, field) in obj.members() {
        match name.as_str() {
            "type" => {
                if field.as_str() != Some("server_vad") {
                    return Err(
                        "turn_detection.type must be server_vad (semantic_vad is not supported)"
                            .to_string(),
                    );
                }
            }
            "threshold" => {
                let Some(t) = as_f64(field) else {
                    return Err("turn_detection.threshold is in [0, 1]".to_string());
                };
                if !(0.0..=1.0).contains(&t) {
                    return Err("turn_detection.threshold is in [0, 1]".to_string());
                }
                settings.threshold = t;
            }
            "prefix_padding_ms" | "silence_duration_ms" => {
                let Some(v) = as_f64(field) else {
                    return Err(format!("turn_detection.{name} is an integer in [0, 10000]"));
                };
                if !(0.0..=10_000.0).contains(&v) || v != v.floor() {
                    return Err(format!("turn_detection.{name} is an integer in [0, 10000]"));
                }
                if name == "prefix_padding_ms" {
                    settings.prefix_padding_ms = v as i32;
                } else {
                    settings.silence_duration_ms = v as i32;
                }
            }
            "create_response" | "interrupt_response" => {
                let Some(b) = field.as_bool() else {
                    return Err(format!("turn_detection.{name} must be a boolean"));
                };
                if name == "create_response" {
                    settings.create_response = b;
                } else {
                    settings.interrupt_response = b;
                }
            }
            "idle_timeout_ms" if is_null(field) => {}
            _ => return Err(format!("unsupported turn_detection field: {name}")),
        }
    }
    if obj.get("type").is_none() {
        return Err("turn_detection needs type server_vad".to_string());
    }
    Ok(Some(settings))
}

fn turn_detection_json(vad: Option<VadSettings>) -> Value {
    let Some(vad) = vad else {
        return Value::Null;
    };
    let mut obj = Object::new();
    push_member(&mut obj, "type", Value::String("server_vad".to_string()));
    push_member(&mut obj, "threshold", num_f64(vad.threshold));
    push_member(
        &mut obj,
        "prefix_padding_ms",
        num_i64(i64::from(vad.prefix_padding_ms)),
    );
    push_member(
        &mut obj,
        "silence_duration_ms",
        num_i64(i64::from(vad.silence_duration_ms)),
    );
    push_member(
        &mut obj,
        "create_response",
        Value::Bool(vad.create_response),
    );
    push_member(
        &mut obj,
        "interrupt_response",
        Value::Bool(vad.interrupt_response),
    );
    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct FakeHost {
        events: Vec<String>,
        configured: Vec<Settings>,
        vad_updates: Vec<Option<VadSettings>>,
        audio_bytes: usize,
        committed_calls: usize,
        cleared_calls: usize,
        truncated_samples: Option<usize>,
        user_texts: Vec<String>,
        tool_outputs: Vec<Vec<String>>,
        transcripts: Vec<Transcription>,
        transcript: Option<String>,
        deleted_turns: usize,
        delete_fails: bool,
        recorded_bytes: usize,
    }

    impl Host for FakeHost {
        fn send_event(&mut self, json: &str) -> bool {
            self.events.push(json.to_string());
            true
        }

        fn configure(&mut self, settings: &Settings) -> Result<(), String> {
            self.configured.push(settings.clone());
            Ok(())
        }

        fn on_vad_settings(&mut self, vad: Option<VadSettings>) {
            self.vad_updates.push(vad);
        }

        fn append_audio(&mut self, pcm: &[u8], record: bool) -> Result<(), String> {
            self.audio_bytes += pcm.len();
            if record {
                self.recorded_bytes += pcm.len();
            }
            Ok(())
        }

        fn commit_audio(&mut self) -> Result<bool, String> {
            self.committed_calls += 1;
            let had = self.audio_bytes > 0;
            self.audio_bytes = 0;
            Ok(had)
        }

        fn clear_audio(&mut self) -> Result<(), String> {
            self.cleared_calls += 1;
            self.audio_bytes = 0;
            Ok(())
        }

        fn truncate_reply(&mut self, samples: usize) -> Result<(), String> {
            self.truncated_samples = Some(samples);
            Ok(())
        }

        fn add_user_text(&mut self, text: &str) -> Result<(), String> {
            self.user_texts.push(text.to_string());
            Ok(())
        }

        fn add_tool_outputs(&mut self, outputs: &[String]) -> Result<(), String> {
            self.tool_outputs.push(outputs.to_vec());
            Ok(())
        }

        fn serves_model(&mut self, name: &str) -> bool {
            name == "voice"
        }

        fn transcribe_turn(&mut self, transcription: &Transcription) -> Result<String, String> {
            self.transcripts.push(transcription.clone());
            self.transcript
                .clone()
                .ok_or_else(|| "decoder failed".to_string())
        }

        fn delete_user_turn(&mut self) -> Result<(), String> {
            if self.delete_fails {
                return Err("cannot delete".to_string());
            }
            self.deleted_turns += 1;
            Ok(())
        }
    }

    fn event_types(host: &FakeHost) -> Vec<String> {
        host.events.iter().map(|e| event_type(e)).collect()
    }

    fn append_event(bytes: usize) -> String {
        format!(
            r#"{{"type":"input_audio_buffer.append","audio":"{}"}}"#,
            encode_base64(&vec![1u8; bytes])
        )
    }

    // Open WebUI 0.13 voice calls: input transcription, a voice, server turn
    // detection that waits for response.create, and deletion of rejected
    // turns.
    #[test]
    fn open_webui_call_transcribes_turns() {
        let session = Session::new("ow".to_string(), "voice".to_string());
        let mut host = FakeHost {
            transcript: Some("Hello there.".to_string()),
            ..FakeHost::default()
        };
        let update = br#"{"type":"session.update","session":{"type":"realtime","output_modalities":["audio"],"instructions":"Be brief.","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"voice","language":"en"},"turn_detection":{"type":"server_vad","interrupt_response":false,"create_response":false}},"output":{"format":{"type":"audio/pcm","rate":24000},"voice":"female"}},"tools":[{"type":"function","name":"generate_chat_completion","parameters":{"type":"object"}}],"tool_choice":"auto"}}"#;
        assert_eq!(
            session.handle_client_event(update, false, &mut host),
            Ok(false)
        );
        assert_eq!(event_types(&host), vec!["session.updated"]);
        let updated = parse_value(host.events[0].as_bytes(), Limits::NONE).unwrap();
        let config = updated.as_object().unwrap().get("session").unwrap();
        let audio = config.as_object().unwrap().get("audio").unwrap();
        let audio = audio.as_object().unwrap();
        let input = audio.get("input").unwrap().as_object().unwrap();
        let transcription = input.get("transcription").unwrap().as_object().unwrap();
        assert_eq!(member_str(transcription, "model"), "voice");
        let output = audio.get("output").unwrap().as_object().unwrap();
        assert_eq!(member_str(output, "voice"), "female");
        assert_eq!(host.configured.last().unwrap().voice, "female");
        let vad = host.vad_updates.last().unwrap().unwrap();
        assert!(!vad.create_response && !vad.interrupt_response);

        // A server turn without create_response transcribes and waits.
        host.events.clear();
        let (item, _) = session.vad_speech_started(0);
        assert_eq!(
            session.handle_vad_audio(&[1u8; 4800], false, &mut host),
            Ok(())
        );
        assert_eq!(
            session.handle_vad_commit(&item, vad.create_response, false, &mut host),
            Ok(false)
        );
        assert_eq!(
            event_types(&host),
            vec![
                "input_audio_buffer.committed",
                "conversation.item.created",
                "conversation.item.input_audio_transcription.delta",
                "conversation.item.input_audio_transcription.completed",
            ]
        );
        assert!(host.events[3].contains(&format!(r#""item_id":"{item}""#)));
        assert!(host.events[3].contains(r#""transcript":"Hello there.""#));
        assert_eq!(
            host.transcripts,
            vec![Transcription {
                language: "en".to_string(),
                prompt: String::new(),
            }]
        );

        // The client answers transcribed turns itself.
        host.events.clear();
        assert_eq!(
            session.handle_client_event(br#"{"type":"response.create"}"#, false, &mut host),
            Ok(true)
        );
        let mut send = |_: &str| true;
        let (settings, _) = session.response_start(&mut send).unwrap();
        assert!(settings.audio_output && settings.voice == "female");
        let outcome = ResponseOutcome {
            ok: true,
            cancelled: false,
            incomplete: false,
            input_tokens: 1,
            output_tokens: 1,
            calls: Vec::new(),
            failure: String::new(),
        };
        assert!(session.response_finish(&outcome, &mut send).is_ok());

        // A turn without a transcript is deleted by the client.
        host.transcript = None;
        host.events.clear();
        let reply = session.state.lock().unwrap().previous_item.clone();
        assert_eq!(
            session.handle_client_event(append_event(4800).as_bytes(), false, &mut host),
            Ok(false)
        );
        assert_eq!(
            session.handle_client_event(
                br#"{"type":"input_audio_buffer.commit"}"#,
                true,
                &mut host
            ),
            Ok(false)
        );
        let types = event_types(&host);
        assert_eq!(
            types.last().unwrap(),
            "conversation.item.input_audio_transcription.failed"
        );
        assert!(host.events.last().unwrap().contains("decoder failed"));
        let turn = parse_value(host.events[0].as_bytes(), Limits::NONE).unwrap();
        let turn = member_str(turn.as_object().unwrap(), "item_id").to_string();
        host.events.clear();
        let delete = format!(r#"{{"type":"conversation.item.delete","item_id":"{turn}"}}"#);
        assert_eq!(
            session.handle_client_event(delete.as_bytes(), false, &mut host),
            Ok(false)
        );
        assert_eq!(event_types(&host), vec!["conversation.item.deleted"]);
        assert_eq!(host.deleted_turns, 1);
        assert_eq!(session.state.lock().unwrap().previous_item, reply);

        // Only the latest user message can go: not twice, not a reply.
        host.events.clear();
        assert_eq!(
            session.handle_client_event(delete.as_bytes(), false, &mut host),
            Ok(false)
        );
        let reply_delete = format!(r#"{{"type":"conversation.item.delete","item_id":"{reply}"}}"#);
        assert_eq!(
            session.handle_client_event(reply_delete.as_bytes(), false, &mut host),
            Ok(false)
        );
        assert_eq!(event_types(&host), vec!["error", "error"]);
        assert_eq!(host.deleted_turns, 1);
        assert_eq!(host.recorded_bytes, 9600);
    }

    fn finish(session: &Session, ok: bool) {
        let mut send = |_: &str| true;
        session.response_start(&mut send).unwrap();
        let outcome = ResponseOutcome {
            ok,
            cancelled: false,
            incomplete: false,
            input_tokens: 1,
            output_tokens: 1,
            calls: Vec::new(),
            failure: if ok {
                String::new()
            } else {
                "failed".to_string()
            },
        };
        session.response_finish(&outcome, &mut send).unwrap();
    }

    fn commit_spoken_turn(session: &Session, host: &mut FakeHost) -> String {
        host.events.clear();
        session
            .handle_client_event(append_event(4800).as_bytes(), false, host)
            .unwrap();
        session
            .handle_client_event(br#"{"type":"input_audio_buffer.commit"}"#, true, host)
            .unwrap();
        let committed = parse_value(host.events[0].as_bytes(), Limits::NONE).unwrap();
        member_str(committed.as_object().unwrap(), "item_id").to_string()
    }

    fn delete(session: &Session, host: &mut FakeHost, item: &str, audio_open: bool) -> String {
        host.events.clear();
        let event = format!(r#"{{"type":"conversation.item.delete","item_id":"{item}"}}"#);
        session
            .handle_client_event(event.as_bytes(), audio_open, host)
            .unwrap();
        event_type(host.events.last().unwrap())
    }

    #[test]
    fn transcripts_precede_replies_and_deletion_keeps_state() {
        let session = Session::new("d".to_string(), "voice".to_string());
        let mut host = FakeHost {
            transcript: Some("Hi.".to_string()),
            ..FakeHost::default()
        };
        session
            .handle_client_event(
                br#"{"type":"session.update","session":{"output_modalities":["text","audio"],"audio":{"input":{"transcription":{"model":null}},"output":{"voice":"female"}}}}"#,
                false,
                &mut host,
            )
            .unwrap();
        assert_eq!(event_types(&host), vec!["session.updated"]);
        assert!(host.configured.last().unwrap().audio_output);
        session
            .handle_client_event(
                br#"{"type":"session.update","session":{"audio":{"output":{"voice":null}}}}"#,
                false,
                &mut host,
            )
            .unwrap();
        assert_eq!(host.configured.last().unwrap().voice, "");

        // With create_response the transcript still comes before the reply.
        host.events.clear();
        let (item, _) = session.vad_speech_started(0);
        session
            .handle_vad_audio(&[1u8; 4800], false, &mut host)
            .unwrap();
        assert_eq!(
            session.handle_vad_commit(&item, true, false, &mut host),
            Ok(true)
        );
        assert_eq!(
            event_types(&host).last().unwrap(),
            "conversation.item.input_audio_transcription.completed"
        );
        // A reply that adds no item leaves the turn deletable.
        finish(&session, false);
        assert_eq!(
            delete(&session, &mut host, &item, false),
            "conversation.item.deleted"
        );

        // Open audio and a failing conversation keep the turn and the state.
        let turn = commit_spoken_turn(&session, &mut host);
        assert_eq!(delete(&session, &mut host, &turn, true), "error");
        host.delete_fails = true;
        assert_eq!(delete(&session, &mut host, &turn, false), "error");
        assert_eq!(session.state.lock().unwrap().previous_item, turn);
        host.delete_fails = false;
        assert_eq!(delete(&session, &mut host, "", false), "error");
        assert_eq!(
            delete(&session, &mut host, &turn, false),
            "conversation.item.deleted"
        );
        assert_eq!(host.deleted_turns, 2);

        // Text messages can be deleted; client item ids are checked.
        host.events.clear();
        session
            .handle_client_event(
                br#"{"type":"conversation.item.create","item":{"type":"message","role":"user","id":"","content":[{"type":"input_text","text":"x"}]}}"#,
                false,
                &mut host,
            )
            .unwrap();
        assert_eq!(event_types(&host), vec!["error"]);
        session
            .handle_client_event(
                br#"{"type":"conversation.item.create","item":{"type":"message","role":"user","id":"mine","content":[{"type":"input_text","text":"x"}]}}"#,
                false,
                &mut host,
            )
            .unwrap();
        assert_eq!(
            delete(&session, &mut host, "mine", false),
            "conversation.item.deleted"
        );
    }

    #[test]
    fn session_update_options_validate() {
        let session = Session::new("v".to_string(), "voice".to_string());
        let mut host = FakeHost::default();
        let mut errors = Vec::new();
        for update in [
            r#"{"type":"session.update","session":{"audio":{"input":{"transcription":{"model":"whisper-1"}}}}}"#,
            r#"{"type":"session.update","session":{"audio":{"input":{"transcription":{"language":7}}}}}"#,
            r#"{"type":"session.update","session":{"audio":{"input":{"transcription":{"model":"voice","delay":1}}}}}"#,
            r#"{"type":"session.update","session":{"audio":{"output":{"voice":7}}}}"#,
            r#"{"type":"session.update","session":{"output_modalities":["audio","audio"]}}"#,
            r#"{"type":"session.update","session":{"output_modalities":[]}}"#,
        ] {
            host.events.clear();
            assert_eq!(
                session.handle_client_event(update.as_bytes(), false, &mut host),
                Ok(false)
            );
            assert_eq!(event_types(&host), vec!["error"], "{update}");
            errors.push(host.events[0].clone());
        }
        assert!(errors[0].contains("transcription.model must name the served model"));
        assert!(host.configured.is_empty());
        assert!(session.state.lock().unwrap().transcription.is_none());

        // Both modalities speak; transcription can be switched off again.
        for (update, audio) in [
            (
                r#"{"type":"session.update","session":{"output_modalities":["audio","text"],"audio":{"input":{"transcription":{"model":"voice","prompt":"Names: Gufo."}}}}}"#,
                true,
            ),
            (
                r#"{"type":"session.update","session":{"output_modalities":["text"],"audio":{"input":{"transcription":null}}}}"#,
                false,
            ),
        ] {
            host.events.clear();
            assert_eq!(
                session.handle_client_event(update.as_bytes(), false, &mut host),
                Ok(false)
            );
            assert_eq!(event_types(&host), vec!["session.updated"], "{update}");
            assert_eq!(host.configured.last().unwrap().audio_output, audio);
        }
        assert!(session.state.lock().unwrap().transcription.is_none());

        // Without transcription a commit sends no transcript events.
        host.events.clear();
        session
            .handle_client_event(append_event(4800).as_bytes(), false, &mut host)
            .unwrap();
        session
            .handle_client_event(br#"{"type":"input_audio_buffer.commit"}"#, true, &mut host)
            .unwrap();
        assert_eq!(
            event_types(&host),
            vec!["input_audio_buffer.committed", "conversation.item.created"]
        );
        assert!(host.transcripts.is_empty());
        assert_eq!(host.recorded_bytes, 0);
    }

    fn event_type(raw: &str) -> String {
        let v = parse_value(raw.as_bytes(), Limits::NONE).unwrap();
        v.as_object()
            .and_then(|o| o.get("type"))
            .and_then(Value::as_str)
            .unwrap()
            .to_string()
    }

    #[test]
    fn recorded_conversation_flow_matches_protocol() {
        let session = Session::new("req1".to_string(), "voice".to_string());
        let mut host = FakeHost::default();

        let created = session.session_created();
        assert_eq!(event_type(&created), "session.created");
        assert!(created.contains(r#""id":"sess_req1""#));
        assert!(created.contains(r#""model":"voice""#));

        let pcm = vec![1u8; 4800];
        let append = format!(
            r#"{{"type":"input_audio_buffer.append","audio":"{}"}}"#,
            encode_base64(&pcm)
        );
        assert_eq!(
            session.handle_client_event(append.as_bytes(), false, &mut host),
            Ok(false)
        );
        assert_eq!(host.audio_bytes, 4800);

        assert_eq!(
            session.handle_client_event(
                br#"{"type":"input_audio_buffer.commit"}"#,
                true,
                &mut host
            ),
            Ok(false)
        );
        assert_eq!(
            host.events
                .iter()
                .map(|e| event_type(e))
                .collect::<Vec<_>>(),
            vec!["input_audio_buffer.committed", "conversation.item.created"]
        );

        host.events.clear();
        assert_eq!(
            session.handle_client_event(br#"{"type":"response.create"}"#, false, &mut host),
            Ok(true)
        );
        let mut send = |s: &str| {
            host.events.push(s.to_string());
            true
        };
        let (settings, resp_id) = session.response_start(&mut send).unwrap();
        assert!(settings.audio_output);
        assert_eq!(resp_id, "resp_req1_1");
        assert!(session.response_text("Hi", &mut send).is_ok());
        assert!(session.response_audio(&[0, 1, 2, 3], &mut send).is_ok());
        let outcome = ResponseOutcome {
            ok: true,
            cancelled: false,
            incomplete: false,
            input_tokens: 10,
            output_tokens: 5,
            calls: Vec::new(),
            failure: String::new(),
        };
        assert!(session.response_finish(&outcome, &mut send).is_ok());
        assert_eq!(
            host.events
                .iter()
                .map(|e| event_type(e))
                .collect::<Vec<_>>(),
            vec![
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_audio_transcript.delta",
                "response.output_audio.delta",
                "response.output_audio.done",
                "response.output_audio_transcript.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.done",
            ]
        );
    }

    #[test]
    fn cancel_and_validation_errors() {
        let session = Session::new("r2".to_string(), "voice".to_string());
        match session.classify_client_event(br#"{"type":"response.cancel"}"#, false, false) {
            ClientDecision::Error(e) => {
                assert!(e.contains(r#""code":"response_cancel_not_active""#));
            }
            other => panic!("expected error, got {other:?}"),
        }
        assert_eq!(
            session.classify_client_event(br#"{"type":"response.cancel"}"#, false, true),
            ClientDecision::Cancel
        );
        match session.classify_client_event(br#"{"type":"_vad.commit"}"#, false, false) {
            ClientDecision::Error(e) => {
                assert!(e.contains("unsupported realtime conversation event: _vad.commit"));
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn tool_call_and_truncation() {
        let session = Session::new("r3".to_string(), "voice".to_string());
        let mut host = FakeHost::default();
        assert_eq!(
            session.handle_client_event(
                br#"{"type":"session.update","session":{"output_modalities":["text"],"tools":[{"type":"function","name":"set_volume","parameters":{"type":"object"}}]}}"#,
                false,
                &mut host
            ),
            Ok(false)
        );
        assert_eq!(host.configured.len(), 1);
        assert!(!host.configured[0].audio_output);
        assert_eq!(
            host.configured[0].tools,
            vec![r#"{"name":"set_volume","parameters":{"type":"object"}}"#]
        );

        host.events.clear();
        assert_eq!(
            session.handle_client_event(br#"{"type":"response.create"}"#, false, &mut host),
            Ok(true)
        );
        let mut send = |s: &str| {
            host.events.push(s.to_string());
            true
        };
        assert!(session.response_start(&mut send).is_ok());
        let outcome = ResponseOutcome {
            ok: true,
            cancelled: false,
            incomplete: false,
            input_tokens: 4,
            output_tokens: 2,
            calls: vec![FunctionCall {
                name: "set_volume".to_string(),
                arguments: r#"{"volume":30}"#.to_string(),
            }],
            failure: String::new(),
        };
        assert!(session.response_finish(&outcome, &mut send).is_ok());
        assert_eq!(
            host.events
                .iter()
                .map(|e| event_type(e))
                .collect::<Vec<_>>(),
            vec![
                "response.created",
                "response.output_item.added",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.done",
            ]
        );

        // Submit function_call_output and verify it flushes on response.create.
        host.events.clear();
        assert_eq!(
            session.handle_client_event(
                br#"{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_item_r3_1","output":"{\"ok\":true}"}}"#,
                false,
                &mut host
            ),
            Ok(false)
        );
        assert_eq!(event_type(&host.events[0]), "conversation.item.created");
        assert!(host.tool_outputs.is_empty());

        host.events.clear();
        assert_eq!(
            session.handle_client_event(br#"{"type":"response.create"}"#, false, &mut host),
            Ok(true)
        );
        assert_eq!(host.tool_outputs, vec![vec![r#"{"ok":true}"#.to_string()]]);

        // Text response + reasoning delta, then truncate the assistant item.
        let mut send = |s: &str| {
            host.events.push(s.to_string());
            true
        };
        assert!(session.response_start(&mut send).is_ok());
        assert!(session.response_reasoning("thinking", &mut send).is_ok());
        assert!(session.response_text("Done", &mut send).is_ok());
        let text_outcome = ResponseOutcome {
            ok: true,
            cancelled: false,
            incomplete: false,
            input_tokens: 5,
            output_tokens: 3,
            calls: Vec::new(),
            failure: String::new(),
        };
        assert!(session.response_finish(&text_outcome, &mut send).is_ok());
        assert_eq!(
            host.events
                .iter()
                .map(|e| event_type(e))
                .collect::<Vec<_>>(),
            vec![
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.done",
            ]
        );
        assert!(
            host.events
                .last()
                .unwrap()
                .contains(r#""reasoning":"thinking""#)
        );

        let reply_item = parse_value(host.events[1].as_bytes(), Limits::NONE)
            .unwrap()
            .as_object()
            .unwrap()
            .get("item")
            .unwrap()
            .as_object()
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();

        host.events.clear();
        let truncate = format!(
            r#"{{"type":"conversation.item.truncate","item_id":"{reply_item}","content_index":0,"audio_end_ms":40}}"#
        );
        assert_eq!(
            session.handle_client_event(truncate.as_bytes(), false, &mut host),
            Ok(false)
        );
        assert_eq!(host.truncated_samples, Some(960));
        assert_eq!(event_type(&host.events[0]), "conversation.item.truncated");

        // Wrong item_id is rejected.
        host.events.clear();
        assert_eq!(
            session.handle_client_event(
                br#"{"type":"conversation.item.truncate","item_id":"wrong","content_index":0,"audio_end_ms":40}"#,
                false,
                &mut host
            ),
            Ok(false)
        );
        assert_eq!(event_type(&host.events[0]), "error");
    }

    #[test]
    fn vad_turn_detection_and_response_outcomes() {
        let session = Session::new("r4".to_string(), "voice".to_string());
        let mut host = FakeHost::default();

        // Enable server_vad via session.update.
        assert_eq!(
            session.handle_client_event(
                br#"{"type":"session.update","session":{"audio":{"input":{"turn_detection":{"type":"server_vad","silence_duration_ms":200,"idle_timeout_ms":null}}}}}"#,
                false,
                &mut host
            ),
            Ok(false)
        );
        assert_eq!(event_type(&host.events[0]), "session.updated");
        assert_eq!(host.vad_updates.len(), 1);
        assert_eq!(host.vad_updates[0].unwrap().silence_duration_ms, 200);

        // Classify append when VAD is active -> VadAudio.
        let pcm = vec![0u8; 4800];
        let append = format!(
            r#"{{"type":"input_audio_buffer.append","audio":"{}"}}"#,
            encode_base64(&pcm)
        );
        assert_eq!(
            session.classify_client_event(append.as_bytes(), true, false),
            ClientDecision::VadAudio(pcm.clone())
        );

        let (item_id, started_ev) = session.vad_speech_started(500);
        assert_eq!(event_type(&started_ev), "input_audio_buffer.speech_started");
        let stopped_ev = session.vad_speech_stopped(&item_id, 1100);
        assert_eq!(event_type(&stopped_ev), "input_audio_buffer.speech_stopped");

        host.events.clear();
        assert!(session.handle_vad_audio(&pcm, false, &mut host).is_ok());
        assert_eq!(
            session.handle_vad_commit(&item_id, true, false, &mut host),
            Ok(true)
        );
        assert_eq!(
            host.events
                .iter()
                .map(|e| event_type(e))
                .collect::<Vec<_>>(),
            vec!["input_audio_buffer.committed", "conversation.item.created"]
        );

        // Failed response emits status_details.
        host.events.clear();
        let mut send = |s: &str| {
            host.events.push(s.to_string());
            true
        };
        assert!(session.response_start(&mut send).is_ok());
        let failed = ResponseOutcome {
            ok: false,
            cancelled: false,
            incomplete: false,
            input_tokens: 0,
            output_tokens: 0,
            calls: Vec::new(),
            failure: "model error".to_string(),
        };
        assert!(session.response_finish(&failed, &mut send).is_ok());
        assert!(host.events.last().unwrap().contains(r#""status":"failed""#));
        assert!(host.events.last().unwrap().contains(r#""model error""#));

        // session_unavailable helper.
        let unavail = Session::session_unavailable("busy");
        assert!(unavail.contains(r#""code":"session_unavailable""#));
        assert!(unavail.contains(r#""message":"busy""#));
    }
}
