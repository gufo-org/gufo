//! Decodes Chat Completions, Responses and Messages request bodies into the
//! request C++ builds (`ChatRequest`), with the checks, check order, error
//! messages and image reads of the C++ decoders it replaces
//! (src/cli/serve/openai_chat.cpp and http_server.cpp at Gufo 0.35). The
//! helpers the completions endpoints share stay C++ and run through the host
//! (`GufoChatHost`); see include/gufo/chat_request.h.
#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod anthropic;
mod chat;
mod content;
mod draft;
mod error;
mod fields;
mod host;
mod message;
mod reasoning;
mod responses;
mod tool_choice;
mod tools;

use draft::{Draft, Reasoning, ToolChoice, Tools};
use error::Failure;
use gufo_ffi::guard;
use host::{Callbacks, Host};
pub use host::{GufoChatEvent, GufoChatHost, GufoChatMessage, GufoChatRejection, HostField};
use std::ffi::c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoChatDecode {
    Accepted = 0,
    Rejected = 1,
    NoMemory = 2,
    Stopped = 3,
    Internal = 4,
}

/// `GufoChatApi`, as a raw value so unknown ones are a caller bug.
pub type GufoChatApi = i32;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoChatReasoning {
    pub enabled: i32,
    pub effort: i32,
    pub preserve_thinking: i32,
    pub has_budget: u8,
    pub budget_tokens: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoChatBytes {
    pub data: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct GufoChatDefaults {
    pub model_id: GufoChatBytes,
    pub require_model: u8,
    pub reasoning: GufoChatReasoning,
    pub tool_names: *const GufoChatBytes,
    pub tool_name_count: usize,
    pub tool_choice: i32,
}

/// The decode's view of `GufoChatDefaults`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Defaults {
    pub model_id: String,
    pub require_model: bool,
    pub reasoning: Reasoning,
    pub tool_names: Vec<Vec<u8>>,
    pub tool_choice: ToolChoice,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            require_model: true,
            reasoning: Reasoning::default(),
            tool_names: Vec::new(),
            tool_choice: ToolChoice::Auto,
        }
    }
}

fn optional_bool(raw: i32) -> Option<Option<bool>> {
    match raw {
        -1 => Some(None),
        0 => Some(Some(false)),
        1 => Some(Some(true)),
        _ => None,
    }
}

impl GufoChatReasoning {
    fn decode(&self) -> Option<Reasoning> {
        Some(Reasoning {
            enabled: optional_bool(self.enabled)?,
            effort: match self.effort {
                -1 => None,
                raw => Some(
                    u8::try_from(raw)
                        .ok()
                        .filter(|effort| *effort <= host::EFFORT_MAX)?,
                ),
            },
            preserve_thinking: optional_bool(self.preserve_thinking)?,
            budget_tokens: match self.has_budget {
                0 => None,
                1 => Some(self.budget_tokens),
                _ => return None,
            },
        })
    }
}

#[allow(unsafe_code)]
impl GufoChatDefaults {
    /// # Safety
    /// The byte ranges and the name array are readable for the call.
    unsafe fn decode(&self) -> Result<Defaults, GufoChatDecode> {
        // SAFETY: as the caller guarantees for `decode_inner`.
        let defaults = unsafe { self.decode_inner() }.ok_or(GufoChatDecode::Internal)?;
        defaults.map_err(|_| GufoChatDecode::NoMemory)
    }

    /// # Safety
    /// As for `decode`.
    unsafe fn decode_inner(&self) -> Option<Result<Defaults, Failure>> {
        // SAFETY: the caller guarantees the model id bytes.
        let model = unsafe { gufo_ffi::input(self.model_id.data, self.model_id.len) }?;
        // SAFETY: the caller guarantees `tool_name_count` names.
        let names = unsafe { gufo_ffi::input(self.tool_names, self.tool_name_count) }?;
        let mut slices = Vec::new();
        if slices.try_reserve_exact(names.len()).is_err() {
            return Some(Err(Failure::NoMemory));
        }
        for name in names {
            // SAFETY: as above, for each name's bytes.
            slices.push(unsafe { gufo_ffi::input(name.data, name.len) }?);
        }
        let reasoning = self.reasoning.decode()?;
        let tool_choice = ToolChoice::from_raw(self.tool_choice)?;
        let owned = || -> Result<Defaults, Failure> {
            let tool_names = copy_names(&slices)?;
            // Served names are UTF-8; others reach messages as replacements.
            let model_id = match std::str::from_utf8(model) {
                Ok(model) => fields::copy(model)?,
                Err(_) => String::from_utf8_lossy(model).into_owned(),
            };
            Ok(Defaults {
                model_id,
                require_model: self.require_model != 0,
                reasoning,
                tool_names,
                tool_choice,
            })
        };
        Some(owned())
    }
}

/// Owned copies of the request's tool names.
fn copy_names(names: &[impl AsRef<[u8]>]) -> Result<Vec<Vec<u8>>, Failure> {
    let mut out = Vec::new();
    out.try_reserve_exact(names.len())?;
    for name in names {
        let name = name.as_ref();
        let mut owned = Vec::new();
        owned.try_reserve_exact(name.len())?;
        owned.extend_from_slice(name);
        out.push(owned);
    }
    Ok(out)
}

/// Which body a decode reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    Chat,
    Responses,
    Messages,
    ResponseControls,
    ResponseItem,
    AnthropicTools,
}

impl Api {
    fn from_raw(raw: GufoChatApi) -> Option<Self> {
        Some(match raw {
            0 => Self::Chat,
            1 => Self::Responses,
            2 => Self::Messages,
            3 => Self::ResponseControls,
            4 => Self::ResponseItem,
            5 => Self::AnthropicTools,
            _ => return None,
        })
    }
}

/// Decodes `body` as `api` reads it; the draft is complete only on success.
pub fn decode(
    api: Api,
    body: &[u8],
    defaults: &Defaults,
    host: &mut impl Host,
) -> Result<Draft, Failure> {
    let fragment = || -> Result<Draft, Failure> {
        Ok(Draft::new(
            defaults.reasoning,
            Tools::new(copy_names(&defaults.tool_names)?, defaults.tool_choice),
        ))
    };
    match api {
        Api::Chat => chat::decode(body, defaults, host),
        Api::Responses => responses::decode(body, defaults, host),
        Api::Messages => anthropic::decode(body, defaults, host),
        Api::ResponseControls => {
            let mut draft = fragment()?;
            responses::decode_controls(body, &mut draft, host).map(|()| draft)
        }
        Api::ResponseItem => {
            let mut draft = fragment()?;
            responses::decode_item(body, &mut draft, host).map(|()| draft)
        }
        Api::AnthropicTools => {
            let mut draft = fragment()?;
            anthropic::decode_tools(body, &mut draft, host).map(|()| draft)
        }
    }
}

fn run(api: Api, body: &[u8], defaults: &Defaults, host: &mut Callbacks) -> GufoChatDecode {
    let result = decode(api, body, defaults, host).and_then(|draft| draft.emit(host));
    match result {
        Ok(()) => GufoChatDecode::Accepted,
        Err(Failure::Rejected(rejection)) => {
            host.reject(rejection.status, rejection.code, &rejection.message);
            GufoChatDecode::Rejected
        }
        Err(Failure::HostRejected) => GufoChatDecode::Rejected,
        // Every caller names its rejection; an unnamed one is a bug.
        Err(Failure::Invalid(_)) | Err(Failure::Internal) => GufoChatDecode::Internal,
        Err(Failure::NoMemory) => GufoChatDecode::NoMemory,
        Err(Failure::Stopped) => GufoChatDecode::Stopped,
    }
}

/// Decodes `body` as `api` reads it and emits the request to `host`, or
/// rejects it. See include/gufo/chat_request.h.
///
/// # Safety
/// `body` points to `len` readable bytes, unchanged during the call, or `len`
/// is 0; `defaults` and `host` point to valid values whose byte ranges are
/// readable; each callback may be called with `context` until this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_decode(
    api: GufoChatApi,
    body: *const u8,
    len: usize,
    defaults: *const GufoChatDefaults,
    host: *const GufoChatHost,
    context: *mut c_void,
) -> GufoChatDecode {
    guard(GufoChatDecode::Internal, || {
        let Some(api) = Api::from_raw(api) else {
            return GufoChatDecode::Internal;
        };
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(body) = (unsafe { gufo_ffi::input(body, len) }) else {
            return GufoChatDecode::Internal;
        };
        // SAFETY: the caller passes valid `defaults` and `host` or null.
        let (Some(defaults), Some(host)) = (unsafe { defaults.as_ref() }, unsafe { host.as_ref() })
        else {
            return GufoChatDecode::Internal;
        };
        // SAFETY: the caller guarantees the byte ranges `defaults` names.
        let defaults = match unsafe { defaults.decode() } {
            Ok(defaults) => defaults,
            Err(status) => return status,
        };
        let Some(mut callbacks) = Callbacks::new(host, context) else {
            return GufoChatDecode::Internal;
        };
        run(api, body, &defaults, &mut callbacks)
    })
}

#[cfg(test)]
mod tests;
