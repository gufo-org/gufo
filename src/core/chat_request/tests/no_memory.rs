//! Client-sized allocations fail with `NoMemory` instead of aborting. A
//! global allocator refuses a second live block of 8 MiB or more while armed,
//! so the 10 MiB value the parser builds fits and any copy of it fails.
#![allow(unsafe_code)]

use gufo_chat_request::{
    GufoChatBytes, GufoChatDecode, GufoChatDefaults, GufoChatEvent, GufoChatHost, GufoChatMessage,
    GufoChatReasoning, GufoChatRejection, HostField, gufo_chat_decode,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::{c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};

const BIG: usize = 8 << 20;
static ARMED: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// Takes one more big block, or refuses it.
fn take_big() -> bool {
    if LIVE.fetch_add(1, SeqCst) >= 1 {
        LIVE.fetch_sub(1, SeqCst);
        return false;
    }
    true
}

struct OneBigBlock;

// SAFETY: every call forwards to `System` with the same arguments, or
// returns null, which `GlobalAlloc` allows for a failed allocation.
unsafe impl GlobalAlloc for OneBigBlock {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= BIG && ARMED.load(SeqCst) && !take_big() {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's layout contract holds for `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        if layout.size() >= BIG && ARMED.load(SeqCst) {
            LIVE.fetch_sub(1, SeqCst);
        }
        // SAFETY: `block` came from `System` with `layout`.
        unsafe { System.dealloc(block, layout) }
    }

    unsafe fn realloc(&self, block: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ARMED.load(SeqCst) {
            let was = layout.size() >= BIG;
            let will = size >= BIG;
            if will && !was && !take_big() {
                return std::ptr::null_mut();
            }
            if was && !will {
                LIVE.fetch_sub(1, SeqCst);
            }
        }
        // SAFETY: as in `alloc` and `dealloc`.
        unsafe { System.realloc(block, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: OneBigBlock = OneBigBlock;

unsafe extern "C" fn serves(_: *mut c_void, _: *const u8, _: usize) -> c_int {
    1
}
unsafe extern "C" fn image(
    _: *mut c_void,
    _: *const u8,
    _: usize,
    _: *mut u32,
    _: *const GufoChatMessage,
) -> c_int {
    0
}
unsafe extern "C" fn schema(
    _: *mut c_void,
    _: *const u8,
    _: usize,
    _: *const GufoChatMessage,
) -> c_int {
    0
}
unsafe extern "C" fn field(_: *mut c_void, _: HostField, _: i32, _: *const u8, _: usize) -> c_int {
    0
}
unsafe extern "C" fn emit(_: *mut c_void, _: *const GufoChatEvent) -> c_int {
    0
}
unsafe extern "C" fn reject(_: *mut c_void, _: *const GufoChatRejection) {}

fn decode(api: i32, body: &[u8]) -> GufoChatDecode {
    let host = GufoChatHost {
        serves_model: Some(serves),
        read_image: Some(image),
        compile_schema: Some(schema),
        host_field: Some(field),
        emit: Some(emit),
        reject: Some(reject),
    };
    let defaults = GufoChatDefaults {
        model_id: GufoChatBytes {
            data: b"m".as_ptr(),
            len: 1,
        },
        require_model: 1,
        reasoning: GufoChatReasoning {
            enabled: -1,
            effort: -1,
            preserve_thinking: -1,
            has_budget: 0,
            budget_tokens: 0,
        },
        tool_names: std::ptr::null(),
        tool_name_count: 0,
        tool_choice: 0,
    };
    ARMED.store(true, SeqCst);
    // SAFETY: every pointer is valid for the call; the callbacks ignore the
    // null context.
    let status = unsafe {
        gufo_chat_decode(
            api,
            body.as_ptr(),
            body.len(),
            &defaults,
            &host,
            std::ptr::null_mut(),
        )
    };
    ARMED.store(false, SeqCst);
    status
}

#[test]
fn copies_of_large_values_report_no_memory() {
    let big = "x".repeat(10 << 20);
    let bodies = [
        // A tool's description, copied into its definition.
        (
            0,
            format!(
                r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"tools":[{{"type":"function","function":{{"name":"f","description":"{big}","parameters":{{"type":"object"}}}}}}]}}"#
            ),
        ),
        // A Messages schema, written into the definition.
        (
            2,
            format!(
                r#"{{"model":"m","max_tokens":5,"messages":[{{"role":"user","content":"hi"}}],"tools":[{{"name":"f","input_schema":{{"type":"object","description":"{big}"}}}}]}}"#
            ),
        ),
        // A custom tool call's input, written into its arguments.
        (
            1,
            format!(
                r#"{{"model":"m","input":[{{"type":"custom_tool_call","call_id":"c","name":"f","input":"{big}"}}]}}"#
            ),
        ),
    ];
    drop(big);
    for (api, body) in &bodies {
        assert_eq!(
            decode(*api, body.as_bytes()),
            GufoChatDecode::NoMemory,
            "api {api}"
        );
    }
    // A large value that moves into the request needs no copy.
    let content = format!(
        r#"{{"model":"m","messages":[{{"role":"user","content":"{}"}}]}}"#,
        "x".repeat(10 << 20)
    );
    assert_eq!(decode(0, content.as_bytes()), GufoChatDecode::Accepted);
}
