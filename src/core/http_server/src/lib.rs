//! HTTP/1.1 transport and WebSocket upgrade driver built on `tokio` and `hyper`,
//! exposed to C++ via a narrow C ABI.

#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod bridge;
mod connection;
mod listener;
mod loading;
mod platform;
mod websocket;

use std::ffi::{c_int, c_void};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use gufo_ffi::{guard, input};
use http::{HeaderName, HeaderValue};

pub use bridge::{GufoHttpCancel, GufoHttpResponder, GufoHttpStream};
pub use listener::GufoHttpServer;
pub use loading::GufoLoadingResponder;
pub use websocket::GufoWsSession;

use bridge::HeartbeatCallback;
use connection::{ConnectionOptions, HandlerCallback};
use listener::DeviceLostCallback;
use websocket::WsHandlerCallback;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GufoHttpSliceView {
    pub data: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GufoHttpHeaderView {
    pub name: GufoHttpSliceView,
    pub value: GufoHttpSliceView,
}

#[repr(C)]
pub struct GufoHttpRequestView {
    pub method: GufoHttpSliceView,
    pub path: GufoHttpSliceView,
    pub query: GufoHttpSliceView,
    pub body: GufoHttpSliceView,
    pub headers: *const GufoHttpHeaderView,
    pub headers_len: usize,
    pub peer_addr: GufoHttpSliceView,
    pub http11: c_int,
    pub cancel: *mut GufoHttpCancel,
    /// Nonzero when the transport refused the request before reading it
    /// whole: the HTTP status (400, 413, 431 or 500) to answer with. The body
    /// is then empty and the connection closes after the response.
    pub rejection: c_int,
}

pub type GufoHttpBytesSink = Option<extern "C" fn(*mut c_void, *const u8, usize)>;
pub type GufoHttpHandlerFn =
    Option<extern "C" fn(*mut c_void, *const GufoHttpRequestView, *mut GufoHttpResponder)>;
pub type GufoHttpDeviceLostFn = Option<extern "C" fn(*mut c_void) -> c_int>;
pub type GufoWsHandlerFn = Option<extern "C" fn(*mut c_void, *mut GufoWsSession)>;

#[repr(C)]
pub struct GufoHttpServerConfig {
    pub host: *const u8,
    pub host_len: usize,
    pub port: c_int,
    pub max_request_body_bytes: usize,
    pub max_connections: usize,
    pub sse_heartbeat_interval_ms: u64,
    pub keep_alive: c_int,
    pub keep_alive_idle_timeout_ms: u64,
    pub max_requests_per_connection: usize,
    pub handler: GufoHttpHandlerFn,
    pub handler_context: *mut c_void,
    pub device_lost: GufoHttpDeviceLostFn,
    pub device_lost_context: *mut c_void,
}

fn emit_bytes(sink: GufoHttpBytesSink, ctx: *mut c_void, bytes: &[u8]) {
    if let Some(f) = sink {
        f(ctx, bytes.as_ptr(), bytes.len());
    }
}

#[allow(unsafe_code)]
unsafe fn collect_headers(
    headers: *const GufoHttpHeaderView,
    len: usize,
) -> Option<Vec<(HeaderName, HeaderValue)>> {
    if len == 0 {
        return Some(Vec::new());
    }
    if headers.is_null() {
        return None;
    }
    // SAFETY: caller guarantees `headers` points to `len` valid
    // `GufoHttpHeaderView` elements for the duration of this call.
    let slice = unsafe { std::slice::from_raw_parts(headers, len) };
    let mut out = Vec::new();
    out.try_reserve(len).ok()?;
    for h in slice {
        // SAFETY: caller guarantees each header's name and value slices are
        // valid for the duration of this call.
        let name_bytes = unsafe { input(h.name.data, h.name.len) }?;
        // SAFETY: same as above.
        let val_bytes = unsafe { input(h.value.data, h.value.len) }?;
        if let (Ok(name), Ok(val)) = (
            HeaderName::from_bytes(name_bytes),
            HeaderValue::from_bytes(val_bytes),
        ) {
            out.push((name, val));
        }
    }
    Some(out)
}

/// # Safety
/// `host` must be valid for `host_len` bytes (or null when `host_len == 0`).
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_is_listen_address(host: *const u8, host_len: usize) -> c_int {
    guard(0, || {
        // SAFETY: caller guarantees `host` is valid for `host_len` bytes.
        let Some(bytes) = (unsafe { input(host, host_len) }) else {
            return 0;
        };
        let Ok(s) = std::str::from_utf8(bytes) else {
            return 0;
        };
        c_int::from(platform::parse_listen_address(s, 0).is_some())
    })
}

/// # Safety
/// `config` must point to a valid `GufoHttpServerConfig` for the duration of
/// this call.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_server_start(
    config: *const GufoHttpServerConfig,
    error_sink: GufoHttpBytesSink,
    error_context: *mut c_void,
) -> *mut GufoHttpServer {
    guard(std::ptr::null_mut(), || {
        if config.is_null() {
            emit_bytes(error_sink, error_context, b"null server config");
            return std::ptr::null_mut();
        }
        // SAFETY: caller guarantees `config` is non-null and valid.
        let cfg = unsafe { &*config };
        // SAFETY: caller guarantees `cfg.host` is valid for `cfg.host_len`.
        let Some(host_bytes) = (unsafe { input(cfg.host, cfg.host_len) }) else {
            emit_bytes(error_sink, error_context, b"invalid host pointer");
            return std::ptr::null_mut();
        };
        let Ok(host) = std::str::from_utf8(host_bytes) else {
            emit_bytes(error_sink, error_context, b"host is not valid UTF-8");
            return std::ptr::null_mut();
        };
        let Some(handler_fn) = cfg.handler else {
            emit_bytes(error_sink, error_context, b"missing request handler");
            return std::ptr::null_mut();
        };
        let options = ConnectionOptions {
            max_request_body_bytes: cfg.max_request_body_bytes,
            max_connections: cfg.max_connections.max(1),
            sse_heartbeat_interval: Duration::from_millis(cfg.sse_heartbeat_interval_ms),
            keep_alive: cfg.keep_alive != 0,
            keep_alive_idle_timeout: Duration::from_millis(cfg.keep_alive_idle_timeout_ms),
            max_requests_per_connection: cfg.max_requests_per_connection.max(1),
            handler: Arc::new(HandlerCallback {
                func: handler_fn,
                ctx: cfg.handler_context as usize,
            }),
        };
        let device_lost = cfg.device_lost.map(|func| DeviceLostCallback {
            func,
            ctx: cfg.device_lost_context as usize,
        });
        match GufoHttpServer::start(host, cfg.port, options, device_lost) {
            Ok(server) => Box::into_raw(Box::new(server)),
            Err(err) => {
                emit_bytes(error_sink, error_context, err.as_bytes());
                std::ptr::null_mut()
            }
        }
    })
}

/// # Safety
/// `server` must be a valid pointer returned by `gufo_http_server_start`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_server_port(server: *const GufoHttpServer) -> c_int {
    guard(0, || {
        if server.is_null() {
            return 0;
        }
        // SAFETY: caller guarantees `server` is valid.
        c_int::from(unsafe { &*server }.port())
    })
}

/// # Safety
/// `server` must be a valid pointer returned by `gufo_http_server_start`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_server_run(server: *mut GufoHttpServer) -> c_int {
    guard(-1, || {
        if server.is_null() {
            return -1;
        }
        // SAFETY: caller guarantees `server` is valid and alive for the duration of `run`.
        match unsafe { &*server }.run() {
            Ok(()) => 0,
            Err(_) => -1,
        }
    })
}

/// # Safety
/// `server` must be null or a valid pointer returned by `gufo_http_server_start`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_server_stop(server: *mut GufoHttpServer) {
    guard((), || {
        if !server.is_null() {
            // SAFETY: caller guarantees `server` is valid.
            unsafe { &*server }.stop();
        }
    });
}

/// # Safety
/// `server` must be null or a pointer from `gufo_http_server_start` freed once.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_server_free(server: *mut GufoHttpServer) {
    guard((), || {
        if !server.is_null() {
            // SAFETY: caller transfers ownership of `server` once.
            drop(unsafe { Box::from_raw(server) });
        }
    });
}

/// # Safety
/// `cancel` must be null or a valid pointer from `GufoHttpRequestView::cancel`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_cancel_is_cancelled(cancel: *const GufoHttpCancel) -> c_int {
    guard(0, || {
        if cancel.is_null() {
            return 0;
        }
        // SAFETY: caller guarantees `cancel` is valid during the handler call.
        c_int::from(unsafe { &*cancel }.is_cancelled())
    })
}

/// # Safety
/// `responder` must be valid; `headers` and `body` must be valid for their lengths.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_respond_buffered(
    responder: *mut GufoHttpResponder,
    status: c_int,
    headers: *const GufoHttpHeaderView,
    headers_len: usize,
    body: *const u8,
    body_len: usize,
) -> c_int {
    guard(0, || {
        if responder.is_null() {
            return 0;
        }
        // SAFETY: caller guarantees `headers` is valid for `headers_len`.
        let Some(hdrs) = (unsafe { collect_headers(headers, headers_len) }) else {
            return 0;
        };
        // SAFETY: caller guarantees `body` is valid for `body_len`.
        let Some(body_slice) = (unsafe { input(body, body_len) }) else {
            return 0;
        };
        let status_u16 = u16::try_from(status).unwrap_or(500);
        // SAFETY: caller guarantees `responder` is valid and uniquely accessed.
        let ok = unsafe { &mut *responder }.respond_buffered(
            status_u16,
            hdrs,
            Bytes::copy_from_slice(body_slice),
        );
        c_int::from(ok)
    })
}

/// # Safety
/// `responder` and `headers` must be valid for the duration of this call.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_respond_stream_begin(
    responder: *mut GufoHttpResponder,
    status: c_int,
    headers: *const GufoHttpHeaderView,
    headers_len: usize,
    is_sse: c_int,
    defer_headers: c_int,
    on_heartbeat: GufoHttpBytesSink,
    on_heartbeat_context: *mut c_void,
) -> *mut GufoHttpStream {
    guard(std::ptr::null_mut(), || {
        if responder.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: caller guarantees `headers` is valid for `headers_len`.
        let Some(hdrs) = (unsafe { collect_headers(headers, headers_len) }) else {
            return std::ptr::null_mut();
        };
        let status_u16 = u16::try_from(status).unwrap_or(500);
        let hb = HeartbeatCallback::new(on_heartbeat, on_heartbeat_context);
        // SAFETY: caller guarantees `responder` is valid and uniquely accessed.
        match unsafe { &mut *responder }.respond_stream_begin(
            status_u16,
            hdrs,
            is_sse != 0,
            defer_headers != 0,
            hb,
        ) {
            Some(stream) => Box::into_raw(Box::new(stream)),
            None => std::ptr::null_mut(),
        }
    })
}

/// # Safety
/// `stream` must be a valid pointer from `gufo_http_respond_stream_begin`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_stream_write(
    stream: *mut GufoHttpStream,
    data: *const u8,
    len: usize,
) -> c_int {
    guard(0, || {
        if stream.is_null() {
            return 0;
        }
        // SAFETY: caller guarantees `data` is valid for `len`.
        let Some(slice) = (unsafe { input(data, len) }) else {
            return 0;
        };
        // SAFETY: caller guarantees `stream` is valid and uniquely accessed.
        c_int::from(unsafe { &mut *stream }.write(slice))
    })
}

/// # Safety
/// `stream` must be null or a pointer from `gufo_http_respond_stream_begin`
/// consumed once.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_stream_end(
    stream: *mut GufoHttpStream,
    complete: c_int,
) -> c_int {
    guard(0, || {
        if stream.is_null() {
            return 0;
        }
        // SAFETY: caller transfers ownership of `stream` once.
        let s = unsafe { Box::from_raw(stream) };
        c_int::from(s.finish(complete != 0))
    })
}

/// # Safety
/// `stream` must be a deferred stream not yet ended; `responder` must be the
/// same responder that created it.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_stream_abort_deferred(
    stream: *mut GufoHttpStream,
    responder: *mut GufoHttpResponder,
) {
    guard((), || {
        if stream.is_null() {
            return;
        }
        // SAFETY: caller transfers ownership of `stream` once.
        let s = unsafe { Box::from_raw(stream) };
        if !responder.is_null() {
            // SAFETY: caller guarantees `responder` is valid.
            s.abort_deferred(unsafe { &mut *responder });
        }
    });
}

/// # Safety
/// `responder` and `headers` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_http_respond_websocket(
    responder: *mut GufoHttpResponder,
    status: c_int,
    headers: *const GufoHttpHeaderView,
    headers_len: usize,
    ws_handler: GufoWsHandlerFn,
    ws_context: *mut c_void,
) -> c_int {
    guard(0, || {
        if responder.is_null() {
            return 0;
        }
        let Some(handler_fn) = ws_handler else {
            return 0;
        };
        // SAFETY: caller guarantees `headers` is valid for `headers_len`.
        let Some(hdrs) = (unsafe { collect_headers(headers, headers_len) }) else {
            return 0;
        };
        let status_u16 = u16::try_from(status).unwrap_or(101);
        let cb = WsHandlerCallback::new(handler_fn, ws_context);
        // SAFETY: caller guarantees `responder` is valid.
        c_int::from(unsafe { &mut *responder }.respond_websocket(status_u16, hdrs, cb))
    })
}

/// # Safety
/// `session` must be a valid pointer passed to `GufoWsHandlerFn`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_ws_session_receive(
    session: *mut GufoWsSession,
    sink: GufoHttpBytesSink,
    sink_context: *mut c_void,
) -> c_int {
    guard(0, || {
        if session.is_null() {
            return 0;
        }
        // SAFETY: caller guarantees `session` is valid.
        match unsafe { &*session }.receive() {
            Some(msg) => {
                emit_bytes(sink, sink_context, &msg);
                1
            }
            None => 0,
        }
    })
}

/// # Safety
/// `session` must be valid; `data` must be valid for `len` bytes.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_ws_session_send_text(
    session: *mut GufoWsSession,
    data: *const u8,
    len: usize,
) -> c_int {
    guard(0, || {
        if session.is_null() {
            return 0;
        }
        // SAFETY: caller guarantees `data` is valid for `len`.
        let Some(slice) = (unsafe { input(data, len) }) else {
            return 0;
        };
        // SAFETY: caller guarantees `session` is valid.
        c_int::from(unsafe { &*session }.send_text(slice))
    })
}

/// # Safety
/// `session` must be valid; `data` must be valid for `len` bytes.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_ws_session_send_binary(
    session: *mut GufoWsSession,
    data: *const u8,
    len: usize,
) -> c_int {
    guard(0, || {
        if session.is_null() {
            return 0;
        }
        // SAFETY: caller guarantees `data` is valid for `len`.
        let Some(slice) = (unsafe { input(data, len) }) else {
            return 0;
        };
        // SAFETY: caller guarantees `session` is valid.
        c_int::from(unsafe { &*session }.send_binary(slice))
    })
}

/// # Safety
/// `session` must be null or valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_ws_session_close(session: *mut GufoWsSession, code: u16) {
    guard((), || {
        if !session.is_null() {
            // SAFETY: caller guarantees `session` is valid.
            unsafe { &*session }.close(code);
        }
    });
}

/// # Safety
/// `session` must be null or valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_ws_session_is_closed(session: *const GufoWsSession) -> c_int {
    guard(1, || {
        if session.is_null() {
            return 1;
        }
        // SAFETY: caller guarantees `session` is valid.
        c_int::from(unsafe { &*session }.is_closed())
    })
}

/// # Safety
/// `host` must be valid for `host_len` bytes.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_loading_responder_start(
    host: *const u8,
    host_len: usize,
    port: c_int,
) -> *mut GufoLoadingResponder {
    guard(std::ptr::null_mut(), || {
        // SAFETY: caller guarantees `host` is valid for `host_len`.
        let Some(bytes) = (unsafe { input(host, host_len) }) else {
            return std::ptr::null_mut();
        };
        let Ok(host_str) = std::str::from_utf8(bytes) else {
            return std::ptr::null_mut();
        };
        match GufoLoadingResponder::start(host_str, port) {
            Some(r) => Box::into_raw(Box::new(r)),
            None => std::ptr::null_mut(),
        }
    })
}

/// # Safety
/// `responder` must be null or valid; `message` must be valid for `message_len`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_loading_responder_set_progress(
    responder: *mut GufoLoadingResponder,
    progress: f64,
    message: *const u8,
    message_len: usize,
) {
    guard((), || {
        if responder.is_null() {
            return;
        }
        // SAFETY: caller guarantees `message` is valid for `message_len`.
        let Some(bytes) = (unsafe { input(message, message_len) }) else {
            return;
        };
        let msg = std::str::from_utf8(bytes).unwrap_or("");
        // SAFETY: caller guarantees `responder` is valid.
        unsafe { &*responder }.set_progress(progress, msg);
    });
}

/// # Safety
/// `responder` must be null or valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_loading_responder_stop(responder: *mut GufoLoadingResponder) {
    guard((), || {
        if !responder.is_null() {
            // SAFETY: caller guarantees `responder` is valid; `stop` takes `&self`.
            unsafe { &*responder }.stop();
        }
    });
}

/// # Safety
/// `responder` must be null or a pointer from `gufo_loading_responder_start`
/// freed once.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_loading_responder_free(responder: *mut GufoLoadingResponder) {
    guard((), || {
        if !responder.is_null() {
            // SAFETY: caller transfers ownership of `responder` once.
            drop(unsafe { Box::from_raw(responder) });
        }
    });
}
