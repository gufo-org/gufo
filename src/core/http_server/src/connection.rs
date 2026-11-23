//! Per-connection hyper HTTP/1.1 driver, request framing checks, keep-alive
//! parking, and C++ handler dispatch.

use std::collections::HashMap;
use std::ffi::c_void;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use http::{HeaderValue, Method, Request, Response, StatusCode, Version};
use http_body_util::{BodyExt, Limited};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::TcpStream;
use tokio::runtime::Handle;
use tokio::sync::{Notify, oneshot, watch};

use crate::bridge::{GufoHttpCancel, GufoHttpResponder, ResponseBody};
use crate::platform::format_peer_addr;
use crate::{GufoHttpHeaderView, GufoHttpRequestView, GufoHttpSliceView};

pub(crate) struct HandlerCallback {
    pub(crate) func: extern "C" fn(*mut c_void, *const GufoHttpRequestView, *mut GufoHttpResponder),
    pub(crate) ctx: usize,
}

impl HandlerCallback {
    fn call(&self, req: &GufoHttpRequestView, responder: &mut GufoHttpResponder) {
        (self.func)(
            self.ctx as *mut c_void,
            req as *const GufoHttpRequestView,
            responder as *mut GufoHttpResponder,
        );
    }
}

pub(crate) struct ConnectionOptions {
    pub(crate) max_request_body_bytes: usize,
    pub(crate) max_connections: usize,
    pub(crate) sse_heartbeat_interval: Duration,
    pub(crate) keep_alive: bool,
    pub(crate) keep_alive_idle_timeout: Duration,
    pub(crate) max_requests_per_connection: usize,
    pub(crate) handler: Arc<HandlerCallback>,
}

struct TrackedConnection {
    serving: bool,
    parked_since: Option<Instant>,
    evict: Arc<Notify>,
}

/// Tracks active and parked keep-alive connections to enforce `max_connections`
/// and evict the longest-idle parked connection when a new client arrives.
pub(crate) struct ConnectionTracker {
    next_id: AtomicUsize,
    conns: Mutex<HashMap<usize, TrackedConnection>>,
}

impl ConnectionTracker {
    pub(crate) fn new() -> Self {
        Self {
            next_id: AtomicUsize::new(1),
            conns: Mutex::new(HashMap::new()),
        }
    }

    /// Tries to admit a new connection. Returns `Some((id, evict_notify))` if
    /// admitted, or `None` if `max_connections` active connections are already
    /// serving.
    pub(crate) fn try_admit(&self, max_connections: usize) -> Option<(usize, Arc<Notify>)> {
        let mut guard = self.conns.lock().ok()?;
        let serving = guard.values().filter(|c| c.serving).count();
        if serving >= max_connections {
            return None;
        }
        let parked = guard.values().filter(|c| c.parked_since.is_some()).count();
        if serving + parked >= max_connections {
            let oldest_id = guard
                .iter()
                .filter_map(|(&id, c)| c.parked_since.map(|t| (id, t)))
                .min_by_key(|&(_, t)| t)
                .map(|(id, _)| id);
            if let Some(id) = oldest_id {
                if let Some(entry) = guard.get_mut(&id) {
                    entry.parked_since = None;
                    entry.evict.notify_one();
                }
            }
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let evict = Arc::new(Notify::new());
        guard.insert(
            id,
            TrackedConnection {
                serving: true,
                parked_since: None,
                evict: Arc::clone(&evict),
            },
        );
        Some((id, evict))
    }

    pub(crate) fn mark_serving(&self, id: usize) {
        if let Ok(mut guard) = self.conns.lock() {
            if let Some(entry) = guard.get_mut(&id) {
                entry.serving = true;
                entry.parked_since = None;
            }
        }
    }

    pub(crate) fn mark_parked(&self, id: usize) {
        if let Ok(mut guard) = self.conns.lock() {
            if let Some(entry) = guard.get_mut(&id) {
                entry.serving = false;
                entry.parked_since = Some(Instant::now());
            }
        }
    }

    pub(crate) fn remove(&self, id: usize) {
        if let Ok(mut guard) = self.conns.lock() {
            guard.remove(&id);
        }
    }

    pub(crate) fn evict_all(&self) {
        if let Ok(mut guard) = self.conns.lock() {
            for entry in guard.values_mut() {
                entry.parked_since = None;
                entry.evict.notify_one();
            }
        }
    }
}

/// Writes an immediate 503 overloaded response and closes the socket when
/// `max_connections` is reached.
pub(crate) async fn reject_overloaded(mut stream: TcpStream) {
    let body = r#"{"error":{"message":"connection limit reached","type":"server_error","code":"overloaded"}}"#;
    let resp = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nRetry-After: 1\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(resp.as_bytes()).await;
    let _ = stream.shutdown().await;
}

pub(crate) struct ConnectionContext {
    pub(crate) conn_id: usize,
    pub(crate) evict: Arc<Notify>,
    pub(crate) shutdown: watch::Receiver<bool>,
    pub(crate) tracker: Arc<ConnectionTracker>,
    pub(crate) options: Arc<ConnectionOptions>,
    pub(crate) runtime: Handle,
}

struct TrackedStream {
    inner: TcpStream,
    pipelined: Arc<AtomicBool>,
}

impl AsyncRead for TrackedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let before = buf.filled().len();
        let res = Pin::new(&mut self.inner).poll_read(cx, buf);
        if let Poll::Ready(Ok(())) = &res {
            let newly_read = &buf.filled()[before..];
            if let Some(pos) = newly_read.windows(4).position(|w| w == b"\r\n\r\n") {
                if pos + 4 < newly_read.len() {
                    self.pipelined.store(true, Ordering::Release);
                }
            }
        }
        res
    }
}

impl AsyncWrite for TrackedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

pub(crate) async fn serve_connection(
    stream: TcpStream,
    peer_addr: Option<std::net::SocketAddr>,
    ctx: ConnectionContext,
) {
    let ConnectionContext {
        conn_id,
        evict,
        mut shutdown,
        tracker,
        options,
        runtime,
    } = ctx;
    let _ = stream.set_nodelay(true);
    let peer_str = peer_addr
        .map(format_peer_addr)
        .unwrap_or_else(|| "anonymous".to_string());
    let served_count = Arc::new(AtomicUsize::new(0));
    let pipelined = Arc::new(AtomicBool::new(false));

    let io = TokioIo::new(TrackedStream {
        inner: stream,
        pipelined: Arc::clone(&pipelined),
    });
    let mut builder = http1::Builder::new();
    builder
        .keep_alive(true)
        .title_case_headers(true)
        .max_headers(101)
        .timer(TokioTimer::new());
    if options.keep_alive && !options.keep_alive_idle_timeout.is_zero() {
        builder.header_read_timeout(options.keep_alive_idle_timeout);
    }

    let tracker_svc = Arc::clone(&tracker);
    let options_svc = Arc::clone(&options);
    let runtime_svc = runtime.clone();
    let shutdown_svc = shutdown.clone();
    let served_svc = Arc::clone(&served_count);
    let peer_svc = Arc::new(peer_str);
    let pipelined_svc = Arc::clone(&pipelined);

    let service = service_fn(move |req: Request<Incoming>| {
        let tracker = Arc::clone(&tracker_svc);
        let options = Arc::clone(&options_svc);
        let runtime = runtime_svc.clone();
        let shutdown = shutdown_svc.clone();
        let served = Arc::clone(&served_svc);
        let peer = Arc::clone(&peer_svc);
        let pipelined = Arc::clone(&pipelined_svc);
        async move {
            tracker.mark_serving(conn_id);
            let n = served.fetch_add(1, Ordering::AcqRel) + 1;
            let client_wants_keep = client_requests_keep_alive(&req);
            let force_close = !options.keep_alive
                || !client_wants_keep
                || n >= options.max_requests_per_connection;
            let has_body = req.headers().contains_key(http::header::CONTENT_LENGTH);
            let resp =
                handle_hyper_request(req, &peer, force_close, &options, &runtime, shutdown).await;
            let saw_trailing = pipelined.swap(false, Ordering::AcqRel);
            let is_pipelined = !has_body && saw_trailing;
            if !force_close && !is_pipelined {
                tracker.mark_parked(conn_id);
            }
            Ok::<_, std::convert::Infallible>(resp)
        }
    });

    let mut conn = Box::pin(builder.serve_connection(io, service).with_upgrades());
    tokio::select! {
        _ = &mut conn => {}
        _ = evict.notified() => {
            Pin::as_mut(&mut conn).graceful_shutdown();
            let _ = tokio::time::timeout(Duration::from_secs(2), &mut conn).await;
        }
        _ = shutdown.changed() => {
            Pin::as_mut(&mut conn).graceful_shutdown();
            let _ = tokio::time::timeout(Duration::from_secs(2), &mut conn).await;
        }
    }

    tracker.remove(conn_id);
}

struct CancelOnDrop {
    cancel: Arc<AtomicBool>,
    armed: bool,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancel.store(true, Ordering::Release);
        }
    }
}

fn client_requests_keep_alive(req: &Request<Incoming>) -> bool {
    let conn_has = |token: &str| {
        req.headers()
            .get_all(http::header::CONNECTION)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .any(|v| {
                v.split(',')
                    .any(|item| item.trim().eq_ignore_ascii_case(token))
            })
    };
    if req.version() == Version::HTTP_11 {
        !conn_has("close")
    } else {
        conn_has("keep-alive")
    }
}

async fn handle_hyper_request(
    mut req: Request<Incoming>,
    peer_addr: &str,
    force_close: bool,
    options: &ConnectionOptions,
    runtime: &Handle,
    shutdown: watch::Receiver<bool>,
) -> Response<ResponseBody> {
    if req.headers().len() > 100 {
        return json_error(
            StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE,
            "request has more than 100 header fields",
            "invalid_request_error",
            "request_header_fields_too_large",
            true,
        );
    }

    // Reject Transfer-Encoding on requests to preserve existing framing contract.
    if req.headers().contains_key(http::header::TRANSFER_ENCODING) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "malformed request",
            "invalid_request_error",
            "bad_request",
            true,
        );
    }

    // Check Content-Length before reading the body.
    if let Some(cl_val) = req.headers().get(http::header::CONTENT_LENGTH) {
        let Ok(cl_str) = cl_val.to_str() else {
            return json_error(
                StatusCode::BAD_REQUEST,
                "malformed request",
                "invalid_request_error",
                "bad_request",
                true,
            );
        };
        let Ok(cl) = cl_str.trim().parse::<u64>() else {
            return json_error(
                StatusCode::BAD_REQUEST,
                "malformed request",
                "invalid_request_error",
                "bad_request",
                true,
            );
        };
        if cl > options.max_request_body_bytes as u64 {
            return json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body is too large",
                "invalid_request_error",
                "payload_too_large",
                true,
            );
        }
    }

    let is_upgrade = req
        .headers()
        .get(http::header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("websocket"))
        });
    let on_upgrade = if is_upgrade {
        Some(hyper::upgrade::on(&mut req))
    } else {
        None
    };

    let method = req.method().as_str().to_string();
    let is_head = *req.method() == Method::HEAD;
    let path = req.uri().path().to_string();
    let query = req.uri().query().unwrap_or("").to_string();
    let http11 = req.version() == Version::HTTP_11;
    let headers: Vec<(String, Vec<u8>)> = req
        .headers()
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.as_bytes().to_vec()))
        .collect();

    let body_bytes = if is_upgrade {
        Bytes::new()
    } else {
        let limited = Limited::new(req.into_body(), options.max_request_body_bytes);
        match limited.collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(err) => {
                if err
                    .downcast_ref::<http_body_util::LengthLimitError>()
                    .is_some()
                {
                    return json_error(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "request body is too large",
                        "invalid_request_error",
                        "payload_too_large",
                        true,
                    );
                }
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "malformed request",
                    "invalid_request_error",
                    "bad_request",
                    true,
                );
            }
        }
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let mut cancel_guard = CancelOnDrop {
        cancel: Arc::clone(&cancel),
        armed: true,
    };
    let (resp_tx, resp_rx) = oneshot::channel();
    let handler = Arc::clone(&options.handler);
    let peer = peer_addr.to_string();
    let rt_handle = runtime.clone();
    let sse_interval = options.sse_heartbeat_interval;
    let cancel_for_handler = Arc::clone(&cancel);

    runtime.spawn_blocking(move || {
        let header_views: Vec<GufoHttpHeaderView> = headers
            .iter()
            .map(|(k, v)| GufoHttpHeaderView {
                name: GufoHttpSliceView {
                    data: k.as_ptr(),
                    len: k.len(),
                },
                value: GufoHttpSliceView {
                    data: v.as_ptr(),
                    len: v.len(),
                },
            })
            .collect();
        let mut cancel_token = GufoHttpCancel::new(Arc::clone(&cancel_for_handler));
        let req_view = GufoHttpRequestView {
            method: GufoHttpSliceView {
                data: method.as_ptr(),
                len: method.len(),
            },
            path: GufoHttpSliceView {
                data: path.as_ptr(),
                len: path.len(),
            },
            query: GufoHttpSliceView {
                data: query.as_ptr(),
                len: query.len(),
            },
            body: GufoHttpSliceView {
                data: body_bytes.as_ptr(),
                len: body_bytes.len(),
            },
            headers: if header_views.is_empty() {
                std::ptr::null()
            } else {
                header_views.as_ptr()
            },
            headers_len: header_views.len(),
            peer_addr: GufoHttpSliceView {
                data: peer.as_ptr(),
                len: peer.len(),
            },
            http11: i32::from(http11),
            cancel: &mut cancel_token as *mut GufoHttpCancel,
        };
        let mut responder = GufoHttpResponder::new(
            resp_tx,
            on_upgrade,
            rt_handle,
            shutdown,
            cancel_for_handler,
            is_head,
            force_close || is_upgrade,
            sse_interval,
        );
        handler.call(&req_view, &mut responder);
        if !responder.has_responded() {
            let _ = responder.respond_buffered(
                500,
                Vec::new(),
                Bytes::from_static(
                    br#"{"error":{"message":"handler did not send a response","type":"internal_error","code":"server_exception"}}"#,
                ),
            );
        }
    });

    match resp_rx.await {
        Ok(resp) => {
            cancel_guard.armed = false;
            resp
        }
        Err(_) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal server error",
            "internal_error",
            "server_exception",
            true,
        ),
    }
}

fn json_error(
    status: StatusCode,
    message: &str,
    err_type: &str,
    code: &str,
    close: bool,
) -> Response<ResponseBody> {
    let body =
        format!(r#"{{"error":{{"message":"{message}","type":"{err_type}","code":"{code}"}}}}"#,);
    let mut builder = Response::builder()
        .status(status)
        .header(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )
        .header(http::header::CONTENT_LENGTH, body.len().to_string());
    if close {
        builder = builder.header(http::header::CONNECTION, HeaderValue::from_static("close"));
    }
    builder
        .body(ResponseBody::Full(Some(Bytes::from(body))))
        .unwrap_or_else(|_| Response::new(ResponseBody::Empty))
}
