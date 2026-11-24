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
use hyper::body::{Body, Frame, Incoming, SizeHint};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::TcpStream;
use tokio::runtime::Handle;
use tokio::sync::{Notify, oneshot, watch};
use tokio::time::Sleep;

use crate::bridge::{GufoHttpCancel, GufoHttpResponder, ResponseBody, StreamAbortError};
use crate::platform::format_peer_addr;
use crate::{GufoHttpHeaderView, GufoHttpRequestView, GufoHttpSliceView};

/// A request body that stops arriving, or a client that stops reading its
/// response, fails after this long without progress, like the socket timeouts
/// the C++ transport set.
const STALL_TIMEOUT: Duration = Duration::from_secs(120);

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

    fn set_state(&self, id: usize, serving: bool, parked: bool) {
        if let Ok(mut guard) = self.conns.lock() {
            if let Some(entry) = guard.get_mut(&id) {
                entry.serving = serving;
                entry.parked_since = parked.then(Instant::now);
            }
        }
    }

    fn parked_since(&self, id: usize) -> Option<Instant> {
        self.conns
            .lock()
            .ok()
            .and_then(|guard| guard.get(&id).and_then(|entry| entry.parked_since))
    }

    fn remove(&self, id: usize) {
        if let Ok(mut guard) = self.conns.lock() {
            guard.remove(&id);
        }
    }

    /// True once every admitted connection and the handlers it started have
    /// finished.
    pub(crate) fn is_empty(&self) -> bool {
        self.conns.lock().map_or(true, |guard| guard.is_empty())
    }
}

/// An admitted connection's entry in the tracker. The connection task and
/// every handler it starts hold the slot, so a handler still running after
/// its client left, or a WebSocket session after the upgrade, keeps the
/// entry until it returns.
pub(crate) struct ConnSlot {
    tracker: Arc<ConnectionTracker>,
    id: usize,
    /// Signalled each time the connection parks, for its idle timer.
    parked: Notify,
}

impl ConnSlot {
    pub(crate) fn new(tracker: Arc<ConnectionTracker>, id: usize) -> Self {
        Self {
            tracker,
            id,
            parked: Notify::new(),
        }
    }

    /// A request is in progress: it counts toward the limit.
    fn mark_serving(&self) {
        self.tracker.set_state(self.id, true, false);
    }

    /// The connection waits idle for its next request and may be evicted.
    fn mark_parked(&self) {
        self.tracker.set_state(self.id, false, true);
        self.parked.notify_one();
    }

    /// Resolves once the connection has stayed parked for `timeout`. A new
    /// request or an eviction leaves the parked state and restarts the wait.
    async fn idle_expired(&self, timeout: Duration) {
        loop {
            match self.tracker.parked_since(self.id) {
                Some(since) => {
                    let deadline = since + timeout;
                    if Instant::now() >= deadline {
                        return;
                    }
                    tokio::time::sleep_until(deadline.into()).await;
                }
                None => self.parked.notified().await,
            }
        }
    }

    /// The response is done and the connection is closing.
    fn mark_released(&self) {
        self.tracker.set_state(self.id, false, false);
    }
}

impl Drop for ConnSlot {
    fn drop(&mut self) {
        self.tracker.remove(self.id);
    }
}

/// What the connection's slot becomes once a response body has been written.
#[derive(Clone, Copy)]
enum AfterResponse {
    /// Idle keep-alive: parked and evictable.
    Park,
    /// The connection closes after this response.
    Release,
    /// A pipelined request already arrived, or an upgraded session continues:
    /// the connection stays serving.
    Keep,
}

/// A response body that updates the connection slot once hyper has finished
/// with it, so a streamed response counts as serving until its last chunk.
pub(crate) struct ConnectionBody {
    inner: ResponseBody,
    slot: Arc<ConnSlot>,
    after: AfterResponse,
}

impl Drop for ConnectionBody {
    fn drop(&mut self) {
        match self.after {
            AfterResponse::Park => self.slot.mark_parked(),
            AfterResponse::Release => self.slot.mark_released(),
            AfterResponse::Keep => {}
        }
    }
}

impl Body for ConnectionBody {
    type Data = Bytes;
    type Error = StreamAbortError;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.get_mut().inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
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
    pub(crate) slot: Arc<ConnSlot>,
    pub(crate) evict: Arc<Notify>,
    pub(crate) shutdown: watch::Receiver<bool>,
    pub(crate) options: Arc<ConnectionOptions>,
    pub(crate) runtime: Handle,
}

struct TrackedStream {
    inner: TcpStream,
    pipelined: Arc<AtomicBool>,
    /// Armed while a write cannot make progress; a client that stops reading
    /// for `stall_timeout` fails the connection, which cancels its handler.
    write_stall: Option<Pin<Box<Sleep>>>,
    stall_timeout: Duration,
}

impl TrackedStream {
    fn track_write<T>(
        &mut self,
        cx: &mut Context<'_>,
        res: Poll<io::Result<T>>,
    ) -> Poll<io::Result<T>> {
        if res.is_ready() {
            self.write_stall = None;
            return res;
        }
        let stall = self
            .write_stall
            .get_or_insert_with(|| Box::pin(tokio::time::sleep(self.stall_timeout)));
        if stall.as_mut().poll(cx).is_ready() {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "client stopped reading the response",
            )));
        }
        Poll::Pending
    }
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
        let res = Pin::new(&mut self.inner).poll_write(cx, buf);
        self.track_write(cx, res)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let res = Pin::new(&mut self.inner).poll_write_vectored(cx, bufs);
        self.track_write(cx, res)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
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
        slot,
        evict,
        mut shutdown,
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
        write_stall: None,
        stall_timeout: STALL_TIMEOUT,
    });
    let mut builder = http1::Builder::new();
    builder
        .keep_alive(true)
        .title_case_headers(true)
        .max_headers(101)
        .timer(TokioTimer::new());
    // hyper times every request head, including the wait before it, with one
    // timeout. It gets the stall allowance, so a first request may arrive as
    // late as before; the keep-alive idle limit is timed separately, from the
    // moment a response parks the connection.
    let idle_timeout = (options.keep_alive && !options.keep_alive_idle_timeout.is_zero())
        .then_some(options.keep_alive_idle_timeout);
    builder.header_read_timeout(idle_timeout.map_or(STALL_TIMEOUT, |idle| idle.max(STALL_TIMEOUT)));

    let slot_svc = Arc::clone(&slot);
    let options_svc = Arc::clone(&options);
    let runtime_svc = runtime.clone();
    let shutdown_svc = shutdown.clone();
    let served_svc = Arc::clone(&served_count);
    let peer_svc = Arc::new(peer_str);
    let pipelined_svc = Arc::clone(&pipelined);

    let service = service_fn(move |req: Request<Incoming>| {
        let slot = Arc::clone(&slot_svc);
        let options = Arc::clone(&options_svc);
        let runtime = runtime_svc.clone();
        let shutdown = shutdown_svc.clone();
        let served = Arc::clone(&served_svc);
        let peer = Arc::clone(&peer_svc);
        let pipelined = Arc::clone(&pipelined_svc);
        async move {
            slot.mark_serving();
            let n = served.fetch_add(1, Ordering::AcqRel) + 1;
            let client_wants_keep = client_requests_keep_alive(&req);
            let force_close = !options.keep_alive
                || !client_wants_keep
                || n >= options.max_requests_per_connection;
            let has_body = req.headers().contains_key(http::header::CONTENT_LENGTH);
            let resp = handle_hyper_request(
                req,
                &peer,
                force_close,
                &options,
                &runtime,
                shutdown,
                Arc::clone(&slot),
            )
            .await;
            let saw_trailing = pipelined.swap(false, Ordering::AcqRel);
            let closing = resp
                .headers()
                .get(http::header::CONNECTION)
                .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"close"));
            let after = if resp.status() == StatusCode::SWITCHING_PROTOCOLS {
                AfterResponse::Keep
            } else if closing {
                AfterResponse::Release
            } else if !has_body && saw_trailing {
                AfterResponse::Keep
            } else {
                AfterResponse::Park
            };
            Ok::<_, std::convert::Infallible>(resp.map(|inner| ConnectionBody {
                inner,
                slot,
                after,
            }))
        }
    });

    let mut conn = Box::pin(builder.serve_connection(io, service).with_upgrades());
    tokio::select! {
        _ = &mut conn => {}
        _ = async {
            match idle_timeout {
                Some(timeout) => slot.idle_expired(timeout).await,
                None => std::future::pending().await,
            }
        } => {
            Pin::as_mut(&mut conn).graceful_shutdown();
            let _ = tokio::time::timeout(Duration::from_secs(2), &mut conn).await;
        }
        _ = evict.notified() => {
            Pin::as_mut(&mut conn).graceful_shutdown();
            let _ = tokio::time::timeout(Duration::from_secs(2), &mut conn).await;
        }
        // Shutdown closes the connection at once, like the C++ transport's
        // socket shutdown: dropping it cancels a request in progress, so
        // `run` returns as soon as handlers observe the cancellation.
        // `wait_for` also sees a shutdown sent before this task first ran.
        _ = async { drop(shutdown.wait_for(|stopped| *stopped).await) } => {}
    }
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

/// Status a request is refused with before its handler runs, or `None`.
fn reject_head(req: &Request<Incoming>, options: &ConnectionOptions) -> Option<u16> {
    if req.headers().len() > 100 {
        return Some(431);
    }
    // Requests keep Content-Length framing only, as before.
    if req.headers().contains_key(http::header::TRANSFER_ENCODING) {
        return Some(400);
    }
    if let Some(cl_val) = req.headers().get(http::header::CONTENT_LENGTH) {
        let Some(cl) = cl_val
            .to_str()
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
        else {
            return Some(400);
        };
        if cl > options.max_request_body_bytes as u64 {
            return Some(413);
        }
    }
    None
}

/// Reads a whole request body. Each frame must arrive within the stall
/// timeout. Errors carry the status the request is refused with.
async fn read_body(body: Incoming, limit: usize) -> Result<Bytes, u16> {
    let mut body = Limited::new(body, limit);
    let mut out: Vec<u8> = Vec::new();
    loop {
        let frame = match tokio::time::timeout(STALL_TIMEOUT, body.frame()).await {
            Err(_) => return Err(400),
            Ok(None) => break,
            Ok(Some(Err(err))) => {
                return Err(
                    if err
                        .downcast_ref::<http_body_util::LengthLimitError>()
                        .is_some()
                    {
                        413
                    } else {
                        400
                    },
                );
            }
            Ok(Some(Ok(frame))) => frame,
        };
        if let Ok(data) = frame.into_data() {
            if out.try_reserve(data.len()).is_err() {
                return Err(500);
            }
            out.extend_from_slice(&data);
        }
    }
    Ok(Bytes::from(out))
}

async fn handle_hyper_request(
    mut req: Request<Incoming>,
    peer_addr: &str,
    force_close: bool,
    options: &ConnectionOptions,
    runtime: &Handle,
    shutdown: watch::Receiver<bool>,
    slot: Arc<ConnSlot>,
) -> Response<ResponseBody> {
    // A refused request still reaches the handler, which answers it with the
    // same headers, request ID and log line as any other response.
    let mut rejection = reject_head(&req, options);

    let is_upgrade = rejection.is_none()
        && req
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

    let body_bytes = if rejection.is_some() || is_upgrade {
        Bytes::new()
    } else {
        match read_body(req.into_body(), options.max_request_body_bytes).await {
            Ok(bytes) => bytes,
            Err(status) => {
                rejection = Some(status);
                Bytes::new()
            }
        }
    };
    // Unread or partial body bytes cannot be followed by another request.
    let force_close = force_close || rejection.is_some();

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
        // The connection stays counted until its handler returns.
        let _slot = slot;
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
            rejection: rejection.map_or(0, i32::from),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stalled_client_fails_the_write() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let client = TcpStream::connect(listener.local_addr().expect("address"))
            .await
            .expect("connect");
        let (server, _) = listener.accept().await.expect("accept");
        let mut stream = TrackedStream {
            inner: server,
            pipelined: Arc::new(AtomicBool::new(false)),
            write_stall: None,
            stall_timeout: Duration::from_millis(100),
        };
        // The client never reads, so the socket buffers fill and stay full.
        let chunk = vec![0u8; 1 << 20];
        let error = loop {
            if let Err(error) = stream.write_all(&chunk).await {
                break error;
            }
        };
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        drop(client);
    }
}
