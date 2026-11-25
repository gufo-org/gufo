//! Bridge between hyper's async request/response pipeline and C++ handlers
//! running on blocking threads.

use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode};
use hyper::body::{Body, Frame, SizeHint};
use hyper::upgrade::OnUpgrade;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::{Instant, Sleep};

use crate::websocket::{IDLE_TIMEOUT, WsHandlerCallback, run_websocket_session};

/// Bounded channel capacity for streaming responses. Keeps memory bounded while
/// letting the producer run ahead of socket flushes by a few chunks.
const STREAM_CHANNEL_CAPACITY: usize = 4;

const SSE_PING: &[u8] = b": ping\n\n";

/// Shared cancellation flag observed by C++ handlers via `is_cancelled`.
pub struct GufoHttpCancel {
    cancelled: Arc<AtomicBool>,
}

impl GufoHttpCancel {
    pub(crate) fn new(cancelled: Arc<AtomicBool>) -> Self {
        Self { cancelled }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Optional callback invoked when the Rust stream emits an SSE heartbeat ping
/// (used by `--trace` to record the ping bytes).
pub(crate) struct HeartbeatCallback {
    func: extern "C" fn(*mut std::ffi::c_void, *const u8, usize),
    ctx: usize,
}

impl HeartbeatCallback {
    pub(crate) fn new(
        func: Option<extern "C" fn(*mut std::ffi::c_void, *const u8, usize)>,
        ctx: *mut std::ffi::c_void,
    ) -> Option<Self> {
        func.map(|f| Self {
            func: f,
            ctx: ctx as usize,
        })
    }

    fn notify(&self, bytes: &[u8]) {
        (self.func)(
            self.ctx as *mut std::ffi::c_void,
            bytes.as_ptr(),
            bytes.len(),
        );
    }
}

/// Error returned from `ResponseBody::poll_frame` when a streaming response
/// fails mid-stream and must remain unterminated (e.g. raw PCM audio).
#[derive(Debug)]
pub(crate) struct StreamAbortError;

impl fmt::Display for StreamAbortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("stream aborted before completion")
    }
}

impl std::error::Error for StreamAbortError {}

pub(crate) enum StreamItem {
    Chunk(Bytes),
    /// `true` = normal EOF (send terminating chunk); `false` = abort without
    /// terminating chunk.
    End(bool),
}

pub(crate) enum ResponseBody {
    Empty,
    Full(Option<Bytes>),
    Stream {
        rx: mpsc::Receiver<StreamItem>,
        heartbeat: Option<Pin<Box<Sleep>>>,
        heartbeat_interval: Duration,
        on_heartbeat: Arc<Mutex<Option<HeartbeatCallback>>>,
        cancel: Arc<AtomicBool>,
        done: bool,
        yielded_chunk: bool,
        pending_abort: bool,
    },
}

impl Drop for ResponseBody {
    fn drop(&mut self) {
        if let Self::Stream { cancel, done, .. } = self {
            if !*done {
                cancel.store(true, Ordering::Release);
            }
        }
    }
}

impl Body for ResponseBody {
    type Data = Bytes;
    type Error = StreamAbortError;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        match this {
            Self::Empty => Poll::Ready(None),
            Self::Full(opt) => match opt.take() {
                Some(bytes) if !bytes.is_empty() => Poll::Ready(Some(Ok(Frame::data(bytes)))),
                _ => Poll::Ready(None),
            },
            Self::Stream {
                rx,
                heartbeat,
                heartbeat_interval,
                on_heartbeat,
                done,
                yielded_chunk,
                pending_abort,
                ..
            } => {
                if *done {
                    return Poll::Ready(None);
                }
                if *pending_abort {
                    *done = true;
                    return Poll::Ready(Some(Err(StreamAbortError)));
                }
                match rx.poll_recv(cx) {
                    Poll::Ready(Some(StreamItem::Chunk(bytes))) => {
                        if let Some(sleep) = heartbeat.as_mut() {
                            sleep.as_mut().reset(Instant::now() + *heartbeat_interval);
                        }
                        if bytes.is_empty() {
                            cx.waker().wake_by_ref();
                            return Poll::Pending;
                        }
                        *yielded_chunk = true;
                        Poll::Ready(Some(Ok(Frame::data(bytes))))
                    }
                    Poll::Ready(Some(StreamItem::End(complete))) => {
                        if complete {
                            *done = true;
                            Poll::Ready(None)
                        } else if *yielded_chunk {
                            *yielded_chunk = false;
                            *pending_abort = true;
                            cx.waker().wake_by_ref();
                            Poll::Pending
                        } else {
                            *done = true;
                            Poll::Ready(Some(Err(StreamAbortError)))
                        }
                    }
                    Poll::Ready(None) => {
                        *done = true;
                        Poll::Ready(None)
                    }
                    Poll::Pending => {
                        if !heartbeat_interval.is_zero() {
                            let sleep = heartbeat.get_or_insert_with(|| {
                                Box::pin(tokio::time::sleep(*heartbeat_interval))
                            });
                            if sleep.as_mut().poll(cx).is_ready() {
                                sleep.as_mut().reset(Instant::now() + *heartbeat_interval);
                                if let Ok(guard) = on_heartbeat.lock() {
                                    if let Some(cb) = guard.as_ref() {
                                        cb.notify(SSE_PING);
                                    }
                                }
                                return Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(
                                    SSE_PING,
                                )))));
                            }
                        }
                        Poll::Pending
                    }
                }
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::Full(opt) => opt.as_ref().is_none_or(Bytes::is_empty),
            Self::Stream { done, .. } => *done,
        }
    }

    fn size_hint(&self) -> SizeHint {
        match self {
            Self::Empty => SizeHint::with_exact(0),
            Self::Full(Some(bytes)) => SizeHint::with_exact(bytes.len() as u64),
            Self::Full(None) => SizeHint::with_exact(0),
            Self::Stream { .. } => SizeHint::default(),
        }
    }
}

/// Per-request responder passed to the C++ handler on a blocking thread.
pub struct GufoHttpResponder {
    tx: Option<oneshot::Sender<Response<ResponseBody>>>,
    upgrade: Option<OnUpgrade>,
    runtime: Handle,
    shutdown: watch::Receiver<bool>,
    cancel: Arc<AtomicBool>,
    is_head: bool,
    force_close: bool,
    sse_heartbeat_interval: Duration,
}

impl GufoHttpResponder {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        tx: oneshot::Sender<Response<ResponseBody>>,
        upgrade: Option<OnUpgrade>,
        runtime: Handle,
        shutdown: watch::Receiver<bool>,
        cancel: Arc<AtomicBool>,
        is_head: bool,
        force_close: bool,
        sse_heartbeat_interval: Duration,
    ) -> Self {
        Self {
            tx: Some(tx),
            upgrade,
            runtime,
            shutdown,
            cancel,
            is_head,
            force_close,
            sse_heartbeat_interval,
        }
    }

    pub(crate) fn has_responded(&self) -> bool {
        self.tx.is_none()
    }

    pub(crate) fn respond_buffered(
        &mut self,
        status: u16,
        headers: Vec<(HeaderName, HeaderValue)>,
        body: Bytes,
    ) -> bool {
        let Some(tx) = self.tx.take() else {
            return false;
        };
        let status_code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut builder = Response::builder().status(status_code);
        if let Some(map) = builder.headers_mut() {
            apply_headers(map, headers);
            if status_code != StatusCode::NO_CONTENT
                && status_code != StatusCode::NOT_MODIFIED
                && !status_code.is_informational()
                && !map.contains_key(http::header::CONTENT_LENGTH)
            {
                if let Ok(val) = HeaderValue::from_str(&body.len().to_string()) {
                    map.insert(http::header::CONTENT_LENGTH, val);
                }
            }
            if !map.contains_key(http::header::CONNECTION) {
                map.insert(
                    http::header::CONNECTION,
                    if self.force_close {
                        HeaderValue::from_static("close")
                    } else {
                        HeaderValue::from_static("keep-alive")
                    },
                );
            }
        }
        let resp_body = if self.is_head
            || status_code == StatusCode::NO_CONTENT
            || status_code == StatusCode::NOT_MODIFIED
            || status_code.is_informational()
        {
            ResponseBody::Empty
        } else {
            ResponseBody::Full(Some(body))
        };
        let Ok(response) = builder.body(resp_body) else {
            return false;
        };
        tx.send(response).is_ok()
    }

    pub(crate) fn respond_stream_begin(
        &mut self,
        status: u16,
        headers: Vec<(HeaderName, HeaderValue)>,
        is_sse: bool,
        defer_headers: bool,
        on_heartbeat: Option<HeartbeatCallback>,
    ) -> Option<GufoHttpStream> {
        let tx = self.tx.take()?;
        let (chunk_tx, chunk_rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
        let heartbeat_interval = if is_sse && !self.sse_heartbeat_interval.is_zero() {
            self.sse_heartbeat_interval
        } else {
            Duration::ZERO
        };

        let on_heartbeat = Arc::new(Mutex::new(on_heartbeat));
        let pending_head = PendingStreamHead {
            status,
            headers,
            force_close: self.force_close || self.is_head,
            chunk_rx: Some(chunk_rx),
            heartbeat_interval,
            on_heartbeat: Arc::clone(&on_heartbeat),
            cancel: Arc::clone(&self.cancel),
        };

        let mut stream = GufoHttpStream {
            responder_tx: Some(tx),
            pending_head: Some(pending_head),
            chunk_tx,
            on_heartbeat,
            started: false,
            connected: true,
        };

        if !defer_headers {
            stream.commit_headers();
        }
        Some(stream)
    }

    pub(crate) fn restore_tx(&mut self, tx: oneshot::Sender<Response<ResponseBody>>) {
        self.tx = Some(tx);
    }

    /// Sends the upgrade response, then runs the session on this handler
    /// thread once hyper hands over the connection. Returns after the session
    /// ends, so the request is logged, traced and counted for its whole
    /// duration.
    pub(crate) fn respond_websocket(
        &mut self,
        status: u16,
        headers: Vec<(HeaderName, HeaderValue)>,
        ws_handler: WsHandlerCallback,
    ) -> bool {
        let Some(tx) = self.tx.take() else {
            return false;
        };
        let Some(on_upgrade) = self.upgrade.take() else {
            return false;
        };
        let status_code = StatusCode::from_u16(status).unwrap_or(StatusCode::SWITCHING_PROTOCOLS);
        let mut builder = Response::builder().status(status_code);
        if let Some(map) = builder.headers_mut() {
            apply_headers(map, headers);
        }
        let Ok(response) = builder.body(ResponseBody::Empty) else {
            return false;
        };
        if tx.send(response).is_err() {
            return false;
        }
        let Ok(upgraded) = self.runtime.block_on(on_upgrade) else {
            return false;
        };
        run_websocket_session(
            upgraded,
            &self.runtime,
            self.shutdown.clone(),
            ws_handler,
            IDLE_TIMEOUT,
        );
        true
    }
}

struct PendingStreamHead {
    status: u16,
    headers: Vec<(HeaderName, HeaderValue)>,
    force_close: bool,
    chunk_rx: Option<mpsc::Receiver<StreamItem>>,
    heartbeat_interval: Duration,
    on_heartbeat: Arc<Mutex<Option<HeartbeatCallback>>>,
    cancel: Arc<AtomicBool>,
}

/// Active streaming response handle owned by the C++ handler during
/// `streaming_body`.
pub struct GufoHttpStream {
    responder_tx: Option<oneshot::Sender<Response<ResponseBody>>>,
    pending_head: Option<PendingStreamHead>,
    chunk_tx: mpsc::Sender<StreamItem>,
    on_heartbeat: Arc<Mutex<Option<HeartbeatCallback>>>,
    started: bool,
    connected: bool,
}

impl Drop for GufoHttpStream {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.on_heartbeat.lock() {
            *guard = None;
        }
    }
}

impl GufoHttpStream {
    fn commit_headers(&mut self) {
        if self.started {
            return;
        }
        self.started = true;
        let (Some(tx), Some(mut head)) = (self.responder_tx.take(), self.pending_head.take())
        else {
            self.connected = false;
            return;
        };
        let Some(chunk_rx) = head.chunk_rx.take() else {
            self.connected = false;
            return;
        };
        let status_code =
            StatusCode::from_u16(head.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut builder = Response::builder().status(status_code);
        if let Some(map) = builder.headers_mut() {
            apply_headers(map, head.headers);
            if !map.contains_key(http::header::CONNECTION) {
                map.insert(
                    http::header::CONNECTION,
                    if head.force_close {
                        HeaderValue::from_static("close")
                    } else {
                        HeaderValue::from_static("keep-alive")
                    },
                );
            }
        }
        let body = ResponseBody::Stream {
            rx: chunk_rx,
            heartbeat: None,
            heartbeat_interval: head.heartbeat_interval,
            on_heartbeat: head.on_heartbeat,
            cancel: head.cancel,
            done: false,
            yielded_chunk: false,
            pending_abort: false,
        };
        let Ok(response) = builder.body(body) else {
            self.connected = false;
            return;
        };
        if tx.send(response).is_err() {
            self.connected = false;
        }
    }

    pub(crate) fn write(&mut self, data: &[u8]) -> bool {
        if !self.connected {
            return false;
        }
        if !self.started {
            self.commit_headers();
            if !self.connected {
                return false;
            }
        }
        if data.is_empty() {
            return true;
        }
        const MAX_CHUNK_BYTES: usize = 64 * 1024;
        for piece in data.chunks(MAX_CHUNK_BYTES) {
            if self
                .chunk_tx
                .blocking_send(StreamItem::Chunk(Bytes::copy_from_slice(piece)))
                .is_err()
            {
                self.connected = false;
                return false;
            }
        }
        true
    }

    pub(crate) fn finish(mut self, complete: bool) -> bool {
        if !self.started && complete {
            self.commit_headers();
        }
        if self.connected
            && self.started
            && self
                .chunk_tx
                .blocking_send(StreamItem::End(complete))
                .is_err()
        {
            self.connected = false;
        }
        self.connected
    }

    pub(crate) fn abort_deferred(mut self, responder: &mut GufoHttpResponder) {
        if !self.started {
            if let Some(tx) = self.responder_tx.take() {
                responder.restore_tx(tx);
            }
        }
    }
}

fn apply_headers(map: &mut HeaderMap, headers: Vec<(HeaderName, HeaderValue)>) {
    for (name, value) in headers {
        map.append(name, value);
    }
}
