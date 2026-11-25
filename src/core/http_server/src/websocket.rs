//! WebSocket session backed by hyper's `Upgraded` stream and `gufo-websocket`'s
//! RFC 6455 frame reader.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use gufo_websocket::{Event, Reader};
use hyper::upgrade::Upgraded;
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncReadExt, AsyncWriteExt, WriteHalf};
use tokio::runtime::Handle;
use tokio::sync::{Mutex as AsyncMutex, Notify, watch};

const MAX_MESSAGE_BYTES: usize = 4 << 20;
const MAX_QUEUE_BYTES: usize = 8 << 20;
/// Bounds the queue of small messages too, which the byte budget alone allows
/// to grow without limit (empty frames cost no bytes).
const MAX_QUEUE_MESSAGES: usize = 128;
const SEND_TIMEOUT: Duration = Duration::from_secs(5);
/// A session where neither side sent anything for this long closes with 1001,
/// like the socket receive timeout the C++ transport used.
pub(crate) const IDLE_TIMEOUT: Duration = Duration::from_secs(120);
/// Unread client bytes are drained this long before the socket closes: on
/// Windows, closing with unread data aborts the connection with RST and
/// discards the close frame just sent.
const CLOSE_DRAIN: Duration = Duration::from_millis(200);
/// How often a waiting reader checks the idle timeout.
const IDLE_POLL: Duration = Duration::from_millis(500);

const OPCODE_TEXT: u8 = 0x01;
const OPCODE_BINARY: u8 = 0x02;
const OPCODE_CLOSE: u8 = 0x08;
const OPCODE_PONG: u8 = 0x0A;

type Writer = Arc<AsyncMutex<WriteHalf<TokioIo<Upgraded>>>>;

pub(crate) struct WsHandlerCallback {
    func: extern "C" fn(*mut std::ffi::c_void, *mut GufoWsSession),
    ctx: usize,
}

impl WsHandlerCallback {
    pub(crate) fn new(
        func: extern "C" fn(*mut std::ffi::c_void, *mut GufoWsSession),
        ctx: *mut std::ffi::c_void,
    ) -> Self {
        Self {
            func,
            ctx: ctx as usize,
        }
    }

    fn call(self, session: *mut GufoWsSession) {
        (self.func)(self.ctx as *mut std::ffi::c_void, session);
    }
}

struct MessageQueue {
    messages: VecDeque<Vec<u8>>,
    queued_bytes: usize,
    closed: bool,
}

/// State shared by the session, its frame reader and the shutdown watcher.
struct Shared {
    queue: Mutex<MessageQueue>,
    ready: Condvar,
    closed: AtomicBool,
    /// Wakes a reader waiting for client bytes once the session closes.
    reader_wake: Notify,
    epoch: Instant,
    last_activity_ms: AtomicU64,
    idle_timeout: Duration,
}

impl Shared {
    /// Marks the session closed; true for the call that closed it.
    fn mark_closed(&self) -> bool {
        let was_open = {
            // `receive` checks the flag under this lock, so a close cannot
            // slip between its check and its wait.
            let mut guard = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            guard.closed = true;
            !self.closed.swap(true, Ordering::AcqRel)
        };
        self.ready.notify_all();
        self.reader_wake.notify_one();
        was_open
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn touch(&self) {
        let elapsed = self.epoch.elapsed().as_millis() as u64;
        self.last_activity_ms.store(elapsed, Ordering::Release);
    }

    fn idle(&self) -> bool {
        let now_ms = self.epoch.elapsed().as_millis() as u64;
        let last = self.last_activity_ms.load(Ordering::Acquire);
        now_ms.saturating_sub(last) >= self.idle_timeout.as_millis() as u64
    }
}

/// Opaque WebSocket session passed to C++ `WebSocket`.
pub struct GufoWsSession {
    runtime: Handle,
    writer: Writer,
    shared: Arc<Shared>,
}

impl GufoWsSession {
    /// Waits for the next client message. Returns `None` once the session has
    /// closed, even with messages still queued, as the C++ transport did.
    pub(crate) fn receive(&self) -> Option<Vec<u8>> {
        let mut guard = self.shared.queue.lock().ok()?;
        while guard.messages.is_empty() && !guard.closed {
            guard = self.shared.ready.wait(guard).ok()?;
        }
        if guard.closed {
            return None;
        }
        let msg = guard.messages.pop_front()?;
        guard.queued_bytes = guard.queued_bytes.saturating_sub(msg.len());
        Some(msg)
    }

    pub(crate) fn send_text(&self, data: &[u8]) -> bool {
        self.send_frame(OPCODE_TEXT, data)
    }

    pub(crate) fn send_binary(&self, data: &[u8]) -> bool {
        self.send_frame(OPCODE_BINARY, data)
    }

    pub(crate) fn close(&self, code: u16) {
        if self.shared.mark_closed() {
            let _ = write_frame(
                &self.runtime,
                &self.writer,
                OPCODE_CLOSE,
                &code.to_be_bytes(),
            );
        }
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.shared.is_closed()
    }

    fn send_frame(&self, opcode: u8, payload: &[u8]) -> bool {
        if self.shared.is_closed() {
            return false;
        }
        let ok = write_frame(&self.runtime, &self.writer, opcode, payload);
        if ok {
            self.shared.touch();
        } else {
            self.shared.mark_closed();
        }
        ok
    }
}

fn encode_frame_header(opcode: u8, len: usize) -> Vec<u8> {
    let mut header = Vec::with_capacity(10);
    header.push(0x80 | (opcode & 0x0F));
    if len < 126 {
        header.push(len as u8);
    } else if len <= u16::MAX as usize {
        header.push(126);
        header.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        header.push(127);
        header.extend_from_slice(&(len as u64).to_be_bytes());
    }
    header
}

/// Writes one unmasked final frame from a blocking thread. A client that
/// stops reading fails the write after `SEND_TIMEOUT`.
fn write_frame(runtime: &Handle, writer: &Writer, opcode: u8, payload: &[u8]) -> bool {
    let header = encode_frame_header(opcode, payload.len());
    runtime.block_on(async {
        let mut guard = writer.lock().await;
        let write = async {
            guard.write_all(&header).await?;
            if !payload.is_empty() {
                guard.write_all(payload).await?;
            }
            guard.flush().await
        };
        matches!(tokio::time::timeout(SEND_TIMEOUT, write).await, Ok(Ok(())))
    })
}

/// Sends a close frame with `code` if this call closes the session.
fn close_with(runtime: &Handle, writer: &Writer, shared: &Shared, payload: &[u8]) {
    if shared.mark_closed() {
        let _ = write_frame(runtime, writer, OPCODE_CLOSE, payload);
    }
}

/// Runs one WebSocket session on the calling blocking thread: a reader thread
/// parses client frames into the message queue while `handler` consumes it.
/// Returns once the handler has returned, the reader has stopped and the
/// socket has closed.
pub(crate) fn run_websocket_session(
    upgraded: Upgraded,
    runtime: &Handle,
    shutdown: watch::Receiver<bool>,
    handler: WsHandlerCallback,
    idle_timeout: Duration,
) {
    let (mut read_half, write_half) = tokio::io::split(TokioIo::new(upgraded));
    let writer: Writer = Arc::new(AsyncMutex::new(write_half));
    let shared = Arc::new(Shared {
        queue: Mutex::new(MessageQueue {
            messages: VecDeque::new(),
            queued_bytes: 0,
            closed: false,
        }),
        ready: Condvar::new(),
        closed: AtomicBool::new(false),
        reader_wake: Notify::new(),
        epoch: Instant::now(),
        last_activity_ms: AtomicU64::new(0),
        idle_timeout,
    });

    let reader_runtime = runtime.clone();
    let reader_writer = Arc::clone(&writer);
    let reader_shared = Arc::clone(&shared);
    let reader_task = runtime.spawn_blocking(move || {
        let runtime = reader_runtime;
        let writer = reader_writer;
        let shared = reader_shared;
        let mut frame_reader = Reader::new(MAX_MESSAGE_BYTES);
        let mut timed_out = false;
        while !shared.is_closed() {
            let event = frame_reader.next(&mut |buf: &mut [u8]| {
                let mut offset = 0;
                while offset < buf.len() {
                    if shared.is_closed() {
                        return false;
                    }
                    let res = runtime.block_on(async {
                        tokio::select! {
                            read = read_half.read(&mut buf[offset..]) => Some(read),
                            _ = shared.reader_wake.notified() => None,
                            _ = tokio::time::sleep(IDLE_POLL) => Some(Err(
                                std::io::ErrorKind::WouldBlock.into(),
                            )),
                        }
                    });
                    match res {
                        None | Some(Ok(0)) => return false,
                        Some(Ok(n)) => offset += n,
                        Some(Err(err)) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            // A client that only listens to streamed output
                            // is not idle.
                            if shared.idle() {
                                timed_out = true;
                                return false;
                            }
                        }
                        Some(Err(_)) => return false,
                    }
                }
                true
            });

            // Replies go out only while the session is open, so none follows
            // the server's own close frame.
            match event {
                Event::End => {
                    if timed_out {
                        close_with(&runtime, &writer, &shared, &1001u16.to_be_bytes());
                    }
                    break;
                }
                Event::Message(msg) => {
                    shared.touch();
                    let Ok(mut guard) = shared.queue.lock() else {
                        break;
                    };
                    if guard.closed {
                        break;
                    }
                    let mut owned = Vec::new();
                    let refusal = if guard.messages.len() >= MAX_QUEUE_MESSAGES
                        || guard.queued_bytes.saturating_add(msg.len()) > MAX_QUEUE_BYTES
                    {
                        Some(1009u16)
                    } else if owned.try_reserve_exact(msg.len()).is_err() {
                        Some(1011u16)
                    } else {
                        None
                    };
                    if let Some(code) = refusal {
                        drop(guard);
                        close_with(&runtime, &writer, &shared, &code.to_be_bytes());
                        break;
                    }
                    owned.extend_from_slice(msg);
                    guard.queued_bytes += owned.len();
                    guard.messages.push_back(owned);
                    drop(guard);
                    shared.ready.notify_one();
                }
                Event::Ping(payload) => {
                    if !shared.is_closed() && !write_frame(&runtime, &writer, OPCODE_PONG, payload)
                    {
                        break;
                    }
                }
                Event::Close(payload) => {
                    close_with(&runtime, &writer, &shared, payload);
                    break;
                }
                Event::Error(code) => {
                    close_with(&runtime, &writer, &shared, &code.to_be_bytes());
                    break;
                }
            }
        }
        shared.mark_closed();
        read_half
    });

    // Server shutdown closes the session, which wakes a handler waiting in
    // `receive` and makes its sends fail.
    let watcher_shared = Arc::clone(&shared);
    let mut watcher_shutdown = shutdown;
    let watcher = runtime.spawn(async move {
        let _ = watcher_shutdown.wait_for(|stopped| *stopped).await;
        watcher_shared.mark_closed();
    });

    let mut session = GufoWsSession {
        runtime: runtime.clone(),
        writer: Arc::clone(&writer),
        shared,
    };
    handler.call(&mut session);
    session.close(1000);
    watcher.abort();
    let read_half = runtime.block_on(reader_task).ok();
    runtime.block_on(async {
        if let Some(mut read_half) = read_half {
            let drain = async {
                let mut sink = [0u8; 2048];
                while matches!(read_half.read(&mut sink).await, Ok(n) if n > 0) {}
            };
            let _ = tokio::time::timeout(CLOSE_DRAIN, drain).await;
        }
        let _ = writer.lock().await.shutdown().await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use http::{Request, Response, StatusCode};
    use http_body_util::Empty;
    use hyper::body::Incoming;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use tokio::net::{TcpListener, TcpStream};

    extern "C" fn receive_until_closed(
        _context: *mut std::ffi::c_void,
        session: *mut GufoWsSession,
    ) {
        // SAFETY: `run_websocket_session` passes a live session for the call.
        #[allow(unsafe_code)]
        let session = unsafe { &*session };
        while session.receive().is_some() {}
    }

    async fn read_until(stream: &mut TcpStream, buffer: &mut Vec<u8>, len: usize) {
        while buffer.len() < len {
            let mut chunk = [0u8; 256];
            let n = stream.read(&mut chunk).await.expect("read");
            assert!(n > 0, "connection closed early");
            buffer.extend_from_slice(&chunk[..n]);
        }
    }

    // A session where neither side speaks closes with 1001 after the idle
    // limit, then drains and closes the socket.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn silent_session_closes_after_idle_limit() {
        const IDLE: Duration = Duration::from_millis(300);
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            let service = service_fn(move |mut req: Request<Incoming>| {
                let on_upgrade = hyper::upgrade::on(&mut req);
                let shutdown = shutdown_rx.clone();
                let runtime = Handle::current();
                tokio::task::spawn_blocking(move || {
                    let upgraded = runtime.block_on(on_upgrade).expect("upgrade");
                    let handler =
                        WsHandlerCallback::new(receive_until_closed, std::ptr::null_mut());
                    run_websocket_session(upgraded, &runtime, shutdown, handler, IDLE);
                });
                async {
                    Response::builder()
                        .status(StatusCode::SWITCHING_PROTOCOLS)
                        .header(http::header::UPGRADE, "websocket")
                        .header(http::header::CONNECTION, "Upgrade")
                        .body(Empty::<Bytes>::new())
                }
            });
            let _ = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .with_upgrades()
                .await;
        });

        let mut client = TcpStream::connect(address).await.expect("connect");
        client
            .write_all(
                b"GET / HTTP/1.1\r\nHost: test\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
            )
            .await
            .expect("write");
        let started = Instant::now();
        let mut buffer = Vec::new();
        while !buffer.windows(4).any(|w| w == b"\r\n\r\n") {
            let len = buffer.len() + 1;
            read_until(&mut client, &mut buffer, len).await;
        }
        let head_end = buffer
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("head")
            + 4;
        assert!(buffer.starts_with(b"HTTP/1.1 101"));
        let mut frame = buffer.split_off(head_end);
        read_until(&mut client, &mut frame, 4).await;
        assert_eq!(&frame[..4], b"\x88\x02\x03\xE9");
        assert!(started.elapsed() >= IDLE);
        // The server drains and then sends FIN without waiting for a reply.
        let mut rest = [0u8; 16];
        let n = tokio::time::timeout(Duration::from_secs(2), client.read(&mut rest))
            .await
            .expect("server closes the socket")
            .expect("read");
        assert_eq!(n, 0);
        server.await.expect("server task");
    }
}
