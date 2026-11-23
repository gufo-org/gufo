//! WebSocket session backed by hyper's `Upgraded` stream and `gufo-websocket`'s
//! RFC 6455 frame reader.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use gufo_websocket::{Event, Reader};
use hyper::upgrade::OnUpgrade;
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncReadExt, AsyncWriteExt, WriteHalf};
use tokio::runtime::Handle;
use tokio::sync::{Mutex as AsyncMutex, watch};

const MAX_MESSAGE_BYTES: usize = 4 << 20;
const MAX_QUEUE_BYTES: usize = 8 << 20;
const SEND_TIMEOUT: Duration = Duration::from_secs(5);
const IDLE_TIMEOUT: Duration = Duration::from_secs(120);

const OPCODE_TEXT: u8 = 0x01;
const OPCODE_BINARY: u8 = 0x02;
const OPCODE_CLOSE: u8 = 0x08;
const OPCODE_PONG: u8 = 0x0A;

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

/// Opaque WebSocket session passed to C++ `WebSocket`.
pub struct GufoWsSession {
    runtime: Handle,
    writer: Arc<AsyncMutex<WriteHalf<TokioIo<hyper::upgrade::Upgraded>>>>,
    queue: Arc<(Mutex<MessageQueue>, Condvar)>,
    closed: Arc<AtomicBool>,
    epoch: Instant,
    last_send_ms: Arc<AtomicU64>,
}

impl GufoWsSession {
    pub(crate) fn receive(&self) -> Option<Vec<u8>> {
        let (lock, cvar) = &*self.queue;
        let mut guard = lock.lock().ok()?;
        while guard.messages.is_empty() && !guard.closed {
            guard = cvar.wait(guard).ok()?;
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
        if self.mark_closed() {
            let payload = code.to_be_bytes();
            let _ = self.write_raw_frame(OPCODE_CLOSE, &payload);
        }
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn mark_closed(&self) -> bool {
        let was_open = !self.closed.swap(true, Ordering::AcqRel);
        if was_open {
            let (lock, cvar) = &*self.queue;
            if let Ok(mut guard) = lock.lock() {
                guard.closed = true;
            }
            cvar.notify_all();
        }
        was_open
    }

    fn send_frame(&self, opcode: u8, payload: &[u8]) -> bool {
        if self.closed.load(Ordering::Acquire) {
            return false;
        }
        let ok = self.write_raw_frame(opcode, payload);
        if ok {
            let elapsed = self.epoch.elapsed().as_millis() as u64;
            self.last_send_ms.store(elapsed, Ordering::Release);
        } else {
            self.mark_closed();
        }
        ok
    }

    fn write_raw_frame(&self, opcode: u8, payload: &[u8]) -> bool {
        let header = encode_frame_header(opcode, payload.len());
        let writer = Arc::clone(&self.writer);
        let payload_copy = payload.to_vec();
        self.runtime.block_on(async move {
            let mut guard = writer.lock().await;
            let write_fut = async {
                guard.write_all(&header).await?;
                if !payload_copy.is_empty() {
                    guard.write_all(&payload_copy).await?;
                }
                guard.flush().await
            };
            matches!(
                tokio::time::timeout(SEND_TIMEOUT, write_fut).await,
                Ok(Ok(()))
            )
        })
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

pub(crate) async fn run_websocket_upgrade(
    on_upgrade: OnUpgrade,
    runtime: Handle,
    mut shutdown: watch::Receiver<bool>,
    handler: WsHandlerCallback,
) {
    let Ok(upgraded) = on_upgrade.await else {
        // Notify C++ with null session so it can free `ws_context`.
        handler.call(std::ptr::null_mut());
        return;
    };
    let io = TokioIo::new(upgraded);
    let (mut read_half, write_half) = tokio::io::split(io);

    let writer = Arc::new(AsyncMutex::new(write_half));
    let queue = Arc::new((
        Mutex::new(MessageQueue {
            messages: VecDeque::new(),
            queued_bytes: 0,
            closed: false,
        }),
        Condvar::new(),
    ));
    let closed = Arc::new(AtomicBool::new(false));
    let epoch = Instant::now();
    let last_send_ms = Arc::new(AtomicU64::new(0));

    let reader_runtime = runtime.clone();
    let reader_writer = Arc::clone(&writer);
    let reader_queue = Arc::clone(&queue);
    let reader_closed = Arc::clone(&closed);
    let reader_last_send = Arc::clone(&last_send_ms);

    let reader_task = runtime.spawn_blocking(move || {
        let mut frame_reader = Reader::new(MAX_MESSAGE_BYTES);
        let send_control = |opcode: u8, payload: &[u8]| {
            let header = encode_frame_header(opcode, payload.len());
            let w = Arc::clone(&reader_writer);
            let p = payload.to_vec();
            reader_runtime.block_on(async move {
                let mut guard = w.lock().await;
                let fut = async {
                    guard.write_all(&header).await?;
                    if !p.is_empty() {
                        guard.write_all(&p).await?;
                    }
                    guard.flush().await
                };
                let _ = tokio::time::timeout(SEND_TIMEOUT, fut).await;
            });
        };

        let mut timed_out = false;
        loop {
            if reader_closed.load(Ordering::Acquire) {
                break;
            }
            let event = frame_reader.next(&mut |buf: &mut [u8]| {
                let mut offset = 0;
                while offset < buf.len() {
                    if reader_closed.load(Ordering::Acquire) {
                        return false;
                    }
                    let res = reader_runtime.block_on(async {
                        tokio::time::timeout(
                            Duration::from_millis(500),
                            read_half.read(&mut buf[offset..]),
                        )
                        .await
                    });
                    match res {
                        Ok(Ok(0)) => return false,
                        Ok(Ok(n)) => offset += n,
                        Ok(Err(_)) => return false,
                        Err(_) => {
                            let now_ms = epoch.elapsed().as_millis() as u64;
                            let last = reader_last_send.load(Ordering::Acquire);
                            if now_ms.saturating_sub(last) >= IDLE_TIMEOUT.as_millis() as u64 {
                                timed_out = true;
                                return false;
                            }
                        }
                    }
                }
                true
            });

            match event {
                Event::End => {
                    if timed_out {
                        send_control(OPCODE_CLOSE, &1001u16.to_be_bytes());
                    }
                    break;
                }
                Event::Message(msg) => {
                    let now_ms = epoch.elapsed().as_millis() as u64;
                    reader_last_send.store(now_ms, Ordering::Release);
                    let (lock, cvar) = &*reader_queue;
                    let Ok(mut guard) = lock.lock() else {
                        break;
                    };
                    if guard.queued_bytes.saturating_add(msg.len()) > MAX_QUEUE_BYTES {
                        drop(guard);
                        send_control(OPCODE_CLOSE, &1009u16.to_be_bytes());
                        break;
                    }
                    guard.queued_bytes += msg.len();
                    guard.messages.push_back(msg.to_vec());
                    cvar.notify_one();
                }
                Event::Ping(payload) => {
                    send_control(OPCODE_PONG, payload);
                }
                Event::Close(payload) => {
                    send_control(OPCODE_CLOSE, payload);
                    break;
                }
                Event::Error(code) => {
                    send_control(OPCODE_CLOSE, &code.to_be_bytes());
                    break;
                }
            }
        }

        reader_closed.store(true, Ordering::Release);
        let (lock, cvar) = &*reader_queue;
        if let Ok(mut guard) = lock.lock() {
            guard.closed = true;
        }
        cvar.notify_all();
    });

    let mut session = Box::new(GufoWsSession {
        runtime: runtime.clone(),
        writer: Arc::clone(&writer),
        queue: Arc::clone(&queue),
        closed: Arc::clone(&closed),
        epoch,
        last_send_ms,
    });

    let mut handler_task = runtime.spawn_blocking(move || {
        handler.call(&mut *session);
        session.close(1000);
    });

    tokio::select! {
        _ = &mut handler_task => {}
        _ = shutdown.changed() => {
            closed.store(true, Ordering::Release);
            let (lock, cvar) = &*queue;
            if let Ok(mut guard) = lock.lock() {
                guard.closed = true;
            }
            cvar.notify_all();
            let _ = handler_task.await;
        }
    }
    closed.store(true, Ordering::Release);
    let _ = reader_task.await;
}
