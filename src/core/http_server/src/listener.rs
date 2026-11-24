//! TCP listener, tokio runtime, accept loop, connection limit enforcement,
//! periodic device-loss polling, and graceful shutdown.

use std::ffi::c_void;
use std::net::TcpListener as StdTcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::net::TcpListener;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::watch;

use crate::connection::{
    ConnSlot, ConnectionContext, ConnectionOptions, ConnectionTracker, reject_overloaded,
    serve_connection,
};
use crate::platform::{bind_tcp_listener, is_addr_in_use, parse_listen_address};

pub(crate) struct DeviceLostCallback {
    pub(crate) func: extern "C" fn(*mut c_void) -> i32,
    pub(crate) ctx: usize,
}

impl DeviceLostCallback {
    fn check(&self) -> bool {
        (self.func)(self.ctx as *mut c_void) != 0
    }
}

pub struct GufoHttpServer {
    runtime: Runtime,
    listener: Mutex<Option<StdTcpListener>>,
    port: u16,
    options: Arc<ConnectionOptions>,
    device_lost: Option<Arc<DeviceLostCallback>>,
    stopped: Arc<AtomicBool>,
    shutdown_tx: watch::Sender<bool>,
    tracker: Arc<ConnectionTracker>,
}

impl GufoHttpServer {
    pub(crate) fn start(
        host: &str,
        port: i32,
        options: ConnectionOptions,
        device_lost: Option<DeviceLostCallback>,
    ) -> Result<Self, String> {
        let port_u16 = u16::try_from(port).map_err(|_| {
            "host must be an IPv4 or IPv6 address and port must be between 0 and 65535".to_string()
        })?;
        let addr = parse_listen_address(host, port_u16).ok_or_else(|| {
            "host must be an IPv4 or IPv6 address and port must be between 0 and 65535".to_string()
        })?;
        let std_listener = bind_tcp_listener(addr, 128).map_err(|err| {
            let mut msg = format!("bind() failed on {host}:{port_u16} ({err})");
            if is_addr_in_use(&err) {
                msg.push_str(": the port is already in use, another server (gufo?) listens on it");
            }
            msg
        })?;
        let bound_port = std_listener
            .local_addr()
            .map_err(|err| format!("getsockname() failed: {err}"))?
            .port();

        let runtime = Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("gufo-http")
            .enable_io()
            .enable_time()
            .build()
            .map_err(|err| format!("failed to start tokio runtime: {err}"))?;

        let (shutdown_tx, _) = watch::channel(false);

        Ok(Self {
            runtime,
            listener: Mutex::new(Some(std_listener)),
            port: bound_port,
            options: Arc::new(options),
            device_lost: device_lost.map(Arc::new),
            stopped: Arc::new(AtomicBool::new(false)),
            shutdown_tx,
            tracker: Arc::new(ConnectionTracker::new()),
        })
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn run(&self) -> Result<(), String> {
        let std_listener = {
            let mut guard = self
                .listener
                .lock()
                .map_err(|_| "listener lock poisoned".to_string())?;
            guard
                .take()
                .ok_or_else(|| "HTTP server has already been run".to_string())?
        };

        let stopped = Arc::clone(&self.stopped);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let tracker = Arc::clone(&self.tracker);
        let options = Arc::clone(&self.options);
        let device_lost = self.device_lost.clone();
        let handle = self.runtime.handle().clone();

        self.runtime.block_on(async move {
            let listener = TcpListener::from_std(std_listener)
                .map_err(|err| format!("configure HTTP listener: {err}"))?;

            let mut device_tick = tokio::time::interval(Duration::from_millis(100));
            device_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            while !*shutdown_rx.borrow() && !stopped.load(Ordering::Acquire) {
                tokio::select! {
                    _ = shutdown_rx.changed() => {
                        break;
                    }
                    _ = device_tick.tick() => {
                        if let Some(cb) = device_lost.as_ref() {
                            let _ = cb.check();
                        }
                    }
                    accepted = listener.accept() => {
                        let Ok((stream, peer_addr)) = accepted else {
                            if stopped.load(Ordering::Acquire) {
                                break;
                            }
                            continue;
                        };
                        if stopped.load(Ordering::Acquire) {
                            break;
                        }
                        match tracker.try_admit(options.max_connections) {
                            Some((conn_id, evict)) => {
                                let ctx = ConnectionContext {
                                    slot: Arc::new(ConnSlot::new(Arc::clone(&tracker), conn_id)),
                                    evict,
                                    shutdown: shutdown_rx.clone(),
                                    options: Arc::clone(&options),
                                    runtime: handle.clone(),
                                };
                                tokio::spawn(async move {
                                    serve_connection(stream, Some(peer_addr), ctx).await;
                                });
                            }
                            None => {
                                tokio::spawn(reject_overloaded(stream));
                            }
                        }
                    }
                }
            }
            // Like the joins of the C++ transport, return only once every
            // connection and the handlers it started have finished, so no
            // handler or device-loss hook outlives `run`.
            drop(listener);
            while !tracker.is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Ok(())
        })
    }

    pub(crate) fn stop(&self) {
        // Every connection task watches the shutdown flag and closes at once.
        if !self.stopped.swap(true, Ordering::AcqRel) {
            let _ = self.shutdown_tx.send(true);
        }
    }
}

impl Drop for GufoHttpServer {
    fn drop(&mut self) {
        self.stop();
    }
}
