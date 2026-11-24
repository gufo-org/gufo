//! Best-effort 503 loading responder that holds the listen port while a model
//! loads and reports load progress to health pollers.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use bytes::Bytes;
use http::{HeaderValue, Request, Response, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::runtime::Builder;
use tokio::sync::watch;

use crate::platform::{bind_tcp_listener, parse_listen_address};

struct LoadingState {
    progress: f64,
    message: String,
}

impl LoadingState {
    fn reply_json(&self) -> String {
        let escaped = escape_json(&self.message);
        format!(
            r#"{{"error":{{"message":"Loading model","type":"unavailable_error","code":"loading"}},"progress":{},"message":"{}"}}"#,
            self.progress, escaped
        )
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

pub struct GufoLoadingResponder {
    state: Arc<Mutex<LoadingState>>,
    shutdown_tx: watch::Sender<bool>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl GufoLoadingResponder {
    pub(crate) fn start(host: &str, port: i32) -> Option<Self> {
        if port <= 0 || port > 65535 {
            return None;
        }
        let addr = parse_listen_address(host, port as u16)?;
        let std_listener = bind_tcp_listener(addr, 16).ok()?;

        let state = Arc::new(Mutex::new(LoadingState {
            progress: 0.0,
            message: "Loading model".to_string(),
        }));
        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

        let state_bg = Arc::clone(&state);

        let thread = std::thread::Builder::new()
            .name("gufo-loading".to_string())
            .spawn(move || {
                let Ok(rt) = Builder::new_current_thread()
                    .enable_io()
                    .enable_time()
                    .build()
                else {
                    return;
                };
                rt.block_on(async move {
                    let Ok(listener) = TcpListener::from_std(std_listener) else {
                        return;
                    };
                    while !*shutdown_rx.borrow() {
                        tokio::select! {
                            _ = shutdown_rx.changed() => break,
                            accepted = listener.accept() => {
                                let Ok((stream, _)) = accepted else {
                                    continue;
                                };
                                let state_req = Arc::clone(&state_bg);
                                tokio::spawn(async move {
                                    let _ = stream.set_nodelay(true);
                                    let io = TokioIo::new(stream);
                                    let svc = service_fn(move |req: Request<Incoming>| {
                                        let body = state_req
                                            .lock()
                                            .map(|g| g.reply_json())
                                            .unwrap_or_else(|_| r#"{"error":{"message":"Loading model","type":"unavailable_error","code":"loading"},"progress":0,"message":"Loading model"}"#.to_string());
                                        async move {
                                            let _ = tokio::time::timeout(
                                                Duration::from_millis(500),
                                                req.into_body().collect(),
                                            )
                                            .await;
                                            let resp = Response::builder()
                                                .status(StatusCode::SERVICE_UNAVAILABLE)
                                                .header(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"))
                                                .header(http::header::RETRY_AFTER, HeaderValue::from_static("1"))
                                                .header(http::header::CONTENT_LENGTH, body.len().to_string())
                                                .header(http::header::CONNECTION, HeaderValue::from_static("close"))
                                                .body(Full::new(Bytes::from(body)))
                                                .unwrap_or_else(|_| Response::new(Full::new(Bytes::new())));
                                            Ok::<_, std::convert::Infallible>(resp)
                                        }
                                    });
                                    let _ = tokio::time::timeout(
                                        Duration::from_secs(2),
                                        http1::Builder::new()
                                            .keep_alive(false)
                                            .title_case_headers(true)
                                            .serve_connection(io, svc),
                                    )
                                    .await;
                                });
                            }
                        }
                    }
                });
            })
            .ok()?;

        Some(Self {
            state,
            shutdown_tx,
            thread: Mutex::new(Some(thread)),
        })
    }

    pub(crate) fn set_progress(&self, progress: f64, message: &str) {
        let clamped = if progress.is_nan() {
            0.0
        } else {
            progress.clamp(0.0, 1.0)
        };
        if let Ok(mut guard) = self.state.lock() {
            guard.progress = clamped;
            if !message.is_empty() {
                guard.message = message.to_string();
            }
        }
    }

    pub(crate) fn stop(&self) {
        let _ = self.shutdown_tx.send(true);
        if let Ok(mut guard) = self.thread.lock() {
            if let Some(handle) = guard.take() {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for GufoLoadingResponder {
    fn drop(&mut self) {
        self.stop();
    }
}
