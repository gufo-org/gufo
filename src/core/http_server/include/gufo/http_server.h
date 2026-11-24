#ifndef GUFO_HTTP_SERVER_H_
#define GUFO_HTTP_SERVER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GufoHttpServer GufoHttpServer;
typedef struct GufoHttpResponder GufoHttpResponder;
typedef struct GufoHttpStream GufoHttpStream;
typedef struct GufoHttpCancel GufoHttpCancel;
typedef struct GufoWsSession GufoWsSession;
typedef struct GufoLoadingResponder GufoLoadingResponder;

typedef struct GufoHttpSliceView {
  const uint8_t* data;
  size_t len;
} GufoHttpSliceView;

typedef struct GufoHttpHeaderView {
  GufoHttpSliceView name;
  GufoHttpSliceView value;
} GufoHttpHeaderView;

typedef struct GufoHttpRequestView {
  GufoHttpSliceView method;
  GufoHttpSliceView path;
  GufoHttpSliceView query;
  GufoHttpSliceView body;
  const GufoHttpHeaderView* headers;
  size_t headers_len;
  GufoHttpSliceView peer_addr;
  int http11;
  GufoHttpCancel* cancel;
  /* Nonzero when the transport refused the request before reading it whole:
     the HTTP status (400, 413, 431 or 500) to answer with. The body is then
     empty and the connection closes after the response. */
  int rejection;
} GufoHttpRequestView;

typedef void (*GufoHttpBytesSink)(void* context, const uint8_t* data,
                                  size_t len);

/* Called on a blocking thread for each incoming request.
   Must invoke one of gufo_http_respond_buffered,
   gufo_http_respond_stream_begin, or gufo_http_respond_websocket on responder
   before returning. */
typedef void (*GufoHttpHandlerFn)(void* context,
                                  const GufoHttpRequestView* request,
                                  GufoHttpResponder* responder);

/* Returns nonzero when the device is lost. Polled periodically by the listener
   and after each request. May be NULL. */
typedef int (*GufoHttpDeviceLostFn)(void* context);

/* Called on the request's handler thread once a WebSocket upgrade completes,
   from within gufo_http_respond_websocket. */
typedef void (*GufoWsHandlerFn)(void* context, GufoWsSession* session);

typedef struct GufoHttpServerConfig {
  const uint8_t* host;
  size_t host_len;
  int port;
  size_t max_request_body_bytes;
  size_t max_connections;
  uint64_t sse_heartbeat_interval_ms;
  int keep_alive;
  uint64_t keep_alive_idle_timeout_ms;
  size_t max_requests_per_connection;
  GufoHttpHandlerFn handler;
  void* handler_context;
  GufoHttpDeviceLostFn device_lost;
  void* device_lost_context;
} GufoHttpServerConfig;

/* Returns 1 if host is a valid numeric IPv4 or IPv6 literal (with optional
   %<scope> on IPv6), 0 otherwise. */
int gufo_http_is_listen_address(const uint8_t* host, size_t host_len);

/* Binds and listens. Returns NULL on failure and passes an error message to
   error_sink (if non-NULL). */
GufoHttpServer* gufo_http_server_start(const GufoHttpServerConfig* config,
                                       GufoHttpBytesSink error_sink,
                                       void* error_context);

/* Returns the bound port (useful when config->port == 0). */
int gufo_http_server_port(const GufoHttpServer* server);

/* Blocks serving connections until gufo_http_server_stop is called.
   Returns 0 on normal shutdown, -1 on internal error. */
int gufo_http_server_run(GufoHttpServer* server);

/* Signals the server to stop accepting and shuts down active connections.
   Safe to call from any thread. */
void gufo_http_server_stop(GufoHttpServer* server);

/* Frees the server handle. Calls stop if not already stopped. */
void gufo_http_server_free(GufoHttpServer* server);

/* Returns 1 if the client has disconnected or the request was cancelled. */
int gufo_http_cancel_is_cancelled(const GufoHttpCancel* cancel);

/* Sends a buffered response. Returns 1 if delivered, 0 if disconnected. */
int gufo_http_respond_buffered(GufoHttpResponder* responder, int status,
                               const GufoHttpHeaderView* headers,
                               size_t headers_len, const uint8_t* body,
                               size_t body_len);

/* Begins a streaming response.
   If defer_headers is nonzero, headers are not sent until the first
   gufo_http_stream_write (so if the handler fails before writing, it may still
   call gufo_http_respond_buffered after calling gufo_http_stream_abort).
   on_heartbeat (if non-NULL) is called when an SSE heartbeat ping is sent. */
GufoHttpStream* gufo_http_respond_stream_begin(
    GufoHttpResponder* responder, int status, const GufoHttpHeaderView* headers,
    size_t headers_len, int is_sse, int defer_headers,
    GufoHttpBytesSink on_heartbeat, void* on_heartbeat_context);

/* Writes a chunk to the stream. Returns 1 on success, 0 if client is gone. */
int gufo_http_stream_write(GufoHttpStream* stream, const uint8_t* data,
                           size_t len);

/* Finishes and frees the stream. If complete is 0 and headers were already
   sent, the stream terminates with an error (leaving chunked encoding
   unterminated). Returns 1 if the client remained connected, 0 otherwise. */
int gufo_http_stream_end(GufoHttpStream* stream, int complete);

/* Aborts a deferred stream before headers were sent, freeing the stream handle
   so the caller can send a buffered error response on the same responder. */
void gufo_http_stream_abort_deferred(GufoHttpStream* stream,
                                     GufoHttpResponder* responder);

/* Upgrades the connection to a WebSocket (status 101), runs ws_handler on the
   calling thread once the upgrade completes and returns after the session has
   ended. Returns 1 if the session ran, 0 if the upgrade failed, in which case
   ws_handler is not called. */
int gufo_http_respond_websocket(GufoHttpResponder* responder, int status,
                                const GufoHttpHeaderView* headers,
                                size_t headers_len, GufoWsHandlerFn ws_handler,
                                void* ws_context);

/* WebSocket session operations (valid inside GufoWsHandlerFn). */
int gufo_ws_session_receive(GufoWsSession* session, GufoHttpBytesSink sink,
                            void* sink_context);
int gufo_ws_session_send_text(GufoWsSession* session, const uint8_t* data,
                              size_t len);
int gufo_ws_session_send_binary(GufoWsSession* session, const uint8_t* data,
                                size_t len);
void gufo_ws_session_close(GufoWsSession* session, uint16_t code);
int gufo_ws_session_is_closed(const GufoWsSession* session);

/* LoadingResponder: best-effort 503 responder while a model loads.
   Returns NULL if port <= 0 or the address cannot be bound. */
GufoLoadingResponder* gufo_loading_responder_start(const uint8_t* host,
                                                   size_t host_len, int port);
void gufo_loading_responder_set_progress(GufoLoadingResponder* responder,
                                         double progress,
                                         const uint8_t* message,
                                         size_t message_len);
void gufo_loading_responder_stop(GufoLoadingResponder* responder);
void gufo_loading_responder_free(GufoLoadingResponder* responder);

#ifdef __cplusplus
}
#endif

#endif  // GUFO_HTTP_SERVER_H_
