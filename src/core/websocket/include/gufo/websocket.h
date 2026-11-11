#ifndef GUFO_WEBSOCKET_H_
#define GUFO_WEBSOCKET_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust WebSocket frame reader: the frames a client
   sends (RFC 6455, section 5). src/cli/serve/websocket.cpp reads its
   connections with it. */
typedef struct GufoWsReader GufoWsReader;

typedef enum GufoWsEventKind {
  /* The read callback failed before a whole frame arrived. */
  GUFO_WS_END = 0,
  /* A complete text message in UTF-8. */
  GUFO_WS_MESSAGE = 1,
  /* A ping, to answer with a pong carrying its payload. */
  GUFO_WS_PING = 2,
  /* The client's close frame, whose payload the reply echoes. */
  GUFO_WS_CLOSE = 3,
  /* A protocol violation, to answer with a close frame with `code`: 1002 for
     a malformed frame, 1007 for text or a close reason that is not UTF-8,
     1009 for more than max_message bytes, 1011 if the payload could not be
     allocated. */
  GUFO_WS_ERROR = 4,
  /* A caller bug (null pointer) or a reader bug. */
  GUFO_WS_INTERNAL = 5,
} GufoWsEventKind;

typedef struct GufoWsEvent {
  GufoWsEventKind kind;
  uint16_t code;
  /* The text of a message, or the payload of a ping or close frame; valid
     until the next call with the same reader. */
  const uint8_t* data;
  size_t len;
} GufoWsEvent;

/* Fills len bytes at data and returns nonzero, or returns 0 once the
   connection is over. */
typedef int (*GufoWsReadFn)(void* context, uint8_t* data, size_t len);

/* A reader for messages of at most max_message bytes, or NULL if it could
   not be allocated. */
GufoWsReader* gufo_ws_reader_new(size_t max_message);

void gufo_ws_reader_free(GufoWsReader* reader);

/* Reads frames through read until the next event, which it stores in *event.
   Pongs are skipped and fragments joined. After an end, close, error or
   internal event, the connection is over. */
void gufo_ws_reader_next(GufoWsReader* reader, GufoWsReadFn read, void* context,
                         GufoWsEvent* event);

#ifdef __cplusplus
}
#endif

#endif
