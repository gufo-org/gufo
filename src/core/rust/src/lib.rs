//! The single static library that C++ links. Two Rust static libraries in one
//! binary would each carry their own standard library, so every component
//! crate is linked here instead.

#![deny(unsafe_code)]

// Linking a component keeps its `#[unsafe(no_mangle)]` C functions in the
// static library; an unused dependency would not be linked at all.
extern crate gufo_audio_decode;
extern crate gufo_base64;
extern crate gufo_cache_manifest;
extern crate gufo_chat_request;
extern crate gufo_generation_parse;
extern crate gufo_gguf;
extern crate gufo_http;
extern crate gufo_http_server;
extern crate gufo_jpeg_decode;
extern crate gufo_json;
extern crate gufo_json_schema;
extern crate gufo_png;
extern crate gufo_websocket;
