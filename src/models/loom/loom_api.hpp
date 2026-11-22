// The Loom runtime, libloom_text_embed, loaded when the first Loom package
// opens. Gufo starts, and serves every other model, without it; the library
// drives the GPU through Loom's own driver, so it is never loaded into a
// process that only uses HIP.
//
// Each loom_text_embed_* call below goes through the loaded library. Tests
// that link a fake library directly define GUFO_LOOM_LINKED.
#ifndef GUFO_LOOM_API_HPP_
#define GUFO_LOOM_API_HPP_

#include <loom_text_embed.h>

#ifndef GUFO_LOOM_LINKED

#include <filesystem>
#include <gufo/ext/platform/library.hpp>
#include <gufo/ext/platform/os_info.hpp>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#define GUFO_LOOM_FUNCTIONS(X)           \
  X(loom_text_embed_chat_prepare)        \
  X(loom_text_embed_checkpoint_export)   \
  X(loom_text_embed_checkpoint_import)   \
  X(loom_text_embed_checkpoint_release)  \
  X(loom_text_embed_close)               \
  X(loom_text_embed_decode_batch)        \
  X(loom_text_embed_decode_batch_tokens) \
  X(loom_text_embed_decoder_feed)        \
  X(loom_text_embed_decoder_free)        \
  X(loom_text_embed_decoder_open)        \
  X(loom_text_embed_decoder_reset)       \
  X(loom_text_embed_detokenize)          \
  X(loom_text_embed_error)               \
  X(loom_text_embed_image_layout)        \
  X(loom_text_embed_info)                \
  X(loom_text_embed_is_stop_token)       \
  X(loom_text_embed_logits_info)         \
  X(loom_text_embed_memory)              \
  X(loom_text_embed_open_ex)             \
  X(loom_text_embed_output_close)        \
  X(loom_text_embed_output_feed)         \
  X(loom_text_embed_output_tools)        \
  X(loom_text_embed_query_gpu)           \
  X(loom_text_embed_request_free)        \
  X(loom_text_embed_request_logits)      \
  X(loom_text_embed_request_media)       \
  X(loom_text_embed_request_project)     \
  X(loom_text_embed_row_begin)           \
  X(loom_text_embed_row_bind)            \
  X(loom_text_embed_row_decode)          \
  X(loom_text_embed_row_end)             \
  X(loom_text_embed_row_extend)          \
  X(loom_text_embed_row_extent)          \
  X(loom_text_embed_row_is_eos)          \
  X(loom_text_embed_row_is_logit)        \
  X(loom_text_embed_row_logits)          \
  X(loom_text_embed_row_pin)             \
  X(loom_text_embed_row_position)        \
  X(loom_text_embed_row_prefill)         \
  X(loom_text_embed_row_prefill_request) \
  X(loom_text_embed_row_reset)           \
  X(loom_text_embed_row_restore)         \
  X(loom_text_embed_row_token)           \
  X(loom_text_embed_tokenize)            \
  X(loom_text_embed_verify_batch)        \
  X(loom_text_embed_vocabulary_size)

namespace gufo::loom {

struct Api {
#define GUFO_LOOM_MEMBER(name) decltype(&::name) name = nullptr;
  GUFO_LOOM_FUNCTIONS(GUFO_LOOM_MEMBER)
#undef GUFO_LOOM_MEMBER
};

// The library's file name, and where a build or an install puts it: beside
// gufo.exe on Windows, <prefix>/lib/gufo on Linux; else the build tree, else
// the library search path.
#ifdef _WIN32
inline constexpr const char* kLibrary = "loom_text_embed.dll";
#else
inline constexpr const char* kLibrary = "libloom_text_embed.so";
#endif

// The loaded library, or an error naming every place it was looked for.
inline const Api& LoadApi() {
  static std::once_flag once;
  static std::unique_ptr<platform::Library> library;
  static Api api;
  static std::string error;
  std::call_once(once, [] {
    std::vector<std::filesystem::path> candidates;
    try {
      const auto exe = platform::ExecutablePath();
#ifdef _WIN32
      candidates.push_back(exe.parent_path() / kLibrary);
#else
      candidates.push_back(exe.parent_path().parent_path() / "lib" / "gufo" /
                           kLibrary);
#endif
    } catch (const std::exception&) {
    }
#ifdef GUFO_LOOM_BUILD_LIBRARY
    candidates.emplace_back(GUFO_LOOM_BUILD_LIBRARY);
#endif
    std::string tried;
    for (const auto& path : candidates) {
      tried += "\n  " + path.string();
      std::error_code ec;
      if (!std::filesystem::is_regular_file(path, ec))
        continue;
      try {
        library = std::make_unique<platform::Library>(path.string());
        break;
      } catch (const platform::LibraryError& e) {
        error = e.what();
        return;
      }
    }
    if (library == nullptr) {
      // Then the loader's search path: the portable Linux artifact runs Gufo
      // through a bundled ELF loader whose --library-path holds lib/gufo.
      try {
        library = std::make_unique<platform::Library>(kLibrary);
      } catch (const platform::LibraryError& e) {
        error = std::string(kLibrary) + " was not found; looked in:" + tried +
                "\n  the library search path (" + e.what() + ")";
        return;
      }
    }
    try {
#define GUFO_LOOM_BIND(name) \
  api.name = reinterpret_cast<decltype(api.name)>(library->Symbol(#name));
      GUFO_LOOM_FUNCTIONS(GUFO_LOOM_BIND)
#undef GUFO_LOOM_BIND
    } catch (const platform::LibraryError& e) {
      error =
          std::string(e.what()) + " (the library is from another Gufo build)";
      library.reset();
    }
  });
  if (!error.empty())
    throw std::runtime_error("loom: the Loom runtime is not available: " +
                             error);
  return api;
}

}  // namespace gufo::loom

// A macro's own name is not expanded again inside it, so each one reaches the
// loaded function of the same name.
// clang-format off
#define loom_text_embed_chat_prepare (::gufo::loom::LoadApi().loom_text_embed_chat_prepare)
#define loom_text_embed_checkpoint_export (::gufo::loom::LoadApi().loom_text_embed_checkpoint_export)
#define loom_text_embed_checkpoint_import (::gufo::loom::LoadApi().loom_text_embed_checkpoint_import)
#define loom_text_embed_checkpoint_release (::gufo::loom::LoadApi().loom_text_embed_checkpoint_release)
#define loom_text_embed_close (::gufo::loom::LoadApi().loom_text_embed_close)
#define loom_text_embed_decode_batch (::gufo::loom::LoadApi().loom_text_embed_decode_batch)
#define loom_text_embed_decode_batch_tokens (::gufo::loom::LoadApi().loom_text_embed_decode_batch_tokens)
#define loom_text_embed_decoder_feed (::gufo::loom::LoadApi().loom_text_embed_decoder_feed)
#define loom_text_embed_decoder_free (::gufo::loom::LoadApi().loom_text_embed_decoder_free)
#define loom_text_embed_decoder_open (::gufo::loom::LoadApi().loom_text_embed_decoder_open)
#define loom_text_embed_decoder_reset (::gufo::loom::LoadApi().loom_text_embed_decoder_reset)
#define loom_text_embed_detokenize (::gufo::loom::LoadApi().loom_text_embed_detokenize)
#define loom_text_embed_error (::gufo::loom::LoadApi().loom_text_embed_error)
#define loom_text_embed_image_layout (::gufo::loom::LoadApi().loom_text_embed_image_layout)
#define loom_text_embed_info (::gufo::loom::LoadApi().loom_text_embed_info)
#define loom_text_embed_is_stop_token (::gufo::loom::LoadApi().loom_text_embed_is_stop_token)
#define loom_text_embed_logits_info (::gufo::loom::LoadApi().loom_text_embed_logits_info)
#define loom_text_embed_memory (::gufo::loom::LoadApi().loom_text_embed_memory)
#define loom_text_embed_open_ex (::gufo::loom::LoadApi().loom_text_embed_open_ex)
#define loom_text_embed_output_close (::gufo::loom::LoadApi().loom_text_embed_output_close)
#define loom_text_embed_output_feed (::gufo::loom::LoadApi().loom_text_embed_output_feed)
#define loom_text_embed_output_tools (::gufo::loom::LoadApi().loom_text_embed_output_tools)
#define loom_text_embed_query_gpu (::gufo::loom::LoadApi().loom_text_embed_query_gpu)
#define loom_text_embed_request_free (::gufo::loom::LoadApi().loom_text_embed_request_free)
#define loom_text_embed_request_logits (::gufo::loom::LoadApi().loom_text_embed_request_logits)
#define loom_text_embed_request_media (::gufo::loom::LoadApi().loom_text_embed_request_media)
#define loom_text_embed_request_project (::gufo::loom::LoadApi().loom_text_embed_request_project)
#define loom_text_embed_row_begin (::gufo::loom::LoadApi().loom_text_embed_row_begin)
#define loom_text_embed_row_bind (::gufo::loom::LoadApi().loom_text_embed_row_bind)
#define loom_text_embed_row_decode (::gufo::loom::LoadApi().loom_text_embed_row_decode)
#define loom_text_embed_row_end (::gufo::loom::LoadApi().loom_text_embed_row_end)
#define loom_text_embed_row_extend (::gufo::loom::LoadApi().loom_text_embed_row_extend)
#define loom_text_embed_row_extent (::gufo::loom::LoadApi().loom_text_embed_row_extent)
#define loom_text_embed_row_is_eos (::gufo::loom::LoadApi().loom_text_embed_row_is_eos)
#define loom_text_embed_row_is_logit (::gufo::loom::LoadApi().loom_text_embed_row_is_logit)
#define loom_text_embed_row_logits (::gufo::loom::LoadApi().loom_text_embed_row_logits)
#define loom_text_embed_row_pin (::gufo::loom::LoadApi().loom_text_embed_row_pin)
#define loom_text_embed_row_position (::gufo::loom::LoadApi().loom_text_embed_row_position)
#define loom_text_embed_row_prefill (::gufo::loom::LoadApi().loom_text_embed_row_prefill)
#define loom_text_embed_row_prefill_request (::gufo::loom::LoadApi().loom_text_embed_row_prefill_request)
#define loom_text_embed_row_reset (::gufo::loom::LoadApi().loom_text_embed_row_reset)
#define loom_text_embed_row_restore (::gufo::loom::LoadApi().loom_text_embed_row_restore)
#define loom_text_embed_row_token (::gufo::loom::LoadApi().loom_text_embed_row_token)
#define loom_text_embed_tokenize (::gufo::loom::LoadApi().loom_text_embed_tokenize)
#define loom_text_embed_verify_batch (::gufo::loom::LoadApi().loom_text_embed_verify_batch)
#define loom_text_embed_vocabulary_size (::gufo::loom::LoadApi().loom_text_embed_vocabulary_size)
// clang-format on

#endif  // GUFO_LOOM_LINKED

#endif  // GUFO_LOOM_API_HPP_
