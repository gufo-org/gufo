// Loom text models embedded in another serving process. See loom_text_embed.h.

#include "loom_text_embed.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <direct.h>
#include <process.h>
#include <windows.h>
#else
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "iree/base/api.h"
#include "iree/base/internal/json.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tooling/device_util.h"
#include "loom/runtime/device/device.h"
#include "loom/runtime/host/host.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/device_configuration.h"
#include "loom/service/finite/component.h"
#include "loom/service/kind.h"
#include "loom/serve/service/text/input.h"
#include "loom/serve/service/text/media.h"
#include "loom/serve/service/text/media_input.h"
#include "loom/serve/service/text/media_projector.h"
#include "loom/serve/service/text/chat.h"
#include "loom/serve/service/text/model.h"
#include "loom/serve/service/text/model_instance.h"
#include "loom/runtime/storage/snapshot.h"
#include "loom/runtime/storage/memory.h"
#include "loom/runtime/execution/execution.h"
#include "loom/serve/service/text/schedule.h"
#include "loom/serve/service/text/tool_calls.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/configured.h"
#include "amdf/api.h"
#include "amdf/gpu.h"
#include "amdf/memory.h"

#if defined(_WIN32)
// exports.def names the exported functions.
#define LOOM_TEXT_EMBED_EXPORT
#else
#define LOOM_TEXT_EMBED_EXPORT __attribute__((visibility("default")))
#endif

// The few host facilities this library needs beyond Loom and IREE.

// Lets other threads run while polling Loom for progress.
static void loom_text_embed_pause(unsigned microseconds) {
#if defined(_WIN32)
  (void)microseconds;
  Sleep(0);
#else
  usleep(microseconds);
#endif
}

static char* loom_text_embed_strndup(const char* text, size_t length) {
  char* copy = (char*)malloc(length + 1);
  if (copy) {
    memcpy(copy, text, length);
    copy[length] = 0;
  }
  return copy;
}

static char* loom_text_embed_strdup(const char* text) {
  return loom_text_embed_strndup(text, strlen(text));
}

// Creates a private directory `<temp>/loom_embed_*` and writes its path to
// `directory`. Returns false with errno set on failure.
static bool loom_text_embed_make_temp_directory(char* directory,
                                                size_t capacity) {
#if defined(_WIN32)
  char base[MAX_PATH + 1];
  const DWORD length = GetTempPathA(sizeof(base), base);
  if (length == 0 || length > MAX_PATH) {
    errno = ENOENT;
    return false;
  }
  static volatile LONG counter = 0;
  for (int attempt = 0; attempt < 100; ++attempt) {
    snprintf(directory, capacity, "%sloom_embed_%d_%ld", base, _getpid(),
             (long)InterlockedIncrement(&counter));
    if (_mkdir(directory) == 0) return true;
    if (errno != EEXIST) return false;
  }
  return false;
#else
  const char* base = getenv("TMPDIR");
  snprintf(directory, capacity, "%s/loom_embed_XXXXXX",
           base && base[0] ? base : "/tmp");
  return mkdtemp(directory) != NULL;
#endif
}

static void loom_text_embed_remove_file(const char* path) {
#if defined(_WIN32)
  _unlink(path);
#else
  unlink(path);
#endif
}

static void loom_text_embed_remove_directory(const char* path) {
#if defined(_WIN32)
  _rmdir(path);
#else
  rmdir(path);
#endif
}

struct loom_text_embed_t {
  iree_allocator_t allocator;
  loom_runtime_host_t* host;
  loom_serve_device_configuration_t devices;
  loom_runtime_device_group_t* group;
  loom_serve_model_configuration_t configuration;
  loom_serve_model_configuration_t resolved;
  bool has_resolved;
  // Reserved consumed-position high-water per row; zero when unreserved.
  uint64_t* extents;
  loom_serve_text_model_t* model;
  // The package's image component and its projector, when it has vision.
  loom_service_finite_component_t* image;
  loom_serve_text_media_projector_t* projector;
  iree_host_size_t shape_count;
  loom_serve_packing_shape_t* shapes;
  iree_host_size_t prefill_capacity;
  uint32_t rows;
  uint32_t draft_depth;
  uint64_t request_serial;
  char model_id[256];
  char error[2048];
  // ABI 2: logits. The package declares "llm.logits"; the handle may be
  // opened with it on, which rewrites the card (a private copy in card_dir)
  // and leaves the package's own files untouched.
  bool logits_declared;
  bool logits_enabled;
  char* package_dir;
  char* card_dir;
  char* card_path;
  // Export region of the epoch outputs allocation: byte offset of slot 0,
  // bytes per slot, floats per row, slots (widest epoch's span capacity).
  uint64_t logits_offset;
  uint64_t logits_stride;
  uint32_t logits_width;
  uint32_t logits_slots;
  // Per row: asked for logits by its bound request; host copy of the logit
  // row of its pending selection and whether that copy is current.
  uint8_t* logit_flag;
  uint8_t* logit_valid;
  float** logit_rows;
};

struct loom_text_embed_request_t {
  loom_text_embed_t* embed;
  char* body;
  uint64_t serial;
  loom_serve_text_chat_t chat;
  // Checked input with image spans, and its projected features.
  loom_serve_text_input_t input;
  bool has_input;
  loom_serve_text_media_request_t* media;
  // Generated text so far, and whether a reasoning span is open on output.
  iree_string_builder_t response;
  // The body asked for logits ("logits": true).
  bool logits;
  bool reasoning_open;
  bool finished;
  // Response offsets: where the last reasoning span ended, and the framing
  // between reasoning and the first answer text (the model's own close).
  size_t reasoning_end;
  size_t close_begin;
  size_t close_end;
  bool reasoned;
  bool closed;
};

static const loom_serve_model_configuration_t* loom_text_embed_config(
    const loom_text_embed_t* embed) {
  return embed->has_resolved ? &embed->resolved : &embed->configuration;
}

// Records a failed status as the handle's error and returns nonzero.
static int loom_text_embed_fail(char* error, size_t error_size,
                                iree_status_t status) {
  if (iree_status_is_ok(status)) return 0;
  iree_allocator_t allocator = iree_allocator_system();
  char* text = NULL;
  iree_host_size_t length = 0;
  if (error && error_size) {
    if (iree_status_to_string(status, &allocator, &text, &length)) {
      snprintf(error, error_size, "%.*s", (int)length, text);
      iree_allocator_free(allocator, text);
    } else {
      snprintf(error, error_size, "%s",
               iree_status_code_string(iree_status_code(status)));
    }
  }
  iree_status_free(status);
  return 2;
}

#define EMBED_RETURN(embed, expr) \
  return loom_text_embed_fail((embed)->error, sizeof((embed)->error), (expr))

// Loom streams weights through io_uring with pinned staging buffers, which
// need more locked memory than a stock Linux host allows (RLIMIT_MEMLOCK of
// 8 MiB; ROCm/hrx-system#3332). Raises the soft limit to the hard one; when
// that is still below 64 MiB, loads parameters as loom-serve's
// --parameter_mode=preload does (whole files read into host memory) instead.
#define LOOM_TEXT_EMBED_MEMLOCK_BYTES (64ull << 20)

static iree_status_t loom_text_embed_select_parameter_mode(void) {
#if defined(_WIN32)
  // Windows has no locked-memory limit to raise.
  return iree_ok_status();
#else
  struct rlimit limit;
  if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0) return iree_ok_status();
  if (limit.rlim_cur != RLIM_INFINITY &&
      limit.rlim_cur < LOOM_TEXT_EMBED_MEMLOCK_BYTES &&
      limit.rlim_cur < limit.rlim_max) {
    limit.rlim_cur = limit.rlim_max;
    if (setrlimit(RLIMIT_MEMLOCK, &limit) != 0) getrlimit(RLIMIT_MEMLOCK, &limit);
  }
  if (limit.rlim_cur == RLIM_INFINITY ||
      limit.rlim_cur >= LOOM_TEXT_EMBED_MEMLOCK_BYTES) {
    return iree_ok_status();
  }
  static bool reported = false;
  if (!reported) {
    reported = true;
    fprintf(stderr,
            "loom: locked-memory limit %llu KiB is below %llu MiB: weights are "
            "read into host memory at load (raise `ulimit -l` to stream them "
            "from disk)\n",
            (unsigned long long)(limit.rlim_cur >> 10),
            (unsigned long long)(LOOM_TEXT_EMBED_MEMLOCK_BYTES >> 20));
  }
  // String flags keep pointers into argv: static storage.
  static char program[] = "loom_text_embed";
  static char flag[] = "--parameter_mode=preload";
  char* argv[] = {program, flag, NULL};
  int argc = 2;
  char** args = argv;
  return iree_flags_parse(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &args);
#endif
}

static iree_status_t loom_text_embed_open_devices(
    loom_text_embed_t* embed, const loom_text_embed_options_t* options,
    bool elastic) {
  IREE_RETURN_IF_ERROR(loom_text_embed_select_parameter_mode());
  static const loomc_iree_hal_target_provider_t* providers[1];
  providers[0] = loomc_amdgpu_iree_hal_target_provider();
  static loom_runtime_compiler_options_t compiler_options;
  compiler_options = (loom_runtime_compiler_options_t){
      .target_environment_create = loomc_target_environment_create_configured,
      .target_providers = providers,
      .target_provider_count = 1,
  };
  IREE_RETURN_IF_ERROR(loom_runtime_host_create(8, NULL, &compiler_options,
                                              embed->allocator, &embed->host));
  const char* uri = options->device_uri && options->device_uri[0]
                        ? options->device_uri
                        : "amdgpu://?completion_spin_ns=4294967295";
  IREE_RETURN_IF_ERROR(loom_serve_device_configuration_initialize_uri(
      iree_make_cstring_view(uri), embed->allocator, &embed->devices));
  if (embed->devices.count != 1 || embed->devices.pool_count < 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected one device and pool");
  }
  // As the loom-fun cards serve: the pool is fixed (backing committed at
  // load, faster than elastic on 0.2.0) unless a budget or checkpoint images
  // need elastic backing.
  embed->devices.pools[0].options.elastic = elastic || options->memory_bytes;
  embed->devices.pools[0].options.memory_limit = options->memory_bytes;
  loom_runtime_device_options_t* device = &embed->devices.options[0];
  iree_hal_driver_t* driver = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_acquire_driver(
      iree_hal_available_driver_registry(), iree_uri_schema(device->uri),
      &driver));
  device->driver = driver;
  iree_status_t status = loom_runtime_device_group_allocate(
      embed->host, 1, embed->devices.options, embed->allocator, &embed->group);
  for (iree_host_size_t i = 0;
       i < embed->devices.pool_count && iree_status_is_ok(status); ++i) {
    loom_runtime_pool_t* pool = NULL;
    status = loom_runtime_device_group_create_pool(
        embed->group, embed->devices.pools[i].binding,
        &embed->devices.pools[i].options, &pool);
  }
  iree_hal_driver_release(driver);
  device->driver = NULL;
  return status;
}

// Owner wakes are not needed: image work is driven by polling.
static void loom_text_embed_wake(void* user_data) { (void)user_data; }

//===----------------------------------------------------------------------===//
// Logits capability: card handling and the export region.
//===----------------------------------------------------------------------===//

#define LOOM_TEXT_EMBED_LOGITS_KEY "\"llm.logits\""
// The epoch outputs allocation (storage.loom slot 5): token feedback first,
// then, when the package is loaded with llm.logits, one row of logits per span.
#define LOOM_TEXT_EMBED_OUTPUTS_ALLOCATION 5

static char* loom_text_embed_read_file(const char* path, size_t* out_size) {
  FILE* file = fopen(path, "rb");
  if (!file) return NULL;
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);
  char* data = size >= 0 ? (char*)malloc((size_t)size + 1) : NULL;
  if (data && fread(data, 1, (size_t)size, file) != (size_t)size) {
    free(data);
    data = NULL;
  }
  fclose(file);
  if (data) {
    data[size] = 0;
    *out_size = (size_t)size;
  }
  return data;
}

// Visits every "llm.logits" member in a card's text; when `set` is nonzero
// rewrites a false value to true (same length) in place. Returns the number of
// members found, or -1 for a value that is not a boolean.
static int loom_text_embed_scan_logits(char* text, int set) {
  int found = 0;
  for (char* at = strstr(text, LOOM_TEXT_EMBED_LOGITS_KEY); at;
       at = strstr(at + 1, LOOM_TEXT_EMBED_LOGITS_KEY)) {
    char* cursor = at + strlen(LOOM_TEXT_EMBED_LOGITS_KEY);
    while (isspace((unsigned char)*cursor)) ++cursor;
    if (*cursor != ':') continue;
    ++cursor;
    while (isspace((unsigned char)*cursor)) ++cursor;
    if (!strncmp(cursor, "false", 5)) {
      if (set) memcpy(cursor, "true ", 5);
    } else if (strncmp(cursor, "true", 4)) {
      return -1;
    }
    ++found;
  }
  return found;
}

static iree_status_t loom_text_embed_prepare_card(
    loom_text_embed_t* embed, const char* path, bool enable) {
  size_t size = 0;
  char* text = loom_text_embed_read_file(path, &size);
  if (!text) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "cannot read %s", path);
  }
  const int declared = loom_text_embed_scan_logits(text, 0);
  embed->logits_declared = declared > 0;
  iree_status_t status = iree_ok_status();
  if (enable) {
    if (declared < 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "llm.logits must be a boolean in the card");
    } else if (declared == 0) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "this package does not declare llm.logits; "
                                "regenerate it with a logits-capable kit");
    }
  }
  if (iree_status_is_ok(status) && enable) {
    loom_text_embed_scan_logits(text, 1);
    char directory[4096];
    if (!loom_text_embed_make_temp_directory(directory, sizeof(directory))) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "cannot create a temporary card directory: %s",
                                strerror(errno));
    } else {
      embed->card_dir = loom_text_embed_strdup(directory);
      char card[4200];
      snprintf(card, sizeof(card), "%s/model.json", directory);
      embed->card_path = loom_text_embed_strdup(card);
      FILE* file = fopen(card, "wb");
      if (!file || fwrite(text, 1, size, file) != size) {
        status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "cannot write %s: %s", card, strerror(errno));
      }
      if (file) fclose(file);
    }
  }
  free(text);
  return status;
}

static void loom_text_embed_remove_card(loom_text_embed_t* embed) {
  if (embed->card_path) {
    loom_text_embed_remove_file(embed->card_path);
    free(embed->card_path);
    embed->card_path = NULL;
  }
  if (embed->card_dir) {
    loom_text_embed_remove_directory(embed->card_dir);
    free(embed->card_dir);
    embed->card_dir = NULL;
  }
}

// Widest epoch shape's span capacity (the storage program sizes its per-span
// regions by it).
static iree_host_size_t loom_text_embed_span_capacity(
    const loom_text_embed_t* embed) {
  iree_host_size_t spans = 0;
  for (iree_host_size_t i = 0; i < embed->shape_count; ++i) {
    spans = iree_max(spans, embed->shapes[i].span_capacity);
  }
  return spans;
}

// The export region of the epoch outputs allocation, as the package's storage
// program sizes it (starter kit, LOGITS.md): slot 0 starts at byte
// LOOM_TEXT_EMBED_LOGITS_OFFSET (after the 8192 span tokens), and each of the
// min(widest epoch's spans, resident rows) slots is ceil(vocab / 128) * 512
// bytes. The stride is derived from the allocation's size and checked.
#define LOOM_TEXT_EMBED_LOGITS_OFFSET 32768

static iree_status_t loom_text_embed_logits_layout(loom_text_embed_t* embed) {
  loom_serve_text_state_t* state = &embed->model->state;
  if (!state->cache.capacity &&
      !iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "logits need the packed epoch path (pool_capacity)");
  }
  iree_hal_buffer_t* outputs =
      *loom_serve_text_state_allocation(state,
                                        LOOM_TEXT_EMBED_OUTPUTS_ALLOCATION);
  const uint64_t length = outputs ? iree_hal_buffer_byte_length(outputs) : 0;
  const uint64_t slots =
      iree_min(loom_text_embed_span_capacity(embed), embed->rows);
  const uint64_t offset = LOOM_TEXT_EMBED_LOGITS_OFFSET;
  if (!slots || length <= offset || (length - offset) % slots ||
      ((length - offset) / slots) % 512) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "outputs allocation (%" PRIu64 " bytes, %" PRIu64
        " slots) has no logits region; the package was not built with "
        "llm.logits or uses another layout",
        length, slots);
  }
  embed->logits_offset = offset;
  embed->logits_stride = (length - offset) / slots;
  embed->logits_width = (uint32_t)(embed->logits_stride / 4);
  embed->logits_slots = (uint32_t)slots;
  return iree_ok_status();
}

// The KV pool's token positions for `rows` rows of `row_context` (at most the
// native context), rounded up to a multiple of the source's page size. Page
// sizes are powers of two dividing the native context, so the largest such
// divisor up to 4096 is one.
static iree_status_t loom_text_embed_pool_tokens(
    const loom_serve_text_configuration_t* text, uint32_t rows,
    uint64_t row_context, iree_host_size_t* out_tokens) {
  const uint64_t context = text->context_capacity;
  uint64_t granule = context & (~context + 1);
  if (granule > 4096 || granule == 0) granule = 4096;
  const uint64_t per_row = row_context < context ? row_context : context;
  uint64_t tokens = 0;
  if (!iree_checked_mul_u64(per_row, rows, &tokens) ||
      tokens > INT64_MAX - granule) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "KV pool of %u rows of %" PRIu64 " tokens", rows,
                            per_row);
  }
  tokens = (tokens + granule - 1) / granule * granule;
  if (tokens > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "KV pool of %" PRIu64 " tokens", tokens);
  }
  *out_tokens = (iree_host_size_t)tokens;
  return iree_ok_status();
}

static iree_status_t loom_text_embed_open_model(
    loom_text_embed_t* embed, const loom_text_embed_options_ex_t* ex) {
  const loom_text_embed_options_t* options = &ex->base;
  char path[4096];
  const size_t length = strlen(options->model_path);
  const bool json = length >= 5 && !strcmp(options->model_path + length - 5,
                                            ".json");
  snprintf(path, sizeof(path), json ? "%s" : "%s/model.json",
           options->model_path);
  {
    iree_string_view_t directory =
        iree_file_path_dirname(iree_make_cstring_view(path));
    embed->package_dir = loom_text_embed_strndup(directory.size ? directory.data : ".",
                                 directory.size ? directory.size : 1);
  }
  IREE_RETURN_IF_ERROR(loom_text_embed_prepare_card(embed, path, ex->logits));
  const char* card_path = ex->logits ? embed->card_path : path;
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_load(
      iree_make_cstring_view(card_path),
      iree_make_cstring_view(options->preset ? options->preset : ""),
      embed->allocator, &embed->configuration));
  if (ex->logits) {
    // The card was read from its private copy; everything it names (programs,
    // parameters, tokenizer) is still the package's.
    const iree_string_view_t package = iree_make_cstring_view(embed->package_dir);
    embed->configuration.sources.base_directory = package;
    if (embed->configuration.resolved_sources) {
      embed->configuration.resolved_sources->base_directory = package;
    }
  }
  if (embed->configuration.sources.reference.path.size) {
    IREE_RETURN_IF_ERROR(loom_serve_model_configuration_resolve(
        &embed->configuration, embed->allocator, &embed->resolved));
    embed->has_resolved = true;
  }
  const loom_serve_model_configuration_t* configuration =
      loom_text_embed_config(embed);
  if (configuration->kind != LOOM_SERVICE_KIND_TEXT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "the package's selected service is not text");
  }
  const loom_serve_text_configuration_t* text = &configuration->text;
  snprintf(embed->model_id, sizeof(embed->model_id), "%.*s",
           (int)configuration->name.size, configuration->name.data);
  embed->prefill_capacity = text->prefill_capacity;
  embed->draft_depth = (uint32_t)text->draft_depth;
  embed->rows = options->rows ? options->rows : (uint32_t)text->rows;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      embed->allocator, sizeof(uint64_t) * embed->rows,
      (void**)&embed->extents));
  memset(embed->extents, 0, sizeof(uint64_t) * embed->rows);
  embed->shape_count =
      loom_serve_text_default_shapes(text->prefill_capacity, NULL);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      embed->allocator, embed->shape_count, sizeof(*embed->shapes),
      (void**)&embed->shapes));
  loom_serve_text_default_shapes(text->prefill_capacity, embed->shapes);
  loom_runtime_memory_pool_t* memory_pool = loom_runtime_pool_memory(
      loom_runtime_device_group_pool_at(embed->group, 0));
  iree_host_size_t pool_tokens = 0;
  if (ex->struct_size >= offsetof(loom_text_embed_options_ex_t, row_context) +
                             sizeof(ex->row_context) &&
      ex->row_context) {
    IREE_RETURN_IF_ERROR(loom_text_embed_pool_tokens(
        text, embed->rows, ex->row_context, &pool_tokens));
  }
  const loom_serve_text_options_t model_options = {
      .mode = LOOM_SERVE_TEXT_MODE_GENERATE,
      .sources = configuration->sources,
      .prefill_capacity = text->prefill_capacity,
      .context_capacity = text->context_capacity,
      // The positions the caller asked for; else as loom-serve registers a
      // text model: an elastic pool sizes the cache from its byte budget, a
      // fixed one holds one native context.
      .pool_capacity = pool_tokens   ? pool_tokens
                       : memory_pool ? 0
                                     : text->context_capacity,
      .cache_limit_bytes = !pool_tokens && memory_pool
                               ? loom_runtime_memory_pool_limit(memory_pool)
                               : 0,
      .epoch_count = embed->shape_count,
      .epoch_shapes = embed->shapes,
      .draft_depth = text->draft_depth,
      .kernel_sanitizer = {.type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
                           .structure_size = sizeof(loomc_sanitizer_options_t)},
      .weights_path = text->weights,
      .tokenizer_path = text->tokenizer,
      .row_count = embed->rows,
      .checkpoint_capacity = options->checkpoints ? options->checkpoints : 64,
      .options = configuration->options,
  };
  IREE_RETURN_IF_ERROR(loom_serve_text_model_allocate(
      loom_runtime_device_group_at(embed->group, 0),
      loom_runtime_device_group_pool_at(embed->group, 0), &model_options,
      embed->allocator, &embed->model));
  if (text->image.contract || text->image.prepare.size) {
    // The package's image component, as loom-serve registers it; prepared on
    // the first image request.
    loom_service_finite_model_options_t image_options = {
        .sources = configuration->sources,
        .asset_directory = text->image.assets,
        .options = text->image.options,
        .specialization_cache = text->image.specialization_cache,
    };
    image_options.sources.component = text->image.contract;
    image_options.sources.prepare = text->image.prepare;
    image_options.sources.control = text->image.control;
    IREE_RETURN_IF_ERROR(loom_service_finite_component_allocate(
        loom_runtime_device_group_at(embed->group, 0),
        loom_runtime_device_group_pool_at(embed->group, 0), &image_options,
        (loom_runtime_wake_callback_t){loom_text_embed_wake, NULL},
        embed->allocator, &embed->image));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_model_activate(embed->model));
  if (ex->logits) {
    IREE_RETURN_IF_ERROR(loom_text_embed_logits_layout(embed));
    embed->logits_enabled = true;
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(embed->allocator, 2 * embed->rows,
                                              (void**)&embed->logit_flag));
  memset(embed->logit_flag, 0, 2 * embed->rows);
  embed->logit_valid = embed->logit_flag + embed->rows;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      embed->allocator, embed->rows * sizeof(float*),
      (void**)&embed->logit_rows));
  memset(embed->logit_rows, 0, embed->rows * sizeof(float*));
  return iree_ok_status();
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_open_ex(
    const loom_text_embed_options_ex_t* options, loom_text_embed_t** out_embed,
    char* error, size_t error_size) {
  *out_embed = NULL;
  iree_allocator_t allocator = iree_allocator_system();
  loom_text_embed_t* embed = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*embed), (void**)&embed);
  if (!iree_status_is_ok(status)) {
    return loom_text_embed_fail(error, error_size, status);
  }
  memset(embed, 0, sizeof(*embed));
  embed->allocator = allocator;
  const bool elastic =
      options->struct_size >= offsetof(loom_text_embed_options_ex_t, elastic) +
                                  sizeof(options->elastic) &&
      options->elastic;
  status = loom_text_embed_open_devices(embed, &options->base, elastic);
  if (iree_status_is_ok(status)) {
    status = loom_text_embed_open_model(embed, options);
  }
  if (!iree_status_is_ok(status)) {
    loom_text_embed_close(embed);
    return loom_text_embed_fail(error, error_size, status);
  }
  *out_embed = embed;
  return 0;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_open(
    const loom_text_embed_options_t* options, loom_text_embed_t** out_embed,
    char* error, size_t error_size) {
  const loom_text_embed_options_ex_t ex = {
      .struct_size = sizeof(ex), .base = *options, .logits = 0};
  return loom_text_embed_open_ex(&ex, out_embed, error, error_size);
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_close(loom_text_embed_t* embed) {
  if (!embed) return;
  if (embed->projector) {
    loom_serve_text_media_projector_free(embed->projector);
  }
  if (embed->image) {
    bool retired = false;
    for (int i = 0; i < 100000 && !retired; ++i) {
      iree_status_ignore(
          loom_service_finite_component_retire(embed->image, &retired));
      if (!retired) loom_text_embed_pause(100);
    }
    loom_service_finite_component_free(embed->image);
  }
  if (embed->model) {
    iree_status_ignore(loom_serve_text_model_free(embed->model));
  }
  if (embed->group) {
    iree_status_ignore(loom_runtime_device_group_free(embed->group));
  }
  loom_serve_device_configuration_deinitialize(&embed->devices);
  if (embed->logit_rows) {
    for (uint32_t i = 0; i < embed->rows; ++i) free(embed->logit_rows[i]);
  }
  iree_allocator_free(embed->allocator, embed->logit_rows);
  iree_allocator_free(embed->allocator, embed->logit_flag);
  iree_allocator_free(embed->allocator, embed->extents);
  iree_allocator_free(embed->allocator, embed->shapes);
  if (embed->has_resolved) {
    loom_serve_model_configuration_deinitialize(&embed->resolved);
  }
  loom_serve_model_configuration_deinitialize(&embed->configuration);
  if (embed->host) loom_runtime_host_release(embed->host);
  loom_text_embed_remove_card(embed);
  free(embed->package_dir);
  iree_allocator_free(embed->allocator, embed);
}

LOOM_TEXT_EMBED_EXPORT const char* loom_text_embed_error(
    const loom_text_embed_t* embed) {
  return embed->error;
}

static iree_host_size_t loom_text_embed_max_spans(
    const loom_text_embed_t* embed) {
  iree_host_size_t spans = 0;
  for (iree_host_size_t i = 0; i < embed->shape_count; ++i) {
    spans = iree_max(spans, embed->shapes[i].span_capacity);
  }
  return iree_min(spans, (iree_host_size_t)embed->rows);
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_info(
    const loom_text_embed_t* embed, loom_text_embed_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  memcpy(out_info->model_id, embed->model_id, sizeof(out_info->model_id));
  out_info->context_capacity =
      (uint32_t)loom_serve_text_model_context_capacity(embed->model);
  out_info->prefill_capacity = (uint32_t)embed->prefill_capacity;
  out_info->rows = embed->rows;
  out_info->draft_depth =
      (uint32_t)loom_serve_text_model_draft_depth(embed->model);
  out_info->max_epoch_spans = (uint32_t)loom_text_embed_max_spans(embed);
  out_info->max_images =
      embed->image ? (uint32_t)loom_serve_text_model_chat_policy(embed->model)
                         ->images.maximum_count
                   : 0;
}

// Removes the top-level `"logits": <value>` member from the request body (in place; the body is a NUL-terminated copy owned by the request).
// This loom (hrx-system v0.1.0) validates the top-level chat fields against a closed set and rejects `logits` ("unsupported chat field"),
// while the flag is the embed layer's own: it is applied to the bound sampling record instead (loom_text_embed_apply_logits_flag).
static void loom_text_embed_strip_logits(char* body, size_t* size, iree_string_view_t value) {
  if (!value.data || value.data < body || value.data + value.size > body + *size) return;
  char* end = (char*)value.data + value.size;
  char* key_end = (char*)value.data;
  while (key_end > body && isspace((unsigned char)key_end[-1])) --key_end;
  if (key_end == body || key_end[-1] != ':') return;
  --key_end;
  while (key_end > body && isspace((unsigned char)key_end[-1])) --key_end;
  static const char kKey[] = "\"logits\"";
  const size_t key_length = sizeof(kKey) - 1;
  if ((size_t)(key_end - body) < key_length || memcmp(key_end - key_length, kKey, key_length) != 0) return;
  char* start = key_end - key_length;
  char* tail = end;
  while (tail < body + *size && isspace((unsigned char)*tail)) ++tail;
  if (tail < body + *size && *tail == ',') {
    end = tail + 1;                               // a following comma goes with the member
    while (end < body + *size && isspace((unsigned char)*end)) ++end;
  } else {
    char* before = start;
    while (before > body && isspace((unsigned char)before[-1])) --before;
    if (before > body && before[-1] == ',') start = before - 1;   // else the preceding comma
  }
  memmove(start, end, (size_t)(body + *size - end) + 1);
  *size -= (size_t)(end - start);
}

// Sets bit 30 of the bound sampling record's top_k (control.loom: the per-span export flag of the package). The record the package
// returned is replaced by a patched copy (it is 16 bytes: temperature, top_p, top_k, seed/nonce).
static iree_status_t loom_text_embed_apply_logits_flag(loom_text_embed_t* embed, loom_text_embed_request_t* request) {
  iree_vm_buffer_t* options = request->chat.options;
  if (!options) return iree_ok_status();
  iree_const_byte_span_t span;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(options, 0, 4 * sizeof(uint32_t), &span));
  uint32_t record[4];
  memcpy(record, span.data, sizeof(record));
  record[2] |= 0x40000000u;
  iree_vm_buffer_t* patched = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                            iree_make_const_byte_span(record, sizeof(record)), 1, embed->allocator,
                                            &patched));
  iree_vm_buffer_release(options);
  request->chat.options = patched;
  return iree_ok_status();
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_chat_prepare(
    loom_text_embed_t* embed, const char* body, size_t body_size,
    int32_t* tokens, size_t capacity, size_t* out_count,
    loom_text_embed_request_t** out_request) {
  *out_count = 0;
  if (out_request) *out_request = NULL;
  loom_text_embed_request_t* request = NULL;
  iree_status_t status = iree_allocator_malloc(
      embed->allocator, sizeof(*request) + body_size + 1, (void**)&request);
  if (!iree_status_is_ok(status)) EMBED_RETURN(embed, status);
  memset(request, 0, sizeof(*request));
  request->embed = embed;
  request->serial = embed->request_serial + 1;
  iree_string_builder_initialize(embed->allocator, &request->response);
  request->body = (char*)(request + 1);
  memcpy(request->body, body, body_size);
  request->body[body_size] = 0;
  {
    // "logits": true asks for the full logit row of every selection.
    iree_string_view_t value = iree_string_view_empty();
    iree_status_t lookup = iree_json_try_lookup_object_value(
        iree_make_string_view(request->body, body_size), IREE_SV("logits"),
        &value);
    if (iree_status_is_ok(lookup)) {
      request->logits = value.data && iree_string_view_equal(value, IREE_SV("true"));
      loom_text_embed_strip_logits(request->body, &body_size, value);
    } else {
      iree_status_ignore(lookup);
    }
  }
  const loom_serve_model_configuration_t* configuration =
      loom_text_embed_config(embed);
  status = loom_serve_text_chat_initialize(
      loom_serve_text_model_chat_policy(embed->model), configuration->name,
      iree_make_string_view(request->body, body_size), ++embed->request_serial,
      /*default_max_tokens=*/1, embed->allocator, &request->chat);
  if (iree_status_is_ok(status) && request->logits) {
    status = loom_text_embed_apply_logits_flag(embed, request);
    if (!iree_status_is_ok(status)) loom_serve_text_chat_deinitialize(&request->chat);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_text_media_span_t* media = NULL;
    iree_host_size_t media_count = 0;
    iree_host_size_t count = 0;
    iree_vm_buffer_t* token_storage = NULL;
    status = loom_serve_text_chat_prepare_input(
        &request->chat, 0, LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,
        LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,
        tokens ? capacity
               : loom_serve_text_model_context_capacity(embed->model),
        0, embed->allocator, tokens ? &token_storage : NULL, &count, &media,
        &media_count);
    if (iree_status_is_ok(status) && token_storage) {
      memcpy(tokens, iree_vm_buffer_const_data(token_storage),
             count * sizeof(int32_t));
    }
    iree_vm_buffer_release(token_storage);
    iree_allocator_free(embed->allocator, media);
    if (iree_status_is_ok(status) && media_count) {
      if (!embed->image) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "this package does not accept images");
      } else {
        // Image requests keep the checked input with its media spans.
        const iree_host_size_t capacity =
            loom_serve_text_model_context_capacity(embed->model);
        status = loom_serve_text_input_prepare(
            &request->chat, (loom_serve_text_input_prefix_t){0}, false,
            capacity, &request->input, embed->allocator);
        if (iree_status_is_ok(status)) {
          request->has_input = true;
          count = request->input.count;
          if (tokens) {
            if (count > capacity) {
              status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                        "prompt exceeds the token buffer");
            } else {
              memcpy(tokens, request->input.tokens, count * sizeof(int32_t));
            }
          }
        }
      }
    }
    *out_count = count;
    if (!iree_status_is_ok(status)) {
      loom_serve_text_chat_deinitialize(&request->chat);
    }
  }
  if (!iree_status_is_ok(status) || !out_request) {
    if (iree_status_is_ok(status)) {
      loom_serve_text_chat_deinitialize(&request->chat);
    }
    iree_string_builder_deinitialize(&request->response);
    iree_allocator_free(embed->allocator, request);
    EMBED_RETURN(embed, status);
  }
  *out_request = request;
  return 0;
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_request_free(
    loom_text_embed_request_t* request) {
  if (!request) return;
  if (request->media) {
    if (request->embed->projector) {
      while (!loom_serve_text_media_projector_cancel(request->embed->projector,
                                                     request->media)) {
        bool progress = false;
        iree_status_ignore(loom_serve_text_media_projector_advance(
            request->embed->projector, 0, &progress));
        loom_text_embed_pause(100);
      }
    }
    iree_status_ignore(loom_serve_text_media_request_take_status(request->media));
    loom_serve_text_media_request_free(request->media);
  }
  if (request->has_input) {
    loom_serve_text_input_deinitialize(&request->input,
                                       request->embed->allocator);
  }
  iree_string_builder_deinitialize(&request->response);
  loom_serve_text_chat_deinitialize(&request->chat);
  iree_allocator_free(request->embed->allocator, request);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_tokenize(
    loom_text_embed_t* embed, const char* text, size_t text_size,
    int match_special, int32_t* tokens, size_t capacity, size_t* out_count) {
  const iree_tokenizer_t* tokenizer =
      loom_serve_text_model_tokenizer(embed->model);
  int32_t scratch[1];
  iree_tokenizer_token_output_t output = {
      .capacity = tokens ? capacity : 0,
      .token_ids = tokens ? tokens : scratch,
  };
  iree_host_size_t count = 0;
  iree_tokenizer_encode_flags_t flags = IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
  if (!match_special) {
    flags |= IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING;
  }
  iree_status_t status = iree_tokenizer_encode(
      tokenizer, iree_make_string_view(text, text_size), flags, output,
      embed->allocator, &count);
  *out_count = count;
  if (!tokens && iree_status_is_resource_exhausted(status)) {
    iree_status_ignore(status);
    return 0;
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_detokenize(
    loom_text_embed_t* embed, const int32_t* tokens, size_t count, char* text,
    size_t capacity, size_t* out_size) {
  const iree_tokenizer_t* tokenizer =
      loom_serve_text_model_tokenizer(embed->model);
  char scratch[1];
  iree_host_size_t size = 0;
  iree_status_t status = iree_tokenizer_decode(
      tokenizer, (iree_tokenizer_token_id_list_t){count, tokens},
      IREE_TOKENIZER_DECODE_FLAG_NONE,
      iree_make_mutable_string_view(text ? text : scratch, text ? capacity : 0),
      embed->allocator, &size);
  *out_size = size;
  if (!text && iree_status_is_resource_exhausted(status)) {
    iree_status_ignore(status);
    return 0;
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int32_t
loom_text_embed_vocabulary_size(loom_text_embed_t* embed) {
  return (int32_t)iree_tokenizer_vocab_capacity(
      iree_tokenizer_vocab(loom_serve_text_model_tokenizer(embed->model)));
}

//===----------------------------------------------------------------------===//
// ABI 2: logit rows.
//===----------------------------------------------------------------------===//

#define LOOM_TEXT_EMBED_MAX_BATCH 256

// Binds a request's logits flag to a row. A request that asks for logits on a
// handle opened without them is refused rather than silently ignored.
static iree_status_t loom_text_embed_set_logit_row(
    loom_text_embed_t* embed, uint32_t row,
    const loom_text_embed_request_t* request) {
  if (row >= embed->rows) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "row %u", row);
  }
  if (request->logits && !embed->logits_enabled) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "request asks for logits but this handle was opened without them");
  }
  const uint8_t flag = request->logits ? 1 : 0;
  if (embed->logit_flag[row] != flag) embed->logit_valid[row] = 0;
  embed->logit_flag[row] = flag;
  return iree_ok_status();
}

static void loom_text_embed_clear_logit_row(loom_text_embed_t* embed,
                                            uint32_t row) {
  if (row >= embed->rows) return;
  embed->logit_flag[row] = 0;
  embed->logit_valid[row] = 0;
}

// After a selecting epoch: downloads the logit row of every flagged row (span
// i of the epoch sits in export slot i) into that row's host copy, and marks
// the other rows' copies stale. One transfer batch, behind the epoch's
// completion.
static iree_status_t loom_text_embed_collect_logits(loom_text_embed_t* embed,
                                                    const uint32_t* rows,
                                                    size_t count) {
  if (!embed->logits_enabled) return iree_ok_status();
  loom_serve_text_state_t* state = &embed->model->state;
  iree_hal_buffer_t* outputs =
      *loom_serve_text_state_allocation(state,
                                        LOOM_TEXT_EMBED_OUTPUTS_ALLOCATION);
  iree_hal_transfer_operation_t operations[64];
  uint32_t targets[64];
  iree_host_size_t pending = 0;
  iree_status_t status = iree_ok_status();
  bool drained = false;
  for (size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const uint32_t row = rows[i];
    if (!embed->logit_flag[row]) {
      embed->logit_valid[row] = 0;
      continue;
    }
    if (i >= embed->logits_slots) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "epoch span %zu has no logits slot", i);
      break;
    }
    if (!embed->logit_rows[row]) {
      embed->logit_rows[row] = (float*)malloc(embed->logits_stride);
      if (!embed->logit_rows[row]) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "logit row buffer");
        break;
      }
    }
    if (!drained) {
      status = loom_runtime_execution_drain(state->execution);
      drained = true;
      if (!iree_status_is_ok(status)) break;
    }
    embed->logit_valid[row] = 0;
    operations[pending] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer = outputs,
                     .source_offset = embed->logits_offset +
                                      (uint64_t)i * embed->logits_stride,
                     .target = (uint8_t*)embed->logit_rows[row],
                     .length = embed->logits_stride}};
    targets[pending++] = row;
    if (pending == IREE_ARRAYSIZE(operations)) {
      uint64_t completion = 0;
      status = loom_runtime_execution_transfer(state->execution, pending,
                                             operations, &completion);
      for (iree_host_size_t k = 0; k < pending; ++k) {
        embed->logit_valid[targets[k]] = iree_status_is_ok(status);
      }
      pending = 0;
    }
  }
  if (iree_status_is_ok(status) && pending) {
    uint64_t completion = 0;
    status = loom_runtime_execution_transfer(state->execution, pending,
                                           operations, &completion);
    for (iree_host_size_t k = 0; k < pending; ++k) {
      embed->logit_valid[targets[k]] = iree_status_is_ok(status);
    }
  }
  if (drained) {
    status = iree_status_join(status, loom_runtime_execution_drain(state->execution));
  }
  return status;
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_logits_info(
    const loom_text_embed_t* embed, loom_text_embed_logits_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  out_info->supported = embed->logits_declared;
  out_info->enabled = embed->logits_enabled;
  out_info->width = embed->logits_width;
  out_info->max_rows = embed->logits_slots;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_request_logits(
    const loom_text_embed_request_t* request) {
  return request->logits ? 1 : 0;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_is_logit(
    loom_text_embed_t* embed, uint32_t row) {
  return row < embed->rows && embed->logit_flag[row];
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_logits(
    loom_text_embed_t* embed, uint32_t row, float* out, size_t capacity,
    size_t* out_count) {
  *out_count = 0;
  if (row >= embed->rows || !embed->logit_flag[row] ||
      !embed->logit_valid[row]) {
    EMBED_RETURN(embed, iree_make_status(
                            IREE_STATUS_FAILED_PRECONDITION,
                            "row %u has no logits for its pending selection",
                            row));
  }
  if (capacity < embed->logits_width) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "logits need %u floats, got %zu",
                                         embed->logits_width, capacity));
  }
  memcpy(out, embed->logit_rows[row], embed->logits_stride);
  *out_count = embed->logits_width;
  return 0;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_is_stop_token(
    const loom_text_embed_t* embed, int32_t token) {
  return token == embed->model->source.eos_token;
}

static loom_serve_text_row_t* loom_text_embed_row(loom_text_embed_t* embed,
                                                  uint32_t row) {
  return loom_serve_text_model_row(embed->model, row);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_begin(
    loom_text_embed_t* embed, uint32_t row,
    const loom_text_embed_request_t* request, uint64_t extent, int fresh) {
  loom_serve_text_row_t* handle = loom_text_embed_row(embed, row);
  loom_serve_text_row_release_reservation(handle);
  embed->extents[row] = 0;
  bool admitted = false;
  iree_status_t status = loom_serve_text_row_try_reserve(
      handle,
      fresh ? LOOM_SERVE_TEXT_RESERVE_FRESH : LOOM_SERVE_TEXT_RESERVE_CONTINUE,
      NULL, (iree_host_size_t)extent, &admitted);
  if (iree_status_is_ok(status) && !admitted) return 1;
  if (iree_status_is_ok(status)) embed->extents[row] = extent;
  if (iree_status_is_ok(status) && request) {
    status = loom_text_embed_set_logit_row(embed, row, request);
    if (iree_status_is_ok(status)) {
      status = loom_serve_text_row_bind_request(handle, request->chat.options);
    }
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_extend(loom_text_embed_t* embed,
                                                      uint32_t row,
                                                      uint64_t extent) {
  return loom_text_embed_row_begin(embed, row, NULL, extent, 0);
}

LOOM_TEXT_EMBED_EXPORT uint64_t
loom_text_embed_row_extent(loom_text_embed_t* embed, uint32_t row) {
  return embed->extents[row];
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_bind(
    loom_text_embed_t* embed, uint32_t row,
    const loom_text_embed_request_t* request) {
  iree_status_t status = loom_text_embed_set_logit_row(embed, row, request);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_row_bind_request(loom_text_embed_row(embed, row),
                                              request->chat.options);
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_row_end(loom_text_embed_t* embed,
                                                    uint32_t row) {
  loom_serve_text_row_release_reservation(loom_text_embed_row(embed, row));
  embed->extents[row] = 0;
  loom_text_embed_clear_logit_row(embed, row);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_reset(loom_text_embed_t* embed,
                                                     uint32_t row) {
  loom_text_embed_clear_logit_row(embed, row);
  EMBED_RETURN(embed, loom_serve_text_row_reset(loom_text_embed_row(embed, row)));
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_prefill(
    loom_text_embed_t* embed, uint32_t row, const int32_t* tokens,
    size_t count) {
  loom_serve_text_row_t* handle = loom_text_embed_row(embed, row);
  iree_status_t status = iree_ok_status();
  for (size_t offset = 0; offset < count && iree_status_is_ok(status);) {
    const size_t chunk = iree_min(count - offset, embed->prefill_capacity);
    status = loom_serve_text_row_prefill(handle, chunk, tokens + offset);
    offset += chunk;
  }
  if (iree_status_is_ok(status) && count) {
    // Only the last chunk's selection is the row's pending one.
    status = loom_text_embed_collect_logits(embed, &row, 1);
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_decode(loom_text_embed_t* embed,
                                                      uint32_t row) {
  iree_status_t status =
      loom_serve_text_row_decode(loom_text_embed_row(embed, row));
  if (iree_status_is_ok(status)) {
    status = loom_text_embed_collect_logits(embed, &row, 1);
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int32_t loom_text_embed_row_token(
    loom_text_embed_t* embed, uint32_t row) {
  return loom_serve_text_row_token(loom_text_embed_row(embed, row));
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_is_eos(loom_text_embed_t* embed,
                                                      uint32_t row) {
  return loom_serve_text_row_is_eos(loom_text_embed_row(embed, row));
}

LOOM_TEXT_EMBED_EXPORT uint64_t
loom_text_embed_row_position(loom_text_embed_t* embed, uint32_t row) {
  return loom_serve_text_row_position(loom_text_embed_row(embed, row));
}

// Smallest prepared shape admitting the spans and tokens.
static bool loom_text_embed_shape(const loom_text_embed_t* embed,
                                  iree_host_size_t spans,
                                  iree_host_size_t tokens,
                                  iree_host_size_t* out_index) {
  bool found = false;
  for (iree_host_size_t i = 0; i < embed->shape_count; ++i) {
    const loom_serve_packing_shape_t* shape = &embed->shapes[i];
    if (shape->span_capacity < spans || shape->token_capacity < tokens) {
      continue;
    }
    if (!found ||
        shape->token_capacity < embed->shapes[*out_index].token_capacity ||
        (shape->token_capacity == embed->shapes[*out_index].token_capacity &&
         shape->span_capacity < embed->shapes[*out_index].span_capacity)) {
      *out_index = i;
      found = true;
    }
  }
  return found;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_decode_batch(
    loom_text_embed_t* embed, const uint32_t* rows, size_t count) {
  if (count == 1) return loom_text_embed_row_decode(embed, rows[0]);
  if (count > LOOM_TEXT_EMBED_MAX_BATCH) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "decode batch too wide"));
  }
  iree_host_size_t shape = 0;
  if (!loom_text_embed_shape(embed, count, count, &shape)) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "no shape for %zu decode spans",
                                         count));
  }
  int32_t pending[LOOM_TEXT_EMBED_MAX_BATCH];
  loom_serve_text_span_t spans[LOOM_TEXT_EMBED_MAX_BATCH];
  for (size_t i = 0; i < count; ++i) {
    pending[i] = loom_text_embed_row_token(embed, rows[i]);
    spans[i] = (loom_serve_text_span_t){
        .row_index = rows[i],
        .token_count = 1,
        .token_ids = &pending[i],
        .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
    };
  }
  iree_status_t status =
      loom_serve_text_model_epoch(embed->model, shape, count, spans);
  if (iree_status_is_ok(status)) {
    status = loom_text_embed_collect_logits(embed, rows, count);
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_decode_batch_tokens(
    loom_text_embed_t* embed, const uint32_t* rows, size_t count,
    const int32_t* tokens) {
  if (!count || count > LOOM_TEXT_EMBED_MAX_BATCH) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "decode batch of %zu rows", count));
  }
  iree_host_size_t shape = 0;
  if (!loom_text_embed_shape(embed, count, count, &shape)) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "no shape for %zu decode spans",
                                         count));
  }
  int32_t input[LOOM_TEXT_EMBED_MAX_BATCH];
  loom_serve_text_span_t spans[LOOM_TEXT_EMBED_MAX_BATCH];
  for (size_t i = 0; i < count; ++i) {
    input[i] = tokens[i];
    spans[i] = (loom_serve_text_span_t){
        .row_index = rows[i],
        .token_count = 1,
        .token_ids = &input[i],
        .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
    };
  }
  iree_status_t status =
      loom_serve_text_model_epoch(embed->model, shape, count, spans);
  if (iree_status_is_ok(status)) {
    status = loom_text_embed_collect_logits(embed, rows, count);
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_verify_batch(
    loom_text_embed_t* embed, const uint32_t* rows, size_t count,
    const uint32_t* limits, int32_t* tokens, uint32_t* counts,
    uint32_t* consumed) {
  const iree_host_size_t width = embed->draft_depth + 1;
  for (size_t i = 0; i < count; ++i) {
    if (rows[i] < embed->rows && embed->logit_flag[rows[i]]) {
      EMBED_RETURN(embed, iree_make_status(
                              IREE_STATUS_FAILED_PRECONDITION,
                              "logit rows run undrafted: use decode_batch"));
    }
  }
  if (!embed->draft_depth || count > LOOM_TEXT_EMBED_MAX_BATCH) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                         "drafting is unavailable"));
  }
  iree_host_size_t shape = 0;
  iree_host_size_t total = 0;
  for (size_t i = 0; i < count; ++i) total += limits[i] ? width : 1;
  if (!loom_text_embed_shape(embed, count, total, &shape)) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "no shape for %zu drafted spans",
                                         count));
  }
  int32_t pending[LOOM_TEXT_EMBED_MAX_BATCH];
  loom_serve_text_span_t spans[LOOM_TEXT_EMBED_MAX_BATCH];
  loom_serve_text_result_t results[LOOM_TEXT_EMBED_MAX_BATCH];
  for (size_t i = 0; i < count; ++i) {
    pending[i] = loom_text_embed_row_token(embed, rows[i]);
    spans[i] = (loom_serve_text_span_t){
        .row_index = rows[i],
        .token_count = limits[i] ? width : 1,
        .token_ids = &pending[i],
        .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT |
                 (limits[i] ? LOOM_SERVICE_TEXT_SPAN_FLAG_PROPOSE : 0),
    };
  }
  memset(results, 0, sizeof(results[0]) * count);
  iree_status_t status = loom_serve_text_model_verify(
      embed->model, shape, count, spans, limits, NULL, results);
  if (iree_status_is_ok(status)) {
    for (size_t i = 0; i < count; ++i) {
      counts[i] = results[i].output_count;
      consumed[i] = results[i].consumed_count;
      memcpy(tokens + 16 * i, results[i].tokens, sizeof(results[i].tokens));
    }
  }
  EMBED_RETURN(embed, status);
}

struct loom_text_embed_decoder_t {
  loom_text_embed_t* embed;
  iree_tokenizer_decode_state_t* state;
  iree_host_size_t storage_size;
  uint8_t storage[];
};

static iree_status_t loom_text_embed_decoder_initialize(
    loom_text_embed_decoder_t* decoder) {
  return iree_tokenizer_decode_state_initialize(
      loom_serve_text_model_tokenizer(decoder->embed->model),
      IREE_TOKENIZER_DECODE_FLAG_NONE,
      iree_make_byte_span(decoder->storage, decoder->storage_size),
      &decoder->state);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_decoder_open(
    loom_text_embed_t* embed, loom_text_embed_decoder_t** out_decoder) {
  *out_decoder = NULL;
  iree_host_size_t size = 0;
  iree_status_t status = iree_tokenizer_decode_state_calculate_size(
      loom_serve_text_model_tokenizer(embed->model), &size);
  loom_text_embed_decoder_t* decoder = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(embed->allocator, sizeof(*decoder) + size,
                                   (void**)&decoder);
  }
  if (iree_status_is_ok(status)) {
    decoder->embed = embed;
    decoder->state = NULL;
    decoder->storage_size = size;
    status = loom_text_embed_decoder_initialize(decoder);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(embed->allocator, decoder);
      decoder = NULL;
    }
  }
  *out_decoder = decoder;
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_decoder_feed(
    loom_text_embed_decoder_t* decoder, const int32_t* tokens, size_t count,
    char* text, size_t capacity, size_t* out_size) {
  *out_size = 0;
  size_t fed = 0;
  while (fed < count) {
    iree_host_size_t consumed = 0, written = 0;
    iree_status_t status = iree_tokenizer_decode_state_feed(
        decoder->state,
        (iree_tokenizer_token_id_list_t){count - fed, tokens + fed},
        iree_make_mutable_string_view(text + *out_size, capacity - *out_size),
        &consumed, &written);
    if (!iree_status_is_ok(status)) EMBED_RETURN(decoder->embed, status);
    fed += consumed;
    *out_size += written;
    if (!consumed && !written) {
      EMBED_RETURN(decoder->embed,
                   iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "decoder output buffer full"));
    }
  }
  return 0;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_decoder_reset(
    loom_text_embed_decoder_t* decoder) {
  iree_tokenizer_decode_state_deinitialize(decoder->state);
  decoder->state = NULL;
  EMBED_RETURN(decoder->embed, loom_text_embed_decoder_initialize(decoder));
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_decoder_free(
    loom_text_embed_decoder_t* decoder) {
  if (!decoder) return;
  if (decoder->state) iree_tokenizer_decode_state_deinitialize(decoder->state);
  iree_allocator_free(decoder->embed->allocator, decoder);
}

// Appends to out, failing on overflow.
static iree_status_t loom_text_embed_append(char* out, size_t capacity,
                                            size_t* size,
                                            iree_string_view_t text) {
  if (*size + text.size > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "normalized output buffer full");
  }
  memcpy(out + *size, text.data, text.size);
  *size += text.size;
  return iree_ok_status();
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_output_feed(
    loom_text_embed_request_t* request, const char* text, size_t text_size,
    int final, char* out, size_t capacity, size_t* out_size) {
  loom_text_embed_t* embed = request->embed;
  *out_size = 0;
  if (request->finished) return 0;
  iree_status_t status = iree_string_builder_append_string(
      &request->response, iree_make_string_view(text, text_size));
  const iree_string_view_t response =
      iree_string_builder_view(&request->response);
  const loom_serve_text_chat_output_phase_t phase =
      final ? LOOM_SERVE_TEXT_CHAT_OUTPUT_COMPLETE
            : LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING;
  while (iree_status_is_ok(status)) {
    loom_serve_text_chat_output_t output = {0};
    status = loom_serve_text_chat_output_next(&request->chat, response, phase,
                                              embed->allocator, &output);
    if (!iree_status_is_ok(status) || !output.advanced) break;
    if (output.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_HIDDEN) continue;
    const bool reasoning = output.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_REASONING;
    if (output.text.size) {
      const size_t begin = (size_t)(output.text.data - response.data);
      if (reasoning) {
        request->reasoned = true;
        request->reasoning_end = begin + output.text.size;
      } else if (request->reasoned && !request->closed) {
        request->closed = true;
        request->close_begin = request->reasoning_end;
        request->close_end = begin;
      }
    }
    if (reasoning != request->reasoning_open && output.text.size) {
      status = loom_text_embed_append(
          out, capacity, out_size,
          reasoning ? IREE_SV("<think>") : IREE_SV("</think>"));
      request->reasoning_open = reasoning;
    }
    if (iree_status_is_ok(status)) {
      status = loom_text_embed_append(out, capacity, out_size, output.text);
    }
  }
  if (iree_status_is_ok(status) && final) {
    if (request->reasoning_open) {
      status = loom_text_embed_append(out, capacity, out_size,
                                      IREE_SV("</think>"));
      request->reasoning_open = false;
    }
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_output_close(
    loom_text_embed_request_t* request, char* out, size_t capacity,
    size_t* out_size) {
  *out_size = 0;
  if (!request->closed) return 0;
  const iree_string_view_t response =
      iree_string_builder_view(&request->response);
  EMBED_RETURN(request->embed,
               loom_text_embed_append(
                   out, capacity, out_size,
                   iree_make_string_view(
                       response.data + request->close_begin,
                       request->close_end - request->close_begin)));
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_output_tools(
    loom_text_embed_request_t* request, int truncated, char* json,
    size_t capacity, size_t* out_size) {
  loom_text_embed_t* embed = request->embed;
  *out_size = 0;
  request->finished = true;
  loom_serve_text_tool_calls_t calls;
  loom_serve_text_tool_calls_initialize(embed->allocator, &calls);
  loom_serve_text_chat_scratch_t scratch;
  loom_serve_text_chat_scratch_initialize(embed->allocator, &scratch);
  loom_serve_text_chat_completion_t completion = {0};
  iree_status_t status = loom_serve_text_chat_complete(
      &request->chat, iree_string_builder_view(&request->response),
      truncated != 0, request->serial, &calls, &scratch, embed->allocator,
      &completion);
  iree_string_builder_t builder;
  iree_string_builder_initialize(embed->allocator, &builder);
  if (iree_status_is_ok(status) && calls.count) {
    status = loom_serve_text_tool_calls_write_json(&calls, false, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_text_embed_append(json, capacity, out_size,
                                    iree_string_builder_view(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  loom_serve_text_chat_completion_deinitialize(&completion);
  loom_serve_text_chat_scratch_deinitialize(&scratch);
  loom_serve_text_tool_calls_deinitialize(&calls);
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_pin(
    loom_text_embed_t* embed, uint32_t row,
    loom_text_embed_checkpoint_t** out_checkpoint) {
  loom_serve_text_checkpoint_t* checkpoint = NULL;
  iree_status_t status =
      loom_serve_text_row_try_pin(loom_text_embed_row(embed, row), &checkpoint);
  *out_checkpoint = (loom_text_embed_checkpoint_t*)checkpoint;
  if (iree_status_is_ok(status) && !checkpoint) return 1;
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_restore(
    loom_text_embed_t* embed, uint32_t row,
    const loom_text_embed_checkpoint_t* checkpoint, int* restored) {
  bool done = false;
  loom_serve_text_row_t* handle = loom_text_embed_row(embed, row);
  loom_serve_text_row_release_reservation(handle);
  embed->extents[row] = 0;
  embed->logit_valid[row] = 0;
  iree_status_t status = loom_serve_text_row_try_restore(
      handle, (const loom_serve_text_checkpoint_t*)checkpoint, &done);
  *restored = done;
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_checkpoint_release(
    loom_text_embed_t* embed, loom_text_embed_checkpoint_t* checkpoint) {
  (void)embed;
  loom_serve_text_checkpoint_release(
      (loom_serve_text_checkpoint_t*)checkpoint);
}


//===----------------------------------------------------------------------===//
// Checkpoint images: a checkpoint's retained state as portable host bytes.
//===----------------------------------------------------------------------===//

// Image layout: the fixed part (recurrent state, then carry) and one page per
// block of block_size tokens, each page holding that block of every cache
// plane. Loom's own snapshot order is plane-major; ranges below map between.
typedef struct loom_text_embed_image_plan_t {
  iree_host_size_t count;
  // Device ranges in Loom's snapshot order (for this checkpoint's IDs).
  loom_runtime_snapshot_range_t* device;
  // The same logical ranges at their image offsets (buffer 0 = host image).
  loom_runtime_snapshot_range_t* image;
} loom_text_embed_image_plan_t;

static loom_serve_text_state_t* loom_text_embed_state(loom_text_embed_t* embed) {
  return &embed->model->state;
}

static iree_device_size_t loom_text_embed_recurrent_bytes(
    loom_serve_text_state_t* state) {
  return state->recurrent.buffers && state->recurrent.buffers[0]
             ? iree_hal_buffer_byte_length(state->recurrent.buffers[0])
             : 0;
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_image_layout(
    loom_text_embed_t* embed, loom_text_embed_image_layout_t* out_layout) {
  loom_serve_text_state_t* state = loom_text_embed_state(embed);
  const loom_serve_text_storage_t* storage = &state->storage;
  uint64_t page = 0;
  for (iree_host_size_t r = 0; r < storage->region_count; ++r) {
    page += (uint64_t)storage->regions[r].blocks.count *
            storage->regions[r].blocks.block_bytes;
  }
  out_layout->block_size = (uint32_t)storage->block_size;
  out_layout->fixed_bytes =
      loom_text_embed_recurrent_bytes(state) + storage->carry_stride;
  out_layout->page_bytes = page;
  out_layout->supported = state->memory.buffers[1] != NULL;
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_memory(
    const loom_text_embed_t* embed, loom_text_embed_memory_t* out_memory) {
  memset(out_memory, 0, sizeof(*out_memory));
  // A fixed pool's backing is all native; an elastic one's is its committed
  // virtual memory.
  const loom_runtime_pool_t* pool =
      loom_runtime_device_group_pool_at(embed->group, 0);
  const loom_runtime_pool_statistics_t pool_statistics =
      loom_runtime_pool_statistics(pool);
  out_memory->device_bytes =
      loom_runtime_pool_memory(pool)
          ? pool_statistics.retained.committed_bytes
          : (uint64_t)pool_statistics.storage.bytes_reserved;
  const loom_serve_text_pool_usage_t usage =
      loom_serve_text_model_pool_usage(embed->model);
  out_memory->state_tokens = usage.capacity;
  out_memory->free_tokens = usage.available;
  out_memory->block_size = (uint32_t)usage.block_size;
  const loom_serve_text_storage_t* storage = &embed->model->state.storage;
  for (iree_host_size_t r = 0; r < storage->region_count; ++r) {
    out_memory->page_bytes += (uint64_t)storage->regions[r].blocks.count *
                              storage->regions[r].blocks.block_bytes;
  }
}

// recurrent_slot UINT32_MAX plans a suspended checkpoint: only lengths and
// the image side are meaningful then.
static iree_status_t loom_text_embed_image_plan(
    loom_serve_text_state_t* state, iree_host_size_t checkpoint_index,
    uint32_t recurrent_slot, uint32_t block_count, const uint32_t* blocks,
    loom_text_embed_image_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  const loom_serve_text_storage_t* storage = &state->storage;
  iree_hal_buffer_t* views[5] = {0};
  if (state->recurrent.buffers && state->recurrent.buffers[0]) {
    views[TEXT_RECURRENT - TEXT_CONTROL] =
        state->recurrent.buffers[recurrent_slot == UINT32_MAX ? 0
                                                              : recurrent_slot];
  }
  // Loom merges consecutive physical pages into one range (and may group
  // capture and restore differently), so its ranges are split back into one
  // range per block and plane: the image layout must not depend on page IDs.
  iree_host_size_t merged_count = 0;
  loom_runtime_snapshot_range_t* merged = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_storage_plan_snapshot(
      storage, state->row_count + checkpoint_index, views, true, block_count,
      blocks, state->host_allocator, &merged_count, &merged));
  const iree_host_size_t fixed_count =
      (views[TEXT_RECURRENT - TEXT_CONTROL] ? 1 : 0) +
      (storage->carry_stride ? 1 : 0);
  uint64_t page = 0;
  iree_host_size_t unit_count = 0;
  for (iree_host_size_t r = 0; r < storage->region_count; ++r) {
    page += (uint64_t)storage->regions[r].blocks.count *
            storage->regions[r].blocks.block_bytes;
    unit_count += storage->regions[r].blocks.count;
  }
  out_plan->count = fixed_count + unit_count * block_count;
  iree_status_t status =
      merged_count < fixed_count
          ? iree_make_status(IREE_STATUS_INTERNAL,
                             "snapshot plan has no fixed ranges")
          : iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(state->host_allocator,
                                         2 * out_plan->count,
                                         sizeof(*out_plan->device),
                                         (void**)&out_plan->device);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(state->host_allocator, merged);
    out_plan->count = 0;
    return status;
  }
  out_plan->image = out_plan->device + out_plan->count;
  uint64_t fixed = 0;
  for (iree_host_size_t i = 0; i < fixed_count; ++i) {
    out_plan->device[i] = merged[i];
    out_plan->image[i] = (loom_runtime_snapshot_range_t){0, fixed,
                                                         merged[i].length};
    fixed += merged[i].length;
  }
  // Logical order: per region and plane, blocks in order.
  iree_host_size_t i = fixed_count, source = fixed_count;
  uint64_t consumed = 0, unit_offset = 0;
  for (iree_host_size_t r = 0;
       r < storage->region_count && iree_status_is_ok(status); ++r) {
    const loom_runtime_block_region_t* region = &storage->regions[r].blocks;
    for (iree_host_size_t plane = 0;
         plane < region->count && iree_status_is_ok(status); ++plane) {
      for (uint32_t block = 0; block < block_count; ++block, ++i) {
        if (source >= merged_count ||
            merged[source].length - consumed < region->block_bytes) {
          status = iree_make_status(IREE_STATUS_INTERNAL,
                                    "snapshot ranges do not cover whole pages");
          break;
        }
        out_plan->device[i] = (loom_runtime_snapshot_range_t){
            merged[source].buffer_index, merged[source].offset + consumed,
            region->block_bytes};
        out_plan->image[i] = (loom_runtime_snapshot_range_t){
            0, fixed + block * page + unit_offset, region->block_bytes};
        consumed += region->block_bytes;
        if (consumed == merged[source].length) {
          ++source;
          consumed = 0;
        }
      }
      unit_offset += region->block_bytes;
    }
  }
  if (iree_status_is_ok(status) && source != merged_count) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "snapshot ranges exceed the checkpoint's pages");
  }
  iree_allocator_free(state->host_allocator, merged);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(state->host_allocator, out_plan->device);
    memset(out_plan, 0, sizeof(*out_plan));
  }
  return status;
}

static void loom_text_embed_image_plan_free(loom_serve_text_state_t* state,
                                            loom_text_embed_image_plan_t* plan) {
  iree_allocator_free(state->host_allocator, plan->device);
}

// Moves every range between device state and the image, 64 at a time, then
// waits for the transfers.
static iree_status_t loom_text_embed_image_transfer(
    loom_serve_text_state_t* state, const loom_text_embed_image_plan_t* plan,
    uint8_t* image, bool upload) {
  iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    buffers[i] = *loom_serve_text_state_allocation(state, i);
  }
  iree_hal_transfer_operation_t operations[64];
  iree_host_size_t operation_count = 0;
  uint64_t completion = 0;
  iree_status_t status = loom_runtime_execution_drain(state->execution);
  for (iree_host_size_t i = 0; i < plan->count && iree_status_is_ok(status);
       ++i) {
    const loom_runtime_snapshot_range_t* device = &plan->device[i];
    uint8_t* bytes = image + plan->image[i].offset;
    operations[operation_count++] =
        upload ? (iree_hal_transfer_operation_t){
                     .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
                     .upload = {.source = bytes,
                                .target_buffer = buffers[device->buffer_index],
                                .target_offset = device->offset,
                                .length = device->length}}
               : (iree_hal_transfer_operation_t){
                     .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
                     .download = {.source_buffer = buffers[device->buffer_index],
                                  .source_offset = device->offset,
                                  .target = bytes,
                                  .length = device->length}};
    if (operation_count == IREE_ARRAYSIZE(operations)) {
      status = loom_runtime_execution_transfer(state->execution, operation_count,
                                             operations, &completion);
      operation_count = 0;
    }
  }
  if (iree_status_is_ok(status) && operation_count) {
    status = loom_runtime_execution_transfer(state->execution, operation_count,
                                           operations, &completion);
  }
  return iree_status_join(status, loom_runtime_execution_drain(state->execution));
}

static iree_device_size_t loom_text_embed_image_size(
    const loom_text_embed_image_plan_t* plan) {
  iree_device_size_t size = 0;
  for (iree_host_size_t i = 0; i < plan->count; ++i) size += plan->image[i].length;
  return size;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_checkpoint_export(
    loom_text_embed_t* embed, const loom_text_embed_checkpoint_t* handle,
    uint8_t* image, size_t capacity, size_t* out_size,
    loom_text_embed_checkpoint_info_t* out_info) {
  const loom_serve_text_checkpoint_t* checkpoint =
      (const loom_serve_text_checkpoint_t*)handle;
  loom_serve_text_state_t* state = loom_text_embed_state(embed);
  loom_serve_text_state_checkpoint_t* endpoint = checkpoint->state;
  *out_size = 0;
  out_info->position = endpoint->position;
  out_info->source_position = endpoint->source_position;
  out_info->has_prediction = checkpoint->has_prediction;
  out_info->token = checkpoint->token;
  if (endpoint->snapshot || endpoint->recurrent_slot == UINT32_MAX) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                         "checkpoint is not resident"));
  }
  loom_text_embed_image_plan_t plan;
  iree_status_t status = loom_text_embed_image_plan(
      state, (iree_host_size_t)(endpoint - state->checkpoints.values),
      endpoint->recurrent_slot, endpoint->block_count, endpoint->blocks, &plan);
  if (!iree_status_is_ok(status)) EMBED_RETURN(embed, status);
  const iree_device_size_t size = loom_text_embed_image_size(&plan);
  *out_size = size;
  if (size > capacity) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "checkpoint image needs %" PRIu64 " bytes",
                              (uint64_t)size);
  } else {
    status = loom_text_embed_image_transfer(state, &plan, image, false);
  }
  loom_text_embed_image_plan_free(state, &plan);
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_checkpoint_import(
    loom_text_embed_t* embed, const loom_text_embed_checkpoint_info_t* info,
    const uint8_t* image, size_t size,
    loom_text_embed_checkpoint_t** out_checkpoint) {
  *out_checkpoint = NULL;
  loom_serve_text_state_t* state = loom_text_embed_state(embed);
  if (!state->memory.buffers[1]) {
    EMBED_RETURN(embed,
                 iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "checkpoint images need elastic pooled state"));
  }
  if (info->position == 0 || info->position > state->storage.blocks_per_row *
                                                  state->storage.block_size) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                         "checkpoint image position %u is out "
                                         "of range",
                                         info->position));
  }
  const uint32_t needed = (uint32_t)iree_host_size_ceil_div(
      info->position, state->storage.block_size);
  // Page IDs grow with demand (Loom 0.2.0): after a restart the pool is still
  // empty. Grow it as Loom does when it resumes a suspended row.
  loom_runtime_block_pool_t* pages = &state->cache.pool;
  if (needed > pages->available) {
    const uint32_t maximum =
        (uint32_t)(state->cache.capacity / state->storage.block_size);
    const uint32_t live = pages->capacity - pages->available;
    if (needed <= maximum - live) {
      uint64_t capacity = pages->capacity ? 2ull * pages->capacity : 16;
      capacity = iree_max(capacity, (uint64_t)live + needed);
      capacity = iree_min(capacity, (uint64_t)maximum);
      iree_status_t status =
          loom_runtime_block_pool_grow(pages, (uint32_t)capacity);
      if (!iree_status_is_ok(status)) EMBED_RETURN(embed, status);
    }
  }
  if (!state->checkpoints.pool.available || !state->recurrent.pool.available ||
      needed > pages->available) {
    return 1;
  }
  // As Loom resumes a suspended endpoint: fresh IDs, committed backing, then
  // the image's contents in logical order.
  uint32_t index = 0;
  loom_runtime_block_pool_acquire(&state->checkpoints.pool, 1, &index);
  loom_serve_text_state_checkpoint_t* endpoint =
      &state->checkpoints.values[index];
  endpoint->position = info->position;
  endpoint->source_position = info->source_position;
  loom_runtime_block_pool_acquire(&state->cache.pool, needed, endpoint->blocks);
  endpoint->block_count = needed;
  loom_runtime_block_pool_acquire(&state->recurrent.pool, 1,
                                &endpoint->recurrent_slot);
  loom_text_embed_image_plan_t plan;
  iree_status_t status = loom_text_embed_image_plan(
      state, index, endpoint->recurrent_slot, needed, endpoint->blocks, &plan);
  bool admitted = false;
  if (iree_status_is_ok(status) && loom_text_embed_image_size(&plan) != size) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "checkpoint image is %zu bytes, expected %" PRIu64,
                              size, (uint64_t)loom_text_embed_image_size(&plan));
  }
  loom_runtime_memory_range_t* commitments = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(state->host_allocator, plan.count + 1,
                                         sizeof(*commitments),
                                         (void**)&commitments);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t count = 0;
    for (iree_host_size_t i = 0; i < plan.count; ++i) {
      loom_runtime_virtual_buffer_t* buffer =
          state->memory.buffers[plan.device[i].buffer_index];
      if (buffer) {
        commitments[count++] = (loom_runtime_memory_range_t){
            buffer, plan.device[i].offset, plan.device[i].length};
      }
    }
    status = loom_runtime_memory_pool_try_commit(state->memory.pool, count,
                                               commitments, &admitted);
  }
  iree_allocator_free(state->host_allocator, commitments);
  if (iree_status_is_ok(status) && admitted) {
    status = loom_text_embed_image_transfer(state, &plan, (uint8_t*)image, true);
  }
  if (plan.device) loom_text_embed_image_plan_free(state, &plan);
  if (!iree_status_is_ok(status) || !admitted) {
    loom_serve_text_state_checkpoint_release(endpoint);
    if (!iree_status_is_ok(status)) EMBED_RETURN(embed, status);
    return 1;
  }
  loom_serve_text_checkpoint_t* checkpoint = &embed->model->checkpoints[index];
  *checkpoint = (loom_serve_text_checkpoint_t){
      .model = embed->model,
      .references = 1,
      .state = endpoint,
      .has_prediction = info->has_prediction != 0,
      .token = info->token};
  *out_checkpoint = (loom_text_embed_checkpoint_t*)checkpoint;
  return 0;
}

LOOM_TEXT_EMBED_EXPORT size_t loom_text_embed_request_media(
    const loom_text_embed_request_t* request, uint32_t* offsets,
    uint32_t* counts, size_t capacity) {
  if (!request->has_input) return 0;
  const iree_host_size_t count = request->input.media_span_count;
  for (iree_host_size_t i = 0; i < count && i < capacity; ++i) {
    offsets[i] = request->input.media_spans[i].input_offset;
    counts[i] = request->input.media_spans[i].input_count;
  }
  return count;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_request_project(
    loom_text_embed_request_t* request) {
  loom_text_embed_t* embed = request->embed;
  if (!request->has_input || request->media) return 0;
  // Prepare the image component on first use.
  loom_service_finite_model_t* image_model = NULL;
  iree_status_t status = iree_ok_status();
  for (;;) {
    status = loom_service_finite_component_prepare(embed->image, &image_model);
    if (!iree_status_is_ok(status) || image_model) break;
    loom_service_finite_component_advance(embed->image, false);
    loom_text_embed_pause(200);
  }
  if (iree_status_is_ok(status) && !embed->projector) {
    loom_runtime_host_t* host = loom_runtime_device_host(
        loom_serve_text_model_device(embed->model));
    iree_async_proactor_t* proactor = NULL;
    status = iree_async_proactor_pool_get(loom_runtime_host_proactor_pool(host),
                                          0, &proactor);
    if (iree_status_is_ok(status)) {
      const iree_host_size_t capacity =
          embed->rows *
          loom_serve_text_model_chat_policy(embed->model)->images.maximum_count;
      status = loom_serve_text_media_projector_allocate(
          image_model, capacity ? capacity : 1, loom_runtime_host_executor(host),
          proactor, (loom_runtime_wake_callback_t){loom_text_embed_wake, NULL},
          embed->allocator, &embed->projector);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_media_request_allocate(
        &request->chat, embed->allocator, &request->media);
  }
  // Offer, seal and advance the projection window until every image of the
  // request has published its features.
  bool offered = false;
  while (iree_status_is_ok(status)) {
    status = loom_serve_text_media_request_take_status(request->media);
    if (!iree_status_is_ok(status)) break;
    if (loom_serve_text_media_request_bind(request->media, &request->input, 0))
      break;
    if (!offered) {
      offered = loom_serve_text_media_projector_offer(embed->projector,
                                                      request->media, 0);
      loom_serve_text_media_projector_start(embed->projector);
    }
    bool progress = false;
    status = loom_serve_text_media_projector_advance(embed->projector, 0,
                                                     &progress);
    loom_service_finite_component_advance(embed->image, false);
    if (!progress) loom_text_embed_pause(100);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_text_media_request_begin(request->media);
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_prefill_request(
    loom_text_embed_t* embed, uint32_t row, loom_text_embed_request_t* request,
    size_t offset, size_t count, int select) {
  if (!request->has_input || offset + count > request->input.count) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "request input range"));
  }
  iree_host_size_t shape = 0;
  if (!loom_text_embed_shape(embed, 1, count, &shape)) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "no shape for %zu input tokens",
                                         count));
  }
  const loom_serve_text_span_t span = {
      .row_index = row,
      .token_count = count,
      .token_ids = request->input.tokens + offset,
      .flags = select ? LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT : 0,
      .input_offset = offset,
      .media = request->media
                   ? loom_serve_text_media_request_input(request->media)
                   : NULL,
  };
  iree_status_t status =
      loom_serve_text_model_epoch(embed->model, shape, 1, &span);
  if (iree_status_is_ok(status) && request->media) {
    loom_serve_text_media_request_consume(request->media, offset + count);
  }
  if (iree_status_is_ok(status) && select) {
    status = loom_text_embed_collect_logits(embed, &row, 1);
  }
  EMBED_RETURN(embed, status);
}

// libamdf, the driver under Loom's amdgpu devices, reads the same endpoints on
// Linux (KFD) and Windows (WDDM). The default device URI opens the first GPU
// endpoint. On Windows local memory is known once a device exists, so one is
// created and destroyed here. On Linux a device binds this process's KFD VM to
// its own render file for good (KFD has no inverse of ACQUIRE_VM), so the
// serving device would then fail with EBUSY: memory comes from the render
// node's sysfs instead.
static amdf_status_t loom_text_embed_query_endpoint(
    const amdf_api_t* api, const amdf_gpu_api_t* gpu_api,
    amdf_instance_t* instance, loom_text_embed_gpu_t* out_gpu) {
  uint32_t count = 0;
  amdf_status_t status = api->endpoint_enumerate(instance, 0, NULL, &count);
  if (amdf_status_code(status) == AMDF_STATUS_CODE_BUFFER_TOO_SMALL) {
    status = AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(status) || count == 0) return status;
  amdf_endpoint_summary_t* summaries =
      (amdf_endpoint_summary_t*)calloc(count, sizeof(*summaries));
  if (!summaries) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  status = api->endpoint_enumerate(instance, count, summaries, &count);
  // The first endpoint that answers as a GPU; a summary may not know its
  // engine yet.
  amdf_endpoint_t* endpoint = NULL;
  amdf_gpu_endpoint_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
      .structure_size = sizeof(info),
  };
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < count && !endpoint;
       ++i) {
    if (summaries[i].engine_kind != AMDF_ENGINE_KIND_GPU &&
        summaries[i].engine_kind != AMDF_ENGINE_KIND_UNKNOWN) {
      continue;
    }
    if (!amdf_status_is_ok(
            api->endpoint_open(instance, &summaries[i].id, &endpoint))) {
      endpoint = NULL;
      continue;
    }
    if (!amdf_status_is_ok(gpu_api->endpoint_query_info(endpoint, &info))) {
      api->endpoint_close(endpoint);
      endpoint = NULL;
    }
  }
  free(summaries);
  if (!endpoint) return status;
  {
    snprintf(out_gpu->target, sizeof(out_gpu->target), "gfx%u%u%x",
             info.gfx_ip.major, info.gfx_ip.minor, info.gfx_ip.stepping);
#if !defined(_WIN32)
    amdf_endpoint_info_t endpoint_info = {
        .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
        .structure_size = sizeof(endpoint_info),
    };
    // Memory is a refinement: the target stands without it.
    if (amdf_status_is_ok(api->endpoint_query_info(endpoint, &endpoint_info)) &&
        endpoint_info.native_identity.type ==
            AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE) {
      char path[96];
      snprintf(path, sizeof(path),
               "/sys/dev/char/%u:%u/device/mem_info_vram_total",
               endpoint_info.native_identity.value.linux_device.major,
               endpoint_info.native_identity.value.linux_device.minor);
      FILE* file = fopen(path, "r");
      if (file) {
        unsigned long long bytes = 0;
        if (fscanf(file, "%llu", &bytes) == 1) out_gpu->memory_bytes = bytes;
        fclose(file);
      }
    }
#else
    const amdf_gpu_device_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
        .structure_size = sizeof(create_info),
    };
    amdf_device_t* device = NULL;
    // Memory is a refinement: the target stands without it.
    if (amdf_status_is_ok(
            gpu_api->device_create(endpoint, &create_info, &device))) {
      amdf_memory_scope_t* scopes[8] = {0};
      uint32_t scope_count = 0;
      amdf_status_t scope_status = api->device_enumerate_memory_scopes(
          device, 8, scopes, &scope_count);
      if (amdf_status_code(scope_status) ==
          AMDF_STATUS_CODE_BUFFER_TOO_SMALL) {
        scope_status = AMDF_STATUS_OK;
        if (scope_count > 8) scope_count = 8;
      }
      for (uint32_t i = 0; amdf_status_is_ok(scope_status) && i < scope_count;
           ++i) {
        amdf_memory_scope_info_t scope = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
            .structure_size = sizeof(scope),
        };
        if (amdf_status_is_ok(api->memory_scope_query_info(scopes[i], &scope)) &&
            scope.kind == AMDF_MEMORY_SCOPE_KIND_LOCAL &&
            scope.capacity_byte_length > out_gpu->memory_bytes) {
          out_gpu->memory_bytes = scope.capacity_byte_length;
        }
      }
      api->device_destroy(device);
    }
#endif
  }
  api->endpoint_close(endpoint);
  return status;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_query_gpu(
    loom_text_embed_gpu_t* out_gpu, char* error, size_t error_size) {
  memset(out_gpu, 0, sizeof(*out_gpu));
  const amdf_api_t* api = NULL;
  const void* extension = NULL;
  amdf_instance_t* instance = NULL;
  amdf_status_t status =
      amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api);
  if (amdf_status_is_ok(status)) {
    status = api->query_extension(AMDF_EXTENSION_GPU,
                                  AMDF_GPU_EXTENSION_VERSION_1,
                                  AMDF_GPU_EXTENSION_VERSION_LATEST, &extension);
  }
  if (amdf_status_is_ok(status)) {
    const amdf_instance_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .structure_size = sizeof(create_info),
    };
    status = api->instance_create(&create_info, &instance);
  }
  if (amdf_status_is_ok(status)) {
    status = loom_text_embed_query_endpoint(
        api, (const amdf_gpu_api_t*)extension, instance, out_gpu);
    api->instance_destroy(instance);
  }
  if (amdf_status_is_ok(status)) return 0;
  if (error && error_size) {
    snprintf(error, error_size, "libamdf status domain %u code %u",
             (unsigned)amdf_status_domain(status),
             (unsigned)amdf_status_code(status));
  }
  return 2;
}
