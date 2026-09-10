/* DeepSeek V4 Flash GGUF loading, validation, and resident ROCm weight binding. */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <ctype.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <vector>

#include "dspark_internal.h"
#include "model.h"
#include "model_data_internal.h"
#include "native_internal.h"

#include "../kernels/rocm/resident_api.h"

enum class ds4_log_type : uint8_t {
    default_log,
    prefill,
    generation,
    kv_cache,
    tool,
    warning,
    timing,
    ok,
    error,
};

static uint32_t g_ds4_compress_ratios[DS4_N_LAYER] = {0};

static int g_ds4_lock_fd = -1;

/* =========================================================================
 * GGUF Quant Block Formats.
 * =========================================================================
 *
 * These layouts and IQ2 tables match the GGUF quantized tensor format,
 * reduced to only the formats ds4.c currently reads:
 *   - Q2_K routed down experts
 *   - Q4_K routed experts in the high-memory variant
 *   - IQ2_XXS routed gate/up experts
 */
#define QK_K 256

struct block_q2_K {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
};

struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
};

struct block_iq2_xxs {
    uint16_t d;
    uint16_t qs[QK_K / 8];
};

static_assert(sizeof(block_q2_K) == 84);
static_assert(sizeof(block_q4_K) == 144);
static_assert(sizeof(block_iq2_xxs) == 66);

/* =========================================================================
 * Shared Helpers, Allocation Guards, and Cursor Reads.
 * =========================================================================
 *
 * This section holds process-wide utilities used by all later stages:
 * fatal-error helpers, allocation wrappers, and the small byte cursor used to
 * parse GGUF metadata.
 */

#define DS4_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */
struct ds4_cursor {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
};

static void ds4_die(const char *msg) {
    fprintf(stderr, "ds4: %s\n", msg);
    exit(1);
}

/* Attention compression is read from GGUF metadata after validating that it
 * matches the exact layout expected for the loaded model shape. */
uint32_t ds4_layer_compress_ratio(uint32_t il) {
    if (il >= DS4_N_LAYER) ds4_die("DeepSeek4 layer index is outside the loaded model layout");
    return g_ds4_compress_ratios[il];
}

static uint32_t ds4_expected_layer_compress_ratio(uint32_t il) {
    if (il >= DS4_N_LAYER) ds4_die("DeepSeek4 layer index is outside the loaded model layout");
    if (il < 2) return 0;
    return (il & 1u) == 0 ? 4u : 128u;
}

static void ds4_die_errno(const char *what, const char *path) {
    fprintf(stderr, "ds4: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static bool ds4_streq(ds4_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static bool ds4_str_eq(ds4_str a, ds4_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

static uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const auto *p = static_cast<const uint8_t *>(ptr);
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

void *ds4_xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) ds4_die("out of memory");
    return p;
}

void *ds4_xmalloc(size_t size) {
    void *p = malloc(size);
    if (!p) ds4_die("out of memory");
    return p;
}

void *ds4_xrealloc(void *ptr, size_t size) {
    void *p = realloc(ptr, size);
    if (!p) ds4_die("out of memory");
    return p;
}

double ds4_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static const char *ds4_log_color_code(ds4_log_type type) {
    switch (type) {
    case ds4_log_type::prefill:
    case ds4_log_type::timing:
        return "\x1b[36m";
    case ds4_log_type::generation:
    case ds4_log_type::ok:
        return "\x1b[32m";
    case ds4_log_type::kv_cache:
        return "\x1b[33m";
    case ds4_log_type::tool:
        return "\x1b[90m";
    case ds4_log_type::warning:
        return "\x1b[38;5;208m";
    case ds4_log_type::error:
        return "\x1b[31m";
    case ds4_log_type::default_log:
        return "";
    }
    return "";
}

bool ds4_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

static void ds4_vlog(FILE *fp, ds4_log_type type, const char *fmt, va_list ap) {
    const bool colorize =
        type != ds4_log_type::default_log && ds4_log_is_tty(fp);
    if (colorize) fputs(ds4_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ds4_vlog(fp, type, fmt, ap);
    va_end(ap);
}

static void cursor_error(ds4_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(ds4_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(ds4_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(ds4_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(ds4_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(ds4_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(ds4_cursor *c, ds4_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

uint64_t ds4_align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

/* =========================================================================
 * GGUF Parsing and Model Mapping.
 * =========================================================================
 *
 * The loader maps the model once, records metadata/tensor descriptors, and
 * leaves tensor bytes in place.  Inference code accesses weights by adding
 * tensor offsets to the mapping instead of copying the GGUF into private
 * structures.
 */

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

struct gguf_type_info {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
};

static constexpr auto gguf_types = [] {
    std::array<gguf_type_info, 31> types{};
    types[0] = {"f32", 1, 4};
    types[1] = {"f16", 1, 2};
    types[2] = {"q4_0", 32, 18};
    types[3] = {"q4_1", 32, 20};
    types[6] = {"q5_0", 32, 22};
    types[7] = {"q5_1", 32, 24};
    types[8] = {"q8_0", 32, 34};
    types[9] = {"q8_1", 32, 40};
    types[10] = {"q2_k", 256, 84};
    types[11] = {"q3_k", 256, 110};
    types[12] = {"q4_k", 256, 144};
    types[13] = {"q5_k", 256, 176};
    types[14] = {"q6_k", 256, 210};
    types[15] = {"q8_k", 256, 292};
    types[16] = {"iq2_xxs", 256, 66};
    types[17] = {"iq2_xs", 256, 74};
    types[18] = {"iq3_xxs", 256, 98};
    types[19] = {"iq1_s", 256, 110};
    types[20] = {"iq4_nl", 256, 50};
    types[21] = {"iq3_s", 256, 110};
    types[22] = {"iq2_s", 256, 82};
    types[23] = {"iq4_xs", 256, 136};
    types[24] = {"i8", 1, 1};
    types[25] = {"i16", 1, 2};
    types[26] = {"i32", 1, 4};
    types[27] = {"i64", 1, 8};
    types[28] = {"f64", 1, 8};
    types[29] = {"iq1_m", 256, 56};
    types[30] = {"bf16", 1, 2};
    return types;
}();


static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(ds4_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        ds4_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;

        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

static const gguf_type_info *tensor_type(uint32_t type) {
    if (type >= gguf_types.size() || gguf_types[type].name == nullptr) {
        return nullptr;
    }
    return &gguf_types[type];
}

const char *ds4_tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static ds4_cursor cursor_at(const ds4_model *m, uint64_t pos) {
    ds4_cursor c{m->map, m->size, pos, {0}};
    return c;
}

static ds4_kv *model_find_kv(const ds4_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (ds4_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return nullptr;
}

static bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static bool model_get_u64(const ds4_model *m, const char *key, uint64_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) {
        return cursor_u64(&c, out);
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}

static bool model_get_f32(const ds4_model *m, const char *key, float *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) {
        return cursor_read(&c, out, sizeof(*out));
    }
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0.0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    return false;
}

static bool model_get_bool(const ds4_model *m, const char *key, bool *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_BOOL) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    uint8_t v = 0;
    if (!cursor_read(&c, &v, sizeof(v))) return false;
    *out = v != 0;
    return true;
}

struct ds4_array_ref {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
};

static bool model_get_array(const ds4_model *m, const char *key, ds4_array_ref *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

bool ds4_model_string_array_begin(const ds4_model *model,
                                  const char *key,
                                  ds4_string_iterator *iterator) {
    if (!model || !key || !iterator) return false;

    ds4_array_ref array;
    if (!model_get_array(model, key, &array) ||
        array.type != GGUF_VALUE_STRING) {
        return false;
    }
    *iterator = ds4_string_iterator{model, array.data_pos, array.len};
    return true;
}

bool ds4_model_string_array_next(ds4_string_iterator *iterator,
                                 ds4_string_view *value) {
    if (!iterator || !value || !iterator->model ||
        iterator->remaining == 0) {
        return false;
    }

    ds4_cursor cursor = cursor_at(iterator->model, iterator->pos);
    ds4_str string;
    if (!cursor_string(&cursor, &string)) return false;
    iterator->pos = cursor.pos;
    iterator->remaining--;
    *value = ds4_string_view{string.ptr, string.len};
    return true;
}

static void model_close(ds4_model *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

/* Read the GGUF metadata table.  Values stay in the mmap; we store offsets so
 * later validation can decode only the keys it needs. */
static void parse_metadata(ds4_model *m, ds4_cursor *c) {
    m->kv = static_cast<ds4_kv *>(
        calloc(static_cast<size_t>(m->n_kv), sizeof(m->kv[0])));
    if (!m->kv) ds4_die("out of memory while allocating metadata table");

    m->alignment = 32;

    for (uint64_t i = 0; i < m->n_kv; i++) {
        ds4_kv *kv = &m->kv[i];

        if (!cursor_string(c, &kv->key)) ds4_die(c->error);
        if (!cursor_u32(c, &kv->type)) ds4_die(c->error);

        kv->value_pos = c->pos;

        if (ds4_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32)
        {
            ds4_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) ds4_die(c->error);
    }
}

/* Read the tensor directory and convert relative GGUF offsets to absolute
 * mmap offsets.  Tensor bytes are still never copied here. */
static void parse_tensors(ds4_model *m, ds4_cursor *c) {
    m->tensors = static_cast<ds4_tensor *>(
        calloc(static_cast<size_t>(m->n_tensors), sizeof(m->tensors[0])));
    if (!m->tensors) ds4_die("out of memory while allocating tensor table");

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];

        if (!cursor_string(c, &t->name)) ds4_die(c->error);
        if (!cursor_u32(c, &t->ndim)) ds4_die(c->error);
        if (t->ndim == 0 || t->ndim > DS4_MAX_DIMS) {
            ds4_die("tensor has an unsupported number of dimensions");
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) ds4_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                ds4_die("tensor element count overflow");
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) ds4_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) ds4_die(c->error);

        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            ds4_log(stderr,
                ds4_log_type::warning,
                "ds4: warning: tensor %.*s has unsupported GGUF type %u\n",
                (int)t->name.len, t->name.ptr, t->type);
        }
    }

    m->tensor_data_pos = ds4_align_up(c->pos, m->alignment);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) {
            ds4_die("tensor offset overflow");
        }
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset))
        {
            ds4_die("tensor points outside GGUF file");
        }
        if (t->bytes > m->max_tensor_bytes) {
            m->max_tensor_bytes = t->bytes;
        }
    }
}

/* Open and map the GGUF once for resident ROCm tensor views. */
static void model_open(ds4_model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) ds4_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
    if (st.st_size < 32) ds4_die("model file is too small to be GGUF");

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) ds4_die_errno("cannot mmap model", path);

    m->fd = fd;
    m->map = static_cast<const uint8_t *>(map);
    m->size = (uint64_t)st.st_size;

    ds4_cursor c = cursor_at(m, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) ds4_die(c.error);
    if (magic != DS4_GGUF_MAGIC) ds4_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) ds4_die(c.error);

    if (m->version != 3) ds4_die("only GGUF v3 is supported");

    parse_metadata(m, &c);
    parse_tensors(m, &c);

}

static ds4_tensor *model_find_tensor(const ds4_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len &&
            memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

struct accelerator_tensor_span {
    uint64_t off;
    uint64_t end;
};

static uint64_t accelerator_rocm_preload_span_bytes(void) {
    return 1024ull * 1048576ull;
}

static bool accelerator_cache_model_tensor_spans(const ds4_model *m, uint64_t *cached_out) {
    std::vector<accelerator_tensor_span> spans;
    spans.reserve(static_cast<size_t>(m->n_tensors));
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        if (t->bytes == 0) continue;
        if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) {
            return false;
        }
        spans.push_back(
            accelerator_tensor_span{t->abs_offset, t->abs_offset + t->bytes});
    }
    std::sort(spans.begin(), spans.end(), [](const auto &left, const auto &right) {
        return left.off < right.off ||
               (left.off == right.off && left.end < right.end);
    });

    const uint64_t max_span = accelerator_rocm_preload_span_bytes();
    uint64_t cached = 0;
    uint64_t merged = 0;
    for (size_t i = 0; i < spans.size();) {
        uint64_t off = spans[i].off;
        uint64_t end = spans[i].end;
        i++;
        while (i < spans.size() && spans[i].off <= end + 65536u &&
               spans[i].end - off <= max_span) {
            if (spans[i].end > end) end = spans[i].end;
            i++;
        }
        while (off < end) {
            uint64_t chunk_end = end;
            if (chunk_end - off > max_span) chunk_end = off + max_span;
            char label[96];
            snprintf(label, sizeof(label), "tensor-span:%" PRIu64, merged);
            if (ds4_gpu_cache_model_range(m->map, m->size, off, chunk_end - off, label) == 0) {
                fprintf(stderr,
                        "ds4: accelerator failed to cache model tensor span %" PRIu64
                        " at offset %" PRIu64 "\n",
                        merged, off);
                return false;
            }
            cached += chunk_end - off;
            merged++;
            off = chunk_end;
        }
    }
    if (cached_out) *cached_out = cached;
    return true;
}

static bool accelerator_cache_model_tensors(const ds4_model *m) {
    if (!m || !m->map || m->size == 0) return false;

    const double t0 = ds4_now_seconds();
    uint64_t cached = 0;
    const bool cache_ok = accelerator_cache_model_tensor_spans(m, &cached);
    ds4_gpu_release_model_staging();
    if (!cache_ok) return false;
    if (cached != 0) {
        const double t1 = ds4_now_seconds();
        if (ds4_log_is_tty(stderr)) fputc('\n', stderr);
        fprintf(stderr,
                "ds4: ROCm startup model cache prepared %.2f GiB of tensor spans in %.3fs\n",
                (double)cached / 1073741824.0,
                t1 - t0);
    }
    return true;
}

/* =========================================================================
 * Fixed Weight Binding and Model Validation.
 * =========================================================================
 *
 * The GGUF tensor directory is converted into a DS4-specific pointer table.
 * After this section, the rest of the program addresses tensors by semantic
 * fields such as layer->attn_q_a or layer->ffn_gate_exps rather than by string
 * lookup.  Shape validation is intentionally strict.
 */

static uint32_t required_u32(const ds4_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static float required_f32(const ds4_model *m, const char *key) {
    float v = 0.0f;
    if (!model_get_f32(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static bool required_bool(const ds4_model *m, const char *key) {
    bool v = false;
    if (!model_get_bool(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static ds4_tensor *required_tensor(const ds4_model *m, const char *name) {
    ds4_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "ds4: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

static ds4_tensor *tensor_by_namef(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static ds4_tensor *required_tensorf(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return required_tensor(m, name);
}

static void tensor_expect_layout(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                ds4_tensor_type_name(t->type),
                ds4_tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

static void tensor_expect_optional(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (t) tensor_expect_layout(t, type, ndim, d0, d1, d2);
}

static bool tensor_is_routed_expert_type(uint32_t type) {
    return type == DS4_TENSOR_IQ2_XXS ||
           type == DS4_TENSOR_Q2_K ||
           type == DS4_TENSOR_Q4_K;
}

static uint64_t routed_expert_block_bytes(uint32_t type) {
    switch (type) {
    case DS4_TENSOR_IQ2_XXS: return sizeof(block_iq2_xxs);
    case DS4_TENSOR_Q2_K:    return sizeof(block_q2_K);
    case DS4_TENSOR_Q4_K:    return sizeof(block_q4_K);
    default:                 ds4_die("unsupported routed expert tensor type");
    }
    return 0;
}

uint64_t ds4_routed_expert_row_bytes(const ds4_tensor *t) {
    if ((t->dim[0] % QK_K) != 0) ds4_die("routed expert row is not QK_K aligned");
    return (t->dim[0] / QK_K) * routed_expert_block_bytes(t->type);
}

static void tensor_expect_routed_expert(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!tensor_is_routed_expert_type(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %u (%s), expected a routed expert quant type\n",
                (int)t->name.len,
                t->name.ptr,
                t->type,
                ds4_tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

/* Verify every tensor type and dimension used by the specialized pipeline.
 * After this succeeds, inference code can rely on fixed DS4 constants. */
static void weights_validate_layout(const ds4_weights *w) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;

    tensor_expect_layout(w->token_embd,      DS4_TENSOR_F16,  2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    tensor_expect_layout(w->output_hc_base,  DS4_TENSOR_F32,  1, DS4_N_HC, 0, 0);
    tensor_expect_layout(w->output_hc_fn,    DS4_TENSOR_F16,  2, hc_dim, DS4_N_HC, 0);
    tensor_expect_layout(w->output_hc_scale, DS4_TENSOR_F32,  1, 1, 0, 0);
    tensor_expect_layout(w->output_norm,     DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->output,          DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);

        tensor_expect_layout(l->hc_attn_fn,     DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_attn_scale,  DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_attn_base,   DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->attn_norm,      DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->attn_q_a,       DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
        tensor_expect_layout(l->attn_q_a_norm,  DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
        tensor_expect_layout(l->attn_q_b,       DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, q_dim, 0);
        tensor_expect_layout(l->attn_kv,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
        tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32,  1, DS4_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks,     DS4_TENSOR_F32,  1, DS4_N_HEAD, 0, 0);
        tensor_expect_layout(l->attn_output_a,  DS4_TENSOR_Q8_0, 2, DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_layout(l->attn_output_b,  DS4_TENSOR_Q8_0, 2, out_low_dim, DS4_N_EMBD, 0);

        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t comp_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            tensor_expect_layout(l->attn_compressor_ape,  DS4_TENSOR_F16, 2, comp_width, ratio, 0);
            tensor_expect_layout(l->attn_compressor_kv,   DS4_TENSOR_F16, 2, DS4_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_gate, DS4_TENSOR_F16, 2, DS4_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
        }
        if (ratio == 4) {
            const uint64_t index_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_width = 2u * DS4_N_INDEXER_HEAD_DIM;
            tensor_expect_layout(l->indexer_attn_q_b,          DS4_TENSOR_F16, 2, DS4_N_LORA_Q, index_q_dim, 0);
            tensor_expect_layout(l->indexer_proj,              DS4_TENSOR_F16, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD, 0);
            tensor_expect_layout(l->indexer_compressor_ape,    DS4_TENSOR_F16, 2, index_width, ratio, 0);
            tensor_expect_layout(l->indexer_compressor_kv,     DS4_TENSOR_F16, 2, DS4_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_gate,   DS4_TENSOR_F16, 2, DS4_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_norm,   DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        }

        tensor_expect_layout(l->hc_ffn_fn,      DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_ffn_scale,   DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_ffn_base,    DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->ffn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->ffn_gate_inp,   DS4_TENSOR_F16,  2, DS4_N_EMBD, DS4_N_EXPERT, 0);
        tensor_expect_optional(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
        tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
        tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
        tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
        if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            fprintf(stderr, "ds4: routed gate/up experts use different quant types in layer %u\n", il);
            exit(1);
        }
        tensor_expect_layout(l->ffn_gate_shexp, DS4_TENSOR_Q8_0,    2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_layout(l->ffn_up_shexp,   DS4_TENSOR_Q8_0,    2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_layout(l->ffn_down_shexp, DS4_TENSOR_Q8_0,    2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
        if (il < DS4_N_HASH_LAYER) {
            tensor_expect_layout(l->ffn_gate_tid2eid, DS4_TENSOR_I32, 2, DS4_N_EXPERT_USED, DS4_N_VOCAB, 0);
        }
    }
}

static bool ds4_shape_matches_metadata(
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_hash_layer,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    return DS4_N_LAYER == n_layer &&
           DS4_N_EMBD == n_embd &&
           DS4_N_VOCAB == n_vocab &&
           DS4_N_HEAD == n_head &&
           DS4_N_HEAD_KV == n_head_kv &&
           DS4_N_HEAD_DIM == n_head_dim &&
           DS4_N_VALUE_DIM == n_value_dim &&
           DS4_N_ROT == n_rot &&
           DS4_N_LORA_Q == n_lora_q &&
           DS4_N_LORA_O == n_lora_o &&
           DS4_N_OUT_GROUP == n_out_group &&
           DS4_N_EXPERT == n_expert &&
           DS4_N_EXPERT_USED == n_expert_used &&
           DS4_N_FF_EXP == n_ff_exp &&
           DS4_N_EXPERT_SHARED == n_expert_shared &&
           DS4_N_HASH_LAYER == n_hash_layer &&
           DS4_N_SWA == n_swa &&
           DS4_N_INDEXER_HEAD == n_indexer_head &&
           DS4_N_INDEXER_HEAD_DIM == n_indexer_head_dim &&
           DS4_N_INDEXER_TOP_K == n_indexer_top_k &&
           DS4_N_HC == n_hc &&
           DS4_N_HC_SINKHORN_ITER == n_hc_sinkhorn_iter;
}

static void ds4_select_shape_from_metadata(
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_hash_layer,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    if (ds4_shape_matches_metadata(
            n_layer, n_embd, n_vocab, n_head, n_head_kv,
            n_head_dim, n_value_dim, n_rot, n_lora_q,
            n_lora_o, n_out_group, n_expert,
            n_expert_used, n_ff_exp, n_expert_shared,
            n_hash_layer, n_swa, n_indexer_head,
            n_indexer_head_dim, n_indexer_top_k, n_hc,
            n_hc_sinkhorn_iter)) return;

    fprintf(stderr,
            "ds4: unsupported DeepSeek4 shape: layers=%u embd=%u heads=%u "
            "q_lora=%u out_groups=%u experts=%u ff_exp=%u indexer_top_k=%u\n",
            n_layer,
            n_embd,
            n_head,
            n_lora_q,
            n_out_group,
            n_expert,
            n_ff_exp,
            n_indexer_top_k);
    exit(1);
}

static void validate_compress_ratio_metadata(const ds4_model *m) {
    const char *key = "deepseek4.attention.compress_ratios";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "ds4: required int32/uint32 array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.attention.compress_ratios is shorter than the layer count");
    }

    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) ds4_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if (v < 0) ds4_die("metadata array contains a negative value");
            got = (uint32_t)v;
        }

        const uint32_t expected = ds4_expected_layer_compress_ratio(il);
        if (got != expected) {
            fprintf(stderr,
                    "ds4: unexpected DeepSeek4 compression ratio at layer %u for %s: got %u, expected %u\n",
                    il, DS4_MODEL_SHAPE_NAME, got, expected);
            exit(1);
        }
        g_ds4_compress_ratios[il] = got;
    }
}

static void config_expect_f32(const char *name, float got, float expected);

static void validate_swiglu_clamp_metadata(const ds4_model *m) {
    const char *key = "deepseek4.swiglu_clamp_exp";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_FLOAT32 && arr.type != GGUF_VALUE_FLOAT64)) {
        fprintf(stderr, "ds4: required float array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.swiglu_clamp_exp is shorter than the layer count");
    }

    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < DS4_N_LAYER; i++) {
        float got = 0.0f;
        if (arr.type == GGUF_VALUE_FLOAT32) {
            if (!cursor_read(&c, &got, sizeof(got))) ds4_die(c.error);
        } else {
            double v = 0.0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            got = (float)v;
        }
        config_expect_f32("swiglu_clamp_exp", got, DS4_SWIGLU_CLAMP_EXP);
    }
}

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%u for %s, got %u\n",
            name, expected, DS4_MODEL_SHAPE_NAME, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "ds4: expected %s=%.9g for %s, got %.9g\n",
            name, (double)expected, DS4_MODEL_SHAPE_NAME, (double)got);
    exit(1);
}

static void config_expect_bool(const char *name, bool got, bool expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%s for %s, got %s\n",
            name, expected ? "true" : "false", DS4_MODEL_SHAPE_NAME, got ? "true" : "false");
    exit(1);
}

static void config_validate_fixed_shape(uint32_t n_layer) {
    config_expect_u32("block_count",                  n_layer,                 DS4_N_LAYER);
}

/* Validate metadata values that affect semantics: attention shape, HC count,
 * expert routing, RoPE scaling, compression ratios, and SwiGLU clamp. */
static void config_validate_model(const ds4_model *m) {
    const uint32_t n_layer = required_u32(m, "deepseek4.block_count");
    const uint32_t n_embd = required_u32(m, "deepseek4.embedding_length");
    const uint32_t n_vocab = required_u32(m, "deepseek4.vocab_size");
    const uint32_t n_head = required_u32(m, "deepseek4.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "deepseek4.attention.head_count_kv");
    const uint32_t n_head_dim = required_u32(m, "deepseek4.attention.key_length");
    const uint32_t n_value_dim = required_u32(m, "deepseek4.attention.value_length");
    const uint32_t n_rot = required_u32(m, "deepseek4.rope.dimension_count");
    const uint32_t n_lora_q = required_u32(m, "deepseek4.attention.q_lora_rank");
    const uint32_t n_lora_o = required_u32(m, "deepseek4.attention.output_lora_rank");
    const uint32_t n_out_group = required_u32(m, "deepseek4.attention.output_group_count");
    const uint32_t n_expert = required_u32(m, "deepseek4.expert_count");
    const uint32_t n_expert_used = required_u32(m, "deepseek4.expert_used_count");
    const uint32_t n_ff_exp = required_u32(m, "deepseek4.expert_feed_forward_length");
    const uint32_t n_expert_shared = required_u32(m, "deepseek4.expert_shared_count");
    const uint32_t n_hash_layer = required_u32(m, "deepseek4.hash_layer_count");
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);
    const uint32_t n_swa = required_u32(m, "deepseek4.attention.sliding_window");
    const uint32_t n_indexer_head = required_u32(m, "deepseek4.attention.indexer.head_count");
    const uint32_t n_indexer_head_dim = required_u32(m, "deepseek4.attention.indexer.key_length");
    const uint32_t n_indexer_top_k = required_u32(m, "deepseek4.attention.indexer.top_k");
    const uint32_t n_hc = required_u32(m, "deepseek4.hyper_connection.count");
    const uint32_t n_hc_sinkhorn_iter = required_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations");

    ds4_select_shape_from_metadata(n_layer,
                                   n_embd,
                                   n_vocab,
                                   n_head,
                                   n_head_kv,
                                   n_head_dim,
                                   n_value_dim,
                                   n_rot,
                                   n_lora_q,
                                   n_lora_o,
                                   n_out_group,
                                   n_expert,
                                   n_expert_used,
                                   n_ff_exp,
                                   n_expert_shared,
                                   n_hash_layer,
                                   n_swa,
                                   n_indexer_head,
                                   n_indexer_head_dim,
                                   n_indexer_top_k,
                                   n_hc,
                                   n_hc_sinkhorn_iter);

    config_expect_u32("embedding_length",            n_embd,         DS4_N_EMBD);
    config_expect_u32("vocab_size",                  n_vocab,        DS4_N_VOCAB);
    config_expect_u32("attention.head_count",        n_head,         DS4_N_HEAD);
    config_expect_u32("attention.key_length",        n_head_dim,     DS4_N_HEAD_DIM);
    config_expect_u32("attention.head_count_kv",     n_head_kv,      DS4_N_HEAD_KV);
    config_expect_u32("attention.value_length",      n_value_dim,    DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count",        n_rot,          DS4_N_ROT);
    config_expect_u32("attention.output_group_count", n_out_group,    DS4_N_OUT_GROUP);
    config_expect_u32("attention.q_lora_rank",       n_lora_q,        DS4_N_LORA_Q);
    config_expect_u32("attention.output_lora_rank",  n_lora_o,        DS4_N_LORA_O);
    config_expect_u32("expert_count",               n_expert,        DS4_N_EXPERT);
    config_expect_u32("expert_used_count",          n_expert_used,   DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", n_ff_exp,        DS4_N_FF_EXP);
    config_expect_u32("expert_shared_count",         n_expert_shared, DS4_N_EXPERT_SHARED);
    config_expect_u32("hash_layer_count",            n_hash_layer,    DS4_N_HASH_LAYER);
    config_expect_u32("expert_group_count",         n_expert_groups, 0);
    config_expect_u32("expert_group_used_count",    n_group_used,    0);

    config_expect_u32("attention.sliding_window",     n_swa,                   DS4_N_SWA);
    config_expect_u32("attention.indexer.head_count", n_indexer_head,     DS4_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length", n_indexer_head_dim, DS4_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k",      n_indexer_top_k,    DS4_N_INDEXER_TOP_K);
    config_expect_u32("hyper_connection.count", n_hc, DS4_N_HC);
    config_expect_u32("hyper_connection.sinkhorn_iterations", n_hc_sinkhorn_iter, DS4_N_HC_SINKHORN_ITER);

    config_validate_fixed_shape(n_layer);
    validate_compress_ratio_metadata(m);

    validate_swiglu_clamp_metadata(m);

    uint64_t rope_orig_ctx = DS4_ROPE_ORIG_CTX;
    model_get_u64(m, "deepseek4.rope.scaling.original_context_length", &rope_orig_ctx);
    if (rope_orig_ctx != DS4_ROPE_ORIG_CTX) {
        fprintf(stderr, "ds4: expected rope.scaling.original_context_length=%" PRIu64
                " for %s, got %" PRIu64 "\n",
                (uint64_t)DS4_ROPE_ORIG_CTX, DS4_MODEL_SHAPE_NAME, rope_orig_ctx);
        exit(1);
    }
    const float rope_freq_base = required_f32(m, "deepseek4.rope.freq_base");
    config_expect_f32("rope.freq_base", rope_freq_base, DS4_ROPE_FREQ_BASE);
    float rope_scale_factor = DS4_ROPE_SCALE_FACTOR;
    model_get_f32(m, "deepseek4.rope.scaling.factor", &rope_scale_factor);
    config_expect_f32("rope.scaling.factor", rope_scale_factor, DS4_ROPE_SCALE_FACTOR);
    float rope_yarn_beta_fast = DS4_ROPE_YARN_BETA_FAST;
    model_get_f32(m, "deepseek4.rope.scaling.yarn_beta_fast", &rope_yarn_beta_fast);
    config_expect_f32("rope.scaling.yarn_beta_fast", rope_yarn_beta_fast, DS4_ROPE_YARN_BETA_FAST);
    float rope_yarn_beta_slow = DS4_ROPE_YARN_BETA_SLOW;
    model_get_f32(m, "deepseek4.rope.scaling.yarn_beta_slow", &rope_yarn_beta_slow);
    config_expect_f32("rope.scaling.yarn_beta_slow", rope_yarn_beta_slow, DS4_ROPE_YARN_BETA_SLOW);
    const float compress_rope_freq_base = required_f32(m, "deepseek4.attention.compress_rope_freq_base");
    config_expect_f32("attention.compress_rope_freq_base", compress_rope_freq_base, DS4_COMPRESS_ROPE_FREQ_BASE);
    const float expert_weight_scale = required_f32(m, "deepseek4.expert_weights_scale");
    config_expect_f32("expert_weights_scale", expert_weight_scale, DS4_EXPERT_WEIGHT_SCALE);
    const float rms_eps = required_f32(m, "deepseek4.attention.layer_norm_rms_epsilon");
    config_expect_f32("attention.layer_norm_rms_epsilon", rms_eps, DS4_RMS_EPS);
    const float hc_eps = required_f32(m, "deepseek4.hyper_connection.epsilon");
    config_expect_f32("hyper_connection.epsilon", hc_eps, DS4_HC_EPS);
    const bool expert_weight_norm = required_bool(m, "deepseek4.expert_weights_norm");
    config_expect_bool("expert_weights_norm", expert_weight_norm, true);
}

/* Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers. */
static void weights_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->token_embd       = required_tensor(m, "token_embd.weight");
    w->output_hc_base   = required_tensor(m, "output_hc_base.weight");
    w->output_hc_fn     = required_tensor(m, "output_hc_fn.weight");
    w->output_hc_scale  = required_tensor(m, "output_hc_scale.weight");
    w->output_norm      = required_tensor(m, "output_norm.weight");
    w->output           = required_tensor(m, "output.weight");

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_weights *l = &w->layer[il];
        const uint32_t compress_ratio = ds4_layer_compress_ratio(il);

        l->hc_attn_fn      = required_tensorf(m, "blk.%u.hc_attn_fn.weight", il);
        l->hc_attn_scale   = required_tensorf(m, "blk.%u.hc_attn_scale.weight", il);
        l->hc_attn_base    = required_tensorf(m, "blk.%u.hc_attn_base.weight", il);
        l->attn_norm       = required_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->attn_q_a        = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
        l->attn_q_a_norm   = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
        l->attn_q_b        = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
        l->attn_kv         = required_tensorf(m, "blk.%u.attn_kv.weight", il);
        l->attn_kv_a_norm  = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
        l->attn_sinks      = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
        l->attn_output_a   = required_tensorf(m, "blk.%u.attn_output_a.weight", il);
        l->attn_output_b   = required_tensorf(m, "blk.%u.attn_output_b.weight", il);
        if (compress_ratio != 0) {
            l->attn_compressor_ape  = required_tensorf(m, "blk.%u.attn_compressor_ape.weight", il);
            l->attn_compressor_kv   = required_tensorf(m, "blk.%u.attn_compressor_kv.weight", il);
            l->attn_compressor_gate = required_tensorf(m, "blk.%u.attn_compressor_gate.weight", il);
            l->attn_compressor_norm = required_tensorf(m, "blk.%u.attn_compressor_norm.weight", il);
        }
        if (compress_ratio == 4) {
            l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);
            l->indexer_proj     = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
            l->indexer_compressor_ape  = required_tensorf(m, "blk.%u.indexer_compressor_ape.weight", il);
            l->indexer_compressor_kv   = required_tensorf(m, "blk.%u.indexer_compressor_kv.weight", il);
            l->indexer_compressor_gate = required_tensorf(m, "blk.%u.indexer_compressor_gate.weight", il);
            l->indexer_compressor_norm = required_tensorf(m, "blk.%u.indexer_compressor_norm.weight", il);
        }
        l->hc_ffn_fn       = required_tensorf(m, "blk.%u.hc_ffn_fn.weight", il);
        l->hc_ffn_scale    = required_tensorf(m, "blk.%u.hc_ffn_scale.weight", il);
        l->hc_ffn_base     = required_tensorf(m, "blk.%u.hc_ffn_base.weight", il);
        l->ffn_norm        = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
        l->ffn_gate_inp    = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
        l->ffn_exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b.bias", il);
        l->ffn_gate_exps   = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
        l->ffn_up_exps     = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
        l->ffn_down_exps   = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
        l->ffn_gate_shexp  = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
        l->ffn_up_shexp    = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
        l->ffn_down_shexp  = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);

        if (il < DS4_N_HASH_LAYER) {
            l->ffn_gate_tid2eid = required_tensorf(m, "blk.%u.ffn_gate_tid2eid.weight", il);
        }
    }

    weights_validate_layout(w);
}

static void weights_free(ds4_weights *w) {
    memset(w, 0, sizeof(*w));
}

/* =========================================================================
 * DSpark Support Model.
 * =========================================================================
 *
 * The DSpark drafter lives in its own GGUF. Its blocks reuse the DS4 block
 * layout, so binding produces ordinary ds4_layer_weights and the existing
 * batched block kernels drive it without a parallel implementation. Validation
 * is as strict as the target's: a support model that does not match this
 * checkpoint's shape must fail at load, not silently draft garbage.
 */

static ds4_tensor *dspark_required_tensor_stage(const ds4_model *m,
                                               const char *suffix,
                                               uint32_t stage) {
    char name[128];
    snprintf(name, sizeof(name), "mtp.%u.%s", stage, suffix);
    ds4_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "ds4: DSpark support model is missing tensor %s\n", name);
        exit(1);
    }
    return t;
}

static ds4_tensor *dspark_optional_tensor_stage(const ds4_model *m,
                                               const char *suffix,
                                               uint32_t stage) {
    char name[128];
    snprintf(name, sizeof(name), "mtp.%u.%s", stage, suffix);
    return model_find_tensor(m, name);
}

static uint32_t dspark_required_u32(const ds4_model *m, const char *key) {
    uint32_t value = 0;
    if (!model_get_u32(m, key, &value)) {
        fprintf(stderr, "ds4: DSpark support model is missing metadata key %s\n", key);
        exit(1);
    }
    return value;
}

uint32_t ds4_dspark_feature_width(const ds4_dspark_model *dspark) {
    if (!dspark) return 0;
    return dspark->n_target_layers * DS4_N_EMBD;
}

static void dspark_validate_metadata(ds4_dspark_model *d) {
    const ds4_model *m = d->model;
    ds4_str architecture = {};
    ds4_kv *kv = model_find_kv(m, "general.architecture");
    if (kv && kv->type == GGUF_VALUE_STRING) {
        ds4_cursor c = cursor_at(m, kv->value_pos);
        (void)cursor_string(&c, &architecture);
    }
    if (!ds4_streq(architecture, "deepseek4-dspark")) {
        fprintf(stderr,
                "ds4: expected general.architecture=deepseek4-dspark for the "
                "DSpark support model, got '%.*s'\n",
                (int)architecture.len,
                architecture.ptr ? architecture.ptr : "");
        exit(1);
    }

    d->n_stages = dspark_required_u32(m, "dspark.stage_count");
    d->block_size = dspark_required_u32(m, "dspark.block_size");
    d->markov_rank = dspark_required_u32(m, "dspark.markov_rank");
    d->noise_token_id = dspark_required_u32(m, "dspark.noise_token_id");

    if (d->n_stages == 0 || d->n_stages > DS4_DSPARK_MAX_STAGES) {
        fprintf(stderr, "ds4: DSpark stage_count=%u is outside 1..%u\n",
                d->n_stages, DS4_DSPARK_MAX_STAGES);
        exit(1);
    }
    if (d->block_size < 2 || d->block_size > DS4_DSPARK_MAX_BLOCK) {
        fprintf(stderr, "ds4: DSpark block_size=%u is outside 2..%u\n",
                d->block_size, DS4_DSPARK_MAX_BLOCK);
        exit(1);
    }
    if (d->markov_rank == 0 || d->markov_rank % 32u != 0) {
        fprintf(stderr,
                "ds4: DSpark markov_rank=%u must be a non-zero multiple of the "
                "32-element Q8_0 block\n",
                d->markov_rank);
        exit(1);
    }
    if (d->noise_token_id >= DS4_N_VOCAB) {
        fprintf(stderr, "ds4: DSpark noise_token_id=%u is outside the vocabulary\n",
                d->noise_token_id);
        exit(1);
    }

    const uint32_t layer_count = dspark_required_u32(m, "dspark.n_layers");
    if (layer_count != d->n_stages) {
        fprintf(stderr, "ds4: DSpark n_layers=%u disagrees with stage_count=%u\n",
                layer_count, d->n_stages);
        exit(1);
    }

    ds4_array_ref layers = {};
    if (!model_get_array(m, "dspark.target_layer_ids", &layers) ||
        layers.type != GGUF_VALUE_UINT32 ||
        layers.len == 0 || layers.len > DS4_DSPARK_MAX_TARGET_LAYERS) {
        fprintf(stderr,
                "ds4: DSpark target_layer_ids must list 1..%u uint32 target layers\n",
                DS4_DSPARK_MAX_TARGET_LAYERS);
        exit(1);
    }
    d->n_target_layers = (uint32_t)layers.len;
    ds4_cursor c = cursor_at(m, layers.data_pos);
    for (uint32_t i = 0; i < d->n_target_layers; i++) {
        uint32_t value = 0;
        if (!cursor_u32(&c, &value) || value >= DS4_N_LAYER) {
            fprintf(stderr,
                    "ds4: DSpark target_layer_ids[%u] is not a valid target layer\n",
                    i);
            exit(1);
        }
        if (i > 0 && value <= d->target_layer_ids[i - 1]) {
            fprintf(stderr,
                    "ds4: DSpark target_layer_ids must be strictly increasing\n");
            exit(1);
        }
        d->target_layer_ids[i] = value;
    }
}

static void dspark_bind_stage_block(ds4_layer_weights *l,
                                    const ds4_model *m,
                                    uint32_t stage) {
    memset(l, 0, sizeof(*l));
    l->hc_attn_fn      = dspark_required_tensor_stage(m, "hc_attn_fn.weight", stage);
    l->hc_attn_scale   = dspark_required_tensor_stage(m, "hc_attn_scale.weight", stage);
    l->hc_attn_base    = dspark_required_tensor_stage(m, "hc_attn_base.weight", stage);
    l->attn_norm       = dspark_required_tensor_stage(m, "attn_norm.weight", stage);
    l->attn_q_a        = dspark_required_tensor_stage(m, "attn_q_a.weight", stage);
    l->attn_q_a_norm   = dspark_required_tensor_stage(m, "attn_q_a_norm.weight", stage);
    l->attn_q_b        = dspark_required_tensor_stage(m, "attn_q_b.weight", stage);
    l->attn_kv         = dspark_required_tensor_stage(m, "attn_kv.weight", stage);
    l->attn_kv_a_norm  = dspark_required_tensor_stage(m, "attn_kv_a_norm.weight", stage);
    l->attn_sinks      = dspark_required_tensor_stage(m, "attn_sinks.weight", stage);
    l->attn_output_a   = dspark_required_tensor_stage(m, "attn_output_a.weight", stage);
    l->attn_output_b   = dspark_required_tensor_stage(m, "attn_output_b.weight", stage);
    l->hc_ffn_fn       = dspark_required_tensor_stage(m, "hc_ffn_fn.weight", stage);
    l->hc_ffn_scale    = dspark_required_tensor_stage(m, "hc_ffn_scale.weight", stage);
    l->hc_ffn_base     = dspark_required_tensor_stage(m, "hc_ffn_base.weight", stage);
    l->ffn_norm        = dspark_required_tensor_stage(m, "ffn_norm.weight", stage);
    l->ffn_gate_inp    = dspark_required_tensor_stage(m, "ffn_gate_inp.weight", stage);
    l->ffn_exp_probs_b = dspark_optional_tensor_stage(m, "exp_probs_b.bias", stage);
    l->ffn_gate_exps   = dspark_required_tensor_stage(m, "ffn_gate_exps.weight", stage);
    l->ffn_up_exps     = dspark_required_tensor_stage(m, "ffn_up_exps.weight", stage);
    l->ffn_down_exps   = dspark_required_tensor_stage(m, "ffn_down_exps.weight", stage);
    l->ffn_gate_shexp  = dspark_required_tensor_stage(m, "ffn_gate_shexp.weight", stage);
    l->ffn_up_shexp    = dspark_required_tensor_stage(m, "ffn_up_shexp.weight", stage);
    l->ffn_down_shexp  = dspark_required_tensor_stage(m, "ffn_down_shexp.weight", stage);
}

static void dspark_validate_stage_block(const ds4_layer_weights *l, uint32_t stage) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    (void)stage;

    tensor_expect_layout(l->hc_attn_fn,     DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
    tensor_expect_layout(l->hc_attn_scale,  DS4_TENSOR_F32,  1, 3, 0, 0);
    tensor_expect_layout(l->hc_attn_base,   DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
    tensor_expect_layout(l->attn_norm,      DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(l->attn_q_a,       DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
    tensor_expect_layout(l->attn_q_a_norm,  DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
    tensor_expect_layout(l->attn_q_b,       DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, q_dim, 0);
    tensor_expect_layout(l->attn_kv,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
    tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32,  1, DS4_N_HEAD_DIM, 0, 0);
    tensor_expect_layout(l->attn_sinks,     DS4_TENSOR_F32,  1, DS4_N_HEAD, 0, 0);
    tensor_expect_layout(l->attn_output_a,  DS4_TENSOR_Q8_0, 2,
                         DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
    tensor_expect_layout(l->attn_output_b,  DS4_TENSOR_Q8_0, 2, out_low_dim, DS4_N_EMBD, 0);
    tensor_expect_layout(l->hc_ffn_fn,      DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
    tensor_expect_layout(l->hc_ffn_scale,   DS4_TENSOR_F32,  1, 3, 0, 0);
    tensor_expect_layout(l->hc_ffn_base,    DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
    tensor_expect_layout(l->ffn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    /* The support model ships a Q8_0 router projection where the target ships
     * F16; both are dense and the router kernel consumes logits, not weights. */
    tensor_expect_layout(l->ffn_gate_inp,   DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
    tensor_expect_optional(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
    tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
    if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
        fprintf(stderr, "ds4: DSpark stage %u routed gate/up experts use different quant types\n",
                stage);
        exit(1);
    }
    tensor_expect_layout(l->ffn_gate_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
    tensor_expect_layout(l->ffn_up_shexp,   DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
    tensor_expect_layout(l->ffn_down_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
}

static void dspark_bind(ds4_dspark_model *d) {
    const ds4_model *m = d->model;
    const uint32_t last = d->n_stages - 1u;
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;

    for (uint32_t stage = 0; stage < d->n_stages; stage++) {
        ds4_dspark_stage_weights *s = &d->stage[stage];
        dspark_bind_stage_block(&s->block, m, stage);
        dspark_validate_stage_block(&s->block, stage);
    }

    ds4_dspark_stage_weights *first = &d->stage[0];
    first->main_proj = dspark_required_tensor_stage(m, "main_proj.weight", 0);
    first->main_norm = dspark_required_tensor_stage(m, "main_norm.weight", 0);
    tensor_expect_layout(first->main_proj, DS4_TENSOR_Q8_0, 2,
                         ds4_dspark_feature_width(d), DS4_N_EMBD, 0);
    tensor_expect_layout(first->main_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);

    ds4_dspark_stage_weights *final_stage = &d->stage[last];
    final_stage->norm = dspark_required_tensor_stage(m, "norm.weight", last);
    final_stage->hc_head_base = dspark_required_tensor_stage(m, "hc_head_base.weight", last);
    final_stage->hc_head_fn = dspark_required_tensor_stage(m, "hc_head_fn.weight", last);
    final_stage->hc_head_scale = dspark_required_tensor_stage(m, "hc_head_scale.weight", last);
    final_stage->markov_w1 =
        dspark_required_tensor_stage(m, "markov_head.markov_w1.weight", last);
    final_stage->markov_w2 =
        dspark_required_tensor_stage(m, "markov_head.markov_w2.weight", last);
    tensor_expect_layout(final_stage->norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(final_stage->hc_head_base, DS4_TENSOR_F32, 1, DS4_N_HC, 0, 0);
    tensor_expect_layout(final_stage->hc_head_fn, DS4_TENSOR_F16, 2, hc_dim, DS4_N_HC, 0);
    tensor_expect_layout(final_stage->hc_head_scale, DS4_TENSOR_F32, 1, 1, 0, 0);
    tensor_expect_layout(final_stage->markov_w1, DS4_TENSOR_Q8_0, 2,
                         d->markov_rank, DS4_N_VOCAB, 0);
    tensor_expect_layout(final_stage->markov_w2, DS4_TENSOR_Q8_0, 2,
                         d->markov_rank, DS4_N_VOCAB, 0);
    // Validate the artifact's confidence head; the runtime controller uses
    // observed acceptance and does not execute this projection.
    tensor_expect_layout(
        dspark_required_tensor_stage(m, "confidence_head.proj.weight", last),
        DS4_TENSOR_Q8_0, 2, (uint64_t)DS4_N_EMBD + d->markov_rank, 1, 0);
}

/* Copy every bound DSpark tensor into the support arena. The residency policy
 * is selected by the ROCm runtime; Strix Halo defaults to managed storage so
 * prompt processing retains its device-memory headroom. */
static bool dspark_cache_tensors(const ds4_dspark_model *d) {
    const ds4_model *m = d->model;
    uint64_t total = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        /* 256-byte span alignment matches ds4_gpu_cache_support_range. */
        total = ds4_align_up(total, 256u) + m->tensors[i].bytes;
    }
    if (!ds4_gpu_reserve_support_map(m->map, m->size, total)) {
        fprintf(stderr, "ds4: failed to reserve %.2f GiB for the DSpark support model\n",
                (double)total / 1073741824.0);
        return false;
    }
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        if (!ds4_gpu_cache_support_range(m->map,
                                        m->size,
                                        t->abs_offset,
                                        t->bytes,
                                        "dspark_tensor")) {
            fprintf(stderr, "ds4: failed to cache DSpark tensor %.*s\n",
                    (int)t->name.len, t->name.ptr);
            return false;
        }
    }
    fprintf(stderr, "ds4: DSpark support model cached %.2f GiB of tensor spans\n",
            (double)total / 1073741824.0);
    return true;
}

int ds4_dspark_open(ds4_dspark_model **out, const char *path) {
    if (!out || !path || !path[0]) return 1;
    *out = NULL;

    auto *d = static_cast<ds4_dspark_model *>(
        ds4_xcalloc(1, sizeof(ds4_dspark_model)));
    d->model = static_cast<ds4_model *>(ds4_xcalloc(1, sizeof(ds4_model)));
    d->model->fd = -1;

    model_open(d->model, path);
    dspark_validate_metadata(d);
    dspark_bind(d);
    if (!dspark_cache_tensors(d)) {
        ds4_dspark_close(d);
        return 1;
    }

    fprintf(stderr,
            "ds4: DSpark support model loaded stages=%u block=%u markov_rank=%u "
            "noise_token=%u target_layers=%u\n",
            d->n_stages,
            d->block_size,
            d->markov_rank,
            d->noise_token_id,
            d->n_target_layers);
    *out = d;
    return 0;
}

void ds4_dspark_close(ds4_dspark_model *d) {
    if (!d) return;
    ds4_gpu_release_support_map();
    model_close(d->model);
    free(d->model);
    free(d);
}

/* =========================================================================
 * Engine API and Process Lock.
 * =========================================================================
 *
 * The public entry points acquire the single instance lock, open the GGUF with
 * the backend-appropriate mmap policy, and expose tokenized prompt operations
 * to the CLI and server.
 */

static void ds4_release_instance_lock(void) {
    if (g_ds4_lock_fd >= 0) {
        close(g_ds4_lock_fd);
        g_ds4_lock_fd = -1;
    }
}

/* Refuse to start a second ds4 process.  The model can map tens of GiB, so a
 * stale accidental second run is more dangerous than a normal CLI error. */
static void ds4_acquire_instance_lock(void) {
    const char *path = getenv("DS4_LOCK_FILE");
    if (!path || !path[0]) path = "/tmp/ds4.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "ds4: failed to open lock file %s: %s\n", path, strerror(errno));
        exit(2);
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                char *end = NULL;
                owner = strtol(buf, &end, 10);
            }
            if (owner > 0) {
                fprintf(stderr, "ds4: another ds4 process is already running (pid %ld); refusing to start\n", owner);
            } else {
                fprintf(stderr, "ds4: another ds4 process is already running; refusing to start\n");
            }
            close(fd);
            exit(2);
        }
        fprintf(stderr, "ds4: failed to lock %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }

    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "ds4: failed to truncate lock file %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }
    dprintf(fd, "%ld\n", (long)getpid());
    g_ds4_lock_fd = fd;
    atexit(ds4_release_instance_lock);
}

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    if (!out || !opt || !opt->model_path || !opt->model_path[0]) return 1;

    auto *e = static_cast<ds4_engine *>(ds4_xcalloc(1, sizeof(ds4_engine)));
    e->model = static_cast<ds4_model *>(ds4_xcalloc(1, sizeof(ds4_model)));
    e->weights =
        static_cast<ds4_weights *>(ds4_xcalloc(1, sizeof(ds4_weights)));
    e->model->fd = -1;
    ds4_acquire_instance_lock();

    model_open(e->model, opt->model_path);
    e->vocab = ds4_vocab_create(e->model);
    config_validate_model(e->model);
    weights_bind(e->weights, e->model);

    e->rocm_ready = ds4_gpu_init() != 0;
    if (!e->rocm_ready) {
        fprintf(stderr, "ds4: ROCm backend unavailable; aborting startup\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }

    (void)ds4_gpu_set_model_fd(e->model->fd);
    if (!ds4_gpu_set_model_map_range(e->model->map,
                                     e->model->size,
                                     e->model->tensor_data_pos,
                                     e->model->size - e->model->tensor_data_pos,
                                     e->model->max_tensor_bytes)) {
        fprintf(stderr,
                "ds4: ROCm failed to map model views; aborting startup. "
                "This is commonly caused by insufficient memory or accelerator VM budget.\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    if (!accelerator_cache_model_tensors(e->model)) {
        fprintf(stderr, "ds4: ROCm failed to prepare startup model cache\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }

    /* The DSpark drafter is loaded last so a failure here cannot leave the
     * target model half-initialized. */
    if (opt->dspark_model_path && opt->dspark_model_path[0]) {
        if (ds4_dspark_open(&e->dspark, opt->dspark_model_path) != 0) {
            fprintf(stderr, "ds4: failed to load DSpark support model %s\n",
                    opt->dspark_model_path);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
    }

    *out = e;
    return 0;
}

bool ds4_engine_has_dspark(const ds4_engine *e) {
    return e != NULL && e->dspark != NULL;
}

int ds4_engine_vocab_size(const ds4_engine *e) {
    return e ? ds4_vocab_size(e->vocab) : 0;
}

const char *ds4_engine_model_name(const ds4_engine *e) {
    (void)e;
    return DS4_MODEL_SHAPE_NAME;
}

void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
    ds4_rocm_graph_destroy(e->batch_workspace);
    e->batch_workspace = NULL;
    ds4_dspark_close(e->dspark);
    e->dspark = NULL;
    weights_free(e->weights);
    ds4_vocab_destroy(e->vocab);
    model_close(e->model);
    ds4_gpu_cleanup();
    ds4_release_instance_lock();
    free(e->weights);
    free(e->model);
    free(e);
}
