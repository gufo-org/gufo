#ifndef GUFO_CACHE_MANIFEST_H_
#define GUFO_CACHE_MANIFEST_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust disk manifest codec and checksum. The
   `GUFOMNF2` manifest of cache format versions 2 and 3 and CRC-64/ECMA-182;
   src/cache/disk.cpp encodes, decodes and checksums with it, while files,
   directories, recovery, durability and retention stay there. */
typedef enum GufoCacheStatus {
  GUFO_CACHE_COMPLETE = 0,
  /* The manifest breaks the format, its limits or its geometry. */
  GUFO_CACHE_REJECTED = 1,
  GUFO_CACHE_NO_MEMORY = 2,
  /* A callback returned nonzero or no buffer. */
  GUFO_CACHE_STOPPED = 3,
  /* A caller bug (null pointer, missing callback) or a codec bug. */
  GUFO_CACHE_INTERNAL = 4,
} GufoCacheStatus;

/* ComponentKind and the last CheckpointPurpose in src/cache, and the
   manifest limits of src/core/cache_manifest/src/manifest.rs. */
enum {
  GUFO_CACHE_APPEND_ROWS = 0,
  GUFO_CACHE_PRIVATE_STATE = 1,
  GUFO_CACHE_WINDOW_ROWS = 2,
  GUFO_CACHE_MAX_PURPOSE = 3,
  GUFO_CACHE_FORMAT_VERSION = 2,
  GUFO_CACHE_WINDOW_FORMAT_VERSION = 3,
  GUFO_CACHE_MAX_COMPONENTS = 256,
  GUFO_CACHE_MAX_REFERENCES = 65536,
  GUFO_CACHE_MAX_MANIFEST_BYTES = 16 * 1024 * 1024,
};

typedef struct GufoCachePayload {
  uint8_t file[16];
  uint64_t bytes;
  uint64_t checksum;
} GufoCachePayload;

/* A component's descriptor, position and references. window_rows is zero
   unless kind is GUFO_CACHE_WINDOW_ROWS. The has_ fields are 0 or 1. */
typedef struct GufoCacheComponent {
  uint32_t id;
  uint32_t layout_version;
  uint8_t kind;
  uint64_t row_bytes;
  uint64_t rows_per_chunk;
  uint64_t state_bytes;
  uint64_t window_rows;
  uint32_t position_id;
  uint64_t valid_rows;
  const GufoCachePayload* chunks;
  size_t chunk_count;
  int has_tail;
  GufoCachePayload tail;
  int has_private_state;
  GufoCachePayload private_state;
} GufoCacheComponent;

typedef struct GufoCacheManifest {
  uint64_t checkpoint;
  uint64_t lineage;
  uint8_t compatibility[32];
  uint8_t purpose;
  uint32_t rank;
  const uint8_t* input;
  size_t input_len;
  const uint32_t* tokens;
  size_t token_count;
  const GufoCacheComponent* components;
  size_t component_count;
} GufoCacheManifest;

/* What a decoded manifest holds: input bytes, tokens, components, their full
   chunks, and every payload reference, tails and private states included. */
typedef struct GufoCacheManifestPlan {
  size_t input_bytes;
  size_t tokens;
  size_t components;
  size_t chunks;
  size_t references;
} GufoCacheManifestPlan;

/* Each callback returns nonzero to stop. Its arguments, and the bytes and
   payloads they point to, are valid during the call only. */
typedef int (*GufoCacheBeginFn)(void* context,
                                const GufoCacheManifest* manifest);
typedef int (*GufoCacheTokensFn)(void* context, const uint32_t* tokens,
                                 size_t count);
typedef int (*GufoCacheComponentFn)(void* context,
                                    const GufoCacheComponent* component);
typedef int (*GufoCachePayloadsFn)(void* context,
                                   const GufoCachePayload* payloads,
                                   size_t count);
/* Returns a writable buffer of len bytes, valid until the encoding returns,
   or null to stop. */
typedef uint8_t* (*GufoCacheBufferFn)(void* context, size_t len);

/* Receives a decoded manifest. begin gets the fixed fields and the input,
   which points into the decoded bytes, with token_count and component_count
   but no tokens or components. tokens then gets the tokens in blocks, in
   order. For each component, component gets it with chunk_count but no
   chunks, which chunks then gets in blocks. Empty blocks are skipped. */
typedef struct GufoCacheManifestSink {
  void* context;
  GufoCacheBeginFn begin;
  GufoCacheTokensFn tokens;
  GufoCacheComponentFn component;
  GufoCachePayloadsFn chunks;
} GufoCacheManifestSink;

/* Writes to *checksum the CRC-64/ECMA-182 of data[0, len) after bytes whose
   CRC is seed (zero at the start): no reflection or final xor. */
GufoCacheStatus gufo_cache_checksum(const uint8_t* data, size_t len,
                                    uint64_t seed, uint64_t* checksum);

/* Reads the counts of the manifest in data[0, len) into *plan, so the
   caller can reserve its storage; checks its length and structure but not
   its magic, version, checksum or geometry, which decoding does. */
GufoCacheStatus gufo_cache_manifest_plan(const uint8_t* data, size_t len,
                                         GufoCacheManifestPlan* plan);

/* Decodes and validates the manifest in data[0, len), then reports it to
   sink, which takes nothing from a rejected manifest. Entries taken before
   another failure are void. */
GufoCacheStatus gufo_cache_manifest_decode(const uint8_t* data, size_t len,
                                           const GufoCacheManifestSink* sink);

/* Validates and encodes manifest into the buffer that buffer returns for
   its exact encoded length. The buffer's contents are void unless the
   encoding completes. */
GufoCacheStatus gufo_cache_manifest_encode(const GufoCacheManifest* manifest,
                                           GufoCacheBufferFn buffer,
                                           void* context);

#ifdef __cplusplus
}
#endif

#endif
