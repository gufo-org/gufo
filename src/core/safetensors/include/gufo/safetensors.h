#ifndef GUFO_SAFETENSORS_H_
#define GUFO_SAFETENSORS_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust safetensors header reader. A file is an
   8-byte little-endian header length, a JSON header of tensor entries and
   an optional "__metadata__" object of strings, then the payload the tensors
   tile. src/core/safetensors.cpp wraps it for the model loaders; the tensor
   data stays in the caller's file or mapping. */
typedef enum GufoSafetensorsParse {
  GUFO_SAFETENSORS_COMPLETE = 0,
  /* The file breaks the format; the reject callback got the reason. */
  GUFO_SAFETENSORS_REJECTED = 1,
  GUFO_SAFETENSORS_NO_MEMORY = 2,
  /* A callback returned nonzero. */
  GUFO_SAFETENSORS_STOPPED = 3,
  /* A caller bug (null pointer, missing callback) or a parser bug. */
  GUFO_SAFETENSORS_INTERNAL = 4,
} GufoSafetensorsParse;

/* Why a file is rejected: it breaks the format, or one of Gufo's limits
   (a header of at most 32 MiB with at least one tensor, at most 5
   dimensions, each at most 2^31 - 1). */
typedef enum GufoSafetensorsReason {
  /* The file is shorter than the length prefix. */
  GUFO_SAFETENSORS_FILE = 1,
  /* The header length is 0, above 32 MiB or past the file. */
  GUFO_SAFETENSORS_LENGTH = 2,
  /* The header is not JSON. */
  GUFO_SAFETENSORS_JSON = 3,
  /* The header is not a JSON object. */
  GUFO_SAFETENSORS_NOT_OBJECT = 4,
  /* "__metadata__" is neither null nor an object of strings. */
  GUFO_SAFETENSORS_METADATA = 5,
  /* A tensor entry is not an object. */
  GUFO_SAFETENSORS_ENTRY = 6,
  /* The type is missing, not a string or not one the format names. */
  GUFO_SAFETENSORS_DTYPE = 7,
  /* "shape" or "data_offsets" is missing or not an array, the shape has
     more than 5 dimensions or the offsets are not two. */
  GUFO_SAFETENSORS_LAYOUT = 8,
  /* A dimension is not an integer from 0 to 2^31 - 1, the shape holds more
     bytes than the payload, or, without elements, its dimensions multiplied
     in order overflow 64 bits before the first 0. */
  GUFO_SAFETENSORS_DIMENSION = 9,
  /* An offset is not an integer, or the offsets do not span exactly the
     tensor's bytes within the payload. */
  GUFO_SAFETENSORS_EXTENT = 10,
  /* Two tensors share bytes, or a tensor without elements lies inside
     another's bytes. */
  GUFO_SAFETENSORS_OVERLAP = 11,
  /* Payload bytes before a tensor belong to none. */
  GUFO_SAFETENSORS_GAP = 12,
  /* Payload bytes after the last tensor belong to none. */
  GUFO_SAFETENSORS_TRAILING = 13,
  /* The header holds no tensor. */
  GUFO_SAFETENSORS_EMPTY = 14,
} GufoSafetensorsReason;

/* The element types the format names (its spelling in the comments), as in
   safetensors 0.8. New types take the next values. */
typedef enum GufoSafetensorsDtype {
  GUFO_SAFETENSORS_BOOL = 0,    /* BOOL, 8 bits */
  GUFO_SAFETENSORS_F4 = 1,      /* F4, 4 bits */
  GUFO_SAFETENSORS_F6_E2M3 = 2, /* F6_E2M3, 6 bits */
  GUFO_SAFETENSORS_F6_E3M2 = 3, /* F6_E3M2, 6 bits */
  GUFO_SAFETENSORS_U8 = 4,
  GUFO_SAFETENSORS_I8 = 5,
  GUFO_SAFETENSORS_F8_E5M2 = 6,
  GUFO_SAFETENSORS_F8_E4M3 = 7,
  GUFO_SAFETENSORS_F8_E8M0 = 8,
  GUFO_SAFETENSORS_I16 = 9,
  GUFO_SAFETENSORS_U16 = 10,
  GUFO_SAFETENSORS_F16 = 11,
  GUFO_SAFETENSORS_BF16 = 12,
  GUFO_SAFETENSORS_I32 = 13,
  GUFO_SAFETENSORS_U32 = 14,
  GUFO_SAFETENSORS_F32 = 15,
  GUFO_SAFETENSORS_C64 = 16, /* C64, 64 bits */
  GUFO_SAFETENSORS_F64 = 17,
  GUFO_SAFETENSORS_I64 = 18,
  GUFO_SAFETENSORS_U64 = 19,
  GUFO_SAFETENSORS_F8_E4M3FNUZ = 20, /* F8_E4M3FNUZ, 8 bits */
  GUFO_SAFETENSORS_F8_E5M2FNUZ = 21, /* F8_E5M2FNUZ, 8 bits */
} GufoSafetensorsDtype;

/* Bytes valid during a callback only. */
typedef struct GufoSafetensorsSlice {
  const uint8_t* data;
  size_t len;
} GufoSafetensorsSlice;

/* The tensor or metadata key that fails (empty for the file or the header
   as a whole) and what is wrong, in words, to follow
   "invalid safetensors header: " and the name, or
   "invalid safetensors file: " for GUFO_SAFETENSORS_FILE. */
typedef struct GufoSafetensorsRejection {
  GufoSafetensorsReason reason;
  GufoSafetensorsSlice name;
  GufoSafetensorsSlice message;
} GufoSafetensorsRejection;

typedef struct GufoSafetensorsMetadata {
  GufoSafetensorsSlice key;
  GufoSafetensorsSlice value;
} GufoSafetensorsMetadata;

/* One tensor: its shape holds rank dimensions, valid during the callback
   only; begin and end are offsets in the payload. A tensor without elements
   has begin == end between the others or at either end of the payload and
   takes no part in the tiling. */
typedef struct GufoSafetensorsTensor {
  GufoSafetensorsSlice name;
  GufoSafetensorsDtype dtype;
  const uint64_t* shape;
  size_t rank;
  uint64_t begin;
  uint64_t end;
} GufoSafetensorsTensor;

/* Takes the rejection. */
typedef void (*GufoSafetensorsRejectFn)(
    void* context, const GufoSafetensorsRejection* rejection);
/* Takes one metadata entry; returns nonzero to stop. */
typedef int (*GufoSafetensorsMetadataFn)(void* context,
                                         const GufoSafetensorsMetadata* entry);
/* Takes one tensor; returns nonzero to stop. */
typedef int (*GufoSafetensorsTensorFn)(void* context,
                                       const GufoSafetensorsTensor* tensor);

/* The spelling of the type whose value is dtype, such as "BF16"; empty for
   a value that names no type. The bytes live as long as the program. */
GufoSafetensorsSlice gufo_safetensors_dtype_name(uint32_t dtype);
/* The size in bits of one element of the type whose value is dtype; 0 for a
   value that names no type. */
uint32_t gufo_safetensors_dtype_bits(uint32_t dtype);

/* Reads the header length from the first len bytes of a file of file_size
   bytes and checks it against the file. reject may be null. */
GufoSafetensorsParse gufo_safetensors_header_length(
    const uint8_t* prefix, size_t len, uint64_t file_size,
    GufoSafetensorsRejectFn reject, void* context, uint64_t* header_len);

/* Checks the header in header[0, len), which follows the length prefix of a
   file of file_size bytes, then passes its "__metadata__" entries to
   metadata and its tensors to tensor, both in header order. The payload
   starts 8 + len bytes into the file. On GUFO_SAFETENSORS_REJECTED, reject
   (which may be null) got the reason and no entry was passed. */
GufoSafetensorsParse gufo_safetensors_parse_header(
    const uint8_t* header, size_t len, uint64_t file_size,
    GufoSafetensorsMetadataFn metadata, GufoSafetensorsTensorFn tensor,
    GufoSafetensorsRejectFn reject, void* context);

#ifdef __cplusplus
}
#endif

#endif
