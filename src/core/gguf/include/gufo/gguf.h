#ifndef GUFO_GGUF_H_
#define GUFO_GGUF_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust GGUF header reader: the version, the typed
   metadata and the tensor index of GGUF versions 2 and 3, read without
   copying. src/core/gguf_reader.cpp parses every file it opens with it; the
   tensor data stays in its mapping. */
typedef enum GufoGgufParse {
  GUFO_GGUF_COMPLETE = 0,
  /* The file breaks the format or the limits; the error sink got the
     message unless the file is merely malformed or truncated. */
  GUFO_GGUF_REJECTED = 1,
  GUFO_GGUF_NO_MEMORY = 2,
  /* A callback returned nonzero. */
  GUFO_GGUF_STOPPED = 3,
  /* A caller bug (null pointer, missing callback) or a parser bug. */
  GUFO_GGUF_INTERNAL = 4,
} GufoGgufParse;

/* Bytes inside the parsed buffer. */
typedef struct GufoGgufSlice {
  const uint8_t* data;
  size_t len;
} GufoGgufSlice;

/* The field of GufoGgufMetadata that holds the value. Arrays widen integers
   and booleans to 64 bits and floats to double. */
typedef enum GufoGgufKind {
  GUFO_GGUF_UNSIGNED = 0,
  GUFO_GGUF_SIGNED = 1,
  GUFO_GGUF_FLOAT = 2,
  /* unsigned_value is 0 or 1. */
  GUFO_GGUF_BOOL = 3,
  GUFO_GGUF_STRING = 4,
  GUFO_GGUF_STRINGS = 5,
  GUFO_GGUF_UNSIGNEDS = 6,
  GUFO_GGUF_SIGNEDS = 7,
  GUFO_GGUF_FLOATS = 8,
} GufoGgufKind;

/* One metadata entry; its strings point into the parsed buffer. */
typedef struct GufoGgufMetadata {
  GufoGgufSlice key;
  /* The GGUF value type as the file records it (9 for an array). */
  uint32_t value_type;
  GufoGgufKind kind;
  uint64_t unsigned_value;
  int64_t signed_value;
  double float_value;
  GufoGgufSlice string;
  /* The items of an array, which came in blocks before the entry. */
  size_t count;
} GufoGgufMetadata;

/* A block of array items: count items of the kind's field, valid during the
   callback only, from index first of the array's len items on. Strings
   point into the parsed buffer; an empty block's field may point nowhere. */
typedef struct GufoGgufItems {
  GufoGgufKind kind;
  const GufoGgufSlice* strings;
  const uint64_t* unsigneds;
  const int64_t* signeds;
  const double* floats;
  size_t count;
  size_t first;
  size_t len;
} GufoGgufItems;

/* One tensor. Its dimensions are valid during the callback only; its name
   points into the parsed buffer. */
typedef struct GufoGgufTensor {
  GufoGgufSlice name;
  const uint64_t* dimensions;
  size_t dimension_count;
  uint32_t ggml_type;
  /* 0 when tensor_bytes could not size it; size then spans the bytes the
     file reserves for it, up to the next tensor's offset or the data end. */
  int sized;
  /* The offset in the data section, as the file records it. */
  uint64_t offset;
  /* Where the data starts in the parsed buffer, and its bytes. */
  size_t start;
  size_t size;
} GufoGgufTensor;

/* The caller's descriptor budget: the host memory its descriptors may take
   and what it charges per metadata entry, per tensor and per string of a
   string array. Numbers in arrays are charged 8 bytes, key, name and string
   bytes one each. */
typedef struct GufoGgufLimits {
  size_t descriptor_budget;
  size_t metadata_entry_bytes;
  size_t tensor_entry_bytes;
  size_t string_item_bytes;
  /* Accept tensors tensor_bytes cannot size, unless they have no elements. */
  int allow_unsized_tensors;
} GufoGgufLimits;

typedef struct GufoGgufHeader {
  uint32_t version;
  /* The tensor data alignment: general.alignment, or 32. */
  uint64_t alignment;
} GufoGgufHeader;

/* Returns the encoded bytes of a tensor of this storage type and shape
   (elements is 0 when a dimension is 0 or their product overflows), or 0
   when they cannot be sized. */
typedef size_t (*GufoGgufTensorBytesFn)(void* context, uint32_t ggml_type,
                                        uint64_t elements,
                                        uint64_t first_dimension);
/* Takes the entry counts, each at most len; returns nonzero to stop. */
typedef int (*GufoGgufCountsFn)(void* context, size_t metadata_count,
                                size_t tensor_count);
/* Takes a block of array items; returns nonzero to stop. */
typedef int (*GufoGgufItemsFn)(void* context, const GufoGgufItems* items);
/* Takes one metadata entry; returns nonzero to stop. */
typedef int (*GufoGgufMetadataFn)(void* context, const GufoGgufMetadata* entry);
/* Takes one tensor; returns nonzero to stop. */
typedef int (*GufoGgufTensorFn)(void* context, const GufoGgufTensor* tensor);
/* Takes bytes valid during the call only. */
typedef void (*GufoGgufBytesFn)(void* context, const uint8_t* data, size_t len);

/* Parses the header of the GGUF file in data[0, len): counts takes the
   entry counts once they passed the limits, metadata each entry in file order
   once its checks pass, tensor each tensor once all tensors passed theirs.
   Before the entry of an array, items takes its items in blocks, in order;
   each array starts with a block at index 0, empty for an empty array.
   On GUFO_GGUF_COMPLETE, *header holds the version and alignment. Entries
   taken before a failure are void. error may be null; no message is then
   made. */
GufoGgufParse gufo_gguf_parse_header(
    const uint8_t* data, size_t len, const GufoGgufLimits* limits,
    GufoGgufTensorBytesFn tensor_bytes, GufoGgufCountsFn counts,
    GufoGgufItemsFn items, GufoGgufMetadataFn metadata, GufoGgufTensorFn tensor,
    GufoGgufBytesFn error, void* context, GufoGgufHeader* header);

#ifdef __cplusplus
}
#endif

#endif
