#ifndef GUFO_TOKENIZER_JSON_H
#define GUFO_TOKENIZER_JSON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GufoTokenizerJson GufoTokenizerJson;

typedef enum GufoTokenizerJsonStatus {
  GUFO_TOKENIZER_JSON_OK = 0,
  GUFO_TOKENIZER_JSON_INVALID = 1,
  GUFO_TOKENIZER_JSON_INTERNAL = 2,
} GufoTokenizerJsonStatus;

typedef struct GufoTokenizerJsonBytes {
  const uint8_t* data;
  size_t len;
} GufoTokenizerJsonBytes;

typedef struct GufoTokenizerJsonVocabEntry {
  GufoTokenizerJsonBytes token;
  uint32_t id;
} GufoTokenizerJsonVocabEntry;

typedef struct GufoTokenizerJsonMergeEntry {
  GufoTokenizerJsonBytes left;
  GufoTokenizerJsonBytes right;
  bool was_string;
} GufoTokenizerJsonMergeEntry;

typedef struct GufoTokenizerJsonAddedToken {
  GufoTokenizerJsonBytes content;
  uint32_t id;
  bool has_single_word;
  bool single_word;
  bool has_lstrip;
  bool lstrip;
  bool has_rstrip;
  bool rstrip;
  bool has_normalized;
  bool normalized;
  bool flags_valid;
} GufoTokenizerJsonAddedToken;

typedef struct GufoTokenizerJsonMeta {
  bool truncation_is_null;
  bool padding_is_null;

  bool has_model;
  bool has_vocab;
  bool has_merges;
  bool has_added_tokens;

  GufoTokenizerJsonBytes model_type;
  bool has_dropout;
  bool dropout_is_null;
  GufoTokenizerJsonBytes continuing_subword_prefix;
  GufoTokenizerJsonBytes end_of_word_suffix;
  bool has_byte_fallback;
  bool byte_fallback;
  bool has_ignore_merges;
  bool ignore_merges;
  bool has_unk_token;
  bool unk_token_is_null;
  GufoTokenizerJsonBytes unk_token;

  bool has_normalizer;
  GufoTokenizerJsonBytes normalizer_type;
  GufoTokenizerJsonBytes normalizer_pattern_string;
  GufoTokenizerJsonBytes normalizer_content;

  bool has_pre_tokenizer;
  GufoTokenizerJsonBytes pre_tokenizer_type;
  GufoTokenizerJsonBytes pre_tokenizer_pattern_string;
  GufoTokenizerJsonBytes pre_tokenizer_behavior;
  bool has_pre_tokenizer_invert;
  bool pre_tokenizer_invert;

  size_t pre_tokenizer_sub_count;
  GufoTokenizerJsonBytes split_type;
  GufoTokenizerJsonBytes split_behavior;
  bool has_split_invert;
  bool split_invert;
  GufoTokenizerJsonBytes split_regex;
  GufoTokenizerJsonBytes byte_level_type;
  bool has_add_prefix_space;
  bool add_prefix_space;
  bool has_use_regex;
  bool use_regex;

  bool has_post_processor;
  bool post_processor_valid;
  GufoTokenizerJsonBytes post_processor_error;
} GufoTokenizerJsonMeta;

typedef void (*GufoTokenizerJsonErrorFn)(void* ctx, const uint8_t* data,
                                         size_t len);

GufoTokenizerJsonStatus gufo_tokenizer_json_parse(
    const uint8_t* data, size_t len, uint32_t max_token_id,
    GufoTokenizerJson** out, GufoTokenizerJsonErrorFn on_error,
    void* error_ctx);

void gufo_tokenizer_json_free(GufoTokenizerJson* handle);

GufoTokenizerJsonStatus gufo_tokenizer_json_meta(
    const GufoTokenizerJson* handle, GufoTokenizerJsonMeta* out);

size_t gufo_tokenizer_json_vocab_len(const GufoTokenizerJson* handle);
GufoTokenizerJsonStatus gufo_tokenizer_json_vocab_at(
    const GufoTokenizerJson* handle, size_t index,
    GufoTokenizerJsonVocabEntry* out);

size_t gufo_tokenizer_json_merges_len(const GufoTokenizerJson* handle);
GufoTokenizerJsonStatus gufo_tokenizer_json_merge_at(
    const GufoTokenizerJson* handle, size_t index,
    GufoTokenizerJsonMergeEntry* out);

size_t gufo_tokenizer_json_added_len(const GufoTokenizerJson* handle);
GufoTokenizerJsonStatus gufo_tokenizer_json_added_at(
    const GufoTokenizerJson* handle, size_t index,
    GufoTokenizerJsonAddedToken* out);

GufoTokenizerJsonStatus gufo_tokenizer_json_prefix_ids(
    const GufoTokenizerJson* handle, const uint32_t** data, size_t* len);

GufoTokenizerJsonStatus gufo_tokenizer_json_suffix_ids(
    const GufoTokenizerJson* handle, const uint32_t** data, size_t* len);

#ifdef __cplusplus
}
#endif

#endif  // GUFO_TOKENIZER_JSON_H
