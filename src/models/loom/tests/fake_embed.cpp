// See fake_embed.hpp. Implements the loom_text_embed.h functions the runner
// uses.
#include "fake_embed.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>

#include "loom_text_embed.h"

namespace fake {
State& state() {
  static State instance;
  return instance;
}
void reset() {
  state() = State{};
}
int Argmax(const float* row) {
  int best = 0;
  for (int i = 1; i < kVocabulary; ++i)
    if (row[i] > row[best])
      best = i;
  return best;
}
}  // namespace fake

struct loom_text_embed_t {
  struct Row {
    std::uint64_t extent{0};
    std::uint64_t position{0};
    int pending{0};
    bool has_pending{false};
    bool flag{false};
    bool valid{false};
    std::vector<float> logits;
    bool eos{false};
  };
  std::vector<Row> rows;
  bool logits{false};
  char error[256]{};
};
struct loom_text_embed_request_t {
  std::string body;
  bool logits{false};
};
struct loom_text_embed_decoder_t {};

namespace {
void Select(loom_text_embed_t* e, std::uint32_t r, int last) {
  auto& row = e->rows[r];
  std::vector<float> out(fake::kWidth, -3e38F);
  fake::state().script(last, static_cast<int>(row.position), out.data());
  for (int i = fake::kVocabulary; i < fake::kWidth; ++i)
    out[i] = -3e38F;
  row.pending = fake::Argmax(out.data());
  row.eos = row.pending == fake::kEos;
  row.has_pending = true;
  if (row.flag) {
    row.logits = out;
    row.valid = true;
  } else {
    row.valid = false;
  }
}
int Fail(loom_text_embed_t* e, const char* text) {
  std::snprintf(e->error, sizeof(e->error), "%s", text);
  return 2;
}
std::string Render(const std::string& body) {
  const auto& st = fake::state();
  if (body.find("Q7reasoning9") != std::string::npos)
    return "<|a|>Q7reasoning9" + st.reasoning_close + "Q7answer9";
  if (body.find("qprobe") != std::string::npos)
    return "<tools>qprobe</tools>" + st.tool_call_text;
  return "PROMPT:" + body.substr(0, 8);
}
}  // namespace

extern "C" {
int loom_text_embed_open_ex(const loom_text_embed_options_ex_t* o,
                            loom_text_embed_t** out, char*, size_t) {
  auto* e = new loom_text_embed_t;
  e->rows.resize(o->base.rows ? o->base.rows : 4);
  e->logits = o->logits != 0 && fake::state().declares_logits;
  *out = e;
  return 0;
}
int loom_text_embed_open(const loom_text_embed_options_t* o,
                         loom_text_embed_t** out, char* err, size_t n) {
  loom_text_embed_options_ex_t ex{sizeof(ex), *o, 0};
  return loom_text_embed_open_ex(&ex, out, err, n);
}
void loom_text_embed_close(loom_text_embed_t* e) {
  delete e;
}
const char* loom_text_embed_error(const loom_text_embed_t* e) {
  return e->error;
}
void loom_text_embed_info(const loom_text_embed_t* e,
                          loom_text_embed_info_t* i) {
  std::memset(i, 0, sizeof(*i));
  std::snprintf(i->model_id, sizeof(i->model_id), "fake");
  i->context_capacity = 4096;
  i->prefill_capacity = 64;
  i->rows = static_cast<std::uint32_t>(e->rows.size());
  i->max_epoch_spans = i->rows;
  i->draft_depth = fake::state().draft_depth;
}
void loom_text_embed_logits_info(const loom_text_embed_t* e,
                                 loom_text_embed_logits_info_t* i) {
  std::memset(i, 0, sizeof(*i));
  i->supported = fake::state().declares_logits;
  i->enabled = e->logits;
  i->width = e->logits ? fake::kWidth : 0;
  i->max_rows = static_cast<std::uint32_t>(e->rows.size());
}
int loom_text_embed_chat_prepare(loom_text_embed_t*, const char* body, size_t n,
                                 int32_t* tokens, size_t capacity,
                                 size_t* out_count,
                                 loom_text_embed_request_t** out_request) {
  const std::string text = Render(std::string(body, n));
  if (tokens) {
    for (size_t i = 0; i < text.size() && i < capacity; ++i)
      tokens[i] = static_cast<unsigned char>(text[i]);
  }
  *out_count = tokens ? text.size() : 0;
  if (out_request) {
    auto* r = new loom_text_embed_request_t;
    r->body.assign(body, n);
    r->logits = r->body.find("\"logits\":true") != std::string::npos ||
                r->body.find("\"logits\": true") != std::string::npos;
    *out_request = r;
  }
  return 0;
}
void loom_text_embed_request_free(loom_text_embed_request_t* r) {
  delete r;
}
size_t loom_text_embed_request_media(const loom_text_embed_request_t*,
                                     uint32_t*, uint32_t*, size_t) {
  return 0;
}
int loom_text_embed_request_project(loom_text_embed_request_t*) {
  return 0;
}
int loom_text_embed_row_prefill_request(loom_text_embed_t*, uint32_t,
                                        loom_text_embed_request_t*, size_t,
                                        size_t, int) {
  return 2;
}
int loom_text_embed_output_feed(loom_text_embed_request_t*, const char* text,
                                size_t n, int, char* out, size_t capacity,
                                size_t* out_size) {
  *out_size = std::min(n, capacity);
  std::memcpy(out, text, *out_size);
  return 0;
}
int loom_text_embed_output_close(loom_text_embed_request_t*, char*, size_t,
                                 size_t* out_size) {
  *out_size = 0;
  return 0;
}
int loom_text_embed_output_tools(loom_text_embed_request_t*, int, char*, size_t,
                                 size_t* out_size) {
  *out_size = 0;
  return 0;
}
int loom_text_embed_tokenize(loom_text_embed_t*, const char* text, size_t n,
                             int, int32_t* tokens, size_t capacity,
                             size_t* out) {
  *out = n;
  if (tokens)
    for (size_t i = 0; i < n && i < capacity; ++i)
      tokens[i] = static_cast<unsigned char>(text[i]);
  return 0;
}
int loom_text_embed_detokenize(loom_text_embed_t*, const int32_t* tokens,
                               size_t count, char* text, size_t capacity,
                               size_t* out_size) {
  std::string s;
  for (size_t i = 0; i < count; ++i)
    if (tokens[i] >= 0 && tokens[i] < fake::kBytes)
      s += static_cast<char>(tokens[i]);
  *out_size = s.size();
  if (text) {
    if (s.size() > capacity)
      return 2;
    std::memcpy(text, s.data(), s.size());
  }
  return 0;
}
int32_t loom_text_embed_vocabulary_size(loom_text_embed_t*) {
  return fake::kVocabulary;
}
int loom_text_embed_decoder_open(loom_text_embed_t*,
                                 loom_text_embed_decoder_t** d) {
  *d = new loom_text_embed_decoder_t;
  return 0;
}
int loom_text_embed_decoder_feed(loom_text_embed_decoder_t*,
                                 const int32_t* tokens, size_t count,
                                 char* text, size_t capacity,
                                 size_t* out_size) {
  size_t n = 0;
  for (size_t i = 0; i < count; ++i)
    if (tokens[i] >= 0 && tokens[i] < fake::kBytes && n < capacity)
      text[n++] = static_cast<char>(tokens[i]);
  *out_size = n;
  return 0;
}
int loom_text_embed_decoder_reset(loom_text_embed_decoder_t*) {
  return 0;
}
void loom_text_embed_decoder_free(loom_text_embed_decoder_t* d) {
  delete d;
}

int loom_text_embed_row_begin(loom_text_embed_t* e, uint32_t row,
                              const loom_text_embed_request_t* request,
                              uint64_t extent, int fresh) {
  auto& r = e->rows[row];
  r.extent = extent;
  if (fresh) {
    r.position = 0;
    r.has_pending = false;
    r.valid = false;
  }
  if (request) {
    if (request->logits && !e->logits)
      return Fail(e, "no logits");
    r.flag = request->logits;
  }
  return 0;
}
int loom_text_embed_row_extend(loom_text_embed_t* e, uint32_t row,
                               uint64_t extent) {
  e->rows[row].extent = extent;
  return 0;
}
uint64_t loom_text_embed_row_extent(loom_text_embed_t* e, uint32_t row) {
  return e->rows[row].extent;
}
int loom_text_embed_row_bind(loom_text_embed_t* e, uint32_t row,
                             const loom_text_embed_request_t* request) {
  auto& r = e->rows[row];
  if (request->logits && !e->logits)
    return Fail(e, "no logits");
  if (r.flag != request->logits)
    r.valid = false;
  r.flag = request->logits;
  return 0;
}
void loom_text_embed_row_end(loom_text_embed_t* e, uint32_t row) {
  e->rows[row].extent = 0;
  e->rows[row].flag = false;
  e->rows[row].valid = false;
}
int loom_text_embed_row_reset(loom_text_embed_t* e, uint32_t row) {
  e->rows[row] = loom_text_embed_t::Row{};
  return 0;
}
int loom_text_embed_row_prefill(loom_text_embed_t* e, uint32_t row,
                                const int32_t* tokens, size_t count) {
  auto& r = e->rows[row];
  r.position += count;
  Select(e, row, tokens[count - 1]);
  fake::Epoch epoch;
  epoch.rows = {row};
  epoch.tokens = {tokens[count - 1]};
  epoch.exported = {r.valid};
  fake::state().epochs.push_back(epoch);
  return 0;
}
int loom_text_embed_row_decode(loom_text_embed_t* e, uint32_t row) {
  const int token = e->rows[row].pending;
  const std::uint32_t rows[1] = {row};
  const int32_t tokens[1] = {token};
  e->rows[row].position += 1;
  Select(e, row, token);
  fake::Epoch epoch;
  epoch.rows = {rows[0]};
  epoch.tokens = {tokens[0]};
  epoch.exported = {e->rows[row].valid};
  fake::state().epochs.push_back(epoch);
  return 0;
}
int32_t loom_text_embed_row_token(loom_text_embed_t* e, uint32_t row) {
  return e->rows[row].pending;
}
int loom_text_embed_row_is_eos(loom_text_embed_t* e, uint32_t row) {
  return e->rows[row].eos;
}
uint64_t loom_text_embed_row_position(loom_text_embed_t* e, uint32_t row) {
  return e->rows[row].position;
}
int loom_text_embed_decode_batch(loom_text_embed_t* e, const uint32_t* rows,
                                 size_t count) {
  if (count == 1)
    return loom_text_embed_row_decode(e, rows[0]);
  fake::Epoch epoch;
  for (size_t i = 0; i < count; ++i) {
    auto& r = e->rows[rows[i]];
    const int token = r.pending;
    r.position += 1;
    Select(e, rows[i], token);
    epoch.rows.push_back(rows[i]);
    epoch.tokens.push_back(token);
    epoch.exported.push_back(r.valid);
  }
  fake::state().epochs.push_back(epoch);
  return 0;
}
int loom_text_embed_decode_batch_tokens(loom_text_embed_t* e,
                                        const uint32_t* rows, size_t count,
                                        const int32_t* tokens) {
  fake::Epoch epoch;
  epoch.host_tokens = true;
  for (size_t i = 0; i < count; ++i) {
    auto& r = e->rows[rows[i]];
    r.position += 1;
    Select(e, rows[i], tokens[i]);
    epoch.rows.push_back(rows[i]);
    epoch.tokens.push_back(tokens[i]);
    epoch.exported.push_back(r.valid);
  }
  fake::state().epochs.push_back(epoch);
  return 0;
}
int loom_text_embed_verify_batch(loom_text_embed_t* e, const uint32_t*, size_t,
                                 const uint32_t*, int32_t*, uint32_t*,
                                 uint32_t*) {
  return Fail(e, "no drafting");
}
int loom_text_embed_row_logits(loom_text_embed_t* e, uint32_t row, float* out,
                               size_t capacity, size_t* out_count) {
  *out_count = 0;
  auto& r = e->rows[row];
  if (!r.flag || !r.valid)
    return Fail(e, "row has no logits");
  if (capacity < fake::kWidth)
    return Fail(e, "capacity");
  std::memcpy(out, r.logits.data(), sizeof(float) * fake::kWidth);
  *out_count = fake::kWidth;
  return 0;
}
int loom_text_embed_request_logits(const loom_text_embed_request_t* r) {
  return r->logits;
}
int loom_text_embed_row_is_logit(loom_text_embed_t* e, uint32_t row) {
  return e->rows[row].flag;
}
int loom_text_embed_is_stop_token(const loom_text_embed_t*, int32_t token) {
  return token == fake::kEos;
}
void loom_text_embed_memory(const loom_text_embed_t*,
                            loom_text_embed_memory_t* memory) {
  *memory = {};
}
int loom_text_embed_query_gpu(loom_text_embed_gpu_t* gpu, char*, size_t) {
  *gpu = {};
  std::snprintf(gpu->target, sizeof(gpu->target), "gfx1151");
  return 0;
}
int loom_text_embed_row_pin(loom_text_embed_t*, uint32_t,
                            loom_text_embed_checkpoint_t**) {
  return 1;
}
int loom_text_embed_row_restore(loom_text_embed_t*, uint32_t,
                                const loom_text_embed_checkpoint_t*, int*) {
  return 1;
}
void loom_text_embed_checkpoint_release(loom_text_embed_t*,
                                        loom_text_embed_checkpoint_t*) {}
void loom_text_embed_image_layout(loom_text_embed_t*,
                                  loom_text_embed_image_layout_t* l) {
  std::memset(l, 0, sizeof(*l));
}
int loom_text_embed_checkpoint_export(loom_text_embed_t*,
                                      const loom_text_embed_checkpoint_t*,
                                      uint8_t*, size_t, size_t*,
                                      loom_text_embed_checkpoint_info_t*) {
  return 1;
}
int loom_text_embed_checkpoint_import(loom_text_embed_t*,
                                      const loom_text_embed_checkpoint_info_t*,
                                      const uint8_t*, size_t,
                                      loom_text_embed_checkpoint_t**) {
  return 1;
}
}  // extern "C"
