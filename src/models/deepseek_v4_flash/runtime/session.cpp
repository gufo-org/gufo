#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <memory>
#include <new>

#include "dspark_internal.h"
#include "dspark_policy.h"
#include "native_internal.h"

#define DS4_SESSION_NEG_INF (-1.0e30f)

struct ds4_session {
    ds4_engine *engine;
    ds4_rocm_graph *graph;
    ds4_tokens checkpoint;
    std::unique_ptr<float[]> logits;
    ds4_session_cancel_fn cancel;
    void *cancel_user_data;
    uint32_t prefill_capacity;
    int context_size;
    bool checkpoint_valid;
    bool use_dspark;
    ds4_dspark_request_state dspark_state{};
    /* Verified target rows for a sampled concurrent cycle; grown on demand. */
    std::unique_ptr<float[]> spec_rows;
    uint32_t spec_rows_cap = 0;
};

static void set_error(char *error,
                      size_t error_capacity,
                      const char *message) {
    if (error && error_capacity != 0) {
        snprintf(error, error_capacity, "%s", message);
    }
}

static bool session_cancelled(const ds4_session *session) {
    return session && session->cancel &&
           session->cancel(session->cancel_user_data);
}

static void session_dspark_reset_request_state(ds4_session *session,
                                               bool preserve_force_plain) {
    if (!session) return;
    const bool force_plain =
        !session->use_dspark ||
        (preserve_force_plain && session->dspark_state.force_plain_request);
    session->dspark_state = {};
    session->dspark_state.force_plain_request = force_plain;
    session->dspark_state.plain_only = force_plain;
    ds4_rocm_graph_dspark_set_capture_enabled(session->graph, !force_plain);
}

int ds4_session_create(ds4_session** out, ds4_engine* engine, int context_size,
                       bool use_dspark) {
  if (!out || !engine || context_size <= 0 || (use_dspark && !engine->dspark))
    return 1;

  auto session = std::unique_ptr<ds4_session>(new (std::nothrow) ds4_session{});
  if (!session)
    return 1;

  session->engine = engine;
  session->use_dspark = use_dspark;
  session->context_size = context_size;
  session->prefill_capacity = ds4_rocm_graph_prefill_capacity(context_size);
  session->graph =
      ds4_rocm_graph_create(engine, context_size, session->prefill_capacity);
  if (!session->graph) {
    return 1;
  }

  const int vocabulary_size = ds4_engine_vocab_size(engine);
  if (vocabulary_size <= 0 ||
      (size_t)vocabulary_size > SIZE_MAX / sizeof(session->logits[0])) {
    ds4_rocm_graph_destroy(session->graph);
    return 1;
  }
  session->logits.reset(new (std::nothrow) float[vocabulary_size]);
  if (!session->logits) {
    ds4_rocm_graph_destroy(session->graph);
    return 1;
  }

  /* Prefill captures target features and seeds the attached support model's
   * KV. */
  if (use_dspark) {
    if (!ds4_rocm_graph_dspark_attach(session->graph, engine, engine->dspark)) {
      ds4_rocm_graph_destroy(session->graph);
      return 1;
    }
  }

  *out = session.release();
  return 0;
}

void ds4_session_free(ds4_session *session) {
    if (!session) return;
    ds4_rocm_graph_destroy(session->graph);
    ds4_tokens_free(&session->checkpoint);
    delete session;
}

bool ds4_session_dspark_enabled(const ds4_session* session) {
  return session && session->use_dspark;
}

void ds4_session_set_cancel(ds4_session *session,
                            ds4_session_cancel_fn callback,
                            void *user_data) {
    if (!session) return;
    session->cancel = callback;
    session->cancel_user_data = user_data;
}

void ds4_session_invalidate(ds4_session *session) {
    if (!session) return;
    session->checkpoint_valid = false;
    session_dspark_reset_request_state(session, false);
}

void ds4_session_begin_request(ds4_session* session) {
  if (!session)
    return;
  const bool incomplete_support =
      session->use_dspark && ds4_rocm_graph_dspark_context_len(session->graph) <
                                 static_cast<uint32_t>(session->checkpoint.len);
  session_dspark_reset_request_state(session, false);
  if (incomplete_support)
    ds4_session_prepare_batch_execution(session);
}

void ds4_session_prepare_batch_execution(ds4_session *session) {
    if (!session || ds4_rocm_graph_dspark_block_size(session->graph) == 0u) {
        return;
    }
    session->dspark_state.force_plain_request = true;
    session->dspark_state.plain_only = true;
    ds4_rocm_graph_dspark_set_capture_enabled(session->graph, false);
}

/*
 * Evaluate one target token and extend the support ring with the sampled target
 * features captured by that decode. An injection after disabled capture cannot
 * claim contiguous coverage across the missing support rows.
 */
static bool session_commit_and_extend(ds4_session *session, int token) {
    const uint32_t position = (uint32_t)session->checkpoint.len;
    if (!ds4_rocm_graph_eval(session->graph,
                             session->engine,
                             token,
                             position,
                             session->logits.get())) {
        return false;
    }
    ds4_tokens_push(&session->checkpoint, token);
    if (ds4_rocm_graph_dspark_block_size(session->graph) != 0 &&
        ds4_rocm_graph_dspark_capture_ready(session->graph)) {
      return ds4_rocm_graph_dspark_inject(session->graph, position, 1u);
    }
    return true;
}

static bool sessions_commit_and_extend_batch(
    const ds4_session_dspark_batch_item* items,
    const std::array<int, 8>& tokens, size_t item_count) {
  if (!items || item_count < 2u || item_count > 8u)
    return false;
  std::array<ds4_rocm_batch_item, 8> batch{};
  for (size_t index = 0; index < item_count; ++index) {
    ds4_session* session = items[index].session;
    batch[index] = {
        .graph = session->graph,
        .token = tokens[index],
        .position = static_cast<uint32_t>(session->checkpoint.len),
        .logits = session->logits.get(),
    };
  }
  if (!ds4_rocm_graph_eval_batch(items[0].session->engine, batch.data(),
                                 item_count)) {
    return false;
  }
  for (size_t index = 0; index < item_count; ++index) {
    ds4_session* session = items[index].session;
    const uint32_t position = static_cast<uint32_t>(session->checkpoint.len);
    ds4_tokens_push(&session->checkpoint, tokens[index]);
    if (ds4_rocm_graph_dspark_block_size(session->graph) != 0u &&
        ds4_rocm_graph_dspark_capture_ready(session->graph) &&
        !ds4_rocm_graph_dspark_inject(session->graph, position, 1u)) {
      return false;
    }
  }
  return true;
}

/*
 * Materialize the latest captured target batch into the support KV ring only
 * when drafting actually needs it. The capture may be an initial prompt or a
 * resumed-prefill suffix; contiguous injections compose without repeating
 * already seeded rows.
 */
static bool session_dspark_seed_pending(ds4_session *session,
                                        uint32_t required_context) {
    if (!session || ds4_rocm_graph_dspark_block_size(session->graph) == 0) {
        return false;
    }
    uint32_t context =
        ds4_rocm_graph_dspark_context_len(session->graph);
    if (context >= required_context) return true;

    uint32_t start = 0;
    const uint32_t rows =
        ds4_rocm_graph_dspark_batch_capture_rows(session->graph, &start);
    if (rows == 0 || start > context || start + rows < required_context) {
        return false;
    }
    if (!ds4_rocm_graph_dspark_inject(session->graph, start, rows)) {
        return false;
    }
    context = ds4_rocm_graph_dspark_context_len(session->graph);
    return context >= required_context;
}

/*
 * The target-known token that starts the next block. Greedy cycles take the
 * frontier argmax. Sampled cycles emit the token the previous verification
 * rejected (already drawn, so the request RNG must not advance again) or draw
 * one from the frontier logits with the request sampler.
 */
static int session_dspark_anchor(ds4_session* session,
                                 const ds4_dspark_sampler* sampler,
                                 int vocabulary_size) {
  if (sampler == nullptr) {
    return ds4_sample_argmax(session->logits.get(), (uint32_t)vocabulary_size);
  }
  const int pending = session->dspark_state.pending_anchor;
  session->dspark_state.pending_anchor = -1;
  if (pending >= 0) {
    if (sampler->accept) sampler->accept(sampler->ctx, pending);
    return pending;
  }
  return sampler->sample(sampler->ctx, session->logits.get(),
                         (uint32_t)vocabulary_size);
}

static float* session_spec_rows(ds4_session* session, uint32_t rows,
                                int vocabulary_size) {
  if (rows == 0u) return nullptr;
  if (session->spec_rows_cap < rows) {
    session->spec_rows.reset(
        new (std::nothrow) float[(size_t)rows * (size_t)vocabulary_size]);
    session->spec_rows_cap = session->spec_rows ? rows : 0u;
  }
  return session->spec_rows.get();
}

int ds4_session_dspark_take_pending_anchor(ds4_session* session) {
  if (!session) return -1;
  const int pending = session->dspark_state.pending_anchor;
  session->dspark_state.pending_anchor = -1;
  return pending;
}

int ds4_session_sync(ds4_session *session,
                     const ds4_tokens *prompt,
                     char *error,
                     size_t error_capacity) {
  if (!session || !prompt || !prompt->v || prompt->len <= 0 ||
      prompt->len >= session->context_size) {
    set_error(error, error_capacity, "prompt exceeds context");
    return 1;
  }
  const int vocabulary_size = ds4_engine_vocab_size(session->engine);
  if (std::any_of(prompt->v, prompt->v + prompt->len,
                  [vocabulary_size](int token) {
                    return token < 0 || token >= vocabulary_size;
                  })) {
    set_error(error, error_capacity, "prompt contains an invalid token ID");
    return 1;
  }
    if (session_cancelled(session)) {
        set_error(error, error_capacity, "prefill cancelled");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }

    if (session->checkpoint_valid && prompt->len >= session->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &session->checkpoint)) {
      const int suffix = prompt->len - session->checkpoint.len;
      if (suffix > 0) {
        /* Materialize any pending capture before suffix prefill reuses its
         * storage. */
        (void)session_dspark_seed_pending(session,
                                          (uint32_t)session->checkpoint.len);
      }
      const uint32_t resume_minimum =
          ds4_rocm_graph_resume_prefill_min_tokens();
      if (suffix > 0 && (uint32_t)suffix >= resume_minimum) {
        const bool ok = ds4_rocm_graph_prefill_range(
            session->graph, session->engine, prompt,
            (uint32_t)session->checkpoint.len, (uint32_t)suffix,
            session->logits.get());
        if (!ok) {
          set_error(error, error_capacity,
                    "ROCm resumed prefill failed while extending checkpoint");
          session->checkpoint_valid = false;
          return 1;
        }
        ds4_tokens_copy(&session->checkpoint, prompt);
        session->checkpoint_valid = true;
        return 0;
      }

      for (int i = session->checkpoint.len; i < prompt->len; ++i) {
        if (session_cancelled(session)) {
          set_error(error, error_capacity, "prefill cancelled");
          return DS4_SESSION_SYNC_INTERRUPTED;
        }
        if (!session_commit_and_extend(session, prompt->v[i])) {
          set_error(error, error_capacity,
                    "ROCm decode failed while extending checkpoint");
          session->checkpoint_valid = false;
          return 1;
        }
      }
      return 0;
    }

    session->checkpoint_valid = false;
    session_dspark_reset_request_state(session, true);
    if (!ds4_rocm_graph_reset(session->graph)) {
        set_error(error, error_capacity, "ROCm prefill state reset failed");
        return 1;
    }
    if (!ds4_rocm_graph_prefill(session->graph,
                                session->engine,
                                prompt,
                                session->logits.get())) {
        set_error(error, error_capacity, "ROCm prefill failed");
        return 1;
    }
    ds4_tokens_copy(&session->checkpoint, prompt);
    session->checkpoint_valid = true;
    return 0;
}

int ds4_session_argmax(const ds4_session *session) {
  if (!session || !session->checkpoint_valid || !session->logits)
    return -1;
  return ds4_sample_argmax(session->logits.get(),
                           (uint32_t)ds4_engine_vocab_size(session->engine));
}

int ds4_session_sample(const ds4_session *session,
                       float temperature,
                       int top_k,
                       float top_p,
                       float min_p,
                       uint64_t *rng_state) {
  if (!session || !session->checkpoint_valid || !session->logits || !rng_state)
    return -1;
  return ds4_sample_top_p_min_p(
      session->logits.get(), (uint32_t)ds4_engine_vocab_size(session->engine),
      temperature, top_k, top_p, min_p, rng_state);
}

int ds4_session_copy_logits(const ds4_session *session,
                            float *out,
                            int capacity) {
  if (!session || !session->checkpoint_valid || !out)
    return 0;
  const int vocabulary_size = ds4_engine_vocab_size(session->engine);
  if (capacity < vocabulary_size)
    return 0;
  memcpy(out, session->logits.get(), (size_t)vocabulary_size * sizeof(out[0]));
  return vocabulary_size;
}

int ds4_session_eval(ds4_session *session,
                     int token,
                     char *error,
                     size_t error_capacity) {
    if (!session) {
        set_error(error, error_capacity, "invalid session");
        return 1;
    }
    if (!session->checkpoint_valid ||
        session->checkpoint.len >= session->context_size || token < 0 ||
        token >= ds4_engine_vocab_size(session->engine)) {
      set_error(error, error_capacity,
                "decode needs a valid checkpoint, token, and context position");
      return 1;
    }
    if (session_cancelled(session)) {
        set_error(error, error_capacity, "decode cancelled");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }
    if (!session_commit_and_extend(session, token)) {
        set_error(error, error_capacity, "ROCm decode failed");
        session->checkpoint_valid = false;
        return 1;
    }
    session->checkpoint_valid = true;
    return 0;
}

int ds4_sessions_eval_batch(const ds4_session_batch_item *items,
                            size_t item_count,
                            char *error,
                            size_t error_capacity) {
    if (!items || item_count < 2 || item_count > 8) {
        set_error(error,
                  error_capacity,
                  "batch decode requires two to eight sessions");
        return 1;
    }

    ds4_engine *engine = nullptr;
    std::array<ds4_rocm_batch_item, 8> graph_items{};
    size_t graph_item_count = 0;
    for (size_t index = 0; index < item_count; ++index) {
        ds4_session *session = items[index].session;
        if (!session || !session->checkpoint_valid) {
            set_error(error,
                      error_capacity,
                      "batch decode contains an invalid session");
            return 1;
        }
        if (engine == nullptr) {
            engine = session->engine;
        } else if (session->engine != engine) {
            set_error(error,
                      error_capacity,
                      "batch decode sessions do not share one model");
            return 1;
        }
        if (session_cancelled(session)) {
            set_error(error, error_capacity, "batch decode cancelled");
            return DS4_SESSION_SYNC_INTERRUPTED;
        }
        if (items[index].token < 0 ||
            items[index].token >= ds4_engine_vocab_size(session->engine)) {
          set_error(error, error_capacity,
                    "batch decode contains an invalid token");
          return 1;
        }
        if (session->checkpoint.len < 0 ||
            session->checkpoint.len >= session->context_size) {
          set_error(error, error_capacity,
                    "batch decode exceeds a session context");
          return 1;
        }
        for (size_t previous = 0; previous < index; ++previous) {
            if (items[previous].session == session) {
                set_error(error,
                          error_capacity,
                          "batch decode contains a duplicate session");
                return 1;
            }
        }
        graph_items[graph_item_count++] = {
            .graph = session->graph,
            .token = items[index].token,
            .position = static_cast<uint32_t>(session->checkpoint.len),
            .logits = session->logits.get(),
        };
    }

    if (!ds4_rocm_graph_eval_batch(engine,
                                   graph_items.data(),
                                   graph_item_count)) {
        for (size_t index = 0; index < item_count; ++index) {
            items[index].session->checkpoint_valid = false;
        }
        set_error(error, error_capacity, "ROCm batch decode failed");
        return 1;
    }

    for (size_t index = 0; index < item_count; ++index) {
        ds4_tokens_push(&items[index].session->checkpoint, items[index].token);
        items[index].session->checkpoint_valid = true;
        /*
         * The multi-session target path does not capture per-session sampled
         * hidden states for the support ring. Keep this request on exact target
         * batching after its first concurrent step; a cold request reset
         * enables DSpark again.
         */
        items[index].session->dspark_state.plain_only = true;
        items[index].session->dspark_state.force_plain_request = true;
        ds4_rocm_graph_dspark_set_capture_enabled(
            items[index].session->graph, false);
    }
    return 0;
}

uint64_t ds4_session_payload_bytes(const ds4_session *session) {
    if (!session || !session->checkpoint_valid) return 0;
    return ds4_rocm_graph_snapshot_bytes(session->graph,
                                         &session->checkpoint);
}

int ds4_session_save_snapshot(const ds4_session *session,
                              ds4_session_snapshot *snapshot,
                              char *error,
                              size_t error_capacity) {
    if (!session || !session->checkpoint_valid) {
        set_error(error, error_capacity, "session has no valid checkpoint");
        return 1;
    }
    return ds4_rocm_graph_save_snapshot(
        session->graph, &session->checkpoint, session->logits.get(),
        session->prefill_capacity, session->context_size,
        &session->dspark_state, snapshot, error, error_capacity);
}

int ds4_session_load_snapshot(ds4_session *session,
                              const ds4_session_snapshot *snapshot,
                              char *error,
                              size_t error_capacity) {
    if (!session) {
        set_error(error, error_capacity, "invalid session snapshot load");
        return 1;
    }
    session->checkpoint_valid = false;
    const int result = ds4_rocm_graph_load_snapshot(
        session->graph, &session->checkpoint, session->logits.get(),
        session->prefill_capacity, session->context_size,
        &session->dspark_state, snapshot, error, error_capacity);
    session->checkpoint_valid = result == 0;
    return result;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snapshot) {
    if (!snapshot) return;
    free(snapshot->ptr);
    memset(snapshot, 0, sizeof(*snapshot));
}

/* C1 retains the established four-probe/64-token acceptance policy. Concurrent
 * requests use a separate recent-cost window after their batched cycle. */
static void session_dspark_note_cycle(ds4_session* session,
                                      bool concurrent_batch) {
  if (concurrent_batch) {
    return;
  }
  constexpr uint64_t kEarlyCycles = 4u;
  constexpr double kEarlyRejectRate = 0.25;
  constexpr uint64_t kWarmupCycles = 8u;
  constexpr double kBreakEvenAcceptance = 0.48;
  constexpr uint32_t kSerialLowAcceptanceBackoff = DS4_DSPARK_RETRY_TOKENS;

  const auto& state = session->dspark_state;
  const uint64_t drafted = state.support_drafted - state.policy_start_drafted;
  const uint64_t accepted =
      state.support_accepted - state.policy_start_accepted;
  const uint64_t steps = state.steps - state.policy_start_steps;
  if (drafted == 0u || steps < kEarlyCycles)
    return;
  const double acceptance =
      static_cast<double>(accepted) / static_cast<double>(drafted);
  if (steps < kWarmupCycles && acceptance >= kEarlyRejectRate) {
    return;
  }
  if (acceptance >= kBreakEvenAcceptance) {
    session->dspark_state.skip_remaining = 0u;
    return;
  }
  session->dspark_state.skip_remaining = kSerialLowAcceptanceBackoff;
}

// Cost units are hundredths of one ordinary batch decode step, calibrated
// offline on gfx1151 at 0/4K/16K with tg128, rounded up to five units.
// See docs/models/deepseek-v4-flash/artifacts/cost-calibration.json. Runtime wall clocks
// never influence token decisions.
static uint32_t session_dspark_cycle_cost(const ds4_session* session,
                                          uint32_t tail, size_t concurrency) {
  return ds4_dspark_cycle_cost(session->checkpoint.len, tail, concurrency);
}

static uint32_t session_dspark_profitable_tail(const ds4_session* session,
                                               uint32_t maximum,
                                               size_t concurrency) {
  uint32_t best_tail = 0;
  uint32_t best_cost = 100;
  for (uint32_t tail = 1; tail <= std::min(maximum, 3u); ++tail) {
    const uint32_t cost = session_dspark_cycle_cost(session, tail, concurrency);
    // Compare cost per emitted token under full acceptance, including the
    // target-known first token. Zero selects ordinary target decoding.
    if (cost * (best_tail + 1u) < best_cost * (tail + 1u)) {
      best_tail = tail;
      best_cost = cost;
    }
  }
  return best_tail;
}

static void session_dspark_use_concurrency(ds4_session* session,
                                           size_t concurrency) {
  auto& state = session->dspark_state;
  if (state.policy_concurrency == concurrency)
    return;
  state.policy_concurrency = static_cast<uint32_t>(concurrency);
  state.policy_start_drafted = state.support_drafted;
  state.policy_start_accepted = state.support_accepted;
  state.policy_start_steps = state.steps;
  state.concurrent_width = 3u;
  state.probe_cycles = state.probe_emitted = state.probe_cost = 0;
  state.skip_remaining = 0;
}

static void session_dspark_note_concurrent_cycle(ds4_session* session,
                                                 uint32_t drafted,
                                                 uint32_t accepted,
                                                 size_t concurrency) {
  if (drafted == 0)
    return;
  auto& state = session->dspark_state;
  // Let C2 build its adaptive tail before judging a cold start.
  if (concurrency <= 2u && state.steps - state.policy_start_steps <= 4u)
    return;
  const uint32_t cost =
      session_dspark_cycle_cost(session, drafted, concurrency);
  // One probe is sufficient when even full acceptance cannot repay the
  // calibrated cycle. Keep a bounded retry so later request membership or
  // context can still resume useful drafting.
  if (cost >= 100u * (drafted + 1u)) {
    state.skip_remaining = DS4_DSPARK_RETRY_TOKENS;
    state.probe_cycles = state.probe_emitted = state.probe_cost = 0;
    return;
  }
  ++state.probe_cycles;
  state.probe_emitted += 1u + accepted;
  state.probe_cost += cost;
  if (state.probe_cycles < 4u)
    return;
  // Retry after a bounded number of target tokens; their captured features keep
  // the support state complete, so a later profitable cohort can resume
  // drafting.
  if (100u * state.probe_emitted < state.probe_cost)
    state.skip_remaining = DS4_DSPARK_RETRY_TOKENS;
  state.probe_cycles = state.probe_emitted = state.probe_cost = 0;
}

static uint32_t session_dspark_adaptive_width(const ds4_session* session,
                                              uint32_t maximum) {
  if (!session || maximum == 0u) {
    return 0u;
  }
  return std::clamp(session->dspark_state.concurrent_width, 1u, maximum);
}

static uint32_t session_dspark_usable_tail(const ds4_session* session,
                                           uint32_t maximum, int emitted_cap) {
  const int context_room = session->context_size - session->checkpoint.len;
  if (emitted_cap <= 1 || context_room <= 2) {
    return 0;
  }
  return std::min({session_dspark_adaptive_width(session, maximum),
                   static_cast<uint32_t>(emitted_cap - 1),
                   static_cast<uint32_t>(context_room - 2)});
}

static void session_dspark_update_adaptive_width(ds4_session* session,
                                                 uint32_t drafted,
                                                 uint32_t accepted,
                                                 uint32_t maximum) {
  if (!session || drafted == 0u || maximum == 0u) {
    return;
  }
  const uint32_t previous = session_dspark_adaptive_width(session, maximum);
  uint32_t next = previous;
  if (accepted >= drafted) {
    next = std::min(previous + 1u, maximum);
  } else if (accepted * 2u < drafted) {
    next = std::max(previous - 1u, 1u);
  }
  session->dspark_state.concurrent_width = next;
  if (getenv("GUFO_DEEPSEEK_DSPARK_WIDTH_TRACE") != nullptr) {
    fprintf(stderr,
            "ds4: DSpark adaptive width previous=%u drafted=%u "
            "accepted=%u next=%u maximum=%u\n",
            previous, drafted, accepted, next, maximum);
  }
}

/*
 * Decide the accepted prefix of a verified block.
 *
 * Greedy requests accept while the target argmax reproduces the draft. Sampled
 * requests use point-mass or p/q verification. The first rejected target or
 * residual draw becomes the next anchor. Row i is the target output after
 * draft i, read from `row_logits`
 * (concurrent cycles) or on demand from the graph (single cycles, where the
 * last read row is the frontier).
 */
static uint32_t session_dspark_accept_prefix(
    ds4_session* session, const ds4_dspark_sampler* sampler,
    const float* row_logits, int vocabulary_size, const int32_t* drafts,
    uint32_t drafted, const int32_t* row_tops, bool stop_at_eos,
    bool* frontier_read) {
  uint32_t accepted = 1;
  *frontier_read = false;
  if (sampler == nullptr) {
    while (accepted < drafted && row_tops[accepted - 1] == drafts[accepted]) {
      if (stop_at_eos && ds4_token_is_stop(session->engine, drafts[accepted])) {
        return accepted;
      }
      accepted++;
    }
    return accepted;
  }
  while (accepted < drafted) {
    int drawn = row_tops[accepted - 1u];
    if (drawn >= 0 && drawn < vocabulary_size &&
        sampler->try_accept_argmax != nullptr &&
        sampler->try_accept_argmax(sampler->ctx, drawn)) {
      // A legal greedy argmax needs no vocabulary download or full mask.
      // A rejection below must still fetch this row as its reusable frontier.
      *frontier_read = false;
    } else {
      const float* row = nullptr;
      if (row_logits != nullptr) {
        row = row_logits + (size_t)(accepted - 1u) * (size_t)vocabulary_size;
      } else if (ds4_rocm_graph_read_spec_logits_row(
                     session->graph, accepted - 1u, session->logits.get())) {
        row = session->logits.get();
        *frontier_read = true;
      } else {
        return 0;
      }
      drawn =
          sampler->verify != nullptr
              ? sampler->verify(sampler->ctx, accepted - 1u, row,
                                (uint32_t)vocabulary_size, drafts[accepted])
              : sampler->sample(sampler->ctx, row, (uint32_t)vocabulary_size);
    }
    if (drawn != drafts[accepted] ||
        (stop_at_eos && ds4_token_is_stop(session->engine, drawn))) {
      // EOS is a completed draw, but must not enter the reusable checkpoint.
      // Emit it as the next anchor without drawing again.
      session->dspark_state.pending_anchor = drawn;
      return accepted;
    }
    accepted++;
  }
  *frontier_read = false;
  return accepted;
}

static int session_dspark_finish_verified(
    ds4_session* session, int length, int vocabulary_size,
    bool concurrent_batch, const ds4_dspark_sampler* sampler,
    const float* row_logits, const int32_t* drafts, uint32_t drafted,
    const int32_t* row_tops, bool stop_at_eos, int* emitted, int* n_emitted,
    char* error, size_t error_capacity) {
  bool frontier_read = false;
  const uint32_t accepted = session_dspark_accept_prefix(
      session, sampler, row_logits, vocabulary_size, drafts, drafted, row_tops,
      stop_at_eos, &frontier_read);
  if (accepted == 0u) {
    set_error(error, error_capacity, "DSpark verified row read failed");
    session->checkpoint_valid = false;
    return 1;
  }
  const bool sampled = sampler != nullptr;
  uint32_t positional_accepted = 0;
  for (uint32_t i = 1; i < drafted; ++i) {
    if (row_tops[i - 1u] == drafts[i])
      positional_accepted++;
  }
  if (getenv("GUFO_DEEPSEEK_DSPARK_TRACE") != nullptr) {
    fprintf(stderr, "ds4: DSpark %s cycle drafted=%u accepted=%u draft=",
            sampled ? "sampled" : "greedy", drafted, accepted);
    for (uint32_t i = 0; i < drafted; ++i) {
      fprintf(stderr, "%s%d", i == 0 ? "" : ",", (int)drafts[i]);
    }
    fprintf(stderr, " tops=");
    for (uint32_t i = 0; i < drafted; ++i) {
      fprintf(stderr, "%s%d", i == 0 ? "" : ",", (int)row_tops[i]);
    }
    fputc('\n', stderr);
  }

  if (accepted == drafted) {
    session->checkpoint.len = length + (int)drafted;
    if (row_logits != nullptr) {
      memcpy(session->logits.get(),
             row_logits + (size_t)(drafted - 1u) * (size_t)vocabulary_size,
             (size_t)vocabulary_size * sizeof(float));
    } else if (!concurrent_batch &&
               !ds4_rocm_graph_read_spec_logits_row(session->graph, drafted - 1u,
                                                    session->logits.get())) {
      set_error(error, error_capacity, "DSpark frontier logits read failed");
      session->checkpoint_valid = false;
      return 1;
    }
    session->checkpoint_valid = true;
    if (ds4_rocm_graph_dspark_capture_ready(session->graph)) {
      (void)ds4_rocm_graph_dspark_inject(session->graph, (uint32_t)length,
                                         drafted);
    }
    for (uint32_t i = 0; i < drafted; ++i) {
      emitted[(*n_emitted)++] = drafts[i];
    }
    session->dspark_state.drafted += drafted;
    session->dspark_state.accepted += accepted;
    session->dspark_state.support_drafted += drafted - 1u;
    session->dspark_state.support_accepted += accepted - 1u;
    session->dspark_state.positional_accepted += positional_accepted;
    session->dspark_state.anchors += 1u;
    session->dspark_state.full_blocks += 1u;
    session->dspark_state.steps += 1;
    session_dspark_note_cycle(session, concurrent_batch);
    return 0;
  }

  if (row_logits != nullptr) {
    // The rejected draw came from this row; keep it as the frontier so an
    // exact target step or a restored prefix sees the same logits.
    memcpy(session->logits.get(),
           row_logits + (size_t)(accepted - 1u) * (size_t)vocabulary_size,
           (size_t)vocabulary_size * sizeof(float));
  }
  bool committed_prefix = false;
  if (accepted <= 4u) {
    committed_prefix =
        (concurrent_batch || frontier_read ||
         ds4_rocm_graph_read_spec_logits_row(session->graph, accepted - 1u,
                                             session->logits.get())) &&
        ds4_rocm_graph_spec_frontier_commit_prefix(session->graph, accepted);
    if (committed_prefix) {
      session->checkpoint.len = length + (int)accepted;
      if (ds4_rocm_graph_dspark_capture_ready(session->graph)) {
        committed_prefix = ds4_rocm_graph_dspark_inject(
            session->graph, (uint32_t)length, accepted);
      }
    }
  } else if (accepted == 5u) {
    committed_prefix =
        ds4_rocm_graph_spec_frontier_commit_prefix(session->graph, 4u);
    if (committed_prefix) {
      session->checkpoint.len = length + 4;
      if (ds4_rocm_graph_dspark_capture_ready(session->graph)) {
        committed_prefix =
            ds4_rocm_graph_dspark_inject(session->graph, (uint32_t)length, 4u);
      }
    }
    if (committed_prefix) {
      committed_prefix = session_commit_and_extend(session, drafts[4]);
    }
  }
  if (committed_prefix) {
    session->checkpoint_valid = true;
    for (uint32_t i = 0; i < accepted; ++i) {
      emitted[(*n_emitted)++] = drafts[i];
    }
    session->dspark_state.drafted += drafted;
    session->dspark_state.accepted += accepted;
    session->dspark_state.support_drafted += drafted - 1u;
    session->dspark_state.support_accepted += accepted - 1u;
    session->dspark_state.positional_accepted += positional_accepted;
    session->dspark_state.anchors += 1u;
    session->dspark_state.steps += 1u;
    session_dspark_note_cycle(session, concurrent_batch);
    return 0;
  }

  if (!ds4_rocm_graph_spec_frontier_restore(session->graph)) {
    set_error(error, error_capacity, "DSpark rollback failed");
    session->checkpoint_valid = false;
    return 1;
  }
  ds4_rocm_graph_dspark_truncate_context(session->graph, (uint32_t)length);
  uint32_t committed = 0;
  for (uint32_t i = 0; i < accepted; ++i) {
    // Greedy replay re-checks each row against the sequential argmax. A
    // sampled prefix was accepted by draws the request already consumed, so
    // it is committed as verified.
    const int sequential_top =
        sampled ? drafts[i]
                : ds4_sample_argmax(session->logits.get(),
                                    (uint32_t)vocabulary_size);
    if (sequential_top < 0) {
      set_error(error, error_capacity,
                "DSpark replay logits contain no finite values");
      session->checkpoint_valid = false;
      return 1;
    }
    if (sequential_top != drafts[i]) {
      if (getenv("GUFO_DEEPSEEK_DSPARK_TRACE") != nullptr) {
        fprintf(stderr,
                "ds4: DSpark replay shortened at row=%u "
                "verify_accept=%d sequential=%d\n",
                i, (int)drafts[i], sequential_top);
      }
      break;
    }
    if (!session_commit_and_extend(session, drafts[i])) {
      set_error(error, error_capacity, "DSpark accepted-prefix replay failed");
      session->checkpoint_valid = false;
      return 1;
    }
    emitted[(*n_emitted)++] = drafts[i];
    committed++;
  }
  session->dspark_state.drafted += drafted;
  session->dspark_state.accepted += committed;
  session->dspark_state.support_drafted += drafted - 1u;
  session->dspark_state.support_accepted +=
      committed > 0u ? committed - 1u : 0u;
  session->dspark_state.positional_accepted += positional_accepted;
  session->dspark_state.anchors += 1u;
  if (committed == drafted)
    session->dspark_state.full_blocks += 1u;
  session->dspark_state.steps += 1;
  session_dspark_note_cycle(session, concurrent_batch);
  return 0;
}

/*
 * One DSpark speculative cycle, greedy or sampled through `sampler`.
 *
 * Emits the target-known anchor plus its accepted support-model tail, or just
 * the target token when drafting is skipped. Accepted verifier prefixes become
 * authoritative cache state. Replay is only an internal safety fallback if a
 * prefix snapshot cannot be committed.
 */
static int session_dspark_step(ds4_session* session, int* emitted,
                               int emitted_cap, int* n_emitted, char* error,
                               size_t error_capacity, uint32_t max_draft_tokens,
                               const ds4_dspark_sampler* sampler,
                               bool stop_at_eos) {
  if (!session || !session->checkpoint_valid || !emitted || !n_emitted ||
      emitted_cap < 1 ||
      (sampler && (!sampler->sample || ((sampler->propose == nullptr) !=
                                        (sampler->verify == nullptr))))) {
    set_error(error, error_capacity, "invalid DSpark step request");
    return 1;
  }
  *n_emitted = 0;
  if (session->checkpoint.len >= session->context_size) {
    set_error(error, error_capacity, "DSpark decode exceeds session context");
    return 1;
  }
  // C1 controls did not repay probabilistic proposal/verification overhead.
  // Keep the qualified point-mass path (and AR seed identity) for this cohort.
  ds4_dspark_sampler scalar_sampler{};
  if (sampler && sampler->propose) {
    scalar_sampler = *sampler;
    scalar_sampler.propose = nullptr;
    scalar_sampler.verify = nullptr;
    sampler = &scalar_sampler;
  }
  session_dspark_use_concurrency(session, 1u);
  const uint32_t block = ds4_rocm_graph_dspark_block_size(session->graph);
  const uint32_t maximum =
      max_draft_tokens == 0u ? block : std::min(block, max_draft_tokens);
  const uint32_t requested_tail =
      session_dspark_usable_tail(session, maximum, emitted_cap);
  const int length = session->checkpoint.len;
  const int vocabulary_size = ds4_engine_vocab_size(session->engine);
  const int target_first =
      session_dspark_anchor(session, sampler, vocabulary_size);
  if (target_first < 0 || target_first >= vocabulary_size) {
    set_error(error, error_capacity,
              "DSpark target logits contain no finite values");
    session->checkpoint_valid = false;
    return 1;
  }
  if (stop_at_eos && ds4_token_is_stop(session->engine, target_first)) {
    emitted[(*n_emitted)++] = target_first;
    return 0;
  }
  const uint32_t required_context = (uint32_t)length;

  (void)session_dspark_seed_pending(session, required_context);

  /* Fall back to one ordinary token whenever a block cannot be drafted or
   * verified: no drafter, no captured features, or not enough context room. */
  bool can_draft =
      !session->dspark_state.plain_only && requested_tail != 0 && length >= 2 &&
      ds4_rocm_graph_dspark_context_len(session->graph) >= required_context;
  if (can_draft && session->dspark_state.skip_remaining != 0) {
    session->dspark_state.skip_remaining--;
    session->dspark_state.skipped++;
    can_draft = false;
  }
  if (!can_draft) {
    if (!session_commit_and_extend(session, target_first)) {
      set_error(error, error_capacity, "DeepSeek decode failed");
      session->checkpoint_valid = false;
      return 1;
    }
    emitted[(*n_emitted)++] = target_first;
    return 0;
  }

  int32_t drafts[DS4_DSPARK_MAX_BLOCK + 1u];
  uint32_t tail_drafted = 0;
  const bool proposed = ds4_rocm_graph_dspark_draft(
      session->graph, session->engine, target_first, (uint32_t)length,
      drafts + 1, &tail_drafted, sampler, requested_tail);
  uint32_t drafted = 0;
  if (proposed && tail_drafted <= DS4_DSPARK_MAX_BLOCK) {
    // Keep the qualified support computation and select only the requested
    // prefix for verification. The server's C1 draft budget used to be
    // ignored here.
    tail_drafted = std::min(tail_drafted, requested_tail);
    drafts[0] = target_first;
    drafted = tail_drafted + 1u;
  }
  const uint32_t drafted_cap = DS4_DSPARK_MAX_BLOCK + 1u;
  if (!proposed || drafted <= 1 || drafted > drafted_cap) {
    if (!session_commit_and_extend(session, target_first)) {
      set_error(error, error_capacity,
                "DeepSeek decode failed after draft failure");
      session->checkpoint_valid = false;
      return 1;
    }
    emitted[(*n_emitted)++] = target_first;
    return 0;
  }

  /* Row 0 is the target-known frontier token. The support model drafts only
   * the tail conditioned on that real token, so every verifier pass commits
   * at least the anchor. */

  int32_t row_tops[DS4_DSPARK_MAX_BLOCK + 1u];
  if (!ds4_rocm_graph_spec_prepare(session->graph, session->engine, drafted) ||
      !ds4_rocm_graph_spec_frontier_save(session->graph)) {
    set_error(error, error_capacity, "DSpark verification setup failed");
    return 1;
  }
  for (uint32_t i = 0; i < drafted; ++i) {
    ds4_tokens_push(&session->checkpoint, drafts[i]);
  }
  const bool verified = ds4_rocm_graph_verify_suffix(
      session->graph, session->engine, &session->checkpoint, (uint32_t)length,
      drafted, row_tops);
  session->checkpoint.len = length;
  if (!verified) {
    set_error(error, error_capacity, "DSpark verification failed");
    session->checkpoint_valid = false;
    return 1;
  }
  const uint64_t accepted_before = session->dspark_state.support_accepted;
  const int status = session_dspark_finish_verified(
      session, length, vocabulary_size, false, sampler, nullptr, drafts,
      drafted, row_tops, stop_at_eos, emitted, n_emitted, error,
      error_capacity);
  if (status == 0) {
    session_dspark_update_adaptive_width(
        session, tail_drafted,
        static_cast<uint32_t>(session->dspark_state.support_accepted -
                              accepted_before),
        maximum);
  }
  return status;
}

int ds4_sessions_dspark_step_batch(const ds4_session_dspark_batch_item* items,
                                   size_t item_count, char* error,
                                   size_t error_capacity) {
  if (!items || item_count == 0u || item_count > 8u) {
    set_error(error, error_capacity,
              "DSpark batch requires one to eight sessions");
    return 1;
  }
  if (item_count == 1u) {
    const auto& item = items[0];
    if (!item.session || !item.session->checkpoint_valid ||
        item.max_draft_tokens == 0u) {
      set_error(error, error_capacity, "DSpark batch contains an invalid session");
      return 1;
    }
    if (session_cancelled(item.session)) {
      set_error(error, error_capacity, "DSpark batch decode cancelled");
      return DS4_SESSION_SYNC_INTERRUPTED;
    }
    return session_dspark_step(
        item.session, item.emitted, item.emitted_cap, item.n_emitted, error,
        error_capacity, item.max_draft_tokens, item.sampler, item.stop_at_eos);
  }

  ds4_engine* engine = nullptr;
  uint32_t block = 0;
  int vocabulary_size = 0;
  std::array<int, 8> lengths{};
  std::array<int, 8> target_first{};
  std::array<bool, 8> can_draft_items{};
  std::array<uint32_t, 8> verify_tails{};
  std::array<uint32_t, 8> policy_tails{};
  bool any_can_draft = false;
  bool any_stopped = false;
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_session_dspark_batch_item& item = items[index];
    ds4_session* session = item.session;
    if (!session || !session->checkpoint_valid || !item.emitted ||
        !item.n_emitted || item.emitted_cap < 1 ||
        (item.sampler &&
         (!item.sampler->sample || ((item.sampler->propose == nullptr) !=
                                    (item.sampler->verify == nullptr))))) {
      set_error(error, error_capacity,
                "DSpark batch contains an invalid session");
      return 1;
    }
    *item.n_emitted = 0;
    if (session->checkpoint.len >= session->context_size) {
      set_error(error, error_capacity,
                "DSpark batch decode exceeds session context");
      return 1;
    }
    if (engine == nullptr) {
      engine = session->engine;
      block = ds4_rocm_graph_dspark_block_size(session->graph);
      vocabulary_size = ds4_engine_vocab_size(engine);
    } else if (session->engine != engine) {
      set_error(error, error_capacity,
                "DSpark batch sessions do not share one model");
      return 1;
    }
    if (session_cancelled(session)) {
      set_error(error, error_capacity, "DSpark batch decode cancelled");
      return DS4_SESSION_SYNC_INTERRUPTED;
    }
    for (size_t previous = 0; previous < index; ++previous) {
      if (items[previous].session == session) {
        set_error(error, error_capacity,
                  "DSpark batch contains a duplicate session");
        return 1;
      }
    }
  }

  // Validate the whole cohort before changing any request's draft policy or KV.
  for (size_t index = 0; index < item_count; ++index) {
    const auto& item = items[index];
    ds4_session* session = item.session;
    session_dspark_use_concurrency(session, item_count);
    const int length = session->checkpoint.len;
    lengths[index] = length;
    target_first[index] =
        session_dspark_anchor(session, item.sampler, vocabulary_size);
    if (target_first[index] < 0 || target_first[index] >= vocabulary_size) {
      set_error(error, error_capacity,
                "DSpark target logits contain no finite values");
      session->checkpoint_valid = false;
      return 1;
    }
    if (item.stop_at_eos &&
        ds4_token_is_stop(session->engine, target_first[index])) {
      any_stopped = true;
      continue;
    }
    (void)session_dspark_seed_pending(session, static_cast<uint32_t>(length));

    const uint32_t maximum =
        std::min({item.max_draft_tokens, block, DS4_DSPARK_MAX_BLOCK, 3u});
    policy_tails[index] =
        session_dspark_profitable_tail(session, maximum, item_count);
    verify_tails[index] = session_dspark_usable_tail(
        session, policy_tails[index], item.emitted_cap);
    bool can_draft =
        !session->dspark_state.plain_only &&
        session_dspark_usable_tail(session, maximum, item.emitted_cap) != 0u &&
        length >= 2 &&
        ds4_rocm_graph_dspark_context_len(session->graph) >=
            static_cast<uint32_t>(length);
    if (can_draft && (policy_tails[index] == 0u ||
                      session->dspark_state.skip_remaining != 0u)) {
      // Reconsider cost each token and keep aging an existing acceptance
      // backoff even while the caller's draft limit is unprofitable.
      if (session->dspark_state.skip_remaining != 0u)
        session->dspark_state.skip_remaining--;
      session->dspark_state.skipped++;
      can_draft = false;
    }
    can_draft_items[index] = can_draft;
    any_can_draft = any_can_draft || can_draft;
  }
  if (any_stopped || !any_can_draft) {
    // Keep completed requests at the frontier before EOS. Surviving requests
    // commit their already selected anchors without advancing their RNG again.
    std::array<ds4_session_dspark_batch_item, 8> active_items{};
    std::array<int, 8> active_tokens{};
    size_t active_count = 0;
    for (size_t index = 0; index < item_count; ++index) {
      if (items[index].stop_at_eos &&
          ds4_token_is_stop(engine, target_first[index])) {
        continue;
      }
      active_items[active_count] = items[index];
      active_tokens[active_count++] = target_first[index];
    }
    const bool committed =
        active_count == 0u ||
        (active_count == 1u
             ? session_commit_and_extend(active_items[0].session,
                                         active_tokens[0])
             : sessions_commit_and_extend_batch(active_items.data(),
                                                active_tokens, active_count));
    if (!committed) {
      for (size_t index = 0; index < active_count; ++index) {
        active_items[index].session->checkpoint_valid = false;
      }
      set_error(error, error_capacity,
                "DeepSeek DSpark fallback batch decode failed");
      return 1;
    }
    for (size_t index = 0; index < item_count; ++index) {
      items[index].emitted[0] = target_first[index];
      *items[index].n_emitted = 1;
    }
    return 0;
  }

  constexpr uint32_t kDraftCapacity = DS4_DSPARK_MAX_BLOCK + 1u;
  std::array<std::array<int32_t, kDraftCapacity>, 8> drafts{};
  std::array<std::array<int32_t, kDraftCapacity>, 8> row_tops{};
  std::array<uint32_t, 8> drafted_counts{};
  std::array<uint32_t, 8> verify_counts{};
  std::array<uint32_t, 8> tail_drafted{};
  const bool trace_timing =
      getenv("GUFO_DEEPSEEK_DSPARK_TIMING") != nullptr;
  const double draft_t0 = trace_timing ? ds4_now_seconds() : 0.0;
  std::array<ds4_rocm_dspark_draft_item, 8> draft_items{};
  size_t draft_item_count = 0;
  size_t single_draft_index = 0;
  for (size_t index = 0; index < item_count; ++index) {
    if (can_draft_items[index]) {
      single_draft_index = index;
      draft_items[draft_item_count++] = {
          .graph = items[index].session->graph,
          .target_next_token = target_first[index],
          .position = static_cast<uint32_t>(lengths[index]),
          .max_draft_tokens = verify_tails[index],
          .tokens = drafts[index].data() + 1,
          .n_tokens = &tail_drafted[index],
          .sampler = items[index].sampler,
          .concurrency = item_count,
      };
    }
  }
  bool proposed_all = false;
  if (draft_item_count >= 2) {
    proposed_all = ds4_rocm_graph_dspark_draft_head_batch(
        engine, draft_items.data(), draft_item_count);
  } else {
    const size_t index = single_draft_index;
    proposed_all = ds4_rocm_graph_dspark_draft(
        items[index].session->graph, engine, target_first[index],
        static_cast<uint32_t>(lengths[index]), drafts[index].data() + 1,
        &tail_drafted[index], items[index].sampler, verify_tails[index],
        item_count);
  }
  const double draft_ms =
      trace_timing ? (ds4_now_seconds() - draft_t0) * 1000.0 : 0.0;

  for (size_t index = 0; index < item_count; ++index) {
    drafts[index][0] = target_first[index];
    if (proposed_all && tail_drafted[index] <= DS4_DSPARK_MAX_BLOCK) {
      const uint32_t verify_tail =
          std::min(tail_drafted[index], verify_tails[index]);
      drafted_counts[index] = can_draft_items[index] ? verify_tail + 1u : 1u;
    }
    proposed_all = proposed_all && drafted_counts[index] != 0u &&
                   drafted_counts[index] <= kDraftCapacity;
  }

  if (proposed_all) {
    // Confidence describes this block, so retry after one budget. Historical
    // rejection retains its longer backoff for persistently poor acceptance.
    for (size_t index = 0; index < item_count; ++index)
      if (can_draft_items[index] && items[index].sampler &&
          items[index].sampler->propose && tail_drafted[index] == 0)
        items[index].session->dspark_state.skip_remaining = verify_tails[index];
    if (std::all_of(drafted_counts.begin(), drafted_counts.begin() + item_count,
                    [](uint32_t count) { return count == 1; }))
      proposed_all = false;
  }
  if (!proposed_all) {
    if (!sessions_commit_and_extend_batch(items, target_first, item_count)) {
      for (size_t index = 0; index < item_count; ++index) {
        items[index].session->checkpoint_valid = false;
      }
      set_error(error, error_capacity,
                "DeepSeek DSpark draft fallback batch decode failed");
      return 1;
    }
    for (size_t index = 0; index < item_count; ++index) {
      items[index].emitted[0] = target_first[index];
      *items[index].n_emitted = 1;
    }
    return 0;
  }

  verify_counts = drafted_counts;
  std::array<ds4_rocm_verify_item, 8> verify_items{};
  std::array<float*, 8> row_logits{};
  for (size_t index = 0; index < item_count; ++index) {
    ds4_session* session = items[index].session;
    for (uint32_t row = 0; row < verify_counts[index]; ++row) {
      ds4_tokens_push(&session->checkpoint, drafts[index][row]);
    }
    if (items[index].sampler != nullptr) {
      row_logits[index] =
          session_spec_rows(session, verify_counts[index], vocabulary_size);
      if (row_logits[index] == nullptr) {
        for (size_t reset = 0; reset < item_count; ++reset) {
          items[reset].session->checkpoint.len = lengths[reset];
        }
        set_error(error, error_capacity,
                  "DSpark sampled verification rows allocation failed");
        return 1;
      }
    }
    verify_items[index] = {
        .graph = session->graph,
        .tokens = &session->checkpoint,
        .start = static_cast<uint32_t>(lengths[index]),
        .n_tokens = verify_counts[index],
        .logical_n_tokens = drafted_counts[index],
        .row_tops = row_tops[index].data(),
        .frontier_logits = session->logits.get(),
        .row_logits = row_logits[index],
    };
  }

  const double verify_t0 = trace_timing ? ds4_now_seconds() : 0.0;
  const bool verified =
      ds4_rocm_graph_verify_batch(engine, verify_items.data(), item_count);
  const double verify_ms =
      trace_timing ? (ds4_now_seconds() - verify_t0) * 1000.0 : 0.0;
  for (size_t index = 0; index < item_count; ++index) {
    items[index].session->checkpoint.len = lengths[index];
  }
  if (!verified) {
    for (size_t index = 0; index < item_count; ++index) {
      items[index].session->checkpoint_valid = false;
    }
    set_error(error, error_capacity, "DSpark batch verification failed");
    return 1;
  }

  const double commit_t0 = trace_timing ? ds4_now_seconds() : 0.0;
  uint32_t total_rows = 0u;
  uint32_t total_emitted = 0u;
  for (size_t index = 0; index < item_count; ++index) {
    total_rows += verify_counts[index];
    const int result = session_dspark_finish_verified(
        items[index].session, lengths[index], vocabulary_size, true,
        items[index].sampler, row_logits[index], drafts[index].data(),
        drafted_counts[index], row_tops[index].data(), items[index].stop_at_eos,
        items[index].emitted, items[index].n_emitted, error, error_capacity);
    if (result != 0) {
      for (size_t remaining = index + 1u; remaining < item_count; ++remaining) {
        items[remaining].session->checkpoint_valid = false;
      }
      return result;
    }
    total_emitted +=
        static_cast<uint32_t>(*items[index].n_emitted);
    if (can_draft_items[index]) {
      const uint32_t maximum = std::min(
          {items[index].max_draft_tokens, block, DS4_DSPARK_MAX_BLOCK});
      const uint32_t policy_maximum = std::min(maximum, policy_tails[index]);
      const uint32_t support_drafted =
          drafted_counts[index] > 0u ? drafted_counts[index] - 1u : 0u;
      const uint32_t support_accepted =
          *items[index].n_emitted > 0
              ? static_cast<uint32_t>(*items[index].n_emitted - 1)
              : 0u;
      session_dspark_note_concurrent_cycle(
          items[index].session, support_drafted, support_accepted, item_count);
      session_dspark_update_adaptive_width(items[index].session,
                                           support_drafted, support_accepted,
                                           policy_maximum);
    }
  }
  if (trace_timing) {
    const double commit_ms =
        (ds4_now_seconds() - commit_t0) * 1000.0;
    fprintf(stderr,
            "ds4: DSpark batch timing c=%zu rows=%u emitted=%u "
            "draft=%.3f ms verify=%.3f ms commit=%.3f ms\n",
            item_count, total_rows, total_emitted, draft_ms, verify_ms,
            commit_ms);
  }
  return 0;
}

void ds4_session_dspark_stats(const ds4_session *session,
                              uint64_t *drafted,
                              uint64_t *accepted,
                              uint64_t *support_drafted,
                              uint64_t *support_accepted,
                              uint64_t *positional_accepted,
                              uint64_t *anchors,
                              uint64_t *full_blocks,
                              uint64_t *steps,
                              uint64_t *skipped,
                              uint32_t *context_tokens) {
    if (!session) return;
    if (drafted)
      *drafted = session->dspark_state.drafted;
    if (accepted)
      *accepted = session->dspark_state.accepted;
    if (support_drafted)
      *support_drafted = session->dspark_state.support_drafted;
    if (support_accepted)
      *support_accepted = session->dspark_state.support_accepted;
    if (positional_accepted) {
      *positional_accepted = session->dspark_state.positional_accepted;
    }
    if (anchors)
      *anchors = session->dspark_state.anchors;
    if (full_blocks)
      *full_blocks = session->dspark_state.full_blocks;
    if (steps)
      *steps = session->dspark_state.steps;
    if (skipped)
      *skipped = session->dspark_state.skipped;
    if (context_tokens) {
        *context_tokens =
            ds4_rocm_graph_dspark_context_len(session->graph);
    }
}

int ds4_session_pos(const ds4_session *session) {
    return session ? session->checkpoint.len : 0;
}

int ds4_session_ctx(const ds4_session *session) {
    return session ? session->context_size : 0;
}

uint32_t ds4_session_prefill_capacity(const ds4_session* session) {
  return session ? session->prefill_capacity : 0;
}
