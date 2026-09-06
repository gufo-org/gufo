#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <memory>
#include <new>

#include "dspark_internal.h"
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
    uint64_t dspark_drafted;
    uint64_t dspark_accepted;
    uint64_t dspark_support_drafted;
    uint64_t dspark_support_accepted;
    uint64_t dspark_positional_accepted;
    uint64_t dspark_anchors;
    uint64_t dspark_full_blocks;
    uint64_t dspark_steps;
    uint64_t dspark_skipped;
    uint32_t dspark_skip_remaining;
    bool dspark_plain_only;
    bool dspark_force_plain_request;
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
        preserve_force_plain && session->dspark_force_plain_request;
    session->dspark_drafted = 0;
    session->dspark_accepted = 0;
    session->dspark_support_drafted = 0;
    session->dspark_support_accepted = 0;
    session->dspark_positional_accepted = 0;
    session->dspark_anchors = 0;
    session->dspark_full_blocks = 0;
    session->dspark_steps = 0;
    session->dspark_skipped = 0;
    session->dspark_skip_remaining = 0;
    session->dspark_force_plain_request = force_plain;
    session->dspark_plain_only = force_plain;
    ds4_rocm_graph_dspark_set_capture_enabled(session->graph, !force_plain);
}

int ds4_session_create(ds4_session **out,
                       ds4_engine *engine,
                       int context_size) {
    if (!out || !engine || context_size <= 0) return 1;

    auto session = std::unique_ptr<ds4_session>(
        new (std::nothrow) ds4_session{});
    if (!session) return 1;

    session->engine = engine;
    session->context_size = context_size;
    session->prefill_capacity =
        ds4_rocm_graph_prefill_capacity(context_size);
    session->graph = ds4_rocm_graph_create(engine,
                                           context_size,
                                           session->prefill_capacity);
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

    /* Attach before prefill so target hidden states are captured. Converting
     * those captures into support KV is deferred until the first draft, keeping
     * ordinary prompt processing unchanged. */
    if (engine->dspark != nullptr) {
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

void ds4_session_prepare_batch_execution(ds4_session *session) {
    if (!session || ds4_rocm_graph_dspark_block_size(session->graph) == 0u) {
        return;
    }
    session->dspark_force_plain_request = true;
    session->dspark_plain_only = true;
    ds4_rocm_graph_dspark_set_capture_enabled(session->graph, false);
}

/*
 * Evaluate one target token and extend the support ring with the sampled target
 * features captured by that decode. If support coverage has a gap (for example
 * after restoring a target-only snapshot), injection stores the row but the
 * graph deliberately leaves its contiguous context length unchanged.
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

int ds4_session_sync(ds4_session *session,
                     const ds4_tokens *prompt,
                     char *error,
                     size_t error_capacity) {
    if (!session || !prompt || prompt->len <= 0 ||
        prompt->len >= session->context_size) {
        set_error(error, error_capacity, "prompt exceeds context");
        return 1;
    }
    if (session_cancelled(session)) {
        set_error(error, error_capacity, "prefill cancelled");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }

    if (session->checkpoint_valid &&
        prompt->len >= session->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &session->checkpoint)) {
        const int suffix = prompt->len - session->checkpoint.len;
        if (suffix > 0) {
            /* Preserve a lazily captured prompt before the resumed prefill
             * reuses the batch capture storage for its suffix. A target-only
             * snapshot has no capture and simply remains unable to draft. */
            (void)session_dspark_seed_pending(
                session, (uint32_t)session->checkpoint.len);
        }
        const uint32_t resume_minimum =
            ds4_rocm_graph_resume_prefill_min_tokens();
        if (suffix > 0 && (uint32_t)suffix >= resume_minimum) {
            const bool ok = ds4_rocm_graph_prefill_range(
                session->graph,
                session->engine,
                prompt,
                (uint32_t)session->checkpoint.len,
                (uint32_t)suffix,
                session->logits.get());
            if (!ok) {
                set_error(error,
                          error_capacity,
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
                set_error(error,
                          error_capacity,
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
    if (!session || !session->logits) return -1;
    return ds4_sample_argmax(session->logits.get(),
                             (uint32_t)ds4_engine_vocab_size(session->engine));
}

int ds4_session_argmax_excluding(const ds4_session *session,
                                 int excluded_token) {
    if (!session || !session->logits) return -1;

    const int vocabulary_size = ds4_engine_vocab_size(session->engine);
    int best = -1;
    float best_logit = DS4_SESSION_NEG_INF;
    for (int i = 0; i < vocabulary_size; ++i) {
        if (i == excluded_token) continue;
        const float value = session->logits[i];
        if (best < 0 || value > best_logit) {
            best = i;
            best_logit = value;
        }
    }
    return best;
}

int ds4_session_sample(const ds4_session *session,
                       float temperature,
                       int top_k,
                       float top_p,
                       float min_p,
                       uint64_t *rng_state) {
    if (!session || !session->logits || !rng_state) return -1;
    return ds4_sample_top_p_min_p(
        session->logits.get(),
        (uint32_t)ds4_engine_vocab_size(session->engine),
        temperature,
        top_k,
        top_p,
        min_p,
        rng_state);
}

int ds4_session_copy_logits(const ds4_session *session,
                            float *out,
                            int capacity) {
    if (!session || !out) return 0;
    const int vocabulary_size = ds4_engine_vocab_size(session->engine);
    if (capacity < vocabulary_size) return 0;
    memcpy(out,
           session->logits.get(),
           (size_t)vocabulary_size * sizeof(out[0]));
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
            set_error(error,
                      error_capacity,
                      "batch decode contains an invalid token");
            return 1;
        }
        if (session->checkpoint.len < 0 ||
            session->checkpoint.len >= session->context_size) {
            set_error(error,
                      error_capacity,
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
        items[index].session->dspark_plain_only = true;
        items[index].session->dspark_force_plain_request = true;
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
    return ds4_rocm_graph_save_snapshot(session->graph,
                                        &session->checkpoint,
                                        session->logits.get(),
                                        session->prefill_capacity,
                                        session->context_size,
                                        snapshot,
                                        error,
                                        error_capacity);
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
    const int result =
        ds4_rocm_graph_load_snapshot(session->graph,
                                     &session->checkpoint,
                                     session->logits.get(),
                                     session->prefill_capacity,
                                     session->context_size,
                                     snapshot,
                                     error,
                                     error_capacity);
    session->checkpoint_valid = result == 0;
    return result;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snapshot) {
    if (!snapshot) return;
    free(snapshot->ptr);
    memset(snapshot, 0, sizeof(*snapshot));
}

/*
 * DSpark verifier self-check.
 *
 * Speculative decoding is only lossless if the batched verification pass agrees
 * with ordinary one-token decode about the target's greedy continuation. The
 * two paths run different kernels with different reduction orders, so agreement
 * is an empirical property of this checkpoint, not something the design can
 * assume. This routine measures it directly: decode a suffix autoregressively,
 * roll back, verify the same suffix as one block, and compare.
 *
 * It also times both paths, which is what decides whether a drafter can pay for
 * itself: verification has to cost far less than the tokens it replaces.
 */
int ds4_session_dspark_selftest(ds4_session *session,
                                int rows,
                                char *error,
                                size_t error_capacity) {
    if (!session || !session->checkpoint_valid || rows < 2) {
        set_error(error, error_capacity, "invalid DSpark self-test request");
        return 1;
    }
    if (rows > (int)DS4_SPEC_MAX_ROWS) {
        set_error(error, error_capacity, "DSpark self-test row count exceeds the verifier cap");
        return 1;
    }
    const int start = session->checkpoint.len;
    if (start + rows >= session->context_size) {
        set_error(error, error_capacity, "DSpark self-test would exceed the context");
        return 1;
    }
    if (!ds4_rocm_graph_spec_prepare(session->graph, session->engine, (uint32_t)rows)) {
        set_error(error, error_capacity, "DSpark verification state allocation failed");
        return 1;
    }

    const int vocabulary_size = ds4_engine_vocab_size(session->engine);
    auto base_logits = std::unique_ptr<float[]>(
        new (std::nothrow) float[vocabulary_size]);
    auto block = std::unique_ptr<int[]>(new (std::nothrow) int[rows + 1]);
    auto row_tops = std::unique_ptr<int32_t[]>(new (std::nothrow) int32_t[rows]);
    if (!base_logits || !block || !row_tops) {
        set_error(error, error_capacity, "DSpark self-test allocation failed");
        return 1;
    }
    memcpy(base_logits.get(),
           session->logits.get(),
           (size_t)vocabulary_size * sizeof(float));

    if (!ds4_rocm_graph_spec_frontier_save(session->graph)) {
        set_error(error, error_capacity, "DSpark frontier save failed");
        return 1;
    }

    /* Autoregressive reference: block[i] is what greedy decode emits at
     * position start + i, so block[i + 1] is the target's continuation after
     * row i and is exactly what the verifier's row i must reproduce. */
    const double ar_t0 = ds4_now_seconds();
    for (int i = 0; i < rows; ++i) {
        block[i] = ds4_sample_argmax(session->logits.get(), (uint32_t)vocabulary_size);
        if (!ds4_rocm_graph_eval(session->graph,
                                 session->engine,
                                 block[i],
                                 (uint32_t)(start + i),
                                 session->logits.get())) {
            set_error(error, error_capacity, "DSpark self-test decode failed");
            session->checkpoint_valid = false;
            return 1;
        }
    }
    const double ar_seconds = ds4_now_seconds() - ar_t0;
    block[rows] = ds4_sample_argmax(session->logits.get(), (uint32_t)vocabulary_size);

    /* Roll back to the pre-suffix frontier and replay the same suffix as one
     * verification block. */
    session->checkpoint.len = start;
    if (!ds4_rocm_graph_spec_frontier_restore(session->graph)) {
        set_error(error, error_capacity, "DSpark frontier restore failed");
        session->checkpoint_valid = false;
        return 1;
    }
    for (int i = 0; i < rows; ++i) {
        ds4_tokens_push(&session->checkpoint, block[i]);
    }

    /* Batched kernels and hipBLASLt plans are selected per row count, so the
     * first verification block at a new width pays setup that steady-state
     * decoding never sees again. Report every repetition rather than one
     * number that silently mixes the two. */
    const int repetitions = 4;
    double verify_seconds = 0.0;
    double verify_best = 0.0;
    for (int rep = 0; rep < repetitions; ++rep) {
        if (rep > 0) {
            session->checkpoint.len = start;
            if (!ds4_rocm_graph_spec_frontier_restore(session->graph)) {
                set_error(error, error_capacity, "DSpark frontier restore between reps failed");
                session->checkpoint_valid = false;
                return 1;
            }
            for (int i = 0; i < rows; ++i) {
                ds4_tokens_push(&session->checkpoint, block[i]);
            }
        }
        const double verify_t0 = ds4_now_seconds();
        const bool verified = ds4_rocm_graph_verify_suffix(session->graph,
                                                          session->engine,
                                                          &session->checkpoint,
                                                          (uint32_t)start,
                                                          (uint32_t)rows,
                                                          row_tops.get());
        verify_seconds = ds4_now_seconds() - verify_t0;
        if (!verified) {
            set_error(error, error_capacity, "DSpark verification pass failed");
            session->checkpoint_valid = false;
            return 1;
        }
        if (rep == 0 || verify_seconds < verify_best) verify_best = verify_seconds;
        fprintf(stderr,
                "ds4: DSpark selftest rows=%d rep=%d verify=%.2f ms\n",
                rows,
                rep,
                verify_seconds * 1000.0);
    }
    verify_seconds = verify_best;

    int agree = 0;
    for (int i = 0; i < rows; ++i) {
        if (row_tops[i] == block[i + 1]) agree++;
    }
    fprintf(stderr,
            "ds4: DSpark selftest rows=%d start=%d agree=%d/%d "
            "autoregressive=%.2f ms (%.2f ms/token) verify_best=%.2f ms "
            "verify_per_row=%.2f ms cost_ratio=%.2fx\n",
            rows,
            start,
            agree,
            rows,
            ar_seconds * 1000.0,
            ar_seconds * 1000.0 / (double)rows,
            verify_seconds * 1000.0,
            verify_seconds * 1000.0 / (double)rows,
            ar_seconds > 0.0 ? verify_seconds / (ar_seconds / (double)rows) : 0.0);
    for (int i = 0; i < rows; ++i) {
        if (row_tops[i] == block[i + 1]) continue;
        fprintf(stderr,
                "ds4: DSpark selftest row %d disagrees: verify=%d autoregressive=%d\n",
                i,
                (int)row_tops[i],
                block[i + 1]);
    }

    /* Restore the frontier once more and confirm rollback leaves the session
     * able to reproduce the original continuation. */
    session->checkpoint.len = start;
    if (!ds4_rocm_graph_spec_frontier_restore(session->graph)) {
        set_error(error, error_capacity, "DSpark frontier restore after verify failed");
        session->checkpoint_valid = false;
        return 1;
    }
    memcpy(session->logits.get(),
           base_logits.get(),
           (size_t)vocabulary_size * sizeof(float));
    if (!ds4_rocm_graph_eval(session->graph,
                             session->engine,
                             block[0],
                             (uint32_t)start,
                             session->logits.get())) {
        set_error(error, error_capacity, "DSpark rollback replay failed");
        session->checkpoint_valid = false;
        return 1;
    }
    const int replayed = ds4_sample_argmax(session->logits.get(),
                                           (uint32_t)vocabulary_size);
    fprintf(stderr,
            "ds4: DSpark selftest rollback replay next=%d expected=%d %s\n",
            replayed,
            block[1],
            replayed == block[1] ? "ok" : "MISMATCH");

    session->checkpoint.len = start;
    ds4_tokens_push(&session->checkpoint, block[0]);
    session->checkpoint_valid = true;
    return agree == rows && replayed == block[1] ? 0 : 1;
}

/*
 * DSpark drafting self-check.
 *
 * Runs whole speculative cycles: seed the drafter's rings from the prompt's
 * captured target features, propose a block, verify it against the target, commit
 * the accepted prefix, and repeat. Reports acceptance and timing per cycle, which
 * is the measurement that says whether the drafter earns its verification block.
 */
int ds4_session_dspark_draft_selftest(ds4_session *session,
                                     int cycles,
                                     char *error,
                                     size_t error_capacity) {
    if (!session || !session->checkpoint_valid || cycles < 1) {
        set_error(error, error_capacity, "invalid DSpark draft self-test request");
        return 1;
    }
    if (!ds4_engine_has_dspark(session->engine)) {
        set_error(error, error_capacity, "no DSpark support model is attached");
        return 1;
    }
    const uint32_t block = ds4_rocm_graph_dspark_block_size(session->graph);
    if (block == 0) {
        set_error(error, error_capacity, "DSpark drafting is not attached to this session");
        return 1;
    }

    if (!session_dspark_seed_pending(
            session, (uint32_t)session->checkpoint.len)) {
        set_error(error,
                  error_capacity,
                  "prompt prefill did not capture DSpark support context");
        return 1;
    }
    const uint32_t context_len =
        ds4_rocm_graph_dspark_context_len(session->graph);
    if (context_len < (uint32_t)session->checkpoint.len) {
        set_error(error,
                  error_capacity,
                  "prompt prefill did not seed the DSpark support cache");
        return 1;
    }
    fprintf(stderr,
            "ds4: DSpark draft selftest prompt_context=%u\n",
            context_len);

    const int vocabulary_size = ds4_engine_vocab_size(session->engine);
    auto drafts = std::unique_ptr<int32_t[]>(new (std::nothrow) int32_t[block]);
    auto row_tops = std::unique_ptr<int32_t[]>(new (std::nothrow) int32_t[block]);
    if (!drafts || !row_tops) {
        set_error(error, error_capacity, "DSpark draft self-test allocation failed");
        return 1;
    }

    std::size_t total_drafted = 0;
    std::size_t total_accepted = 0;
    /* Per-position agreement, counted independently of the accepted prefix.
     * Acceptance is prefix-based, so one early miss discards correct later
     * tokens; this separates drafter quality from prefix fragility. */
    std::size_t total_positional = 0;
    std::size_t total_positional_rows = 0;
    double total_draft_ms = 0.0;
    double total_verify_ms = 0.0;

    for (int cycle = 0; cycle < cycles; ++cycle) {
        const int length = session->checkpoint.len;
        if (length < 1 || length + (int)block + 1 >= session->context_size) break;
        const int last_token = session->checkpoint.v[length - 1];

        /* The stage input leads with an encoder row holding the fused feature of
         * the last committed position, so the rows occupy
         * [length - 1, length - 1 + block_size], and the drafted rows land
         * exactly on the positions being predicted. */
        constexpr int pos_shift = -1;
        if (length + pos_shift < 1) break;
        const uint32_t pos0 = (uint32_t)(length + pos_shift);
        uint32_t drafted = 0;
        const double draft_t0 = ds4_now_seconds();
        const bool proposed = ds4_rocm_graph_dspark_draft(
            session->graph, session->engine, last_token, pos0, drafts.get(),
            &drafted, false);
        const double draft_ms = (ds4_now_seconds() - draft_t0) * 1000.0;
        if (!proposed || drafted == 0) {
            set_error(error, error_capacity, "DSpark draft proposal failed");
            return 1;
        }
        total_draft_ms += draft_ms;
        total_drafted += drafted;

        /* The target's own next token decides whether the first draft survives. */
        const int target_first =
            ds4_sample_argmax(session->logits.get(), (uint32_t)vocabulary_size);
        uint32_t accepted = 0;
        double verify_ms = 0.0;
        if (target_first == drafts[0]) {
            if (!ds4_rocm_graph_spec_prepare(session->graph, session->engine, drafted) ||
                !ds4_rocm_graph_spec_frontier_save(session->graph)) {
                set_error(error, error_capacity, "DSpark verification setup failed");
                return 1;
            }
            for (uint32_t i = 0; i < drafted; ++i) {
                ds4_tokens_push(&session->checkpoint, drafts[i]);
            }
            const double verify_t0 = ds4_now_seconds();
            const bool verified = ds4_rocm_graph_verify_suffix(session->graph,
                                                              session->engine,
                                                              &session->checkpoint,
                                                              (uint32_t)length,
                                                              drafted,
                                                              row_tops.get());
            verify_ms = (ds4_now_seconds() - verify_t0) * 1000.0;
            if (!verified) {
                set_error(error, error_capacity, "DSpark verification failed");
                session->checkpoint_valid = false;
                return 1;
            }
            accepted = 1;
            while (accepted < drafted && row_tops[accepted - 1] == drafts[accepted]) {
                accepted++;
            }
            total_positional += 1;
            total_positional_rows += drafted;
            for (uint32_t i = 1; i < drafted; ++i) {
                if (row_tops[i - 1] == drafts[i]) total_positional++;
            }
            /* Commit only the accepted prefix: rewind, restore the frontier, and
             * replay it so the committed state matches what was accepted. */
            session->checkpoint.len = length;
            if (!ds4_rocm_graph_spec_frontier_restore(session->graph)) {
                set_error(error, error_capacity, "DSpark rollback failed");
                session->checkpoint_valid = false;
                return 1;
            }
            ds4_rocm_graph_dspark_truncate_context(session->graph, (uint32_t)length);
            for (uint32_t i = 0; i < accepted; ++i) {
                if (!session_commit_and_extend(session, drafts[i])) {
                    set_error(error, error_capacity, "DSpark accepted-prefix replay failed");
                    session->checkpoint_valid = false;
                    return 1;
                }
            }
        } else {
            total_positional_rows += 1;
            /* First draft missed: commit the target's own token. */
            if (!session_commit_and_extend(session, target_first)) {
                set_error(error, error_capacity, "DSpark miss replay failed");
                session->checkpoint_valid = false;
                return 1;
            }
        }
        total_accepted += accepted;
        total_verify_ms += verify_ms;

        if (getenv("GUFO_DEEPSEEK_DSPARK_TRACE") != nullptr) {
            /* The four MASK positions start from identical embeddings and are
             * separated only by rope and by attending to each other. Printing the
             * proposal next to the target's own continuation shows immediately
             * whether they differentiate at all. */
            fprintf(stderr, "ds4:   draft =");
            for (uint32_t i = 0; i < drafted; ++i) {
                fprintf(stderr, " %d", (int)drafts[i]);
            }
            fprintf(stderr, "\nds4:   target= %d", target_first);
            if (target_first == drafts[0]) {
                for (uint32_t i = 0; i + 1 < drafted; ++i) {
                    fprintf(stderr, " %d", (int)row_tops[i]);
                }
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr,
                "ds4: DSpark draft cycle=%d drafted=%u accepted=%u draft=%.2f ms "
                "verify=%.2f ms first=%s\n",
                cycle,
                drafted,
                accepted,
                draft_ms,
                verify_ms,
                target_first == drafts[0] ? "hit" : "miss");
    }

    const double emitted = (double)total_accepted + (double)cycles;
    fprintf(stderr,
            "ds4: DSpark draft selftest cycles=%d drafted=%zu accepted=%zu "
            "acceptance=%.1f%% positional=%.1f%% (%zu/%zu) "
            "draft_avg=%.2f ms verify_avg=%.2f ms\n",
            cycles,
            total_drafted,
            total_accepted,
            total_drafted ? 100.0 * (double)total_accepted / (double)total_drafted : 0.0,
            total_positional_rows
                ? 100.0 * (double)total_positional / (double)total_positional_rows
                : 0.0,
            total_positional,
            total_positional_rows,
            total_draft_ms / (double)cycles,
            total_verify_ms / (double)cycles);
    (void)emitted;
    session->checkpoint_valid = true;
    return 0;
}

/*
 * Feed one attempted cycle's outcome to the break-even controller.
 *
 * Serial DSpark keeps the established four-probe/64-token retry policy.
 * Concurrent C2 stops after one clearly poor cycle because another support
 * probe displaces a profitable W2 target step. A new request resets the
 * controller and can use DSpark again. The 48% threshold is the measured
 * seed-plus-five break-even point on gfx1151.
 */
static void session_dspark_note_cycle(ds4_session* session,
                                      bool concurrent_batch) {
  const uint64_t early_cycles = concurrent_batch ? 1u : 4u;
  constexpr double kEarlyRejectRate = 0.25;
  constexpr uint64_t kWarmupCycles = 8u;
  constexpr double kBreakEvenAcceptance = 0.48;
  constexpr uint32_t kSerialLowAcceptanceBackoff = 64u;
  if (getenv("GUFO_DEEPSEEK_DSPARK_DISABLE_BACKOFF") != nullptr) {
    return;
  }
  if (session->dspark_support_drafted == 0u ||
      session->dspark_steps < early_cycles) {
    return;
  }
  const double acceptance = (double)session->dspark_support_accepted /
                            (double)session->dspark_support_drafted;
  if (session->dspark_steps < kWarmupCycles && acceptance >= kEarlyRejectRate) {
    return;
  }
  if (acceptance >= kBreakEvenAcceptance) {
    session->dspark_skip_remaining = 0u;
    return;
  }
  if (concurrent_batch) {
    session->dspark_skip_remaining = 0u;
    session->dspark_plain_only = true;
    ds4_rocm_graph_dspark_set_capture_enabled(session->graph, false);
  } else {
    session->dspark_skip_remaining = kSerialLowAcceptanceBackoff;
  }
}

static int session_dspark_finish_verified(
    ds4_session* session, int length, int vocabulary_size,
    bool concurrent_batch, const int32_t* drafts, uint32_t drafted,
    const int32_t* row_tops, int* emitted, int* n_emitted, char* error,
    size_t error_capacity) {
  uint32_t accepted = 1;
  while (accepted < drafted && row_tops[accepted - 1] == drafts[accepted]) {
    accepted++;
  }
  uint32_t positional_accepted = 0;
  for (uint32_t i = 1; i < drafted; ++i) {
    if (row_tops[i - 1u] == drafts[i])
      positional_accepted++;
  }
  if (getenv("GUFO_DEEPSEEK_DSPARK_TRACE") != nullptr) {
    fprintf(stderr, "ds4: DSpark cycle drafted=%u accepted=%u draft=", drafted,
            accepted);
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
    if (!ds4_rocm_graph_read_spec_logits_row(session->graph, drafted - 1u,
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
    session->dspark_drafted += drafted;
    session->dspark_accepted += accepted;
    session->dspark_support_drafted += drafted - 1u;
    session->dspark_support_accepted += accepted - 1u;
    session->dspark_positional_accepted += positional_accepted;
    session->dspark_anchors += 1u;
    session->dspark_full_blocks += 1u;
    session->dspark_steps += 1;
    session_dspark_note_cycle(session, concurrent_batch);
    return 0;
  }

  bool committed_prefix = false;
  if (accepted <= 4u) {
    committed_prefix =
        ds4_rocm_graph_read_spec_logits_row(session->graph, accepted - 1u,
                                            session->logits.get()) &&
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
    session->dspark_drafted += drafted;
    session->dspark_accepted += accepted;
    session->dspark_support_drafted += drafted - 1u;
    session->dspark_support_accepted += accepted - 1u;
    session->dspark_positional_accepted += positional_accepted;
    session->dspark_anchors += 1u;
    session->dspark_steps += 1u;
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
    const int sequential_top =
        ds4_sample_argmax(session->logits.get(), (uint32_t)vocabulary_size);
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
  session->dspark_drafted += drafted;
  session->dspark_accepted += committed;
  session->dspark_support_drafted += drafted - 1u;
  session->dspark_support_accepted += committed > 0u ? committed - 1u : 0u;
  session->dspark_positional_accepted += positional_accepted;
  session->dspark_anchors += 1u;
  if (committed == drafted)
    session->dspark_full_blocks += 1u;
  session->dspark_steps += 1;
  session_dspark_note_cycle(session, concurrent_batch);
  return 0;
}

/*
 * One greedy DSpark speculative cycle.
 *
 * Emits the target-known anchor plus its accepted support-model tail, or just
 * the target token when drafting is skipped. Accepted verifier prefixes become
 * authoritative cache state. Replay is only an internal safety fallback if a
 * prefix snapshot cannot be committed.
 */
int ds4_session_dspark_step(ds4_session *session,
                           int *emitted,
                           int emitted_cap,
                           int *n_emitted,
                           char *error,
                           size_t error_capacity) {
    if (!session || !emitted || !n_emitted || emitted_cap < 1) {
        set_error(error, error_capacity, "invalid DSpark step request");
        return 1;
    }
    *n_emitted = 0;
    const uint32_t block = ds4_rocm_graph_dspark_block_size(session->graph);
    const int length = session->checkpoint.len;
    const int vocabulary_size = ds4_engine_vocab_size(session->engine);
    const int target_first =
        ds4_sample_argmax(session->logits.get(), (uint32_t)vocabulary_size);
    const uint32_t required_context = (uint32_t)length;

    (void)session_dspark_seed_pending(session, required_context);

    /* Fall back to one ordinary token whenever a block cannot be drafted or
     * verified: no drafter, no captured features, or not enough context room. */
    bool can_draft =
        !session->dspark_plain_only && block != 0 && length >= 2 &&
        length + (int)block + 1 < session->context_size &&
        ds4_rocm_graph_dspark_context_len(session->graph) >=
            required_context &&
        emitted_cap >= (int)block + 1;
    if (can_draft && session->dspark_skip_remaining != 0) {
        session->dspark_skip_remaining--;
        session->dspark_skipped++;
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
        drafts + 1, &tail_drafted, false);
    uint32_t drafted = 0;
    if (proposed && tail_drafted <= DS4_DSPARK_MAX_BLOCK) {
        drafts[0] = target_first;
        drafted = tail_drafted + 1u;
    }
    const uint32_t drafted_cap = DS4_DSPARK_MAX_BLOCK + 1u;
    if (!proposed || drafted == 0 || drafted > drafted_cap) {
        if (!session_commit_and_extend(session, target_first)) {
            set_error(error, error_capacity, "DeepSeek decode failed after draft failure");
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
    const bool verified = ds4_rocm_graph_verify_suffix(session->graph,
                                                      session->engine,
                                                      &session->checkpoint,
                                                      (uint32_t)length,
                                                      drafted,
                                                      row_tops);
    session->checkpoint.len = length;
    if (!verified) {
        set_error(error, error_capacity, "DSpark verification failed");
        session->checkpoint_valid = false;
        return 1;
    }
    return session_dspark_finish_verified(
        session, length, vocabulary_size, false, drafts, drafted, row_tops,
        emitted, n_emitted, error, error_capacity);
}

int ds4_sessions_dspark_step_batch(const ds4_session_dspark_batch_item* items,
                                   size_t item_count, char* error,
                                   size_t error_capacity) {
  if (!items || item_count < 2u || item_count > 8u) {
    set_error(error, error_capacity,
              "DSpark batch requires two to eight sessions");
    return 1;
  }

  ds4_engine* engine = nullptr;
  uint32_t block = 0;
  int vocabulary_size = 0;
  std::array<int, 8> lengths{};
  std::array<int, 8> target_first{};
  bool all_can_draft = true;
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_session_dspark_batch_item& item = items[index];
    ds4_session* session = item.session;
    if (!session || !session->checkpoint_valid || !item.emitted ||
        !item.n_emitted || item.emitted_cap < 1) {
      set_error(error, error_capacity,
                "DSpark batch contains an invalid session");
      return 1;
    }
    *item.n_emitted = 0;
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

    const int length = session->checkpoint.len;
    lengths[index] = length;
    target_first[index] = ds4_sample_argmax(
        session->logits.get(), static_cast<uint32_t>(vocabulary_size));
    (void)session_dspark_seed_pending(session, static_cast<uint32_t>(length));

    bool can_draft =
        !session->dspark_plain_only && block != 0u && length >= 2 &&
        length + static_cast<int>(block) + 1 < session->context_size &&
        ds4_rocm_graph_dspark_context_len(session->graph) >=
            static_cast<uint32_t>(length) &&
        item.emitted_cap >= static_cast<int>(block) + 1;
    if (can_draft && session->dspark_skip_remaining != 0u) {
      session->dspark_skip_remaining--;
      session->dspark_skipped++;
      can_draft = false;
    }
    all_can_draft = all_can_draft && can_draft;
  }

  /*
   * C2 can amortize a confidence-trimmed support proposal against W2.
   * Wider waves stream target weights more efficiently than serial support
   * passes, so keep C4-C8 on their exact W4-W8 route.
   */
  if (item_count > 2u) {
    for (size_t index = 0; index < item_count; ++index) {
      ds4_session* session = items[index].session;
      session->dspark_plain_only = true;
      session->dspark_force_plain_request = true;
      ds4_rocm_graph_dspark_set_capture_enabled(session->graph, false);
    }
    if (!sessions_commit_and_extend_batch(items, target_first, item_count)) {
      for (size_t index = 0; index < item_count; ++index) {
        items[index].session->checkpoint_valid = false;
      }
      set_error(error, error_capacity,
                "DeepSeek DSpark wide target batch decode failed");
      return 1;
    }
    for (size_t index = 0; index < item_count; ++index) {
      items[index].emitted[0] = target_first[index];
      *items[index].n_emitted = 1;
    }
    return 0;
  }

  /*
   * A mixed speculative/plain cycle would need two target launches and lose
   * the weight-streaming benefit. Keep request state intact and advance all
   * members exactly once; a later cycle can batch them again.
   */
  if (!all_can_draft) {
    if (!sessions_commit_and_extend_batch(items, target_first, item_count)) {
      for (size_t index = 0; index < item_count; ++index) {
        items[index].session->checkpoint_valid = false;
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
  std::array<uint32_t, 8> tail_drafted{};
  bool proposed_all = true;
  for (size_t index = 0; index < item_count; ++index) {
    const bool proposed = ds4_rocm_graph_dspark_draft(
        items[index].session->graph, engine, target_first[index],
        static_cast<uint32_t>(lengths[index]), drafts[index].data() + 1,
        &tail_drafted[index], true);
    proposed_all = proposed_all && proposed;
  }

  uint32_t common_tail = DS4_DSPARK_MAX_BLOCK;
  for (size_t index = 0; index < item_count; ++index) {
    if (proposed_all && tail_drafted[index] <= DS4_DSPARK_MAX_BLOCK) {
      drafts[index][0] = target_first[index];
      drafted_counts[index] = tail_drafted[index] + 1u;
      common_tail = std::min(common_tail, tail_drafted[index]);
    }
    proposed_all = proposed_all && drafted_counts[index] != 0u &&
                   drafted_counts[index] <= kDraftCapacity;
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
  if (common_tail == 0u) {
    if (!sessions_commit_and_extend_batch(items, target_first, item_count)) {
      for (size_t index = 0; index < item_count; ++index) {
        items[index].session->checkpoint_valid = false;
      }
      set_error(error, error_capacity,
                "DeepSeek DSpark confidence fallback batch decode failed");
      return 1;
    }
    for (size_t index = 0; index < item_count; ++index) {
      ds4_session* session = items[index].session;
      session->dspark_plain_only = true;
      ds4_rocm_graph_dspark_set_capture_enabled(session->graph, false);
      items[index].emitted[0] = target_first[index];
      *items[index].n_emitted = 1;
    }
    return 0;
  }
  for (size_t index = 0; index < item_count; ++index) {
    drafted_counts[index] = common_tail + 1u;
  }

  std::array<ds4_rocm_verify_item, 8> verify_items{};
  for (size_t index = 0; index < item_count; ++index) {
    ds4_session* session = items[index].session;
    for (uint32_t row = 0; row < drafted_counts[index]; ++row) {
      ds4_tokens_push(&session->checkpoint, drafts[index][row]);
    }
    verify_items[index] = {
        .graph = session->graph,
        .tokens = &session->checkpoint,
        .start = static_cast<uint32_t>(lengths[index]),
        .n_tokens = drafted_counts[index],
        .row_tops = row_tops[index].data(),
    };
  }

  const bool verified =
      ds4_rocm_graph_verify_batch(engine, verify_items.data(), item_count);
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

  for (size_t index = 0; index < item_count; ++index) {
    const int result = session_dspark_finish_verified(
        items[index].session, lengths[index], vocabulary_size, true,
        drafts[index].data(), drafted_counts[index], row_tops[index].data(),
        items[index].emitted, items[index].n_emitted, error, error_capacity);
    if (result != 0) {
      for (size_t remaining = index + 1u; remaining < item_count; ++remaining) {
        items[remaining].session->checkpoint_valid = false;
      }
      return result;
    }
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
    if (drafted) *drafted = session->dspark_drafted;
    if (accepted) *accepted = session->dspark_accepted;
    if (support_drafted) *support_drafted = session->dspark_support_drafted;
    if (support_accepted) *support_accepted = session->dspark_support_accepted;
    if (positional_accepted) {
        *positional_accepted = session->dspark_positional_accepted;
    }
    if (anchors) *anchors = session->dspark_anchors;
    if (full_blocks) *full_blocks = session->dspark_full_blocks;
    if (steps) *steps = session->dspark_steps;
    if (skipped) *skipped = session->dspark_skipped;
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
