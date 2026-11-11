/*
 * tq_ref.c - reference decoder and reference encoder for TQ2_T / TQK4..TQK7 / TQP5..TQP7,
 * and the block-Hadamard rotation.
 *
 * Copyright 2026 AgentionAI (Laurent Zuijdwijk). Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
 *
 * The decoder is the format definition. The encoder is a plain, correct
 * reference: per 128-weight block it runs a tail-biting Viterbi search over the
 * bitshift trellis for a handful of candidate scales and keeps the best. It is a
 * plain, uncalibrated baseline.
 */
#include "tq_ref.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ types */

int tq_type_k(int type) {
    switch (type) {
        case TQ_TYPE_TQ2_T: return 8;
        case TQ_TYPE_TQK7:  return 7;
        case TQ_TYPE_TQK6:  return 6;
        case TQ_TYPE_TQK5:  return 5;
        case TQ_TYPE_TQK4:  return 4;
        case TQ_TYPE_TQP5:  return 5;
        case TQ_TYPE_TQP6:  return 6;
        case TQ_TYPE_TQP7:  return 7;
        default:            return 0;
    }
}

int tq_type_steps(int type) {
    switch (type) {
        case TQ_TYPE_TQP5: case TQ_TYPE_TQP6: case TQ_TYPE_TQP7: return TQP_STEPS;
        default: return tq_type_k(type) ? TQ_STEPS : 0;
    }
}

size_t tq_block_bytes(int type) {
    const int k = tq_type_k(type);
    return k ? (size_t) (2 + tq_type_steps(type) * k / 8) : 0;
}

const char *tq_type_name(int type) {
    switch (type) {
        case TQ_TYPE_TQ2_T: return "TQ2_T";
        case TQ_TYPE_TQK7:  return "TQK7";
        case TQ_TYPE_TQK6:  return "TQK6";
        case TQ_TYPE_TQK5:  return "TQK5";
        case TQ_TYPE_TQK4:  return "TQK4";
        case TQ_TYPE_TQP5:  return "TQP5";
        case TQ_TYPE_TQP6:  return "TQP6";
        case TQ_TYPE_TQP7:  return "TQP7";
        default:            return NULL;
    }
}

size_t tq_row_bytes(int type, int64_t n) {
    if (n < 0 || n % TQ_BLOCK) return 0;
    return (size_t) (n / TQ_BLOCK) * tq_block_bytes(type);
}

/* --------------------------------------------------------------- binary16 */

static uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float    bits_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

float tq_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t       man  = h & 0x3ffu;
    if (exp == 0x1f) return bits_f32(sign | 0x7f800000u | (man << 13)); /* inf / nan */
    if (exp != 0)    return bits_f32(sign | ((exp + 112u) << 23) | (man << 13));
    if (man == 0)    return bits_f32(sign);
    /* subnormal: value = man * 2^-24 */
    int e = -1;
    do { man <<= 1; ++e; } while (!(man & 0x400u));
    return bits_f32(sign | ((uint32_t) (112 - e) << 23) | ((man & 0x3ffu) << 13));
}

uint16_t tq_fp32_to_fp16(float f) {
    const uint32_t u    = f32_bits(f);
    const uint16_t sign = (uint16_t) ((u >> 16) & 0x8000u);
    const uint32_t a    = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return (uint16_t) (sign | 0x7c00u | (a > 0x7f800000u ? 0x200u : 0u));
    if (a >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);             /* rounds to inf */
    if (a < 0x38800000u) {                                                /* fp16 subnormal or 0 */
        if (a < 0x33000000u) return sign;                                 /* < 2^-25 -> 0 */
        const uint32_t e     = a >> 23;
        const uint32_t m     = (a & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126u - e;                                  /* 14..24 */
        uint32_t       r     = m >> shift;
        const uint32_t rem   = m & ((1u << shift) - 1u);
        const uint32_t half  = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return (uint16_t) (sign | r);
    }
    uint32_t r = ((a >> 13) - (112u << 10));
    const uint32_t rem = a & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    return (uint16_t) (sign | r);
}

/* ---------------------------------------------------------------- decoder */

/* Circular nsteps*K-bit LSB-first stream; 16-bit window starting at bit (nsteps - 1 - t)*K. */
static uint32_t stream_state(const uint8_t *qs, int k, int nsteps, int t) {
    const uint32_t nbytes = (uint32_t) (nsteps * k / 8);
    const uint32_t off    = (uint32_t) (nsteps - 1 - t) * (uint32_t) k;
    const uint32_t i0     = off >> 3;
    const uint32_t w      = (uint32_t) qs[i0]
                          | ((uint32_t) qs[(i0 + 1) % nbytes] << 8)
                          | ((uint32_t) qs[(i0 + 2) % nbytes] << 16);
    return (w >> (off & 7)) & 0xffffu;
}

uint32_t tq_state(const uint8_t *qs, int k, int t) {
    if (k == 8) {
        /* TQ2_T: one byte per step; state = previous step's byte (circular) : this byte. */
        return ((uint32_t) qs[(t + TQ_STEPS - 1) % TQ_STEPS] << 8) | qs[t];
    }
    return stream_state(qs, k, TQ_STEPS, t); /* TQK */
}

uint32_t tq_state_tqp(const uint8_t *qs, int k, int t) {
    return stream_state(qs, k, TQP_STEPS, t);
}

static uint32_t block_state(const uint8_t *qs, int k, int nsteps, int t) {
    return nsteps == TQP_STEPS ? stream_state(qs, k, TQP_STEPS, t) : tq_state(qs, k, t);
}

int tq_dequantize_row(int type, const void *src, float *dst, int64_t n) {
    const int k = tq_type_k(type);
    if (!k || !src || !dst || n < 0 || n % TQ_BLOCK) return -1;
    const size_t   bs = tq_block_bytes(type);
    const int      ns = tq_type_steps(type);
    const uint32_t np = ns == TQP_STEPS ? 1 : 2;  /* codebook pairs per step */
    const uint8_t *b  = (const uint8_t *) src;
    for (int64_t i = 0; i < n / TQ_BLOCK; ++i, b += bs) {
        const float    d  = tq_fp16_to_fp32((uint16_t) (b[0] | (b[1] << 8)));
        const uint8_t *qs = b + 2;
        for (int t = 0; t < ns; ++t) {
            const uint32_t s = block_state(qs, k, ns, t);
            for (uint32_t p = 0; p < np; ++p) {
                const uint32_t idx = tq_codebook_index(s, p);
                *dst++ = d * tq_fp16_to_fp32(tq_codebook_f16[2 * idx + 0]);
                *dst++ = d * tq_fp16_to_fp32(tq_codebook_f16[2 * idx + 1]);
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------- encoder */

#define NSTATE      (1u << TQ_STATE_BITS)
#ifndef TQ_TB_CANDIDATES
#define TQ_TB_CANDIDATES 4   /* tail-biting start/end candidates tried per scale */
#endif
#define N_CAND TQ_TB_CANDIDATES
#define N_SCALES    8      /* scale grid points                                */
static const float k_scale_grid[N_SCALES] = { 0.70f, 0.78f, 0.86f, 0.94f, 1.02f, 1.10f, 1.20f, 1.32f };

typedef struct {
    float   cb[2 * TQ_CB_SIZE];        /* codebook as f32                          */
    float  *cost;                      /* [nsteps][NSTATE] step costs              */
    float  *dp;                        /* [nsteps + 1][NSTATE] path metrics        */
    float  *m;                         /* [NSTATE >> 4] best predecessor metric    */
    int     k, nsteps;                 /* bits per step; 32 (4 values) or 64 (2)   */
} tq_work;

/* Step costs for target y (unit-scale) with weights w: cost[t][s] = sum_j w_j (y_j - c_j(s))^2. */
static void build_costs(tq_work *W, const float *y, const float *w) {
    float e0[TQ_CB_SIZE], e1[TQ_CB_SIZE];
    if (W->nsteps == TQP_STEPS) { /* pair trellis: 2 values per step, first pair of the state */
        for (int t = 0; t < TQP_STEPS; ++t) {
            const float *yt = y + 2 * t, *wt = w + 2 * t;
            for (int i = 0; i < TQ_CB_SIZE; ++i) {
                const float a = yt[0] - W->cb[2 * i], b = yt[1] - W->cb[2 * i + 1];
                e0[i] = wt[0] * a * a + wt[1] * b * b;
            }
            float *ct = W->cost + (size_t) t * NSTATE;
            for (uint32_t s = 0; s < NSTATE; ++s) ct[s] = e0[(s * 0x9e3779b1u) >> 21];
        }
        return;
    }
    for (int t = 0; t < TQ_STEPS; ++t) {
        const float *yt = y + 4 * t, *wt = w + 4 * t;
        for (int i = 0; i < TQ_CB_SIZE; ++i) {
            const float c0 = W->cb[2 * i], c1 = W->cb[2 * i + 1];
            const float a = yt[0] - c0, b = yt[1] - c1, c = yt[2] - c0, d = yt[3] - c1;
            e0[i] = wt[0] * a * a + wt[1] * b * b;
            e1[i] = wt[2] * c * c + wt[3] * d * d;
        }
        float *ct = W->cost + (size_t) t * NSTATE;
        for (uint32_t s = 0; s < NSTATE; ++s) {
            const uint32_t x = s * 0x9e3779b1u;
            ct[s] = e0[x >> 21] + e1[(x >> 10) & 2047u];
        }
    }
}

/* Viterbi recursion for a compile-time K (lets the compiler vectorise the inner loops).
 * dp[t+1][s] = cost[t][s] + min_h dp[t][(s >> K) | (h << (15 - K))]. */
#define TQ_DEFINE_FORWARD(K)                                                                   \
static void forward_k##K(tq_work *W) {                                                       \
    enum { NQ = NSTATE >> K, NH = 1 << K };                                                   \
    float *m = W->m;                                                                          \
    for (int t = 0; t < W->nsteps; ++t) {                                                     \
        const float *dp = W->dp + (size_t) t * NSTATE;                                        \
        float *dpn = W->dp + (size_t) (t + 1) * NSTATE;                                       \
        const float *ct = W->cost + (size_t) t * NSTATE;                                      \
        memcpy(m, dp, NQ * sizeof(float));                                                    \
        for (int h = 1; h < NH; ++h) {                                                        \
            const float *row = dp + (size_t) h * NQ;                                          \
            for (int q = 0; q < NQ; ++q) m[q] = row[q] < m[q] ? row[q] : m[q];                \
        }                                                                                     \
        for (int q = 0; q < NQ; ++q) {                                                        \
            const float mq = m[q];                                                            \
            for (int v = 0; v < NH; ++v) dpn[(q << K) | v] = ct[(q << K) | v] + mq;           \
        }                                                                                     \
    }                                                                                         \
}
TQ_DEFINE_FORWARD(4)
TQ_DEFINE_FORWARD(5)
TQ_DEFINE_FORWARD(6)
TQ_DEFINE_FORWARD(7)
TQ_DEFINE_FORWARD(8)

/* One Viterbi pass. States are the 15 low bits of the shift register:
 *   s_t = ((s_{t-1} << K) | v_t) & TQ_STATE_MASK.
 * start < 0: free initial state; else s_{-1} = start. end < 0: best final state; else the last
 * state = end. Writes the nsteps states to path and returns the path cost. After the call
 * W->dp + nsteps*NSTATE holds the final metrics. */
static float viterbi(tq_work *W, int k, int32_t start, int32_t end, uint32_t *path) {
    const uint32_t hs = TQ_STATE_BITS - k, nh = 1u << k;
    float *dp0 = W->dp;
    for (uint32_t s = 0; s < NSTATE; ++s) dp0[s] = (start < 0 || (uint32_t) start == s) ? 0.0f : INFINITY;
    switch (k) {
        case 4: forward_k4(W); break;
        case 5: forward_k5(W); break;
        case 6: forward_k6(W); break;
        case 7: forward_k7(W); break;
        default: forward_k8(W); break;
    }
    const int ns = W->nsteps;
    const float *fin = W->dp + (size_t) ns * NSTATE;
    uint32_t s;
    if (end >= 0) {
        s = (uint32_t) end;
    } else {
        s = 0;
        for (uint32_t i = 1; i < NSTATE; ++i) if (fin[i] < fin[s]) s = i;
    }
    const float total = fin[s];
    /* traceback: re-find each step's best predecessor (first minimum, as in the forward pass) */
    path[ns - 1] = s;
    for (int t = ns - 1; t > 0; --t) {
        const float *dp = W->dp + (size_t) t * NSTATE;
        const uint32_t q = s >> k;
        uint32_t best = q;
        for (uint32_t h = 1; h < nh; ++h) if (dp[q | (h << hs)] < dp[best]) best = q | (h << hs);
        s = best;
        path[t - 1] = s;
    }
    return total;
}

/* Best tail-biting path (state before step 0 == last state) for the current cost table. */
static float viterbi_tailbiting(tq_work *W, int k, uint32_t *path, float *scratch_dp) {
    uint32_t tmp[TQP_STEPS];
    viterbi(W, k, -1, -1, tmp);
    memcpy(scratch_dp, W->dp + (size_t) W->nsteps * NSTATE, NSTATE * sizeof(float));
    /* candidates: the N_CAND best final states of the free run */
    int32_t cand[N_CAND];
    float   cv[N_CAND];
    for (int c = 0; c < N_CAND; ++c) { cand[c] = -1; cv[c] = INFINITY; }
    for (uint32_t s = 0; s < NSTATE; ++s) {
        const float v = scratch_dp[s];
        if (v >= cv[N_CAND - 1]) continue;
        int c = N_CAND - 1;
        while (c > 0 && cv[c - 1] > v) { cv[c] = cv[c - 1]; cand[c] = cand[c - 1]; --c; }
        cv[c] = v; cand[c] = (int32_t) s;
    }
    float best = INFINITY;
    for (int c = 0; c < N_CAND; ++c) {
        if (cand[c] < 0) continue;
        const float e = viterbi(W, k, cand[c], cand[c], tmp);
        if (e < best) { best = e; memcpy(path, tmp, (size_t) W->nsteps * sizeof(uint32_t)); }
    }
    return best;
}

static void path_values(const tq_work *W, const uint32_t *path, float *c) {
    if (W->nsteps == TQP_STEPS) {
        for (int t = 0; t < TQP_STEPS; ++t) {
            const uint32_t idx = tq_codebook_index(path[t], 0);
            c[2 * t + 0] = W->cb[2 * idx + 0];
            c[2 * t + 1] = W->cb[2 * idx + 1];
        }
        return;
    }
    for (int t = 0; t < TQ_STEPS; ++t)
        for (uint32_t p = 0; p < 2; ++p) {
            const uint32_t idx = tq_codebook_index(path[t], p);
            c[4 * t + 2 * p + 0] = W->cb[2 * idx + 0];
            c[4 * t + 2 * p + 1] = W->cb[2 * idx + 1];
        }
}

/* Weighted least-squares scale for fixed codebook values, rounded to binary16; returns the
 * weighted error at that rounded scale. */
static float fit_scale(const float *x, const float *w, const float *c, uint16_t *dh) {
    double sxc = 0.0, scc = 0.0;
    for (int j = 0; j < TQ_BLOCK; ++j) { sxc += w[j] * x[j] * c[j]; scc += w[j] * c[j] * c[j]; }
    float d = scc > 0.0 ? (float) (sxc / scc) : 0.0f;
    if (d > 65504.0f) d = 65504.0f;
    if (d < -65504.0f) d = -65504.0f;
    *dh = tq_fp32_to_fp16(d);
    const float dr = tq_fp16_to_fp32(*dh);
    double err = 0.0;
    for (int j = 0; j < TQ_BLOCK; ++j) { const double e = x[j] - dr * c[j]; err += w[j] * e * e; }
    return (float) err;
}

static void write_block(uint8_t *b, int k, int ns, uint16_t dh, const uint32_t *path) {
    b[0] = (uint8_t) (dh & 0xff);
    b[1] = (uint8_t) (dh >> 8);
    uint8_t *qs = b + 2;
    memset(qs, 0, (size_t) (ns * k / 8));
    if (k == 8) {
        for (int t = 0; t < TQ_STEPS; ++t) qs[t] = (uint8_t) (path[t] & 0xffu);
        return;
    }
    for (int t = 0; t < ns; ++t) { /* step t's new bits at stream bit (ns - 1 - t)*K */
        const uint32_t v = path[t] & ((1u << k) - 1u);
        const int off = (ns - 1 - t) * k;
        for (int i = 0; i < k; ++i)
            if ((v >> i) & 1u) qs[(off + i) >> 3] |= (uint8_t) (1u << ((off + i) & 7));
    }
}

static void encode_block(tq_work *W, int k, const float *x, const float *imp, uint8_t *out, float *scratch_dp) {
    float w[TQ_BLOCK], y[TQ_BLOCK], c[TQ_BLOCK];
    double sw = 0.0, swxx = 0.0, sxx = 0.0;
    for (int j = 0; j < TQ_BLOCK; ++j) {
        w[j] = (imp && imp[j] > 0.0f && isfinite(imp[j])) ? imp[j] : (imp ? 0.0f : 1.0f);
        sw += w[j];
        swxx += w[j] * x[j] * x[j];
        sxx += (double) x[j] * x[j];
    }
    if (sw <= 0.0) { for (int j = 0; j < TQ_BLOCK; ++j) w[j] = 1.0f; sw = TQ_BLOCK; swxx = sxx; }
    if (sxx == 0.0 || swxx == 0.0) {
        const uint32_t zero[TQP_STEPS] = { 0 };
        write_block(out, k, W->nsteps, 0, zero);
        return;
    }
    const float sigma = (float) sqrt(swxx / sw);

    uint32_t path[TQP_STEPS], best_path[TQP_STEPS];
    uint16_t dh, best_dh = 0;
    float best_err = INFINITY, best_d = sigma;

    for (int g = 0; g <= N_SCALES; ++g) {
        /* grid points, then one refinement at the best least-squares scale found */
        const float d0 = g < N_SCALES ? sigma * k_scale_grid[g] : best_d;
        if (!(d0 > 0.0f)) continue;
        for (int j = 0; j < TQ_BLOCK; ++j) y[j] = x[j] / d0;
        build_costs(W, y, w);
        viterbi_tailbiting(W, k, path, scratch_dp);
        path_values(W, path, c);
        const float err = fit_scale(x, w, c, &dh);
        if (err < best_err && tq_fp16_to_fp32(dh) > 0.0f) {
            best_err = err; best_dh = dh; best_d = tq_fp16_to_fp32(dh);
            memcpy(best_path, path, (size_t) W->nsteps * sizeof(uint32_t));
        }
    }
    if (!isfinite(best_err)) { /* degenerate block: keep a valid all-zero path at scale sigma */
        memset(best_path, 0, sizeof(best_path));
        best_dh = tq_fp32_to_fp16(sigma);
    }
    write_block(out, k, W->nsteps, best_dh, best_path);
}

size_t tq_quantize_row_ref(int type, const float *src, void *dst, int64_t n, const float *imp) {
    const int k = tq_type_k(type);
    if (!k || !src || !dst || n < 0 || n % TQ_BLOCK) return 0;
    tq_work *W = (tq_work *) calloc(1, sizeof(tq_work));
    if (!W) return 0;
    W->k      = k;
    W->nsteps = tq_type_steps(type);
    W->cost = (float *) malloc((size_t) W->nsteps * NSTATE * sizeof(float));
    W->dp   = (float *) malloc((size_t) (W->nsteps + 1) * NSTATE * sizeof(float));
    W->m    = (float *) malloc((NSTATE >> 4) * sizeof(float));
    float *scratch = (float *) malloc(NSTATE * sizeof(float));
    size_t written = 0;
    if (W->cost && W->dp && W->m && scratch) {
        for (int i = 0; i < 2 * TQ_CB_SIZE; ++i) W->cb[i] = tq_fp16_to_fp32(tq_codebook_f16[i]);
        const size_t bs = tq_block_bytes(type);
        uint8_t *out = (uint8_t *) dst;
        for (int64_t i = 0; i < n / TQ_BLOCK; ++i)
            encode_block(W, k, src + i * TQ_BLOCK, imp ? imp + i * TQ_BLOCK : NULL, out + i * bs, scratch);
        written = (size_t) (n / TQ_BLOCK) * bs;
    }
    free(scratch); free(W->m); free(W->dp); free(W->cost); free(W);
    return written;
}

/* ------------------------------------------------------- named wrappers */

void dequantize_row_tq2_t(const void *x, float *y, int64_t n) { tq_dequantize_row(TQ_TYPE_TQ2_T, x, y, n); }
void dequantize_row_tqk4 (const void *x, float *y, int64_t n) { tq_dequantize_row(TQ_TYPE_TQK4,  x, y, n); }
void dequantize_row_tqk5 (const void *x, float *y, int64_t n) { tq_dequantize_row(TQ_TYPE_TQK5,  x, y, n); }
void dequantize_row_tqk6 (const void *x, float *y, int64_t n) { tq_dequantize_row(TQ_TYPE_TQK6,  x, y, n); }
void dequantize_row_tqk7 (const void *x, float *y, int64_t n) { tq_dequantize_row(TQ_TYPE_TQK7,  x, y, n); }
void dequantize_row_tqp5 (const void *x, float *y, int64_t n) { tq_dequantize_row(TQ_TYPE_TQP5,  x, y, n); }
void dequantize_row_tqp6 (const void *x, float *y, int64_t n) { tq_dequantize_row(TQ_TYPE_TQP6,  x, y, n); }
void dequantize_row_tqp7 (const void *x, float *y, int64_t n) { tq_dequantize_row(TQ_TYPE_TQP7,  x, y, n); }

size_t quantize_row_tq2_t_ref(const float *x, void *y, int64_t n, const float *imp) { return tq_quantize_row_ref(TQ_TYPE_TQ2_T, x, y, n, imp); }
size_t quantize_row_tqk4_ref (const float *x, void *y, int64_t n, const float *imp) { return tq_quantize_row_ref(TQ_TYPE_TQK4,  x, y, n, imp); }
size_t quantize_row_tqk5_ref (const float *x, void *y, int64_t n, const float *imp) { return tq_quantize_row_ref(TQ_TYPE_TQK5,  x, y, n, imp); }
size_t quantize_row_tqk6_ref (const float *x, void *y, int64_t n, const float *imp) { return tq_quantize_row_ref(TQ_TYPE_TQK6,  x, y, n, imp); }
size_t quantize_row_tqk7_ref (const float *x, void *y, int64_t n, const float *imp) { return tq_quantize_row_ref(TQ_TYPE_TQK7,  x, y, n, imp); }
size_t quantize_row_tqp5_ref (const float *x, void *y, int64_t n, const float *imp) { return tq_quantize_row_ref(TQ_TYPE_TQP5,  x, y, n, imp); }
size_t quantize_row_tqp6_ref (const float *x, void *y, int64_t n, const float *imp) { return tq_quantize_row_ref(TQ_TYPE_TQP6,  x, y, n, imp); }
size_t quantize_row_tqp7_ref (const float *x, void *y, int64_t n, const float *imp) { return tq_quantize_row_ref(TQ_TYPE_TQP7,  x, y, n, imp); }

/* ------------------------------------------------- block-Hadamard rotation */

/* In-place normalised Walsh-Hadamard transform of each block-wide slice: v <- H_b v.
 * H_b[r][c] = (-1)^popcount(r & c) / sqrt(b); the butterfly below computes exactly that
 * product (up to rounding). */
static int fwht_blocks(float *v, int64_t n, int block) {
    if (!v || block < 1 || (block & (block - 1)) || n < 0 || n % block) return -1;
    const float scale = 1.0f / sqrtf((float) block);
    for (int64_t o = 0; o < n; o += block) {
        float *x = v + o;
        for (int len = 1; len < block; len <<= 1)
            for (int i = 0; i < block; i += len << 1)
                for (int j = i; j < i + len; ++j) {
                    const float a = x[j], b = x[j + len];
                    x[j] = a + b;
                    x[j + len] = a - b;
                }
        for (int j = 0; j < block; ++j) x[j] *= scale;
    }
    return 0;
}

int tq_hadamard_rotate(float *v, int64_t n, int block, const int8_t *signs) {
    if (!v || block < 1 || (block & (block - 1)) || n < 0 || n % block) return -1;
    if (signs) for (int64_t i = 0; i < n; ++i) if (signs[i] < 0) v[i] = -v[i];
    return fwht_blocks(v, n, block);
}

int tq_hadamard_unrotate(float *v, int64_t n, int block, const int8_t *signs) {
    if (fwht_blocks(v, n, block)) return -1;
    if (signs) for (int64_t i = 0; i < n; ++i) if (signs[i] < 0) v[i] = -v[i];
    return 0;
}
