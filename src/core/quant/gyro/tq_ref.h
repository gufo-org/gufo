/*
 * tq_ref - reference decoder and reference encoder for the Gyro trellis tensor
 * formats TQ2_T, TQK4..TQK7 and TQP5..TQP7, and the Gyro block-Hadamard rotation.
 *
 * Copyright 2026 AgentionAI (Laurent Zuijdwijk). Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
 *
 * Tensor format: trellis-coded quantization with Hadamard rotation, in the
 * family of QTIP (Tseng et al., 2024). Format, encoder and kernels: AgentionAI.
 *
 * Dependency-free C11. See SPEC.md for the normative byte-level definition.
 */
#ifndef TQ_REF_H
#define TQ_REF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF / ggml tensor type ids. */
enum tq_type {
    TQ_TYPE_TQ2_T = 144, /* K = 8, 2.125 bpw, 34-byte block */
    TQ_TYPE_TQK6  = 145, /* K = 6, 1.625 bpw, 26-byte block */
    TQ_TYPE_TQK7  = 146, /* K = 7, 1.875 bpw, 30-byte block */
    TQ_TYPE_TQK5  = 150, /* K = 5, 1.375 bpw, 22-byte block */
    TQ_TYPE_TQK4  = 151, /* K = 4, 1.125 bpw, 18-byte block */
    TQ_TYPE_TQP5  = 147, /* K = 5, pair trellis, 2.625 bpw, 42-byte block */
    TQ_TYPE_TQP6  = 148, /* K = 6, pair trellis, 3.125 bpw, 50-byte block */
    TQ_TYPE_TQP7  = 149  /* K = 7, pair trellis, 3.625 bpw, 58-byte block */
};

#define TQ_BLOCK       128   /* weights per block                       */
#define TQ_STEPS       32    /* trellis steps per block, TQ2_T / TQK (4 weights each) */
#define TQP_STEPS      64    /* trellis steps per block, TQP (2 weights each)         */
#define TQ_STATE_BITS  15    /* bits of the state that reach the codebook */
#define TQ_STATE_MASK  ((1u << TQ_STATE_BITS) - 1u)
#define TQ_CB_SIZE     2048  /* 2-D codebook points                     */

/* The normative 2048-point codebook: point i = (tq_codebook_f16[2i], tq_codebook_f16[2i+1]),
 * IEEE-754 binary16 bit patterns. */
extern const uint16_t tq_codebook_f16[2 * TQ_CB_SIZE];

/* Type metadata. All return 0 / NULL for an unknown type id. */
int         tq_type_k(int type);           /* new bits per step: 4..8              */
int         tq_type_steps(int type);       /* 32 (TQ2_T, TQK) or 64 (TQP)          */
size_t      tq_block_bytes(int type);      /* 2 + steps*K/8                        */
const char *tq_type_name(int type);
size_t      tq_row_bytes(int type, int64_t n); /* n % 128 == 0                 */

/* binary16 <-> binary32 (exact / round-to-nearest-even). */
float    tq_fp16_to_fp32(uint16_t h);
uint16_t tq_fp32_to_fp16(float f);

/* Codebook index (0..2047) of pair p (0 or 1) of state s (any width; masked to 15 bits). */
static inline uint32_t tq_codebook_index(uint32_t s, uint32_t p) {
    const uint32_t x = (s & TQ_STATE_MASK) * 0x9e3779b1u;
    return p ? ((x >> 10) & 2047u) : (x >> 21);
}

/* 16-bit state of step t (0..31) of one block's qs bytes, for K = 4..8 (K = 8 is TQ2_T). */
uint32_t tq_state(const uint8_t *qs, int k, int t);
/* 16-bit state of step t (0..63) of a TQP block's qs bytes, K = 5..7. */
uint32_t tq_state_tqp(const uint8_t *qs, int k, int t);

/* Decode n weights (n % 128 == 0) of the given type. Returns 0 on success, -1 on bad args. */
int tq_dequantize_row(int type, const void *src, float *dst, int64_t n);

/* Plain reference encoder: per block, tail-biting Viterbi over the trellis minimising the
 * (optionally importance-weighted) squared error, with a simple block-scale search.
 * imp: NULL, or n non-negative per-weight importances (e.g. an imatrix row, repeated per
 * row of the tensor). Returns bytes written, 0 on bad args or allocation failure. */
size_t tq_quantize_row_ref(int type, const float *src, void *dst, int64_t n, const float *imp);

/* Block-Hadamard rotation (SPEC.md section 8, GGUF keys prism.hadamard.*).
 * T v = H_b (s . v) on every block-wide slice of v, H_b the normalised Sylvester-Walsh-Hadamard
 * matrix (entries (-1)^popcount(r & c) / sqrt(b)), s a +1/-1 sign vector of length n (NULL:
 * all +1). A rotated weight row is stored as T w; the runtime applies T to the matmul input,
 * so (T w) . (T x) = w . x. tq_hadamard_rotate applies T, tq_hadamard_unrotate applies T^-1
 * (v <- s . (H_b v)). block: power of two >= 1 dividing n. Return 0, or -1 on bad args. */
int tq_hadamard_rotate  (float *v, int64_t n, int block, const int8_t *signs);
int tq_hadamard_unrotate(float *v, int64_t n, int block, const int8_t *signs);

/* ggml-style named wrappers. */
void dequantize_row_tq2_t(const void *x, float *y, int64_t n);
void dequantize_row_tqk4 (const void *x, float *y, int64_t n);
void dequantize_row_tqk5 (const void *x, float *y, int64_t n);
void dequantize_row_tqk6 (const void *x, float *y, int64_t n);
void dequantize_row_tqk7 (const void *x, float *y, int64_t n);
void dequantize_row_tqp5 (const void *x, float *y, int64_t n);
void dequantize_row_tqp6 (const void *x, float *y, int64_t n);
void dequantize_row_tqp7 (const void *x, float *y, int64_t n);
size_t quantize_row_tq2_t_ref(const float *x, void *y, int64_t n, const float *imp);
size_t quantize_row_tqk4_ref (const float *x, void *y, int64_t n, const float *imp);
size_t quantize_row_tqk5_ref (const float *x, void *y, int64_t n, const float *imp);
size_t quantize_row_tqk6_ref (const float *x, void *y, int64_t n, const float *imp);
size_t quantize_row_tqk7_ref (const float *x, void *y, int64_t n, const float *imp);
size_t quantize_row_tqp5_ref (const float *x, void *y, int64_t n, const float *imp);
size_t quantize_row_tqp6_ref (const float *x, void *y, int64_t n, const float *imp);
size_t quantize_row_tqp7_ref (const float *x, void *y, int64_t n, const float *imp);

#ifdef __cplusplus
}
#endif

#endif /* TQ_REF_H */
