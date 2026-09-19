"""Quality metrics: KL divergence, perplexity/NLL, top-k agreement, logit RMSE.

Matched-token teacher forcing per docs/TESTING.md.
"""

from __future__ import annotations

import numpy as np


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax along the specified axis."""
    m = np.max(z, axis=axis, keepdims=True)
    zz = z - m
    exp_zz = np.exp(zz)
    sum_exp = np.sum(exp_zz, axis=axis, keepdims=True)
    return zz - np.log(sum_exp)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the specified axis."""
    zz = z - np.max(z, axis=axis, keepdims=True)
    exp_zz = np.exp(zz)
    return exp_zz / np.sum(exp_zz, axis=axis, keepdims=True)


def kl_divergence_log_space(lp_t: np.ndarray, lp_c: np.ndarray, axis: int = -1) -> np.ndarray:
    """Computes KL(P_t || P_c) = sum P_t * (log P_t - log P_c) directly in log-space."""
    p_t = np.exp(lp_t)
    return np.sum(p_t * (lp_t - lp_c), axis=axis)


def compare_logits(teacher_logits: np.ndarray,
                   candidate_logits: np.ndarray,
                   target_tokens: np.ndarray | list | None = None,
                   top_k: int = 5) -> dict:
    """Calculates full distribution and distance metrics between teacher and candidate logit tensors.

    teacher_logits, candidate_logits: float32 numpy [positions, vocab].
    Returns dict of aggregates.
    """
    if teacher_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"Shape mismatch: teacher {teacher_logits.shape} vs candidate {candidate_logits.shape}"
        )

    t_nonfinite = int(np.isnan(teacher_logits).sum() + np.isinf(teacher_logits).sum())
    c_nonfinite = int(np.isnan(candidate_logits).sum() + np.isinf(candidate_logits).sum())

    if t_nonfinite > 0 or c_nonfinite > 0:
        return {
            "positions": teacher_logits.shape[0],
            "nonfinite_teacher": t_nonfinite,
            "nonfinite_candidate": c_nonfinite,
            "valid": False,
        }

    T, V = teacher_logits.shape
    lp_t = log_softmax(teacher_logits)
    lp_c = log_softmax(candidate_logits)
    p_t = np.exp(lp_t)
    p_c = np.exp(lp_c)

    kl = kl_divergence_log_space(lp_t, lp_c, axis=-1)

    top1_t = np.argmax(teacher_logits, axis=1)
    top1_c = np.argmax(candidate_logits, axis=1)
    topk_t = np.argsort(teacher_logits, axis=1)[:, -top_k:]
    topk_c = np.argsort(candidate_logits, axis=1)[:, -top_k:]

    top1_agree = float((top1_t == top1_c).mean())

    # top-k set overlap per position
    topk_overlap = []
    for i in range(T):
        s_t = set(topk_t[i].tolist())
        s_c = set(topk_c[i].tolist())
        topk_overlap.append(len(s_t & s_c) / float(top_k))
    topk_agree = float(np.mean(topk_overlap))

    # candidate probability on teacher top-1 token
    cand_prob_on_t1 = p_c[np.arange(T), top1_t]

    # teacher top-1 rank in candidate distribution (0 = exact match)
    # rank is count of candidate logits strictly greater than candidate logit at top1_t
    cand_at_t1 = candidate_logits[np.arange(T), top1_t][:, None]
    rank_displacement = np.sum(candidate_logits > cand_at_t1, axis=1)

    # Logit and log-prob differences
    logit_diff = teacher_logits - candidate_logits
    logit_rmse = float(np.sqrt(np.mean(logit_diff ** 2)))
    logit_max_abs = float(np.max(np.abs(logit_diff)))

    lp_diff = lp_t - lp_c
    lp_rmse = float(np.sqrt(np.mean(lp_diff ** 2)))
    lp_max_abs = float(np.max(np.abs(lp_diff)))

    nll = None
    if target_tokens is not None:
        target_tokens = np.asarray(target_tokens)
        if len(target_tokens) == T:
            nll = -np.log(np.clip(p_c[np.arange(T), target_tokens], 1e-30, 1.0))

    metrics = {
        "positions": T,
        "vocab_size": V,
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "kl_p95": float(np.percentile(kl, 95)),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_p99_9": float(np.percentile(kl, 99.9)),
        "kl_max": float(kl.max()),
        "top1_agreement": top1_agree,
        f"top{top_k}_agreement": topk_agree,
        "teacher_top1_candidate_prob_mean": float(cand_prob_on_t1.mean()),
        "teacher_top1_rank_displacement_mean": float(rank_displacement.mean()),
        "teacher_top1_rank_displacement_max": int(rank_displacement.max()),
        "logit_rmse": logit_rmse,
        "logit_max_abs_diff": logit_max_abs,
        "norm_log_prob_rmse": lp_rmse,
        "norm_log_prob_max_diff": lp_max_abs,
        "nonfinite_teacher": t_nonfinite,
        "nonfinite_candidate": c_nonfinite,
        "valid": True,
    }
    if nll is not None:
        metrics["nll_mean"] = float(nll.mean())
        metrics["perplexity"] = float(np.exp(nll.mean()))

    return metrics
