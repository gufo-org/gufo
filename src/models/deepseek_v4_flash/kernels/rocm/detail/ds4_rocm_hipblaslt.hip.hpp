/* HIP-only hipBLASLt state and helpers.
 * Included from ds4_rocm.hip.cpp under __HIP_PLATFORM_AMD__ to keep ROCm
 * planning/cache code out of the ROCM host runtime body. */

static hipblasLtHandle_t g_hipblaslt;
static int g_hipblaslt_ready;
struct hip_hipblaslt_gemm_plan {
    uint32_t out_dim;
    uint32_t n_tok;
    uint32_t in_dim;
    hipblasOperation_t op_a;
    hipDataType output_type;
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t a_desc;
    hipblasLtMatrixLayout_t b_desc;
    hipblasLtMatrixLayout_t c_desc;
    hipblasLtMatrixLayout_t d_desc;
    hipblasLtMatmulAlgo_t algo;
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    int tuned;
};
static std::vector<hip_hipblaslt_gemm_plan> g_hipblaslt_gemm_plans;

static void hipblaslt_gemm_plan_clear(void) {
    for (size_t i = 0; i < g_hipblaslt_gemm_plans.size(); i++) {
        hip_hipblaslt_gemm_plan &p = g_hipblaslt_gemm_plans[i];
        if (p.d_desc) (void)hipblasLtMatrixLayoutDestroy(p.d_desc);
        if (p.c_desc) (void)hipblasLtMatrixLayoutDestroy(p.c_desc);
        if (p.b_desc) (void)hipblasLtMatrixLayoutDestroy(p.b_desc);
        if (p.a_desc) (void)hipblasLtMatrixLayoutDestroy(p.a_desc);
        if (p.desc) (void)hipblasLtMatmulDescDestroy(p.desc);
    }
    g_hipblaslt_gemm_plans.clear();
}

static int hipblaslt_ok(hipblasStatus_t st, const char *what) {
    if (st == HIPBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: hipBLASLt %s failed: status %d\n", what, (int)st);
    return 0;
}

static hip_hipblaslt_gemm_plan *hipblaslt_gemm_plan_get(
        uint32_t out_dim,
        uint32_t n_tok,
        uint32_t in_dim,
        hipblasOperation_t op_a,
        hipDataType output_type,
        const char *label) {
    for (size_t i = 0; i < g_hipblaslt_gemm_plans.size(); i++) {
        hip_hipblaslt_gemm_plan &p = g_hipblaslt_gemm_plans[i];
        if (p.out_dim == out_dim && p.n_tok == n_tok && p.in_dim == in_dim &&
            p.op_a == op_a && p.output_type == output_type) {
            return &p;
        }
    }

    hipblasLtMatmulDesc_t desc = NULL;
    hipblasLtMatrixLayout_t a_desc = NULL, b_desc = NULL, c_desc = NULL, d_desc = NULL;
    hipblasLtMatmulPreference_t pref = NULL;
    hipblasLtMatmulHeuristicResult_t heur[16];
    int returned = 0;
    int ok = 0;
    do {
        if (!hipblaslt_ok(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                          "matmul desc create")) break;
        hipblasOperation_t op_b = HIPBLAS_OP_N;
        if (!hipblaslt_ok(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                          &op_a, sizeof(op_a)),
                          "set transA")) break;
        if (!hipblaslt_ok(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                          &op_b, sizeof(op_b)),
                          "set transB")) break;
        const uint32_t a_rows = op_a == HIPBLAS_OP_T ? in_dim : out_dim;
        const uint32_t a_cols = op_a == HIPBLAS_OP_T ? out_dim : in_dim;
        const uint32_t a_ld = a_rows;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&a_desc, HIP_R_16F,
                                                       a_rows, a_cols, a_ld),
                          "A layout create")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&b_desc, HIP_R_16F, in_dim, n_tok, in_dim),
                          "B layout create")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&c_desc, output_type,
                                                       out_dim, n_tok, out_dim),
                          "C layout create")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&d_desc, output_type,
                                                       out_dim, n_tok, out_dim),
                          "D layout create")) break;
        if (!hipblaslt_ok(hipblasLtMatmulPreferenceCreate(&pref), "preference create")) break;
        const size_t max_workspace = 0;
        if (!hipblaslt_ok(hipblasLtMatmulPreferenceSetAttribute(
                                  pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                  &max_workspace, sizeof(max_workspace)),
                          "set max workspace")) break;
        if (!hipblaslt_ok(hipblasLtMatmulAlgoGetHeuristic(g_hipblaslt, desc,
                                                          a_desc, b_desc, c_desc, d_desc,
                                                          pref, 16, heur, &returned),
                          "algo heuristic")) break;
        if (returned <= 0) {
            fprintf(stderr, "ds4: hipBLASLt no algo for %s m=%u n=%u k=%u\n",
                    label ? label : "gemm", out_dim, n_tok, in_dim);
            break;
        }
        ok = 1;
    } while (0);
    if (pref) (void)hipblasLtMatmulPreferenceDestroy(pref);
    if (!ok) {
        if (d_desc) (void)hipblasLtMatrixLayoutDestroy(d_desc);
        if (c_desc) (void)hipblasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) (void)hipblasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) (void)hipblasLtMatrixLayoutDestroy(a_desc);
        if (desc) (void)hipblasLtMatmulDescDestroy(desc);
        return NULL;
    }

    hip_hipblaslt_gemm_plan p;
    p.out_dim = out_dim;
    p.n_tok = n_tok;
    p.in_dim = in_dim;
    p.op_a = op_a;
    p.output_type = output_type;
    p.desc = desc;
    p.a_desc = a_desc;
    p.b_desc = b_desc;
    p.c_desc = c_desc;
    p.d_desc = d_desc;
    memset(&p.algo, 0, sizeof(p.algo));
    p.candidates.assign(heur, heur + returned);
    p.tuned = 0;
    g_hipblaslt_gemm_plans.push_back(p);
    return &g_hipblaslt_gemm_plans.back();
}

static int hipblaslt_gemm_f16_launch(
        hip_hipblaslt_gemm_plan *p,
        void *out,
        const __half *a,
        const __half *b,
        const hipblasLtMatmulAlgo_t *algo) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return hipblasLtMatmul(g_hipblaslt, p->desc, &alpha,
                           a, p->a_desc,
                           b, p->b_desc,
                           &beta,
                           out, p->c_desc,
                           out, p->d_desc,
                           algo,
                           NULL, 0, 0) == HIPBLAS_STATUS_SUCCESS;
}

static int hipblaslt_gemm_plan_tune(
        hip_hipblaslt_gemm_plan *p,
        void *out,
        const __half *a,
        const __half *b,
        const char *label) {
    if (!p || p->candidates.empty()) return 0;
    static const int tune_allowed = [] {
        const char *env = getenv("GUFO_DEEPSEEK_ROCM_HIPBLASLT_TUNE");
        return env == NULL || env[0] != '0';
    }();
    if (!tune_allowed) {
        /* Pre-plan-cache behavior: take the heuristic's first entry as-is. */
        p->algo = p->candidates[0].algo;
        p->tuned = 1;
        return 1;
    }
    // Runtime timing made the near-tied 64x2048x4096 shape alternate between
    // algorithms with different FP accumulation order. Pin the profiled
    // gfx1151 choices so identical inputs produce identical logits.
    /* Candidate 0 is the heuristic's own first choice, which is what this
     * backend used before the plan cache learned to keep the whole candidate
     * list. Profiled picks (4 for most shapes, 5 and 6 for three of the
     * `in_dim == 4096` projections) are faster on gfx1151, but they change the
     * accumulation order of nearly every dense projection, and the retained
     * 128-token trajectory envelope has no margin for that. Keep the
     * heuristic's choice and leave the sweep reachable through the environment.
     */
    size_t gfx1151_preferred_candidate = 0u;
    if (p->n_tok >= DS4_ROCM_WIDE_PREFILL_ROWS) {
        gfx1151_preferred_candidate = 4u;
        if (p->op_a == HIPBLAS_OP_T && p->in_dim == 4096u) {
            if (p->out_dim == 256u) gfx1151_preferred_candidate = 6u;
            else if (p->out_dim == 512u) gfx1151_preferred_candidate = 5u;
        }
    }
    {
        const char *env = getenv("GUFO_DEEPSEEK_ROCM_HIPBLASLT_CANDIDATE");
        if (env != NULL && env[0] != '\0') {
            char *end = NULL;
            const unsigned long value = strtoul(env, &end, 10);
            if (end != env && end != NULL && *end == '\0' && value < 16ul) {
                gfx1151_preferred_candidate = (size_t)value;
            }
        }
    }
    if (g_rocm_gfx1151 &&
        p->candidates.size() > gfx1151_preferred_candidate) {
        const hipblasLtMatmulHeuristicResult_t &candidate =
            p->candidates[gfx1151_preferred_candidate];
        if (candidate.state == HIPBLAS_STATUS_SUCCESS &&
            candidate.workspaceSize == 0u &&
            hipblaslt_gemm_f16_launch(
                p, out, a, b, &candidate.algo)) {
            p->algo = candidate.algo;
            p->tuned = 1;
            fprintf(stderr,
                    "ds4: ROCm hipBLASLt selected fixed %s candidate %zu/%zu "
                    "(opA=%c m=%u n=%u k=%u)\n",
                    label ? label : "gemm",
                    gfx1151_preferred_candidate,
                    p->candidates.size(),
                    p->op_a == HIPBLAS_OP_T ? 'T' : 'N',
                    p->out_dim,
                    p->n_tok,
                    p->in_dim);
            return 1;
        }
    }

    hipEvent_t begin = NULL, end = NULL;
    if (hipEventCreate(&begin) != hipSuccess ||
        hipEventCreate(&end) != hipSuccess) {
        if (begin) (void)hipEventDestroy(begin);
        if (end) (void)hipEventDestroy(end);
        return 0;
    }

    float best_ms = INFINITY;
    int best_index = -1;
    const int tune_iterations = 3;
    for (size_t i = 0; i < p->candidates.size(); i++) {
        const hipblasLtMatmulHeuristicResult_t &candidate = p->candidates[i];
        if (candidate.state != HIPBLAS_STATUS_SUCCESS ||
            candidate.workspaceSize != 0u ||
            !hipblaslt_gemm_f16_launch(p, out, a, b, &candidate.algo)) {
            continue;
        }
        if (hipEventRecord(begin, 0) != hipSuccess) continue;
        int launched = 1;
        for (int it = 0; it < tune_iterations; it++) {
            if (!hipblaslt_gemm_f16_launch(p, out, a, b, &candidate.algo)) {
                launched = 0;
                break;
            }
        }
        if (!launched ||
            hipEventRecord(end, 0) != hipSuccess ||
            hipEventSynchronize(end) != hipSuccess) {
            continue;
        }
        float elapsed = 0.0f;
        if (hipEventElapsedTime(&elapsed, begin, end) != hipSuccess) continue;
        const float mean_ms = elapsed / (float)tune_iterations;
        if (mean_ms < best_ms) {
            best_ms = mean_ms;
            best_index = (int)i;
        }
    }
    (void)hipEventDestroy(end);
    (void)hipEventDestroy(begin);
    if (best_index < 0) return 0;

    p->algo = p->candidates[(size_t)best_index].algo;
    p->tuned = 1;
    fprintf(stderr,
            "ds4: ROCm hipBLASLt selected %s candidate %d/%zu "
            "(opA=%c m=%u n=%u k=%u, %.3f ms)\n",
            label ? label : "gemm",
            best_index,
            p->candidates.size(),
            p->op_a == HIPBLAS_OP_T ? 'T' : 'N',
            p->out_dim,
            p->n_tok,
            p->in_dim,
            best_ms);
    return 1;
}

/* Opt-in: send projections that ship on hipBLAS through hipBLASLt instead.
 *
 * Worth 363.9 to 410.5 tok/s at a 4,096-token prompt, and it is the only
 * accelerated prefill route that the retained envelope rejects. Width scoping
 * does not save it: the pinned trajectory's own prompt prefill is wide, and
 * swapping libraries for those projections is a coarser perturbation than
 * swapping algorithms within one, so the trajectory lands at 114/128 with rank
 * sum 150 against a 116/142 floor. Every other route below, MMQ included, keeps
 * the trajectory at 116/128 rank sum 142. Enable with
 * GUFO_DEEPSEEK_ROCM_HIPBLASLT_ROUTING=1. */
enum {
    DS4_ROCM_LT_ROUTE_Q8_F32 = 1u,   /* dense Q8 projection, F32 result */
    DS4_ROCM_LT_ROUTE_Q8_F16 = 2u,   /* dense Q8 projection, F16 result */
    DS4_ROCM_LT_ROUTE_F16    = 4u,   /* F16-weight projection */
    DS4_ROCM_LT_ROUTE_ATTN_B = 8u,   /* attention output B fallback */
    DS4_ROCM_LT_ROUTE_F16_PAIR = 16u /* paired F16 projection */
};

/* Which projections move from hipBLAS to hipBLASLt, as a bitmask.
 *
 * hipBLASLt picks a different Tensile kernel for the same shape, worth up to
 * 363.9 -> 410.5 tok/s at a 4,096-token prompt, but it is the one accelerated
 * prefill route the retained envelope can reject: width scoping does not save
 * it, because the pinned trajectory's own prompt prefill is wide. The default
 * keeps the subset that leaves the trajectory at 116/128 rank sum 142; set
 * GUFO_DEEPSEEK_ROCM_HIPBLASLT_ROUTING to a mask (31 for all of them) to
 * explore, or 0 to disable. */
static int hipblaslt_route_mask(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = DS4_ROCM_LT_ROUTE_DEFAULT_MASK;
        const char *env = getenv("GUFO_DEEPSEEK_ROCM_HIPBLASLT_ROUTING");
        if (env != NULL && env[0] != '\0') {
            char *end = NULL;
            const unsigned long value = strtoul(env, &end, 10);
            if (end != env && end != NULL && *end == '\0' && value <= 31ul) {
                cached = (int)value;
            }
        }
    }
    return cached;
}

static int hipblaslt_route_enabled(unsigned int which) {
    return (hipblaslt_route_mask() & (int)which) != 0;
}

static int hipblaslt_gemm_f16(
        void *out,
        const __half *a,
        const __half *b,
        uint32_t out_dim,
        uint32_t n_tok,
        uint32_t in_dim,
        hipblasOperation_t op_a,
        hipDataType output_type,
        const char *label) {
    if (!g_hipblaslt_ready || !out || !a || !b ||
        out_dim == 0 || n_tok == 0 || in_dim == 0) return 0;
    hip_hipblaslt_gemm_plan *p = hipblaslt_gemm_plan_get(
        out_dim, n_tok, in_dim, op_a, output_type, label);
    if (!p) return 0;
    if (!p->tuned && !hipblaslt_gemm_plan_tune(p, out, a, b, label)) return 0;
    return hipblaslt_gemm_f16_launch(p, out, a, b, &p->algo);
}

static int hipblaslt_gemm_tn_f16_out_f16(
        __half *out,
        const __half *w_rowmajor_out_in,
        const __half *x_rowmajor_tok_in,
        uint32_t out_dim,
        uint32_t n_tok,
        uint32_t in_dim,
        const char *label) {
    return hipblaslt_gemm_f16(out,
                              w_rowmajor_out_in,
                              x_rowmajor_tok_in,
                              out_dim,
                              n_tok,
                              in_dim,
                              HIPBLAS_OP_T,
                              HIP_R_16F,
                              label);
}
