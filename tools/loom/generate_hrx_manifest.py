#!/usr/bin/env python3
"""Generates the HRX artifact manifest (JSON + C++ header) for compiled Loom kernels."""

import argparse
import hashlib
import json
import os
import sys

KERNEL_METADATA = [
    {
        "name": "qwen_swiglu",
        "filename": "qwen_fused_swiglu_bf16.fb",
        "export_name": "qwen_fused_swiglu_bf16",
        "binding_count": 4,
        "binding_order": ["input", "gate", "up", "output"],
        "scalar_constants": ["row_capacity"],
        "tensor_encoding": "bf16",
        "fixed_dimensions": {"row_capacity": 17408, "hidden": 5120},
        "workgroup_size": [128, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_rmsnorm_qkv",
        "filename": "qwen_fused_rmsnorm_qkv_bf16.fb",
        "export_name": "qwen_fused_rmsnorm_qkv_bf16",
        "binding_count": 4,
        "binding_order": ["input", "gamma", "qkv_weight", "output"],
        "scalar_constants": ["row_capacity"],
        "tensor_encoding": "bf16",
        "fixed_dimensions": {"row_capacity": 10240, "hidden": 5120},
        "workgroup_size": [128, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_rope_kv",
        "filename": "qwen_fused_rope_kv_cache_bf16.fb",
        "export_name": "qwen_fused_rope_kv_cache_bf16",
        "binding_count": 7,
        "binding_order": ["q", "k", "v", "cos", "sin", "k_cache", "v_cache"],
        "scalar_constants": ["max_heads"],
        "tensor_encoding": "bf16",
        "fixed_dimensions": {"max_heads": 24, "head_dim": 256, "rotary_dim": 64},
        "workgroup_size": [128, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_down_residual",
        "filename": "qwen_fused_down_residual_bf16.fb",
        "export_name": "qwen_fused_down_residual_bf16",
        "binding_count": 4,
        "binding_order": ["input", "down_weight", "residual", "output"],
        "scalar_constants": ["hidden_dim"],
        "tensor_encoding": "bf16",
        "fixed_dimensions": {"hidden_dim": 5120},
        "workgroup_size": [128, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_rmsnorm",
        "filename": "qwen_rmsnorm_f32.fb",
        "export_name": "qwen_rmsnorm_f32",
        "binding_count": 3,
        "binding_order": ["input", "gamma", "output"],
        "scalar_constants": [],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"hidden": 5120},
        "workgroup_size": [128, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_residual_add",
        "filename": "qwen_residual_add_f32.fb",
        "export_name": "qwen_residual_add_f32",
        "binding_count": 3,
        "binding_order": ["left", "right", "output"],
        "scalar_constants": ["elements"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"element_capacity": 17408},
        "workgroup_size": [256, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_swiglu_pointwise",
        "filename": "qwen_swiglu_pointwise_f32.fb",
        "export_name": "qwen_swiglu_pointwise_f32",
        "binding_count": 3,
        "binding_order": ["gate", "up", "output"],
        "scalar_constants": ["elements"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"element_capacity": 17408},
        "workgroup_size": [256, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_split_q_gate",
        "filename": "qwen_split_q_gate_f32.fb",
        "export_name": "qwen_split_q_gate_f32",
        "binding_count": 3,
        "binding_order": ["q_gate", "query", "gate"],
        "scalar_constants": [],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"query_dim": 6144, "gate_dim": 6144},
        "workgroup_size": [192, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_copy",
        "filename": "qwen_copy_f32.fb",
        "export_name": "qwen_copy_f32",
        "binding_count": 2,
        "binding_order": ["source", "destination"],
        "scalar_constants": ["elements"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"element_capacity": 50331648},
        "workgroup_size": [256, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_q8_decode_oracle",
        "filename": "qwen_q8_0_decode_oracle.fb",
        "export_name": "qwen_q8_0_decode_oracle",
        "binding_count": 3,
        "binding_order": ["weight", "input", "output"],
        "scalar_constants": ["rows"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"hidden": 5120},
        "workgroup_size": [160, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_q8_embedding",
        "filename": "qwen_q8_0_embedding_k5120.fb",
        "export_name": "qwen_q8_0_embedding_k5120",
        "binding_count": 2,
        "binding_order": ["embedding", "output"],
        "scalar_constants": ["token", "vocab_size"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 5120, "vocab_size": 248320},
        "workgroup_size": [32, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_q8_gemv_k5120",
        "filename": "qwen_q8_0_gemv_k5120.fb",
        "export_name": "qwen_q8_0_gemv_k5120",
        "binding_count": 3,
        "binding_order": ["weight", "input", "output"],
        "scalar_constants": ["rows"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 5120, "row_capacity": 17408},
        "workgroup_size": [160, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_q8_gemv_k6144",
        "filename": "qwen_q8_0_gemv_k6144.fb",
        "export_name": "qwen_q8_0_gemv_k6144",
        "binding_count": 3,
        "binding_order": ["weight", "input", "output"],
        "scalar_constants": ["rows"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 6144, "row_capacity": 5120},
        "workgroup_size": [192, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_q8_gemv_k17408",
        "filename": "qwen_q8_0_gemv_k17408.fb",
        "export_name": "qwen_q8_0_gemv_k17408",
        "binding_count": 3,
        "binding_order": ["weight", "input", "output"],
        "scalar_constants": ["rows"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 17408, "row_capacity": 5120},
        "workgroup_size": [544, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_q8_gemv_k17408_wg256",
        "filename": "qwen_q8_0_gemv_k17408_wg256.fb",
        "export_name": "qwen_q8_0_gemv_k17408_wg256",
        "binding_count": 3,
        "binding_order": ["weight", "input", "output"],
        "scalar_constants": ["rows"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 17408, "row_capacity": 5120},
        "workgroup_size": [256, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_rmsnorm_batch",
        "filename": "qwen_rmsnorm_batch_f32.fb",
        "export_name": "qwen_rmsnorm_batch_f32",
        "binding_count": 3,
        "binding_order": ["input", "gamma", "output"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"hidden": 5120, "token_capacity": 128},
        "workgroup_size": [160, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_residual_add_batch",
        "filename": "qwen_residual_add_batch_f32.fb",
        "export_name": "qwen_residual_add_batch_f32",
        "binding_count": 3,
        "binding_order": ["left", "right", "output"],
        "scalar_constants": ["elements"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"element_capacity": 655360},
        "workgroup_size": [32, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_swiglu_pointwise_batch",
        "filename": "qwen_swiglu_pointwise_batch_f32.fb",
        "export_name": "qwen_swiglu_pointwise_batch_f32",
        "binding_count": 2,
        "binding_order": ["pairs", "output"],
        "scalar_constants": ["elements"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"element_capacity": 2228224, "ffn": 17408},
        "workgroup_size": [32, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_deltanet_prepare_batch",
        "filename": "qwen_deltanet_prepare_batch_f32.fb",
        "export_name": "qwen_deltanet_prepare_batch_f32",
        "binding_count": 3,
        "binding_order": ["prepared", "a", "dt"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"heads": 48, "token_capacity": 128},
        "workgroup_size": [48, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_ssm_conv_batch",
        "filename": "qwen_ssm_conv_silu_batch_f32.fb",
        "export_name": "qwen_ssm_conv_silu_batch_f32",
        "binding_count": 4,
        "binding_order": ["input", "weights", "state", "output"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"channels": 10240, "taps": 4, "token_capacity": 128},
        "workgroup_size": [32, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_deltanet_recurrence_batch",
        "filename": "qwen_deltanet_recurrence_batch_f32.fb",
        "export_name": "qwen_deltanet_recurrence_batch_f32",
        "binding_count": 4,
        "binding_order": ["conv", "prepared", "state", "readout"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"heads": 48, "dim": 128, "token_capacity": 128},
        "workgroup_size": [128, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_deltanet_readout_batch",
        "filename": "qwen_deltanet_readout_batch_f32.fb",
        "export_name": "qwen_deltanet_readout_batch_f32",
        "binding_count": 4,
        "binding_order": ["readout", "norm", "gate", "output"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"heads": 48, "dim": 128, "token_capacity": 128},
        "workgroup_size": [128, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_gemm_i8_blocked_k5120_t128",
        "filename": "qwen_q8_0_gemm_i8_blocked_k5120_t128.fb",
        "export_name": "qwen_q8_0_gemm_i8_blocked_k5120_t128",
        "binding_count": 4,
        "binding_order": ["weight", "act", "scales", "output"],
        "scalar_constants": ["rows", "tokens"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 5120, "row_capacity": 248320, "tokens": 128},
        "workgroup_size": [256, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_activation_quantize_blocked_k5120",
        "filename": "qwen_activation_quantize_blocked_k5120.fb",
        "export_name": "qwen_activation_quantize_blocked_k5120",
        "binding_count": 3,
        "binding_order": ["input", "output", "scales"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"k": 5120, "token_capacity": 128},
        "workgroup_size": [160, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_gemm_i8_blocked_k6144_t128",
        "filename": "qwen_q8_0_gemm_i8_blocked_k6144_t128.fb",
        "export_name": "qwen_q8_0_gemm_i8_blocked_k6144_t128",
        "binding_count": 4,
        "binding_order": ["weight", "act", "scales", "output"],
        "scalar_constants": ["rows", "tokens"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 6144, "row_capacity": 5120, "tokens": 128},
        "workgroup_size": [256, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_activation_quantize_blocked_k6144",
        "filename": "qwen_activation_quantize_blocked_k6144.fb",
        "export_name": "qwen_activation_quantize_blocked_k6144",
        "binding_count": 3,
        "binding_order": ["input", "output", "scales"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"k": 6144, "token_capacity": 128},
        "workgroup_size": [192, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_gemm_i8_blocked_k17408_t128",
        "filename": "qwen_q8_0_gemm_i8_blocked_k17408_t128.fb",
        "export_name": "qwen_q8_0_gemm_i8_blocked_k17408_t128",
        "binding_count": 4,
        "binding_order": ["weight", "act", "scales", "output"],
        "scalar_constants": ["rows", "tokens"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 17408, "row_capacity": 5120, "tokens": 128},
        "workgroup_size": [256, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_activation_quantize_blocked_k17408",
        "filename": "qwen_activation_quantize_blocked_k17408.fb",
        "export_name": "qwen_activation_quantize_blocked_k17408",
        "binding_count": 3,
        "binding_order": ["input", "output", "scales"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"k": 17408, "token_capacity": 128},
        "workgroup_size": [544, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_gemm_i8_k5120_t8",
        "filename": "qwen_q8_0_gemm_i8_k5120_t8.fb",
        "export_name": "qwen_q8_0_gemm_i8_k5120_t8",
        "binding_count": 4,
        "binding_order": ["weight", "act", "scales", "output"],
        "scalar_constants": ["rows", "tokens"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 5120, "row_capacity": 248320, "tokens": 8},
        "workgroup_size": [160, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_activation_quantize_k5120",
        "filename": "qwen_activation_quantize_k5120.fb",
        "export_name": "qwen_activation_quantize_k5120",
        "binding_count": 3,
        "binding_order": ["input", "output", "scales"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"k": 5120, "token_capacity": 8},
        "workgroup_size": [160, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_gemm_i8_wmma_k5120_t8",
        "filename": "qwen_q8_0_gemm_i8_wmma_k5120_t8.fb",
        "export_name": "qwen_q8_0_gemm_i8_wmma_k5120_t8",
        "binding_count": 4,
        "binding_order": ["weight", "act", "scales", "output"],
        "scalar_constants": ["rows", "tokens"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {
            "k": 5120,
            "row_capacity": 248320,
            "logical_tokens": 8,
            "physical_tokens": 16,
        },
        "workgroup_size": [32, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_activation_quantize_wmma_k5120",
        "filename": "qwen_activation_quantize_wmma_k5120.fb",
        "export_name": "qwen_activation_quantize_wmma_k5120",
        "binding_count": 3,
        "binding_order": ["input", "output", "scales"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {
            "k": 5120,
            "logical_tokens": 8,
            "physical_tokens": 16,
            "scale_layout": "block-major",
        },
        "workgroup_size": [160, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_gemm_i8_k6144_t8",
        "filename": "qwen_q8_0_gemm_i8_k6144_t8.fb",
        "export_name": "qwen_q8_0_gemm_i8_k6144_t8",
        "binding_count": 4,
        "binding_order": ["weight", "act", "scales", "output"],
        "scalar_constants": ["rows", "tokens"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 6144, "row_capacity": 5120, "tokens": 8},
        "workgroup_size": [192, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_activation_quantize_k6144",
        "filename": "qwen_activation_quantize_k6144.fb",
        "export_name": "qwen_activation_quantize_k6144",
        "binding_count": 3,
        "binding_order": ["input", "output", "scales"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"k": 6144, "token_capacity": 8},
        "workgroup_size": [192, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_gemm_i8_k17408_t8",
        "filename": "qwen_q8_0_gemm_i8_k17408_t8.fb",
        "export_name": "qwen_q8_0_gemm_i8_k17408_t8",
        "binding_count": 4,
        "binding_order": ["weight", "act", "scales", "output"],
        "scalar_constants": ["rows", "tokens"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 17408, "row_capacity": 5120, "tokens": 8},
        "workgroup_size": [544, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_activation_quantize_k17408",
        "filename": "qwen_activation_quantize_k17408.fb",
        "export_name": "qwen_activation_quantize_k17408",
        "binding_count": 3,
        "binding_order": ["input", "output", "scales"],
        "scalar_constants": ["tokens"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"k": 17408, "token_capacity": 8},
        "workgroup_size": [544, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_gemm_k5120_t8",
        "filename": "qwen_q8_0_gemm_k5120_t8.fb",
        "export_name": "qwen_q8_0_gemm_k5120_t8",
        "binding_count": 3,
        "binding_order": ["weight", "input", "output"],
        "scalar_constants": ["rows", "tokens"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 5120, "row_capacity": 248320, "tokens": 8},
        "workgroup_size": [160, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_gemm_k6144_t8",
        "filename": "qwen_q8_0_gemm_k6144_t8.fb",
        "export_name": "qwen_q8_0_gemm_k6144_t8",
        "binding_count": 3,
        "binding_order": ["weight", "input", "output"],
        "scalar_constants": ["rows", "tokens"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 6144, "row_capacity": 5120, "tokens": 8},
        "workgroup_size": [192, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_gemm_k17408_t8",
        "filename": "qwen_q8_0_gemm_k17408_t8.fb",
        "export_name": "qwen_q8_0_gemm_k17408_t8",
        "binding_count": 3,
        "binding_order": ["weight", "input", "output"],
        "scalar_constants": ["rows", "tokens"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 17408, "row_capacity": 5120, "tokens": 8},
        "workgroup_size": [544, 1, 1],
        "subgroup_size": 32,
        "optional": True,
    },
    {
        "name": "qwen_q8_vocab_gemv_k5120",
        "filename": "qwen_q8_0_vocab_gemv_k5120.fb",
        "export_name": "qwen_q8_0_vocab_gemv_k5120",
        "binding_count": 3,
        "binding_order": ["weight", "input", "output"],
        "scalar_constants": ["rows"],
        "tensor_encoding": "q8_0",
        "fixed_dimensions": {"k": 5120, "row_capacity": 248320},
        "workgroup_size": [160, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_per_head_rmsnorm",
        "filename": "qwen_per_head_rmsnorm_f32.fb",
        "export_name": "qwen_per_head_rmsnorm_f32",
        "binding_count": 3,
        "binding_order": ["input", "gamma", "output"],
        "scalar_constants": ["heads"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"max_heads": 24, "head_dim": 256},
        "workgroup_size": [128, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_attention_decode",
        "filename": "qwen_attention_decode_f32.fb",
        "export_name": "qwen_attention_decode_f32",
        "binding_count": 5,
        "binding_order": ["query", "gate", "key_cache", "value_cache", "output"],
        "scalar_constants": ["position"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"query_heads": 24, "kv_heads": 4, "head_dim": 256, "max_context": 131072},
        "workgroup_size": [192, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_ssm_conv",
        "filename": "qwen_ssm_conv_silu_f32.fb",
        "export_name": "qwen_ssm_conv_silu_f32",
        "binding_count": 4,
        "binding_order": ["input", "weights", "state", "output"],
        "scalar_constants": [],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"qkv_dim": 10240, "conv_kernel": 4},
        "workgroup_size": [256, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_deltanet_prepare",
        "filename": "qwen_deltanet_prepare_f32.fb",
        "export_name": "qwen_deltanet_prepare_f32",
        "binding_count": 4,
        "binding_order": ["alpha", "beta", "a", "dt"],
        "scalar_constants": [],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"num_heads": 48},
        "workgroup_size": [48, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_deltanet_recurrence",
        "filename": "qwen_deltanet_recurrence_f32.fb",
        "export_name": "qwen_deltanet_recurrence_f32",
        "binding_count": 7,
        "binding_order": ["conv", "alpha_decay", "beta_correction", "norm", "gate", "state", "output"],
        "scalar_constants": [],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"num_heads": 48, "head_dim": 128},
        "workgroup_size": [128, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
    {
        "name": "qwen_argmax",
        "filename": "qwen_argmax_f32.fb",
        "export_name": "qwen_argmax_f32",
        "binding_count": 2,
        "binding_order": ["logits", "token"],
        "scalar_constants": ["elements"],
        "tensor_encoding": "f32",
        "fixed_dimensions": {"elements": 248320},
        "workgroup_size": [1, 1, 1],
        "subgroup_size": 32,
        "optional": False,
    },
]


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate HRX Loom artifact manifest")
    parser.add_argument("--kernels-dir", required=True, help="Path to directory containing .fb files")
    parser.add_argument("--output-json", required=True, help="Path to write hrx_manifest.json")
    parser.add_argument("--output-header", required=True, help="Path to write C++ manifest header")
    args = parser.parse_args()

    kernels_dir = args.kernels_dir.strip('"\'')
    output_json = args.output_json.strip('"\'')
    output_header = args.output_header.strip('"\'')

    entries = []
    for meta in KERNEL_METADATA:
        fb_path = os.path.join(kernels_dir, meta["filename"])
        entry = dict(meta)
        if os.path.exists(fb_path):
            entry["sha256"] = compute_sha256(fb_path)
            entry["present"] = True
        else:
            entry["sha256"] = ""
            entry["present"] = False
            if not meta["optional"]:
                print(f"Warning: required artifact {meta['filename']} missing in {kernels_dir}", file=sys.stderr)
        entries.append(entry)

    manifest_data = {
        "schema_version": "1.0.0",
        "model_kind": "qwen3.8-27b",
        "target": "gfx1151",
        "wave_size": 32,
        "hrx_abi_revision": "hrx-loom-v1",
        "entries": entries,
    }

    print(f"Generated HRX manifest JSON: {output_json} ({len(entries)} entries)")
    print(f"Generated HRX manifest header: {output_header}")

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
        f.write("\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_header)), exist_ok=True)
    with open(output_header, "w", encoding="utf-8") as f:
        f.write("// Generated by generate_hrx_manifest.py. Do not edit.\n")
        f.write("#ifndef GUFO_GENERATED_HRX_MANIFEST_HPP_\n")
        f.write("#define GUFO_GENERATED_HRX_MANIFEST_HPP_\n\n")
        f.write("#include <array>\n#include <cstdint>\n#include <string_view>\n\n")
        f.write("namespace gufo::hrx::generated {\n\n")
        f.write('inline constexpr std::string_view kSchemaVersion = "1.0.0";\n')
        f.write('inline constexpr std::string_view kModelKind = "qwen3.8-27b";\n')
        f.write('inline constexpr std::string_view kTarget = "gfx1151";\n')
        f.write("inline constexpr std::uint32_t kWaveSize = 32;\n")
        f.write('inline constexpr std::string_view kHrxAbiRevision = "hrx-loom-v1";\n\n')
        f.write("struct StaticArtifactManifestEntry {\n")
        f.write("  std::string_view name;\n")
        f.write("  std::string_view filename;\n")
        f.write("  std::string_view export_name;\n")
        f.write("  std::uint32_t binding_count;\n")
        f.write("  std::string_view tensor_encoding;\n")
        f.write("  std::uint32_t workgroup_size_x;\n")
        f.write("  std::uint32_t workgroup_size_y;\n")
        f.write("  std::uint32_t workgroup_size_z;\n")
        f.write("  std::uint32_t subgroup_size;\n")
        f.write("  std::string_view sha256;\n")
        f.write("  bool optional;\n")
        f.write("};\n\n")
        f.write(f"inline constexpr std::array<StaticArtifactManifestEntry, {len(entries)}> kStaticManifestEntries{{\n")
        for e in entries:
            wg = e["workgroup_size"]
            opt = "true" if e["optional"] else "false"
            f.write(f'    StaticArtifactManifestEntry{{"{e["name"]}", "{e["filename"]}", "{e["export_name"]}", {e["binding_count"]}, "{e["tensor_encoding"]}", {wg[0]}, {wg[1]}, {wg[2]}, {e["subgroup_size"]}, "{e["sha256"]}", {opt}}},\n')
        f.write("};\n\n")
        f.write("}  // namespace gufo::hrx::generated\n\n")
        f.write("#endif  // GUFO_GENERATED_HRX_MANIFEST_HPP_\n")


if __name__ == "__main__":
    main()
