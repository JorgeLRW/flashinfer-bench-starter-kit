"""
FlashInfer fused_moe — ORIGINAL SEED BASELINE (unmodified)
==========================================================
Track:  moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Model:  DeepSeek-V3 / DeepSeek-R1

This is the UNMODIFIED FlashInfer starter-kit seed kernel.
Used as the control group / baseline for the A/B comparison:
  A) OpenEvolve evolving FROM this baseline
  B) Human-optimized kernel (kernel.py) starting from this baseline

Pure PyTorch reference — correct but slow.
"""

import torch
import triton
import triton.language as tl


# ── Fixed geometry ──────────────────────────────────────────────────────────
H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4


def kernel(
    routing_logits,        # [T, 256]       float32
    routing_bias,          # [256]          bfloat16
    hidden_states,         # [T, 7168]      float8_e4m3fn
    hidden_states_scale,   # [56, T]        float32
    gemm1_weights,         # [32, 4096, 7168] float8_e4m3fn
    gemm1_weights_scale,   # [32, 32, 56]   float32
    gemm2_weights,         # [32, 7168, 2048] float8_e4m3fn
    gemm2_weights_scale,   # [32, 56, 16]   float32
    local_expert_offset,   # int32 scalar
    routed_scaling_factor, # float32 scalar
    output,                # [T, 7168]      bfloat16  (DPS — write here)
):
    """FP8 block-scale fused MoE with DeepSeek-V3 no-aux routing (DPS)."""

    # EVOLVE-BLOCK-START
    T = routing_logits.shape[0]
    device = hidden_states.device

    # 1) FP8 block-scale dequantisation of hidden_states
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)
    A_scale_TH = A_scale.permute(1, 0).contiguous()
    A_scale_expanded = (
        A_scale_TH
        .unsqueeze(-1)
        .expand(T, H // BLOCK, BLOCK)
        .reshape(T, H)
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded

    # NOTE: Weight dequant moved into per-expert loop below to save GPU memory.
    # The original all-at-once dequant uses ~19 GB; per-expert uses ~0.2 GB.

    # 2) DeepSeek-V3 no-aux routing
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    group_size = E_GLOBAL // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask
        .unsqueeze(2)
        .expand(T, N_GROUP, group_size)
        .reshape(T, E_GLOBAL)
    )

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # 3) Per-expert compute: GEMM1 → SwiGLU → GEMM2, weighted accumulate
    #    Weight dequant happens per-expert to stay within GPU memory.
    result = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_LOCAL):
        ge = local_start + le
        if ge < 0 or ge >= E_GLOBAL:
            continue
        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue
        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

        A_e = A.index_select(0, token_idx)

        # Lazy per-expert weight dequant
        w13_fp32 = gemm1_weights[le].to(torch.float32)
        s13 = gemm1_weights_scale[le].to(torch.float32)
        s13 = torch.repeat_interleave(s13, BLOCK, dim=0)
        s13 = torch.repeat_interleave(s13, BLOCK, dim=1)
        W13_e = w13_fp32 * s13

        G1 = A_e.matmul(W13_e.t())

        X1 = G1[:, :I]
        X2 = G1[:, I:]
        silu_X2 = X2 * torch.sigmoid(X2)
        C = silu_X2 * X1

        w2_fp32 = gemm2_weights[le].to(torch.float32)
        s2 = gemm2_weights_scale[le].to(torch.float32)
        s2 = torch.repeat_interleave(s2, BLOCK, dim=0)
        s2 = torch.repeat_interleave(s2, BLOCK, dim=1)
        W2_e = w2_fp32 * s2

        O = C.matmul(W2_e.t())

        w_tok = weights.index_select(0, token_idx)[:, ge].unsqueeze(1)
        result.index_add_(0, token_idx, O * w_tok)

    output.copy_(result.to(torch.bfloat16))
    # EVOLVE-BLOCK-END
