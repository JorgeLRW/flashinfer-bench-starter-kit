"""
FlashInfer fused_moe Triton Kernel — Seed Implementation
=========================================================
Track:  moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Model:  DeepSeek-V3 / DeepSeek-R1

This is the **seed / initial program** for OpenEvolve.  It implements the full
fused MoE operation in plain PyTorch (correct but slow).  OpenEvolve will
iteratively evolve this into high-performance Triton kernels.

The function signature uses Destination Passing Style (DPS):
    kernel(input1, …, inputN, output)  — writes result into pre-allocated output.

Geometry (constants for this definition):
    H  = 7168   (hidden_size)
    I  = 2048   (intermediate_size)
    E  = 256    (num_experts, global)
    EL = 32     (num_local_experts)
    BLOCK = 128 (quantisation block size)
    TOP_K = 8, N_GROUP = 8, TOPK_GROUP = 4  (routing)

Optimization targets (for the LLM to focus on):
    • Replace the per-expert Python loop with fused Triton kernels
    • Fuse dequantisation into the GEMMs
    • Exploit FP8 tensor-core instructions (tl.dot on fp8 operands)
    • Use tiling / shared-memory staging for the two large GEMMs
    • Vectorise the routing / top-k selection
"""

import torch
import triton
import triton.language as tl


# ── Fixed geometry (def) ─────────────────────────────────────────────
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
    # ─── Everything below is the evolution target. ───────────────────────
    # The function signature above MUST stay fixed (DPS contract).

    T = routing_logits.shape[0]
    device = hidden_states.device

    # ────────────────────────────────────────────────────────────────────
    # 1) FP8 block-scale dequantisation
    # ────────────────────────────────────────────────────────────────────

    # Hidden states: [T, H], scale: [H/128, T]  (transposed block layout)
    A_fp32 = hidden_states.to(torch.float32)                          # [T, H]
    A_scale = hidden_states_scale.to(torch.float32)                   # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()                   # [T, H/128]
    A_scale_expanded = (
        A_scale_TH
        .unsqueeze(-1)
        .expand(T, H // BLOCK, BLOCK)
        .reshape(T, H)
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded                                     # [T, H]

    # W13 (gate+up): [EL, 2I, H], scale: [EL, 2I/128, H/128]
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13 = torch.repeat_interleave(S13, BLOCK, dim=1)                  # [EL, 2I, H/128]
    S13 = torch.repeat_interleave(S13, BLOCK, dim=2)                  # [EL, 2I, H]
    W13 = W13_fp32 * S13                                              # [EL, 2I, H]

    # W2 (down): [EL, H, I], scale: [EL, H/128, I/128]
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2 = torch.repeat_interleave(S2, BLOCK, dim=1)                    # [EL, H, I/128]
    S2 = torch.repeat_interleave(S2, BLOCK, dim=2)                    # [EL, H, I]
    W2 = W2_fp32 * S2                                                 # [EL, H, I]

    # ^^ block-scale quantization for GEMMs in fp32

    # ────────────────────────────────────────────────────────────────────
    # 2) DeepSeek-V3 no-aux routing
    # ────────────────────────────────────────────────────────────────────

    # **risky simply because we compute assuming bias exists** maybe handle if routing_bias is None
    logits = routing_logits.to(torch.float32)                         # [T, E]
    bias = routing_bias.to(torch.float32).reshape(-1)                 # [E]

    # Sigmoid activations (consider logits + bias for blending)
    s = torch.sigmoid(logits)                                         # [T, E]
    s_with_bias = s + bias                                            # [T, E]

    # ^^ maps each expert logit to an independent score according to the sigmoid (per deepseek)

    # Group-level selection (8 groups of 32 experts each)
    # note: all scored by sum, top 4 groups kept, global top 8 selected
    group_size = E_GLOBAL // N_GROUP                                  # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)           # [T, 8, 32]

    # Per-group score = sum of top-2 values
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)                               # [T, 8]

    # Keep top-4 groups
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask
        .unsqueeze(2)
        .expand(T, N_GROUP, group_size)
        .reshape(T, E_GLOBAL)
    )

    # Global top-8 experts within kept groups
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    # Combination weights (normalised sigmoid WITHOUT bias, then scaled)
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M                                                   # [T, E]
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor         # [T, E]

    # ────────────────────────────────────────────────────────────────────
    # 3) Per-expert compute: GEMM1 → SwiGLU → GEMM2, weighted accumulate
    # ────────────────────────────────────────────────────────────────────

    result = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_LOCAL):
        ge = local_start + le
        if ge < 0 or ge >= E_GLOBAL:
            continue

        # Tokens that selected this expert
        sel_mask = (topk_idx == ge).any(dim=1)                        # [T] bool
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

        # Gather
        # memcpy heavy, materializes full w matrices
        A_e = A.index_select(0, token_idx)                            # [Tk, H]
        W13_e = W13[le]                                               # [2I, H]
        W2_e = W2[le]                                                 # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] → [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())

        # SwiGLU
        X1 = G1[:, :I]                                                # gate
        X2 = G1[:, I:]                                                # up
        silu_X2 = X2 * torch.sigmoid(X2)
        C = silu_X2 * X1                                              # [Tk, I]

        # GEMM2: [Tk, I] @ [I, H] → [Tk, H]
        O = C.matmul(W2_e.t())

        # Weighted accumulate
        w_tok = weights.index_select(0, token_idx)[:, ge].unsqueeze(1)
        result.index_add_(0, token_idx, O * w_tok)

    # Write to DPS output
    output.copy_(result.to(torch.bfloat16))

    # EVOLVE-BLOCK-END
