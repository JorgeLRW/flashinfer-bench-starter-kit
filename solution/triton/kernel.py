"""
FlashInfer fused_moe â€” ORIGINAL SEED BASELINE (unmodified)
==========================================================
Track:  moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Model:  DeepSeek-V3 / DeepSeek-R1

This is the UNMODIFIED FlashInfer starter-kit seed kernel.
Used as the control group / baseline for the A/B comparison:
  A) OpenEvolve evolving FROM this baseline
  B) Human-optimized kernel (kernel.py) starting from this baseline

Pure PyTorch reference â€” correct but slow.
"""

import torch
import triton
import triton.language as tl


# â”€â”€ Fixed geometry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    output,                # [T, 7168]      bfloat16  (DPS â€” write here)
):
    """FP8 block-scale fused MoE with DeepSeek-V3 no-aux routing (DPS)."""

    # EVOLVE-BLOCK-START
    T = routing_logits.shape[0]
    device = hidden_states.device
    torch.backends.cuda.matmul.allow_tf32 = True

    # 1) FP8 block-scale dequantisation of hidden_states (fp32 for accuracy)
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
    A = (A_fp32 * A_scale_expanded).contiguous()  # [T, H] fp32

    # 2) DeepSeek-V3 no-aux routing (in-place sigmoid to save memory)
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    logits.sigmoid_()
    s = logits
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
    weights = (weights / weights_sum) * routed_scaling_factor  # [T, 256] fp32

    # 3) Per-expert compute: GEMM1 → SwiGLU → GEMM2, weighted accumulate
    #    Weight dequant happens per-expert to stay within GPU memory.
    result = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    # Group tokens by local expert to avoid O(T*E_LOCAL) scans
    flat_tok = torch.arange(T, device=device, dtype=torch.int64).repeat_interleave(TOP_K)
    flat_ge = topk_idx.reshape(-1)
    mask = (flat_ge >= local_start) & (flat_ge < local_start + E_LOCAL)
    if mask.any():
        ge_l = (flat_ge[mask] - local_start).to(torch.int64)  # [N_assigned]
        tok_l = flat_tok[mask]                                 # [N_assigned]
        order = torch.argsort(ge_l)
        ge_sorted = ge_l[order]
        tok_sorted = tok_l[order]
        counts = torch.bincount(ge_sorted, minlength=E_LOCAL)
        offsets = torch.zeros(E_LOCAL + 1, device=device, dtype=torch.int64)
        offsets[1:] = torch.cumsum(counts, dim=0)
    else:
        tok_sorted = torch.empty((0,), device=device, dtype=torch.int64)
        offsets = torch.zeros(E_LOCAL + 1, device=device, dtype=torch.int64)

    # Iterate only non-empty local experts to reduce Python overhead
    nonempty = torch.nonzero(offsets[1:] > offsets[:-1], as_tuple=False).squeeze(1).tolist()
    for le in nonempty:
        start = int(offsets[le].item())
        end = int(offsets[le + 1].item())
        token_idx = tok_sorted[start:end]

        A_e = A.index_select(0, token_idx).contiguous()  # [N_tok_e, H] fp32

        # GEMM1 dequant with blockwise broadcast and matmul (fp32 for accuracy)
        w13_fp32 = gemm1_weights[le].to(torch.float32).view(32, BLOCK, 56, BLOCK)   # [32,128,56,128]
        s13 = gemm1_weights_scale[le].to(torch.float32).view(32, 1, 56, 1)          # [32,1,56,1]
        G1 = torch.nn.functional.linear(A_e, (w13_fp32 * s13).view(4096, 7168).contiguous())

        # SwiGLU
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        C = torch.nn.functional.silu(X2) * X1  # [N_tok_e, I] fp32

        # Pre-scale activations by routing weight before GEMM2
        w_tok = weights.index_select(0, token_idx)[:, local_start + le].unsqueeze(1)  # [N_tok_e,1] fp32
        C = C * w_tok

        # GEMM2 dequant with blockwise broadcast and matmul
        w2_fp32 = gemm2_weights[le].to(torch.float32).view(56, BLOCK, 16, BLOCK)    # [56,128,16,128]
        s2 = gemm2_weights_scale[le].to(torch.float32).view(56, 1, 16, 1)           # [56,1,16,1]
        O = torch.nn.functional.linear(C, (w2_fp32 * s2).view(7168, 2048).contiguous())  # [N_tok_e, H] fp32

        result.index_add_(0, token_idx, O)

    output.copy_(result.to(torch.bfloat16))
    # EVOLVE-BLOCK-END

