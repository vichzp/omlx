# SPDX-License-Identifier: Apache-2.0
"""LiLiCorr block drafter (NVIDIA Nemotron 3.5 Bolt).

LiLiCorr is DFlash2's backbone (grouped dynamic conv around attention and the
MLP, sliding-window attention with sinks, a non-causal block) with the
candidate selector replaced by a small lattice transformer. For each draft
slot it takes the target lm_head's top-k candidates and scores transitions
between candidates of neighbouring slots, with an anchor vector from the last
committed position's projected context scoring the first slot. Proposals walk
the lattice: slot ``s`` picks from row ``logits[s, previous]``.

Ported from vLLM's ``LiLiCorrHead``. The checkpoint (EigenLabs' MLX repack of
NVIDIA's release) stores the large projections as NVFP4 with a per-tensor
global scale, which ``nn.QuantizedLinear`` does not model; they load into
the ModelOpt loader's ``ScaledQuantizedLinear``.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

from ...patches.mlx_lm_mtp import batch_generator as bg
from ...patches.qwen38_modelopt_mixed import ScaledQuantizedLinear
from ...utils.sampling import top_k_indices
from . import selection

# The checkpoint's per-tensor global_scale is the amax form: the linear's
# output scale is global_scale / (448 * 6), applied after the NVFP4 matmul.
_GLOBAL_SCALE_DIV = 448.0 * 6.0


def is_lilicorr_config(config: dict) -> bool:
    return (config.get("dflash_config") or {}).get("projector_type") == "lilicorr"


def _normalize(x: mx.array, eps: float) -> mx.array:
    x32 = x.astype(mx.float32)
    norm = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
    return (x32 / mx.maximum(norm, eps)).astype(x.dtype)


class _MLP(nn.Module):
    """Non-gated SiLU MLP with biases (vLLM's NemotronHMLP as LiLiCorr uses it)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.up_proj = nn.Linear(d_in, d_hidden, bias=True)
        self.down_proj = nn.Linear(d_hidden, d_out, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.up_proj(x)))


class _LatticeAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.in_proj_weight = mx.zeros((3 * hidden, hidden))
        self.in_proj_bias = mx.zeros((3 * hidden,))
        self.out_proj = nn.Linear(hidden, hidden, bias=True)

    def __call__(self, x: mx.array, bias: mx.array) -> mx.array:
        batch, length, hidden = x.shape
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias
        q, k, v = (
            t.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            for t in mx.split(qkv, 3, axis=-1)
        )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=bias.astype(q.dtype)
        )
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, hidden))


class _LatticeLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden, eps=eps)
        self.attn = _LatticeAttention(hidden, heads)
        self.mlp_norm = nn.RMSNorm(hidden, eps=eps)
        self.mlp = _MLP(hidden, int(hidden * mlp_ratio), hidden)

    def __call__(self, x: mx.array, bias: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x), bias)
        return x + self.mlp(self.mlp_norm(x))


class LiLiCorrHead(nn.Module):
    def __init__(self, model_hidden: int, block_size: int, eps: float, cfg: dict):
        super().__init__()
        hidden = int(cfg["lilicorr_hidden_size"] or model_hidden)
        self.block_size = int(block_size)
        self.slots = self.block_size - 1
        self.topk = int(cfg["lilicorr_candidate_topk"])
        self.hidden = hidden
        self.heads = int(cfg["lilicorr_num_heads"])
        self.eps = float(cfg["lilicorr_vector_eps"])
        self.logit_scale = float(cfg["lilicorr_logit_scale"])
        factor = int(cfg["lilicorr_factor_dim"])
        self.token_proj = nn.Linear(model_hidden, hidden, bias=True)
        self.pass_hidden_proj = nn.Linear(model_hidden, hidden, bias=True)
        self.context_proj = nn.Linear(model_hidden, hidden, bias=True)
        self.feature_norm = nn.LayerNorm(5)
        self.feature_mlp = _MLP(5, hidden, hidden)
        self.slot_embedding = mx.zeros((1, 1, self.slots, 1, hidden))
        self.rank_embedding = mx.zeros((1, 1, 1, self.topk, hidden))
        self.relative_slot_bias = mx.zeros((self.heads, 2 * self.block_size - 1))
        self.same_slot_bias = mx.zeros((self.heads,))
        self.layers = [
            _LatticeLayer(hidden, self.heads, float(cfg["lilicorr_mlp_ratio"]), eps)
            for _ in range(int(cfg["lilicorr_num_layers"]))
        ]
        self.output_norm = nn.RMSNorm(hidden, eps=eps)
        self.anchor_norm = nn.RMSNorm(hidden, eps=eps)
        self.factor_input_proj = nn.Linear(3 * hidden, hidden, bias=True)
        self.out_head = nn.Linear(hidden, factor, bias=True)
        self.in_head = nn.Linear(hidden, factor, bias=True)
        self.anchor_out_head = nn.Linear(hidden, factor, bias=True)

    def _attention_bias(self, n_slots: int) -> mx.array:
        slot = mx.repeat(mx.arange(self.slots), self.topk)
        rel = slot[:, None] - slot[None, :] + self.block_size - 1
        bias = self.relative_slot_bias[:, rel]
        same = (slot[:, None] == slot[None, :]).astype(bias.dtype)
        bias = bias + same[None] * self.same_slot_bias[:, None, None]
        n = n_slots * self.topk
        return bias[None, :, :n, :n]

    def __call__(self, token_emb, cand_lp, pass_hidden, anchor_hidden):
        """Transition logits ``(B, S, K_prev, K_next)``; row 0 is the anchor's.

        ``token_emb`` (B, S, K, Hm), ``cand_lp`` (B, S, K) sorted descending,
        ``pass_hidden`` (B, S, Hm), ``anchor_hidden`` (B, Hm).
        """
        batch, slots, k = cand_lp.shape
        dtype = self.slot_embedding.dtype
        tokens = self.token_proj(token_emb.astype(dtype))
        passed = self.pass_hidden_proj(pass_hidden.astype(dtype))[:, :, None, :]
        lp = cand_lp.astype(mx.float32)
        rank = (mx.arange(k, dtype=mx.float32) / max(k - 1, 1))[None, None, :]
        top1 = (mx.arange(k) == 0).astype(mx.float32)[None, None, :]
        features = mx.stack(
            [
                lp,
                mx.exp(lp),
                lp - lp.max(axis=-1, keepdims=True),
                mx.broadcast_to(rank, lp.shape),
                mx.broadcast_to(top1, lp.shape),
            ],
            axis=-1,
        )
        h = (
            tokens
            + passed
            + self.feature_mlp(self.feature_norm(features.astype(tokens.dtype)))
        )
        h = h + self.slot_embedding[:, 0, :slots] + self.rank_embedding[:, 0]
        h = h.reshape(batch, slots * k, self.hidden)
        anchor = self.context_proj(anchor_hidden.astype(dtype))
        bias = self._attention_bias(slots)
        for layer in self.layers:
            h = layer(h, bias)
        h = self.output_norm(h).reshape(batch, slots, k, self.hidden)
        anchor = self.anchor_norm(anchor)
        weight, b = self.factor_input_proj.weight, self.factor_input_proj.bias
        w_self = weight[:, : self.hidden]
        w_anchor = weight[:, self.hidden : 2 * self.hidden]
        w_cross = weight[:, 2 * self.hidden :]
        row = anchor[:, None, None, :]
        factor = nn.silu(h @ w_self.T + b + row @ w_anchor.T + (h * row) @ w_cross.T)
        out_vec = _normalize(self.out_head(factor), self.eps)
        in_vec = _normalize(self.in_head(factor), self.eps)
        anchor_out = _normalize(self.anchor_out_head(anchor), self.eps)
        start = mx.sum(anchor_out[:, None, :] * in_vec[:, 0], axis=-1)
        pairs = out_vec[:, :-1] @ in_vec[:, 1:].transpose(0, 1, 3, 2)
        start = self.logit_scale * start.astype(mx.float32)
        pairs = self.logit_scale * pairs.astype(mx.float32)
        first = mx.broadcast_to(start[:, None, None, :], (batch, 1, k, k))
        return mx.concatenate([first, pairs], axis=1)


class LiLiCorrDraftModel(DFlash2DraftModel):
    """DFlash2 backbone with the LiLiCorr head in place of the selector."""

    def __init__(self, raw: dict):
        dflash = dict(raw["dflash_config"])
        dflash.update(
            selector_rank=1, selector_top_k=int(dflash["lilicorr_candidate_topk"])
        )
        config = DFlash2Config.from_dict(
            {
                **raw,
                "dflash_config": dflash,
                "num_target_layers": 1 + max(dflash["target_layer_ids"]),
            }
        )
        super().__init__(config)
        del self.candidate_selector
        self.lilicorr = LiLiCorrHead(
            config.hidden_size, config.block_size, config.rms_norm_eps, dflash
        )

    def bind(self, target_model, tokenizer=None):
        language = getattr(target_model, "language_model", target_model)
        backbone = getattr(language, "backbone", None)
        if backbone is None or not hasattr(backbone, "embeddings"):
            return super().bind(target_model)
        # Nemotron-H keeps its embeddings under ``backbone``.
        self.embed_tokens = backbone.embeddings
        self.embed_scale = 1.0
        self.lm_head = language.lm_head
        return self


def load_lilicorr(path: str | Path) -> LiLiCorrDraftModel:
    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    model = LiLiCorrDraftModel(raw)
    # The checkpoint lists its NVFP4 projections; the rest stay dense.
    quantized = {
        k for k, v in raw.get("quantization", {}).items() if isinstance(v, dict)
    }
    leaves = tree_flatten(
        model.leaf_modules(), is_leaf=lambda module: isinstance(module, nn.Module)
    )
    model.update_modules(
        tree_unflatten(
            [
                (name, ScaledQuantizedLinear.from_linear(module, "scaled_nvfp4"))
                for name, module in leaves
                if name in quantized
            ]
        )
    )
    # Marks the drafter as already quantized for load_dflash_drafter.
    model.config.quantization = raw.get("quantization")
    weights = mx.load(str(path / "model.safetensors"))
    for key, value in weights.items():
        if key.endswith(".global_scale"):
            weights[key] = value / _GLOBAL_SCALE_DIV
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model


def select(
    model: LiLiCorrDraftModel, draft_hidden, logits, h_ctx, lengths, counts, samplers
) -> list[tuple]:
    """Proposals per row: ``(tokens (1, S), q)``, q empty for greedy rows.

    As in ``selection.select_sampled``, each slot picks among its top-k
    candidates, here from the lattice row of the previous pick. Sampled rows
    draw through their sampler's filters and expose that density as q.
    """
    head = model.lilicorr
    # The anchor is the projected context of each row's newest committed
    # position (a predraft commits ``counts`` rows).
    if counts is None:
        last = mx.array([max(n, 1) - 1 for n in lengths], dtype=mx.int32)
    else:
        last = mx.concatenate(counts).astype(mx.int32) - 1
    anchor_hidden = mx.take_along_axis(h_ctx, last[:, None, None], axis=1)[:, 0]
    lp = logits.astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    candidates = top_k_indices(lp, head.topk)
    cand_lp = mx.take_along_axis(lp, candidates, axis=-1)
    # The head embeds each candidate's rank, so order them best first.
    order = mx.argsort(-cand_lp, axis=-1)
    candidates = mx.take_along_axis(candidates, order, axis=-1).astype(mx.int32)
    cand_lp = mx.take_along_axis(cand_lp, order, axis=-1)
    scores = head(model.embed_tokens(candidates), cand_lp, draft_hidden, anchor_hidden)
    out = []
    for index, sampler in enumerate(samplers):
        previous, picks, densities = 0, [], []
        for slot in range(logits.shape[1]):
            row = scores[index : index + 1, slot, previous]
            if sampler is None:
                pick = mx.argmax(row, axis=-1)
            else:
                pick, density = selection.sample_candidates(row, sampler)
                densities.append(density)
            previous = pick.reshape(())
            picks.append(candidates[index, slot][previous].reshape(1))
        q = (
            bg.SparseDraftQ(
                candidates[index], mx.concatenate(densities), logits.shape[-1]
            )
            if densities
            else []
        )
        out.append((mx.concatenate(picks)[None], q))
    return out
