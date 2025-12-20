# extended_attention_qwen.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import os
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

# ---------- helpers ----------
def _coerce_timestep_to_int(t: Optional[Union[int, float, torch.Tensor]]) -> Optional[int]:
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return int(t.reshape(-1)[0].item())
    return int(t)

def _normalize_timestep_range(
    rng: Optional[Union[int, Tuple[int, int], List[int]]]
) -> Optional[Tuple[int, int]]:
    if rng is None:
        return None
    if isinstance(rng, int):
        # first rng steps: [0, rng)
        return (0, rng)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return (int(rng[0]), int(rng[1]))
    raise TypeError("timestep_range must be None, int, or a (start, end) tuple")

def _pick_pivot_layer_bucket(
    pc: Dict[Any, Any],
    layer_idx: int,
    t_idx: Optional[int],
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Allowed structure:
      A) Plane: {layer_idx: {"hidden": T, "attn": T}, ...}
      B) Temporal Overlap: {timestep: {layer_idx: {"hidden": T, "attn": T}, ...}, ...}
    """
    if not isinstance(pc, dict) or not pc:
        return None

    sample_val = next(iter(pc.values()))
    if isinstance(sample_val, dict) and ("hidden" in sample_val or "attn" in sample_val):
        return pc.get(layer_idx)

    ts_keys = [k for k in pc.keys() if isinstance(k, int)]
    if not ts_keys:
        return None

    if t_idx is not None and t_idx in pc:
        bucket = pc[t_idx]
    elif t_idx is not None:
        nearest = min(ts_keys, key=lambda kk: abs(kk - t_idx))
        bucket = pc[nearest]
    else:
        bucket = pc[min(ts_keys)]

    if isinstance(bucket, dict):
        return bucket.get(layer_idx)
    return None

def resize_flow(flow: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
    assert flow.ndim == 4 and flow.shape[1] == 2, "flow must be [B,2,H,W]"
    B, C, H, W = flow.shape
    if (H, W) == (H_out, W_out):
        return flow
    fr = F.interpolate(flow, size=(H_out, W_out), mode="bilinear", align_corners=True)
    fr[:, 0] *= (W_out / max(W, 1))
    fr[:, 1] *= (H_out / max(H, 1))
    return fr

def backward_warp(latent_hw, flow_t2p):
    """
    latent_hw: [B,C,H,W]  (features to warp)
    flow_t2p : [B,2,H,W]  (target→pivot)  → backward pull: grid = base + flow
    """
    B, C, H, W = latent_hw.shape
    device, dtype = latent_hw.device, latent_hw.dtype
    flow = flow_t2p.to(device=device, dtype=dtype)

    yy = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W)
    xx = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W)

    gx = ((xx.unsqueeze(0) + flow[:, 0]) / max(W - 1, 1)) * 2 - 1
    gy = ((yy.unsqueeze(0) + flow[:, 1]) / max(H - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)  # [B,H,W,2]

    return F.grid_sample(latent_hw, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def fb_consistency_mask(F_t2p, B_p2t, t_abs=0.5, t_rel=0.05):
    """
    F_t2p: [B,2,H,W] (target→pivot)
    B_p2t: [B,2,H,W] (pivot→target), but defined on pivot grid
    t_abs: absolute threshold in px
    t_rel: relative threshold in px
    """
    B, _, H, W = F_t2p.shape
    device = F_t2p.device
    dtype  = F_t2p.dtype

    yy = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W)
    xx = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W)
    x_prime = xx.unsqueeze(0) + F_t2p[:, 0]  # [B,H,W]
    y_prime = yy.unsqueeze(0) + F_t2p[:, 1]  # [B,H,W]
    gx = (x_prime / max(W - 1, 1)) * 2 - 1
    gy = (y_prime / max(H - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)  # [B,H,W,2]

    B_samp = F.grid_sample(B_p2t, grid, mode="bilinear", padding_mode="border", align_corners=True)  # [B,2,H,W]

    err = F_t2p + B_samp
    err_norm = torch.linalg.norm(err, dim=1)  # [B,H,W]
    F_norm = torch.linalg.norm(F_t2p, dim=1)  # [B,H,W]

    thr = torch.maximum(torch.full_like(F_norm, float(t_abs)), t_rel * F_norm)
    mask_hard = (err_norm <= thr).float()

    eps = 1e-8
    soft = torch.exp(- (err_norm / (thr + eps))**2)
    return soft.clamp(0, 1), mask_hard


class QwenDoubleStreamProcessor:
    """
    Extended Attention processor for Qwen Double-Stream:

    - Customize the self-attn of the image stream to 'frame extension (temporal extended K/V)'
    - Keep the text stream safe with the original joint attention calculation (text conditioning stability)
    """
    _attention_backend = None

    def __init__(self, layer_index: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0."
            )
        self.layer_index = layer_index

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # image stream
        encoder_hidden_states: torch.FloatTensor,  # text stream
        encoder_hidden_states_mask: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,

        timestep_index: Optional[int] = None,
        token_replace: bool = False,
        Hs: int = 56,
        Ws: int = 74,
        Ss: int = 2,

        # --- flow args ---
        ref_flows: Optional[List[torch.Tensor]] = None,
        ref_flows_backward: Optional[List[torch.Tensor]] = None,

        # --- NEW: disk cache root ---
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> torch.FloatTensor:

        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        # Compute QKV for image stream (sample projections)
        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

class QwenDoubleStreamExtendedAttnProcessor:
    """
    Extended Attention processor for Qwen Double-Stream:

    - 이미지 스트림의 self-attn을 '프레임 확장(temporal extended K/V)'으로 커스터마이즈.
    - 텍스트 스트림은 원래 조인트 어텐션 계산으로 안전하게 유지(텍스트 conditioning 안정성).
    """
    _attention_backend = None

    def __init__(self, layer_index: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0."
            )
        self.layer_index = layer_index

    # ---------- disk cache helpers ----------
    @staticmethod
    def _pivot_path(cache_dir: str, mode: str, timestep: int, layer: int) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{mode}_t{timestep:04d}_L{layer:02d}.pt")

    @classmethod
    def _save_pivot_pair(cls, cache_dir: str, mode: str, timestep: int, layer: int,
                         tokA: torch.Tensor, tokB: torch.Tensor) -> None:
        """Save two image-stream token blocks (e.g., indices [2] and [3]) to disk."""
        path = cls._pivot_path(cache_dir, mode, timestep, layer)
        torch.save({"A": tokA.detach().cpu(), "B": tokB.detach().cpu()}, path)

    @classmethod
    def _load_pivot_pair(cls, cache_dir: str, mode: str, timestep: int, layer: int,
                         device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        path = cls._pivot_path(cache_dir, mode, timestep, layer)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        obj = torch.load(path, map_location="cpu")
        A = obj["A"].to(device=device, dtype=dtype, non_blocking=True)
        B = obj["B"].to(device=device, dtype=dtype, non_blocking=True)
        return A, B

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # image stream
        encoder_hidden_states: torch.FloatTensor,  # text stream
        encoder_hidden_states_mask: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,

        timestep_index: Optional[int] = None,
        token_replace: bool = False,
        Hs: int = 56,
        Ws: int = 74,
        Ss: int = 2,

        # --- flow args ---
        ref_flows: Optional[List[torch.Tensor]] = None,
        ref_flows_backward: Optional[List[torch.Tensor]] = None,

        # --- NEW: disk cache root ---
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> torch.FloatTensor:

        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")
        
        B, S, D = hidden_states.shape
        seq_txt = encoder_hidden_states.shape[1]
        H = attn.heads  # num heads

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # (B, S, H, Dh) -> (1, B*S, H, Dh) -> repeat(B,...)
        img_key_ext   = img_key.reshape(1, B * S, H, -1).repeat(B, 1, 1, 1)
        img_value_ext = img_value.reshape(1, B * S, H, -1).repeat(B, 1, 1, 1)

        # Concatenate for joint attention
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key_ext], dim=1)
        joint_value = torch.cat([txt_value, img_value_ext], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout
        txt_attn_output = attn.to_add_out(txt_attn_output)

        # ======================================================
        # 2) LOAD pivot tokens from disk and REPLACE (then re-save)
        # ======================================================
        t_idx = _coerce_timestep_to_int(timestep_index)

        if token_replace and (t_idx is not None) and (cache_dir is not None):
            device = hidden_states.device
            dtype  = hidden_states.dtype

            mode = "right"
            batch_idx = [2, 3]
            overlap_idx = [0, 1]

            pa0 = pa1 = None
            pa0, pa1 = self._load_pivot_pair(cache_dir, mode, t_idx, self.layer_index, device, dtype)
            pivot_attn_full = [pa0, pa1]
            pivot_attn_flow = [pa1, pa1]
            if (pivot_attn_flow[0] is not None) and (pivot_attn_flow[1] is not None):
                target_attn_list = []
                # === Prepare optical flow ===
                expected_flows_len = 2
                flows_fwd = ref_flows or [None] * expected_flows_len
                flows_bwd = ref_flows_backward or [None] * expected_flows_len
                flows_fwd += [None] * (expected_flows_len - len(flows_fwd))
                flows_bwd += [None] * (expected_flows_len - len(flows_bwd))

                # flow_tok_f_list, flow_soft_mask_list = [], []
                flow_tok_f_list, flow_hard_mask_list = [], []

                for fwd, bwd in zip(flows_fwd, flows_bwd):
                    flow_f = resize_flow(fwd, Hs, Ws) if fwd is not None else None
                    flow_b = resize_flow(bwd, Hs, Ws) if bwd is not None else None
                    if (flow_f is not None) and (flow_b is not None):
                        soft_mask, hard_mask = fb_consistency_mask(flow_f, flow_b)
                    else:
                        soft_mask = None
                    flow_tok_f_list.append(flow_f)
                    flow_hard_mask_list.append(hard_mask)
                    # flow_soft_mask_list.append(soft_mask)

                # per pivot stream
                for p_idx, pa in enumerate(pivot_attn_flow):

                    pa = pa.unsqueeze(0)  # [1, S_img, D]
                    pivot_chunks = pa.split(Hs * Ws, dim=1)
                    base_chunks  = img_attn_output[batch_idx[p_idx]].unsqueeze(0).split(Hs * Ws, dim=1)

                    warped_chunks = []

                    for c_idx, (ch_pivot, ch_base) in enumerate(zip(pivot_chunks, base_chunks)):

                        flow_f   = flow_tok_f_list[p_idx]
                        hard_mask = flow_hard_mask_list[p_idx]
                        p_hw = ch_pivot.transpose(1, 2).reshape(1, D, Hs, Ws)
                        b_hw = ch_base.transpose(1, 2).reshape(1, D, Hs, Ws)

                        if flow_f is not None:
                            p_warp = backward_warp(p_hw, flow_f)
                            if hard_mask is not None:
                                m = hard_mask.unsqueeze(1).to(p_warp.dtype)
                                p_warp = p_warp * m + b_hw * (1 - m)
                        else:
                            p_warp = b_hw
                        ch_out = p_warp.reshape(1, D, Hs * Ws).transpose(1, 2)
 
                        warped_chunks.append(ch_out)

                    out_tok = torch.cat(warped_chunks, dim=1)
                    target_attn_list.append(out_tok)

                # write back into current attention outputs
                for i, out_tok in enumerate(target_attn_list):
                    img_attn_output[batch_idx[i]] = out_tok.view_as(img_attn_output[batch_idx[i]])
                
            for i, prev_pa in enumerate(pivot_attn_full):
                img_attn_output[overlap_idx[i]] = prev_pa

        # ======================================================
        # 3) Cache update (right token)
        # ======================================================
        if t_idx is not None:
            if cache_dir is not None:
                self._save_pivot_pair(
                    cache_dir, "right", int(t_idx), self.layer_index,
                    img_attn_output[2], img_attn_output[3]
                )

        return img_attn_output, txt_attn_output

def register_extended_attn(transformer, layer_range: tuple = None):
    blocks = transformer.transformer_blocks
    total_layers = len(blocks)
    
    if layer_range is not None:
        start, end = layer_range
        exattn_targets = blocks[start:end+1]
        exattn_range_desc = f"layers {start} to {end}"

        token_targets = blocks[end+1:]
        token_range_desc = f"layers {end+1} to {total_layers-1}"
    else:
        targets = blocks
        range_desc = "all layers"
    
    for i, blk in enumerate(exattn_targets):
        blk.attn.processor = QwenDoubleStreamExtendedAttnProcessor(layer_index=i)
    for i, blk in enumerate(token_targets):
        blk.attn.processor = QwenDoubleStreamProcessor(layer_index=i)

    print("Token all layer registered successfully.")
    print(f"Total layers: {total_layers}, Registered: {len(exattn_targets)} ({exattn_range_desc})")
    print(f"Total layers: {total_layers}, Registered: {len(token_targets)} ({token_range_desc})")
    print(f"Processor: {exattn_targets[0].__class__.__name__} -> {exattn_targets[0].attn.processor.__class__.__name__}")
    print(f"Processor: {token_targets[0].__class__.__name__} -> {token_targets[0].attn.processor.__class__.__name__}")

