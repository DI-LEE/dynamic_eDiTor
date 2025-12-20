# qwen_blend_from_cache.py
import math
from typing import Dict, Optional, Callable, List, Union, Any
import torch
import torch.nn.functional as F

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    QwenImageEditPipeline,
    calculate_dimensions,
    PipelineImageInput,
)
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput


# ----------------------------- helpers (outside the class) -----------------------------

def compute_temporal_weights_from_distances(
    distances: List[float],
) -> List[torch.Tensor]:
    """
    distances: nonnegative distances per pivot (smaller = more weight).
    Softmax(-beta * d_k) to get normalized weights, then scale by alpha (cap).
    Returns K scalars as 0-D tensors on (device, dtype).
    """
    raw = [1.0 / (d + 1.0) for d in distances]
    S = sum(raw) or 1.0
    weights = [w / S for w in raw]
    return weights

def make_step_selector(selector):
    if selector is None:
        return lambda _: True
    if isinstance(selector, int):
        a, b = 0, int(selector)
        return lambda i, A=a, B=b: A <= i < B
    if (isinstance(selector, (tuple, list)) and len(selector) == 2
        and all(isinstance(x, int) for x in selector)):
        a, b = int(selector[0]), int(selector[1])
        return lambda i, A=a, B=b: A <= i < B
    if isinstance(selector, range):
        a, b, s = selector.start or 0, selector.stop, selector.step or 1
        return lambda i, A=a, B=b, S=s: (A <= i < B) and ((i - A) % S == 0)
    if isinstance(selector, (list, set)):
        S = set(int(x) for x in selector)
        return lambda i, SS=S: i in SS
    raise TypeError("Unsupported selector type")

def _coerce_timestep_range_for_processor(selector):
    if isinstance(selector, int):
        return selector
    if (isinstance(selector, (tuple, list)) and len(selector) == 2
        and all(isinstance(x, int) for x in selector)):
        return (int(selector[0]), int(selector[1]))
    return None 

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None

# --------------------------- main pipeline (inherits Qwen) ----------------------------
def qwenimage_pipeline_ours(
    pipe,
    image,                  
    prompt,                  
    *,
    generator=None,
    num_inference_steps=8,
    height=None,
    width=None,
    token_replace: bool = False,
    ref_flows: Optional[List[torch.Tensor]] = None,     # list of [B,2,H,W], pivot->target
    ref_flows_backward: Optional[List[torch.Tensor]] = None, # list of [B,2,H,W], target->pivot
    prev_right_cache: Optional[Dict[str, torch.Tensor]] = None,
    prev_bottom_cache: Optional[Dict[str, torch.Tensor]] = None,
    is_cache_bottom: bool = False,
    bottom_token_replace: bool = False,
):
    image_arg = image

    right_cache = {}
    bottom_cache = {}
    attention_kwargs = {
        "timestep_index": 0,
        "token_replace": token_replace,
        "ref_flows": ref_flows,
        "ref_flows_backward": ref_flows_backward,
        "Hs": height//pipe.vae_scale_factor//2, # 1184//8//2=74 778//8//2=48
        "Ws": width//pipe.vae_scale_factor//2, # 896//8//2=56 778//8//2=48
        "Ss": 2,
        "right_cache": right_cache,
        "bottom_cache": bottom_cache,
        "prev_right_cache": prev_right_cache,
        "prev_bottom_cache": prev_bottom_cache,
        "is_cache_bottom": is_cache_bottom,
        "bottom_token_replace": bottom_token_replace,
    }

    def on_step_end(pipeline, step: int, timestep, cb_kwargs: dict):
        next_step = step + 1
        pipeline.attention_kwargs["timestep_index"] = next_step
        return {}
    
    out = pipe(
        image=image_arg,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        height=height,
        width=width,
        attention_kwargs=attention_kwargs,
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    return out[0], right_cache, bottom_cache
