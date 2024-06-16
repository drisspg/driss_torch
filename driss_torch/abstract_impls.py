import torch
from torch.library import register_fake


@register_fake("DrissTorch::saturated_cast")
def saturated_cast_meta(
    x: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
    transpose: bool = False,
):
    return torch.empty_like(x, dtype=out_dtype)


@register_fake("DrissTorch::amax")
def amax_meta(
    x: torch.Tensor,
):
    return torch.empty(1, dtype=torch.float32, device=x.device)
