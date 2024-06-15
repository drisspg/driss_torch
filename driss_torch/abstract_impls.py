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
