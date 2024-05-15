"""Defines a subclass that can be backed by unified memory:  https://developer.nvidia.com/blog/unified-memory-cuda-beginners/"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch


aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
UM_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    """Use this decorator to implement a function for an aten op in __torch_dispatch__"""

    def decorator(func):
        for op in aten_ops:
            UM_OPS_TABLE[op] = func
        return func

    return decorator


@dataclass
class SubclassTensorArgs:
    original_shape: torch.Size
    original_strides: Tuple
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool


class UMTensor(torch.Tensor):
    """Unified Memory backed class Tensor Subclass"""

    def __new__(
        cls,
        # Args related for base tensor construction
        tensor_meta: SubclassTensorArgs,
        # Args stored on the instance
        block_size: int,
        n_blocks: int,
        scaler_block_size: int,
        quantized_scalers: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_mean: torch.Tensor,
        quantized_data: torch.Tensor,
        nf4: torch.Tensor,
    ):
        """Create a new NF4Tensor object
        Args:
            tensor_meta: Metadata for the tensor
            block_size: Size of the quantization block
            n_blocks: Number of blocks to cover the full tensor
            scaler_block_size: Block size for the scalar quantization
            quantized_scalers: Quantized scalers data' represented a uint8 tensor
            quantization_factor: Quantization factor, single scalar represented as torch.Tensor
            scaler_mean: Mean of the scalers
            quantized_data: Quantized data represented as uint8 tensor
            nf4: NF4 tensor LUT for the quantization and dequantization

        """

        nf4tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor_meta.original_shape,
            tensor_meta.original_strides,
            tensor_meta.storage_offset,
            dtype=tensor_meta.dtype,
            device=tensor_meta.device,
            requires_grad=tensor_meta.requires_grad,
        )
        return nf4tensor
