from pathlib import Path

import torch

lib_path = Path(__file__).parent / ".." / "build" / "libdriss_torch.so"
torch.ops.load_library(str(lib_path.resolve()))
torch.ops.load_library(lib_path)
torch.ops.import_module("driss_torch.abstract_impls")


ops = torch.ops.DrissTorch


def list_ops():
    raise NotImplementedError("This function does not do what I think it should.")
    return ops.__dir__()


def saturated_cast(
    x: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
    transpose: bool = False,
) -> torch.Tensor:
    """This op takes in a tensor and returns the fp8 saturated casted version of it.
    Args;
        x: The input tensor.
        out_dtype: The output data type, must be a float8 dtype.
        scale: An on device tensor, this is expected to be a singleton tensor whose value will be multiplied
            x before casting
        transpose: If true will transpose the input tensor during casting
    Returns:
        The output tensor.
    """
    return ops.saturated_cast(x, scale, out_dtype, transpose)


def saturated_cast_amax(
    x: torch.Tensor,
    amax: torch.Tensor,
    out_dtype: torch.dtype,
    transpose: bool = False,
) -> torch.Tensor:
    """This op takes in a tensor and returns the fp8 saturated casted version of it.
    Args;
        x: The input tensor.
        out_dtype: The output data type, must be a float8 dtype.
        scale: An on device tensor, this is expected to be a singleton tensor whose value will be multiplied
            x before casting
        transpose: If true will transpose the input tensor during casting
    Returns:
        The output tensor.
    """
    return ops.saturated_cast_amax(x, amax, out_dtype, transpose)
