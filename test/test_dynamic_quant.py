import pytest
import torch
from driss_torch import dynamic_scaled_quant
from torchao.float8.float8_utils import tensor_to_scale



def eager_scaled_quant(
    a: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
):
    """Quantize tensor to fp8 using a delayed scaled and calculate abs_max

    Args:
        a: Input tensor to quantize
        scale: Scale to apply to input tensor, calculated from previous abs_max
        fp8_dtype: FP8 datatype to quantize to
    """
    out = a * scale
    out = torch.where(out > torch.finfo(fp8_dtype).max, torch.finfo(fp8_dtype).max, out)
    out = torch.where(
        out < -1 * torch.finfo(fp8_dtype).max, -1 * torch.finfo(fp8_dtype).max, out
    )
    return out.to(fp8_dtype)


@pytest.mark.parametrize("num_rows", [3, 64, 512, 4096])
@pytest.mark.parametrize("num_cols", [7, 17, 127, 512, 3212, 4097])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("in_dtype", [torch.float32])
def test_cast(
    num_rows: int, num_cols: int, in_dtype: torch.dtype, fp8_dtype: torch.dtype
):
    a = torch.rand(num_rows, num_cols, dtype=in_dtype, device="cuda")
    scale = tensor_to_scale(a, fp8_dtype)

    cast_pytorch = eager_scaled_quant(a, scale, fp8_dtype)

    cast_custom = dynamic_scaled_quant(a, fp8_dtype)

    custom_fp32 = cast_custom.to(torch.float32)
    pytorch_fp32 = cast_pytorch.to(torch.float32)
    torch.testing.assert_close(custom_fp32, pytorch_fp32)
