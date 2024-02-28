import pytest
import torch
from driss_torch import saturated_cast
from float8_experimental.float8_utils import tensor_to_scale


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
    out = torch.where(out < -1 * torch.finfo(fp8_dtype).max, -1 * torch.finfo(fp8_dtype).max, out)
    return out.to(fp8_dtype)


@pytest.mark.parametrize("num_rows", [3, 64, 512, 4096])
@pytest.mark.parametrize("num_cols", [7, 17, 127, 512, 3212, 4097])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float32])
def test_cast(num_rows: int, num_cols: int, in_dtype: torch.dtype, fp8_dtype: torch.dtype):
    # This is a bad test since since the cast is not saturating
    # but torch.rand is gaussian(0, 1) so it should be fine and we wont
    # exceed the range but we should do this right.
    a = torch.rand(num_rows, num_cols, dtype=in_dtype, device="cuda")
    scale = tensor_to_scale(a, fp8_dtype)

    cast_pytorch = eager_scaled_quant(a, scale, fp8_dtype)
    cast_custom = saturated_cast(a, scale, fp8_dtype)

    custom_fp32 = cast_custom.to(torch.float32)
    pytorch_fp32 = cast_pytorch.to(torch.float32)
    torch.testing.assert_close(custom_fp32, pytorch_fp32)


@pytest.mark.xfail(reason="This test is failing, we need to investigate", strict=True)
def test_cast_edge_bug():
    a = torch.Tensor([0.3223, 0.3223, 0.3223]).to(device="cuda", dtype=torch.bfloat16).view(2, 1)
    scale = torch.Tensor([57344.0]).to("cuda")
    cast_pytorch = eager_scaled_quant(a, scale, torch.float8_e5m2)
    cast_custom = saturated_cast(a, scale, torch.float8_e5m2)

    custom_fp32 = cast_custom.to(torch.float32)
    pytorch_fp32 = cast_pytorch.to(torch.float32)
    MAX_P_output = a.to(torch.float64) * scale.to(torch.float64)
    print("Custom diff is ", torch.max(torch.abs(MAX_P_output - custom_fp32)))
    print("PyTorch diff is ", torch.max(torch.abs(MAX_P_output - pytorch_fp32)))
    # The closest bit pattern is 0|11101|01 = 20480.0
    # Custom is producing 0|11101|00 = 16384.0 which is wrong unless rounding toward zero is set
    torch.testing.assert_close(custom_fp32, pytorch_fp32)
