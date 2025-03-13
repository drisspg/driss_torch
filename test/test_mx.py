import pytest
import torch
from torchao.prototype.mx_formats.mx_tensor import MXTensor, to_mx, ScaleCalculationMode
from transformer_nuggets.mx.to_blocked import to_blocked
from driss_torch import mx_fp8_bf16, mx_fp4_bf16, mx_fp8_quantize
from torchao.float8.float8_utils import compute_error


def run_matrix_test(M: int, K: int, N: int, format: str = "fp8") -> float:
    """Run matrix multiplication test with given dimensions and format."""
    dtype = torch.bfloat16
    device = torch.device("cuda")

    a = torch.rand((M, K), dtype=dtype, device=device)
    b = torch.rand((N, K), dtype=dtype, device=device)

    fmt = torch.float8_e4m3fn if format == "fp8" else "fp4_e2m1"
    mx_func = mx_fp8_bf16 if format == "fp8" else mx_fp4_bf16

    a_mx = MXTensor.to_mx(a, fmt, 32)
    b_mx = MXTensor.to_mx(b, fmt, 32)

    a_data = a_mx._data
    b_data = b_mx._data
    assert b_data.is_contiguous()
    b_data = b_data.transpose(-1, -2)

    a_scale = a_mx._scale_e8m0.view(M, K // 32)
    b_scale = b_mx._scale_e8m0.view(N, K // 32)

    a_scale_block = to_blocked(a_scale)
    b_scale_block = to_blocked(b_scale)

    out_hp = a_mx.to_dtype(torch.bfloat16) @ b_mx.to_dtype(torch.bfloat16).transpose(
        -1, -2
    )
    out = mx_func(a_data, b_data, a_scale_block, b_scale_block)

    return compute_error(out_hp, out).item()


@pytest.mark.parametrize(
    "size",
    [
        (128, 128, 128),
        (256, 256, 256),
        (384, 384, 384),  # Small
        (512, 512, 512),
        (768, 768, 768),  # Medium
        (1024, 1024, 1024),
        (8192, 8192, 8192),  # Large
        (128, 256, 384),
        (256, 384, 512),  # Non-square
        (129, 256, 384),
        (133, 512, 528),  # Non-aligned
    ],
    ids=lambda x: f"{x[0]}x{x[1]}x{x[2]}",
)
@pytest.mark.parametrize("format", ["fp8", "fp4"])
def test_matrix_multiplication(size, format):
    """Test matrix multiplication with various dimensions and formats."""
    M, K, N = size
    sqnr = run_matrix_test(M, K, N, format)
    threshold = 80
    assert (
        sqnr > threshold
    ), f"{format} SQNR {sqnr} below threshold for dims {M}x{K}x{N}"


@pytest.mark.parametrize(
    "size",
    [
        (128, 32),
        (256, 32),
        (8192, 32),
        (128, 64),
        (128, 128),
        (128, 8192),
        (256, 64),
        (256, 128),
        (8192, 8192),
    ],
    ids=lambda x: f"{x[0]}x{x[1]}",
)
def test_mx_cast(size):
    """Test MXTensor cast."""
    dtype = torch.bfloat16
    device = torch.device("cuda")
    a = torch.rand(size, dtype=dtype, device=device)
    ref_scale, ref_data = to_mx(a, torch.float8_e4m3fn, 32, ScaleCalculationMode.CEIL)
    data, scale = mx_fp8_quantize(a)

    torch.testing.assert_close(data, ref_data)
    torch.testing.assert_close(scale, ref_scale.view(torch.float8_e8m0fnu))
