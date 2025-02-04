import pytest
import torch
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from transformer_nuggets.mx.to_blocked import to_blocked
from driss_torch import mx_fp8_bf16
from torchao.float8.float8_utils import compute_error


def run_matrix_test(M: int, K: int, N: int) -> float:
    """
    Run matrix multiplication test with given dimensions.

    Args:
        M, K, N: Matrix dimensions
        profile: Whether to enable profiling

    Returns:
        tuple of (cublas_sqnr, cutlass_sqnr)
    """
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Initialize matrices
    a = torch.rand((M, K), dtype=dtype, device=device)
    b = torch.rand((N, K), dtype=dtype, device=device)

    a_mx = MXTensor.to_mx(a, torch.float8_e4m3fn, 32)
    b_mx = MXTensor.to_mx(b, torch.float8_e4m3fn, 32)

    a_fp8 = a_mx._data
    b_fp8 = b_mx._data
    assert b_fp8.is_contiguous()
    b_fp8 = b_fp8.transpose(-1, -2)

    # Get scales
    a_scale_e8 = a_mx._scale_e8m0.view(M, K // 32)
    b_scale_e8 = b_mx._scale_e8m0.view(N, K // 32)

    a_scale_block = to_blocked(a_scale_e8)
    b_scale_block = to_blocked(b_scale_e8)

    # Get reference output
    out_hp = a_mx.to_dtype(torch.bfloat16) @ b_mx.to_dtype(torch.bfloat16).transpose(
        -1, -2
    )

    # Run CUTLASS implementation
    out_e8_fp8 = mx_fp8_bf16(a_fp8, b_fp8, a_scale_block, b_scale_block)

    # Calculate CUTLASS metrics
    cutlass_sqnr = compute_error(out_hp, out_e8_fp8)

    return cutlass_sqnr.item()


@pytest.mark.parametrize(
    "size",
    [
        # Small matrices
        (128, 128, 128),
        (256, 256, 256),
        (384, 384, 384),
        # Medium matrices
        (512, 512, 512),
        (640, 640, 640),
        (768, 768, 768),
        # Large matrices
        (896, 896, 896),
        (1024, 1024, 1024),
        # Very large matrices
        (8192, 8192, 8192),
        # Non-square matrices
        (128, 256, 384),
        (256, 384, 512),
        (384, 512, 640),
        # Non-aligned matrices
        (129, 256, 384),
        (256, 384, 536),
        (133, 512, 528),
    ],
    ids=lambda x: f"{x[0]}x{x[1]}x{x[2]}",
)
def test_matrix_multiplication(size):
    """Test matrix multiplication with various dimensions."""
    M, K, N = size
    cutlass_sqnr = run_matrix_test(M, K, N)
    assert (
        cutlass_sqnr > 20
    ), f"CUTLASS SQNR {cutlass_sqnr} below threshold for dims {M}x{K}x{N}"
