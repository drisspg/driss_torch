import torch
from torchao.float8.inference import (
    addmm_float8_unwrapped_inference,
    preprocess_data,
    Float8MMConfig,
)
from driss_torch import sweep_mm
import pytest


def get_scales(A_fp8, B_fp8):
    a_scale = torch.ones((A_fp8.size(0), 1), device="cuda", dtype=torch.float32)
    b_scale = torch.ones((B_fp8.size(1), 1), device="cuda", dtype=torch.float32).T
    return a_scale, b_scale


@pytest.mark.parametrize("swizzle", [1, 2, 4, 8, 16])
def test_functional(swizzle, M=1024, K=16384 * 8, N=16384):
    device = "cuda"
    use_fast_accum = True

    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    A_fp8 = A.to(torch.float8_e4m3fn)
    B_fp8 = B.to(torch.float8_e4m3fn)
    A_fp8, B_fp8 = preprocess_data(A_fp8, B_fp8, Float8MMConfig(use_fast_accum=True))

    a_scale, b_scale = get_scales(A, B)

    out_sweep = sweep_mm(
        A_fp8,
        B_fp8,
        a_scale,
        b_scale,
        None,
        torch.bfloat16,
        use_fast_accum,
        1,
        2,
        1,
        True,
        swizzle,  # Using the parametrized swizzle value
    )

    out_mm = addmm_float8_unwrapped_inference(
        A_fp8, a_scale, B_fp8, b_scale, torch.bfloat16, use_fast_accum=True
    )

    torch.testing.assert_close(out_sweep, out_mm)
