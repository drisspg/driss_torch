import pytest
import torch
from driss_torch import amax


def _eager_amax(a):
    return torch.abs(a).max().to(torch.float32).expand(1)


@pytest.mark.parametrize("num_rows", [4, 64, 512, 4096])
@pytest.mark.parametrize("num_cols", [7, 17, 127, 512, 3212, 4097])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float32])
def test_cast(num_rows: int, num_cols: int, in_dtype: torch.dtype):
    a = torch.randn(num_rows, num_cols, dtype=in_dtype, device="cuda")
    custom_amax = amax(a)
    eager_amax = _eager_amax(a)
    print(f"{custom_amax=}")
    print(f"{eager_amax=}")
    torch.testing.assert_close(custom_amax, eager_amax)
