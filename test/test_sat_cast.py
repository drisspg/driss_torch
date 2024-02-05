import pytest
import torch
from driss_torch import saturated_cast


@pytest.mark.parametrize("num_rows", [64, 512, 4096])
@pytest.mark.parametrize("num_cols", [512, 3212, 4097])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_cast(num_rows: int, num_cols: int, dtype: torch.dtype):
    # This is a bad test since since the cast is not saturating
    # but torch.rand is gaussian(0, 1) so it should be fine and we wont
    # exceed the range but we should do this right.
    a = torch.rand(num_rows, num_cols, dtype=torch.bfloat16, device="cuda")

    cast_pytorch = a.to(dtype)
    cast_custom = saturated_cast(a, dtype)

    torch.testing.assert_close(cast_pytorch, cast_custom)
