import torch
from driss_torch import add_one


def test_add_one():
    shape = (3, 3, 3)
    a = torch.randint(0, 10, shape, dtype=torch.float).cuda()

    torch.testing.assert_close(add_one(a), a + 1)
