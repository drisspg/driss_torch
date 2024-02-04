from pathlib import Path

import torch

lib_path = Path(__file__).parent / ".." / "build" / "libdriss_torch.so"
torch.ops.load_library(str(lib_path.resolve()))
torch.ops.load_library(lib_path)


ops = torch.ops.DrissTorch


def list_ops():
    return ops.__dir__()


def add_one(x: torch.Tensor) -> torch.Tensor:
    return ops.add_one(x)
