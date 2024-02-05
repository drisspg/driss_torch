
#pragma once
#include <torch/torch.h>

namespace driss_torch {
at::Tensor add_one(const at::Tensor &input);
} // namespace driss_torch
