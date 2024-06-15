#pragma once

#include <torch/torch.h>

namespace driss_torch {
at::Tensor amax(const at::Tensor &input);
} // namespace driss_torch
