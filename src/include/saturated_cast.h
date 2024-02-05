#pragma once

#include <torch/torch.h>

namespace driss_torch {

at::Tensor saturated_cast(const at::Tensor &input, at::ScalarType dtype);
} // namespace driss_torch
