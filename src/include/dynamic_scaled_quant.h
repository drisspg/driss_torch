#pragma once

#include <torch/torch.h>

namespace driss_torch {

at::Tensor dynamic_scaled_quant(const at::Tensor &input,
                                at::ScalarType fp8_dtype);
} // namespace driss_torch
