#pragma once

#include <torch/torch.h>

namespace driss_torch {

at::Tensor saturated_cast(const at::Tensor &input, at::ScalarType dtype, const at::Tensor& attn_mask, bool transpose);
at::Tensor saturated_cast_meta(const at::Tensor &input, at::ScalarType dtype, const at::Tensor& attn_mask, bool transpose);
} // namespace driss_torch
