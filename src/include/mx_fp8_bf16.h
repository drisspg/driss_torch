#pragma once

#include <torch/torch.h>

namespace driss_torch {

at::Tensor mx_fp8_bf16(at::Tensor a, at::Tensor b, at::Tensor a_scale,
                       at::Tensor b_scale);

 
at::Tensor mx_fp4_bf16(at::Tensor a, at::Tensor b, at::Tensor a_scale,
                       at::Tensor b_scale);           
} // namespace driss_torch
