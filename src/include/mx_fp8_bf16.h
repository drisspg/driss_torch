#pragma once

#include <torch/torch.h>

namespace driss_torch {

at::Tensor mx_fp8_bf16(at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale,
                       at::Tensor w_scale);
} // namespace driss_torch
