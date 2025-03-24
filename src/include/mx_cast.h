#pragma once

#include <torch/torch.h>

namespace driss_torch {


std::tuple<at::Tensor, at::Tensor> mx_fp8_quantize(
    at::Tensor input,
    int64_t block_size = 32,
    int64_t axis = 1,
    bool transpose = false,
    c10::ScalarType fp8_type = at::kFloat8_e4m3fn);


} // namespace driss_torch
