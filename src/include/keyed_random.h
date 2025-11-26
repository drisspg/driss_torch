#pragma once
#include <torch/types.h>

namespace driss_torch {

torch::Tensor keyed_random_uniform(torch::Tensor keys, int64_t cols);

}  // namespace driss_torch
