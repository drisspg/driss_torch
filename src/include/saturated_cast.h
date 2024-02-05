#pragma once

#include <torch/torch.h>

using namespace at;

Tensor saturated_cast(const Tensor &input, ScalarType dtype);
