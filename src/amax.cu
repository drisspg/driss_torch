#include "include/saturated_cast.h"
#include "utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

namespace driss_torch {
using namespace at;

namespace {

} // namespace


Tensor amax(const Tensor &input){
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kFloat,
              "Input tensor must be of type BFloat16 or Float, but got ",
              input.dtype());
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D, but got ", input.dim());
    auto amax_output = torch::empty({1}, input.options().dtype(at::kFloat));


    return amax_output;
}

} // namespace driss_torch
