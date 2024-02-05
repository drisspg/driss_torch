#include "saturated_cast.h"
#include "utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace driss_torch {
using namespace at;

namespace {
__global__ void saturated_cast_kernel(nv_bfloat16 *input,
                                      __nv_fp8_storage_t *output, int n_rows,
                                      int n_cols,
                                      __nv_fp8_interpretation_t out_dtype) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Assume row major
  const int global_index = row * n_cols + col;
  if (row < n_rows && col < n_cols) {
    output[global_index] = __nv_cvt_bfloat16raw_to_fp8(
        input[global_index], __nv_saturation_t::__NV_SATFINITE, out_dtype);
  }
}

__nv_fp8_interpretation_t dtype_map(const ScalarType dtype) {
  switch (dtype) {
  case at::kFloat8_e4m3fn:
    return __nv_fp8_interpretation_t::__NV_E4M3;
  case at::kFloat8_e5m2:
    return __nv_fp8_interpretation_t::__NV_E5M2;
  default:
    TORCH_CHECK(false, "Invalid dtype");
  }
}
} // namespace

Tensor saturated_cast(const Tensor &input, ScalarType dtype) {
  TORCH_CHECK(dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e5m2,
              "Output tensor must be of type Float8_e4m3fn or Float8_e5m2")
  auto output = torch::empty(input.sizes(), input.options().dtype(dtype));
  const int n_rows = input.size(0);
  const int n_cols = input.size(1);

  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "Input tensor must be of type BFloat16");

  const int block_size = 32;
  const dim3 grid(ceil_div(n_cols, block_size), ceil_div(n_rows, block_size));
  const dim3 block(block_size, block_size);
  saturated_cast_kernel<<<grid, block>>>(
      static_cast<nv_bfloat16 *>(input.data_ptr()),
      static_cast<__nv_fp8_storage_t *>(output.data_ptr()), n_rows, n_cols,
      dtype_map(dtype));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace driss_torch
