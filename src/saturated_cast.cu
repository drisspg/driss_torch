#include "include/saturated_cast.h"
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
__global__ void
saturated_cast_kernel_single(nv_bfloat16 *input, __nv_fp8_storage_t *output,
                             int n_rows, int n_cols,
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

__global__ void
saturated_cast_kernel_double(nv_bfloat162 *input, __nv_fp8x2_storage_t *output,
                             int n_rows, int n_cols,
                             __nv_fp8_interpretation_t out_dtype) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Assume row major
  const int global_index = row * n_cols + col;
  if (row < n_rows && col < n_cols) {
    output[global_index] = __nv_cvt_bfloat16raw2_to_fp8x2(
        input[global_index], __nv_saturation_t::__NV_SATFINITE, out_dtype);
  }
}

__global__ void
saturated_cast_kernel_flat(nv_bfloat162 *input, __nv_fp8x2_storage_t *output,
                             int64_t numel,
                             __nv_fp8_interpretation_t out_dtype) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index <numel) {
    output[global_index] = __nv_cvt_bfloat16raw2_to_fp8x2(
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

void dispatch_best_kernel(const Tensor &input, const Tensor &output,
                          __nv_fp8_interpretation_t out_dtype, bool transpose) {
  const int n_rows = input.size(0);
  const int n_cols = input.size(1);
  const int block_size_x = 32;
  const int block_size_y = 32;
  const dim3 block(block_size_x, block_size_y);
  const dim3 grid(ceil_div(n_cols, block_size_x), ceil_div(n_rows, block_size_y));
  if (!transpose){
    const auto numel = input.numel();
    if (numel%2 == 0){
      const dim3 block(256);
      const dim3 grid(ceil_div(numel, block.x));
      saturated_cast_kernel_flat<<<grid, block>>>(
        static_cast<nv_bfloat162 *>(input.data_ptr()),
        static_cast<__nv_fp8x2_storage_t *>(output.data_ptr()), numel/2, out_dtype);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }

  if (n_cols % 2 == 0) {
    // We cast to a 2x8 type, so we need to divide the number of columns by 2
    saturated_cast_kernel_double<<<grid, block>>>(
        static_cast<nv_bfloat162 *>(input.data_ptr()),
        static_cast<__nv_fp8x2_storage_t *>(output.data_ptr()), n_rows,
        n_cols / 2, out_dtype);
  } else {
    saturated_cast_kernel_single<<<grid, block>>>(
        static_cast<nv_bfloat16 *>(input.data_ptr()),
        static_cast<__nv_fp8_storage_t *>(output.data_ptr()), n_rows, n_cols,
        out_dtype);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

Tensor saturated_cast(const Tensor &input, ScalarType dtype, const c10::optional<Tensor>& attn_mask, bool transpose) {
  TORCH_CHECK(dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e5m2,
              "Output tensor must be of type Float8_e4m3fn or Float8_e5m2")
  auto output = torch::empty(input.sizes(), input.options().dtype(dtype));

  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "Input tensor must be of type BFloat16");
  dispatch_best_kernel(input, output, dtype_map(dtype), transpose);
  return output;
}

} // namespace driss_torch
