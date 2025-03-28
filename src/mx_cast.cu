#include "cute/pointer.hpp"
#include "include/mx_cast.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// CUTLASS includes
#include <cute/tensor.hpp>
#include <cutlass/util/device_memory.h>

#include <cmath>

namespace driss_torch_kernels {

using namespace cute;

template <typename inpt_type>
__device__ __forceinline__ __nv_fp8_storage_t
convert_to_fp8(inpt_type scaled_input, __nv_fp8_interpretation_t fp8_dtype) {
  return __nv_cvt_float_to_fp8(scaled_input, __nv_saturation_t::__NV_SATFINITE,
                               fp8_dtype);
}

__device__ __forceinline__ float compute_row_scales(const float abs_val, int row_idx, int col_idx, auto num_rows) {
  typedef cub::WarpReduce<int> WarpReduce;  // Changed to int reduction
  __shared__ typename WarpReduce::TempStorage temp_storage[num_rows];

  // Compute log2 and ceiling to next integer immediately
  float log2_val_float = log2f(abs_val);
  int log2_val = (int)ceilf(log2_val_float);  // Round up to next integer

  // Perform reduction using integers
  int max_log2 = WarpReduce(temp_storage[row_idx]).Reduce(log2_val, cub::Max());
  max_log2 = __shfl_sync(0xffffffff, max_log2, 0);

  // For E4M3, subtract log2(256.0) = 8 as integer
  int scale_log2 = max_log2 - 8;

  // Convert back to float only at the end
  return powf(2.0f, (float)scale_log2);
}


template <class Element, class TensorInput, class TensorOutput,
          class TensorScale, class ThrShape, class BlockShape>
__global__ void mx_fp8_quantize_kernel(TensorInput input, TensorOutput output,
                                       TensorScale scale, ThrShape thr_shape, BlockShape blck_shape) {

  auto tid = threadIdx.x + threadIdx.y * blockDim.x;
  // Slice the tensors to obtain a view into each tile.
  Tensor tile_input = input(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor tile_output = output(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor tile_scale = scale(make_coord(_, _), blockIdx.x, blockIdx.y);

  constexpr auto num_rows = size<0>(thr_shape);
  const int num_tiles = get<0>(blck_shape)/num_rows;

  // Tile divide for input
  auto tiled_input_tensor = tiled_divide(tile_input, thr_shape);
  auto tiled_ouput_tenosr = tiled_divide(tile_output, thr_shape);
  auto tiled_scale_tensor_e8m0 = tiled_divide(tile_scale, make_shape(get<0>(thr_shape), 1));

  auto row_idx = threadIdx.y;
  auto col_idx = threadIdx.x;

  #pragma unroll
  for (auto i = 0; i < num_tiles; i++) {

    auto sub_tiled_input = tiled_input_tensor(make_coord(_, _), i, 0);
    auto sub_tile_out = tiled_ouput_tenosr(make_coord(_, _), i, 0);
    auto sub_tiled_scale_e8m0 = tiled_scale_tensor_e8m0(make_coord(_, _), i, 0);

    auto inpt = sub_tiled_input(row_idx, col_idx);

    // Calculate absolute values
    auto abs = std::abs(static_cast<float>(inpt));

    // // Compute row scales
    auto scale = compute_row_scales(abs, row_idx, col_idx, num_rows);
    __syncthreads();

    // // Apply scaling and convert to FP8
    auto inverse_scale = 1 / scale;
    auto scaled = static_cast<float>(inpt) * inverse_scale;
    auto out = convert_to_fp8(scaled, __NV_E4M3);

    // Write output directly to global memory
    sub_tile_out(row_idx, col_idx) = out;

    // Write scale directly to global memory (only one thread per row)
    if (col_idx == 0) {
      auto converted = __nv_cvt_float_to_e8m0(scale, __NV_SATFINITE, cudaRoundMode::cudaRoundPosInf);
      sub_tiled_scale_e8m0(row_idx, 0) = converted;
    }
  }

}

} // namespace driss_torch_kernels

namespace driss_torch {
using namespace cute;

std::tuple<at::Tensor, at::Tensor> mx_fp8_quantize(at::Tensor input,
                                                   int64_t block_size,
                                                   int64_t axis, bool transpose,
                                                   c10::ScalarType fp8_type) {

  TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kBFloat16,
              "Input tensor must be float, half, or bfloat16");
  TORCH_CHECK(block_size > 0 && block_size <= 32,
              "Block size must be positive and <= 32");
  TORCH_CHECK(input.dim() == 2,
              "Input tensor must be 2D for CUTLASS implementation");
  TORCH_CHECK(fp8_type == at::kFloat8_e4m3fn || fp8_type == at::kFloat8_e5m2,
              "FP8 type must be Float8_e4m3fn or Float8_e5m2");

  // Get tensor dimensions
  auto input_shape = input.sizes();
  auto input_strides = input.strides();

  TORCH_CHECK(input.dim() == 2, "Only support 2d tensor for now");
  int64_t m = input_shape[0];
  int64_t n = input_shape[1];
  auto num_k_blocks = cutlass::ceil_div(n, 32);
  auto total_blocks = m * num_k_blocks;

  // Create output tensors
  auto output = at::empty_like(input, input.options().dtype(fp8_type));
  auto scale =
      at::empty({total_blocks}, input.options().dtype(at::kFloat8_e8m0fnu));

  auto tensor_input_shape = make_shape(m, n);
  auto tensor_scale_shape = make_shape(m, num_k_blocks);

  auto input_ptr = static_cast<__nv_bfloat16 *>(input.data_ptr());
  auto scale_ptr = static_cast<__nv_fp8_storage_t *>(scale.data_ptr());
  auto output_ptr = static_cast<__nv_fp8_storage_t *>(output.data_ptr());

  Tensor tensor_input =
      make_tensor(make_gmem_ptr(input_ptr), make_layout(tensor_input_shape, LayoutRight()));
  Tensor tensor_scale =
      make_tensor(make_gmem_ptr(scale_ptr), make_layout(tensor_scale_shape, LayoutRight()));
  Tensor tensor_ouput =
      make_tensor(make_gmem_ptr(output_ptr), make_layout(tensor_input_shape, LayoutRight()));

  // Keep it easy for now
  auto block_shape = make_shape(Int<128>{}, Int<32>{});
  auto scale_shape = make_shape(get<0>(block_shape), Int<1>{});
  TORCH_CHECK(evenly_divides(tensor_input_shape, block_shape),
              "Need block shape to evenly divide the input tensor for now");


  // These will be used to determine the CUDA kernel grid dimensions.
  Tensor tiled_tensor_input = tiled_divide(tensor_input, block_shape);
  Tensor tiled_tensor_ouput = tiled_divide(tensor_ouput, block_shape);
  Tensor tiled_tensor_scale = tiled_divide(tensor_scale, scale_shape);

  // Thread arrangement
  auto thr_shape = make_shape(Int<8>{}, Int<32>{});
  dim3 gridDim(
      size<1>(tiled_tensor_input),
      size<2>(tiled_tensor_input));
  dim3 blockDim(size<1>(thr_shape), size<0>(thr_shape));

  //
  // Launch the kernel
  //
  driss_torch_kernels::mx_fp8_quantize_kernel<__nv_bfloat16>
      <<<gridDim, blockDim>>>(tiled_tensor_input, tiled_tensor_ouput,
                              tiled_tensor_scale, thr_shape, block_shape);

  // Check for CUDA errors
  C10_CUDA_CHECK(cudaGetLastError());

  return {output, scale};
}

} // namespace driss_torch
