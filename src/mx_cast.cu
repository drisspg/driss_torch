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

// CUTLASS includes
#include <cute/tensor.hpp>
#include <cutlass/util/device_memory.h>

#include <cmath>

namespace driss_torch_kernels {

using namespace cute;

__device__ __forceinline__ float roundToPowerOf2(float v) {
  if (v <= 0)
    return (v == 0) ? 1.0f : 0.0f;

  int exp;
  float mantissa = frexpf(v, &exp);

  // If mantissa is exactly 0.5, we're already at a power of 2
  if (mantissa == 0.5f) {
    return v;
  }

  // Check if we should round up or down
  if (mantissa > 0.5f) {
    exp++; // Round up
  }

  // Return 2^exp
  return ldexpf(1.0f, exp - 1);
}

__device__ __forceinline__ float
get_fp8_max_pow_2(__nv_fp8_interpretation_t fp8_dtype) {
  switch (fp8_dtype) {
  case __NV_E4M3:
    return 256.0f;
  case __NV_E5M2:
    return 57344.0f;
  default:
    return 0.0f; // Invalid dtype
  }
}

template <typename inpt_type>
__device__ __forceinline__ __nv_fp8_storage_t
convert_to_fp8(inpt_type scaled_input, __nv_fp8_interpretation_t fp8_dtype) {
  return __nv_cvt_float_to_fp8(scaled_input, __nv_saturation_t::__NV_SATFINITE,
                               fp8_dtype);
}

__device__ __forceinline__ void compute_row_scales(const float abs_val,
                                                   float *row_scales,
                                                   int row_idx, int col_idx) {
  // Use warp shuffle to compute the maximum
  float max_val = abs_val;
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
    max_val = max(max_val, other_val);
  }

  if (col_idx == 0) {
    row_scales[row_idx] = roundToPowerOf2(max_val) / get_fp8_max_pow_2(__NV_E4M3);
  }
}

template <typename Element, int InputSize, int NumRows>
struct SharedMemory {
  Element input_data[sizeof(Element) * InputSize];
  float row_scales[InputSize];
};


template <class Element, class TensorInput, class TensorOutput,
          class TensorScale, class ThrShape, class BlockShape>
__global__ void mx_fp8_quantize_kernel(TensorInput input, TensorOutput output,
                                       TensorScale scale, ThrShape thr_shape, BlockShape blck_shape) {

  // Slice the tensors to obtain a view into each tile.
  Tensor tile_input = input(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor tile_output = output(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor tile_scale = scale(make_coord(_, _), blockIdx.x, blockIdx.y);

  // Define shared memory
  auto smem_shape = thr_shape;
  constexpr auto num_rows = size<0>(thr_shape);
  auto smem_layout = make_layout(smem_shape, LayoutRight());

  __shared__ SharedMemory<Element, size(smem_shape), num_rows> smem;

  using SmemLayout = decltype(smem_layout);
  using SmemArray = array_aligned<Element, cosize_v<SmemLayout>>;
  SmemArray &input_smem = *reinterpret_cast<SmemArray *>(smem.input_data);

  // Create a tensor view into shared memory with the 32x32 shape
  auto smem_ptr = make_smem_ptr(input_smem.data());
  auto smem_tensor = make_tensor(smem_ptr, smem_layout);

  auto smem_tiled_input = tiled_divide(tile_input, smem_shape);
  auto sub_tiled_output = tiled_divide(tile_output, smem_shape);
  auto sub_tiled_scale = tiled_divide(tile_scale, smem_shape);

  auto row_idx = threadIdx.y;
  auto col_idx = threadIdx.x;

  #pragma unroll
  for (auto i{0}; i < get<0>(blck_shape)/num_rows; i++) {
    auto input_slice = smem_tiled_input(make_coord(_, _), i, 0);
    auto scale_slice = sub_tiled_scale(make_coord(_, _), i, 0);
    int32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    cooperative_copy<size(thr_shape)>(tid, input_slice, smem_tensor);
    __syncthreads();
    auto abs =
        std::abs(static_cast<float>(smem_tensor(row_idx, col_idx)));
    // Compute column max and store in col_maxes
    compute_row_scales(abs, smem.row_scales, row_idx, col_idx);
    __syncthreads();

    auto scale = smem.row_scales[row_idx];
    auto inverse_scale = 1 / smem.row_scales[row_idx];
    auto scaled = static_cast<float>(smem_tensor(row_idx, col_idx)) * inverse_scale;
    auto out = convert_to_fp8(scaled, __NV_E4M3);
    auto output_slice = sub_tiled_output(make_coord(_, _), i, 0);
    output_slice(row_idx, col_idx) = out;
    if (col_idx == 0) {
      auto converted = __nv_cvt_float_to_e8m0(scale, __NV_SATFINITE, cudaRoundMode::cudaRoundPosInf);
      scale_slice(row_idx, 0) = converted;
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
      size<2>(tiled_tensor_input)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size<1>(thr_shape), size<0>(thr_shape));

  //
  // Launch the kernel
  //
  driss_torch_kernels::mx_fp8_quantize_kernel<__nv_bfloat16>
      <<<gridDim, blockDim>>>(tiled_tensor_input, tiled_tensor_ouput,
                              tiled_tensor_scale, thr_shape, block_shape);
  // tiled_copy);

  // Check for CUDA errors
  C10_CUDA_CHECK(cudaGetLastError());

  return {output, scale};
}

} // namespace driss_torch
