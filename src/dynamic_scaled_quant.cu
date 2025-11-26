/**
 * @file dynamic_fp8_cast.cu
 * @brief coop grid based fp8 dynamic cast kernel
 */

#include "include/dynamic_scaled_quant.h"
#include "utils.h"

#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <cooperative_groups.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace driss_torch {
using namespace at;
namespace {
namespace cg = cooperative_groups;
/*
Calculate the max value in a block of int_type values
*/
template <typename inpt_type, int BLOCK_SIZE>
__device__ __forceinline__ inpt_type block_max(inpt_type const thread_data) {
  using BlockReduce =
      cub::BlockReduce<inpt_type, BLOCK_SIZE,
                       cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Reduce(
      thread_data, ::cuda::maximum<inpt_type>{});
}

/*
Given the fp8_dtype enum return the maximum possible value for that type
*/
__device__ __forceinline__ float
get_fp8_max(__nv_fp8_interpretation_t fp8_dtype) {
  switch (fp8_dtype) {
  case __NV_E4M3:
    return 448.0f;
  case __NV_E5M2:
    return 57344.0f;
  default:
    return 0.0f; // Invalid dtype
  }
}

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

__nv_fp8_interpretation_t at_to_cuda(ScalarType fp8_type) {
  switch (fp8_type) {
  case ScalarType::Float8_e4m3fn:
    return __NV_E4M3;
  case ScalarType::Float8_e5m2:
    return __NV_E5M2;
  default:
    TORCH_CHECK(false, "no sir! That aint an fp8!");
  }
}

template <typename inpt_type>
__device__ __forceinline__ __nv_fp8_storage_t
convert_to_fp8(inpt_type scaled_input, __nv_fp8_interpretation_t fp8_dtype) {
  return __nv_cvt_float_to_fp8(scaled_input, __nv_saturation_t::__NV_SATFINITE,
                               fp8_dtype);
}

template <typename inpt_type, int BLOCK_SIZE>
__global__ void
dynamic_scaled_quant(inpt_type const *input, __nv_fp8_storage_t *output,
                     inpt_type *global_amax, const int64_t numel,
                     __nv_fp8_interpretation_t fp8_dtype) {

  // Get grid and block level groups
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  const auto local_idx = cg::thread_rank(block);
  const auto stride = gridDim.x * blockDim.x;
  inpt_type local_max{0};

  // First grid-strided loop accumulating amaxs
  for (auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;
       global_idx < numel; global_idx += stride) {
    auto thread_val = std::abs(input[global_idx]);
    local_max = std::max(thread_val, local_max);
  }
  inpt_type temp_max;
  temp_max = block_max<inpt_type, BLOCK_SIZE>(std::abs(local_max));

  block.sync();
  if (local_idx == 0) {
    atomicMaxFloat(global_amax, temp_max);
  }
  grid.sync();

  auto scale = get_fp8_max(fp8_dtype) / (global_amax[0] + 1e-12);
  // Second grid-strided loop for quantization
  for (auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;
       global_idx < numel; global_idx += stride) {
    auto scaled_input = scale * input[global_idx];
    output[global_idx] = convert_to_fp8(scaled_input, fp8_dtype);
  }
}
void launch_kernel(const Tensor &input, Tensor &output, Tensor &global_amax,
                   ScalarType fp8_dtype) {
  constexpr int blockSize = 512;
  auto numel = input.numel();

  int numSMs;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

  int numBlocksPerSM;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, dynamic_scaled_quant<float, blockSize>, blockSize, 0));

  auto numBlocks = std::min(ceil_div(numel, blockSize),
                            static_cast<int64_t>(numBlocksPerSM * numSMs));

  auto input_ptr = input.data_ptr<float>();
  auto output_ptr = static_cast<__nv_fp8_storage_t *>(output.data_ptr());
  auto amax_ptr = global_amax.data_ptr<float>();

  auto cuda_fp8_type = at_to_cuda(fp8_dtype);

  void *kernel_args[] = {&input_ptr, &output_ptr, &amax_ptr, &numel,
                         &cuda_fp8_type};

  auto stream = at::cuda::getCurrentCUDAStream();

  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(dynamic_scaled_quant<float, blockSize>),
      numBlocks, blockSize, kernel_args,
      0, // no shared memory
      stream));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

Tensor dynamic_scaled_quant(const Tensor &input, ScalarType fp8_dtype) {
  TORCH_CHECK(fp8_dtype == at::kFloat8_e4m3fn || fp8_dtype == at::kFloat8_e5m2,
              "Output tensor must be of type Float8_e4m3fn or Float8_e5m2")

  TORCH_CHECK(input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kFloat,
              "Input tensor must be of type BFloat16 or Float, but got ",
              input.dtype());
  TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D, but got ",
              input.dim());
  TORCH_CHECK(input.is_contiguous());

  auto casted_output =
      torch::empty_like(input, input.options().dtype(fp8_dtype));

  auto global_amax = at::zeros({1}, input.options());

  launch_kernel(input, casted_output, global_amax, fp8_dtype);
  return casted_output;
}

} // namespace driss_torch
