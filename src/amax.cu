#include "include/amax.h"
#include "utils.h"

#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace driss_torch {
using namespace at;

namespace {

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old;
  old = !signbit(value)
            ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

using namespace cooperative_groups;

__device__ float reduce_amax(thread_group g, float *temp, float val) {
  int lane = g.thread_rank();

  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (int i = g.size() / 2; i > 0; i /= 2) {
    temp[lane] = val;
    g.sync(); // wait for all threads to store
    if (lane < i) {
      // I dont think we need another max;
      val = max(val, temp[lane + i]);
    }
    g.sync(); // wait for all threads to load
  }
  return val; // note: only thread 0 will return full sum
}

template <int coarse_factor, typename PackedType>
__device__ float thread_amax(PackedType *input, size_t n) {
  float amax = -std::numeric_limits<float>::infinity();

  int gIdx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = gIdx; i < n / 2 /*pack_width*/; i += blockDim.x * gridDim.x) {
    auto slice2 = input[i];
    if constexpr (std::is_same_v<PackedType, nv_bfloat162>) {
      auto abs = __habs2(slice2);
      auto local_max = __hmax(abs.x, abs.y);
      amax = max(amax, __bfloat162float(local_max));
    } else {
      amax = max(amax, abs(slice2.x));
      amax = max(amax, abs(slice2.y));
    }
  }
  return amax;
}

template <int coarse_factor, typename PackedType, typename UnpackedType>
__global__ void amax_kernel(float *amax, PackedType *input, int64_t numel) {
  auto g = this_thread_block();
  float my_amax = thread_amax<coarse_factor, PackedType>(input, numel);
  extern __shared__ float temp[];
  float block_max = reduce_amax(g, temp, my_amax);

  if (g.thread_rank() == 0) {
    atomicMaxFloat(amax, block_max);
  }
}

void launch_kernel(const Tensor &input, Tensor &output) {
  constexpr int blockSize = 256;
  constexpr int coarse_factor = 8;
  constexpr int pack_width = 2;
  const auto numel = input.numel();
  int nBlocks = ceil_div(numel, blockSize * coarse_factor * pack_width);
  int sharedBytes = blockSize * sizeof(float);

  if (input.scalar_type() == at::kFloat) {
    amax_kernel<coarse_factor, float2, float>
        <<<nBlocks, blockSize, sharedBytes>>>(
            static_cast<float *>(output.data_ptr()),
            static_cast<float2 *>(input.data_ptr()), numel);
  } else {
    amax_kernel<coarse_factor, nv_bfloat162, nv_bfloat16>
        <<<nBlocks, blockSize, sharedBytes>>>(
            static_cast<float *>(output.data_ptr()),
            static_cast<nv_bfloat162 *>(input.data_ptr()), numel);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

Tensor amax(const Tensor &input) {
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kFloat,
              "Input tensor must be of type BFloat16 or Float, but got ",
              input.dtype());
  TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D, but got ",
              input.dim());
 TORCH_CHECK(input.numel() % 2 == 0, "Only works for even numels!");

  auto amax_output = torch::full({1}, -std::numeric_limits<float>::infinity(),
                                 input.options().dtype(at::kFloat));
  launch_kernel(input, amax_output);
  return amax_output;
}

} // namespace driss_torch
