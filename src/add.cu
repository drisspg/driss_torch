#include "add.h"
#include "utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/python.h>

namespace driss_torch {

using namespace at;

namespace {
template <class T>
__global__ void add_one_kernel(const T *const input, T *const output,
                               const int64_t N) {
  // Grid-strided loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    output[i] = input[i] + 1;
  }
}
} // namespace

Tensor add_one(const Tensor &input) {
  auto output = torch::zeros_like(input);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "add_one_cuda", [&]() {
    const auto block_size = 128;
    const auto num_blocks =
        std::min(65535L, ceil_div(input.numel(), block_size));
    add_one_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), input.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return output;
}

} // namespace driss_torch
