#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <cstdint>
#include <limits>

#include "include/keyed_random.h"

namespace driss_torch {

namespace {

// Helper for converting uint32_t to float in [0, 1)
// Matches the implementation in ATen/native/cuda/Distributions.cu
struct curand_uniform_wrapper {
  curandStatePhilox4_32_10_t &state;
  __device__ curand_uniform_wrapper(curandStatePhilox4_32_10_t &state): state(state) {}
  __device__ float operator()() {
    uint32_t val = curand(&state);
    constexpr auto MASK = static_cast<uint32_t>((static_cast<uint64_t>(1) << std::numeric_limits<float>::digits) - 1);
    constexpr auto DIVISOR = static_cast<float>(1) / (static_cast<uint32_t>(1) << std::numeric_limits<float>::digits);
    return (val & MASK) * DIVISOR;
  }
};

template <typename scalar_t>
__global__ void keyed_random_uniform_kernel(
    scalar_t* output,
    const int64_t* keys,
    int64_t cols,
    int64_t numel) {

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numel) {
    int64_t r = idx / cols;
    int64_t c = idx % cols;

    int64_t key = keys[r];

    curandStatePhilox4_32_10_t state;
    // seed=key, subsequence=c, offset=0
    curand_init(key, c, 0, &state);

    curand_uniform_wrapper uniform_gen(state);
    output[idx] = static_cast<scalar_t>(uniform_gen());
  }
}

} // namespace

torch::Tensor keyed_random_uniform(torch::Tensor keys, int64_t cols) {
  TORCH_CHECK(keys.is_cuda(), "keys must be a CUDA tensor");
  TORCH_CHECK(keys.dim() == 1, "keys must be 1D");
  TORCH_CHECK(keys.scalar_type() == torch::kLong, "keys must be int64");

  int64_t rows = keys.size(0);
  int64_t numel = rows * cols;

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(keys.device());
  // Default to float32 for now, can expose dtype argument later if needed
  torch::Tensor output = torch::empty({rows, cols}, options);

  if (numel == 0) {
    return output;
  }

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  // Launch kernel
  // We only support float output for now based on curand_uniform_wrapper which returns float.
  // If we need other types, we need to cast.
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, output.scalar_type(), "keyed_random_uniform", ([&] {
    keyed_random_uniform_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        keys.data_ptr<int64_t>(),
        cols,
        numel);
  }));

  return output;
}

} // namespace driss_torch
