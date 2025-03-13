#include "cute/int_tuple.hpp"
#include "cute/pointer.hpp"
#include "cutlass/fast_math.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include "include/mx_cast.h"

// CUTLASS includes
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/util/device_memory.h>

#include <memory>
#include <vector>
#include <cmath>

namespace driss_torch_kernels {

namespace cg = cooperative_groups;
using namespace cute;

template <class Element, class TensorInput, class TensorOutput, class TensorScale, class Tiled_Copy>
__global__ void mx_fp8_quantize_kernel(TensorInput input, TensorOutput output, TensorScale scale, Tiled_Copy tiled_copy){
// Slice the tensors to obtain a view into each tile.
  Tensor tile_input = input(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_output = output(make_coord(_,_), blockIdx.x, blockIdx.y);
  Tensor tile_scale = scale(make_coord(_,_), blockIdx.x, blockIdx.y);

  // Construct a Tensor corresponding to each thread's slice.
  ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  print(thr_copy);

  Tensor thr_tile_input = thr_copy.partition_S(tile_input);
  Tensor thr_tile_ouput = thr_copy.partition_D(tile_output);
//   Tensor thr_tile_scale = thr_copy.partition_D(tile_scale);

  // Construct a register-backed Tensor with the same shape as each thread's partition
  // Use make_fragment because the first mode is the instruction-local mode
  Tensor input_fragment = make_fragment_like(thr_tile_ouput);
  // auto smem_size = size(tile_shape);

  // // Define shared memory
  // extern __shared__ char shared_memory[];
  // using SmemArray = array_aligned<Element, cosize_v<decltype(tile_input.layout())>>;
  // SmemArray &smem = *reinterpret_cast<SmemArray*>(shared_memory);

  // // Create a tensor view of shared memory with the same shape as the tile
  // Tensor smem_tile = make_tensor(smem.data(), make_layout(make_shape(size<0>(tile_input), size<1>(tile_input))));

  // // Get the thread's slice of the copy operation
  // auto thread_idx = threadIdx.x;
  // auto thr_copy = tiled_copy.get_thread_slice(thread_idx);

  // // Partition the tensors for this thread
  // Tensor thr_tile_input = thr_copy.partition_S(tile_input);       // Thread's slice of global input
  // Tensor thr_smem_tile = thr_copy.partition_D(smem_tile);         // Thread's slice of shared memory
  // Tensor thr_tile_output = thr_copy.partition_D(tile_output);     // Thread's slice of global output

  // // Create register storage for this thread's data
  // Tensor reg_fragment = make_fragment_like(thr_tile_input);

  // // Copy from global memory to shared memory
  // copy(tiled_copy, tile_input, smem_tile);

  // // Ensure all threads complete their copy before proceeding
  // __syncthreads();

  // // Copy from shared memory to registers (if needed for processing)
  // copy(thr_smem_tile, reg_fragment);

  // Copy from GMEM to RMEM and from RMEM to GMEM
  // copy(tiled_copy, thr_tile_input, input_fragment);
//   copy(tiled_copy, fragment, thr_tile_ouput);

}

} // driss_torch_kernels namespace

namespace driss_torch {
using namespace cute;

std::tuple<at::Tensor, at::Tensor> mx_fp8_quantize(
    at::Tensor input,
    int64_t block_size,
    int64_t axis,
    bool transpose,
    c10::ScalarType fp8_type) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kFloat ||
               input.scalar_type() == at::kBFloat16,
               "Input tensor must be float, half, or bfloat16");
    TORCH_CHECK(block_size > 0 && block_size <= 32, "Block size must be positive and <= 32");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D for CUTLASS implementation");
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
    auto scale = at::empty({total_blocks}, input.options().dtype(at::kFloat8_e8m0fnu));

    auto tensor_input_shape = make_shape(m, n);
    auto tensor_scale_shape = make_shape(m, num_k_blocks);


    auto input_ptr = static_cast<__nv_bfloat16*>(input.data_ptr());
    auto scale_ptr = static_cast<__nv_fp8_storage_t*>(scale.data_ptr());
    auto output_ptr = static_cast<__nv_fp8_storage_t*>(output.data_ptr());

    Tensor tensor_input = make_tensor(make_gmem_ptr(input_ptr), make_layout(tensor_input_shape));
    Tensor tensor_scale = make_tensor(make_gmem_ptr(scale_ptr), make_layout(tensor_scale_shape));
    Tensor tensor_ouput = make_tensor(make_gmem_ptr(output_ptr), make_layout(tensor_input_shape));

    // Keep it easy for now
    auto block_shape = make_shape(Int<128>{}, Int<32>{});
    auto scale_shape = make_shape(get<0>(block_shape), Int<1>{});
    TORCH_CHECK(evenly_divides(tensor_input_shape, block_shape), "Need block shape to evenly divide the input tensor for now");

    // These will be used to determine the CUDA kernel grid dimensions.
    Tensor tiled_tensor_input = tiled_divide(tensor_input, block_shape);
    Tensor tiled_tensor_ouput = tiled_divide(tensor_ouput, block_shape);
    Tensor tiled_tensor_scale = tiled_divide(tensor_scale, scale_shape);

    // Construct a TiledCopy with a specific access pattern
    //   This version uses a
    //   (1) Layout-of-Threads to describe the number and arrangement of threads (e.g. row-major, col-major, etc),
    //   (2) Layout-of-Values that each thread will access.

    // Thread arrangement
    Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<32>{}));

    // Value arrangement per thread
    Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));
    // Define `AccessType` which controls the size of the actual memory access instruction.
    using CopyOp = UniversalCopy<uint_byte_t<sizeof(__nv_fp8_storage_t) * size(val_layout)>>;     // A very specific access width copy instruction
    //using CopyOp = UniversalCopy<cutlass::AlignedArray<Element, size(val_layout)>>;  // A more generic type that supports many copy strategies
    // using CopyOp = AutoVectorizingCopy;                                              // An adaptable-width instruction that assumes maximal alignment of inputs

    // A Copy_Atom corresponds to one CopyOperation applied to Tensors of type Element.
    using Atom = Copy_Atom<CopyOp, uint8_t>;

    // Construct tiled copy, a tiling of copy atoms.
    //
    // Note, this assumes the vector and thread layouts are aligned with contigous data
    // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
    // reads. Alternative value layouts are also possible, though incompatible layouts
    // will result in compile time errors.
    TiledCopy tiled_copy = make_tiled_copy(Atom{},             // Access strategy
                                            thr_layout,         // thread layout (e.g. 32x4 Col-Major)
                                            val_layout);        // value layout (e.g. 4x1)



    dim3 gridDim (size<1>(tiled_tensor_input), size<2>(tiled_tensor_input));   // Grid shape corresponds to modes m' and n'
    dim3 blockDim(size(thr_layout));

        //
        // Launch the kernel
        //
        driss_torch_kernels::mx_fp8_quantize_kernel<__nv_bfloat16><<< gridDim, blockDim >>>(
            tiled_tensor_input,
            tiled_tensor_ouput,
            tiled_tensor_scale,
            tiled_copy);


    // Check for CUDA errors
    C10_CUDA_CHECK(cudaGetLastError());

    return {output, scale};
}

} // namespace driss_torch
