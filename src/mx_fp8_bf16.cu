#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>


namespace driss_torch {
using namespace at;
namespace {
using namespace cute;
void run_gemm(at::Tensor& a, at::Tensor& b, at::Tensor& a_scale,
             at::Tensor& b_scale, at::Tensor& out) {
 int M = a.size(0);
 int K = a.size(1);
 int N = b.size(1);
std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;
  // A matrix configuration
  using         ElementA    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand
  using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  constexpr int AlignmentA  = 16;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand
  using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
  constexpr int AlignmentB  = 128;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
  using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
  using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
  using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
  constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
  // Kernel functional config
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ArchTag             = cutlass::arch::Sm100;                           // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

  // Kernel Perf config
  using MmaTileShape        = Shape<_128,_128,_128>;                          // MMA's tile size
  using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster
  using PerSmTileShape_MNK  = Shape<_128,_128,_128>;                          // Threadblock-level tile size

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      PerSmTileShape_MNK, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto or using targeted scheduling policy
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,                                                   // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Reference device GEMM implementation type
  using StrideA   = typename Gemm::GemmKernel::StrideA;
  using StrideB   = typename Gemm::GemmKernel::StrideB;
  using StrideC   = typename Gemm::GemmKernel::StrideC;
  using StrideD   = typename Gemm::GemmKernel::StrideD;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using Sm100BlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm100BlkScaledConfig;

  // Initialize strides using packed stride configuration
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

  // Create layouts with proper shapes and strides
  auto layout_A = make_layout(make_shape(M, K, 1), stride_A);
  auto layout_B = make_layout(make_shape(K, N, 1), stride_B);
  auto layout_D = make_layout(make_shape(M, N, 1), stride_D);

  // Initialize scale factor layouts using block scaled configuration
  auto layout_SFA = Sm100BlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Sm100BlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  using DtypeA = ElementA::DataType;
  using DtypeB = ElementB::DataType;
  using DtypeScaleA = ElementA::ScaleFactorType;
  using DtypeScaleB = ElementB::ScaleFactorType;
  using DtypeOut = ElementD;

  Gemm gemm;

  auto A_ptr = reinterpret_cast<DtypeA*>(a.data_ptr());
  auto B_ptr = reinterpret_cast<DtypeB*>(b.data_ptr());
  auto SFA_ptr = reinterpret_cast<DtypeScaleA*>(a_scale.data_ptr());
  auto SFB_ptr = reinterpret_cast<DtypeScaleB*>(b_scale.data_ptr());
  auto out_ptr = reinterpret_cast<DtypeOut*>(out.data_ptr());

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    { // Mainloop arguments
      A_ptr, stride_A,
      B_ptr, stride_B,
      SFA_ptr, layout_SFA,
      SFB_ptr, layout_SFB
    },
    { // Epilogue arguments
      {1.0, 0.0},
      nullptr, StrideC{},  // No bias for now
      out_ptr, stride_D
    }
  };

  // arguments.scheduler.max_swizzle_size = 8;

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Cutlass cannot implement");
  // Allocate workspace memory
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  std::cout<<"Workspace size: "<< workspace_size<<std::endl;
  auto workspace = a.new_empty(
      {static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte));


  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Cutlass cannot initialize");

  status = gemm.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Cutlass cannot run", cutlass::cutlassGetStatusString(status));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

}
}

at::Tensor mx_fp8_bf16(at::Tensor a, at::Tensor b, at::Tensor a_scale,
                    at::Tensor b_scale) {
 TORCH_CHECK(a.is_cuda(), "a must be CUDA tensor");
 TORCH_CHECK(b.is_cuda(), "b must be CUDA tensor");
 TORCH_CHECK(a_scale.is_cuda(), "a_scale must be CUDA tensor");
 TORCH_CHECK(b_scale.is_cuda(), "b_scale must be CUDA tensor");

 auto out = at::empty({a.size(0), b.size(1)},
                     a.options().dtype(at::kBFloat16));

 run_gemm(a, b, a_scale, b_scale, out);
 return out;
}

} // namespace driss_torch
