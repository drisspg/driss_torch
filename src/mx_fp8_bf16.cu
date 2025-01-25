#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

#include "torch/torch.h"
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
void run_gemm(at::Tensor& XQ, at::Tensor& WQ, at::Tensor& x_scale,
             at::Tensor& w_scale, at::Tensor& out) {
 int M = XQ.size(0);
 int N = WQ.size(1);
 int K = XQ.size(1);

  // A matrix configuration
  using         ElementA    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand
  using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand
  using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
  constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

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
  using ClusterShape        = Shape<_2,_2,_1>;                                // Shape of the threadblocks in a cluster
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
  using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
  using StrideB   = typename Gemm::GemmKernel::StrideB;
  using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
  using StrideC   = typename Gemm::GemmKernel::StrideC;
  using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
  using StrideD   = typename Gemm::GemmKernel::StrideD;
  using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

  using DtypeA = ElementA::DataType;
  using DtypeB = ElementB::DataType;
  using DtypeScaleA = ElementA::ScaleFactorType;
  using DtypeScaleB = ElementB::ScaleFactorType;
  using DtypeOut = ElementD;

  /// Initialization
  StrideA stride_A;
  // LayoutA layout_A;
  LayoutSFA layout_SFA;
  StrideB stride_B;
  // LayoutB layout_B;
  LayoutSFB layout_SFB;
  StrideC stride_C;
  // LayoutC layout_C;
  StrideD stride_D;
  // LayoutD layout_D;
  // uint64_t seed;

  Gemm gemm;

  auto A_ptr = reinterpret_cast<DtypeA*>(XQ.data_ptr());
  auto B_ptr = reinterpret_cast<DtypeB*>(WQ.data_ptr());
  auto SFA_ptr = reinterpret_cast<DtypeScaleA*>(x_scale.data_ptr());
  auto SFB_ptr = reinterpret_cast<DtypeScaleB*>(w_scale.data_ptr());
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
      {1,0},
      nullptr, stride_C,
      out_ptr, stride_D
    }
  };

  // arguments.scheduler.max_swizzle_size = 8;

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Cutlass cannot implement");

  // Allocate workspace memory
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = XQ.new_empty(
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

at::Tensor mx_fp8_bf16(at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale,
                    at::Tensor w_scale) {
 TORCH_CHECK(XQ.is_cuda(), "XQ must be CUDA tensor");
 TORCH_CHECK(WQ.is_cuda(), "WQ must be CUDA tensor");
 TORCH_CHECK(x_scale.is_cuda(), "x_scale must be CUDA tensor");
 TORCH_CHECK(w_scale.is_cuda(), "w_scale must be CUDA tensor");

 auto out = at::empty({XQ.size(0), WQ.size(1)},
                     XQ.options().dtype(at::kBFloat16));

 run_gemm(XQ, WQ, x_scale, w_scale, out);
 return out;
}

} // namespace driss_torch
