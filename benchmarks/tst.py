from cutlass_library.manifest import Manifest
from cutlass_library.generator import GeneratorTarget
from cutlass_library.generator import (
    GenerateSM100_TensorOp_mixed_8bits_UMMA_gemm_with_block_scaled,
)


# Create Options object for manifest configuration
class Options:
    def __init__(self):
        self.kernels = "all"  # Generate all kernels
        self.curr_build_dir = "."
        self.operations = "all"
        self.ignore_kernels = ""
        self.exclude_kernels = ""
        self.kernel_filter_file = None
        self.architectures = "100"  # For SM100
        self.filter_by_cc = True
        self.disable_full_archs_compilation = False
        self.instantiation_level = "max"

def GenerateCustomE4M3Gemm(manifest, cuda_version):
    from cutlass_library import (
        LayoutType,
        MathOperation,
        DataType,
        MathInstruction,
        OpcodeClass,
        TileDescription,
        DynamicClusterShape,
        EpilogueScheduleType,
        TileSchedulerType,
        KernelScheduleType,
    )
    from cutlass_library.generator import CreateGemmUniversal3xOperator, CudaToolkitVersionSatisfies

    if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
        return

    layouts = [
        [
            [LayoutType.RowMajor, 128],
            [LayoutType.ColumnMajor, 128],
            [LayoutType.RowMajor, 0],
        ]
    ]

    instruction_sizes = [[128, 128, 32]]  # Single instruction size for simplicity

    min_cc = 100
    max_cc = 100

    math_instructions = []
    for instr_size in instruction_sizes:
        math_instructions.append(
            MathInstruction(
                instr_size,
                DataType.e4m3,  # A type
                DataType.e4m3,  # B type
                DataType.f32,  # Accumulator type
                OpcodeClass.BlockScaledTensorOp,
                MathOperation.multiply_add,
                DataType.ue8m0,
            )
        )

    cluster_shapes = [[1, 1, 1]]  # Single cluster shape

    for math_inst in math_instructions:
        tile_descriptions = []
        for cluster_shape in cluster_shapes:
            multiplier = (
                (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
            )
            tile_descriptions.append(
                TileDescription(
                    [
                        math_inst.instruction_shape[0] * multiplier[0],
                        math_inst.instruction_shape[1] * multiplier[1],
                        math_inst.instruction_shape[2] * 4 * multiplier[2],
                    ],
                    0,
                    [4, 1, 1],
                    math_inst,
                    min_cc,
                    max_cc,
                    cluster_shape,
                )
            )

        data_type = {
            "a_type": math_inst.element_a,
            "b_type": math_inst.element_b,
            "c_type": DataType.void,  # No bias
            "d_type": DataType.bf16,  # Output in bf16
            "acc_type": DataType.f32,  # Accumulate in f32
            "epi_type": DataType.f32,
            "sf_type": math_inst.element_scale_factor,
            "sfd_type": {"type": DataType.void, "vector_size": None, "layout": None},
        }

        CreateGemmUniversal3xOperator(
            manifest,
            layouts,
            tile_descriptions,
            data_type,
            [
                [
                    KernelScheduleType.Mxf8f6f4TmaWarpSpecialized1SmSm100,
                    EpilogueScheduleType.TmaWarpSpecialized1Sm,
                ]
            ],
            tile_schedulers=[TileSchedulerType.Default],
        )



if __name__ == "__main__":
    args = Options()
    manifest = Manifest(args)
    GenerateCustomE4M3Gemm(manifest, cuda_version="12.8")
    manifest.emit()


    # # Initialize manifest with options
    # args = Options()
    # manifest = Manifest(args)

    # # Generate the kernels
    # cuda_version = "12.8"
    # GenerateSM100_TensorOp_mixed_8bits_UMMA_gemm_with_block_scaled(manifest, cuda_version)

    # # Emit the manifest
    # manifest.emit(GeneratorTarget.Library)
