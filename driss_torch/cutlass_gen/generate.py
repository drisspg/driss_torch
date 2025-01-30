import cutlass_library.generator as cutlass_generator
import cutlass_library.manifest as cutlass_manifest
from cutlass_library.generator import *


from dataclasses import dataclass
from typing import Optional
import logging


@dataclass
class ManifestArgs:
    """
    Arguments for CUTLASS Manifest configuration based on ArgumentParser definition
    """

    # Operation and kernel filtering
    operations: str = "all"
    kernels: str = ""
    ignore_kernels: str = ""
    exclude_kernels: str = ""
    kernel_filter_file: Optional[str] = None

    # Build directories
    build_dir: str = "."
    curr_build_dir: str = "."

    # Target and architecture configuration
    generator_target: str = "library"
    architectures: str = "80;90;100"
    filter_by_cc: str = "True"
    cuda_version: str = "12.8.0"

    # Output and interface configuration
    selected_kernel_list: Optional[str] = None
    interface_dir: Optional[str] = None

    # Build options
    disable_full_archs_compilation: bool = False
    log_level: str = "info"
    instantiation_level: str = ""

    @property
    def numeric_log_level(self) -> int:
        """Convert string log level to numeric logging level"""
        return getattr(logging, self.log_level.upper(), logging.INFO)

    @property
    def generator_targets(self) -> list[str]:
        """Parse generator target string into list"""
        return self.generator_target.split(",")

    @property
    def should_emit_library(self) -> bool:
        """Check if library should be emitted"""
        return "library" in self.generator_targets

    @property
    def is_blackwell_enabled(self) -> bool:
        """Check if Blackwell architecture is enabled"""
        return self.architectures == "100a"

    @property
    def compute_capabilities(self) -> list[int]:
        """Parse architectures string into compute capabilities list"""
        if not self.architectures:
            return [50]

        # Check for comma usage instead of semicolons
        if "," in self.architectures:
            raise ValueError(
                "The list of architectures must be semicolon-delimited.\n"
                "Don't use commas to separate the architectures; use semicolons.\n"
                f"You specified the list as: {self.architectures}"
            )

        # Process architectures including conditional ones
        arch_list = self.architectures.split(";")
        arch_conditional = ["90a", "100a"]

        # Remove 'a' suffix from conditional architectures
        processed_arch = [
            x.split("a")[0] if x in arch_conditional else x for x in arch_list
        ]

        return [int(x) for x in processed_arch]

    def write_selected_kernels(self, selected_kernels: list[str]) -> None:
        """Write selected kernels to file if specified"""
        if self.selected_kernel_list is not None and selected_kernels:
            with open(self.selected_kernel_list, "w") as file_writer:
                for kernel in selected_kernels:
                    file_writer.write(f"{kernel}\n")

    def configure_logging(self) -> None:
        """Configure logging based on specified level"""
        logging.basicConfig(level=self.numeric_log_level)



def GenerateSM100_TensorOp_FP8_BF16_gemm(manifest, cuda_version):
    # SM100 MMA with FP8 inputs and BF16 outputs
    # if not CudaToolkitVersionSatisfies(cuda_version, 12, 8):
    #     return

    layouts = [
        [[LayoutType.RowMajor, 128], [LayoutType.ColumnMajor, 128], [LayoutType.RowMajor, 0]],
        [[LayoutType.ColumnMajor, 128], [LayoutType.RowMajor, 128], [LayoutType.RowMajor, 0]],
    ]

    # Define instruction sizes for 1SM and 2SM cases
    instruction_sizes_1sm = [
        [128, 128, 32], [128, 256, 32],  # 1SM cases
    ]

    instruction_sizes_2sm = [
        [256, 128, 32], [256, 256, 32],  # 2SM cases
    ]

    # Only using E4M3 (FP8) for inputs
    ab_types = [DataType.e4m3]
    acc_types = [DataType.f32]  # Keep F32 accumulation for precision

    def tile_schedulers(sfdtype):
        if sfdtype["type"] == DataType.void:
            return [TileSchedulerType.Default]
        else:
            return [TileSchedulerType.Default, TileSchedulerType.StreamK]

    min_cc = 100
    max_cc = 100
    epi_type = DataType.f32

    # Generate math instructions for 1SM
    math_instructions_1sm = []
    for instr_size in instruction_sizes_1sm:
        math_instructions_1sm.append(
            MathInstruction(
                instr_size,
                DataType.e4m3, DataType.e4m3, DataType.f32,  # FP8 inputs, F32 accumulation
                OpcodeClass.BlockScaledTensorOp,
                MathOperation.multiply_add,
                DataType.ue8m0)
        )

    # Generate math instructions for 2SM
    math_instructions_2sm = []
    for instr_size in instruction_sizes_2sm:
        math_instructions_2sm.append(
            MathInstruction(
                instr_size,
                DataType.e4m3, DataType.e4m3, DataType.f32,  # FP8 inputs, F32 accumulation
                OpcodeClass.BlockScaledTensorOp,
                MathOperation.multiply_add,
                DataType.ue8m0)
        )

    # Cluster shapes for 1SM and 2SM
    cluster_shapes_1sm = [
        [1, 1, 1],
        [2, 1, 1],
        [4, 4, 1]
    ]

    cluster_shapes_2sm = [
        [2, 1, 1],
        [4, 1, 1],
        [4, 4, 1]
    ]

    # Generate 1SM kernels
    for math_inst in math_instructions_1sm:
        tile_descriptions = []
        for cluster_shape in cluster_shapes_1sm:
            multiplier_1sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else cluster_shape
            tile_descriptions.append(
                TileDescription([
                    math_inst.instruction_shape[0] * multiplier_1sm[0],
                    math_inst.instruction_shape[1] * multiplier_1sm[1],
                    math_inst.instruction_shape[2] * 4 * multiplier_1sm[2]],
                    0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

        # Define data type configurations - now with BF16 output
        data_types = [
            {
                "a_type": math_inst.element_a,
                "b_type": math_inst.element_b,
                "c_type": DataType.void,
                "d_type": DataType.bf16,  # BF16 output
                "acc_type": math_inst.element_accumulator,
                "epi_type": epi_type,
                "sf_type": math_inst.element_scale_factor,
                "sfd_type": {"type": DataType.void, "vector_size": None, "layout": None}
            },
            {
                "a_type": math_inst.element_a,
                "b_type": math_inst.element_b,
                "c_type": DataType.f16,
                "d_type": DataType.bf16,  # BF16 output
                "acc_type": math_inst.element_accumulator,
                "epi_type": epi_type,
                "sf_type": math_inst.element_scale_factor,
                "sfd_type": {"type": DataType.void, "vector_size": None, "layout": None}
            }
        ]

        # Set alignment for output based on BF16 size
        for layout in layouts:
            layout[2][1] = 128 // DataTypeSize[DataType.bf16]

        for data_type in data_types:
            CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
                [[KernelScheduleType.Mxf8f6f4TmaWarpSpecialized1SmSm100, EpilogueScheduleType.TmaWarpSpecialized1Sm]],
                tile_schedulers=tile_schedulers(data_type["sfd_type"]))

    # Generate 2SM kernels with similar modifications
    for math_inst in math_instructions_2sm:
        tile_descriptions = []
        for cluster_shape in cluster_shapes_2sm:
            multiplier_2sm = (1, 1, 1) if cluster_shape == DynamicClusterShape else (
            cluster_shape[0] // 2, cluster_shape[1], cluster_shape[2])
            tile_descriptions.append(
                TileDescription([
                    math_inst.instruction_shape[0] * multiplier_2sm[0],
                    math_inst.instruction_shape[1] * multiplier_2sm[1],
                    math_inst.instruction_shape[2] * 4 * multiplier_2sm[2]],
                    0, [4, 1, 1], math_inst, min_cc, max_cc, cluster_shape))

        # Define data types for 2SM - with BF16 output
        data_types = [
            {
                "a_type": math_inst.element_a,
                "b_type": math_inst.element_b,
                "c_type": DataType.void,
                "d_type": DataType.bf16,
                "acc_type": math_inst.element_accumulator,
                "epi_type": epi_type,
                "sf_type": math_inst.element_scale_factor,
                "sfd_type": {"type": DataType.void, "vector_size": None, "layout": None}
            },
            {
                "a_type": math_inst.element_a,
                "b_type": math_inst.element_b,
                "c_type": DataType.f16,
                "d_type": DataType.bf16,
                "acc_type": math_inst.element_accumulator,
                "epi_type": epi_type,
                "sf_type": math_inst.element_scale_factor,
                "sfd_type": {"type": DataType.ue8m0, "vector_size": 32, "layout": LayoutType.RowMajor}
            }
        ]

        # Handle alignments and kernel generation for 2SM cases
        for data_type in data_types:
            for layout in layouts:
                layout[0][1] = get_tma_alignment_elt(data_type["a_type"])
                layout[1][1] = get_tma_alignment_elt(data_type["b_type"])
                layout[2][1] = get_tma_alignment_elt(data_type["d_type"])
                
                for tile in tile_descriptions:
                    math_inst = tile.math_instruction
                    # Alignment checks
                    if layout[0][0] == LayoutType.ColumnMajor:
                        if math_inst.instruction_shape[0] // 2 % layout[0][1] != 0:
                            continue
                    else:
                        if tile.threadblock_shape[2] // tile.cluster_shape[2] % layout[0][1] != 0:
                            continue

                    if layout[1][0] == LayoutType.RowMajor:
                        if math_inst.instruction_shape[1] // 2 % layout[1][1] != 0:
                            continue
                    else:
                        if tile.threadblock_shape[2] // tile.cluster_shape[2] % layout[1][1] != 0:
                            continue

                    if math_inst.instruction_shape[0] == 128:
                        CreateGemmUniversal3xOperator(manifest, [layout], [tile], [data_type],
                            [[KernelScheduleType.Mxf8f6f4TmaWarpSpecialized2SmSm100, EpilogueScheduleType.TmaWarpSpecialized2Sm]],
                            tile_schedulers=tile_schedulers(data_type["sfd_type"]))
                    else:
                        CreateGemmUniversal3xOperator(manifest, [layout], [tile], [data_type],
                            [[KernelScheduleType.Mxf8f6f4TmaWarpSpecialized2SmSm100, EpilogueScheduleType.ScheduleAuto]],
                            tile_schedulers=tile_schedulers(data_type["sfd_type"]))

def get_blockscaled_configs():
    args = ManifestArgs(architectures="100", kernels="ue8m0xe4m3_ue8m0xe4m3_f32*tnt")

    manifest = cutlass_manifest.Manifest(args)
    cuda_version = "100"

    # cutlass_generator.GenerateSM100(
    #     manifest, cuda_version
    # )

    GenerateSM100_TensorOp_FP8_BF16_gemm(manifest, cuda_version)
    print(manifest.args)
    print(f"Num Kernels {len(manifest.operations)}")
    print(manifest.operations)


if __name__ == "__main__":
    get_blockscaled_configs()
