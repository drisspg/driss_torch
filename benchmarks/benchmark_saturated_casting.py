import itertools
from dataclasses import dataclass

from typing import List

import torch

from driss_torch import saturated_cast

from tabulate import tabulate
from tqdm import tqdm

from transformer_nuggets.fp8.scaled_quant import eager_scaled_quant, scaled_quant
from transformer_nuggets.utils import benchmark_torch_function_in_microseconds

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


def eager_scaled_quant(
    a: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
):
    """Quantize tensor to fp8 using a delayed scaled and calculate abs_max

    Args:
        a: Input tensor to quantize
        scale: Scale to apply to input tensor, calculated from previous abs_max
        fp8_dtype: FP8 datatype to quantize to
    """
    out = a * scale
    out = torch.where(out > torch.finfo(fp8_dtype).max, torch.finfo(fp8_dtype).max, out)
    out = torch.where(out < -1 * torch.finfo(fp8_dtype).max, -1 * torch.finfo(fp8_dtype).max, out)
    return out.to(fp8_dtype)


@dataclass(frozen=True)
class ExperimentConfig:
    num_rows: int
    num_cols: int
    high_precision_dtype: torch.dtype
    low_precision_dtype: torch.dtype


@dataclass(frozen=True)
class ExperimentResult:
    cuda_time: float
    pytorch_time: float
    compiled_pytorch_time: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    sizes = [2**9, 2**10, 2**11, 2**12]
    high_precision_dtypes = [torch.bfloat16]
    low_precision_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2]
    saturated = [True, False]
    configs = []
    for size, high_precision_dtype, low_precision_dtype, sat in itertools.product(
        sizes, high_precision_dtypes, low_precision_dtypes, saturated
    ):
        configs.append(
            ExperimentConfig(
                num_rows=size,
                num_cols=size,
                high_precision_dtype=high_precision_dtype,
                low_precision_dtype=low_precision_dtype,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    high_precision_tensor = torch.randn(
        config.num_rows, config.num_cols, dtype=config.high_precision_dtype, device=device
    )
    cuda_hp_tensor = high_precision_tensor.clone()

    eager_abs_max = torch.abs(high_precision_tensor).max().to(torch.float32)

    scale = torch.finfo(config.low_precision_dtype).max / eager_abs_max
    scale = scale.to(torch.float32)
    scale = torch.ones(1, dtype=torch.float32, device=device)

    # Correctness check:
    cuda_out = saturated_cast(cuda_hp_tensor, config.low_precision_dtype)
    cuda_out_hp = cuda_out.to(config.high_precision_dtype)

    eager_out = eager_scaled_quant(high_precision_tensor, scale, config.low_precision_dtype).to(
        config.high_precision_dtype
    )
    eager_out_hp = eager_out.to(config.high_precision_dtype)

    torch.testing.assert_close(cuda_out_hp, eager_out_hp, rtol=1e-3, atol=1e-3)

    cuda_time = benchmark_torch_function_in_microseconds(
        saturated_cast,
        cuda_hp_tensor,
        config.low_precision_dtype,
    )
    pytorch_time = benchmark_torch_function_in_microseconds(
        eager_scaled_quant,
        high_precision_tensor,
        scale,
        config.low_precision_dtype,
    )
    compiled_pytorch_fn = torch.compile(eager_scaled_quant, fullgraph=True)
    compiled_pytorch_time = benchmark_torch_function_in_microseconds(
        compiled_pytorch_fn,
        high_precision_tensor,
        scale,
        config.low_precision_dtype,
    )
    return ExperimentResult(
        cuda_time=cuda_time,
        pytorch_time=pytorch_time,
        compiled_pytorch_time=compiled_pytorch_time,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "num_rows",
        "num_cols",
        "high_precision_dtype",
        "low_precision_dtype",
        "cuda_time",
        "pytorch_time",
        "compiled_pytorch_time",
    ]
    rows = []
    for experiment in experiments:
        rows.append(
            [
                experiment.config.num_rows,
                experiment.config.num_cols,
                experiment.config.high_precision_dtype,
                experiment.config.low_precision_dtype,
                experiment.result.cuda_time,
                experiment.result.pytorch_time,
                experiment.result.compiled_pytorch_time,
            ]
        )
    print(tabulate(rows, headers=headers))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    main()
