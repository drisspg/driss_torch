#include <c10/core/DispatchKey.h>
#include <torch/library.h>

// Custom up headers
#include "amax.h"
#include "dynamic_scaled_quant.h"
#include "saturated_cast.h"

TORCH_LIBRARY(DrissTorch, m) {
  m.impl_abstract_pystub("driss_torch.abstract_impls");
  //   Saturated cast func from bf16 to fp8 types
  m.def("saturated_cast(Tensor input, Tensor scale, ScalarType dtype, bool "
        "transpose) -> Tensor");
  m.impl("saturated_cast", c10::DispatchKey::CUDA,
         TORCH_FN(driss_torch::saturated_cast));
  //   Amax func
  m.def("amax(Tensor input) -> Tensor");
  m.impl("amax", c10::DispatchKey::CUDA, TORCH_FN(driss_torch::amax));
  // Dynamic cast to fp8
  m.def("dynamic_scaled_quant(Tensor input, ScalarType dtype) -> Tensor");
  m.impl("dynamic_scaled_quant", c10::DispatchKey::CUDA,
         TORCH_FN(driss_torch::dynamic_scaled_quant));
}
