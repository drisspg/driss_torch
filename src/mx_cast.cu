#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <cudnn_frontend.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <memory>
#include <vector>

namespace cudnn_wrapper {

namespace fe = cudnn_frontend;

// Helper to create and manage CUDNN handle
class CudnnHandleManager {
public:
    CudnnHandleManager() {
        CUDNN_CHECK(cudnnCreate(&handle_));
    }

    ~CudnnHandleManager() {
        cudnnDestroy(handle_);
    }

    cudnnHandle_t get() const { return handle_; }

private:
    cudnnHandle_t handle_;
};

// Helper macro for checking CUDNN errors
#define CUDNN_CHECK(expr)                                               \
    do {                                                                \
        cudnnStatus_t status = (expr);                                  \
        if (status != CUDNN_STATUS_SUCCESS) {                           \
            auto msg = cudnnGetErrorString(status);                     \
            TORCH_CHECK(false, "cuDNN error: ", msg);                   \
        }                                                               \
    } while (0)

// Main function to quantize a tensor to MXFP8 format
std::vector<at::Tensor> block_scale_quantize_fp8(
    at::Tensor input,
    int64_t block_size = 32,
    int64_t axis = 1,
    bool transpose = false) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kFloat ||
               input.scalar_type() == at::kBFloat16,
               "Input tensor must be float, half, or bfloat16");
    TORCH_CHECK(block_size > 0, "Block size must be positive");
    TORCH_CHECK(axis >= 0 && axis < input.dim(), "Axis must be within tensor dimensions");

    // Get current CUDA device
    at::cuda::CUDAGuard device_guard(input.device());

    // Create CUDNN handle
    CudnnHandleManager handle_manager;
    auto handle = handle_manager.get();

    // Set the stream
    CUDNN_CHECK(cudnnSetStream(handle, at::cuda::getCurrentCUDAStream()));

    // Create graph
    fe::graph::Graph graph;

    // Configure graph data types
    auto input_data_type = fe::DataType_t::FLOAT;
    if (input.scalar_type() == at::kHalf) {
        input_data_type = fe::DataType_t::HALF;
    } else if (input.scalar_type() == at::kBFloat16) {
        input_data_type = fe::DataType_t::BFLOAT16;
    }

    graph.set_io_data_type(input_data_type)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // Get tensor dimensions
    std::vector<int64_t> dims_vec(input.sizes().begin(), input.sizes().end());
    std::vector<int64_t> strides_vec(input.strides().begin(), input.strides().end());

    // Create tensor attributes
    auto tensor_attrs = fe::graph::Tensor_attributes()
                          .set_data_type(input_data_type);

    // Set dimensions and strides
    if (dims_vec.size() == 2) {
        tensor_attrs.set_dim({static_cast<int>(dims_vec[0]),
                            static_cast<int>(dims_vec[1]), 1, 1})
                  .set_stride({static_cast<int>(strides_vec[0]),
                             static_cast<int>(strides_vec[1]), 0, 0});
    } else if (dims_vec.size() == 3) {
        tensor_attrs.set_dim({static_cast<int>(dims_vec[0]),
                            static_cast<int>(dims_vec[1]),
                            static_cast<int>(dims_vec[2]), 1})
                  .set_stride({static_cast<int>(strides_vec[0]),
                             static_cast<int>(strides_vec[1]),
                             static_cast<int>(strides_vec[2]), 0});
    } else if (dims_vec.size() == 4) {
        tensor_attrs.set_dim({static_cast<int>(dims_vec[0]),
                            static_cast<int>(dims_vec[1]),
                            static_cast<int>(dims_vec[2]),
                            static_cast<int>(dims_vec[3])})
                  .set_stride({static_cast<int>(strides_vec[0]),
                             static_cast<int>(strides_vec[1]),
                             static_cast<int>(strides_vec[2]),
                             static_cast<int>(strides_vec[3])});
    } else {
        TORCH_CHECK(false, "Input tensor must have 2, 3, or 4 dimensions");
    }

    // Add tensor to graph
    auto X = graph.tensor(tensor_attrs);

    // Configure quantization
    auto quantize_attrs = fe::graph::Block_scale_quantize_attributes()
                            .set_block_size(static_cast<int>(block_size))
                            .set_axis(static_cast<int>(axis))
                            .set_transpose(transpose);

    // Add quantization operation to graph
    auto [Y, mx_scale] = graph.block_scale_quantize(X, quantize_attrs);

    // Set outputs
    Y->set_output(true).set_data_type(fe::DataType_t::FP8_E5M2);
    mx_scale->set_output(true).set_data_type(fe::DataType_t::FP8_E8M0);

    // Validate and build the graph
    TORCH_CHECK(graph.validate().is_good(), "Graph validation failed");
    TORCH_CHECK(graph.build_operation_graph(handle).is_good(), "Build operation graph failed");
    TORCH_CHECK(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good(), "Create execution plans failed");
    TORCH_CHECK(graph.check_support(handle).is_good(), "Check support failed");
    TORCH_CHECK(graph.build_plans(handle).is_good(), "Build plans failed");

    // Calculate output sizes
    auto total_elements = input.numel();
    auto scale_elements = total_elements / block_size;

    // Create output tensors
    auto options = input.options().dtype(at::kChar); // int8_t for FP8
    auto output = at::empty(input.sizes(), options);
    auto scale = at::empty({scale_elements}, options);

    // Create workspace
    auto workspace_size = graph.get_workspace_size();
    auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                              at::TensorOptions().dtype(at::kChar).device(input.device()));

    // Set up variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, input.data_ptr()},
        {Y, output.data_ptr()},
        {mx_scale, scale.data_ptr()}
    };

    // Execute the graph
    TORCH_CHECK(graph.execute(handle, variant_pack, workspace.data_ptr()).is_good(),
               "Graph execution failed");

    // Return both quantized tensor and scale factors
    return {output, scale};
}

// Alternative function for NVFP4 quantization
std::vector<at::Tensor> block_scale_quantize_fp4(
    at::Tensor input,
    int64_t block_size = 16,
    int64_t axis = 1,
    bool transpose = false) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kFloat ||
               input.scalar_type() == at::kBFloat16,
               "Input tensor must be float, half, or bfloat16");

    // Similar implementation as above but with FP4_E2M1 data type
    CudnnHandleManager handle_manager;
    auto handle = handle_manager.get();
    CUDNN_CHECK(cudnnSetStream(handle, at::cuda::getCurrentCUDAStream()));

    fe::graph::Graph graph;

    auto input_data_type = fe::DataType_t::FLOAT;
    if (input.scalar_type() == at::kHalf) {
        input_data_type = fe::DataType_t::HALF;
    } else if (input.scalar_type() == at::kBFloat16) {
        input_data_type = fe::DataType_t::BFLOAT16;
    }

    graph.set_io_data_type(input_data_type)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // Setup tensor and quantization operation similar to FP8 version
    // but with FP4_E2M1 output data type

    std::vector<int64_t> dims_vec(input.sizes().begin(), input.sizes().end());
    std::vector<int64_t> strides_vec(input.strides().begin(), input.strides().end());

    auto tensor_attrs = fe::graph::Tensor_attributes()
                          .set_data_type(input_data_type);

    if (dims_vec.size() == 2) {
        tensor_attrs.set_dim({static_cast<int>(dims_vec[0]),
                            static_cast<int>(dims_vec[1]), 1, 1})
                  .set_stride({static_cast<int>(strides_vec[0]),
                             static_cast<int>(strides_vec[1]), 0, 0});
    } else if (dims_vec.size() == 3) {
        tensor_attrs.set_dim({static_cast<int>(dims_vec[0]),
                            static_cast<int>(dims_vec[1]),
                            static_cast<int>(dims_vec[2]), 1})
                  .set_stride({static_cast<int>(strides_vec[0]),
                             static_cast<int>(strides_vec[1]),
                             static_cast<int>(strides_vec[2]), 0});
    } else if (dims_vec.size() == 4) {
        tensor_attrs.set_dim({static_cast<int>(dims_vec[0]),
                            static_cast<int>(dims_vec[1]),
                            static_cast<int>(dims_vec[2]),
                            static_cast<int>(dims_vec[3])})
                  .set_stride({static_cast<int>(strides_vec[0]),
                             static_cast<int>(strides_vec[1]),
                             static_cast<int>(strides_vec[2]),
                             static_cast<int>(strides_vec[3])});
    } else {
        TORCH_CHECK(false, "Input tensor must have 2, 3, or 4 dimensions");
    }

    auto X = graph.tensor(tensor_attrs);

    auto quantize_attrs = fe::graph::Block_scale_quantize_attributes()
                            .set_block_size(static_cast<int>(block_size))
                            .set_axis(static_cast<int>(axis))
                            .set_transpose(transpose);

    auto [Y, mx_scale] = graph.block_scale_quantize(X, quantize_attrs);

    // Use FP4_E2M1 for output instead of FP8_E5M2
    Y->set_output(true).set_data_type(fe::DataType_t::FP4_E2M1);
    // Use FP8_E4M3 for scale factors
    mx_scale->set_output(true).set_data_type(fe::DataType_t::FP8_E4M3);

    // Build and execute the graph
    TORCH_CHECK(graph.validate().is_good(), "Graph validation failed");
    TORCH_CHECK(graph.build_operation_graph(handle).is_good(), "Build operation graph failed");
    TORCH_CHECK(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good(), "Create execution plans failed");
    TORCH_CHECK(graph.check_support(handle).is_good(), "Check support failed");
    TORCH_CHECK(graph.build_plans(handle).is_good(), "Build plans failed");

    // Calculate output sizes (FP4 is packed 2 elements per byte)
    auto total_elements = input.numel();
    auto scale_elements = total_elements / block_size;

    // Create output tensors
    auto options = input.options().dtype(at::kChar);
    auto output = at::empty({total_elements / 2}, options); // Half the size for FP4
    auto scale = at::empty({scale_elements}, options);

    auto workspace_size = graph.get_workspace_size();
    auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                              at::TensorOptions().dtype(at::kChar).device(input.device()));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, input.data_ptr()},
        {Y, output.data_ptr()},
        {mx_scale, scale.data_ptr()}
    };

    TORCH_CHECK(graph.execute(handle, variant_pack, workspace.data_ptr()).is_good(),
               "Graph execution failed");

    return {output, scale};
}

} // namespace cudnn_wrapper


namespace driss_torch {

std::vector<torch::Tensor> mx_fp8_quantize(
    torch::Tensor input,
    int64_t block_size = 32,
    int64_t axis = 1,
    bool transpose = false) {
    return cudnn_wrapper::block_scale_quantize_fp8(input, block_size, axis, transpose);
}

std::vector<torch::Tensor> mx_fp4_quantize(
    torch::Tensor input,
    int64_t block_size = 16,
    int64_t axis = 1,
    bool transpose = false) {
    return cudnn_wrapper::block_scale_quantize_fp4(input, block_size, axis, transpose);
}



} // driss_torch
// Wrapper functions for PyTorch extension
