cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(driss_torch LANGUAGES CXX CUDA)

# Set the C++ standard for all targets
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable better clangd support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Python3 REQUIRED COMPONENTS Development)
execute_process(
        COMMAND "${Python3_EXECUTABLE}" "-c" "import torch;print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE PT_CMAKE_PREFIX
        COMMAND_ECHO STDOUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
)

# cache CUDA_ARCHITECTURES, which seems to be reset by Torch
set(TMP_STORE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${PT_CMAKE_PREFIX})

find_package(Torch REQUIRED CONFIG)
# set(CMAKE_CUDA_ARCHITECTURES ${TMP_STORE_CUDA_ARCHITECTURES})

#  simple_cuda source files
file(GLOB_RECURSE SIMPLE_CUDA_CU_SOURCES src/*.cu)
file(GLOB_RECURSE SIMPLE_CUDA_CPP_SOURCES src/*.cpp)
MESSAGE(STATUS "SIMPLE_CUDA_CU_SOURCES: ${SIMPLE_CUDA_CU_SOURCES}")
MESSAGE(STATUS "SIMPLE_CUDA_CPP_SOURCES: ${SIMPLE_CUDA_CPP_SOURCES}")

add_library(driss_torch SHARED
    ${SIMPLE_CUDA_CU_SOURCES}
    ${SIMPLE_CUDA_CPP_SOURCES}
)
# target_compile_features(cuda_torch PRIVATE cxx_std_11)
target_link_libraries(driss_torch PRIVATE ${TORCH_LIBRARIES} Python3::Python)

# Use if the default GCC version gives issues
# target_compile_options(driss_torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ccbin g++-9>)

# Use a variant of this if you're on an earlier cmake than 3.18
# target_compile_options(pytorch_cmake_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)
