cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX CUDA)

# Set the C++ standard for all targets
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c" "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE PT_CMAKE_PREFIX
    COMMAND_ECHO STDOUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

# Set CUDA architecture to SM100a
set(CMAKE_CUDA_ARCHITECTURES 90a)
set(TORCH_CUDA_ARCH_LIST "9.0a")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${PT_CMAKE_PREFIX})
find_package(Torch REQUIRED CONFIG)

# ---- cuDNN dependency setup ----
# Set system-installed cuDNN paths directly
set(CUDNN_INCLUDE_DIR "/usr/include" CACHE PATH "cuDNN include directory")
set(CUDNN_LIBRARY "/usr/lib64/libcudnn.so" CACHE FILEPATH "cuDNN library")

# Add cuDNN frontend from submodule
set(CUDNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn-frontend/include" CACHE PATH "cuDNN frontend include directory")

# Verify cuDNN frontend header exists in the submodule
if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h")
    message(FATAL_ERROR "Could not find cudnn_frontend.h in submodule. Please make sure the submodule is properly initialized.")
endif()

message(STATUS "Using cuDNN frontend from submodule: ${CUDNN_FRONTEND_INCLUDE_DIR}")

# driss_torch source files - exclude scaled_mm_kernels by default
file(GLOB_RECURSE CU_SOURCES "src/*.cu")
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
list(FILTER CU_SOURCES EXCLUDE REGEX "scaled_mm_kernels")

# Only add scaled_mm_kernels if BUILD_SWEEP_MM is ON
if(BUILD_SWEEP_MM)
    file(GLOB_RECURSE SWEEP_MM_SOURCES "src/scaled_mm_kernels/*.cu")
    list(APPEND CU_SOURCES ${SWEEP_MM_SOURCES})
    add_definitions(-DBUILD_SWEEP_MM)
endif()

add_library(driss_torch SHARED
    ${CU_SOURCES}
    ${CPP_SOURCES}
)

# Set the library output directory
set_target_properties(driss_torch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/driss_torch/lib"
)

# Check for CUTLASS
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include/cutlass/cutlass.h")
    message(FATAL_ERROR "The Cutlass submodule was not downloaded! Please update submodules and try again.")
endif()

# Include CUTLASS headers without building the entire library
target_include_directories(driss_torch PUBLIC
    src/include
    src/scaled_mm_kernels/
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/library/include
    ${CUDNN_INCLUDE_DIR}
    ${CUDNN_FRONTEND_INCLUDE_DIR}
)

# Create a cuDNN target for linking
add_library(cudnn INTERFACE)
target_include_directories(cudnn INTERFACE
    ${CUDNN_INCLUDE_DIR}
    ${CUDNN_FRONTEND_INCLUDE_DIR}
)
target_link_libraries(cudnn INTERFACE ${CUDNN_LIBRARY})

# Link the library to the Torch library and cuDNN
target_link_libraries(driss_torch PRIVATE ${TORCH_LIBRARIES} cudnn)

# Install the library to the wheel distribution
install(TARGETS driss_torch
    LIBRARY DESTINATION driss_torch/lib
)

# Print some configuration information
message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "cuDNN include directory: ${CUDNN_INCLUDE_DIR}")
message(STATUS "cuDNN frontend include directory: ${CUDNN_FRONTEND_INCLUDE_DIR}")
message(STATUS "cuDNN library: ${CUDNN_LIBRARY}")
message(STATUS "PyTorch prefix path: ${PT_CMAKE_PREFIX}")
