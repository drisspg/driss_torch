#include "include/utils.h"

extern "C" {
using namespace driss_torch;

/**
 * @brief Measures the execution time of a CUDA kernel.
 *
 * This function measures the execution time of a CUDA kernel by recording
 * CUDA events before and after the kernel launch. The time is measured in
 * milliseconds.
 *
 * @param kernelLauncher A std::function object that launches the CUDA kernel.
 * This function should take no arguments and have no return value. The
 * function should launch the CUDA kernel with the desired grid and block
 * dimensions.
 *
 * @return The execution time of the CUDA kernel in milliseconds.
 *
 * @note The caller is responsible for ensuring that any device memory
 * accessed by the kernel is allocated before this function is called and
 * deallocated after this function returns. The caller is also responsible
 * for handling any CUDA errors that may occur during the kernel execution.
 */
extern "C" float kernel_time(std::function<void()> kernelLauncher) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  kernelLauncher();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsedTime;
}

} // extern "C"
