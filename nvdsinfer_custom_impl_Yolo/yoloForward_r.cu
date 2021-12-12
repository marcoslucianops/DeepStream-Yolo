/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

inline __device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void gpuYoloLayer_r(const float* input, float* output, const uint gridSizeX, const uint gridSizeY, const uint numOutputClasses,
                               const uint numBBoxes, const float scale_x_y)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
    uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x_id >= gridSizeX) || (y_id >= gridSizeY) || (z_id >= numBBoxes))
    {
        return;
    }

    const int numGridCells = gridSizeX * gridSizeY;
    const int bbindex = y_id * gridSizeX + x_id;

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]) * 2.0 - 0.5;

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]) * 2.0 - 0.5;

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]
        = pow(sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]) * 2, 2);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]
        = pow(sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]) * 2, 2);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]);

    for (uint i = 0; i < numOutputClasses; ++i)
    {
        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]
            = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]);
    }
}

cudaError_t cudaYoloLayer_r(const void* input, void* output, const uint& batchSize, const uint& gridSizeX, const uint& gridSizeY,
                            const uint& numOutputClasses, const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream,
                            const float modelScale);

cudaError_t cudaYoloLayer_r(const void* input, void* output, const uint& batchSize, const uint& gridSizeX, const uint& gridSizeY,
                            const uint& numOutputClasses, const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream,
                            const float modelScale)
{
    dim3 threads_per_block(16, 16, 4);
    dim3 number_of_blocks((gridSizeX / threads_per_block.x) + 1,
                          (gridSizeY / threads_per_block.y) + 1,
                          (numBBoxes / threads_per_block.z) + 1);
    for (unsigned int batch = 0; batch < batchSize; ++batch)
    {
        gpuYoloLayer_r<<<number_of_blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const float*>(input) + (batch * outputSize),
            reinterpret_cast<float*>(output) + (batch * outputSize), gridSizeX, gridSizeY, numOutputClasses,
            numBBoxes, modelScale);
    }
    return cudaGetLastError();
}
