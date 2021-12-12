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

__global__ void gpuRegionLayer(const float* input, float* output, const uint gridSizeX, const uint gridSizeY, const uint numOutputClasses,
                               const uint numBBoxes)
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
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]);

    float temp = 1.0;
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < numOutputClasses; ++i){
        int val = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < numOutputClasses; ++i){
        float e = exp(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))] / temp - largest / temp);
        sum += e;
        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))] = e;
    }
    for(i = 0; i < numOutputClasses; ++i){
        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))] /= sum;
    }
}

cudaError_t cudaYoloLayer_v2(const void* input, void* output, const uint& batchSize, const uint& gridSizeX, const uint& gridSizeY,
                            const uint& numOutputClasses, const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream);

cudaError_t cudaYoloLayer_v2(const void* input, void* output, const uint& batchSize, const uint& gridSizeX, const uint& gridSizeY,
                            const uint& numOutputClasses, const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream)
{
    dim3 threads_per_block(16, 16, 4);
    dim3 number_of_blocks((gridSizeX / threads_per_block.x) + 1,
                          (gridSizeY / threads_per_block.y) + 1,
                          (numBBoxes / threads_per_block.z) + 1);
    for (unsigned int batch = 0; batch < batchSize; ++batch)
    {
        gpuRegionLayer<<<number_of_blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const float*>(input) + (batch * outputSize),
            reinterpret_cast<float*>(output) + (batch * outputSize), gridSizeX, gridSizeY, numOutputClasses,
            numBBoxes);
    }
    return cudaGetLastError();
}
