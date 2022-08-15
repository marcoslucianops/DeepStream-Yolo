/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <stdint.h>
#include <stdio.h>

__global__ void gpuYoloLayer_e(
    const float* cls, const float* reg, int* num_detections, float* detection_boxes, float* detection_scores,
    int* detection_classes, const float scoreThreshold, const uint netWidth, const uint netHeight,
    const uint numOutputClasses, const uint64_t outputSize)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_id >= outputSize)
        return;

    float maxProb = 0.0f;
    int maxIndex = -1;

    for (uint i = 0; i < numOutputClasses; ++i)
    {
        float prob
            = cls[x_id * numOutputClasses + i];

        if (prob > maxProb)
        {
            maxProb = prob;
            maxIndex = i;
        }
    }

    if (maxProb < scoreThreshold)
        return;

    int count = (int)atomicAdd(num_detections, 1);

    detection_boxes[count * 4 + 0] = reg[x_id * 4 + 0];
    detection_boxes[count * 4 + 1] = reg[x_id * 4 + 1];
    detection_boxes[count * 4 + 2] = reg[x_id * 4 + 2];
    detection_boxes[count * 4 + 3] = reg[x_id * 4 + 3];
    detection_scores[count] = maxProb;
    detection_classes[count] = maxIndex;
}

cudaError_t cudaYoloLayer_e(
    const void* cls, const void* reg, void* num_detections, void* detection_boxes, void* detection_scores,
    void* detection_classes, const uint& batchSize, uint64_t& outputSize, const float& scoreThreshold, const uint& netWidth,
    const uint& netHeight, const uint& numOutputClasses, cudaStream_t stream);

cudaError_t cudaYoloLayer_e(
    const void* cls, const void* reg, void* num_detections, void* detection_boxes, void* detection_scores,
    void* detection_classes, const uint& batchSize, uint64_t& outputSize, const float& scoreThreshold, const uint& netWidth,
    const uint& netHeight, const uint& numOutputClasses, cudaStream_t stream)
{
    int threads_per_block = 16;
    int number_of_blocks = (outputSize / threads_per_block) + 1;

    for (unsigned int batch = 0; batch < batchSize; ++batch)
    {
        gpuYoloLayer_e<<<number_of_blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const float*>(cls) + (batch * numOutputClasses * outputSize),
            reinterpret_cast<const float*>(reg) + (batch * 4 * outputSize),
            reinterpret_cast<int*>(num_detections) + (batch),
            reinterpret_cast<float*>(detection_boxes) + (batch * 4 * outputSize),
            reinterpret_cast<float*>(detection_scores) + (batch * outputSize),
            reinterpret_cast<int*>(detection_classes) + (batch * outputSize),
            scoreThreshold, netWidth, netHeight, numOutputClasses, outputSize);
    }
    return cudaGetLastError();
}
