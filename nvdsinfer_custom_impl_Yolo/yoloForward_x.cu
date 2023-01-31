/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <stdint.h>

__global__ void gpuYoloLayer_x(const float* input, int* num_detections, float* detection_boxes, float* detection_scores,
    int* detection_classes, const float scoreThreshold, const uint netWidth, const uint netHeight,
    const uint numOutputClasses, const uint64_t outputSize, const float* anchors, const int* mask)
{
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (x_id >= outputSize)
    return;

  const float objectness = input[x_id * (5 + numOutputClasses) + 4];

  if (objectness < scoreThreshold)
    return;

  int count = (int)atomicAdd(num_detections, 1);

  float x = (input[x_id * (5 + numOutputClasses) + 0] + anchors[x_id * 2]) * mask[x_id];

  float y = (input[x_id * (5 + numOutputClasses) + 1] + anchors[x_id * 2 + 1]) * mask[x_id];

  float w = __expf(input[x_id * (5 + numOutputClasses) + 2]) * mask[x_id];

  float h = __expf(input[x_id * (5 + numOutputClasses) + 3]) * mask[x_id];

  float maxProb = 0.0f;
  int maxIndex = -1;

  for (uint i = 0; i < numOutputClasses; ++i) {
    float prob = input[x_id * (5 + numOutputClasses) + 5 + i];
    if (prob > maxProb) {
      maxProb = prob;
      maxIndex = i;
    }
  }

  detection_boxes[count * 4 + 0] = x - 0.5 * w;
  detection_boxes[count * 4 + 1] = y - 0.5 * h;
  detection_boxes[count * 4 + 2] = x + 0.5 * w;
  detection_boxes[count * 4 + 3] = y + 0.5 * h;
  detection_scores[count] = objectness * maxProb;
  detection_classes[count] = maxIndex;
}

cudaError_t cudaYoloLayer_x(const void* input, void* num_detections, void* detection_boxes, void* detection_scores,
    void* detection_classes, const uint& batchSize, uint64_t& outputSize, const float& scoreThreshold, const uint& netWidth,
    const uint& netHeight, const uint& numOutputClasses, const void* anchors, const void* mask, cudaStream_t stream);

cudaError_t cudaYoloLayer_x(const void* input, void* num_detections, void* detection_boxes, void* detection_scores,
    void* detection_classes, const uint& batchSize, uint64_t& outputSize, const float& scoreThreshold, const uint& netWidth,
    const uint& netHeight, const uint& numOutputClasses, const void* anchors, const void* mask, cudaStream_t stream)
{
  int threads_per_block = 16;
  int number_of_blocks = (outputSize / threads_per_block) + 1;

  for (unsigned int batch = 0; batch < batchSize; ++batch) {
    gpuYoloLayer_x<<<number_of_blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<const float*>(input) + (batch * (5 + numOutputClasses) * outputSize),
        reinterpret_cast<int*>(num_detections) + (batch),
        reinterpret_cast<float*>(detection_boxes) + (batch * 4 * outputSize),
        reinterpret_cast<float*>(detection_scores) + (batch * outputSize),
        reinterpret_cast<int*>(detection_classes) + (batch * outputSize),
        scoreThreshold, netWidth, netHeight, numOutputClasses, outputSize, reinterpret_cast<const float*>(anchors),
        reinterpret_cast<const int*>(mask));
  }
  return cudaGetLastError();
}
