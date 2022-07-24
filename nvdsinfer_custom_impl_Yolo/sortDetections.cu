/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <cub/device/device_radix_sort.cuh>

__global__ void sortOutput(
    int* d_indexes, float* d_scores, float* d_boxes, int* d_classes, float* bboxData, float* scoreData,
    const uint numOutputClasses, const int topk)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_id >= topk)
        return;

    int index = d_indexes[x_id];
    int maxIndex = d_classes[index];
    bboxData[x_id * 4 + 0] = d_boxes[index * 4 + 0];
    bboxData[x_id * 4 + 1] = d_boxes[index * 4 + 1];
    bboxData[x_id * 4 + 2] = d_boxes[index * 4 + 2];
    bboxData[x_id * 4 + 3] = d_boxes[index * 4 + 3];
    scoreData[x_id * numOutputClasses + maxIndex] = d_scores[x_id] - 1.f;
}

cudaError_t sortDetections(
    void* d_indexes, void* d_scores, void* d_boxes, void* d_classes, void* bboxData, void* scoreData, void* countData,
    const uint& batchSize, uint64_t& outputSize, uint& topK, const uint& numOutputClasses, cudaStream_t stream);

cudaError_t sortDetections(
    void* d_indexes, void* d_scores, void* d_boxes, void* d_classes, void* bboxData, void* scoreData, void* countData,
    const uint& batchSize, uint64_t& outputSize, uint& topK, const uint& numOutputClasses, cudaStream_t stream)
{
    for (unsigned int batch = 0; batch < batchSize; ++batch)
    {
        int* _d_indexes = reinterpret_cast<int*>(d_indexes) + (batch * outputSize);
        float* _d_scores = reinterpret_cast<float*>(d_scores) + (batch * outputSize);

        int* _countData = reinterpret_cast<int*>(countData) + (batch);
        int count;
        cudaMemcpy(&count, _countData, sizeof(int), cudaMemcpyDeviceToHost);

        if (count == 0)
        {
            return cudaGetLastError();
        }

        size_t begin_bit = 0;
        size_t end_bit = sizeof(float) * 8;

        float *d_keys_out = NULL;
        int *d_values_out = NULL;

        cudaMalloc((void **)&d_keys_out, count * sizeof(float));
        cudaMalloc((void **)&d_values_out, count * sizeof(int));

        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, _d_scores, d_keys_out, _d_indexes,
        d_values_out, count, begin_bit, end_bit);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, _d_scores, d_keys_out, _d_indexes,
        d_values_out, count, begin_bit, end_bit);

        cudaMemcpy(_d_scores, d_keys_out, count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(_d_indexes, d_values_out, count * sizeof(int), cudaMemcpyDeviceToDevice);

        int _topK = count < topK ? count : topK;

        int threads_per_block = 16;
        int number_of_blocks = 0;

        if (_topK % 2 == 0 && _topK >= threads_per_block)
            number_of_blocks = _topK / threads_per_block;
        else
            number_of_blocks = (_topK / threads_per_block) + 1;

        sortOutput<<<number_of_blocks, threads_per_block, 0, stream>>>(
            _d_indexes, _d_scores, reinterpret_cast<float*>(d_boxes) + (batch * 4 * outputSize),
            reinterpret_cast<int*>(d_classes) + (batch * outputSize),
            reinterpret_cast<float*>(bboxData) + (batch * topK * 4),
            reinterpret_cast<float*>(scoreData) + (batch * topK * numOutputClasses), numOutputClasses, _topK);

        cudaFree(d_keys_out);
        cudaFree(d_values_out);
        cudaFree(d_temp_storage);
    }
    return cudaGetLastError();
}
