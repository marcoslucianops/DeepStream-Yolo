/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "yoloPlugins.h"
#include "NvInferPlugin.h"
#include <cassert>
#include <iostream>
#include <memory>

uint kNUM_CLASSES;

namespace {
    template <typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

cudaError_t cudaYoloLayer_e(
    const void* cls, const void* reg, void* d_indexes, void* d_scores, void* d_boxes, void* d_classes, void* countData,
    const uint& batchSize, uint64_t& outputSize, const float& scoreThreshold, const uint& netWidth, const uint& netHeight,
    const uint& numOutputClasses, cudaStream_t stream);

cudaError_t cudaYoloLayer_r(
    const void* input, void* d_indexes, void* d_scores, void* d_boxes, void* d_classes, void* countData,
    const uint& batchSize, uint64_t& inputSize, uint64_t& outputSize, const float& scoreThreshold, const uint& netWidth,
    const uint& netHeight, const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses, const uint& numBBoxes,
    const float& scaleXY, const void* anchors, const void* mask, cudaStream_t stream);

cudaError_t cudaYoloLayer_nc(
    const void* input, void* d_indexes, void* d_scores, void* d_boxes, void* d_classes, void* countData,
    const uint& batchSize, uint64_t& inputSize, uint64_t& outputSize, const float& scoreThreshold, const uint& netWidth,
    const uint& netHeight, const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses, const uint& numBBoxes,
    const float& scaleXY, const void* anchors, const void* mask, cudaStream_t stream);

cudaError_t cudaYoloLayer(
    const void* input, void* d_indexes, void* d_scores, void* d_boxes, void* d_classes, void* countData,
    const uint& batchSize, uint64_t& inputSize, uint64_t& outputSize, const float& scoreThreshold, const uint& netWidth,
    const uint& netHeight, const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses, const uint& numBBoxes,
    const float& scaleXY, const void* anchors, const void* mask, cudaStream_t stream);

cudaError_t cudaRegionLayer(
    const void* input, void* softmax, void* d_indexes, void* d_scores, void* d_boxes, void* d_classes, void* countData,
    const uint& batchSize, uint64_t& inputSize, uint64_t& outputSize, const float& scoreThreshold, const uint& netWidth,
    const uint& netHeight, const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses, const uint& numBBoxes,
    const void* anchors, cudaStream_t stream);

cudaError_t sortDetections(
    void* d_indexes, void* d_scores, void* d_boxes, void* d_classes, void* bboxData, void* scoreData, void* countData,
    const uint& batchSize, uint64_t& outputSize, uint& topK, const uint& numOutputClasses, cudaStream_t stream);

YoloLayer::YoloLayer (const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data);

    read(d, m_NetWidth);
    read(d, m_NetHeight);
    read(d, m_NumClasses);
    read(d, m_NewCoords);
    read(d, m_OutputSize);
    read(d, m_Type);
    read(d, m_TopK);
    read(d, m_ScoreThreshold);

    if (m_Type != 3) {
        uint yoloTensorsSize;
        read(d, yoloTensorsSize);
        for (uint i = 0; i < yoloTensorsSize; ++i)
        {
            TensorInfo curYoloTensor;
            read(d, curYoloTensor.gridSizeX);
            read(d, curYoloTensor.gridSizeY);
            read(d, curYoloTensor.numBBoxes);
            read(d, curYoloTensor.scaleXY);

            uint anchorsSize;
            read(d, anchorsSize);
            for (uint j = 0; j < anchorsSize; j++)
            {
                float result;
                read(d, result);
                curYoloTensor.anchors.push_back(result);
            }

            uint maskSize;
            read(d, maskSize);
            for (uint j = 0; j < maskSize; j++)
            {
                int result;
                read(d, result);
                curYoloTensor.mask.push_back(result);
            }
            m_YoloTensors.push_back(curYoloTensor);
        }
    }

    kNUM_CLASSES = m_NumClasses;
};

YoloLayer::YoloLayer(
    const uint& netWidth, const uint& netHeight, const uint& numClasses, const uint& newCoords,
    const std::vector<TensorInfo>& yoloTensors, const uint64_t& outputSize, const uint& modelType, const uint& topK,
    const float& scoreThreshold) :
    m_NetWidth(netWidth),
    m_NetHeight(netHeight),
    m_NumClasses(numClasses),
    m_NewCoords(newCoords),
    m_YoloTensors(yoloTensors),
    m_OutputSize(outputSize),
    m_Type(modelType),
    m_TopK(topK),
    m_ScoreThreshold(scoreThreshold)
{
    assert(m_NetWidth > 0);
    assert(m_NetHeight > 0);

    kNUM_CLASSES = m_NumClasses;
};

nvinfer1::Dims
YoloLayer::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept
{
    assert(index < 3);
    if (index == 0) {
        return nvinfer1::Dims{3, {static_cast<int>(m_TopK), 1, 4}};
    }
    return nvinfer1::Dims{2, {static_cast<int>(m_TopK), static_cast<int>(m_NumClasses)}};
}

bool YoloLayer::supportsFormat (
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept {
    return (type == nvinfer1::DataType::kFLOAT &&
            format == nvinfer1::PluginFormat::kLINEAR);
}

void
YoloLayer::configureWithFormat (
    const nvinfer1::Dims* inputDims, int nbInputs,
    const nvinfer1::Dims* outputDims, int nbOutputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) noexcept
{
    assert(nbInputs > 0);
    assert(format == nvinfer1::PluginFormat::kLINEAR);
    assert(inputDims != nullptr);
}

int32_t YoloLayer::enqueue (
    int batchSize, void const* const* inputs, void* const* outputs, void* workspace,	
    cudaStream_t stream) noexcept
{
    void* bboxData = outputs[0];
    void* scoreData = outputs[1];

    CUDA_CHECK(cudaMemsetAsync((float*)bboxData, 0, sizeof(float) * m_TopK * 4 * batchSize, stream));
    CUDA_CHECK(cudaMemsetAsync((float*)scoreData, 0, sizeof(float) * m_TopK * m_NumClasses * batchSize, stream));

    void* countData;
    CUDA_CHECK(cudaMalloc(&countData, sizeof(int) * batchSize));
    CUDA_CHECK(cudaMemsetAsync((int*)countData, 0, sizeof(int) * batchSize, stream));

    void* d_indexes;
    CUDA_CHECK(cudaMalloc(&d_indexes, sizeof(int) * m_OutputSize * batchSize));
    CUDA_CHECK(cudaMemsetAsync((int*)d_indexes, 0, sizeof(int) * m_OutputSize * batchSize, stream));

    void* d_scores;
    CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float) * m_OutputSize * batchSize));
    CUDA_CHECK(cudaMemsetAsync((float*)d_scores, 0, sizeof(float) * m_OutputSize * batchSize, stream));

    void* d_boxes;
    CUDA_CHECK(cudaMalloc(&d_boxes, sizeof(float) * m_OutputSize * 4 * batchSize));
    CUDA_CHECK(cudaMemsetAsync((float*)d_boxes, 0, sizeof(float) * m_OutputSize * 4 * batchSize, stream));

    void* d_classes;
    CUDA_CHECK(cudaMalloc(&d_classes, sizeof(int) * m_OutputSize * batchSize));
    CUDA_CHECK(cudaMemsetAsync((float*)d_classes, 0, sizeof(int) * m_OutputSize * batchSize, stream));

    if (m_Type == 3)
    {
        CUDA_CHECK(cudaYoloLayer_e(
            inputs[0], inputs[1], d_indexes, d_scores, d_boxes, d_classes, countData, batchSize, m_OutputSize,
            m_ScoreThreshold, m_NetWidth, m_NetHeight, m_NumClasses, stream));
    }
    else
    {
        uint yoloTensorsSize = m_YoloTensors.size();
        for (uint i = 0; i < yoloTensorsSize; ++i)
        {
            TensorInfo& curYoloTensor = m_YoloTensors.at(i);

            uint numBBoxes = curYoloTensor.numBBoxes;
            float scaleXY = curYoloTensor.scaleXY;
            uint gridSizeX = curYoloTensor.gridSizeX;
            uint gridSizeY = curYoloTensor.gridSizeY;
            std::vector<float> anchors = curYoloTensor.anchors;
            std::vector<int> mask = curYoloTensor.mask;

            void* v_anchors;
            void* v_mask;
            if (anchors.size() > 0) {
                float* f_anchors = anchors.data();
                CUDA_CHECK(cudaMalloc(&v_anchors, sizeof(float) * anchors.size()));
                CUDA_CHECK(cudaMemcpy(v_anchors, f_anchors, sizeof(float) * anchors.size(), cudaMemcpyHostToDevice));
            }
            if (mask.size() > 0) {
                int* f_mask = mask.data();
                CUDA_CHECK(cudaMalloc(&v_mask, sizeof(int) * mask.size()));
                CUDA_CHECK(cudaMemcpy(v_mask, f_mask, sizeof(int) * mask.size(), cudaMemcpyHostToDevice));
            }

            uint64_t inputSize = gridSizeX * gridSizeY * (numBBoxes * (4 + 1 + m_NumClasses));

            if (m_Type == 2) {  // YOLOR incorrect param: scale_x_y = 2.0
                CUDA_CHECK(cudaYoloLayer_r(
                    inputs[i], d_indexes, d_scores, d_boxes, d_classes, countData, batchSize, inputSize, m_OutputSize,
                    m_ScoreThreshold, m_NetWidth, m_NetHeight, gridSizeX, gridSizeY, m_NumClasses, numBBoxes, 2.0, v_anchors,
                    v_mask, stream));
            }
            else if (m_Type == 1) {
                if (m_NewCoords) {
                    CUDA_CHECK(cudaYoloLayer_nc(
                        inputs[i], d_indexes, d_scores, d_boxes, d_classes, countData, batchSize, inputSize, m_OutputSize,
                        m_ScoreThreshold, m_NetWidth, m_NetHeight, gridSizeX, gridSizeY, m_NumClasses, numBBoxes, scaleXY,
                        v_anchors, v_mask, stream));
                }
                else {
                    CUDA_CHECK(cudaYoloLayer(
                        inputs[i], d_indexes, d_scores, d_boxes, d_classes, countData, batchSize, inputSize, m_OutputSize,
                        m_ScoreThreshold, m_NetWidth, m_NetHeight, gridSizeX, gridSizeY, m_NumClasses, numBBoxes, scaleXY,
                        v_anchors, v_mask, stream));
                }
            }
            else {
                void* softmax;
                CUDA_CHECK(cudaMalloc(&softmax, sizeof(float) * inputSize * batchSize));
                CUDA_CHECK(cudaMemsetAsync((float*)softmax, 0, sizeof(float) * inputSize * batchSize));

                CUDA_CHECK(cudaRegionLayer(
                    inputs[i], softmax, d_indexes, d_scores, d_boxes, d_classes, countData, batchSize, inputSize, m_OutputSize,
                    m_ScoreThreshold, m_NetWidth, m_NetHeight, gridSizeX, gridSizeY, m_NumClasses, numBBoxes, v_anchors,
                    stream));

                CUDA_CHECK(cudaFree(softmax));
            }

            if (anchors.size() > 0) {
                CUDA_CHECK(cudaFree(v_anchors));
            }
            if (mask.size() > 0) {
                CUDA_CHECK(cudaFree(v_mask));
            }
        }
    }

    CUDA_CHECK(sortDetections(
        d_indexes, d_scores, d_boxes, d_classes, bboxData, scoreData, countData, batchSize, m_OutputSize, m_TopK,
        m_NumClasses, stream));

    CUDA_CHECK(cudaFree(countData));
    CUDA_CHECK(cudaFree(d_indexes));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_boxes));
    CUDA_CHECK(cudaFree(d_classes));

    return 0;
}

size_t YoloLayer::getSerializationSize() const noexcept
{
    size_t totalSize = 0;

    totalSize += sizeof(m_NetWidth);
    totalSize += sizeof(m_NetHeight);
    totalSize += sizeof(m_NumClasses);
    totalSize += sizeof(m_NewCoords);
    totalSize += sizeof(m_OutputSize);
    totalSize += sizeof(m_Type);
    totalSize += sizeof(m_TopK);
    totalSize += sizeof(m_ScoreThreshold);

    if (m_Type != 3) {
        uint yoloTensorsSize = m_YoloTensors.size();
        totalSize += sizeof(yoloTensorsSize);

        for (uint i = 0; i < yoloTensorsSize; ++i)
        {
            const TensorInfo& curYoloTensor = m_YoloTensors.at(i);
            totalSize += sizeof(curYoloTensor.gridSizeX);
            totalSize += sizeof(curYoloTensor.gridSizeY);
            totalSize += sizeof(curYoloTensor.numBBoxes);
            totalSize += sizeof(curYoloTensor.scaleXY);
            totalSize += sizeof(uint) + sizeof(curYoloTensor.anchors[0]) * curYoloTensor.anchors.size();
            totalSize += sizeof(uint) + sizeof(curYoloTensor.mask[0]) * curYoloTensor.mask.size();
        }
    }

    return totalSize;
}

void YoloLayer::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer);

    write(d, m_NetWidth);
    write(d, m_NetHeight);
    write(d, m_NumClasses);
    write(d, m_NewCoords);
    write(d, m_OutputSize);
    write(d, m_Type);
    write(d, m_TopK);
    write(d, m_ScoreThreshold);

    if (m_Type != 3) {
        uint yoloTensorsSize = m_YoloTensors.size();
        write(d, yoloTensorsSize);
        for (uint i = 0; i < yoloTensorsSize; ++i)
        {
            const TensorInfo& curYoloTensor = m_YoloTensors.at(i);
            write(d, curYoloTensor.gridSizeX);
            write(d, curYoloTensor.gridSizeY);
            write(d, curYoloTensor.numBBoxes);
            write(d, curYoloTensor.scaleXY);

            uint anchorsSize = curYoloTensor.anchors.size();
            write(d, anchorsSize);
            for (uint j = 0; j < anchorsSize; ++j)
            {
                write(d, curYoloTensor.anchors[j]);
            }

            uint maskSize = curYoloTensor.mask.size();
            write(d, maskSize);
            for (uint j = 0; j < maskSize; ++j)
            {
                write(d, curYoloTensor.mask[j]);
            }
        }
    }
}

nvinfer1::IPluginV2* YoloLayer::clone() const noexcept
{
    return new YoloLayer (
        m_NetWidth, m_NetHeight, m_NumClasses, m_NewCoords, m_YoloTensors, m_OutputSize, m_Type, m_TopK,
        m_ScoreThreshold);
}

REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);
