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

uint kNUM_BBOXES;
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

cudaError_t cudaYoloLayer_r(
    const void* input, void* output, const uint& batchSize, const uint& netWidth, const uint& netHeight,
    const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses, const uint& numBBoxes,
    uint64_t& outputSize, const float& scaleXY, const void* anchors, const void* mask, cudaStream_t stream);

cudaError_t cudaYoloLayer_nc(
    const void* input, void* output, const uint& batchSize, const uint& netWidth, const uint& netHeight,
    const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses, const uint& numBBoxes,
    uint64_t& outputSize, const float& scaleXY, const void* anchors, const void* mask, cudaStream_t stream);

cudaError_t cudaYoloLayer(
    const void* input, void* output, const uint& batchSize, const uint& netWidth, const uint& netHeight,
    const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses, const uint& numBBoxes,
    uint64_t& outputSize, const float& scaleXY, const void* anchors, const void* mask, cudaStream_t stream);

cudaError_t cudaRegionLayer(
    const void* input, void* output, void* softmax, const uint& batchSize, const uint& netWidth,
    const uint& netHeight, const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses,
    const uint& numBBoxes, uint64_t& outputSize, const void* anchors, cudaStream_t stream);

YoloLayer::YoloLayer (const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data);

    read(d, m_NumBBoxes);
    read(d, m_NumClasses);
    read(d, m_NetWidth);
    read(d, m_NetHeight);
    read(d, m_GridSizeX);
    read(d, m_GridSizeY);
    read(d, m_Type);
    read(d, m_NewCoords);
    read(d, m_ScaleXY);
    read(d, m_OutputSize);

    uint anchorsSize;
    read(d, anchorsSize);
    for (uint i = 0; i < anchorsSize; i++) {
        float result;
        read(d, result);
        m_Anchors.push_back(result);
    }

    uint maskSize;
    read(d, maskSize);
    for (uint i = 0; i < maskSize; i++) {
        int result;
        read(d, result);
        m_Mask.push_back(result);
    }

    if (m_Anchors.size() > 0) {
        float* anchors = m_Anchors.data();
        CUDA_CHECK(cudaMallocHost(&p_Anchors, m_Anchors.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(p_Anchors, anchors, m_Anchors.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    if (m_Mask.size() > 0) {
        int* mask = m_Mask.data();
        CUDA_CHECK(cudaMallocHost(&p_Mask, m_Mask.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(p_Mask, mask, m_Mask.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    kNUM_BBOXES = m_NumBBoxes;
    kNUM_CLASSES = m_NumClasses;
};

YoloLayer::YoloLayer (
    const uint& numBBoxes, const uint& numClasses, const uint& netWidth, const uint& netHeight,
    const uint& gridSizeX, const uint& gridSizeY, const uint& modelType, const uint& newCoords,
    const float& scaleXY, const std::vector<float> anchors,
    const std::vector<int> mask) :
    m_NumBBoxes(numBBoxes),
    m_NumClasses(numClasses),
    m_NetWidth(netWidth),
    m_NetHeight(netHeight),
    m_GridSizeX(gridSizeX),
    m_GridSizeY(gridSizeY),
    m_Type(modelType),
    m_NewCoords(newCoords),
    m_ScaleXY(scaleXY),
    m_Anchors(anchors),
    m_Mask(mask)
{
    assert(m_NumBBoxes > 0);
    assert(m_NumClasses > 0);
    assert(m_NetWidth > 0);
    assert(m_NetHeight > 0);
    assert(m_GridSizeX > 0);
    assert(m_GridSizeY > 0);

    m_OutputSize = m_GridSizeX * m_GridSizeY * (m_NumBBoxes * (4 + 1 + m_NumClasses));

    if (m_Anchors.size() > 0) {
        float* anchors = m_Anchors.data();
        CUDA_CHECK(cudaMallocHost(&p_Anchors, m_Anchors.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(p_Anchors, anchors, m_Anchors.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    if (m_Mask.size() > 0) {
        int* mask = m_Mask.data();
        CUDA_CHECK(cudaMallocHost(&p_Mask, m_Mask.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(p_Mask, mask, m_Mask.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    kNUM_BBOXES = m_NumBBoxes;
    kNUM_CLASSES = m_NumClasses;
};

YoloLayer::~YoloLayer()
{
    if (m_Anchors.size() > 0) {
        CUDA_CHECK(cudaFreeHost(p_Anchors));
    }
    if (m_Mask.size() > 0) {
        CUDA_CHECK(cudaFreeHost(p_Mask));
    }
}

nvinfer1::Dims
YoloLayer::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept
{
    assert(index == 0);
    assert(nbInputDims == 1);
    return inputs[0];
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
    assert(nbInputs == 1);
    assert(format == nvinfer1::PluginFormat::kLINEAR);
    assert(inputDims != nullptr);
}

int YoloLayer::enqueue (
    int batchSize, void const* const* inputs, void* const* outputs, void* workspace,	
    cudaStream_t stream) noexcept
{
    if (m_Type == 2) { // YOLOR incorrect param: scale_x_y = 2.0
        CUDA_CHECK(cudaYoloLayer_r(
            inputs[0], outputs[0], batchSize, m_NetWidth, m_NetHeight, m_GridSizeX, m_GridSizeY,
            m_NumClasses, m_NumBBoxes, m_OutputSize, 2.0, p_Anchors, p_Mask, stream));
    }
    else if (m_Type == 1) {
        if (m_NewCoords) {
            CUDA_CHECK(cudaYoloLayer_nc(
                inputs[0], outputs[0], batchSize, m_NetWidth, m_NetHeight, m_GridSizeX, m_GridSizeY,
                m_NumClasses, m_NumBBoxes, m_OutputSize, m_ScaleXY, p_Anchors, p_Mask, stream));
        }
        else {
            CUDA_CHECK(cudaYoloLayer(
                inputs[0], outputs[0], batchSize, m_NetWidth, m_NetHeight, m_GridSizeX, m_GridSizeY,
                m_NumClasses, m_NumBBoxes, m_OutputSize, m_ScaleXY, p_Anchors, p_Mask, stream));
        }
    }
    else {
        void* softmax;
        cudaMallocHost(&softmax, sizeof(outputs[0]));
        cudaMemcpy(softmax, outputs[0], sizeof(outputs[0]), cudaMemcpyHostToDevice);

        CUDA_CHECK(cudaRegionLayer(
            inputs[0], outputs[0], softmax, batchSize, m_NetWidth, m_NetHeight, m_GridSizeX, m_GridSizeY,
            m_NumClasses, m_NumBBoxes, m_OutputSize, p_Anchors, stream));

        CUDA_CHECK(cudaFreeHost(softmax));
    }
    return 0;
}

size_t YoloLayer::getSerializationSize() const noexcept
{
    size_t totalSize = 0;

    totalSize += sizeof(m_NumBBoxes);
    totalSize += sizeof(m_NumClasses);
    totalSize += sizeof(m_NetWidth);
    totalSize += sizeof(m_NetHeight);
    totalSize += sizeof(m_GridSizeX);
    totalSize += sizeof(m_GridSizeY);
    totalSize += sizeof(m_Type);
    totalSize += sizeof(m_NewCoords);
    totalSize += sizeof(m_ScaleXY);
    totalSize += sizeof(m_OutputSize);
    totalSize += sizeof(uint) + sizeof(m_Anchors[0]) * m_Anchors.size();
    totalSize += sizeof(uint) + sizeof(m_Mask[0]) * m_Mask.size();

    return totalSize;
}

void YoloLayer::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer);

    write(d, m_NumBBoxes);
    write(d, m_NumClasses);
    write(d, m_NetWidth);
    write(d, m_NetHeight);
    write(d, m_GridSizeX);
    write(d, m_GridSizeY);
    write(d, m_Type);
    write(d, m_NewCoords);
    write(d, m_ScaleXY);
    write(d, m_OutputSize);

    uint anchorsSize = m_Anchors.size();
    write(d, anchorsSize);
    for (uint i = 0; i < anchorsSize; i++) {
        write(d, m_Anchors[i]);
    }

    uint maskSize = m_Mask.size();
    write(d, maskSize);
    for (uint i = 0; i < maskSize; i++) {
        write(d, m_Mask[i]);
    }
}

nvinfer1::IPluginV2* YoloLayer::clone() const noexcept
{
    return new YoloLayer (
        m_NumBBoxes, m_NumClasses, m_NetWidth, m_NetHeight, m_GridSizeX, m_GridSizeY, m_Type,
        m_NewCoords, m_ScaleXY, m_Anchors, m_Mask);
}

REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);
