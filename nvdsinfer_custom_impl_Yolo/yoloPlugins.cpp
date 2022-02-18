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

 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "yoloPlugins.h"
#include "NvInferPlugin.h"
#include <cassert>
#include <iostream>
#include <memory>

int kMODEL_TYPE;
int kNUM_BBOXES;
int kNUM_CLASSES;
float kBETA_NMS;

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

cudaError_t cudaYoloLayer_r (
    const void* input, void* output, const uint& batchSize,
    const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses,
    const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream, const float scaleXY,
    const void* anchors, const void* mask);

cudaError_t cudaYoloLayer_nc (
    const void* input, void* output, const uint& batchSize,
    const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses,
    const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream, const float scaleXY,
    const void* anchors, const void* mask);

cudaError_t cudaYoloLayer (
    const void* input, void* output, const uint& batchSize,
    const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses,
    const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream, const float scaleXY,
    const void* anchors, const void* mask);

cudaError_t cudaYoloLayer_v2 (
    const void* input, void* output, void* softmax, const uint& batchSize,
    const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses,
    const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream, const void* anchors);

YoloLayer::YoloLayer (const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data);
    read(d, m_NumBoxes);
    read(d, m_NumClasses);
    read(d, m_GridSizeX);
    read(d, m_GridSizeY);
    read(d, m_OutputSize);

    read(d, m_Type);
    read(d, m_NewCoords);
    read(d, m_ScaleXY);
    read(d, m_BetaNMS);

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

    kMODEL_TYPE = m_Type;
    kNUM_BBOXES = m_NumBoxes;
    kNUM_CLASSES = m_NumClasses;
    kBETA_NMS = m_BetaNMS;

    if (m_Anchors.size() > 0) {
        float* m_anchors = m_Anchors.data();
        CHECK(cudaMallocHost(&mAnchors, m_Anchors.size() * sizeof(float)));
        CHECK(cudaMemcpy(mAnchors, m_anchors, m_Anchors.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    if (m_Mask.size() > 0) {
        int* m_mask = m_Mask.data();
        CHECK(cudaMallocHost(&mMask, m_Mask.size() * sizeof(int)));
        CHECK(cudaMemcpy(mMask, m_mask, m_Mask.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
};

YoloLayer::YoloLayer (
    const uint& numBoxes, const uint& numClasses, const uint& gridSizeX, const uint& gridSizeY, const uint modelType, const uint newCoords, const float scaleXY, const float betaNMS, const std::vector<float> anchors, std::vector<int> mask) :
    m_NumBoxes(numBoxes),
    m_NumClasses(numClasses),
    m_GridSizeX(gridSizeX),
    m_GridSizeY(gridSizeY),
    m_Type(modelType),
    m_NewCoords(newCoords),
    m_ScaleXY(scaleXY),
    m_BetaNMS(betaNMS),
    m_Anchors(anchors),
    m_Mask(mask)
{
    assert(m_NumBoxes > 0);
    assert(m_NumClasses > 0);
    assert(m_GridSizeX > 0);
    assert(m_GridSizeY > 0);
    m_OutputSize = m_GridSizeX * m_GridSizeY * (m_NumBoxes * (4 + 1 + m_NumClasses));

    if (m_Anchors.size() > 0) {
        float* m_anchors = m_Anchors.data();
        CHECK(cudaMallocHost(&mAnchors, m_Anchors.size() * sizeof(float)));
        CHECK(cudaMemcpy(mAnchors, m_anchors, m_Anchors.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    if (m_Mask.size() > 0) {
        int* m_mask = m_Mask.data();
        CHECK(cudaMallocHost(&mMask, m_Mask.size() * sizeof(int)));
        CHECK(cudaMemcpy(mMask, m_mask, m_Mask.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
};

YoloLayer::~YoloLayer()
{
    if (m_Anchors.size() > 0) {
        CHECK(cudaFreeHost(mAnchors));
    }
    if (m_Mask.size() > 0) {
        CHECK(cudaFreeHost(mMask));
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
    assert (format == nvinfer1::PluginFormat::kLINEAR);
    assert(inputDims != nullptr);
}

int YoloLayer::enqueue(
    int batchSize, void const* const* inputs, void* const* outputs, void* workspace,	
    cudaStream_t stream) noexcept
{
    if (m_Type == 2) { // YOLOR incorrect param: scale_x_y = 2.0
        CHECK(cudaYoloLayer_r(
            inputs[0], outputs[0], batchSize, m_GridSizeX, m_GridSizeY, m_NumClasses, m_NumBoxes,
            m_OutputSize, stream, 2.0, mAnchors, mMask));
    }
    else if (m_Type == 1) {
        if (m_NewCoords) {
            CHECK(cudaYoloLayer_nc(
                inputs[0], outputs[0], batchSize, m_GridSizeX, m_GridSizeY, m_NumClasses, m_NumBoxes,
                m_OutputSize, stream, m_ScaleXY, mAnchors, mMask));
        }
        else {
            CHECK(cudaYoloLayer(
                inputs[0], outputs[0], batchSize, m_GridSizeX, m_GridSizeY, m_NumClasses, m_NumBoxes,
                m_OutputSize, stream, m_ScaleXY, mAnchors, mMask));
        }
    }
    else {
        void* softmax;
        CHECK(cudaMallocHost(&softmax, sizeof(outputs[0])));
        CHECK(cudaMemcpy(softmax, outputs[0], sizeof(outputs[0]), cudaMemcpyHostToDevice));

        CHECK(cudaYoloLayer_v2(
            inputs[0], outputs[0], softmax, batchSize, m_GridSizeX, m_GridSizeY, m_NumClasses, m_NumBoxes,
            m_OutputSize, stream, mAnchors));

        CHECK(cudaFreeHost(softmax));
    }
    return 0;
}

size_t YoloLayer::getSerializationSize() const noexcept
{
    int anchorsSum = 1;
    for (uint i = 0; i < m_Anchors.size(); i++) {
        anchorsSum += 1;
    }
    int maskSum = 1;
    for (uint i = 0; i < m_Mask.size(); i++) {
        maskSum += 1;
    }

    return sizeof(m_NumBoxes) + sizeof(m_NumClasses) + sizeof(m_GridSizeX) + sizeof(m_GridSizeY) + sizeof(m_OutputSize) + sizeof(m_Type)
            + sizeof(m_NewCoords) + sizeof(m_ScaleXY) + sizeof(m_BetaNMS) + anchorsSum * sizeof(float) + maskSum * sizeof(int);
}

void YoloLayer::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer);
    write(d, m_NumBoxes);
    write(d, m_NumClasses);
    write(d, m_GridSizeX);
    write(d, m_GridSizeY);
    write(d, m_OutputSize);

    write(d, m_Type);
    write(d, m_NewCoords);
    write(d, m_ScaleXY);
    write(d, m_BetaNMS);

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

    kMODEL_TYPE = m_Type;
    kNUM_BBOXES = m_NumBoxes;
    kNUM_CLASSES = m_NumClasses;
    kBETA_NMS = m_BetaNMS;
}

nvinfer1::IPluginV2* YoloLayer::clone() const noexcept
{
    return new YoloLayer (m_NumBoxes, m_NumClasses, m_GridSizeX, m_GridSizeY, m_Type, m_NewCoords, m_ScaleXY, m_BetaNMS, m_Anchors, m_Mask);
}

REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);
