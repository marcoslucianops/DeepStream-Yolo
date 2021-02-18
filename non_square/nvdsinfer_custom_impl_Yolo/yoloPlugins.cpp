/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

int kNUM_CLASSES;
float kBETA_NMS;
std::vector<float> kANCHORS;
std::vector<std::vector<int>> kMASK;

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

cudaError_t cudaYoloLayer (
    const void* input, void* output, const uint& batchSize,
    const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses,
    const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream, const uint modelCoords, const float modelScale, const uint modelType);

YoloLayer::YoloLayer (const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data);
    read(d, m_NumBoxes);
    read(d, m_NumClasses);
    read(d, m_GridSizeX);
    read(d, m_GridSizeY);
    read(d, m_OutputSize);

    read(d, m_type);
    read(d, m_new_coords);
    read(d, m_scale_x_y);
    read(d, m_beta_nms);
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
        uint nMask;
        read(d, nMask);
        std::vector<int> pMask;
        for (uint f = 0; f < nMask; f++) {
            int result;
            read(d, result);
            pMask.push_back(result);
        }
        m_Mask.push_back(pMask);
    }
    kNUM_CLASSES = m_NumClasses;
    kBETA_NMS = m_beta_nms;
    kANCHORS = m_Anchors;
    kMASK = m_Mask;
};

YoloLayer::YoloLayer (
    const uint& numBoxes, const uint& numClasses, const uint& gridSizeX, const uint& gridSizeY, const uint model_type, const uint new_coords, const float scale_x_y, const float beta_nms, const std::vector<float> anchors, std::vector<std::vector<int>> mask) :
    m_NumBoxes(numBoxes),
    m_NumClasses(numClasses),
    m_GridSizeX(gridSizeX),
    m_GridSizeY(gridSizeY),
    m_type(model_type),
    m_new_coords(new_coords),
    m_scale_x_y(scale_x_y),
    m_beta_nms(beta_nms),
    m_Anchors(anchors),
    m_Mask(mask)
{
    assert(m_NumBoxes > 0);
    assert(m_NumClasses > 0);
    assert(m_GridSizeX > 0);
    assert(m_GridSizeY > 0);
    m_OutputSize = m_GridSizeX * m_GridSizeY * (m_NumBoxes * (4 + 1 + m_NumClasses));
};

nvinfer1::Dims
YoloLayer::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    assert(index == 0);
    assert(nbInputDims == 1);
    return inputs[0];
}

bool YoloLayer::supportsFormat (
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const {
    return (type == nvinfer1::DataType::kFLOAT &&
            format == nvinfer1::PluginFormat::kNCHW);
}

void
YoloLayer::configureWithFormat (
    const nvinfer1::Dims* inputDims, int nbInputs,
    const nvinfer1::Dims* outputDims, int nbOutputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
{
    assert(nbInputs == 1);
    assert (format == nvinfer1::PluginFormat::kNCHW);
    assert(inputDims != nullptr);
}

int YoloLayer::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace,
    cudaStream_t stream)
{
    CHECK(cudaYoloLayer(
              inputs[0], outputs[0], batchSize, m_GridSizeX, m_GridSizeY, m_NumClasses, m_NumBoxes,
              m_OutputSize, stream, m_new_coords, m_scale_x_y, m_type));
    return 0;
}

size_t YoloLayer::getSerializationSize() const
{
    int anchorsSum = 1;
    for (uint i = 0; i < m_Anchors.size(); i++) {
        anchorsSum += 1;
    }
    int maskSum = 1;
    for (uint i = 0; i < m_Mask.size(); i++) {
        maskSum += 1;
        for (uint f = 0; f < m_Mask[i].size(); f++) {
            maskSum += 1;
        }
    }

    return sizeof(m_NumBoxes) + sizeof(m_NumClasses) + sizeof(m_GridSizeX) + sizeof(m_GridSizeY) + sizeof(m_OutputSize) + sizeof(m_type)
            + sizeof(m_new_coords) + sizeof(m_scale_x_y) + sizeof(m_beta_nms) + anchorsSum * sizeof(float) + maskSum * sizeof(int);
}

void YoloLayer::serialize(void* buffer) const
{
    char *d = static_cast<char*>(buffer);
    write(d, m_NumBoxes);
    write(d, m_NumClasses);
    write(d, m_GridSizeX);
    write(d, m_GridSizeY);
    write(d, m_OutputSize);

    write(d, m_type);
    write(d, m_new_coords);
    write(d, m_scale_x_y);
    write(d, m_beta_nms);
    uint anchorsSize = m_Anchors.size();
    write(d, anchorsSize);
    for (uint i = 0; i < anchorsSize; i++) {
        write(d, m_Anchors[i]);
    }
    uint maskSize = m_Mask.size();
    write(d, maskSize);
    for (uint i = 0; i < maskSize; i++) {
        uint pMaskSize = m_Mask[i].size();
        write(d, pMaskSize);
        for (uint f = 0; f < pMaskSize; f++) {
            write(d, m_Mask[i][f]);
        }
    }
    kNUM_CLASSES = m_NumClasses;
    kBETA_NMS = m_beta_nms;
    kANCHORS = m_Anchors;
    kMASK = m_Mask;
}

nvinfer1::IPluginV2* YoloLayer::clone() const
{
    return new YoloLayer (m_NumBoxes, m_NumClasses, m_GridSizeX, m_GridSizeY, m_type, m_new_coords, m_scale_x_y, m_beta_nms, m_Anchors, m_Mask);
}

REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);