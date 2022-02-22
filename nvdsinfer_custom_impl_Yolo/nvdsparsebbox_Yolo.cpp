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
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include "nvdsinfer_custom_impl.h"
#include "utils.h"

#include "yoloPlugins.h"

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static NvDsInferParseObjectInfo convertBBox(
    const float& bx, const float& by, const float& bw,
    const float& bh, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;

    float x1 = bx - bw / 2;
    float y1 = by - bh / 2;
    float x2 = x1 + bw;
    float y2 = y1 + bh;

    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);
    x2 = clamp(x2, 0, netW);
    y2 = clamp(y2, 0, netH);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH);

    return b;
}

static void addBBoxProposal(
    const float bx, const float by, const float bw, const float bh,
    const uint& netW, const uint& netH, const int maxIndex,
    const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBox(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo> decodeYoloTensor(
    const float* detections,
    const uint gridSizeW, const uint gridSizeH, const uint numBBoxes,
    const uint numOutputClasses, const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const int numGridCells = gridSizeH * gridSizeW;
                const int bbindex = y * gridSizeW + x;

                const float bx
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                const float by
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                const float bw
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)];
                const float bh
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)];
                const float maxProb
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];
                const int maxIndex
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 5)];

                addBBoxProposal(bx, by, bw, bh, netW, netH, maxIndex, maxProb, binfo);
            }
        }
    }
    return binfo;
}

static bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList,
    const uint &numBBoxes,
    const uint &numClasses)
{
    if (outputLayersInfo.empty())
    {
        std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
        return false;
    }

    if (numClasses != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured: "
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << numClasses << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;

    for (uint idx = 0; idx < outputLayersInfo.size(); ++idx)
    {
        const NvDsInferLayerInfo &layer = outputLayersInfo[idx];

        assert(layer.inferDims.numDims == 3);
        const uint gridSizeH = layer.inferDims.d[1];
        const uint gridSizeW = layer.inferDims.d[2];

        std::vector<NvDsInferParseObjectInfo> outObjs =
            decodeYoloTensor(
                (const float*)(layer.buffer),
                gridSizeW, gridSizeH, numBBoxes, numClasses,
                networkInfo.width, networkInfo.height);

        objects.insert(objects.end(), outObjs.begin(), outObjs.end());
    }

    objectList = objects;

    return true;
}

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    uint numBBoxes = kNUM_BBOXES;
    uint numClasses = kNUM_CLASSES;

    return NvDsInferParseCustomYolo (
        outputLayersInfo, networkInfo, detectionParams, objectList, numBBoxes, numClasses);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
