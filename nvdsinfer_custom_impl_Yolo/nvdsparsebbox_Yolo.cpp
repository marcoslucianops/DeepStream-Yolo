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

static std::vector<NvDsInferParseObjectInfo>
nonMaximumSuppression(const float nmsThresh, std::vector<NvDsInferParseObjectInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU
        = [&overlap1D](NvDsInferParseObjectInfo& bbox1, NvDsInferParseObjectInfo& bbox2) -> float {
        float overlapX
            = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY
            = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const NvDsInferParseObjectInfo& b1, const NvDsInferParseObjectInfo& b2) {
                         return b1.detectionConfidence > b2.detectionConfidence;
                     });
    std::vector<NvDsInferParseObjectInfo> out;
    for (auto i : binfo)
    {
        bool keep = true;
        for (auto j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}

static std::vector<NvDsInferParseObjectInfo>
nmsAllClasses(const float nmsThresh,
        std::vector<NvDsInferParseObjectInfo>& binfo,
        const uint numClasses)
{
    std::vector<NvDsInferParseObjectInfo> result;
    std::vector<std::vector<NvDsInferParseObjectInfo>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.classId).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }
    return result;
}

static NvDsInferParseObjectInfo convertBBox(const float& bx, const float& by, const float& bw,
                                     const float& bh, const int& stride, const uint& netW,
                                     const uint& netH)
{
    NvDsInferParseObjectInfo b;
    float xCenter = bx * stride;
    float yCenter = by * stride;
    float x0 = xCenter - bw / 2;
    float y0 = yCenter - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;

    x0 = clamp(x0, 0, netW);
    y0 = clamp(y0, 0, netH);
    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);

    b.left = x0;
    b.width = clamp(x1 - x0, 0, netW);
    b.top = y0;
    b.height = clamp(y1 - y0, 0, netH);

    return b;
}

static void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                     const uint stride, const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBox(bx, by, bw, bh, stride, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloTensor(
    const float* detections, const std::vector<int> &mask, const std::vector<float> &anchors,
    const uint gridSizeW, const uint gridSizeH, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const uint& netW,
    const uint& netH,
    const float confThresh)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[mask[b] * 2];
                const float ph = anchors[mask[b] * 2 + 1];

                const int numGridCells = gridSizeH * gridSizeW;
                const int bbindex = y * gridSizeW + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                const float bw
                    = pw * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)];
                const float bh
                    = ph * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)];

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                if (maxProb > confThresh)
                {
                    addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
                }
            }
        }
    }
    return binfo;
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV2Tensor(
    const float* detections, const std::vector<float> &anchors,
    const uint gridSizeW, const uint gridSizeH, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const uint& netW,
    const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[b * 2];
                const float ph = anchors[b * 2 + 1];

                const int numGridCells = gridSizeH * gridSizeW;
                const int bbindex = y * gridSizeW + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                const float bw
                    = pw * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)];
                const float bh
                    = ph * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)];

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
            }
        }
    }
    return binfo;
}

static inline std::vector<const NvDsInferLayerInfo*>
SortLayers(const std::vector<NvDsInferLayerInfo> & outputLayersInfo)
{
    std::vector<const NvDsInferLayerInfo*> outLayers;
    for (auto const &layer : outputLayersInfo) {
        outLayers.push_back (&layer);
    }
    std::sort(outLayers.begin(), outLayers.end(),
        [](const NvDsInferLayerInfo* a, const NvDsInferLayerInfo* b) {
            return a->inferDims.d[1] < b->inferDims.d[1];
        });
    return outLayers;
}

static bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList,
    const std::vector<float> &anchors,
    const std::vector<std::vector<int>> &masks,
    const uint &num_classes,
    const float &beta_nms)
{
    const float kCONF_THRESH = detectionParams.perClassThreshold[0];

    const std::vector<const NvDsInferLayerInfo*> sortedLayers =
        SortLayers (outputLayersInfo);

    if (sortedLayers.size() != masks.size()) {
        std::cerr << "ERROR: YOLO output layer.size: " << sortedLayers.size()
                  << " does not match mask.size: " << masks.size() << std::endl;
        return false;
    }

    if (num_classes != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured: "
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << num_classes << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;

    for (uint idx = 0; idx < masks.size(); ++idx) {
        const NvDsInferLayerInfo &layer = *sortedLayers[idx]; // 255 x Grid x Grid

        assert(layer.inferDims.numDims == 3);
        const uint gridSizeH = layer.inferDims.d[1];
        const uint gridSizeW = layer.inferDims.d[2];
        const uint stride = DIVUP(networkInfo.width, gridSizeW);

        std::vector<NvDsInferParseObjectInfo> outObjs =
            decodeYoloTensor((const float*)(layer.buffer), masks[idx], anchors, gridSizeW, gridSizeH, stride, masks[idx].size(),
                       num_classes, networkInfo.width, networkInfo.height, kCONF_THRESH);
        objects.insert(objects.end(), outObjs.begin(), outObjs.end());
    }


    objectList.clear();
    objectList = nmsAllClasses(beta_nms, objects, num_classes);

    return true;
}

static bool NvDsInferParseYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList,
    std::vector<float> &anchors,
    const uint &num_classes)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }
    const uint kNUM_BBOXES = anchors.size() / 2;

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    if (num_classes != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured: "
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << num_classes << std::endl;
    }

    assert(layer.inferDims.numDims == 3);
    const uint gridSizeH = layer.inferDims.d[1];
    const uint gridSizeW = layer.inferDims.d[2];
    const uint stride = DIVUP(networkInfo.width, gridSizeW);
    for (auto& anchor : anchors) {
        anchor *= stride;
    }
    std::vector<NvDsInferParseObjectInfo> objects =
        decodeYoloV2Tensor((const float*)(layer.buffer), anchors, gridSizeW, gridSizeH, stride, kNUM_BBOXES,
                   num_classes, networkInfo.width, networkInfo.height);

    objectList = objects;

    return true;
}

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{

    int num_classes = kNUM_CLASSES;
    float beta_nms = kBETA_NMS;
    std::vector<float> anchors = kANCHORS;
    std::vector<std::vector<int>> mask = kMASK;

    if (mask.size() > 0) {
        return NvDsInferParseYolo (outputLayersInfo, networkInfo, detectionParams, objectList, anchors, mask, num_classes, beta_nms);
    }
    else {
        return NvDsInferParseYoloV2 (outputLayersInfo, networkInfo, detectionParams, objectList, anchors, num_classes);
    }
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);