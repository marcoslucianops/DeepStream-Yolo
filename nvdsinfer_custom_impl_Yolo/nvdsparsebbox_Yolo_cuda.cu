/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "nvdsinfer_custom_impl.h"

extern "C" bool
NvDsInferParseYolo_cuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool
NvDsInferParseYoloE_cuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

__global__ void decodeTensorYolo_cuda(NvDsInferParseObjectInfo *binfo, float* input, int outputSize, int netW, int netH,
    float minPreclusterThreshold)
{
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (x_id >= outputSize)
    return;

  float maxProb = input[x_id * 6 + 4];
  int maxIndex = (int) input[x_id * 6 + 5];

  if (maxProb < minPreclusterThreshold) {
    binfo[x_id].detectionConfidence = 0.0;
    return;
  }

  float bxc = input[x_id * 6 + 0];
  float byc = input[x_id * 6 + 1];
  float bw = input[x_id * 6 + 2];
  float bh = input[x_id * 6 + 3];

  float x0 = bxc - bw / 2;
  float y0 = byc - bh / 2;
  float x1 = x0 + bw;
  float y1 = y0 + bh;

  x0 = fminf(float(netW), fmaxf(float(0.0), x0));
  y0 = fminf(float(netH), fmaxf(float(0.0), y0));
  x1 = fminf(float(netW), fmaxf(float(0.0), x1));
  y1 = fminf(float(netH), fmaxf(float(0.0), y1));

  binfo[x_id].left = x0;
  binfo[x_id].top = y0;
  binfo[x_id].width = fminf(float(netW), fmaxf(float(0.0), x1 - x0));
  binfo[x_id].height = fminf(float(netH), fmaxf(float(0.0), y1 - y0));
  binfo[x_id].detectionConfidence = maxProb;
  binfo[x_id].classId = maxIndex;
}

__global__ void decodeTensorYoloE_cuda(NvDsInferParseObjectInfo *binfo, float* input, int outputSize, int netW, int netH,
    float minPreclusterThreshold)
{
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (x_id >= outputSize)
    return;

  float maxProb = input[x_id * 6 + 4];
  int maxIndex = (int) input[x_id * 6 + 5];

  if (maxProb < minPreclusterThreshold) {
    binfo[x_id].detectionConfidence = 0.0;
    return;
  }

  float x0 = input[x_id * 6 + 0];
  float y0 = input[x_id * 6 + 1];
  float x1 = input[x_id * 6 + 2];
  float y1 = input[x_id * 6 + 3];

  x0 = fminf(float(netW), fmaxf(float(0.0), x0));
  y0 = fminf(float(netH), fmaxf(float(0.0), y0));
  x1 = fminf(float(netW), fmaxf(float(0.0), x1));
  y1 = fminf(float(netH), fmaxf(float(0.0), y1));

  binfo[x_id].left = x0;
  binfo[x_id].top = y0;
  binfo[x_id].width = fminf(float(netW), fmaxf(float(0.0), x1 - x0));
  binfo[x_id].height = fminf(float(netH), fmaxf(float(0.0), y1 - y0));
  binfo[x_id].detectionConfidence = maxProb;
  binfo[x_id].classId = maxIndex;
}

static bool NvDsInferParseCustomYolo_cuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo &layer = outputLayersInfo[0];

  const int outputSize = layer.inferDims.d[0];

  thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

  float minPreclusterThreshold = *(std::min_element(detectionParams.perClassPreclusterThreshold.begin(),
        detectionParams.perClassPreclusterThreshold.end()));

  int threads_per_block = 1024;
  int number_of_blocks = ((outputSize - 1) / threads_per_block) + 1;

  decodeTensorYolo_cuda<<<number_of_blocks, threads_per_block>>>(
      thrust::raw_pointer_cast(objects.data()), (float*) layer.buffer, outputSize, networkInfo.width, networkInfo.height,
      minPreclusterThreshold);

  objectList.resize(outputSize);
  thrust::copy(objects.begin(), objects.end(), objectList.begin());

  return true;
}

static bool NvDsInferParseCustomYoloE_cuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo &layer = outputLayersInfo[0];

  const int outputSize = layer.inferDims.d[0];

  thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

  float minPreclusterThreshold = *(std::min_element(detectionParams.perClassPreclusterThreshold.begin(),
        detectionParams.perClassPreclusterThreshold.end()));

  int threads_per_block = 1024;
  int number_of_blocks = ((outputSize - 1) / threads_per_block) + 1;

  decodeTensorYoloE_cuda<<<number_of_blocks, threads_per_block>>>(
      thrust::raw_pointer_cast(objects.data()), (float*) layer.buffer, outputSize, networkInfo.width, networkInfo.height,
      minPreclusterThreshold);

  objectList.resize(outputSize);
  thrust::copy(objects.begin(), objects.end(), objectList.begin());

  return true;
}

extern "C" bool
NvDsInferParseYolo_cuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYolo_cuda(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParseYoloE_cuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloE_cuda(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo_cuda);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloE_cuda);
