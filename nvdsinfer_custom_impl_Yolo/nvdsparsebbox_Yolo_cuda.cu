/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "nvdsinfer_custom_impl.h"

extern "C" bool
NvDsInferParseYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool
NvDsInferParseYoloOBBCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

__global__ void decodeTensorYoloCuda(NvDsInferParseObjectInfo *binfo, const float* output, const uint outputSize,
    const uint netW, const uint netH, const float* preclusterThreshold)
{
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (x_id >= outputSize) {
    return;
  }

  float maxProb = output[x_id * 6 + 4];
  int maxIndex = (int) output[x_id * 6 + 5];

  if (maxProb < preclusterThreshold[maxIndex]) {
    binfo[x_id].detectionConfidence = 0.0;
    return;
  }

  float bx1 = output[x_id * 6 + 0];
  float by1 = output[x_id * 6 + 1];
  float bx2 = output[x_id * 6 + 2];
  float by2 = output[x_id * 6 + 3];

  bx1 = fminf(float(netW), fmaxf(float(0.0), bx1));
  by1 = fminf(float(netH), fmaxf(float(0.0), by1));
  bx2 = fminf(float(netW), fmaxf(float(0.0), bx2));
  by2 = fminf(float(netH), fmaxf(float(0.0), by2));

  binfo[x_id].left = bx1;
  binfo[x_id].top = by1;
  binfo[x_id].width = fminf(float(netW), fmaxf(float(0.0), bx2 - bx1));
  binfo[x_id].height = fminf(float(netH), fmaxf(float(0.0), by2 - by1));
  binfo[x_id].detectionConfidence = maxProb;
  binfo[x_id].classId = maxIndex;
}

__global__ void decodeTensorYoloOBBCuda(NvDsInferParseObjectInfo *binfo, const float* output, const uint outputSize,
    const uint netW, const uint netH, const float* preclusterThreshold, const uint numClasses)
{
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (x_id >= outputSize) {
    return;
  }

  const uint stride = 4 + numClasses + 1;

  float x_center = output[x_id * stride + 0];
  float y_center = output[x_id * stride + 1];
  float width = output[x_id * stride + 2];
  float height = output[x_id * stride + 3];

  float maxProb = 0.0f;
  int maxIndex = 0;
  for (uint c = 0; c < numClasses; ++c) {
    float prob = output[x_id * stride + 4 + c];
    if (prob > maxProb) {
      maxProb = prob;
      maxIndex = c;
    }
  }

  float angle = output[x_id * stride + 4 + numClasses];

  if (maxProb < preclusterThreshold[maxIndex]) {
    binfo[x_id].detectionConfidence = 0.0;
    return;
  }

  float cos_a = fabsf(cosf(angle));
  float sin_a = fabsf(sinf(angle));
  float half_aabb_w = (width * cos_a + height * sin_a) / 2.0f;
  float half_aabb_h = (width * sin_a + height * cos_a) / 2.0f;

  float bx1 = x_center - half_aabb_w;
  float by1 = y_center - half_aabb_h;
  float bx2 = x_center + half_aabb_w;
  float by2 = y_center + half_aabb_h;

  bx1 = fminf(float(netW), fmaxf(float(0.0), bx1));
  by1 = fminf(float(netH), fmaxf(float(0.0), by1));
  bx2 = fminf(float(netW), fmaxf(float(0.0), bx2));
  by2 = fminf(float(netH), fmaxf(float(0.0), by2));

  binfo[x_id].left = bx1;
  binfo[x_id].top = by1;
  binfo[x_id].width = fminf(float(netW), fmaxf(float(0.0), bx2 - bx1));
  binfo[x_id].height = fminf(float(netH), fmaxf(float(0.0), by2 - by1));
  binfo[x_id].detectionConfidence = maxProb;
  binfo[x_id].classId = maxIndex;
}

static bool NvDsInferParseCustomYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  const uint outputSize = output.inferDims.d[0];

  thrust::device_vector<float> perClassPreclusterThreshold = detectionParams.perClassPreclusterThreshold;

  thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

  int threads_per_block = 1024;
  int number_of_blocks = ((outputSize) / threads_per_block) + 1;

  decodeTensorYoloCuda<<<number_of_blocks, threads_per_block>>>(
      thrust::raw_pointer_cast(objects.data()), (float*) (output.buffer), outputSize, networkInfo.width,
          networkInfo.height, thrust::raw_pointer_cast(perClassPreclusterThreshold.data()));

  objectList.resize(outputSize);
  thrust::copy(objects.begin(), objects.end(), objectList.begin());

  return true;
}

static bool NvDsInferParseCustomYoloOBBCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  const uint outputSize = output.inferDims.d[0];
  const uint numClasses = detectionParams.numClassesConfigured;

  thrust::device_vector<float> perClassPreclusterThreshold = detectionParams.perClassPreclusterThreshold;

  thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

  int threads_per_block = 1024;
  int number_of_blocks = ((outputSize) / threads_per_block) + 1;

  decodeTensorYoloOBBCuda<<<number_of_blocks, threads_per_block>>>(
      thrust::raw_pointer_cast(objects.data()), (float*) (output.buffer), outputSize, networkInfo.width,
          networkInfo.height, thrust::raw_pointer_cast(perClassPreclusterThreshold.data()), numClasses);

  objectList.resize(outputSize);
  thrust::copy(objects.begin(), objects.end(), objectList.begin());

  return true;
}

extern "C" bool
NvDsInferParseYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloCuda(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParseYoloOBBCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloOBBCuda(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloCuda);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloOBBCuda);
