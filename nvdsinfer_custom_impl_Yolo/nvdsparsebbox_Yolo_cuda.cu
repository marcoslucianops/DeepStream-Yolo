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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "nvdsinfer_custom_impl.h"

#include "utils.h"
#include "yoloPlugins.h"

__global__ void decodeTensor_YOLO_ONNX(NvDsInferParseObjectInfo *binfo, const float* detections, const int numClasses,
    const int outputSize, float netW, float netH, const float* preclusterThreshold, int* numDetections)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_id >= outputSize)
      return;

    float maxProb = 0.0f;
    int maxIndex = -1;

    for (uint i = 0; i < numClasses; ++i) {
      float prob = detections[x_id * (5 + numClasses) + 5 + i];
      if (prob > maxProb) {
        maxProb = prob;
        maxIndex = i;
      }
    }

    const float objectness = detections[x_id * (5 + numClasses) + 4];

    if (objectness * maxProb < preclusterThreshold[maxIndex])
      return;

    int count = (int)atomicAdd(numDetections, 1);

    const float bxc = detections[x_id * (5 + numClasses) + 0];
    const float byc = detections[x_id * (5 + numClasses) + 1];
    const float bw = detections[x_id * (5 + numClasses) + 2];
    const float bh = detections[x_id * (5 + numClasses) + 3];

    float x0 = bxc - bw / 2;
    float y0 = byc - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;
    x0 = fminf(float(netW), fmaxf(float(0.0), x0));
    y0 = fminf(float(netH), fmaxf(float(0.0), y0));
    x1 = fminf(float(netW), fmaxf(float(0.0), x1));
    y1 = fminf(float(netH), fmaxf(float(0.0), y1));

    binfo[count].left = x0;
    binfo[count].top = y0;
    binfo[count].width = fminf(float(netW), fmaxf(float(0.0), x1 - x0));
    binfo[count].height = fminf(float(netH), fmaxf(float(0.0), y1 - y0));
    binfo[count].detectionConfidence = objectness * maxProb;
    binfo[count].classId = maxIndex;
}

__global__ void decodeTensor_YOLOV8_ONNX(NvDsInferParseObjectInfo* binfo, const float* detections, const int numClasses,
    const int outputSize, float netW, float netH, const float* preclusterThreshold, int* numDetections)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_id >= outputSize)
      return;

    float maxProb = 0.0f;
    int maxIndex = -1;

    for (uint i = 0; i < numClasses; ++i) {
      float prob = detections[x_id + outputSize * (i + 4)];
      if (prob > maxProb) {
        maxProb = prob;
        maxIndex = i;
      }
    }

    if (maxProb < preclusterThreshold[maxIndex])
      return;

    int count = (int)atomicAdd(numDetections, 1);

    const float bxc = detections[x_id + outputSize * 0];
    const float byc = detections[x_id + outputSize * 1];
    const float bw = detections[x_id + outputSize * 2];
    const float bh = detections[x_id + outputSize * 3];

    float x0 = bxc - bw / 2;
    float y0 = byc - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;
    x0 = fminf(float(netW), fmaxf(float(0.0), x0));
    y0 = fminf(float(netH), fmaxf(float(0.0), y0));
    x1 = fminf(float(netW), fmaxf(float(0.0), x1));
    y1 = fminf(float(netH), fmaxf(float(0.0), y1));

    binfo[count].left = x0;
    binfo[count].top = y0;
    binfo[count].width = fminf(float(netW), fmaxf(float(0.0), x1 - x0));
    binfo[count].height = fminf(float(netH), fmaxf(float(0.0), y1 - y0));
    binfo[count].detectionConfidence = maxProb;
    binfo[count].classId = maxIndex;
}

__global__ void decodeTensor_YOLOX_ONNX(NvDsInferParseObjectInfo *binfo, const float* detections, const int numClasses,
    const int outputSize, float netW, float netH, const int *grid0, const int *grid1, const int *strides,
    const float* preclusterThreshold, int* numDetections)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_id >= outputSize)
      return;

    float maxProb = 0.0f;
    int maxIndex = -1;

    for (uint i = 0; i < numClasses; ++i) {
      float prob = detections[x_id * (5 + numClasses) + 5 + i];
      if (prob > maxProb) {
        maxProb = prob;
        maxIndex = i;
      }
    }

    const float objectness = detections[x_id * (5 + numClasses) + 4];

    if (objectness * maxProb < preclusterThreshold[maxIndex])
      return;

    int count = (int)atomicAdd(numDetections, 1);

    const float bxc = (detections[x_id * (5 + numClasses) + 0] + grid0[x_id]) * strides[x_id];
    const float byc = (detections[x_id * (5 + numClasses) + 1] + grid1[x_id]) * strides[x_id];
    const float bw = __expf(detections[x_id * (5 + numClasses) + 2]) * strides[x_id];
    const float bh = __expf(detections[x_id * (5 + numClasses) + 3]) * strides[x_id];

    float x0 = bxc - bw / 2;
    float y0 = byc - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;
    x0 = fminf(float(netW), fmaxf(float(0.0), x0));
    y0 = fminf(float(netH), fmaxf(float(0.0), y0));
    x1 = fminf(float(netW), fmaxf(float(0.0), x1));
    y1 = fminf(float(netH), fmaxf(float(0.0), y1));

    binfo[count].left = x0;
    binfo[count].top = y0;
    binfo[count].width = fminf(float(netW), fmaxf(float(0.0), x1 - x0));
    binfo[count].height = fminf(float(netH), fmaxf(float(0.0), y1 - y0));
    binfo[count].detectionConfidence = objectness * maxProb;
    binfo[count].classId = maxIndex;
}

__global__ void decodeTensor_YOLO_NAS_ONNX(NvDsInferParseObjectInfo *binfo, const float* scores, const float* boxes,
    const int numClasses, const int outputSize, float netW, float netH, const float* preclusterThreshold, int* numDetections)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_id >= outputSize)
      return;

    float maxProb = 0.0f;
    int maxIndex = -1;

    for (uint i = 0; i < numClasses; ++i) {
      float prob = scores[x_id * numClasses + i];
      if (prob > maxProb) {
        maxProb = prob;
        maxIndex = i;
      }
    }

    if (maxProb < preclusterThreshold[maxIndex])
      return;

    int count = (int)atomicAdd(numDetections, 1);

    float x0 = boxes[x_id * 4 + 0];
    float y0 = boxes[x_id * 4 + 1];
    float x1 = boxes[x_id * 4 + 2];
    float y1 = boxes[x_id * 4 + 3];

    x0 = fminf(float(netW), fmaxf(float(0.0), x0));
    y0 = fminf(float(netH), fmaxf(float(0.0), y0));
    x1 = fminf(float(netW), fmaxf(float(0.0), x1));
    y1 = fminf(float(netH), fmaxf(float(0.0), y1));

    binfo[count].left = x0;
    binfo[count].top = y0;
    binfo[count].width = fminf(float(netW), fmaxf(float(0.0), x1 - x0));
    binfo[count].height = fminf(float(netH), fmaxf(float(0.0), y1 - y0));
    binfo[count].detectionConfidence = maxProb;
    binfo[count].classId = maxIndex;
}

__global__ void decodeTensor_PPYOLOE_ONNX(NvDsInferParseObjectInfo *binfo, const float* scores, const float* boxes,
    const int numClasses, const int outputSize, float netW, float netH, const float* preclusterThreshold, int* numDetections)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_id >= outputSize)
      return;

    float maxProb = 0.0f;
    int maxIndex = -1;

    for (uint i = 0; i < numClasses; ++i) {
      float prob = scores[x_id + outputSize * i];
      if (prob > maxProb) {
        maxProb = prob;
        maxIndex = i;
      }
    }

    if (maxProb < preclusterThreshold[maxIndex])
      return;

    int count = (int)atomicAdd(numDetections, 1);

    float x0 = boxes[x_id * 4 + 0];
    float y0 = boxes[x_id * 4 + 1];
    float x1 = boxes[x_id * 4 + 2];
    float y1 = boxes[x_id * 4 + 3];

    x0 = fminf(float(netW), fmaxf(float(0.0), x0));
    y0 = fminf(float(netH), fmaxf(float(0.0), y0));
    x1 = fminf(float(netW), fmaxf(float(0.0), x1));
    y1 = fminf(float(netH), fmaxf(float(0.0), y1));

    binfo[count].left = x0;
    binfo[count].top = y0;
    binfo[count].width = fminf(float(netW), fmaxf(float(0.0), x1 - x0));
    binfo[count].height = fminf(float(netH), fmaxf(float(0.0), y1 - y0));
    binfo[count].detectionConfidence = maxProb;
    binfo[count].classId = maxIndex;
}

static bool
NvDsInferParseCustom_YOLO_ONNX(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& layer = outputLayersInfo[0];

  const uint outputSize = layer.inferDims.d[0];
  const uint numClasses = layer.inferDims.d[1] - 5;

  if (numClasses != detectionParams.numClassesConfigured) {
    std::cerr << "WARNING: Number of classes mismatch, make sure to set num-detected-classes=" << numClasses
        << " in config_infer file\n" << std::endl;
  }

  thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

  std::vector<int> numDetections = { 0 };
  thrust::device_vector<int> d_numDetections(numDetections);

  thrust::device_vector<float> preclusterThreshold(detectionParams.perClassPreclusterThreshold);

  int threads_per_block = 1024;
  int number_of_blocks = ((outputSize - 1) / threads_per_block) + 1;

  decodeTensor_YOLO_ONNX<<<threads_per_block, number_of_blocks>>>(
      thrust::raw_pointer_cast(objects.data()), (const float*) (layer.buffer), numClasses, outputSize,
      static_cast<float>(networkInfo.width), static_cast<float>(networkInfo.height),
      thrust::raw_pointer_cast(preclusterThreshold.data()), thrust::raw_pointer_cast(d_numDetections.data()));

  thrust::copy(d_numDetections.begin(), d_numDetections.end(), numDetections.begin());
  objectList.resize(numDetections[0]);
  thrust::copy(objects.begin(), objects.begin() + numDetections[0], objectList.begin());

  return true;
}

static bool
NvDsInferParseCustom_YOLOV8_ONNX(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& layer = outputLayersInfo[0];

  const uint numClasses = layer.inferDims.d[0] - 4;
  const uint outputSize = layer.inferDims.d[1];

  if (numClasses != detectionParams.numClassesConfigured) {
    std::cerr << "WARNING: Number of classes mismatch, make sure to set num-detected-classes=" << numClasses
        << " in config_infer file\n" << std::endl;
  }

  thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

  std::vector<int> numDetections = { 0 };
  thrust::device_vector<int> d_numDetections(numDetections);

  thrust::device_vector<float> preclusterThreshold(detectionParams.perClassPreclusterThreshold);

  int threads_per_block = 1024;
  int number_of_blocks = ((outputSize - 1) / threads_per_block) + 1;

  decodeTensor_YOLOV8_ONNX<<<threads_per_block, number_of_blocks>>>(
      thrust::raw_pointer_cast(objects.data()), (const float*) (layer.buffer), numClasses, outputSize,
      static_cast<float>(networkInfo.width), static_cast<float>(networkInfo.height),
      thrust::raw_pointer_cast(preclusterThreshold.data()), thrust::raw_pointer_cast(d_numDetections.data()));

  thrust::copy(d_numDetections.begin(), d_numDetections.end(), numDetections.begin());
  objectList.resize(numDetections[0]);
  thrust::copy(objects.begin(), objects.begin() + numDetections[0], objectList.begin());

  return true;
}

static bool
NvDsInferParseCustom_YOLOX_ONNX(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& layer = outputLayersInfo[0];

  const uint outputSize = layer.inferDims.d[0];
  const uint numClasses = layer.inferDims.d[1] - 5;

  if (numClasses != detectionParams.numClassesConfigured) {
    std::cerr << "WARNING: Number of classes mismatch, make sure to set num-detected-classes=" << numClasses
        << " in config_infer file\n" << std::endl;
  }

  thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

  std::vector<int> numDetections = { 0 };
  thrust::device_vector<int> d_numDetections(numDetections);

  thrust::device_vector<float> preclusterThreshold(detectionParams.perClassPreclusterThreshold);

  std::vector<int> strides = {8, 16, 32};

  std::vector<int> grid0;
  std::vector<int> grid1;
  std::vector<int> gridStrides;

  for (uint s = 0; s < strides.size(); ++s) {
    int num_grid_y = networkInfo.height / strides[s];
    int num_grid_x = networkInfo.width / strides[s];
    for (int g1 = 0; g1 < num_grid_y; ++g1) {
      for (int g0 = 0; g0 < num_grid_x; ++g0) {
        grid0.push_back(g0);
        grid1.push_back(g1);
        gridStrides.push_back(strides[s]);
      }
    }
  }

  thrust::device_vector<int> d_grid0(grid0);
  thrust::device_vector<int> d_grid1(grid1);
  thrust::device_vector<int> d_gridStrides(gridStrides);

  int threads_per_block = 1024;
  int number_of_blocks = ((outputSize - 1) / threads_per_block) + 1;

  decodeTensor_YOLOX_ONNX<<<threads_per_block, number_of_blocks>>>(
      thrust::raw_pointer_cast(objects.data()), (const float*) (layer.buffer), numClasses, outputSize,
      static_cast<float>(networkInfo.width), static_cast<float>(networkInfo.height),
      thrust::raw_pointer_cast(d_grid0.data()), thrust::raw_pointer_cast(d_grid1.data()),
      thrust::raw_pointer_cast(d_gridStrides.data()), thrust::raw_pointer_cast(preclusterThreshold.data()),
      thrust::raw_pointer_cast(d_numDetections.data()));

  thrust::copy(d_numDetections.begin(), d_numDetections.end(), numDetections.begin());
  objectList.resize(numDetections[0]);
  thrust::copy(objects.begin(), objects.begin() + numDetections[0], objectList.begin());

  return true;
}

static bool
NvDsInferParseCustom_YOLO_NAS_ONNX(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& scores = outputLayersInfo[0];
  const NvDsInferLayerInfo& boxes = outputLayersInfo[1];

  const uint outputSize = scores.inferDims.d[0];
  const uint numClasses = scores.inferDims.d[1];

  if (numClasses != detectionParams.numClassesConfigured) {
    std::cerr << "WARNING: Number of classes mismatch, make sure to set num-detected-classes=" << numClasses
        << " in config_infer file\n" << std::endl;
  }

  thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

  std::vector<int> numDetections = { 0 };
  thrust::device_vector<int> d_numDetections(numDetections);

  thrust::device_vector<float> preclusterThreshold(detectionParams.perClassPreclusterThreshold);

  int threads_per_block = 1024;
  int number_of_blocks = ((outputSize - 1) / threads_per_block) + 1;

  decodeTensor_YOLO_NAS_ONNX<<<threads_per_block, number_of_blocks>>>(
      thrust::raw_pointer_cast(objects.data()), (const float*) (scores.buffer), (const float*) (boxes.buffer), numClasses,
      outputSize, static_cast<float>(networkInfo.width), static_cast<float>(networkInfo.height),
      thrust::raw_pointer_cast(preclusterThreshold.data()), thrust::raw_pointer_cast(d_numDetections.data()));

  thrust::copy(d_numDetections.begin(), d_numDetections.end(), numDetections.begin());
  objectList.resize(numDetections[0]);
  thrust::copy(objects.begin(), objects.begin() + numDetections[0], objectList.begin());

  return true;
}

static bool
NvDsInferParseCustom_PPYOLOE_ONNX(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& scores = outputLayersInfo[0];
  const NvDsInferLayerInfo& boxes = outputLayersInfo[1];

  const uint numClasses = scores.inferDims.d[0];
  const uint outputSize = scores.inferDims.d[1];

  if (numClasses != detectionParams.numClassesConfigured) {
    std::cerr << "WARNING: Number of classes mismatch, make sure to set num-detected-classes=" << numClasses
        << " in config_infer file\n" << std::endl;
  }

  thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

  std::vector<int> numDetections = { 0 };
  thrust::device_vector<int> d_numDetections(numDetections);

  thrust::device_vector<float> preclusterThreshold(detectionParams.perClassPreclusterThreshold);

  int threads_per_block = 1024;
  int number_of_blocks = ((outputSize - 1) / threads_per_block) + 1;

  decodeTensor_PPYOLOE_ONNX<<<threads_per_block, number_of_blocks>>>(
      thrust::raw_pointer_cast(objects.data()), (const float*) (scores.buffer), (const float*) (boxes.buffer), numClasses,
      outputSize, static_cast<float>(networkInfo.width), static_cast<float>(networkInfo.height),
      thrust::raw_pointer_cast(preclusterThreshold.data()), thrust::raw_pointer_cast(d_numDetections.data()));

  thrust::copy(d_numDetections.begin(), d_numDetections.end(), numDetections.begin());
  objectList.resize(numDetections[0]);
  thrust::copy(objects.begin(), objects.begin() + numDetections[0], objectList.begin());

  return true;
}

extern "C" bool
NvDsInferParse_YOLO_ONNX(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustom_YOLO_ONNX(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParse_YOLOV8_ONNX(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustom_YOLOV8_ONNX(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParse_YOLOX_ONNX(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustom_YOLOX_ONNX(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParse_YOLO_NAS_ONNX(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustom_YOLO_NAS_ONNX(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParse_PPYOLOE_ONNX(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustom_PPYOLOE_ONNX(outputLayersInfo, networkInfo, detectionParams, objectList);
}
