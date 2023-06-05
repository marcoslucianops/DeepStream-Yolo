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

#include <algorithm>

#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_context.h"

#include "yolo.h"

#define USE_CUDA_ENGINE_GET_API 1

static bool
getYoloNetworkInfo(NetworkInfo& networkInfo, const NvDsInferContextInitParams* initParams)
{
  std::string onnxWtsFilePath = initParams->onnxFilePath;
  std::string darknetWtsFilePath = initParams->modelFilePath;
  std::string darknetCfgFilePath = initParams->customNetworkConfigFilePath;

  std::string yoloType = onnxWtsFilePath != "" ? "onnx" : "darknet";
  std::string modelName = yoloType == "onnx" ?
      onnxWtsFilePath.substr(0, onnxWtsFilePath.find(".onnx")).substr(onnxWtsFilePath.rfind("/") + 1) :
      darknetWtsFilePath.substr(0, darknetWtsFilePath.find(".weights")).substr(darknetWtsFilePath.rfind("/") + 1);

  std::transform(modelName.begin(), modelName.end(), modelName.begin(), [] (uint8_t c) {
    return std::tolower(c);
  });

  networkInfo.inputBlobName = "input";
  networkInfo.networkType = yoloType;
  networkInfo.modelName = modelName;
  networkInfo.onnxWtsFilePath = onnxWtsFilePath;
  networkInfo.darknetWtsFilePath = darknetWtsFilePath;
  networkInfo.darknetCfgFilePath = darknetCfgFilePath;
  networkInfo.batchSize = initParams->maxBatchSize;
  networkInfo.implicitBatch = initParams->forceImplicitBatchDimension;
  networkInfo.int8CalibPath = initParams->int8CalibrationFilePath;
  networkInfo.deviceType = initParams->useDLA ? "kDLA" : "kGPU";
  networkInfo.numDetectedClasses = initParams->numDetectedClasses;
  networkInfo.clusterMode = initParams->clusterMode;
  networkInfo.scaleFactor = initParams->networkScaleFactor;
  networkInfo.offsets = initParams->offsets;
  networkInfo.workspaceSize = initParams->workspaceSize;

  if (initParams->networkMode == NvDsInferNetworkMode_FP32)
    networkInfo.networkMode = "FP32";
  else if (initParams->networkMode == NvDsInferNetworkMode_INT8)
    networkInfo.networkMode = "INT8";
  else if (initParams->networkMode == NvDsInferNetworkMode_FP16)
    networkInfo.networkMode = "FP16";

  if (yoloType == "onnx") {
    if (!fileExists(networkInfo.onnxWtsFilePath)) {
      std::cerr << "ONNX model file does not exist\n" << std::endl;
      return false;
    }
  }
  else {
    if (!fileExists(networkInfo.darknetWtsFilePath)) {
      std::cerr << "Darknet weights file does not exist\n" << std::endl;
      return false;
    }
    else if (!fileExists(networkInfo.darknetCfgFilePath)) {
      std::cerr << "Darknet cfg file does not exist\n" << std::endl;
      return false;
    }
  }

  return true;
}

#if !USE_CUDA_ENGINE_GET_API
IModelParser*
NvDsInferCreateModelParser(const NvDsInferContextInitParams* initParams)
{
  NetworkInfo networkInfo;
  if (!getYoloNetworkInfo(networkInfo, initParams))
    return nullptr;

  return new Yolo(networkInfo);
}
#else

#if NV_TENSORRT_MAJOR >= 8
extern "C" bool
NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder* const builder, nvinfer1::IBuilderConfig* const builderConfig,
    const NvDsInferContextInitParams* const initParams, nvinfer1::DataType dataType, nvinfer1::ICudaEngine*& cudaEngine);

extern "C" bool
NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder* const builder, nvinfer1::IBuilderConfig* const builderConfig,
    const NvDsInferContextInitParams* const initParams, nvinfer1::DataType dataType, nvinfer1::ICudaEngine*& cudaEngine)
#else
extern "C" bool
NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder* const builder, const NvDsInferContextInitParams* const initParams,
    nvinfer1::DataType dataType, nvinfer1::ICudaEngine*& cudaEngine);

extern "C" bool
NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder* const builder, const NvDsInferContextInitParams* const initParams,
    nvinfer1::DataType dataType, nvinfer1::ICudaEngine*& cudaEngine)
#endif

{
  NetworkInfo networkInfo;
  if (!getYoloNetworkInfo(networkInfo, initParams))
    return false;

  Yolo yolo(networkInfo);

#if NV_TENSORRT_MAJOR >= 8
  cudaEngine = yolo.createEngine(builder, builderConfig);
#else
  cudaEngine = yolo.createEngine(builder);
#endif

  if (cudaEngine == nullptr) {
    std::cerr << "Failed to build CUDA engine" << std::endl;
    return false;
  }

  return true;
}
#endif
