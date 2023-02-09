/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "c2f_layer.h"

#include <cassert>

#include "convolutional_layer.h"

nvinfer1::ITensor*
c2fLayer(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, std::string weightsType, float eps, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "c2f");
  assert(block.find("n") != block.end());
  assert(block.find("shortcut") != block.end());
  assert(block.find("filters") != block.end());

  int n = std::stoi(block.at("n"));
  bool shortcut = (block.at("shortcut") == "1");
  int filters = std::stoi(block.at("filters"));

  nvinfer1::Dims inputDims = input->getDimensions();

  nvinfer1::ISliceLayer* sliceLt = network->addSlice(*input,nvinfer1::Dims{3, {0, 0, 0}},
      nvinfer1::Dims{3, {inputDims.d[0] / 2, inputDims.d[1], inputDims.d[2]}}, nvinfer1::Dims{3, {1, 1, 1}});
  assert(sliceLt != nullptr);
  std::string sliceLtLayerName = "slice_lt_" + std::to_string(layerIdx);
  sliceLt->setName(sliceLtLayerName.c_str());
  nvinfer1::ITensor* lt = sliceLt->getOutput(0);

  nvinfer1::ISliceLayer* sliceRb = network->addSlice(*input,nvinfer1::Dims{3, {inputDims.d[0] / 2, 0, 0}},
      nvinfer1::Dims{3, {inputDims.d[0] / 2, inputDims.d[1], inputDims.d[2]}}, nvinfer1::Dims{3, {1, 1, 1}});
  assert(sliceRb != nullptr);
  std::string sliceRbLayerName = "slice_rb_" + std::to_string(layerIdx);
  sliceRb->setName(sliceRbLayerName.c_str());
  nvinfer1::ITensor* rb = sliceRb->getOutput(0);

  std::vector<nvinfer1::ITensor*> concatInputs;
  concatInputs.push_back(lt);
  concatInputs.push_back(rb);
  output = rb;

  for (int i = 0; i < n; ++i) {
    std::string cv1MlayerName = "c2f_1_" + std::to_string(i + 1) + "_";
    nvinfer1::ITensor* cv1M = convolutionalLayer(layerIdx, block, weights, trtWeights, weightPtr, weightsType, filters, eps,
        output, network, cv1MlayerName);
    assert(cv1M != nullptr);

    std::string cv2MlayerName = "c2f_2_" + std::to_string(i + 1) + "_";
    nvinfer1::ITensor* cv2M = convolutionalLayer(layerIdx, block, weights, trtWeights, weightPtr, weightsType, filters, eps,
        cv1M, network, cv2MlayerName);
    assert(cv2M != nullptr);

    if (shortcut) {
      nvinfer1::IElementWiseLayer* ew = network->addElementWise(*output, *cv2M, nvinfer1::ElementWiseOperation::kSUM);
      assert(ew != nullptr);
      std::string ewLayerName = "shortcut_c2f_" + std::to_string(i + 1) + "_" + std::to_string(layerIdx);
      ew->setName(ewLayerName.c_str());
      output = ew->getOutput(0);
      concatInputs.push_back(output);
    }
    else {
      output = cv2M;
      concatInputs.push_back(output);
    }
  }

  nvinfer1::IConcatenationLayer* concat = network->addConcatenation(concatInputs.data(), concatInputs.size());
  assert(concat != nullptr);
  std::string concatLayerName = "route_" + std::to_string(layerIdx);
  concat->setName(concatLayerName.c_str());
  concat->setAxis(0);
  output = concat->getOutput(0);

  return output;
}
