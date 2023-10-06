/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "deconvolutional_layer.h"

#include <cassert>

nvinfer1::ITensor*
deconvolutionalLayer(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, int& inputChannels, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network, std::string layerName)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "deconvolutional");
  assert(block.find("filters") != block.end());
  assert(block.find("pad") != block.end());
  assert(block.find("size") != block.end());
  assert(block.find("stride") != block.end());

  int filters = std::stoi(block.at("filters"));
  int padding = std::stoi(block.at("pad"));
  int kernelSize = std::stoi(block.at("size"));
  int stride = std::stoi(block.at("stride"));
  int bias = filters;

  int groups = 1;
  if (block.find("groups") != block.end())
    groups = std::stoi(block.at("groups"));

  if (block.find("bias") != block.end())
    bias = std::stoi(block.at("bias"));

  int pad;
  if (padding)
    pad = (kernelSize - 1) / 2;
  else
    pad = 0;

  int size = filters * inputChannels * kernelSize * kernelSize / groups;
  std::vector<float> bnBiases;
  std::vector<float> bnWeights;
  std::vector<float> bnRunningMean;
  std::vector<float> bnRunningVar;
  nvinfer1::Weights convWt {nvinfer1::DataType::kFLOAT, nullptr, size};
  nvinfer1::Weights convBias {nvinfer1::DataType::kFLOAT, nullptr, bias};

  float* val;
  if (bias != 0) {
    val = new float[filters];
    for (int i = 0; i < filters; ++i) {
        val[i] = weights[weightPtr];
        ++weightPtr;
    }
    convBias.values = val;
    trtWeights.push_back(convBias);
  }
  val = new float[size];
  for (int i = 0; i < size; ++i) {
      val[i] = weights[weightPtr];
      ++weightPtr;
  }
  convWt.values = val;
  trtWeights.push_back(convWt);

  nvinfer1::IDeconvolutionLayer* conv = network->addDeconvolutionNd(*input, filters,
      nvinfer1::Dims{2, {kernelSize, kernelSize}}, convWt, convBias);
  assert(conv != nullptr);
  std::string convLayerName = "deconv_" + layerName + std::to_string(layerIdx);
  conv->setName(convLayerName.c_str());
  conv->setStrideNd(nvinfer1::Dims{2, {stride, stride}});
  conv->setPaddingNd(nvinfer1::Dims{2, {pad, pad}});

  if (block.find("groups") != block.end())
    conv->setNbGroups(groups);

  output = conv->getOutput(0);

  return output;
}
