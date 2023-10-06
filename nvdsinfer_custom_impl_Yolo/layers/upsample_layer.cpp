/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "upsample_layer.h"

#include <cassert>

nvinfer1::ITensor*
upsampleLayer(int layerIdx, std::map<std::string, std::string>& block, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "upsample");
  assert(block.find("stride") != block.end());

  int stride = std::stoi(block.at("stride"));

  float scale[4] = {1, 1, static_cast<float>(stride), static_cast<float>(stride)};

  nvinfer1::IResizeLayer* resize = network->addResize(*input);
  assert(resize != nullptr);
  std::string resizeLayerName = "upsample_" + std::to_string(layerIdx);
  resize->setName(resizeLayerName.c_str());
  resize->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
  resize->setScales(scale, 4);
  output = resize->getOutput(0);

  return output;
}
