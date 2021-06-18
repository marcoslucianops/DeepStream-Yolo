/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "upsample_layer.h"

nvinfer1::ILayer* upsampleLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "upsample");
    int stride = std::stoi(block.at("stride"));

    nvinfer1::IResizeLayer* resize_layer = network->addResize(*input);
    resize_layer->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    float scale[3] = {1, stride, stride};
    resize_layer->setScales(scale, 3);
    std::string layer_name = "upsample_" + std::to_string(layerIdx);
    resize_layer->setName(layer_name.c_str());
    return resize_layer;
}