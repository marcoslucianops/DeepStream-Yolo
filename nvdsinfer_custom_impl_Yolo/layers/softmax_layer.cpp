/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "softmax_layer.h"

nvinfer1::ITensor* softmaxLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::ITensor* output;

    assert(block.at("type") == "softmax");
    assert(block.find("axes") != block.end());

    int axes = std::stoi(block.at("axes"));

    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*input);
    assert(softmax != nullptr);
    std::string softmaxLayerName = "softmax_" + std::to_string(layerIdx);
    softmax->setName(softmaxLayerName.c_str());
    softmax->setAxes(1 << axes);
    output = softmax->getOutput(0);

    return output;
}
