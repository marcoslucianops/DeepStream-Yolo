/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "cls_layer.h"

nvinfer1::ITensor* clsLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::ITensor* output;

    assert(block.at("type") == "cls");

    nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*input);
    assert(shuffle != nullptr);
    std::string shuffleLayerName = "shuffle_" + std::to_string(layerIdx);
    shuffle->setName(shuffleLayerName.c_str());
    nvinfer1::Permutation permutation;
    permutation.order[0] = 1;
    permutation.order[1] = 0;
    shuffle->setFirstTranspose(permutation);
    output = shuffle->getOutput(0);

    return output;
}
