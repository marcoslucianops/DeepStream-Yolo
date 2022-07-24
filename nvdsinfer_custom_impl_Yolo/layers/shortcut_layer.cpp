/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "shortcut_layer.h"

nvinfer1::ITensor* shortcutLayer(
    int layerIdx,
    std::string mode,
    std::string activation,
    std::string inputVol,
    std::string shortcutVol,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    nvinfer1::ITensor* shortcutInput,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::ITensor* output;

    assert(block.at("type") == "shortcut");

    nvinfer1::ElementWiseOperation operation = nvinfer1::ElementWiseOperation::kSUM;

    if (mode == "mul")
        operation = nvinfer1::ElementWiseOperation::kPROD;

    if (mode == "add" && inputVol != shortcutVol)
    {
        nvinfer1::ISliceLayer* slice = network->addSlice(
            *shortcutInput, nvinfer1::Dims{3, {0, 0, 0}}, input->getDimensions(), nvinfer1::Dims{3, {1, 1, 1}});
        assert(slice != nullptr);
        std::string sliceLayerName = "slice_" + std::to_string(layerIdx);
        slice->setName(sliceLayerName.c_str());
        output = slice->getOutput(0);
    }
    else 
    {
        output = shortcutInput;
    }

    nvinfer1::IElementWiseLayer* shortcut = network->addElementWise(*input, *output, operation);
    assert(shortcut != nullptr);
    std::string shortcutLayerName = "shortcut_" + std::to_string(layerIdx);
    shortcut->setName(shortcutLayerName.c_str());
    output = shortcut->getOutput(0);

    output = activationLayer(layerIdx, activation, output, network);
    assert(output != nullptr);

    return output;
}
