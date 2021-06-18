/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "shortcut_layer.h"

nvinfer1::ILayer* shortcutLayer(
    int layerIdx,
    std::string activation,
    std::string inputVol,
    std::string shortcutVol,
    nvinfer1::ITensor* input,
    nvinfer1::ITensor* shortcutTensor,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::ILayer* output;
    nvinfer1::ITensor* outputTensor;

    if (inputVol != shortcutVol)
    {
        nvinfer1::ISliceLayer* sl = network->addSlice(
            *shortcutTensor,
            nvinfer1::Dims3{0, 0, 0},
            input->getDimensions(),
            nvinfer1::Dims3{1, 1, 1});
        assert(sl != nullptr);
        outputTensor = sl->getOutput(0);
        assert(outputTensor != nullptr);
    } else 
    {
        outputTensor = shortcutTensor;
        assert(outputTensor != nullptr);
    }

    nvinfer1::IElementWiseLayer* ew = network->addElementWise(
        *input, *outputTensor,
        nvinfer1::ElementWiseOperation::kSUM);
    assert(ew != nullptr);

    output = activationLayer(layerIdx, activation, ew, ew->getOutput(0), network);
    assert(output != nullptr);

    return output;
}