/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "channels_layer.h"

nvinfer1::ILayer* channelsLayer(
    std::string type,
    nvinfer1::ITensor* input,
    nvinfer1::ITensor* implicitTensor,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::ILayer* output;

    if (type == "shift") {
    nvinfer1::IElementWiseLayer* ew = network->addElementWise(
        *input, *implicitTensor,
        nvinfer1::ElementWiseOperation::kSUM);
    assert(ew != nullptr);
    output = ew;
    }
    else if (type == "control") {
        nvinfer1::IElementWiseLayer* ew = network->addElementWise(
        *input, *implicitTensor,
        nvinfer1::ElementWiseOperation::kPROD);
    assert(ew != nullptr);
    output = ew;
    }

    return output;
}