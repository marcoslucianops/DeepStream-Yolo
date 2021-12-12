/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <math.h>
#include "implicit_layer.h"

nvinfer1::ILayer* implicitLayer(
    int channels,
    std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights,
    int& weightPtr,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::Weights convWt{nvinfer1::DataType::kFLOAT, nullptr, channels};

    float* val = new float[channels];
    for (int i = 0; i < channels; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convWt.values = val;
    trtWeights.push_back(convWt);

    nvinfer1::IConstantLayer* implicit = network->addConstant(nvinfer1::Dims3{static_cast<int>(channels), 1, 1}, convWt);
    assert(implicit != nullptr);

    return implicit;
}