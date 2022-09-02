/*
 * Created by Pablo Santana :)
 * https://www.github.com/pabsan-0
 */

#include "route_lhalf_layer.h"

nvinfer1::ITensor* route_lhalfLayer(
    int layerIdx,
    std::string& layers,
    std::map<std::string, std::string>& block,
    std::vector<nvinfer1::ITensor*> tensorOutputs,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::ITensor* output;

    assert(block.at("type") == "route_lhalf");
    assert(block.find("layers") != block.end());

    std::string strLayers = block.at("layers");
    assert ( strLayers.find(',') == std::string::npos ) ;
    int idxLayer = std::stoi(trim(strLayers));

    if (idxLayer < 0)
            idxLayer = tensorOutputs.size() + idxLayer;
    assert (idxLayer >= 0 && idxLayer < (int)tensorOutputs.size());
    output = tensorOutputs[idxLayer];
    layers += std::to_string(idxLayer);

    nvinfer1::Dims prevTensorDims = output->getDimensions();
    nvinfer1::ISliceLayer* slice = network->addSlice(
        *output,
        nvinfer1::Dims{3, {0, 0, 0}},
        nvinfer1::Dims{3, {prevTensorDims.d[0] / 2, prevTensorDims.d[1], prevTensorDims.d[2]}},
        nvinfer1::Dims{3, {1, 1, 1}});
    assert(slice != nullptr);
    std::string sliceLayerName = "slice_" + std::to_string(layerIdx);
    slice->setName(sliceLayerName.c_str());
    output = slice->getOutput(0);

    return output;
}
