/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "route_layer.h"

nvinfer1::ILayer* routeLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    std::vector<nvinfer1::ITensor*> tensorOutputs,
    nvinfer1::INetworkDefinition* network)
{
    std::string strLayers = block.at("layers");
    std::vector<int> idxLayers;
    size_t lastPos = 0, pos = 0;
    while ((pos = strLayers.find(',', lastPos)) != std::string::npos) {
        int vL = std::stoi(trim(strLayers.substr(lastPos, pos - lastPos)));
        idxLayers.push_back (vL);
        lastPos = pos + 1;
    }
    if (lastPos < strLayers.length()) {
        std::string lastV = trim(strLayers.substr(lastPos));
        if (!lastV.empty()) {
            idxLayers.push_back (std::stoi(lastV));
        }
    }
    assert (!idxLayers.empty());
    std::vector<nvinfer1::ITensor*> concatInputs;
    for (int idxLayer : idxLayers) {
        if (idxLayer < 0) {
            idxLayer = tensorOutputs.size() + idxLayer;
        }
        assert (idxLayer >= 0 && idxLayer < (int)tensorOutputs.size());
        concatInputs.push_back (tensorOutputs[idxLayer]);
    }

    nvinfer1::IConcatenationLayer* concat =
        network->addConcatenation(concatInputs.data(), concatInputs.size());
    assert(concat != nullptr);
    std::string concatLayerName = "route_" + std::to_string(layerIdx - 1);
    concat->setName(concatLayerName.c_str());
    concat->setAxis(0);

    nvinfer1::ILayer* output = concat;

    if (block.find("groups") != block.end()) {
        nvinfer1::Dims prevTensorDims = output->getOutput(0)->getDimensions();
        int groups = stoi(block.at("groups"));
        int group_id = stoi(block.at("group_id"));
        int startSlice = (prevTensorDims.d[0] / groups) * group_id;
        int channelSlice = (prevTensorDims.d[0] / groups);
        nvinfer1::ISliceLayer* sl = network->addSlice(
            *output->getOutput(0),
            nvinfer1::Dims3{startSlice, 0, 0},
            nvinfer1::Dims3{channelSlice, prevTensorDims.d[1], prevTensorDims.d[2]},
            nvinfer1::Dims3{1, 1, 1});
        assert(sl != nullptr);
        output = sl;
    }

    return output;
}