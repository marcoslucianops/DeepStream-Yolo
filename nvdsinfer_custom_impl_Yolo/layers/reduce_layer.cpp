/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "reduce_layer.h"

nvinfer1::ITensor* reduceLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::ITensor* output;

    assert(block.at("type") == "reduce");
    assert(block.find("mode") != block.end());
    assert(block.find("axes") != block.end());

    std::string mode = block.at("mode");

    nvinfer1::ReduceOperation operation;
    if (mode == "mean")
        operation = nvinfer1::ReduceOperation::kAVG;

    std::string strAxes = block.at("axes");
    std::vector<int32_t> axes;
    size_t lastPos = 0, pos = 0;
    while ((pos = strAxes.find(',', lastPos)) != std::string::npos)
    {
        int vL = std::stoi(trim(strAxes.substr(lastPos, pos - lastPos)));
        axes.push_back(vL);
        lastPos = pos + 1;
    }
    if (lastPos < strAxes.length())
    {
        std::string lastV = trim(strAxes.substr(lastPos));
        if (!lastV.empty())
            axes.push_back(std::stoi(lastV));
    }
    assert(!axes.empty());
    
    uint32_t axisMask = 0;
    for (int axis : axes)
        axisMask |= 1 << axis;
    
    bool keepDims = false;
    if (block.find("keep") != block.end())
        keepDims = std::stoi(block.at("keep")) == 1 ? true : false;

    nvinfer1::IReduceLayer* reduce = network->addReduce(*input, operation, axisMask, keepDims);
    assert(reduce != nullptr);
    std::string reduceLayerName = "reduce_" + std::to_string(layerIdx);
    reduce->setName(reduceLayerName.c_str());
    output = reduce->getOutput(0);

    return output;
}
