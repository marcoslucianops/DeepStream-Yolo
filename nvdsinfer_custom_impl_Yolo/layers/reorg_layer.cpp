/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "reorg_layer.h"

nvinfer1::ITensor* reorgLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::ITensor* output;

    assert(block.at("type") == "reorg");

    nvinfer1::Dims inputDims = input->getDimensions();

    nvinfer1::ISliceLayer *slice1 = network->addSlice(
        *input, nvinfer1::Dims{3, {0, 0, 0}}, nvinfer1::Dims{3, {inputDims.d[0], inputDims.d[1] / 2, inputDims.d[2] / 2}},
        nvinfer1::Dims{3, {1, 2, 2}});
    assert(slice1 != nullptr);
    std::string slice1LayerName = "slice1_" + std::to_string(layerIdx);
    slice1->setName(slice1LayerName.c_str());

    nvinfer1::ISliceLayer *slice2 = network->addSlice(
        *input, nvinfer1::Dims{3, {0, 1, 0}}, nvinfer1::Dims{3, {inputDims.d[0], inputDims.d[1] / 2, inputDims.d[2] / 2}},
        nvinfer1::Dims{3, {1, 2, 2}});
    assert(slice2 != nullptr);
    std::string slice2LayerName = "slice2_" + std::to_string(layerIdx);
    slice2->setName(slice2LayerName.c_str());

    nvinfer1::ISliceLayer *slice3 = network->addSlice(
        *input, nvinfer1::Dims{3, {0, 0, 1}}, nvinfer1::Dims{3, {inputDims.d[0], inputDims.d[1] / 2, inputDims.d[2] / 2}},
        nvinfer1::Dims{3, {1, 2, 2}});
    assert(slice3 != nullptr);
    std::string slice3LayerName = "slice3_" + std::to_string(layerIdx);
    slice3->setName(slice3LayerName.c_str());

    nvinfer1::ISliceLayer *slice4 = network->addSlice(
        *input, nvinfer1::Dims{3, {0, 1, 1}}, nvinfer1::Dims{3, {inputDims.d[0], inputDims.d[1] / 2, inputDims.d[2] / 2}},
        nvinfer1::Dims{3, {1, 2, 2}});
    assert(slice4 != nullptr);
    std::string slice4LayerName = "slice4_" + std::to_string(layerIdx);
    slice4->setName(slice4LayerName.c_str());

    std::vector<nvinfer1::ITensor*> concatInputs;
    concatInputs.push_back(slice1->getOutput(0));
    concatInputs.push_back(slice2->getOutput(0));
    concatInputs.push_back(slice3->getOutput(0));
    concatInputs.push_back(slice4->getOutput(0));

    nvinfer1::IConcatenationLayer* concat = network->addConcatenation(concatInputs.data(), concatInputs.size());
    assert(concat != nullptr);
    std::string concatLayerName = "concat_" + std::to_string(layerIdx);
    concat->setName(concatLayerName.c_str());
    concat->setAxis(0);
    output = concat->getOutput(0);

    return output;
}
