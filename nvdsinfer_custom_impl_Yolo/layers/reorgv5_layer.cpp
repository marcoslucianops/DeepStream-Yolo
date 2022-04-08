/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "reorgv5_layer.h"

nvinfer1::ILayer* reorgV5Layer(
    int layerIdx,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::Dims prevTensorDims = input->getDimensions();

    nvinfer1::ISliceLayer *slice1 = network->addSlice(
        *input,
        nvinfer1::Dims3{0, 0, 0},
        nvinfer1::Dims3{prevTensorDims.d[0], prevTensorDims.d[1] / 2, prevTensorDims.d[2] / 2},
        nvinfer1::Dims3{1, 2, 2});
    assert(slice1 != nullptr);
    std::string slice1LayerName = "slice1_" + std::to_string(layerIdx);
    slice1->setName(slice1LayerName.c_str());

    nvinfer1::ISliceLayer *slice2 = network->addSlice(
        *input,
        nvinfer1::Dims3{0, 1, 0},
        nvinfer1::Dims3{prevTensorDims.d[0], prevTensorDims.d[1] / 2, prevTensorDims.d[2] / 2},
        nvinfer1::Dims3{1, 2, 2});
    assert(slice2 != nullptr);
    std::string slice2LayerName = "slice2_" + std::to_string(layerIdx);
    slice2->setName(slice2LayerName.c_str());

    nvinfer1::ISliceLayer *slice3 = network->addSlice(
        *input,
        nvinfer1::Dims3{0, 0, 1},
        nvinfer1::Dims3{prevTensorDims.d[0], prevTensorDims.d[1] / 2, prevTensorDims.d[2] / 2},
        nvinfer1::Dims3{1, 2, 2});
    assert(slice3 != nullptr);
    std::string slice3LayerName = "slice3_" + std::to_string(layerIdx);
    slice3->setName(slice3LayerName.c_str());

    nvinfer1::ISliceLayer *slice4 = network->addSlice(
        *input,
        nvinfer1::Dims3{0, 1, 1},
        nvinfer1::Dims3{prevTensorDims.d[0], prevTensorDims.d[1] / 2, prevTensorDims.d[2] / 2},
        nvinfer1::Dims3{1, 2, 2});
    assert(slice4 != nullptr);
    std::string slice4LayerName = "slice4_" + std::to_string(layerIdx);
    slice4->setName(slice4LayerName.c_str());

    std::vector<nvinfer1::ITensor*> concatInputs;
    concatInputs.push_back (slice1->getOutput(0));
    concatInputs.push_back (slice2->getOutput(0));
    concatInputs.push_back (slice3->getOutput(0));
    concatInputs.push_back (slice4->getOutput(0));

    nvinfer1::IConcatenationLayer* concat =
        network->addConcatenation(concatInputs.data(), concatInputs.size());
    assert(concat != nullptr);

    return concat;
}
