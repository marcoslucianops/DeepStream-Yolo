/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "reorg_layer.h"

#include <vector>
#include <cassert>

nvinfer1::ITensor*
reorgLayer(int layerIdx, std::map<std::string, std::string>& block, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "reorg" || block.at("type") == "reorg3d");

  int stride = 1;
  if(block.find("stride") != block.end()) {
    stride = std::stoi(block.at("stride"));
  }

  nvinfer1::Dims inputDims = input->getDimensions();

  if (block.at("type") == "reorg3d") {
    nvinfer1::ISliceLayer* slice1 = network->addSlice(*input, nvinfer1::Dims{4, {0, 0, 0, 0}},
        nvinfer1::Dims{4, {inputDims.d[0], inputDims.d[1], inputDims.d[2] / stride, inputDims.d[3] / stride}},
        nvinfer1::Dims{4, {1, 1, stride, stride}});
    assert(slice1 != nullptr);
    std::string slice1LayerName = "slice1_" + std::to_string(layerIdx);
    slice1->setName(slice1LayerName.c_str());

    nvinfer1::ISliceLayer* slice2 = network->addSlice(*input, nvinfer1::Dims{4, {0, 0, 0, 1}},
        nvinfer1::Dims{4, {inputDims.d[0], inputDims.d[1], inputDims.d[2] / stride, inputDims.d[3] / stride}},
        nvinfer1::Dims{4, {1, 1, stride, stride}});
    assert(slice2 != nullptr);
    std::string slice2LayerName = "slice2_" + std::to_string(layerIdx);
    slice2->setName(slice2LayerName.c_str());

    nvinfer1::ISliceLayer* slice3 = network->addSlice(*input, nvinfer1::Dims{4, {0, 0, 1, 0}},
        nvinfer1::Dims{4, {inputDims.d[0], inputDims.d[1], inputDims.d[2] / stride, inputDims.d[3] / stride}},
        nvinfer1::Dims{4, {1, 1, stride, stride}});
    assert(slice3 != nullptr);
    std::string slice3LayerName = "slice3_" + std::to_string(layerIdx);
    slice3->setName(slice3LayerName.c_str());

    nvinfer1::ISliceLayer* slice4 = network->addSlice(*input, nvinfer1::Dims{4, {0, 0, 1, 1}},
        nvinfer1::Dims{4, {inputDims.d[0], inputDims.d[1], inputDims.d[2] / stride, inputDims.d[3] / stride}},
        nvinfer1::Dims{4, {1, 1, stride, stride}});
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
  }
  else {
    nvinfer1::IShuffleLayer* shuffle1 = network->addShuffle(*input);
    assert(shuffle1 != nullptr);
    std::string shuffle1LayerName = "shuffle1_" + std::to_string(layerIdx);
    shuffle1->setName(shuffle1LayerName.c_str());
    nvinfer1::Dims reshapeDims1{6, {inputDims.d[0], inputDims.d[1] / (stride * stride), inputDims.d[2], stride,
        inputDims.d[3], stride}};
    shuffle1->setReshapeDimensions(reshapeDims1);
    nvinfer1::Permutation permutation1{{0, 1, 2, 4, 3, 5}};
    shuffle1->setSecondTranspose(permutation1);
    output = shuffle1->getOutput(0);

    nvinfer1::IShuffleLayer* shuffle2 = network->addShuffle(*output);
    assert(shuffle2 != nullptr);
    std::string shuffle2LayerName = "shuffle2_" + std::to_string(layerIdx);
    shuffle2->setName(shuffle2LayerName.c_str());
    nvinfer1::Dims reshapeDims2{4, {inputDims.d[0], inputDims.d[1] / (stride * stride), inputDims.d[2] * inputDims.d[3],
        stride * stride}};
    shuffle2->setReshapeDimensions(reshapeDims2);
    nvinfer1::Permutation permutation2{{0, 1, 3, 2}};
    shuffle2->setSecondTranspose(permutation2);
    output = shuffle2->getOutput(0);

    nvinfer1::IShuffleLayer* shuffle3 = network->addShuffle(*output);
    assert(shuffle3 != nullptr);
    std::string shuffle3LayerName = "shuffle3_" + std::to_string(layerIdx);
    shuffle3->setName(shuffle3LayerName.c_str());
    nvinfer1::Dims reshapeDims3{4, {inputDims.d[0], inputDims.d[1] / (stride * stride), stride * stride,
        inputDims.d[2] * inputDims.d[3]}};
    shuffle3->setReshapeDimensions(reshapeDims3);
    nvinfer1::Permutation permutation3{{0, 2, 1, 3}};
    shuffle3->setSecondTranspose(permutation3);
    output = shuffle3->getOutput(0);

    nvinfer1::IShuffleLayer* shuffle4 = network->addShuffle(*output);
    assert(shuffle4 != nullptr);
    std::string shuffle4LayerName = "shuffle4_" + std::to_string(layerIdx);
    shuffle4->setName(shuffle4LayerName.c_str());
    nvinfer1::Dims reshapeDims4{4, {inputDims.d[0], inputDims.d[1] * stride * stride, inputDims.d[2] / stride,
        inputDims.d[3] / stride}};
    shuffle4->setReshapeDimensions(reshapeDims4);
    output = shuffle4->getOutput(0);
  }

  return output;
}
