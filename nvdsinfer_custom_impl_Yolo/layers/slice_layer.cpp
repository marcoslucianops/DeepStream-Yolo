/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "slice_layer.h"

#include <cassert>

nvinfer1::ITensor*
sliceLayer(int layerIdx, std::string& name, nvinfer1::ITensor* input, nvinfer1::Dims start, nvinfer1::Dims size,
    nvinfer1::Dims stride, nvinfer1::INetworkDefinition* network, uint batchSize)
{
  nvinfer1::ITensor* output;

  int tensorBatch = input->getDimensions().d[0];

  nvinfer1::ISliceLayer* slice = network->addSlice(*input, start, size, stride);

  if (tensorBatch == -1) {
    int nbDims = size.nbDims;

    nvinfer1::Weights constant1Wt {nvinfer1::DataType::kINT32, nullptr, nbDims};

    int* val1 = new int[nbDims];
    val1[0] = 1;
    for (int i = 1; i < nbDims; ++i) {
      val1[i] = size.d[i];
    }
    constant1Wt.values = val1;

    nvinfer1::IConstantLayer* constant1 = network->addConstant(nvinfer1::Dims{1, {nbDims}}, constant1Wt);
    assert(constant1 != nullptr);
    std::string constant1LayerName = "constant1_" + name + "_" + std::to_string(layerIdx);
    constant1->setName(constant1LayerName.c_str());
    nvinfer1::ITensor* constant1Tensor = constant1->getOutput(0);

    nvinfer1::Weights constant2Wt {nvinfer1::DataType::kINT32, nullptr, nbDims};

    int* val2 = new int[nbDims];
    val2[0] = batchSize;
    for (int i = 1; i < nbDims; ++i) {
      val2[i] = 1;
    }
    constant2Wt.values = val2;

    nvinfer1::IConstantLayer* constant2 = network->addConstant(nvinfer1::Dims{1, {nbDims}}, constant2Wt);
    assert(constant2 != nullptr);
    std::string constant2LayerName = "constant2_" + name + "_" + std::to_string(layerIdx);
    constant2->setName(constant2LayerName.c_str());
    nvinfer1::ITensor* constant2Tensor = constant2->getOutput(0);

    nvinfer1::IElementWiseLayer* newSize = network->addElementWise(*constant1Tensor, *constant2Tensor,
        nvinfer1::ElementWiseOperation::kPROD);
    assert(newSize != nullptr);
    std::string newSizeLayerName = "new_size_" + name + "_" + std::to_string(layerIdx);
    newSize->setName(newSizeLayerName.c_str());
    nvinfer1::ITensor* newSizeTensor = newSize->getOutput(0);

    slice->setInput(2, *newSizeTensor);
  }

  assert(slice != nullptr);
  std::string sliceLayerName = name + "_" + std::to_string(layerIdx);
  slice->setName(sliceLayerName.c_str());
  output = slice->getOutput(0);

  return output;
}
