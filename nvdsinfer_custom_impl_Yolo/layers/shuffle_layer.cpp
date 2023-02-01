/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "shuffle_layer.h"

nvinfer1::ITensor*
shuffleLayer(int layerIdx, std::map<std::string, std::string>& block, nvinfer1::ITensor* input,
    std::vector<nvinfer1::ITensor*> tensorOutputs, nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "shuffle");

  nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*input);
  assert(shuffle != nullptr);
  std::string shuffleLayerName = "shuffle_" + std::to_string(layerIdx);
  shuffle->setName(shuffleLayerName.c_str());

  if (block.find("reshape") != block.end()) {
    nvinfer1::Dims inputTensorDims = input->getDimensions();

    std::string strReshape = block.at("reshape");
    std::vector<int32_t> reshape;
    size_t lastPos = 0, pos = 0;
    while ((pos = strReshape.find(',', lastPos)) != std::string::npos) {
      std::string V = trim(strReshape.substr(lastPos, pos - lastPos));
      if (V == "c")
        reshape.push_back(inputTensorDims.d[0]);
      else if (V == "ch")
        reshape.push_back(inputTensorDims.d[0] * inputTensorDims.d[1]);
      else if (V == "cw")
        reshape.push_back(inputTensorDims.d[0] * inputTensorDims.d[2]);
      else if (V == "h")
        reshape.push_back(inputTensorDims.d[1]);
      else if (V == "hw")
        reshape.push_back(inputTensorDims.d[1] * inputTensorDims.d[2]);
      else if (V == "w")
        reshape.push_back(inputTensorDims.d[2]);
      else if (V == "chw")
        reshape.push_back(inputTensorDims.d[0] * inputTensorDims.d[1] * inputTensorDims.d[2]);
      else
        reshape.push_back(std::stoi(V));
      lastPos = pos + 1;
    }
    if (lastPos < strReshape.length()) {
      std::string lastV = trim(strReshape.substr(lastPos));
      if (!lastV.empty()) {
        if (lastV == "c")
          reshape.push_back(inputTensorDims.d[0]);
        else if (lastV == "ch")
          reshape.push_back(inputTensorDims.d[0] * inputTensorDims.d[1]);
        else if (lastV == "cw")
          reshape.push_back(inputTensorDims.d[0] * inputTensorDims.d[2]);
        else if (lastV == "h")
          reshape.push_back(inputTensorDims.d[1]);
        else if (lastV == "hw")
          reshape.push_back(inputTensorDims.d[1] * inputTensorDims.d[2]);
        else if (lastV == "w")
          reshape.push_back(inputTensorDims.d[2]);
        else if (lastV == "chw")
          reshape.push_back(inputTensorDims.d[0] * inputTensorDims.d[1] * inputTensorDims.d[2]);
        else
          reshape.push_back(std::stoi(lastV));
      }
    }
    assert(!reshape.empty());

    nvinfer1::Dims reshapeDims;
    reshapeDims.nbDims = reshape.size();

    for (uint i = 0; i < reshape.size(); ++i)
      reshapeDims.d[i] = reshape[i];

    shuffle->setReshapeDimensions(reshapeDims);
  }

  if (block.find("transpose1") != block.end()) {
    std::string strTranspose1 = block.at("transpose1");
    std::vector<int32_t> transpose1;
    size_t lastPos = 0, pos = 0;
    while ((pos = strTranspose1.find(',', lastPos)) != std::string::npos) {
      int vL = std::stoi(trim(strTranspose1.substr(lastPos, pos - lastPos)));
      transpose1.push_back(vL);
      lastPos = pos + 1;
    }
    if (lastPos < strTranspose1.length()) {
      std::string lastV = trim(strTranspose1.substr(lastPos));
      if (!lastV.empty())
        transpose1.push_back(std::stoi(lastV));
    }
    assert(!transpose1.empty());

    nvinfer1::Permutation permutation1;
    for (uint i = 0; i < transpose1.size(); ++i)
      permutation1.order[i] = transpose1[i];

    shuffle->setFirstTranspose(permutation1);
  }

  if (block.find("transpose2") != block.end()) {
    std::string strTranspose2 = block.at("transpose2");
    std::vector<int32_t> transpose2;
    size_t lastPos = 0, pos = 0;
    while ((pos = strTranspose2.find(',', lastPos)) != std::string::npos) {
      int vL = std::stoi(trim(strTranspose2.substr(lastPos, pos - lastPos)));
      transpose2.push_back(vL);
      lastPos = pos + 1;
    }
    if (lastPos < strTranspose2.length()) {
      std::string lastV = trim(strTranspose2.substr(lastPos));
      if (!lastV.empty())
        transpose2.push_back(std::stoi(lastV));
    }
    assert(!transpose2.empty());

    nvinfer1::Permutation permutation2;
    for (uint i = 0; i < transpose2.size(); ++i)
      permutation2.order[i] = transpose2[i];

    shuffle->setSecondTranspose(permutation2);
  }

  output = shuffle->getOutput(0);

  return output;
}
