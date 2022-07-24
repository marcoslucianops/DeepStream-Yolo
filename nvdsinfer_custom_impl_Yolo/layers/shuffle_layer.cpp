/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "shuffle_layer.h"

nvinfer1::ITensor* shuffleLayer(
    int layerIdx,
    std::string& layer,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    std::vector<nvinfer1::ITensor*> tensorOutputs,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::ITensor* output;

    assert(block.at("type") == "shuffle");

    nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*input);
    assert(shuffle != nullptr);
    std::string shuffleLayerName = "shuffle_" + std::to_string(layerIdx);
    shuffle->setName(shuffleLayerName.c_str());

    if (block.find("reshape") != block.end())
    {
        std::string strReshape = block.at("reshape");
        std::vector<int32_t> reshape;
        size_t lastPos = 0, pos = 0;
        while ((pos = strReshape.find(',', lastPos)) != std::string::npos)
        {
            int vL = std::stoi(trim(strReshape.substr(lastPos, pos - lastPos)));
            reshape.push_back(vL);
            lastPos = pos + 1;
        }
        if (lastPos < strReshape.length())
        {
            std::string lastV = trim(strReshape.substr(lastPos));
            if (!lastV.empty())
                reshape.push_back(std::stoi(lastV));
        }
        assert(!reshape.empty());

        int from = -1;
        if (block.find("from") != block.end())
            from = std::stoi(block.at("from"));

        if (from < 0)
            from = tensorOutputs.size() + from;

        layer = std::to_string(from);

        nvinfer1::Dims inputTensorDims = tensorOutputs[from]->getDimensions();
        int32_t l = inputTensorDims.d[1] * inputTensorDims.d[2];
        
        nvinfer1::Dims reshapeDims;
        reshapeDims.nbDims = reshape.size();

        for (uint i = 0; i < reshape.size(); ++i)
            if (reshape[i] == 0)
                reshapeDims.d[i] = l;
            else
                reshapeDims.d[i] = reshape[i];

        shuffle->setReshapeDimensions(reshapeDims);
    }

    if (block.find("transpose1") != block.end())
    {
        std::string strTranspose1 = block.at("transpose1");
        std::vector<int32_t> transpose1;
        size_t lastPos = 0, pos = 0;
        while ((pos = strTranspose1.find(',', lastPos)) != std::string::npos)
        {
            int vL = std::stoi(trim(strTranspose1.substr(lastPos, pos - lastPos)));
            transpose1.push_back(vL);
            lastPos = pos + 1;
        }
        if (lastPos < strTranspose1.length())
        {
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

    if (block.find("transpose2") != block.end())
    {
        std::string strTranspose2 = block.at("transpose2");
        std::vector<int32_t> transpose2;
        size_t lastPos = 0, pos = 0;
        while ((pos = strTranspose2.find(',', lastPos)) != std::string::npos)
        {
            int vL = std::stoi(trim(strTranspose2.substr(lastPos, pos - lastPos)));
            transpose2.push_back(vL);
            lastPos = pos + 1;
        }
        if (lastPos < strTranspose2.length())
        {
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
