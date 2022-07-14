/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "maxpool_layer.h"

nvinfer1::ILayer* maxpoolLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "maxpool");
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    int size = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));

    nvinfer1::IPoolingLayer* pool
        = network->addPoolingNd(*input, nvinfer1::PoolingType::kMAX, nvinfer1::Dims{2, {size, size}});
    assert(pool);
    std::string maxpoolLayerName = "maxpool_" + std::to_string(layerIdx);
    pool->setStrideNd(nvinfer1::Dims{2, {stride, stride}});
    if (stride == 1)
    {
        pool->setPaddingNd(nvinfer1::Dims{2, {size / 2, size / 2}});
    }
    pool->setName(maxpoolLayerName.c_str());

    return pool;
}
