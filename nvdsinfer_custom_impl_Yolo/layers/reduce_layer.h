/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __REDUCE_LAYER_H__
#define __REDUCE_LAYER_H__

#include "NvInfer.h"
#include "../utils.h"

nvinfer1::ITensor* reduceLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network);

#endif
