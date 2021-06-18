/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __MAXPOOL_LAYER_H__
#define __MAXPOOL_LAYER_H__

#include <map>
#include <cassert>

#include "NvInfer.h"

nvinfer1::ILayer* maxpoolLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network);

#endif
