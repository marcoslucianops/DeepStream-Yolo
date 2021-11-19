/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __UPSAMPLE_LAYER_H__
#define __UPSAMPLE_LAYER_H__

#include <map>
#include <vector>
#include <cassert>

#include "NvInfer.h"

nvinfer1::ILayer* upsampleLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network);

#endif
