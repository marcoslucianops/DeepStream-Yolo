/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __CHANNELS_LAYER_H__
#define __CHANNELS_LAYER_H__

#include <map>
#include <cassert>

#include "NvInfer.h"

nvinfer1::ILayer* channelsLayer(
    std::string type,
    nvinfer1::ITensor* input,
    nvinfer1::ITensor* implicitTensor,
    nvinfer1::INetworkDefinition* network);

#endif
