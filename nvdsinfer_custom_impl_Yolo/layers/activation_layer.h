/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __ACTIVATION_LAYER_H__
#define __ACTIVATION_LAYER_H__

#include <string>
#include <cassert>

#include "NvInfer.h"

#include "activation_layer.h"

nvinfer1::ILayer* activationLayer(
    int layerIdx,
    std::string activation,
    nvinfer1::ILayer* output,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network);

#endif
