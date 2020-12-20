/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __SHORTCUT_LAYER_H__
#define __SHORTCUT_LAYER_H__

#include "NvInfer.h"

#include "activation_layer.h"

nvinfer1::ILayer* shortcutLayer(
    int layerIdx,
    std::string activation,
    std::string inputVol,
    std::string shortcutVol,
    nvinfer1::ITensor* input,
    nvinfer1::ITensor* shortcutTensor,
    nvinfer1::INetworkDefinition* network);

#endif
