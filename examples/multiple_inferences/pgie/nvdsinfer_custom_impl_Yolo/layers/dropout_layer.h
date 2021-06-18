/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __DROPOUT_LAYER_H__
#define __DROPOUT_LAYER_H__

#include "NvInfer.h"

nvinfer1::ILayer* dropoutLayer(
    float probability,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network);

#endif
