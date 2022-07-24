/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __SHUFFLE_LAYER_H__
#define __SHUFFLE_LAYER_H__

#include "NvInfer.h"
#include "../utils.h"

nvinfer1::ITensor* shuffleLayer(
    int layerIdx,
    std::string& layer,
    std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input,
    std::vector<nvinfer1::ITensor*> tensorOutputs,
    nvinfer1::INetworkDefinition* network);

#endif
