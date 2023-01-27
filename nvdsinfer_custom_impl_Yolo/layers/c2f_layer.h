/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __C2F_LAYER_H__
#define __C2F_LAYER_H__

#include <map>
#include <vector>

#include "NvInfer.h"

nvinfer1::ITensor* c2fLayer(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, std::string weightsType, float eps, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network);

#endif
