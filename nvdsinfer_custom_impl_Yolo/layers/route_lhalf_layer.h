/*
 * Created by Pablo Santana :)
 * https://www.github.com/pabsan-0
 */


#ifndef __ROUTE_LHALF_LAYER_H__
#define __ROUTE_LHALF_LAYER_H__

#include "NvInfer.h"
#include "../utils.h"

nvinfer1::ITensor* route_lhalfLayer(
    int layerIdx,
    std::string& layers,
    std::map<std::string, std::string>& block,
    std::vector<nvinfer1::ITensor*> tensorOutputs,
    nvinfer1::INetworkDefinition* network);

#endif
