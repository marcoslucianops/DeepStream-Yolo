/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __ROUTE_LAYER_H__
#define __ROUTE_LAYER_H__

#include "NvInfer.h"
#include "../utils.h"

nvinfer1::ILayer* routeLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    std::vector<nvinfer1::ITensor*> tensorOutputs,
    nvinfer1::INetworkDefinition* network);

#endif