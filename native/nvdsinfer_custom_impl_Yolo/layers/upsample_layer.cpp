/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "upsample_layer.h"

nvinfer1::ILayer* upsampleLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights,
    int& inputChannels,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "upsample");
    nvinfer1::Dims inpDims = input->getDimensions();
    assert(inpDims.nbDims == 3);
    assert(inpDims.d[1] == inpDims.d[2]);
    int h = inpDims.d[1];
    int w = inpDims.d[2];
    int stride = std::stoi(block.at("stride"));

    nvinfer1::Dims preDims{3,
                           {1, stride * h, w},
                           {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
                            nvinfer1::DimensionType::kSPATIAL}};
    int size = stride * h * w;
    nvinfer1::Weights preMul{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* preWt = new float[size];

    for (int i = 0, idx = 0; i < h; ++i)
    {
        for (int s = 0; s < stride; ++s)
        {
            for (int j = 0; j < w; ++j, ++idx)
            {
                preWt[idx] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    preMul.values = preWt;
    trtWeights.push_back(preMul);
    nvinfer1::IConstantLayer* preM = network->addConstant(preDims, preMul);
    assert(preM != nullptr);
    std::string preLayerName = "preMul_" + std::to_string(layerIdx);
    preM->setName(preLayerName.c_str());

    nvinfer1::Dims postDims{3,
                            {1, h, stride * w},
                            {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
                             nvinfer1::DimensionType::kSPATIAL}};
    size = stride * h * w;
    nvinfer1::Weights postMul{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* postWt = new float[size];

    for (int i = 0, idx = 0; i < h; ++i)
    {
        for (int j = 0; j < stride * w; ++j, ++idx)
        {
            postWt[idx] = (j / stride == i) ? 1.0 : 0.0;
        }
    }
    postMul.values = postWt;
    trtWeights.push_back(postMul);
    nvinfer1::IConstantLayer* post_m = network->addConstant(postDims, postMul);
    assert(post_m != nullptr);
    std::string postLayerName = "postMul_" + std::to_string(layerIdx);
    post_m->setName(postLayerName.c_str());

    nvinfer1::IMatrixMultiplyLayer* mm1
        = network->addMatrixMultiply(*preM->getOutput(0), nvinfer1::MatrixOperation::kNONE, *input,
                                     nvinfer1::MatrixOperation::kNONE);
    assert(mm1 != nullptr);
    std::string mm1LayerName = "mm1_" + std::to_string(layerIdx);
    mm1->setName(mm1LayerName.c_str());
    nvinfer1::IMatrixMultiplyLayer* mm2
        = network->addMatrixMultiply(*mm1->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                     *post_m->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    assert(mm2 != nullptr);
    std::string mm2LayerName = "mm2_" + std::to_string(layerIdx);
    mm2->setName(mm2LayerName.c_str());

    return mm2;
}