/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "reg_layer.h"

nvinfer1::ITensor* regLayer(
    int layerIdx,
    std::map<std::string, std::string>& block,
    std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights,
    int& weightPtr,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
    nvinfer1::ITensor* output;

    assert(block.at("type") == "reg");

    nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*input);
    assert(shuffle != nullptr);
    std::string shuffleLayerName = "shuffle_" + std::to_string(layerIdx);
    shuffle->setName(shuffleLayerName.c_str());
    nvinfer1::Permutation permutation;
    permutation.order[0] = 1;
    permutation.order[1] = 0;
    shuffle->setFirstTranspose(permutation);
    output = shuffle->getOutput(0);
    nvinfer1::Dims shuffleDims = output->getDimensions();

    nvinfer1::ISliceLayer* sliceLt = network->addSlice(
        *output, nvinfer1::Dims{2, {0, 0}}, nvinfer1::Dims{2, {shuffleDims.d[0], 2}}, nvinfer1::Dims{2, {1, 1}});
    assert(sliceLt != nullptr);
    std::string sliceLtLayerName = "slice_lt_" + std::to_string(layerIdx);
    sliceLt->setName(sliceLtLayerName.c_str());
    nvinfer1::ITensor* lt = sliceLt->getOutput(0);

    nvinfer1::ISliceLayer* sliceRb = network->addSlice(
        *output, nvinfer1::Dims{2, {0, 2}}, nvinfer1::Dims{2, {shuffleDims.d[0], 2}}, nvinfer1::Dims{2, {1, 1}});
    assert(sliceRb != nullptr);
    std::string sliceRbLayerName = "slice_rb_" + std::to_string(layerIdx);
    sliceRb->setName(sliceRbLayerName.c_str());
    nvinfer1::ITensor* rb = sliceRb->getOutput(0);

    int channels = shuffleDims.d[0] * 2;
    nvinfer1::Weights anchorPointsWt{nvinfer1::DataType::kFLOAT, nullptr, channels};
    float* val = new float[channels];
    for (int i = 0; i < channels; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    anchorPointsWt.values = val;
    trtWeights.push_back(anchorPointsWt);

    nvinfer1::IConstantLayer* anchorPoints = network->addConstant(nvinfer1::Dims{2, {shuffleDims.d[0], 2}}, anchorPointsWt);
    assert(anchorPoints != nullptr);
    std::string anchorPointsLayerName = "anchor_points_" + std::to_string(layerIdx);
    anchorPoints->setName(anchorPointsLayerName.c_str());
    nvinfer1::ITensor* anchorPointsTensor = anchorPoints->getOutput(0);

    nvinfer1::IElementWiseLayer* x1y1
        = network->addElementWise(*anchorPointsTensor, *lt, nvinfer1::ElementWiseOperation::kSUB);
    assert(x1y1 != nullptr);
    std::string x1y1LayerName = "x1y1_" + std::to_string(layerIdx);
    x1y1->setName(x1y1LayerName.c_str());
    nvinfer1::ITensor* x1y1Tensor = x1y1->getOutput(0);

    nvinfer1::IElementWiseLayer* x2y2
        = network->addElementWise(*rb, *anchorPointsTensor, nvinfer1::ElementWiseOperation::kSUM);
    assert(x2y2 != nullptr);
    std::string x2y2LayerName = "x2y2_" + std::to_string(layerIdx);
    x2y2->setName(x2y2LayerName.c_str());
    nvinfer1::ITensor* x2y2Tensor = x2y2->getOutput(0);

    std::vector<nvinfer1::ITensor*> concatInputs;
    concatInputs.push_back(x1y1Tensor);
    concatInputs.push_back(x2y2Tensor);

    nvinfer1::IConcatenationLayer* concat = network->addConcatenation(concatInputs.data(), concatInputs.size());
    assert(concat != nullptr);
    std::string concatLayerName = "concat_" + std::to_string(layerIdx);
    concat->setName(concatLayerName.c_str());
    concat->setAxis(1);
    output = concat->getOutput(0);

    channels = shuffleDims.d[0];
    nvinfer1::Weights stridePointsWt{nvinfer1::DataType::kFLOAT, nullptr, channels};
    val = new float[channels];
    for (int i = 0; i < channels; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    stridePointsWt.values = val;
    trtWeights.push_back(stridePointsWt);

    nvinfer1::IConstantLayer* stridePoints = network->addConstant(nvinfer1::Dims{2, {shuffleDims.d[0], 1}}, stridePointsWt);
    assert(stridePoints != nullptr);
    std::string stridePointsLayerName = "stride_points_" + std::to_string(layerIdx);
    stridePoints->setName(stridePointsLayerName.c_str());
    nvinfer1::ITensor* stridePointsTensor = stridePoints->getOutput(0);

    nvinfer1::IElementWiseLayer* pred
        = network->addElementWise(*output, *stridePointsTensor, nvinfer1::ElementWiseOperation::kPROD);
    assert(pred != nullptr);
    std::string predLayerName = "pred_" + std::to_string(layerIdx);
    pred->setName(predLayerName.c_str());
    output = pred->getOutput(0);

    return output;
}
