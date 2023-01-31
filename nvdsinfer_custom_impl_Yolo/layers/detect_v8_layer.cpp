/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "detect_v8_layer.h"

#include <cassert>

nvinfer1::ITensor*
detectV8Layer(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "detect_v8");
  assert(block.find("num") != block.end());
  assert(block.find("classes") != block.end());

  int num = std::stoi(block.at("num"));
  int classes = std::stoi(block.at("classes"));
  int reg_max = num / 4;

  nvinfer1::Dims inputDims = input->getDimensions();

  nvinfer1::ISliceLayer* sliceBox = network->addSlice(*input, nvinfer1::Dims{2, {0, 0}},
      nvinfer1::Dims{2, {num, inputDims.d[1]}}, nvinfer1::Dims{2, {1, 1}});
  assert(sliceBox != nullptr);
  std::string sliceBoxLayerName = "slice_box_" + std::to_string(layerIdx);
  sliceBox->setName(sliceBoxLayerName.c_str());
  nvinfer1::ITensor* box = sliceBox->getOutput(0);

  nvinfer1::ISliceLayer* sliceCls = network->addSlice(*input, nvinfer1::Dims{2, {num, 0}},
      nvinfer1::Dims{2, {classes, inputDims.d[1]}}, nvinfer1::Dims{2, {1, 1}});
  assert(sliceCls != nullptr);
  std::string sliceClsLayerName = "slice_cls_" + std::to_string(layerIdx);
  sliceCls->setName(sliceClsLayerName.c_str());
  nvinfer1::ITensor* cls = sliceCls->getOutput(0);

  nvinfer1::IShuffleLayer* shuffle1Box = network->addShuffle(*box);
  assert(shuffle1Box != nullptr);
  std::string shuffle1BoxLayerName = "shuffle1_box_" + std::to_string(layerIdx);
  shuffle1Box->setName(shuffle1BoxLayerName.c_str());
  nvinfer1::Dims reshape1Dims = {3, {4, reg_max, inputDims.d[1]}};
  shuffle1Box->setReshapeDimensions(reshape1Dims);
  nvinfer1::Permutation permutation1Box;
  permutation1Box.order[0] = 1;
  permutation1Box.order[1] = 0;
  permutation1Box.order[2] = 2;
  shuffle1Box->setSecondTranspose(permutation1Box);
  box = shuffle1Box->getOutput(0);

  nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*box);
  assert(softmax != nullptr);
  std::string softmaxLayerName = "softmax_box_" + std::to_string(layerIdx);
  softmax->setName(softmaxLayerName.c_str());
  softmax->setAxes(1 << 0);
  box = softmax->getOutput(0);

  nvinfer1::Weights dflWt {nvinfer1::DataType::kFLOAT, nullptr, reg_max};

  float* val = new float[reg_max];
  for (int i = 0; i < reg_max; ++i) {
    val[i] = i;
  }
  dflWt.values = val;

  nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(*box, 1, nvinfer1::Dims{2, {1, 1}}, dflWt,
      nvinfer1::Weights{});
  assert(conv != nullptr);
  std::string convLayerName = "conv_box_" + std::to_string(layerIdx);
  conv->setName(convLayerName.c_str());
  conv->setStrideNd(nvinfer1::Dims{2, {1, 1}});
  conv->setPaddingNd(nvinfer1::Dims{2, {0, 0}});
  box = conv->getOutput(0);

  nvinfer1::IShuffleLayer* shuffle2Box = network->addShuffle(*box);
  assert(shuffle2Box != nullptr);
  std::string shuffle2BoxLayerName = "shuffle2_box_" + std::to_string(layerIdx);
  shuffle2Box->setName(shuffle2BoxLayerName.c_str());
  nvinfer1::Dims reshape2Dims = {2, {4, inputDims.d[1]}};
  shuffle2Box->setReshapeDimensions(reshape2Dims);
  box = shuffle2Box->getOutput(0);

  nvinfer1::Dims shuffle2BoxDims = box->getDimensions();

  nvinfer1::ISliceLayer* sliceLtBox = network->addSlice(*box, nvinfer1::Dims{2, {0, 0}},
      nvinfer1::Dims{2, {2, shuffle2BoxDims.d[1]}}, nvinfer1::Dims{2, {1, 1}});
  assert(sliceLtBox != nullptr);
  std::string sliceLtBoxLayerName = "slice_lt_box_" + std::to_string(layerIdx);
  sliceLtBox->setName(sliceLtBoxLayerName.c_str());
  nvinfer1::ITensor* lt = sliceLtBox->getOutput(0);

  nvinfer1::ISliceLayer* sliceRbBox = network->addSlice(*box, nvinfer1::Dims{2, {2, 0}},
      nvinfer1::Dims{2, {2, shuffle2BoxDims.d[1]}}, nvinfer1::Dims{2, {1, 1}});
  assert(sliceRbBox != nullptr);
  std::string sliceRbBoxLayerName = "slice_rb_box_" + std::to_string(layerIdx);
  sliceRbBox->setName(sliceRbBoxLayerName.c_str());
  nvinfer1::ITensor* rb = sliceRbBox->getOutput(0);

  int channels = 2 * shuffle2BoxDims.d[1];
  nvinfer1::Weights anchorPointsWt {nvinfer1::DataType::kFLOAT, nullptr, channels};
  val = new float[channels];
  for (int i = 0; i < channels; ++i) {
    val[i] = weights[weightPtr];
    ++weightPtr;
  }
  anchorPointsWt.values = val;
  trtWeights.push_back(anchorPointsWt);

  nvinfer1::IConstantLayer* anchorPoints = network->addConstant(nvinfer1::Dims{2, {2, shuffle2BoxDims.d[1]}},
      anchorPointsWt);
  assert(anchorPoints != nullptr);
  std::string anchorPointsLayerName = "anchor_points_" + std::to_string(layerIdx);
  anchorPoints->setName(anchorPointsLayerName.c_str());
  nvinfer1::ITensor* anchorPointsTensor = anchorPoints->getOutput(0);

  nvinfer1::IElementWiseLayer* x1y1 = network->addElementWise(*anchorPointsTensor, *lt,
      nvinfer1::ElementWiseOperation::kSUB);
  assert(x1y1 != nullptr);
  std::string x1y1LayerName = "x1y1_" + std::to_string(layerIdx);
  x1y1->setName(x1y1LayerName.c_str());
  nvinfer1::ITensor* x1y1Tensor = x1y1->getOutput(0);

  nvinfer1::IElementWiseLayer* x2y2 = network->addElementWise(*rb, *anchorPointsTensor,
      nvinfer1::ElementWiseOperation::kSUM);
  assert(x2y2 != nullptr);
  std::string x2y2LayerName = "x2y2_" + std::to_string(layerIdx);
  x2y2->setName(x2y2LayerName.c_str());
  nvinfer1::ITensor* x2y2Tensor = x2y2->getOutput(0);

  std::vector<nvinfer1::ITensor*> concatBoxInputs;
  concatBoxInputs.push_back(x1y1Tensor);
  concatBoxInputs.push_back(x2y2Tensor);

  nvinfer1::IConcatenationLayer* concatBox = network->addConcatenation(concatBoxInputs.data(), concatBoxInputs.size());
  assert(concatBox != nullptr);
  std::string concatBoxLayerName = "concat_box_" + std::to_string(layerIdx);
  concatBox->setName(concatBoxLayerName.c_str());
  concatBox->setAxis(0);
  box = concatBox->getOutput(0);

  channels = shuffle2BoxDims.d[1];
  nvinfer1::Weights stridePointsWt {nvinfer1::DataType::kFLOAT, nullptr, channels};
  val = new float[channels];
  for (int i = 0; i < channels; ++i) {
    val[i] = weights[weightPtr];
    ++weightPtr;
  }
  stridePointsWt.values = val;
  trtWeights.push_back(stridePointsWt);

  nvinfer1::IConstantLayer* stridePoints = network->addConstant(nvinfer1::Dims{2, {1, shuffle2BoxDims.d[1]}},
      stridePointsWt);
  assert(stridePoints != nullptr);
  std::string stridePointsLayerName = "stride_points_" + std::to_string(layerIdx);
  stridePoints->setName(stridePointsLayerName.c_str());
  nvinfer1::ITensor* stridePointsTensor = stridePoints->getOutput(0);

  nvinfer1::IElementWiseLayer* pred = network->addElementWise(*box, *stridePointsTensor,
      nvinfer1::ElementWiseOperation::kPROD);
  assert(pred != nullptr);
  std::string predLayerName = "pred_" + std::to_string(layerIdx);
  pred->setName(predLayerName.c_str());
  box = pred->getOutput(0);

  nvinfer1::IActivationLayer* sigmoid = network->addActivation(*cls, nvinfer1::ActivationType::kSIGMOID);
  assert(sigmoid != nullptr);
  std::string sigmoidLayerName = "sigmoid_cls_" + std::to_string(layerIdx);
  sigmoid->setName(sigmoidLayerName.c_str());
  cls = sigmoid->getOutput(0);

  std::vector<nvinfer1::ITensor*> concatInputs;
  concatInputs.push_back(box);
  concatInputs.push_back(cls);

  nvinfer1::IConcatenationLayer* concat = network->addConcatenation(concatInputs.data(), concatInputs.size());
  assert(concat != nullptr);
  std::string concatLayerName = "concat_" + std::to_string(layerIdx);
  concat->setName(concatLayerName.c_str());
  concat->setAxis(0);
  output = concat->getOutput(0);

  nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*output);
  assert(shuffle != nullptr);
  std::string shuffleLayerName = "shuffle_" + std::to_string(layerIdx);
  shuffle->setName(shuffleLayerName.c_str());
  nvinfer1::Permutation permutation;
  permutation.order[0] = 1;
  permutation.order[1] = 0;
  shuffle->setFirstTranspose(permutation);
  output = shuffle->getOutput(0);

  return output;
}
