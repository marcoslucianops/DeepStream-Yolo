/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "activation_layer.h"

nvinfer1::ILayer* activationLayer(
    int layerIdx,
    std::string activation,
    nvinfer1::ILayer* output,
    nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
    if (activation == "relu")
    {
        nvinfer1::IActivationLayer* relu = network->addActivation(
            *input, nvinfer1::ActivationType::kRELU);
        assert(relu != nullptr);
        std::string reluLayerName = "relu_" + std::to_string(layerIdx);
        relu->setName(reluLayerName.c_str());
        output = relu;
    }
    else if (activation == "sigmoid" || activation == "logistic")
    {
        nvinfer1::IActivationLayer* sigmoid = network->addActivation(
            *input, nvinfer1::ActivationType::kSIGMOID);
        assert(sigmoid != nullptr);
        std::string sigmoidLayerName = "sigmoid_" + std::to_string(layerIdx);
        sigmoid->setName(sigmoidLayerName.c_str());
        output = sigmoid;
    }
    else if (activation == "tanh")
    {
        nvinfer1::IActivationLayer* tanh = network->addActivation(
            *input, nvinfer1::ActivationType::kTANH);
        assert(tanh != nullptr);
        std::string tanhLayerName = "tanh_" + std::to_string(layerIdx);
        tanh->setName(tanhLayerName.c_str());
        output = tanh;
    }
    else if (activation == "leaky")
    {
        nvinfer1::IActivationLayer* leaky = network->addActivation(
            *input, nvinfer1::ActivationType::kLEAKY_RELU);
        leaky->setAlpha(0.1);
        assert(leaky != nullptr);
        std::string leakyLayerName = "leaky_" + std::to_string(layerIdx);
        leaky->setName(leakyLayerName.c_str());
        output = leaky;
    }
    else if (activation == "softplus")
    {
        nvinfer1::IActivationLayer* softplus = network->addActivation(
            *input, nvinfer1::ActivationType::kSOFTPLUS);
        assert(softplus != nullptr);
        std::string softplusLayerName = "softplus_" + std::to_string(layerIdx);
        softplus->setName(softplusLayerName.c_str());
        output = softplus;
    }
    else if (activation == "mish")
    {
        nvinfer1::IActivationLayer* softplus = network->addActivation(
            *input, nvinfer1::ActivationType::kSOFTPLUS);
        assert(softplus != nullptr);
        std::string softplusLayerName = "softplus_" + std::to_string(layerIdx);
        softplus->setName(softplusLayerName.c_str());
        nvinfer1::IActivationLayer* tanh = network->addActivation(
            *softplus->getOutput(0), nvinfer1::ActivationType::kTANH);
        assert(tanh != nullptr);
        std::string tanhLayerName = "tanh_" + std::to_string(layerIdx);
        tanh->setName(tanhLayerName.c_str());
        nvinfer1::IElementWiseLayer* mish = network->addElementWise(
            *tanh->getOutput(0), *input,
            nvinfer1::ElementWiseOperation::kPROD);
        assert(mish != nullptr);
        std::string mishLayerName = "mish_" + std::to_string(layerIdx);
        mish->setName(mishLayerName.c_str());
        output = mish;
    }
    return output;
}