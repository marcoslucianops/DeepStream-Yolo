/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.

 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "yolo.h"
#include "yoloPlugins.h"

void orderParams(std::vector<std::vector<int>> *maskVector) {
    std::vector<std::vector<int>> maskinput = *maskVector;
    std::vector<int> maskPartial;
    for (uint i = 0; i < maskinput.size(); i++) {
		for (uint j = i + 1; j < maskinput.size(); j++) {
			if (maskinput[i][0] <= maskinput[j][0]) {
				maskPartial = maskinput[i];
				maskinput[i] = maskinput[j];
				maskinput[j] = maskPartial;
            }
		}
	}
    *maskVector = maskinput;
}

Yolo::Yolo(const NetworkInfo& networkInfo)
    : m_NetworkType(networkInfo.networkType), // YOLO type
      m_ConfigFilePath(networkInfo.configFilePath), // YOLO cfg
      m_WtsFilePath(networkInfo.wtsFilePath), // YOLO weights
      m_DeviceType(networkInfo.deviceType), // kDLA, kGPU
      m_InputBlobName(networkInfo.inputBlobName), // data
      m_InputH(0),
      m_InputW(0),
      m_InputC(0),
      m_InputSize(0)
{}

Yolo::~Yolo()
{
    destroyNetworkUtils();
}

nvinfer1::ICudaEngine *Yolo::createEngine (nvinfer1::IBuilder* builder)
{
    assert (builder);

    std::vector<float> weights = loadWeights(m_WtsFilePath, m_NetworkType);
    std::vector<nvinfer1::Weights> trtWeights;

    nvinfer1::INetworkDefinition *network = builder->createNetwork();
    if (parseModel(*network) != NVDSINFER_SUCCESS) {
        network->destroy();
        return nullptr;
    }

    // Build the engine
    std::cout << "Building the TensorRT Engine" << std::endl;
    nvinfer1::ICudaEngine * engine = builder->buildCudaEngine(*network);
    if (engine) {
        std::cout << "Building complete\n" << std::endl;
    } else {
        std::cerr << "Building engine failed\n" << std::endl;
    }

    // destroy
    network->destroy();
    return engine;
}

NvDsInferStatus Yolo::parseModel(nvinfer1::INetworkDefinition& network) {
    destroyNetworkUtils();

    m_ConfigBlocks = parseConfigFile(m_ConfigFilePath);
    parseConfigBlocks();
    orderParams(&m_OutputMasks);

    std::vector<float> weights = loadWeights(m_WtsFilePath, m_NetworkType);
    // build yolo network
    std::cout << "Building YOLO network" << std::endl;
    NvDsInferStatus status = buildYoloNetwork(weights, network);

    if (status == NVDSINFER_SUCCESS) {
        std::cout << "Building YOLO network complete" << std::endl;
    } else {
        std::cerr << "Building YOLO network failed" << std::endl;
    }

    return status;
}

NvDsInferStatus Yolo::buildYoloNetwork(
    std::vector<float>& weights, nvinfer1::INetworkDefinition& network) {
    int weightPtr = 0;
    int channels = m_InputC;

    nvinfer1::ITensor* data =
        network.addInput(m_InputBlobName.c_str(), nvinfer1::DataType::kFLOAT,
            nvinfer1::DimsCHW{static_cast<int>(m_InputC),
                static_cast<int>(m_InputH), static_cast<int>(m_InputW)});
    assert(data != nullptr && data->getDimensions().nbDims > 0);

    nvinfer1::ITensor* previous = data;
    std::vector<nvinfer1::ITensor*> tensorOutputs;
    uint outputTensorCount = 0;

    // build the network using the network API
    for (uint i = 0; i < m_ConfigBlocks.size(); ++i) {
        // check if num. of channels is correct
        assert(getNumChannels(previous) == channels);
        std::string layerIndex = "(" + std::to_string(tensorOutputs.size()) + ")";

        if (m_ConfigBlocks.at(i).at("type") == "net") {
            printLayerInfo("", "layer", "     input", "     outup", "weightPtr");
        }
        
        else if (m_ConfigBlocks.at(i).at("type") == "convolutional") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out = convolutionalLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, channels, previous, &network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            channels = getNumChannels(previous);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerType = "conv_" + m_ConfigBlocks.at(i).at("activation");
            printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));
        }

        else if (m_ConfigBlocks.at(i).at("type") == "dropout") {
            assert(m_ConfigBlocks.at(i).find("probability") != m_ConfigBlocks.at(i).end());
            //float probability = std::stof(m_ConfigBlocks.at(i).at("probability"));
            //nvinfer1::ILayer* out = dropoutLayer(probability, previous, &network);
            //previous = out->getOutput(0);
            //Skip dropout layer
            assert(previous != nullptr);
            tensorOutputs.push_back(previous);
            printLayerInfo(layerIndex, "dropout", "        -", "        -", "    -");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "shortcut") {
            assert(m_ConfigBlocks.at(i).find("activation") != m_ConfigBlocks.at(i).end());
            assert(m_ConfigBlocks.at(i).find("from") != m_ConfigBlocks.at(i).end());
            std::string activation = m_ConfigBlocks.at(i).at("activation");
            int from = stoi(m_ConfigBlocks.at(i).at("from"));
            if (from > 0) {
                from = from - i + 1;
            }
            assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
            assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
            assert(i + from - 1 < i - 2);
            std::string inputVol = dimsToString(previous->getDimensions());
            std::string shortcutVol = dimsToString(tensorOutputs[i + from - 1]->getDimensions());
            nvinfer1::ILayer* out = shortcutLayer(i, activation, inputVol, shortcutVol, previous, tensorOutputs[i + from - 1], &network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerType = "shortcut_" + m_ConfigBlocks.at(i).at("activation") + ": " + std::to_string(i + from - 1);
            printLayerInfo(layerIndex, layerType, "        -", outputVol, "    -");
            if (inputVol != shortcutVol) {
                std::cout << inputVol << " +" << shortcutVol << std::endl;
            }
        }

        else if (m_ConfigBlocks.at(i).at("type") == "route") {
            assert(m_ConfigBlocks.at(i).find("layers") != m_ConfigBlocks.at(i).end());
            nvinfer1::ILayer* out = routeLayer(i, m_ConfigBlocks.at(i), tensorOutputs, &network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            channels = getNumChannels(previous);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            printLayerInfo(layerIndex, "route", "        -", outputVol, std::to_string(weightPtr));
        }

        else if (m_ConfigBlocks.at(i).at("type") == "upsample") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out = upsampleLayer(i - 1, m_ConfigBlocks[i], weights, m_TrtWeights, channels, previous, &network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            printLayerInfo(layerIndex, "upsample", inputVol, outputVol, "    -");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "maxpool") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out = maxpoolLayer(i, m_ConfigBlocks.at(i), previous, &network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            printLayerInfo(layerIndex, "maxpool", inputVol, outputVol, std::to_string(weightPtr));
        }

        else if (m_ConfigBlocks.at(i).at("type") == "yolo") {
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
            assert(prevTensorDims.d[1] == prevTensorDims.d[2]);
            TensorInfo& curYoloTensor = m_OutputTensors.at(outputTensorCount);
            curYoloTensor.gridSize = prevTensorDims.d[1];
            curYoloTensor.stride = m_InputW / curYoloTensor.gridSize;
            m_OutputTensors.at(outputTensorCount).volume = curYoloTensor.gridSize
                * curYoloTensor.gridSize
                * (curYoloTensor.numBBoxes * (5 + curYoloTensor.numClasses));
            std::string layerName = "yolo_" + std::to_string(i);
            curYoloTensor.blobName = layerName;
            int new_coords = 0;
            float scale_x_y = 1;
            float beta_nms = 0.45;
            if (m_ConfigBlocks.at(i).find("new_coords") != m_ConfigBlocks.at(i).end()) {
                new_coords = std::stoi(m_ConfigBlocks.at(i).at("new_coords"));
            }
            if (m_ConfigBlocks.at(i).find("scale_x_y") != m_ConfigBlocks.at(i).end()) {
                scale_x_y = std::stof(m_ConfigBlocks.at(i).at("scale_x_y"));
            }
            if (m_ConfigBlocks.at(i).find("beta_nms") != m_ConfigBlocks.at(i).end()) {
                beta_nms = std::stof(m_ConfigBlocks.at(i).at("beta_nms"));
            }
            nvinfer1::IPluginV2* yoloPlugin
                = new YoloLayer(m_OutputTensors.at(outputTensorCount).numBBoxes,
                                  m_OutputTensors.at(outputTensorCount).numClasses,
                                  m_OutputTensors.at(outputTensorCount).gridSize,
                                  1, new_coords, scale_x_y, beta_nms,
                                  curYoloTensor.anchors,
                                  m_OutputMasks);
            assert(yoloPlugin != nullptr);
            nvinfer1::IPluginV2Layer* yolo =
                network.addPluginV2(&previous, 1, *yoloPlugin);
            assert(yolo != nullptr);
            yolo->setName(layerName.c_str());
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = yolo->getOutput(0);
            assert(previous != nullptr);
            previous->setName(layerName.c_str());
            std::string outputVol = dimsToString(previous->getDimensions());
            network.markOutput(*previous);
            channels = getNumChannels(previous);
            tensorOutputs.push_back(yolo->getOutput(0));
            printLayerInfo(layerIndex, "yolo", inputVol, outputVol, std::to_string(weightPtr));
            ++outputTensorCount;
        }

        //YOLOv2 support
        else if (m_ConfigBlocks.at(i).at("type") == "region") {
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
            assert(prevTensorDims.d[1] == prevTensorDims.d[2]);
            TensorInfo& curRegionTensor = m_OutputTensors.at(outputTensorCount);
            curRegionTensor.gridSize = prevTensorDims.d[1];
            curRegionTensor.stride = m_InputW / curRegionTensor.gridSize;
            m_OutputTensors.at(outputTensorCount).volume = curRegionTensor.gridSize
                * curRegionTensor.gridSize
                * (curRegionTensor.numBBoxes * (5 + curRegionTensor.numClasses));
            std::string layerName = "region_" + std::to_string(i);
            curRegionTensor.blobName = layerName;
            std::vector<std::vector<int>> mask;
            nvinfer1::IPluginV2* regionPlugin
                = new YoloLayer(curRegionTensor.numBBoxes,
                                  curRegionTensor.numClasses,
                                  curRegionTensor.gridSize,
                                  0, 0, 1.0, 0,
                                  curRegionTensor.anchors,
                                  mask);
            assert(regionPlugin != nullptr);
            nvinfer1::IPluginV2Layer* region =
                network.addPluginV2(&previous, 1, *regionPlugin);
            assert(region != nullptr);
            region->setName(layerName.c_str());
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = region->getOutput(0);
            assert(previous != nullptr);
            previous->setName(layerName.c_str());
            std::string outputVol = dimsToString(previous->getDimensions());
            network.markOutput(*previous);
            channels = getNumChannels(previous);
            tensorOutputs.push_back(region->getOutput(0));
            printLayerInfo(layerIndex, "region", inputVol, outputVol, std::to_string(weightPtr));
            ++outputTensorCount;
        }
        else if (m_ConfigBlocks.at(i).at("type") == "reorg") {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::IPluginV2* reorgPlugin = createReorgPlugin(2);
            assert(reorgPlugin != nullptr);
            nvinfer1::IPluginV2Layer* reorg =
                network.addPluginV2(&previous, 1, *reorgPlugin);
            assert(reorg != nullptr);
            std::string layerName = "reorg_" + std::to_string(i);
            reorg->setName(layerName.c_str());
            previous = reorg->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            channels = getNumChannels(previous);
            tensorOutputs.push_back(reorg->getOutput(0));
            printLayerInfo(layerIndex, "reorg", inputVol, outputVol, std::to_string(weightPtr));
        }

        else
        {
            std::cout << "Unsupported layer type --> \""
                      << m_ConfigBlocks.at(i).at("type") << "\"" << std::endl;
            assert(0);
        }
    }

    if ((int)weights.size() != weightPtr)
    {
        std::cout << "Number of unused weights left: " << weights.size() - weightPtr << std::endl;
        assert(0);
    }

    std::cout << "Output YOLO blob names: " << std::endl;
    for (auto& tensor : m_OutputTensors) {
        std::cout << tensor.blobName << std::endl;
    }

    int nbLayers = network.getNbLayers();
    std::cout << "Total number of YOLO layers: " << nbLayers << std::endl;

    return NVDSINFER_SUCCESS;
}

std::vector<std::map<std::string, std::string>>
Yolo::parseConfigFile (const std::string cfgFilePath)
{
    assert(fileExists(cfgFilePath));
    std::ifstream file(cfgFilePath);
    assert(file.good());
    std::string line;
    std::vector<std::map<std::string, std::string>> blocks;
    std::map<std::string, std::string> block;

    while (getline(file, line))
    {
        if (line.size() == 0) continue;
        if (line.front() == '#') continue;
        line = trim(line);
        if (line.front() == '[')
        {
            if (block.size() > 0)
            {
                blocks.push_back(block);
                block.clear();
            }
            std::string key = "type";
            std::string value = trim(line.substr(1, line.size() - 2));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
        else
        {
            int cpos = line.find('=');
            std::string key = trim(line.substr(0, cpos));
            std::string value = trim(line.substr(cpos + 1));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
    }
    blocks.push_back(block);
    return blocks;
}

void Yolo::parseConfigBlocks()
{
    for (auto block : m_ConfigBlocks) {
        if (block.at("type") == "net")
        {
            assert((block.find("height") != block.end())
                   && "Missing 'height' param in network cfg");
            assert((block.find("width") != block.end()) && "Missing 'width' param in network cfg");
            assert((block.find("channels") != block.end())
                   && "Missing 'channels' param in network cfg");

            m_InputH = std::stoul(block.at("height"));
            m_InputW = std::stoul(block.at("width"));
            m_InputC = std::stoul(block.at("channels"));
            assert(m_InputW == m_InputH);
            m_InputSize = m_InputC * m_InputH * m_InputW;
        }
        else if ((block.at("type") == "region") || (block.at("type") == "yolo"))
        {
            assert((block.find("num") != block.end())
                   && std::string("Missing 'num' param in " + block.at("type") + " layer").c_str());
            assert((block.find("classes") != block.end())
                   && std::string("Missing 'classes' param in " + block.at("type") + " layer")
                          .c_str());
            assert((block.find("anchors") != block.end())
                   && std::string("Missing 'anchors' param in " + block.at("type") + " layer")
                          .c_str());

            TensorInfo outputTensor;
            std::string anchorString = block.at("anchors");
            while (!anchorString.empty())
            {
                int npos = anchorString.find_first_of(',');
                if (npos != -1)
                {
                    float anchor = std::stof(trim(anchorString.substr(0, npos)));
                    outputTensor.anchors.push_back(anchor);
                    anchorString.erase(0, npos + 1);
                }
                else
                {
                    float anchor = std::stof(trim(anchorString));
                    outputTensor.anchors.push_back(anchor);
                    break;
                }
            }

            
            if (block.find("mask") != block.end()) {

                std::string maskString = block.at("mask");
                std::vector<int> pMASKS;
                while (!maskString.empty())
                {
                    int npos = maskString.find_first_of(',');
                    if (npos != -1)
                    {
                        int mask = std::stoul(trim(maskString.substr(0, npos)));
                        pMASKS.push_back(mask);
                        outputTensor.masks.push_back(mask);
                        maskString.erase(0, npos + 1);
                    }
                    else
                    {
                        int mask = std::stoul(trim(maskString));
                        pMASKS.push_back(mask);
                        outputTensor.masks.push_back(mask);
                        break;
                    }
                }
                m_OutputMasks.push_back(pMASKS);
            }

            outputTensor.numBBoxes = outputTensor.masks.size() > 0
                ? outputTensor.masks.size()
                : std::stoul(trim(block.at("num")));
            outputTensor.numClasses = std::stoul(block.at("classes"));
            m_OutputTensors.push_back(outputTensor);
        }
    }
}

void Yolo::destroyNetworkUtils() {
    // deallocate the weights
    for (uint i = 0; i < m_TrtWeights.size(); ++i) {
        if (m_TrtWeights[i].count > 0)
            free(const_cast<void*>(m_TrtWeights[i].values));
    }
    m_TrtWeights.clear();
}