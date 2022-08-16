/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "yolo.h"
#include "yoloPlugins.h"
#include <stdlib.h>

#ifdef OPENCV
#include "calibrator.h"
#endif

Yolo::Yolo(const NetworkInfo& networkInfo)
    : m_InputBlobName(networkInfo.inputBlobName),
      m_NetworkType(networkInfo.networkType),
      m_ConfigFilePath(networkInfo.configFilePath),
      m_WtsFilePath(networkInfo.wtsFilePath),
      m_Int8CalibPath(networkInfo.int8CalibPath),
      m_DeviceType(networkInfo.deviceType),
      m_NumDetectedClasses(networkInfo.numDetectedClasses),
      m_ClusterMode(networkInfo.clusterMode),
      m_NetworkMode(networkInfo.networkMode),
      m_ScoreThreshold(networkInfo.scoreThreshold),
      m_InputH(0),
      m_InputW(0),
      m_InputC(0),
      m_InputSize(0),
      m_NumClasses(0),
      m_LetterBox(0),
      m_NewCoords(0),
      m_YoloCount(0)
{}

Yolo::~Yolo()
{
    destroyNetworkUtils();
}

nvinfer1::ICudaEngine *Yolo::createEngine(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config)
{
    assert (builder);

    m_ConfigBlocks = parseConfigFile(m_ConfigFilePath);
    parseConfigBlocks();

    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0);
    if (parseModel(*network) != NVDSINFER_SUCCESS)
    {
        delete network;
        return nullptr;
    }

    std::cout << "Building the TensorRT Engine\n" << std::endl;

    if (m_NumClasses != m_NumDetectedClasses)
    {
        std::cout << "NOTE: Number of classes mismatch, make sure to set num-detected-classes=" << m_NumClasses
                  << " in config_infer file\n" << std::endl;
    }
    if (m_LetterBox == 1)
    {
        std::cout << "NOTE: letter_box is set in cfg file, make sure to set maintain-aspect-ratio=1 in config_infer file"
                  << " to get better accuracy\n" << std::endl;
    }
    if (m_ClusterMode != 2)
    {
        std::cout << "NOTE: Wrong cluster-mode is set, make sure to set cluster-mode=2 in config_infer file\n"
                  << std::endl;
    }

    if (m_NetworkMode == "INT8" && !fileExists(m_Int8CalibPath))
    {
        assert(builder->platformHasFastInt8());
#ifdef OPENCV
        std::string calib_image_list;
        int calib_batch_size;
        if (getenv("INT8_CALIB_IMG_PATH"))
            calib_image_list = getenv("INT8_CALIB_IMG_PATH");
        else
        {
            std::cerr << "INT8_CALIB_IMG_PATH not set" << std::endl;
            std::abort();
        }
        if (getenv("INT8_CALIB_BATCH_SIZE"))
            calib_batch_size = std::stoi(getenv("INT8_CALIB_BATCH_SIZE"));
        else
        {
            std::cerr << "INT8_CALIB_BATCH_SIZE not set" << std::endl;
            std::abort();
        }
        nvinfer1::Int8EntropyCalibrator2 *calibrator = new nvinfer1::Int8EntropyCalibrator2(
            calib_batch_size, m_InputC, m_InputH, m_InputW, m_LetterBox, calib_image_list, m_Int8CalibPath);
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator);
#else
        std::cerr << "OpenCV is required to run INT8 calibrator\n" << std::endl;
        assert(0);
#endif
    }

    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine)
        std::cout << "Building complete\n" << std::endl;
    else
        std::cerr << "Building engine failed\n" << std::endl;

    delete network;
    return engine;
}

NvDsInferStatus Yolo::parseModel(nvinfer1::INetworkDefinition& network) {
    destroyNetworkUtils();

    std::vector<float> weights = loadWeights(m_WtsFilePath, m_NetworkType);
    std::cout << "Building YOLO network\n" << std::endl;
    NvDsInferStatus status = buildYoloNetwork(weights, network);

    if (status == NVDSINFER_SUCCESS)
        std::cout << "Building YOLO network complete" << std::endl;
    else
        std::cerr << "Building YOLO network failed" << std::endl;

    return status;
}

NvDsInferStatus Yolo::buildYoloNetwork(std::vector<float>& weights, nvinfer1::INetworkDefinition& network)
{
    int weightPtr = 0;

    std::string weightsType;
    if (m_WtsFilePath.find(".weights") != std::string::npos)
        weightsType = "weights";
    else
        weightsType = "wts";

    float eps = 1.0e-5;
    if (m_NetworkType.find("yolov5") != std::string::npos || m_NetworkType.find("yolov7") != std::string::npos)
        eps = 1.0e-3;
    else if (m_NetworkType.find("yolor") != std::string::npos)
        eps = 1.0e-4;

    nvinfer1::ITensor* data = network.addInput(
        m_InputBlobName.c_str(), nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims{3, {static_cast<int>(m_InputC), static_cast<int>(m_InputH), static_cast<int>(m_InputW)}});
    assert(data != nullptr && data->getDimensions().nbDims > 0);

    nvinfer1::ITensor* previous = data;
    std::vector<nvinfer1::ITensor*> tensorOutputs;

    nvinfer1::ITensor* yoloTensorInputs[m_YoloCount];
    uint yoloCountInputs = 0;

    int modelType = -1;

    for (uint i = 0; i < m_ConfigBlocks.size(); ++i)
    {
        std::string layerIndex = "(" + std::to_string(tensorOutputs.size()) + ")";

        if (m_ConfigBlocks.at(i).at("type") == "net")
            printLayerInfo("", "Layer", "Input Shape", "Output Shape", "WeightPtr");
        
        else if (m_ConfigBlocks.at(i).at("type") == "convolutional")
        {
            int channels = getNumChannels(previous);
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = convolutionalLayer(
                i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, weightsType, channels, eps, previous, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = "conv_" + m_ConfigBlocks.at(i).at("activation");
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
        }

        else if (m_ConfigBlocks.at(i).at("type") == "batchnorm")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = batchnormLayer(
                i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, weightsType, eps, previous, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = "batchnorm_" + m_ConfigBlocks.at(i).at("activation");
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
        }

        else if (m_ConfigBlocks.at(i).at("type") == "implicit_add" || m_ConfigBlocks.at(i).at("type") == "implicit_mul")
        {
            previous = implicitLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName =  m_ConfigBlocks.at(i).at("type");
            printLayerInfo(layerIndex, layerName, "-", outputVol, std::to_string(weightPtr));
        }

        else if (m_ConfigBlocks.at(i).at("type") == "shift_channels" ||
            m_ConfigBlocks.at(i).at("type") == "control_channels")
        {
            assert(m_ConfigBlocks.at(i).find("from") != m_ConfigBlocks.at(i).end());
            int from = stoi(m_ConfigBlocks.at(i).at("from"));
            if (from > 0)
                from = from - i + 1;
            assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
            assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
            assert(i + from - 1 < i - 2);

            std::string inputVol = dimsToString(previous->getDimensions());
            previous = channelsLayer(i, m_ConfigBlocks.at(i), previous, tensorOutputs[i + from - 1], &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = m_ConfigBlocks.at(i).at("type") + ": " + std::to_string(i + from - 1);
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "shortcut")
        {
            assert(m_ConfigBlocks.at(i).find("from") != m_ConfigBlocks.at(i).end());
            int from = stoi(m_ConfigBlocks.at(i).at("from"));
            if (from > 0)
                from = from - i + 1;
            assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
            assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
            assert(i + from - 1 < i - 2);

            std::string mode = "add";
            if (m_ConfigBlocks.at(i).find("mode") != m_ConfigBlocks.at(i).end())
                mode = m_ConfigBlocks.at(i).at("mode");

            std::string activation = "linear";
            if (m_ConfigBlocks.at(i).find("activation") != m_ConfigBlocks.at(i).end())
                activation = m_ConfigBlocks.at(i).at("activation");

            std::string inputVol = dimsToString(previous->getDimensions());
            std::string shortcutVol = dimsToString(tensorOutputs[i + from - 1]->getDimensions());
            previous = shortcutLayer(
                i, mode, activation, inputVol, shortcutVol, m_ConfigBlocks.at(i), previous, tensorOutputs[i + from - 1],
                &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = "shortcut_" + mode + "_" + activation + ": " + std::to_string(i + from - 1);
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");

            if (mode == "add" && inputVol != shortcutVol)
                std::cout << inputVol << " +" << shortcutVol << std::endl;
        }

        else if (m_ConfigBlocks.at(i).at("type") == "route")
        {
            std::string layers;
            previous = routeLayer(i, layers, m_ConfigBlocks.at(i), tensorOutputs, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = "route: " + layers;
            printLayerInfo(layerIndex, layerName, "-", outputVol, "-");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "upsample")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = upsampleLayer(i, m_ConfigBlocks[i], previous, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = "upsample";
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "maxpool" || m_ConfigBlocks.at(i).at("type") == "avgpool")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = poolingLayer(i, m_ConfigBlocks.at(i), previous, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = m_ConfigBlocks.at(i).at("type");
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "reorg")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            if (m_NetworkType.find("yolov2") != std::string::npos) {
                nvinfer1::IPluginV2* reorgPlugin = createReorgPlugin(2);
                assert(reorgPlugin != nullptr);
                nvinfer1::IPluginV2Layer* reorg = network.addPluginV2(&previous, 1, *reorgPlugin);
                assert(reorg != nullptr);
                std::string reorglayerName = "reorg_" + std::to_string(i);
                reorg->setName(reorglayerName.c_str());
                previous = reorg->getOutput(0);
            }
            else
                previous = reorgLayer(i, m_ConfigBlocks.at(i), previous, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = "reorg";
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "reduce")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = reduceLayer(i, m_ConfigBlocks.at(i), previous, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = "reduce";
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "shuffle")
        {
            std::string layer;
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = shuffleLayer(i, layer, m_ConfigBlocks.at(i), previous, tensorOutputs, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = "shuffle: " + layer;
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "softmax")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = softmaxLayer(i, m_ConfigBlocks.at(i), previous, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            std::string layerName = "softmax";
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "yolo" || m_ConfigBlocks.at(i).at("type") == "region")
        {
            if (m_ConfigBlocks.at(i).at("type") == "yolo")
                if (m_NetworkType.find("yolor") != std::string::npos)
                    modelType = 2;
                else
                    modelType = 1;
            else
                modelType = 0;

            std::string blobName = modelType != 0 ? "yolo_" + std::to_string(i) : "region_" + std::to_string(i);
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
            TensorInfo& curYoloTensor = m_YoloTensors.at(yoloCountInputs);
            curYoloTensor.blobName = blobName;
            curYoloTensor.gridSizeX = prevTensorDims.d[2];
            curYoloTensor.gridSizeY = prevTensorDims.d[1];

            std::string inputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            yoloTensorInputs[yoloCountInputs] = previous;
            ++yoloCountInputs;
            std::string layerName = modelType != 0 ? "yolo" : "region";
            printLayerInfo(layerIndex, layerName, inputVol, "-", "-");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "cls")
        {
            modelType = 3;

            std::string blobName = "cls_" + std::to_string(i);
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
            TensorInfo& curYoloTensor = m_YoloTensors.at(yoloCountInputs);
            curYoloTensor.blobName = blobName;
            curYoloTensor.numBBoxes = prevTensorDims.d[1];
            m_NumClasses = prevTensorDims.d[0];

            std::string inputVol = dimsToString(previous->getDimensions());
            previous = clsLayer(i, m_ConfigBlocks.at(i), previous, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            yoloTensorInputs[yoloCountInputs] = previous;
            ++yoloCountInputs;
            std::string layerName = "cls";
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
        }

        else if (m_ConfigBlocks.at(i).at("type") == "reg")
        {
            modelType = 3;

            std::string blobName = "reg_" + std::to_string(i);
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
            TensorInfo& curYoloTensor = m_YoloTensors.at(yoloCountInputs);
            curYoloTensor.blobName = blobName;
            curYoloTensor.numBBoxes = prevTensorDims.d[1];

            std::string inputVol = dimsToString(previous->getDimensions());
            previous = regLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, previous, &network);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(previous);
            yoloTensorInputs[yoloCountInputs] = previous;
            ++yoloCountInputs;
            std::string layerName = "reg";
            printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
        }

        else
        {
            std::cout << "\nUnsupported layer type --> \"" << m_ConfigBlocks.at(i).at("type") << "\"" << std::endl;
            assert(0);
        }
    }

    if ((int)weights.size() != weightPtr)
    {
        std::cout << "\nNumber of unused weights left: " << weights.size() - weightPtr << std::endl;
        assert(0);
    }

    if (m_YoloCount == yoloCountInputs)
    {
        assert((modelType != -1) && "\nCould not determine model type"); 

        uint64_t outputSize = 0;
        for (uint j = 0; j < yoloCountInputs; ++j)
        {
            TensorInfo& curYoloTensor = m_YoloTensors.at(j);
            if (modelType == 3)
                outputSize = curYoloTensor.numBBoxes;
            else
                outputSize += curYoloTensor.gridSizeX * curYoloTensor.gridSizeY * curYoloTensor.numBBoxes;
        }

        nvinfer1::IPluginV2* yoloPlugin = new YoloLayer(
            m_InputW, m_InputH, m_NumClasses, m_NewCoords, m_YoloTensors, outputSize, modelType, m_ScoreThreshold);
        assert(yoloPlugin != nullptr);
        nvinfer1::IPluginV2Layer* yolo = network.addPluginV2(yoloTensorInputs, m_YoloCount, *yoloPlugin);
        assert(yolo != nullptr);
        std::string yoloLayerName = "yolo";
        yolo->setName(yoloLayerName.c_str());

        std::string outputlayerName;
        nvinfer1::ITensor* num_detections = yolo->getOutput(0);
        outputlayerName = "num_detections";
        num_detections->setName(outputlayerName.c_str());
        nvinfer1::ITensor* detection_boxes = yolo->getOutput(1);
        outputlayerName = "detection_boxes";
        detection_boxes->setName(outputlayerName.c_str());
        nvinfer1::ITensor* detection_scores = yolo->getOutput(2);
        outputlayerName = "detection_scores";
        detection_scores->setName(outputlayerName.c_str());
        nvinfer1::ITensor* detection_classes = yolo->getOutput(3);
        outputlayerName = "detection_classes";
        detection_classes->setName(outputlayerName.c_str());
        network.markOutput(*num_detections);
        network.markOutput(*detection_boxes);
        network.markOutput(*detection_scores);
        network.markOutput(*detection_classes);
    }
    else {
        std::cout << "\nError in yolo cfg file" << std::endl;
        assert(0);
    }

    std::cout << "\nOutput YOLO blob names: " << std::endl;
    for (auto& tensor : m_YoloTensors)
    {
        std::cout << tensor.blobName << std::endl;
    }

    int nbLayers = network.getNbLayers();
    std::cout << "\nTotal number of YOLO layers: " << nbLayers << "\n" << std::endl;

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
        if (line.front() == ' ') continue;
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
    for (auto block : m_ConfigBlocks)
    {
        if (block.at("type") == "net")
        {
            assert((block.find("height") != block.end()) && "Missing 'height' param in network cfg");
            assert((block.find("width") != block.end()) && "Missing 'width' param in network cfg");
            assert((block.find("channels") != block.end()) && "Missing 'channels' param in network cfg");

            m_InputH = std::stoul(block.at("height"));
            m_InputW = std::stoul(block.at("width"));
            m_InputC = std::stoul(block.at("channels"));
            m_InputSize = m_InputC * m_InputH * m_InputW;

            if (block.find("letter_box") != block.end())
            {
                m_LetterBox = std::stoul(block.at("letter_box"));
            }
        }
        else if ((block.at("type") == "region") || (block.at("type") == "yolo"))
        {
            assert((block.find("num") != block.end())
                   && std::string("Missing 'num' param in " + block.at("type") + " layer").c_str());
            assert((block.find("classes") != block.end())
                   && std::string("Missing 'classes' param in " + block.at("type") + " layer").c_str());
            assert((block.find("anchors") != block.end())
                   && std::string("Missing 'anchors' param in " + block.at("type") + " layer").c_str());

            ++m_YoloCount;

            m_NumClasses = std::stoul(block.at("classes"));

            if (block.find("new_coords") != block.end())
            {
                m_NewCoords = std::stoul(block.at("new_coords"));
            }

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

            if (block.find("mask") != block.end())
            {
                std::string maskString = block.at("mask");
                while (!maskString.empty())
                {
                    int npos = maskString.find_first_of(',');
                    if (npos != -1)
                    {
                        int mask = std::stoul(trim(maskString.substr(0, npos)));
                        outputTensor.mask.push_back(mask);
                        maskString.erase(0, npos + 1);
                    }
                    else
                    {
                        int mask = std::stoul(trim(maskString));
                        outputTensor.mask.push_back(mask);
                        break;
                    }
                }
            }

            if (block.find("scale_x_y") != block.end())
            {
                outputTensor.scaleXY = std::stof(block.at("scale_x_y"));
            }
            else
            {
                outputTensor.scaleXY = 1.0;
            }

            outputTensor.numBBoxes
                = outputTensor.mask.size() > 0 ? outputTensor.mask.size() : std::stoul(trim(block.at("num")));
            
            m_YoloTensors.push_back(outputTensor);
        }
        else if ((block.at("type") == "cls") || (block.at("type") == "reg"))
        {
            ++m_YoloCount;
            TensorInfo outputTensor;
            m_YoloTensors.push_back(outputTensor);
        }
    }
}

void Yolo::destroyNetworkUtils()
{
    for (uint i = 0; i < m_TrtWeights.size(); ++i)
        if (m_TrtWeights[i].count > 0)
            free(const_cast<void*>(m_TrtWeights[i].values));
    m_TrtWeights.clear();
}
