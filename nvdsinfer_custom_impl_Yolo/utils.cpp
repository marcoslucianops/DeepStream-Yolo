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

#include "utils.h"

#include <experimental/filesystem>
#include <iomanip>
#include <algorithm>
#include <math.h>

static void leftTrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s)
{
    leftTrim(s);
    rightTrim(s);
    return s;
}

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

bool fileExists(const std::string fileName, bool verbose)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "\nFile does not exist: " << fileName << std::endl;
        return false;
    }
    return true;
}

std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType)
{
    assert(fileExists(weightsFilePath));
    std::cout << "\nLoading pre-trained weights" << std::endl;

    std::vector<float> weights;

    if (weightsFilePath.find(".weights") != std::string::npos) {
        std::ifstream file(weightsFilePath, std::ios_base::binary);
        assert(file.good());
        std::string line;

        if (networkType.find("yolov2") != std::string::npos && networkType.find("yolov2-tiny") == std::string::npos)
        {
            // Remove 4 int32 bytes of data from the stream belonging to the header
            file.ignore(4 * 4);
        }
        else
        {
            // Remove 5 int32 bytes of data from the stream belonging to the header
            file.ignore(4 * 5);
        }

        char floatWeight[4];
        while (!file.eof())
        {
            file.read(floatWeight, 4);
            assert(file.gcount() == 4);
            weights.push_back(*reinterpret_cast<float*>(floatWeight));
            if (file.peek() == std::istream::traits_type::eof()) break;
        }
    }

    else if (weightsFilePath.find(".wts") != std::string::npos) {
        std::ifstream file(weightsFilePath);
        assert(file.good());
        int32_t count;
        file >> count;
        assert(count > 0 && "\nInvalid .wts file.");

        uint32_t floatWeight;
        std::string name;
        uint32_t size;

        while (count--) {
            file >> name >> std::dec >> size;
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                file >> std::hex >> floatWeight;
                weights.push_back(*reinterpret_cast<float *>(&floatWeight));
            };
        }
    }

    else {
        std::cerr << "\nFile " << weightsFilePath << " is not supported" << std::endl;
        std::abort();
    }

    std::cout << "Loading weights of " << networkType << " complete"
            << std::endl;
    std::cout << "Total weights read: " << weights.size() << std::endl;
    return weights;
}

std::string dimsToString(const nvinfer1::Dims d)
{
    std::stringstream s;
    assert(d.nbDims >= 1);
    s << "[";
    for (int i = 0; i < d.nbDims - 1; ++i)
        s << d.d[i] << ", ";
    s << d.d[d.nbDims - 1] << "]";

    return s.str();
}

int getNumChannels(nvinfer1::ITensor* t)
{
    nvinfer1::Dims d = t->getDimensions();
    assert(d.nbDims == 3);

    return d.d[0];
}

void printLayerInfo(
    std::string layerIndex, std::string layerName, std::string layerInput, std::string layerOutput, std::string weightPtr)
{
    std::cout << std::setw(8) << std::left << layerIndex << std::setw(30) << std::left << layerName;
    std::cout << std::setw(20) << std::left << layerInput << std::setw(20) << std::left << layerOutput;
    std::cout << weightPtr << std::endl;
}
