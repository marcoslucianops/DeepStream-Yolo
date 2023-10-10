/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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
 * Edited by Marcos Luciano, Youngjae You
 * https://www.github.com/marcoslucianops
 */

/*==========================================================================*/
// include header
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "nvdsinfer_custom_impl.h"

static bool dict_ready=false;
std::vector<std::string> dict_table;

int argmax(const std::vector<double>& softmax_values) 
{
    return std::distance(softmax_values.begin(), std::max_element(softmax_values.begin(), softmax_values.end()));
}

/* C-linkage to prevent name-mangling */
extern "C" 
bool NvDsInferCustomYolov5ClsParse(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &descString);

extern "C" 
bool NvDsInferCustomYolov5ClsParse(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &descString)
{
    /* Get the number of attributes supported by the classifier. */
    unsigned int numAttributes = outputLayersInfo.size();

    std::ifstream file;

    if(!dict_ready) {
        file.open("imagenet_classes.txt");

        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos) {
                    std::string value = line.substr(colon_pos + 2);  // +2 to skip ": "
                    value = value.substr(0, value.size() - 1);  // remove the trailing comma
                    dict_table.push_back(value);
                }
            }
            dict_ready=true;
            file.close();
        } else {
            std::cerr << "Unable to open file\n";
            return 1;
        }

    }

    /* Iterate through all the output coverage layers of the classifier.
    */
    for (unsigned int l = 0; l < numAttributes; l++)
    {
        /* outputCoverageBuffer for classifiers is usually a softmax layer.
         * The layer is an array of probabilities of the object belonging
         * to each class with each probability being in the range [0,1] and
         * sum all probabilities will be 1.
         */
        NvDsInferDimsCHW dims;

        getDimsCHWFromDims(dims, outputLayersInfo[l].inferDims);
        unsigned int numClasses = dims.c;
        float *outputCoverageBuffer = (float *)outputLayersInfo[l].buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsInferAttribute attr;
        
        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */

        // Verify that the sum is 1 for softmax output
        float prob_sum = 0;
        for (unsigned int c = 0; c < numClasses; c++)
        {
            float probability = outputCoverageBuffer[c];
            
            if (probability > classifierThreshold
                    && probability > maxProbability)
            {
                maxProbability = probability;
                attrFound = true;
                attr.attributeIndex = l;
                attr.attributeValue = c;
                attr.attributeConfidence = probability;
            }

            prob_sum += probability;
        }

        // if (abs(prob_sum - 1.0f) > 0.001)
        //     throw std::runtime_error("softmax output is not valid");

        if (attrFound)
        {
            if (dict_table.size() > attr.attributeIndex &&
                    attr.attributeValue < dict_table.size())
                attr.attributeLabel =
                    strdup(dict_table[attr.attributeValue].c_str());
            else
                attr.attributeLabel = nullptr;
            attrList.push_back(attr);
            if (attr.attributeLabel)
                descString.append(attr.attributeLabel).append(" ");
        }
        else {
            NvDsInferAttribute attr;
            attr.attributeIndex = -1;
            attr.attributeValue = 1; 

            attr.attributeLabel = strdup("Unknown");
            attrList.push_back(attr);
            descString.append(attr.attributeLabel).append(" ");
        }
    }

    return true;
}

// /* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferCustomYolov5ClsParse);


