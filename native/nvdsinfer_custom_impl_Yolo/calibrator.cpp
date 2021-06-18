/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "calibrator.h"
#include <fstream>
#include <iterator>

namespace nvinfer1
{
    int8EntroyCalibrator::int8EntroyCalibrator(const int &batchsize, const int &channels, const int &height, const int &width, const int &letterbox, const std::string &imgPath,
        const std::string &calibTablePath):batchSize(batchsize), inputC(channels), inputH(height), inputW(width), letterBox(letterbox), calibTablePath(calibTablePath), imageIndex(0)
    {
        inputCount = batchsize * channels * height * width;
        std::fstream f(imgPath);
        if (f.is_open())
        {
            std::string temp;
            while (std::getline(f, temp)) imgPaths.push_back(temp);
        }
        batchData = new float[inputCount];
        CUDA_CHECK(cudaMalloc(&deviceInput, inputCount * sizeof(float)));
    }

    int8EntroyCalibrator::~int8EntroyCalibrator()
    {
        CUDA_CHECK(cudaFree(deviceInput));
        if (batchData)
            delete[] batchData;
    }

    bool int8EntroyCalibrator::getBatch(void **bindings, const char **names, int nbBindings)
    {
        if (imageIndex + batchSize > uint(imgPaths.size()))
            return false;

        float* ptr = batchData;
        for (size_t j = imageIndex; j < imageIndex + batchSize; ++j)
        {
            cv::Mat img = cv::imread(imgPaths[j], cv::IMREAD_COLOR);
            std::vector<float>inputData = prepareImage(img, inputC, inputH, inputW, letterBox);

            int len = (int)(inputData.size());
            memcpy(ptr, inputData.data(), len * sizeof(float));

            ptr += inputData.size();
            std::cout << "Load image: " << imgPaths[j] << std::endl;
            std::cout << "Progress: " << (j + 1)*100. / imgPaths.size() << "%" << std::endl;
        }
        imageIndex += batchSize;
        CUDA_CHECK(cudaMemcpy(deviceInput, batchData, inputCount * sizeof(float), cudaMemcpyHostToDevice));
        bindings[0] = deviceInput;
        return true;
    }

    const void* int8EntroyCalibrator::readCalibrationCache(std::size_t &length)
    {
        calibrationCache.clear();
        std::ifstream input(calibTablePath, std::ios::binary);
        input >> std::noskipws;
        if (readCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(calibrationCache));
        }
        length = calibrationCache.size();
        return length ? calibrationCache.data() : nullptr;
    }

    void int8EntroyCalibrator::writeCalibrationCache(const void *cache, std::size_t length)
    {
        std::ofstream output(calibTablePath, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }
}

std::vector<float> prepareImage(cv::Mat& img, int input_c, int input_h, int input_w, int letter_box)
{
    cv::Mat out;
    if (letter_box == 2)
    {
        int image_w = img.cols;
        int image_h = img.rows;
        int resize_w = 0;
        int resize_h = 0;
        int offset_top = 0;
        int offset_bottom = 0;
        int offset_left = 0;
        int offset_right = 0;
        if ((float)input_h / image_h > (float)input_w / image_w)
        {
            resize_w = input_w;
            resize_h = (input_w * image_h) / image_w;
            offset_bottom = input_h - resize_h;
        }
        else
        {
            resize_h = input_h;
            resize_w = (input_h * image_w) / image_h;
            offset_right = input_w - resize_w;
        }
        cv::resize(img, out, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_CUBIC);
        cv::copyMakeBorder(out, out, offset_top, offset_bottom, offset_left, offset_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
    else
    {
        cv::resize(img, out, cv::Size(input_w, input_h), 0, 0, cv::INTER_CUBIC);
    }
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    if (input_c == 3)
    {
        out.convertTo(out, CV_32FC3, 1.0 / 255.0);
    }
    else
    {
        out.convertTo(out, CV_32FC1, 1.0 / 255.0);
    }
    std::vector<cv::Mat> input_channels(input_c);
    cv::split(out, input_channels);
    std::vector<float> result(input_h * input_w * input_c);
    auto data = result.data();
    int channelLength = input_h * input_w;
    for (int i = 0; i < input_c; ++i)
    {
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }
    return result;
}
