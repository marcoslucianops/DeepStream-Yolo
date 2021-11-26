/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "calibrator.h"
#include <fstream>
#include <iterator>

namespace nvinfer1
{
    Int8EntropyCalibrator2::Int8EntropyCalibrator2(const int &batchsize, const int &channels, const int &height, const int &width, const int &letterbox, const std::string &imgPath,
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

    Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
    {
        CUDA_CHECK(cudaFree(deviceInput));
        if (batchData)
            delete[] batchData;
    }

    int Int8EntropyCalibrator2::getBatchSize() const noexcept
    {
        return batchSize;
    }

    bool Int8EntropyCalibrator2::getBatch(void **bindings, const char **names, int nbBindings) noexcept
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

    const void* Int8EntropyCalibrator2::readCalibrationCache(std::size_t &length) noexcept
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

    void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, std::size_t length) noexcept
    {
        std::ofstream output(calibTablePath, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }
}

std::vector<float> prepareImage(cv::Mat& img, int input_c, int input_h, int input_w, int letter_box)
{
    cv::Mat out;
    int image_w = img.cols;
    int image_h = img.rows;
    if (image_w != input_w || image_h != input_h)
    {
        if (letter_box == 1)
        {
            float ratio_w = (float)image_w / (float)input_w;
            float ratio_h = (float)image_h / (float)input_h;
            if (ratio_w > ratio_h)
            {
                int new_width = input_w * ratio_h;
                int x = (image_w - new_width) / 2;
                cv::Rect roi(abs(x), 0, new_width, image_h);
                out = img(roi);
            }
            else if (ratio_w < ratio_h)
            {
                int new_height = input_h * ratio_w;
                int y = (image_h - new_height) / 2;
                cv::Rect roi(0, abs(y), image_w, new_height);
                out = img(roi);
            }
            else {
                out = img;
            }
            cv::resize(out, out, cv::Size(input_w, input_h), 0, 0, cv::INTER_CUBIC);
        }
        else
        {
            cv::resize(img, out, cv::Size(input_w, input_h), 0, 0, cv::INTER_CUBIC);
        }
        cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(img, out, cv::COLOR_BGR2RGB);
    }
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
