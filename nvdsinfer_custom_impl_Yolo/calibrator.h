/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "NvInfer.h"
#include <vector>
#include <string>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#endif

namespace nvinfer1 {
    class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
    public:
        Int8EntropyCalibrator2(const int &batchsize,
                             const int &channels,
                             const int &height,
                             const int &width,
                             const int &letterbox,
                             const std::string &imgPath,
                             const std::string &calibTablePath);

        virtual ~Int8EntropyCalibrator2();
        int getBatchSize() const noexcept override;
        bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
        const void* readCalibrationCache(std::size_t& length) noexcept override;
        void writeCalibrationCache(const void* cache, size_t length) noexcept override;

    private:
        int batchSize;
        int inputC;
        int inputH;
        int inputW;
        int letterBox;
        std::string calibTablePath;
        size_t imageIndex;
        size_t inputCount;
        std::vector<std::string> imgPaths;
        float *batchData{ nullptr };
        void  *deviceInput{ nullptr };
        bool readCache;
        std::vector<char> calibrationCache;
    };
}

std::vector<float> prepareImage(cv::Mat& img, int input_c, int input_h, int input_w, int letter_box);

#endif //CALIBRATOR_H
