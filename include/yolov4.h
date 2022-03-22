//
// Created by linhan on 2022/3/21.
//

#ifndef YOLO_TRT_YOLOV4_H
#define YOLO_TRT_YOLOV4_H

#include <opencv2/opencv.hpp>
#include "loggingRT.h"
#include <map>
#include <iostream>
#include <cassert>
#include <fstream>
#include <NvInfer.h>
#include <dirent.h>
#include <NvOnnxParser.h>
#include <memory>
#include <mutex>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <cuda_runtime.h>
#include "loggingRT.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}

struct ObjStu
{
    int		 id		= -1;
    float	 prob	= 0.f;
    cv::Rect rect;

};

typedef std::vector<ObjStu> detectResult;

class YoloV4
{
public:
    explicit YoloV4(const std::string& configPath);
    ~YoloV4();
    std::vector<detectResult> detect(std::vector<cv::Mat> &batchImg);

private:
    void imgPreProcess(std::vector<cv::Mat> &batchImg);
    void getTrtmodelStream();
    void readParameters(const std::string& configPath);
    nvinfer1::ICudaEngine* createEngine(nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config);
    bool createEngineIfNotExit();
    void createInferenceEngine(nvinfer1::IHostMemory **modelStream);
    void getBindingDimsInfo();
    std::vector<detectResult> postProcessing(float *boxesProb, float *confProb);
    void doInference(nvinfer1::IExecutionContext& context, float* boxesProb, float* confProb, int batchSize);

private:
    int mMaxSupportBatchSize;
    size_t mEngineFileSize;
    std::string mQuantizationInfer;
    std::string mOnnxFile;
    std::string mEngineFile;
    Logger mGlogger;
    char *mTrtModelStream;
    int mInputH;
    int mInputW;
    int mInputC;
    int mOutputBoxesN;
    int mOutputBoxesL;
    int mOutputBoxesDim;
    int mOutputBoxesSize;
    int mOutputConfsN;
    int mOutputConfsDim;
    int mOutputConfsSize;
    unsigned char *mInputData;
    float *mBuff[3];
    nvinfer1::IRuntime *mRuntime;
    nvinfer1::ICudaEngine *mEngine;
    nvinfer1::IExecutionContext *mContext;
    cudaStream_t  mStream;
};

#endif //YOLO_TRT_YOLOV4_H
