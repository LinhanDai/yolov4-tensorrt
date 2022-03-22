//
// Created by linhan on 2022/3/21.
//
#include "yolov4.h"

#include <utility>

extern "C" void cudaPreProcess(float* img_dst, unsigned char* img_source, int width, int height, int channel, int num, cudaStream_t stream);

YoloV4::YoloV4(const std::string& configPath)
{
    readParameters(configPath);
    bool flag = createEngineIfNotExit();
    assert(flag == true && "engine create failure!");
    getTrtmodelStream();
}

YoloV4::~YoloV4()
{
    try
    {
        cudaStreamDestroy(mStream);
        cudaFree(mInputData);
        cudaFree(mBuff[0]);
        cudaFree(mBuff[1]);
        cudaFree(mBuff[2]);
        mContext->destroy();
        mEngine->destroy();
    }
    catch (std::exception &e)
    {
        mGlogger.log(Severity::kERROR, "~YoloV4() error!");
    }
}

void YoloV4::imgPreProcess(std::vector<cv::Mat> &batchImg)
{
    for (int i = 0; i < batchImg.size(); i++)
    {
        cv::Mat &img = batchImg[i];
        cv::resize(img, img, cv::Size(mInputW, mInputH), cv::INTER_LINEAR);
    }
}

void YoloV4::doInference(nvinfer1::IExecutionContext& context, float* boxesProb, float* confProb, int batchSize)
{
    const nvinfer1::ICudaEngine &engine = context.getEngine();
    assert(engine.getNbBindings() == 3), "yolov4 Bindings Dim should be four!";
    nvinfer1::Dims inputDims = engine.getBindingDimensions(0);
    nvinfer1::Dims d = inputDims;
    d.d[0] = batchSize;
    if (!mContext->setBindingDimensions(0, d))
    {
        mGlogger.log(Severity::kERROR, "模型输入维度不正确");
        std::abort();
    }
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    context.enqueueV2((void **)mBuff, mStream, nullptr);
    CHECK(cudaMemcpyAsync(boxesProb, mBuff[1], batchSize * mOutputBoxesSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    CHECK(cudaMemcpyAsync(confProb, mBuff[2], batchSize * mOutputConfsSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);
}

std::vector<detectResult> YoloV4::postProcessing(float *boxesProb, float *confProb)
{

}

std::vector<detectResult> YoloV4::detect(std::vector<cv::Mat> &batchImg)
{
    int batch = batchImg.size();
    imgPreProcess(batchImg);
    int inputSingleByteNum = mInputH * mInputW * mInputC;
    for (size_t i = 0; i < batch; i++)
    {
        cudaMemcpyAsync(mInputData + i * inputSingleByteNum, batchImg[i].data, inputSingleByteNum,
                        cudaMemcpyHostToDevice, mStream);
    }

    cudaPreProcess(mBuff[0], mInputData, mInputW, mInputH, mInputC, batch, mStream);
    float *boxesProb = (float *) malloc(batch * mOutputBoxesSize * sizeof(float));
    float *confProb = (float *) malloc(batch * mOutputConfsSize * sizeof(float));
    doInference(*mContext, boxesProb, confProb, batch);
    for (int i = 0; i < mOutputConfsSize; i++)
    {
        if (confProb[i] > 0.004)
        {
            std::cout << confProb[i] << " ";
        }
    }
    std::vector<detectResult> detectResult = postProcessing(boxesProb, confProb);
    free(boxesProb);
    free(confProb);
    return detectResult;
}

void YoloV4::readParameters(const std::string& configPath)
{
    std::string yamlFile = configPath + "/" + "yolo.yaml";
    cv::FileStorage fs(yamlFile, cv::FileStorage::READ);
    mMaxSupportBatchSize = fs["maxSupportBatchSize"];
    mQuantizationInfer = (std::string) fs["quantizationInfer"];
    mOnnxFile = configPath + "/"  + (std::string) fs["onnxFile"];
    mEngineFile = configPath + "/" + (std::string) fs["engineFile"];
}


nvinfer1::ICudaEngine *YoloV4::createEngine(nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config)
{
    mGlogger.setReportableSeverity(Severity::kERROR);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    assert(network);
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, mGlogger);
    assert(parser);
    bool parsed = parser->parseFromFile(mOnnxFile.c_str(), (int) nvinfer1::ILogger::Severity::kWARNING);
    if (!parsed) {
        mGlogger.log(Severity::kERROR, "onnx file parse error, please check onnx file!");
        std::abort();
    }
    // Build engine
    builder->setMaxBatchSize(mMaxSupportBatchSize);
    config->setMaxWorkspaceSize(1_GiB);
    if (strcmp(mQuantizationInfer.c_str(), "FP16") == 0)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();

    if (inputDims.d[0] == -1)
    {
        nvinfer1::IOptimizationProfile *profileCalib = builder->createOptimizationProfile();
        const auto inputName = "input";
        nvinfer1::Dims batchDim = inputDims;
        batchDim.d[0] = 1;
        // We do not need to check the return of setDimension and setCalibrationProfile here as all dims are explicitly set
        profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, batchDim);
        profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, batchDim);
        batchDim.d[0] = mMaxSupportBatchSize;
        profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, batchDim);
        config->addOptimizationProfile(profileCalib);
    }
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);
    mGlogger.log(Severity::kINFO, "success create engine!");
    //release network
    network->destroy();
    return engine;
}

void YoloV4::getBindingDimsInfo()
{
    int nb = mEngine->getNbBindings();
    assert(nb == 3), "binding total dim should be three!";
    nvinfer1::Dims inputDims = mEngine->getBindingDimensions(0);
    nvinfer1::Dims dInput = inputDims;
    mInputC = dInput.d[1];
    mInputH = dInput.d[2];
    mInputW = dInput.d[3];
    nvinfer1::Dims outPutBoxesDims = mEngine->getBindingDimensions(1);
    nvinfer1::Dims dOutPutBoxes = outPutBoxesDims;
    mOutputBoxesN = dOutPutBoxes.d[1];
    mOutputBoxesL = dOutPutBoxes.d[2];
    mOutputBoxesDim = dOutPutBoxes.d[3];
    mOutputBoxesSize = mOutputBoxesN * mOutputBoxesL * mOutputBoxesDim;
    nvinfer1::Dims outPutConfsDims = mEngine->getBindingDimensions(2);
    nvinfer1::Dims dOutPutConfs = outPutConfsDims;
    mOutputConfsN = dOutPutConfs.d[1];
    mOutputConfsDim = dOutPutConfs.d[2];
    mOutputConfsSize = mOutputConfsN * mOutputConfsDim;
}

void YoloV4::getTrtmodelStream()
{
    std::ifstream file(mEngineFile, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        mEngineFileSize = file.tellg();
        file.seekg(0, file.beg);
        mTrtModelStream = new char[mEngineFileSize];
        assert(mTrtModelStream);
        file.read(mTrtModelStream, mEngineFileSize);
        file.close();
    }
    mRuntime = nvinfer1::createInferRuntime(mGlogger);
    assert(mRuntime);
    mEngine = mRuntime->deserializeCudaEngine(mTrtModelStream, mEngineFileSize, nullptr);
    assert(mEngine);
    mContext = mEngine->createExecutionContext();
    assert(mContext);
    //create stream
    CHECK(cudaStreamCreate(&mStream));
    getBindingDimsInfo();
    //create fixed maximum input buffer
    int inputSingleByteNum = mInputW * mInputH * mInputC;
    int outputSingleBoxByteNum = mOutputBoxesN * mOutputBoxesL * mOutputBoxesDim;
    int outputSingleConfByteNum = mOutputConfsN * mOutputConfsDim;
    CHECK(cudaMalloc(&(mInputData), mMaxSupportBatchSize * inputSingleByteNum));
    CHECK(cudaMalloc(&(mBuff[0]), mMaxSupportBatchSize * inputSingleByteNum * sizeof(float)));
    CHECK(cudaMalloc(&(mBuff[1]), mMaxSupportBatchSize * outputSingleBoxByteNum * sizeof(float)));
    CHECK(cudaMalloc(&(mBuff[2]), mMaxSupportBatchSize * outputSingleConfByteNum * sizeof(float)));
    delete mTrtModelStream;
}

void YoloV4::createInferenceEngine(nvinfer1::IHostMemory **modelStream)
{
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(mGlogger);
    assert(builder);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    assert(config);
    nvinfer1::ICudaEngine *engine = createEngine(builder, config);
    assert(engine != nullptr && "engine create failure!");

    // Serialize the engine
    (*modelStream) = engine->serialize();

    //release all memory
    builder->destroy();
    config->destroy();
    engine->destroy();
}

bool YoloV4::createEngineIfNotExit()
{
    std::ifstream cache(mEngineFile.c_str(), std::ios::binary);
    if (cache)
        return true;
    else
    {
        nvinfer1::IHostMemory* modelStream{nullptr};
        createInferenceEngine(&modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(mEngineFile.c_str(), std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return false;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
    }
    return true;
}