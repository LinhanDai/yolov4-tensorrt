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

void YoloV4::imgPreProcess(std::vector<cv::Mat> &batchImg) const
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

void YoloV4::getMaxConfsData(const float *confProb, int batch,
                             std::vector<std::vector<float>> &maxConfVec,  std::vector<std::vector<int>> &maxConfIndexVec) const
{
    for (int i = 0; i < batch; i ++)
    {
        std::vector<float> confVec;
        std::vector<int> indexVec;
        for (int j = 0; j < mOutputConfsN; j++)
        {
            float maxConf = 0;
            int maxConfIndex = 0;
            for (int k = 0; k < mOutputConfsDim; k++)
            {
                if (confProb[i * mOutputConfsSize + j * mOutputConfsDim + k] > maxConf)
                {
                    maxConf = confProb[i * mOutputConfsSize + j * mOutputConfsDim + k];
                    maxConfIndex = k;
                }
            }
            confVec.push_back(maxConf);
            indexVec.push_back(maxConfIndex);
        }
        maxConfVec.push_back(confVec);
        maxConfIndexVec.push_back(indexVec);
    }
}

void YoloV4::thresholdFilter(const float *boxesProb, std::vector<std::vector<float>> &maxConfVec, std::vector<std::vector<int>> &maxConfIndexVec,
                             std::vector<std::vector<float>> &confFilterVec, std::vector<std::vector<int>> &confIdFilterVec,
                             std::vector<std::vector<ObjPos>> &boxFilterVec,float confThreshold) const
{
    int batch = maxConfVec.size();
    for (int i = 0; i < batch; i++)
    {
        std::vector<ObjPos> pos;
        std::vector<float> conf;
        std::vector<int> id;
        std::vector<float> confVec = maxConfVec[i];
        std::vector<int> confIndexVec = maxConfIndexVec[i];
        for (int j= 0; j < confVec.size(); ++j)
        {
            if (confVec[j] > confThreshold)
            {
                ObjPos boxPos{};
                conf.push_back(confVec[j]);
                id.push_back(confIndexVec[j]);
                boxPos.x1 = boxesProb[i * mOutputBoxesSize + j * mOutputBoxesDim];
                boxPos.y1 = boxesProb[i * mOutputBoxesSize + j * mOutputBoxesDim + 1];
                boxPos.x2 = boxesProb[i * mOutputBoxesSize + j * mOutputBoxesDim + 2];
                boxPos.y2 = boxesProb[i * mOutputBoxesSize + j * mOutputBoxesDim + 3];
                pos.push_back(boxPos);
            }
        }
        confFilterVec.push_back(conf);
        confIdFilterVec.push_back(id);
        boxFilterVec.push_back(pos);
    }
}

void YoloV4::modifyBoundaryValue(int &x1, int &y1, int &x2, int &y2, int imgWidth, int imgHeight)
{
    if (x1 < 0)
    {
        x1 = 0;
    }
    else if (x1 > imgWidth)
    {
        x1 = imgWidth;
    }
    if (x2 < 0)
    {
        x2 = 0;
    }
    else if (x2 > imgWidth)
    {
        x2 = imgWidth;
    }
    if (y1 < 0)
    {
        y1 = 0;
    }
    else if (y1 > imgHeight)
    {
        y1 = imgHeight;
    }
    if (y2 < 0)
    {
        y2 = 0;
    }
    else if (y2 > imgHeight)
    {
        y2 = imgHeight;
    }
}

bool YoloV4::checkDetectRect(int &x1, int &y1, int &x2, int &y2, int imgWidth, int imgHeight)
{
    modifyBoundaryValue(x1, y1, x2, y2, imgWidth, imgHeight);
    if ((x2 - x1 > 0) && (y2 - y1 > 0))
        return true;
    else
        return false;
}

std::vector<detectResult> YoloV4::getDetResult(std::vector<std::vector<float>> &confFilterVec,
                                       std::vector<std::vector<int>> &confIdFilterVec,
                                       std::vector<std::vector<ObjPos>> &boxFilterVec,
                                       std::vector<std::vector<int>> keepVec)
{
    std::vector<detectResult> result;
    int batch = boxFilterVec.size();
    for (int i = 0; i < batch; i++)
    {
        detectResult det;
        for (int j = 0; j < keepVec[i].size(); ++j)
        {
            ObjStu obj{};
            int x1 = boxFilterVec[i][keepVec[i][j]].x1 * mImageSizeBatch[i].width;
            int y1 = boxFilterVec[i][keepVec[i][j]].y1 * mImageSizeBatch[i].height;
            int x2 = boxFilterVec[i][keepVec[i][j]].x2 * mImageSizeBatch[i].width;
            int y2 = boxFilterVec[i][keepVec[i][j]].y2 * mImageSizeBatch[i].height;
            if(checkDetectRect(x1, y1, x2, y2, mImageSizeBatch[i].width, mImageSizeBatch[i].height))
            {
                obj.rect.x = boxFilterVec[i][keepVec[i][j]].x1 * mImageSizeBatch[i].width;
                obj.rect.y = boxFilterVec[i][keepVec[i][j]].y1 * mImageSizeBatch[i].height;
                obj.rect.width = (boxFilterVec[i][keepVec[i][j]].x2 - boxFilterVec[i][keepVec[i][j]].x1) * mImageSizeBatch[i].width;
                obj.rect.height = (boxFilterVec[i][keepVec[i][j]].y2 - boxFilterVec[i][keepVec[i][j]].y1) * mImageSizeBatch[i].height;
                obj.prob = confFilterVec[i][keepVec[i][j]];
                obj.id = confIdFilterVec[i][keepVec[i][j]];
                det.push_back(obj);
            }
        }
        result.push_back(det);
    }
    return result;
}

// Computes IOU between two bounding boxes
float YoloV4::getIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON)
        return 0;

    return in / un;
}


void YoloV4::bubbleSort(std::vector<float> confs, int length, std::vector<int> &indDiff)
{
    for (int m = 0; m < length; m++)
    {
        indDiff[m] = m;
    }
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length - i - 1; j++)
        {
            if (confs[j] < confs[j + 1])
            {
                float temp = confs[j];
                confs[j] = confs[j + 1];
                confs[j + 1] = temp;
                int ind_temp = indDiff[j];
                indDiff[j] = indDiff[j + 1];
                indDiff[j + 1] = ind_temp;
            }
        }
    }
}

std::vector<std::vector<int>> YoloV4::allClassNMS(std::vector<std::vector<float>> &confFilterVec,
                                                  std::vector<std::vector<int>> &confIdFilterVec,
                                                  std::vector<std::vector<ObjPos>> &boxFilterVec,
                                                  float nmsThreshold)
{
    int batch = boxFilterVec.size();
    std::vector<std::vector<int>> keepVec;
    for (int i = 0; i < batch; i ++)
    {
        std::vector<int> keep;
        std::vector<float> confs = confFilterVec[i];
        std::vector<int> ids = confIdFilterVec[i];
        std::vector<ObjPos> boxes = boxFilterVec[i];
        int targetNum = boxes.size();
        std::vector<int> indDiff(targetNum, 0);
        bubbleSort(confs, targetNum, indDiff);
        while (!indDiff.empty())
        {
            int idxSelf = indDiff[0];
            keep.push_back(idxSelf);
            std::vector<float> iouVec;
            for (int j = 1; j < indDiff.size(); j++)
            {
                float iou = getIOU(cv::Rect_<float>(boxes[idxSelf].x1, boxes[idxSelf].y1,
                                            boxes[idxSelf].x2 - boxes[idxSelf].x1,
                                            boxes[idxSelf].y2 - boxes[idxSelf].y1),
                                   cv::Rect_<float>(boxes[indDiff[j]].x1, boxes[indDiff[j]].y1,
                                            boxes[indDiff[j]].x2 - boxes[indDiff[j]].x1,
                                            boxes[indDiff[j]].y2 - boxes[indDiff[j]].y1));
                iouVec.push_back(iou);
            }
            std::vector<int> newIndex;
            for (int j = 0; j < iouVec.size(); j++)
            {
                if (iouVec[j] < nmsThreshold)
                {
                    newIndex.push_back(indDiff[1 + j]);
                }
            }
            indDiff = newIndex;
        }
        keepVec.push_back(keep);
    }
    return keepVec;
}

std::vector<detectResult> YoloV4::postProcessing(float *boxesProb, float *confProb, int batch)
{
    std::vector<std::vector<int>> keepVec;
    std::vector<std::vector<float>> maxConfVec;
    std::vector<std::vector<int>> maxConfIndexVec;
    std::vector<std::vector<float>> confFilterVec;
    std::vector<std::vector<int>> confIdFilterVec;
    std::vector<std::vector<ObjPos>> boxFilterVec;
    getMaxConfsData(confProb, batch, maxConfVec, maxConfIndexVec);
    thresholdFilter(boxesProb, maxConfVec, maxConfIndexVec, confFilterVec, confIdFilterVec, boxFilterVec, mConfTreshold);
    if (mAllClasssNMS)
    {
        keepVec = allClassNMS(confFilterVec, confIdFilterVec, boxFilterVec, mNMSTreshold);
    }
    std::vector<detectResult> result = getDetResult(confFilterVec, confIdFilterVec, boxFilterVec, keepVec);
    return result;
}

void YoloV4::initInputImageSize(std::vector<cv::Mat> &batchImg)
{
    int batch = batchImg.size();
    for (int i = 0; i < batch; i++)
    {
        ImgInfo info{};
        info.width = batchImg[i].cols;
        info.height = batchImg[i].rows;
        mImageSizeBatch.push_back(info);
    }
}

std::vector<detectResult> YoloV4::detect(std::vector<cv::Mat> &batchImg)
{
    std::vector<cv::Mat> detectMatVec;
    for (auto & img : batchImg)
    {
        detectMatVec.push_back(img);
    }
    int batch = detectMatVec.size();
    initInputImageSize(detectMatVec);
    imgPreProcess(detectMatVec);
    int inputSingleByteNum = mInputH * mInputW * mInputC;
    for (size_t i = 0; i < batch; i++)
    {
        cudaMemcpyAsync(mInputData + i * inputSingleByteNum, detectMatVec[i].data, inputSingleByteNum,
                        cudaMemcpyHostToDevice, mStream);
    }
    cudaPreProcess(mBuff[0], mInputData, mInputW, mInputH, mInputC, batch, mStream);
    float *boxesProb = (float *) malloc(batch * mOutputBoxesSize * sizeof(float));
    float *confProb = (float *) malloc(batch * mOutputConfsSize * sizeof(float));
    memset(boxesProb, 0, batch * mOutputBoxesSize * sizeof(float));
    memset(confProb, 0, batch * mOutputConfsSize * sizeof(float));
    doInference(*mContext, boxesProb, confProb, batch);
    std::vector<detectResult> detectResult = postProcessing(boxesProb, confProb, batch);
    free(boxesProb);
    free(confProb);
    mImageSizeBatch.clear();
    return detectResult;
}

void YoloV4::readParameters(const std::string& configPath)
{
    std::string yamlFile = configPath + "/" + "yolo.yaml";
    cv::FileStorage fs(yamlFile, cv::FileStorage::READ);
    mConfTreshold = fs["confTreshold"];
    mNMSTreshold = fs["nmsTreshold"];
    fs["allClassNMS"] >> mAllClasssNMS;
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