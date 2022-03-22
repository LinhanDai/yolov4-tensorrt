//
// Created by linhan on 2022/3/21.
//
#include "yolov4.h"

int main()
{
    std::vector<cv::Mat> imgBatch;
    std::string configPath = "../configs";
    cv::Mat img = cv::imread("../data/2.jpg", cv::IMREAD_COLOR);
    imgBatch.push_back(img);
    std::shared_ptr<YoloV4> yoloObj = std::make_shared<YoloV4>(configPath);
    yoloObj->detect(imgBatch);
    return 0;
}