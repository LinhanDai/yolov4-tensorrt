//
// Created by linhan on 2022/3/21.
//
#include "yolov4.h"

void showResult(const std::vector<detectResult>& result, cv::Mat img)
{
    std::vector<cv::Scalar> colors;
    colors.emplace_back(255, 0, 0);
    colors.emplace_back(0, 255, 0);
    colors.emplace_back(0, 0, 255);
    colors.emplace_back(255, 255, 0);
    colors.emplace_back(255, 0, 255);
    colors.emplace_back(255, 255, 255);
    colors.emplace_back(0, 0, 0);
    for (const auto& batchResult: result)
    {
        for (const auto& r: batchResult)
        {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
            cv::rectangle(img, r.rect, colors[r.id], 1);
            cv::putText(img, stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
        }
    }
    cv::namedWindow("Windows", cv::WINDOW_NORMAL);
    cv::resizeWindow("Windows", img.cols / 2, img.rows / 2);
    cv::imshow("Windows", img);
    cv::waitKey(0);
}

int main()
{
    std::vector<cv::Mat> imgBatch;
    cv::Mat imgClone;
    std::string configPath = "../configs";
    cv::Mat img = cv::imread("../data/2.jpg", cv::IMREAD_COLOR);
    imgClone = img.clone();
    imgBatch.push_back(img);
    std::shared_ptr<YoloV4> yoloObj = std::make_shared<YoloV4>(configPath);
//        double start = cv::getTickCount();
    std::vector<detectResult> result = yoloObj->detect(imgBatch);
//        double end = cv::getTickCount();
//        std::cout << "fps:" << 1 / ((end - start) / cv::getTickFrequency()) << std::endl;

    showResult(result, imgClone);
    return 0;
}