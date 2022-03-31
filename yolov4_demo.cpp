//
// Created by linhan on 2022/3/21.
//
#include "yolov4.h"

void showResult(const std::vector<detectResult>& result, std::vector<cv::Mat> &imgCloneBatch)
{
    std::vector<cv::Scalar> colors;
    colors.emplace_back(255, 0, 0);
    colors.emplace_back(0, 255, 0);
    colors.emplace_back(0, 0, 255);
    colors.emplace_back(255, 255, 0);
    colors.emplace_back(255, 0, 255);
    colors.emplace_back(255, 255, 255);
    colors.emplace_back(0, 0, 0);
    for (int i = 0; i < result.size(); i++)
    {
        detectResult batchResult = result[i];
        for (const auto& r: batchResult)
        {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
            cv::rectangle(imgCloneBatch[i], r.rect, colors[r.id], 2);
            cv::putText(imgCloneBatch[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
        }
        cv::namedWindow("Windows", cv::WINDOW_NORMAL);
        cv::resizeWindow("Windows", imgCloneBatch[i].cols / 2, imgCloneBatch[i].rows / 2);
        cv::imshow("Windows", imgCloneBatch[i]);
        cv::waitKey(0);
    }
}

int main()
{
    std::string configPath = "../configs";
    std::vector<cv::String> images;
    cv::String path("../data/*.jpg");
    cv::glob(path, images);
    std::shared_ptr<YoloV4> yoloObj = std::make_shared<YoloV4>(configPath);
    for (auto image: images)
    {
        std::vector<cv::Mat> batch_img;
        cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
        batch_img.push_back(img);
        double start = cv::getTickCount();
        std::vector<detectResult> result = yoloObj->detect(batch_img);
        double end = cv::getTickCount();
        double fps =  1 / ((end - start) / cv::getTickFrequency());
        std::cout << "fps:" << fps << std::endl;
        showResult(result, batch_img);
    }
    return 0;
}