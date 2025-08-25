#include "P2Fconverter.h"  // 必须包含对应的头文件
#include <opencv2/opencv.hpp>
#include <cmath>

// 只需实现成员函数，不要重复定义类

void IdealPinhole2Fisheye::init(const cv::Size& pinhole_size, const cv::Size& fish_size, float fish_fov_deg,
                               float fx, float fy, float cx, float cy,
                               const cv::Mat& R) {
    fish_eye_size = fish_size;
    fish_fov_rad = fish_fov_deg * CV_PI / 180.0f;

    // 预计算重映射表
    map_x.create(fish_size, CV_32F);
    map_y.create(fish_size, CV_32F);
    valid_mask.create(fish_size, CV_8U);

    const float fish_center_x = fish_size.width * 0.5f;
    const float fish_center_y = fish_size.height * 0.5f;
    const float radius = std::min(fish_center_x, fish_center_y);

    #pragma omp parallel for
    for (int y = 0; y < fish_size.height; ++y) {
        for (int x = 0; x < fish_size.width; ++x) {
            // 1. 鱼眼图像坐标转单位球面坐标
            float nx = (x - fish_center_x) / radius;  // 归一化 [-1,1]
            float ny = (y - fish_center_y) / radius;
            
            float r = std::sqrt(nx*nx + ny*ny);

            // 2. 转换为3D单位向量 (等距投影模型)
            float theta = r * fish_fov_rad * 0.5f;
            float sin_theta = std::sin(theta);
            float cos_theta = std::cos(theta);
            float phi = std::atan2(ny, nx);

            // 单位球面上的点
            cv::Vec3f point_3d(
                sin_theta * std::cos(phi),
                sin_theta * std::sin(phi),
                cos_theta
            );

            // 3. 应用旋转（如果提供）
            if (!R.empty()) {
                cv::Mat rotated = R * cv::Mat(point_3d);
                point_3d = cv::Vec3f(rotated.at<float>(0), rotated.at<float>(1), rotated.at<float>(2));
            }

            // 4. 理想针孔投影
            if (point_3d[2] <= 0) {  // 背面不可见
                valid_mask.at<uchar>(y, x) = 0;
                continue;
            }

            // 归一化平面坐标
            float xu = point_3d[0] / point_3d[2];
            float yu = point_3d[1] / point_3d[2];

            // 计算像素坐标
            float u = fx * xu + cx;
            float v = fy * yu + cy;

            // 检查是否在图像范围内
            if (u >= 0 && u < pinhole_size.width && v >= 0 && v < pinhole_size.height) {
                map_x.at<float>(y, x) = u;
                map_y.at<float>(y, x) = v;
                valid_mask.at<uchar>(y, x) = 255;
            } else {
                valid_mask.at<uchar>(y, x) = 0;
            }
        }
    }
}

cv::Mat IdealPinhole2Fisheye::convert(const cv::Mat& pinhole, int interpolation) const {
    cv::Mat fish_eye;
    cv::remap(pinhole, fish_eye, map_x, map_y, interpolation, cv::BORDER_CONSTANT, cv::Scalar(0));
    fish_eye.setTo(cv::Scalar(0), ~valid_mask);
    return fish_eye;
}