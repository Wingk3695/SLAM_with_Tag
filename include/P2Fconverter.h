#ifndef IDEAL_PINHOLE_2_FISHEYE_H
#define IDEAL_PINHOLE_2_FISHEYE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

class __attribute__((visibility("default"))) IdealPinhole2Fisheye {
private:
    Mat map_x, map_y;       // 重映射表
    Mat valid_mask;         // 有效像素掩码
    Size fish_eye_size;     // 鱼眼图像尺寸
    float fish_fov_rad;     // 鱼眼有效视场角（弧度）

public:
    /**
     * @brief 初始化理想模型转换器
     * @param pinhole_size 针孔图像尺寸
     * @param fish_size 鱼眼图像尺寸
     * @param fish_fov_deg 鱼眼有效视场角（度）
     * @param fx 水平焦距（像素）
     * @param fy 垂直焦距（像素）
     * @param cx 光心x坐标（像素）
     * @param cy 光心y坐标（像素）
     * @param R 可选旋转矩阵 (3x3 CV_32F)
     */
    void init(const Size& pinhole_size, const Size& fish_size, float fish_fov_deg,
              float fx, float fy, float cx, float cy,
              const Mat& R = Mat::eye(3, 3, CV_32F));

    /**
     * @brief 转换单帧图像
     * @param pinhole 输入针孔图像
     * @param interpolation 插值方法 (默认INTER_LINEAR)
     * @return 鱼眼图像
     */
    Mat convert(const Mat& pinhole, int interpolation = INTER_LINEAR) const;
};

#endif // IDEAL_PINHOLE_2_FISHEYE_H