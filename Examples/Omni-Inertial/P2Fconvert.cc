#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

class IdealPinhole2Fisheye {
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
              const Mat& R = Mat::eye(3, 3, CV_32F)) {
        
        fish_eye_size = fish_size;
        fish_fov_rad = fish_fov_deg * CV_PI / 180.0f;

        // 预计算重映射表
        map_x.create(fish_size, CV_32F);
        map_y.create(fish_size, CV_32F);
        valid_mask.create(fish_size, CV_8U);

        const float fish_center_x = fish_size.width * 0.5f;
        const float fish_center_y = fish_size.height * 0.5f;
        const float radius = min(fish_center_x, fish_center_y);

        #pragma omp parallel for
        for (int y = 0; y < fish_size.height; ++y) {
            for (int x = 0; x < fish_size.width; ++x) {
                // 1. 鱼眼图像坐标转单位球面坐标
                float nx = (x - fish_center_x) / radius;  // 归一化 [-1,1]
                float ny = (y - fish_center_y) / radius;
                
                float r = sqrt(nx*nx + ny*ny);
                // if (r > 1.0f) {
                //     valid_mask.at<uchar>(y, x) = 0;
                //     continue;
                // }

                // 2. 转换为3D单位向量 (等距投影模型)
                float theta = r * fish_fov_rad * 0.5f;
                float sin_theta = sin(theta);
                float cos_theta = cos(theta);
                float phi = atan2(ny, nx);

                // 单位球面上的点
                Vec3f point_3d(
                    sin_theta * cos(phi),
                    sin_theta * sin(phi),
                    cos_theta
                );

                // 3. 应用旋转（如果提供）
                if (R.data != nullptr) {
                    Mat rotated = R * Mat(point_3d);
                    point_3d = Vec3f(rotated.at<float>(0), rotated.at<float>(1), rotated.at<float>(2));
                }

                // 4. 理想针孔投影
                if (point_3d[2] <= 0) {  // 背面不可见
                    valid_mask.at<uchar>(y, x) = 0;
                    continue;
                }

                // 归一化平面坐标（理想投影）
                float xu = point_3d[0] / point_3d[2];
                float yu = point_3d[1] / point_3d[2];

                // 计算像素坐标（使用内参）
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

    /**
     * @brief 转换单帧图像
     * @param pinhole 输入针孔图像
     * @param interpolation 插值方法 (默认INTER_LINEAR)
     * @return 鱼眼图像
     */
    Mat convert(const Mat& pinhole, int interpolation = INTER_LINEAR) const {
        Mat fish_eye;
        remap(pinhole, fish_eye, map_x, map_y, interpolation, BORDER_CONSTANT, Scalar(0));
        fish_eye.setTo(Scalar(0), ~valid_mask);
        return fish_eye;
    }

    /**
     * @brief 批量转换图像
     * @param pinholes 输入针孔图像数组
     * @param fish_eyes 输出鱼眼图像数组
     * @param interpolation 插值方法
     */
    void convertBatch(const vector<Mat>& pinholes, vector<Mat>& fish_eyes, int interpolation = INTER_LINEAR) const {
        fish_eyes.resize(pinholes.size());
        #pragma omp parallel for
        for (size_t i = 0; i < pinholes.size(); ++i) {
            remap(pinholes[i], fish_eyes[i], map_x, map_y, interpolation, BORDER_CONSTANT, Scalar(0));
            fish_eyes[i].setTo(Scalar(0), ~valid_mask);
        }
    }
};

// 示例使用
int main() {
    // 1. 设置相机参数（理想模型）
    float fx = 262.51551819f, fy = 262.51551819f;  // 焦距
    float cx = 318.28943253f, cy = 254.40927315f;   // 光心

    // 2. 初始化转换器
    IdealPinhole2Fisheye converter;
    converter.init(Size(640, 480),  // 针孔图像尺寸
                  Size(600, 540),   // 鱼眼图像尺寸
                  85.0f,           // 鱼眼FOV (度)
                  fx, fy, cx, cy); // 内参

    // 3. 实时处理示例
    Mat img = cv::imread("./imgPinhole4F.png",cv::IMREAD_UNCHANGED);

    Mat frame, fish_eye;
    frame = img;

    // 转换并测量耗时
    auto t1 = chrono::high_resolution_clock::now();
    fish_eye = converter.convert(frame);
    auto t2 = chrono::high_resolution_clock::now();
        
    // 显示结果
    imwrite("Pinhole.png", frame);
    imwrite("Fisheye.png", fish_eye);
        
    // 打印处理时间（毫秒）
    cout << "Process time: " 
        << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() 
        << "ms" << endl;

    
    return 0;
}