#include <iostream>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "ORBextractor.h"
#include <include/CameraModels/KannalaBrandt8.h>

using namespace std;
using namespace ORB_SLAM3;

class FeatureExtractor {
public:
    FeatureExtractor() {}
    
    // 模仿Frame::ExtractORB方法
    void ExtractORB(ORBextractor* extractor, const cv::Mat &im, 
                   vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
    {
        vector<int> vLappingArea = {0, 0};
        (*extractor)(im, cv::Mat(), keypoints, descriptors, vLappingArea);
    }
};



// 反投影函数
cv::Point3f unproject(const cv::Point2f &p2D, const std::vector<float> &calib) {
    cv::Point2f pw((p2D.x - calib[2]) / calib[0], (p2D.y - calib[3]) / calib[1]);
    float scale = 1.f;
    float theta_d = sqrtf(pw.x * pw.x + pw.y * pw.y);
    theta_d = fminf(fmaxf(-CV_PI / 2.f, theta_d), CV_PI / 2.f);

    if (theta_d > 1e-8) {
        float theta = theta_d;
        float precision = 1e-9;

        for (int j = 0; j < 10; j++) {
            float theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 = theta4 * theta4;
            float k0_theta2 = calib[4] * theta2, k1_theta4 = calib[5] * theta4;
            float k2_theta6 = calib[6] * theta6, k3_theta8 = calib[7] * theta8;
            float theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                            (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
            theta = theta - theta_fix;
            if (fabsf(theta_fix) < precision)
                break;
        }
        scale = std::tan(theta) / theta_d;
    }

    return cv::Point3f(pw.x * scale, pw.y * scale, 1.f);
}

// 转换
Eigen::Vector3f unprojectEig(const cv::Point2f &p2D, const std::vector<float> &calib) {
    cv::Point3f ray = unproject(p2D, calib);
    return Eigen::Vector3f(ray.x, ray.y, ray.z);
}

// 投影函数
Eigen::Vector2f project(const Eigen::Vector3f &p3D, const std::vector<float> &calib) {
    const float x2_plus_y2 = p3D[0] * p3D[0] + p3D[1] * p3D[1];
    const float theta = atan2f(sqrtf(x2_plus_y2), p3D[2]);
    const float psi = atan2f(p3D[1], p3D[0]);

    const float theta2 = theta * theta;
    const float theta3 = theta * theta2;
    const float theta5 = theta3 * theta2;
    const float theta7 = theta5 * theta2;
    const float theta9 = theta7 * theta2;
    const float r = theta + calib[4] * theta3 + calib[5] * theta5
                    + calib[6] * theta7 + calib[7] * theta9;

    const float u = calib[0] * r * cos(psi) + calib[2];
    const float v = calib[1] * r * sin(psi) + calib[3];
    
    return Eigen::Vector2f(u, v);
}


// 三角化函数
void Triangulate(const cv::Point2f &p1, const cv::Point2f &p2, 
                const Eigen::Matrix<float,3,4> &Tcw1, const Eigen::Matrix<float,3,4> &Tcw2, 
                Eigen::Vector3f &x3D) {
    Eigen::Matrix<float,4,4> A;
    A.row(0) = p1.x*Tcw1.row(2)-Tcw1.row(0);
    A.row(1) = p1.y*Tcw1.row(2)-Tcw1.row(1);
    A.row(2) = p2.x*Tcw2.row(2)-Tcw2.row(0);
    A.row(3) = p2.y*Tcw2.row(2)-Tcw2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f x3Dh = svd.matrixV().col(3);
    x3D = x3Dh.head(3)/x3Dh(3);
}

// 三角化匹配点函数
float TriangulateMatches(
    const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, 
    const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12, 
    const float sigmaLevel, const float unc, 
    const std::vector<float> &calib1, const std::vector<float> &calib2,
    Eigen::Vector3f& p3D) {

    Eigen::Vector3f r1 = unprojectEig(kp1.pt, calib1);
    Eigen::Vector3f r2 = unprojectEig(kp2.pt, calib2);

    // Check parallax
    Eigen::Vector3f r21 = R12 * r2;
    const float cosParallaxRays = r1.dot(r21)/(r1.norm() *r21.norm());

    if(cosParallaxRays > 0.9998){
        cout<<"三角化-1";
        return -1;
    }

    // Parallax is good, so we try to triangulate
    cv::Point2f p11, p22;
    p11.x = r1[0];
    p11.y = r1[1];
    p22.x = r2[0];
    p22.y = r2[1];

    Eigen::Vector3f x3D;
    Eigen::Matrix<float,3,4> Tcw1;
    Tcw1 << Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero();

    Eigen::Matrix<float,3,4> Tcw2;
    Eigen::Matrix3f R21 = R12.transpose();
    Tcw2 << R21, -R21 * t12;

    Triangulate(p11, p22, Tcw1, Tcw2, x3D);

    float z1 = x3D(2);
    if(z1 <= 0){
        cout<<"三角化-2";
        return -2;
    }

    float z2 = R21.row(2).dot(x3D)+Tcw2(2,3);
    if(z2<=0){
        cout<<"三角化-3";
        return -3;
    }

    // Check reprojection error
    Eigen::Vector2f uv1 = project(x3D, calib1);
    float errX1 = uv1(0) - kp1.pt.x;
    float errY1 = uv1(1) - kp1.pt.y;

    if((errX1*errX1+errY1*errY1)>5.991 * sigmaLevel){
        cout<<"三角化-4";
        return -4;
    }

    Eigen::Vector3f x3D2 = R21 * x3D + Tcw2.col(3);
    Eigen::Vector2f uv2 = project(x3D2, calib2);
    float errX2 = uv2(0) - kp2.pt.x;
    float errY2 = uv2(1) - kp2.pt.y;

    if((errX2*errX2+errY2*errY2)>5.991 * unc){
        cout<<"三角化-5";
        return -5;
    }

    p3D = x3D;
    return z1;
}




int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "用法: ./feature_extractor 配置文件路径 左图像路径 右图像路径" << endl;
        return 1;
    }

    // 加载配置文件
    string strSettingsFile = argv[1];
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "无法打开配置文件: " << strSettingsFile << endl;
        return -1;
    }

    // 读取ORB参数
    int nFeatures = fsSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fsSettings["ORBextractor.scaleFactor"];
    int nLevels = fsSettings["ORBextractor.nLevels"];
    int fIniThFAST = fsSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fsSettings["ORBextractor.minThFAST"];

    cout << "ORB提取器参数:" << endl;
    cout << "- 特征点数量: " << nFeatures << endl;
    cout << "- 尺度因子: " << fScaleFactor << endl;
    cout << "- 金字塔层数: " << nLevels << endl;
    cout << "- 初始FAST阈值: " << fIniThFAST << endl;
    cout << "- 最小FAST阈值: " << fMinThFAST << endl;

    // 读取图像
    string leftImagePath = argv[2];
    string rightImagePath = argv[3];
    
    // ========== 读取相机内参和外参 ==========
    float fx1 = fsSettings["Camera1.fx"];
    float fy1 = fsSettings["Camera1.fy"];
    float cx1 = fsSettings["Camera1.cx"];
    float cy1 = fsSettings["Camera1.cy"];
    float fx2 = fsSettings["Camera2.fx"];
    float fy2 = fsSettings["Camera2.fy"];
    float cx2 = fsSettings["Camera2.cx"];
    float cy2 = fsSettings["Camera2.cy"];
    // 读取4x4外参矩阵（右到左）
    cv::Mat T_c1_c2 = fsSettings["Camera.T_c1_c2"].mat();
    cv::Mat R_c1_c2 = T_c1_c2(cv::Rect(0,0,3,3)).clone();
    cv::Mat t_c1_c2 = T_c1_c2(cv::Rect(3,0,1,3)).clone();

    cv::Mat imLeft = cv::imread(leftImagePath, cv::IMREAD_UNCHANGED);
    cv::Mat imRight = cv::imread(rightImagePath, cv::IMREAD_UNCHANGED);
    cv::Mat imGrayLeft, imGrayRight;
    if(imLeft.channels() == 3)
    {
        cv::cvtColor(imLeft, imGrayLeft, cv::COLOR_RGB2GRAY);
        cv::cvtColor(imRight, imGrayRight, cv::COLOR_RGB2GRAY);
    }
    else if(imLeft.channels() == 4)
    {
        cv::cvtColor(imLeft, imGrayLeft, cv::COLOR_RGBA2GRAY);
        cv::cvtColor(imRight, imGrayRight, cv::COLOR_RGBA2GRAY);
    }
    else
    {
        imGrayLeft = imLeft.clone();
        imGrayRight = imRight.clone();
    }


    
    
        // 构建相机内参矩阵
    cv::Mat K1 = (cv::Mat_<double>(3,3) << fx1, 0, cx1, 0, fy1, cy1, 0, 0, 1);
    cv::Mat K2 = (cv::Mat_<double>(3,3) << fx2, 0, cx2, 0, fy2, cy2, 0, 0, 1);
    
    // 确保R_c1_c2是double类型
    cv::Mat R_c1_c2_double;
    R_c1_c2.convertTo(R_c1_c2_double, CV_64F);
    
    // 左相机到正前方的旋转（向右旋转45度）
    double angle_left = -CV_PI / 4.0; // 左相机向右旋转45度到正前方
    cv::Mat rotAxis_left = (cv::Mat_<double>(3,1) << 0, 1, 0); // 绕Y轴旋转
    cv::Mat R_left_to_front;
    cv::Rodrigues(rotAxis_left * angle_left, R_left_to_front);
    
    // 右相机到正前方的旋转
    // 使用与之前相同的方法计算右相机的旋转，但添加额外的修正
    double angle_right = CV_PI / 4.0; // 右相机向左旋转45度到正前方
    cv::Mat rotAxis_right = (cv::Mat_<double>(3,1) << 0, 1, 0);
    cv::Mat R_right_rot;
    cv::Rodrigues(rotAxis_right * angle_right, R_right_rot);
    
    // 添加额外的旋转来修正右相机图像方向
    cv::Mat R_right_fix = cv::Mat::eye(3, 3, CV_64F);
    double fix_angle = -CV_PI / 2.0; // -90度旋转
    R_right_fix.at<double>(0,0) = cos(fix_angle);
    R_right_fix.at<double>(0,1) = -sin(fix_angle);
    R_right_fix.at<double>(1,0) = sin(fix_angle);
    R_right_fix.at<double>(1,1) = cos(fix_angle);
    
    // 组合右相机的旋转
    cv::Mat R_right_to_front = R_right_rot * R_right_fix;
    
    // 计算单应性矩阵
    cv::Mat K1_inv, K2_inv;
    cv::invert(K1, K1_inv);
    cv::invert(K2, K2_inv);
    
    // 左相机的单应性矩阵 - 保持不变
    cv::Mat H_left = K1 * R_left_to_front * K1_inv;
    
    // 右相机的单应性矩阵
    cv::Mat H_right = K2 * R_right_to_front * R_c1_c2_double.t() * K2_inv;
    
    // 计算变换后的图像边界
    std::vector<cv::Point2f> corners_left(4), corners_right(4);
    corners_left[0] = cv::Point2f(0, 0);
    corners_left[1] = cv::Point2f(imGrayLeft.cols, 0);
    corners_left[2] = cv::Point2f(imGrayLeft.cols, imGrayLeft.rows);
    corners_left[3] = cv::Point2f(0, imGrayLeft.rows);
    
    corners_right[0] = cv::Point2f(0, 0);
    corners_right[1] = cv::Point2f(imGrayRight.cols, 0);
    corners_right[2] = cv::Point2f(imGrayRight.cols, imGrayRight.rows);
    corners_right[3] = cv::Point2f(0, imGrayRight.rows);
    
    std::vector<cv::Point2f> corners_left_transformed, corners_right_transformed;
    cv::perspectiveTransform(corners_left, corners_left_transformed, H_left);
    cv::perspectiveTransform(corners_right, corners_right_transformed, H_right);
    
    // 计算所有变换后点的边界
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    
    for (const auto& pt : corners_left_transformed) {
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }
    
    for (const auto& pt : corners_right_transformed) {
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }
    
    // 创建平移矩阵，确保所有像素都在可视范围内
    cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
    T.at<double>(0,2) = -min_x; // x方向的偏移
    T.at<double>(1,2) = -min_y; // y方向的偏移
    
    // 应用平移到单应性矩阵
    cv::Mat H_left_final = T * H_left;
    cv::Mat H_right_final = T * H_right;
    
    // 计算输出图像大小
    int output_width = cvRound(max_x - min_x) + 50;  // 添加一些边距
    int output_height = cvRound(max_y - min_y) + 50;
    cv::Size output_size(output_width, output_height);
    
    // 应用单应性变换
    cv::Mat imRectLeft, imRectRight;
    cv::warpPerspective(imGrayLeft, imRectLeft, H_left_final, output_size);
    cv::warpPerspective(imGrayRight, imRectRight, H_right_final, output_size);
    
    // 缩放图像以适应屏幕显示
    double scale_factor = 0.3; // 缩放为原来的30%
    cv::Mat imLeftSmall, imRightSmall, imRectLeftSmall, imRectRightSmall;
    
    cv::resize(imGrayLeft, imLeftSmall, cv::Size(), scale_factor, scale_factor);
    cv::resize(imGrayRight, imRightSmall, cv::Size(), scale_factor, scale_factor);
    cv::resize(imRectLeft, imRectLeftSmall, cv::Size(), scale_factor, scale_factor);
    cv::resize(imRectRight, imRectRightSmall, cv::Size(), scale_factor, scale_factor);
    
     // 创建合并图像 - 直接取左图左半部分和右图右半部分
    cv::Mat mergedRect;
    if (imRectLeftSmall.size() == imRectRightSmall.size()) {
        // 创建与原图相同大小的合并图像
        mergedRect = cv::Mat(imRectLeftSmall.rows, imRectLeftSmall.cols, imRectLeftSmall.type());
        
        // 计算中点
        int midCol = imRectLeftSmall.cols / 2;
        
        // 复制左图的左半部分
        cv::Mat leftHalf = imRectLeftSmall(cv::Rect(0, 0, midCol, imRectLeftSmall.rows));
        cv::Mat leftROI = mergedRect(cv::Rect(0, 0, midCol, imRectLeftSmall.rows));
        leftHalf.copyTo(leftROI);
        
        // 复制右图的右半部分
        cv::Mat rightHalf = imRectRightSmall(cv::Rect(midCol, 0, imRectRightSmall.cols - midCol, imRectRightSmall.rows));
        cv::Mat rightROI = mergedRect(cv::Rect(midCol, 0, imRectRightSmall.cols - midCol, imRectRightSmall.rows));
        rightHalf.copyTo(rightROI);
        
        // 在中间画一条分割线
        cv::line(mergedRect, cv::Point(midCol, 0), cv::Point(midCol, mergedRect.rows), cv::Scalar(255), 1);
    } else {
        cout << "警告: 左右图像大小不一致，无法简单合并。" << endl;
    }
    
    // 显示缩放后的图像
    cv::imshow("Orin Left", imLeftSmall);
    cv::imshow("Orin Right", imRightSmall);
    cv::imshow("Rec Left", imRectLeftSmall);
    cv::imshow("Rec Right", imRectRightSmall);
     // 如果成功创建了合并图像，则显示
    if (!mergedRect.empty()) {
        cv::imshow("合并投影图 (左右各半)", mergedRect);
    }


    // ORBextractor* pORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    // ORBextractor* pORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    // vector<cv::KeyPoint> mvKeysLeft, mvKeysRight;
    // cv::Mat mDescriptorsLeft, mDescriptorsRight;
    
    // FeatureExtractor extractor;
    // thread threadLeft(&FeatureExtractor::ExtractORB, &extractor, pORBextractorLeft, std::ref(imGrayLeft), std::ref(mvKeysLeft), std::ref(mDescriptorsLeft));
    // thread threadRight(&FeatureExtractor::ExtractORB, &extractor, pORBextractorRight, std::ref(imGrayRight), std::ref(mvKeysRight), std::ref(mDescriptorsRight));
    // threadLeft.join();
    // threadRight.join();
    // cout << "左图像特征点数量: " << mvKeysLeft.size() << endl;
    // cout << "右图像特征点数量: " << mvKeysRight.size() << endl;
    
    // // 可视化特征点
    // cv::Mat imLeftKeypoints, imRightKeypoints;
    // cv::drawKeypoints(imGrayLeft, mvKeysLeft, imLeftKeypoints);
    // cv::drawKeypoints(imGrayRight, mvKeysRight, imRightKeypoints);
    // cv::imshow("Left Image Features", imLeftKeypoints);
    // cv::imshow("Right Image Features", imRightKeypoints);
    

    // cv::BFMatcher bfmatcher(cv::NORM_HAMMING);
    // std::vector<std::vector<cv::DMatch>> knnMatches;
    // bfmatcher.knnMatch(mDescriptorsLeft, mDescriptorsRight, knnMatches, 2);
    // cout << "暴力初始匹配对: " << knnMatches.size() << std::endl;

    // std::vector<cv::DMatch> goodMatches;
    // const float ratio_thresh = 0.8f;
    // for (size_t i = 0; i < knnMatches.size(); i++) {
    //     if (knnMatches[i].size() >= 2 && knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
    //         goodMatches.push_back(knnMatches[i][0]);
    //     }
    // }
    // cout << "无几何约束的有效匹配数: " << goodMatches.size() << endl;
    // cv::Mat imgMatchNoGeom;
    // cv::drawMatches(imGrayLeft, mvKeysLeft, imGrayRight, mvKeysRight, goodMatches, imgMatchNoGeom);
    // cv::imshow("Matches without Geometry", imgMatchNoGeom);

    // // 区域筛选匹配
    // std::vector<cv::DMatch> filteredMatches;
    // int leftThreshold = imGrayLeft.cols  / 2;
    // int rightThreshold = imGrayRight.cols / 2;    

    // for (const auto& match : goodMatches) {
    //     const cv::KeyPoint& leftKP = mvKeysLeft[match.queryIdx];
    //     const cv::KeyPoint& rightKP = mvKeysRight[match.trainIdx];
    //     if (leftKP.pt.x >= leftThreshold && rightKP.pt.x <= rightThreshold) {
    //         filteredMatches.push_back(match);
    //     }
    // }
    // cout << "区域筛选后的匹配数: " << filteredMatches.size() << endl;
    // cv::Mat imgFilteredMatches;
    // cv::drawMatches(imGrayLeft, mvKeysLeft, imGrayRight, mvKeysRight, filteredMatches, imgFilteredMatches);
    // cv::imshow("Filtered Region Matches", imgFilteredMatches);

    // //////////////////////////////////
    // cout << "开始原始三角化过程..." << endl;

    // // 将相机参数组成数组
    // std::vector<float> vCalibration1 = {fx1, fy1, cx1, cy1, 0, 0, 0, 0};
    // std::vector<float> vCalibration2 = {fx2, fy2, cx2, cy2, 0, 0, 0, 0};

    // // 转换为Eigen矩阵
    // Eigen::Matrix3f R12;
    // Eigen::Vector3f t12;
    // cv::cv2eigen(R_c1_c2, R12);
    // cv::cv2eigen(t_c1_c2, t12);

    // // 初始化匹配和深度向量
    // std::vector<int> mvLeftToRightMatch(mvKeysLeft.size(), -1);
    // std::vector<int> mvRightToLeftMatch(mvKeysRight.size(), -1);
    // std::vector<Eigen::Vector3f> mvStereo3Dpoints(mvKeysLeft.size());
    // std::vector<float> mvDepth(mvKeysLeft.size(), -1.0f);

    // // 设置每个金字塔层级的尺度
    // std::vector<float> mvLevelSigma2(nLevels);
    // mvLevelSigma2[0] = 1.0f;
    // for(int i=1; i<nLevels; i++)
    //     mvLevelSigma2[i] = mvLevelSigma2[i-1] * fScaleFactor * fScaleFactor;

    // int nMatches = 0;

    // // 遍历所有匹配点
    // for(const auto& match : goodMatches) {
    //     int queryIdx = match.queryIdx;
    //     int trainIdx = match.trainIdx;
        
    //     // 确保索引有效
    //     if(queryIdx >= 0 && queryIdx < (int)mvKeysLeft.size() && 
    //        trainIdx >= 0 && trainIdx < (int)mvKeysRight.size()) {
            
    //         // 确保octave索引有效
    //         int octaveLeft = mvKeysLeft[queryIdx].octave;
    //         int octaveRight = mvKeysRight[trainIdx].octave;
            
    //         if(octaveLeft >= 0 && octaveLeft < nLevels && 
    //            octaveRight >= 0 && octaveRight < nLevels) {
                
    //             float sigma1 = mvLevelSigma2[octaveLeft];
    //             float sigma2 = mvLevelSigma2[octaveRight];
                
    //             Eigen::Vector3f p3D;
    //             float depth = TriangulateMatches(
    //                 mvKeysLeft[queryIdx], 
    //                 mvKeysRight[trainIdx],
    //                 R12, t12, sigma1, sigma2,
    //                 vCalibration1, vCalibration2, p3D);
                
    //             if(depth > 0.0001f) {
    //                 mvLeftToRightMatch[queryIdx] = trainIdx;
    //                 mvRightToLeftMatch[trainIdx] = queryIdx;
    //                 mvStereo3Dpoints[queryIdx] = p3D;
    //                 mvDepth[queryIdx] = depth;
    //                 nMatches++;
    //             }
    //         }
    //     }
    // }

    // cout << "\n三角化后有效匹配数: " << nMatches << endl;

    // // 可视化三角化后的匹配
    // std::vector<cv::DMatch> triangulatedMatches;
    // for(size_t i = 0; i < mvLeftToRightMatch.size(); i++) {
    //     if(mvLeftToRightMatch[i] >= 0) {
    //         cv::DMatch match;
    //         match.queryIdx = i;
    //         match.trainIdx = mvLeftToRightMatch[i];
    //         triangulatedMatches.push_back(match);
    //     }
    // }

    // cv::Mat imgMatchTriangulated;
    // cv::drawMatches(imGrayLeft, mvKeysLeft, imGrayRight, mvKeysRight, triangulatedMatches, imgMatchTriangulated);
    // cv::imshow("Original Triangulated Matches", imgMatchTriangulated);

    // // 简易三角化
    // std::vector<cv::DMatch> validTriangulatedMatches;
    // for (const auto& match : goodMatches) {
    //     // 左右图像的特征点
    //     cv::KeyPoint kpL = mvKeysLeft[match.queryIdx];
    //     cv::KeyPoint kpR = mvKeysRight[match.trainIdx];

    //     // 像素坐标归一化
    //     cv::Mat pt1 = (cv::Mat_<double>(3,1) << (kpL.pt.x - cx1)/fx1, (kpL.pt.y - cy1)/fy1, 1.0);
    //     cv::Mat pt2 = (cv::Mat_<double>(3,1) << (kpR.pt.x - cx2)/fx2, (kpR.pt.y - cy2)/fy2, 1.0);

    //     // 三角化
    //     cv::Mat P1 = cv::Mat::eye(3,4,CV_64F); // 左相机为参考
    //     cv::Mat P2(3,4,CV_64F);
    //     R_c1_c2.convertTo(P2(cv::Rect(0,0,3,3)), CV_64F);
    //     t_c1_c2.convertTo(P2(cv::Rect(3,0,1,3)), CV_64F);

    //     cv::Mat pts_4d;
    //     cv::triangulatePoints(P1, P2, pt1.rowRange(0,2), pt2.rowRange(0,2), pts_4d);

    //     // 齐次归一化
    //     double w = pts_4d.at<double>(3,0);
    //     if (fabs(w) < 1e-6) continue;
    //     double z = pts_4d.at<double>(2,0) / w;
    //     if (z > 0) { // 只保留深度为正的点
    //         validTriangulatedMatches.push_back(match);
    //     }
    // }
    // cout << "简易三角化约束的有效匹配数: " << validTriangulatedMatches.size() << endl;
    // cv::Mat imgMatchWithGeom;
    // cv::drawMatches(imGrayLeft, mvKeysLeft, imGrayRight, mvKeysRight, validTriangulatedMatches, imgMatchWithGeom);
    // cv::imshow("Matches with cv Geometry", imgMatchWithGeom);

    // // 简易三角化+区域筛选
    // std::vector<cv::DMatch> validTriangulatedFilteredMatches;
    // for (const auto& match : filteredMatches) {  
    //     // 左右图像的特征点
    //     cv::KeyPoint kpL = mvKeysLeft[match.queryIdx];
    //     cv::KeyPoint kpR = mvKeysRight[match.trainIdx];

    //     // 像素坐标归一化
    //     cv::Mat pt1 = (cv::Mat_<double>(3,1) << (kpL.pt.x - cx1)/fx1, (kpL.pt.y - cy1)/fy1, 1.0);
    //     cv::Mat pt2 = (cv::Mat_<double>(3,1) << (kpR.pt.x - cx2)/fx2, (kpR.pt.y - cy2)/fy2, 1.0);

    //     // 三角化
    //     cv::Mat P1 = cv::Mat::eye(3,4,CV_64F); // 左相机为参考
    //     cv::Mat P2(3,4,CV_64F);
    //     R_c1_c2.convertTo(P2(cv::Rect(0,0,3,3)), CV_64F);
    //     t_c1_c2.convertTo(P2(cv::Rect(3,0,1,3)), CV_64F);

    //     cv::Mat pts_4d;
    //     cv::triangulatePoints(P1, P2, pt1.rowRange(0,2), pt2.rowRange(0,2), pts_4d);

    //     // 齐次归一化
    //     double w = pts_4d.at<double>(3,0);
    //     if (fabs(w) < 1e-6) continue;
    //     double z = pts_4d.at<double>(2,0) / w;
    //     if (z > 0) { // 只保留深度为正的点
    //         validTriangulatedFilteredMatches.push_back(match);
    //     }
    // }
    // cout << "简易三角化约束+区域筛选的有效匹配数: " << validTriangulatedFilteredMatches.size() << endl;
    // cv::Mat imgMatchWithGeomAndFilter;
    // cv::drawMatches(imGrayLeft, mvKeysLeft, imGrayRight, mvKeysRight, validTriangulatedFilteredMatches, imgMatchWithGeomAndFilter);
    // cv::imshow("Matches with cv Geometry and filter", imgMatchWithGeomAndFilter);



    cv::waitKey(0);
    return 0;
}
