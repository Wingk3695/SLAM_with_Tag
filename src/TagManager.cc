#include "TagManager.h"
#include "KeyFrame.h"
#include <apriltag.h>
#include <tag36h11.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include "Converter.h"

using namespace ORB_SLAM3;

// 静态成员初始化
apriltag_family_t* TagStorage::mpTagFamily = nullptr;
apriltag_detector_t* TagStorage::mpDetector = nullptr;

TagStorage& TagStorage::Instance() {
    static TagStorage instance;
    return instance;
}

TagStorage::TagStorage() 
    : mLoaded(false){
    // 构造时默认不初始化 AprilTag 检测器
}

TagStorage::~TagStorage() {
    DestroyDetector();
}

void TagStorage::InitDetectorAndLoad(const std::string& filename) {
    if (mpDetector) return; // 已初始化
    // 创建 tag36h11 family
    mpTagFamily = tag36h11_create();
    // 创建检测器
    mpDetector = apriltag_detector_create();
    apriltag_detector_add_family_bits(mpDetector, mpTagFamily, 1);
    // 参数配置
    mpDetector->quad_decimate = 2.0;
    mpDetector->quad_sigma    = 0.0;
    mpDetector->nthreads      = 4;
    mpDetector->debug         = 0;
    mpDetector->refine_edges  = 1;
    // 载入已有Tag
    mLoaded = tagLoad(filename);
    if (mLoaded) {
        std::cout << "[TagStorage] Loaded tag poses from \"" << filename << "\".\n";
    } else {
        std::cout << "[TagStorage] No existing pose file found, will compute from observations.\n";
    }
}

apriltag_detector_t* TagStorage::GetDetector() {
    if (!mpDetector) InitDetectorAndLoad();
    return mpDetector;
}

void TagStorage::DestroyDetector() {
    if (mpDetector) {
        apriltag_detector_destroy(mpDetector);
        mpDetector = nullptr;
    }
    if (mpTagFamily) {
        tag36h11_destroy(mpTagFamily);
        mpTagFamily = nullptr;
    }
}

void TagStorage::tagWrite(int id,
                           KeyFrame* pKF,
                           const Eigen::Matrix3d R_cam_tag,
                           const Eigen::Vector3d t_cam_tag)
{
    if (!pKF) return; // 自检：空指针直接丢弃
    std::lock_guard<std::mutex> lock(mMutex);
    TagObs tempobs;
    tempobs.pKF = pKF;
    tempobs.R_cam_tag = R_cam_tag;
    tempobs.t_cam_tag = t_cam_tag;
    mStorage[id].push_back(tempobs);
}

bool TagStorage::tagRead(int id,
                         Eigen::Matrix3d& R_w_tag_avg,
                         Eigen::Vector3d& t_w_tag_avg)
{
    std::lock_guard<std::mutex> lock(mMutex);

    // 如果已经从文件载入，直接返回缓存
    if(mLoaded){
        auto itRt = mStorageRt.find(id);
        if (itRt != mStorageRt.end()) {
            R_w_tag_avg = itRt->second.first;
            t_w_tag_avg = itRt->second.second;
            return true;
        }
        else{
            return false;
        }
    }

    // 没载入，就从观测累积计算
    auto itObs = mStorage.find(id);
    if (itObs == mStorage.end() || itObs->second.empty())
        return false;

    std::vector<Eigen::Quaterniond> qs;
    std::vector<Eigen::Vector3d> ts;

    for (auto& obs : itObs->second) {
        ORB_SLAM3::KeyFrame* pKF = obs.pKF;
        if (!pKF || pKF->isBad()) continue;

        // 1) 取出 Camera->World 的 4×4 变换
        Sophus::SE3f Twc_SE3 = pKF->GetPoseInverse();
        Eigen::Matrix4f Twc_mat = Twc_SE3.matrix();
        Eigen::Matrix3d R_w_c = Twc_mat.block<3,3>(0,0).cast<double>();
        Eigen::Vector3d t_w_c = Twc_mat.block<3,1>(0,3).cast<double>();

        // 3) 世界到 Tag  = (相机到世界) * (相机到 Tag)
        Eigen::Matrix3d R_w_tag = R_w_c * obs.R_cam_tag;
        Eigen::Vector3d t_w_tag = R_w_c * obs.t_cam_tag + t_w_c;

        qs.emplace_back(R_w_tag);
        ts.emplace_back(t_w_tag);
    }

    if (qs.empty()) return false;

    // 平均四元数
    Eigen::Quaterniond q_avg(0,0,0,0);
    for (auto& q : qs) q_avg.coeffs() += q.coeffs();
    q_avg.normalize();

    // 平均平移
    Eigen::Vector3d t_avg(0,0,0);
    for (auto& t : ts) t_avg += t;
    t_avg /= double(ts.size());

    // —— 计算平移误差 —— 
    // 误差定义为每次观测平移与平均平移的欧式距离
    std::vector<double> errs;
    errs.reserve(ts.size());
    for (auto& t : ts) {
        errs.push_back((t - t_avg).norm());
    }
    // 求最大和平均
    double t_err_max = *std::max_element(errs.begin(), errs.end());
    double t_err_avg = std::accumulate(errs.begin(), errs.end(), 0.0) / errs.size();
    cout<< "id：" << id << "最大误差：" << t_err_max << "m 平均误差：" << t_err_avg << "m\n";


    R_w_tag_avg = q_avg.toRotationMatrix();
    t_w_tag_avg = t_avg;

    // 缓存结果，下次直接用
    mStorageRt[id] = std::make_pair(R_w_tag_avg, t_w_tag_avg);
    return true;
}


void TagStorage::tagCleanup() {
    std::lock_guard<std::mutex> lk(mMutex);
    for (auto it = mStorage.begin(); it != mStorage.end(); ) {
        auto& vec = it->second;
        vec.erase(std::remove_if(vec.begin(), vec.end(), [](const TagObs& obs) {
            return (obs.pKF == nullptr) || (obs.pKF->isBad());
        }), vec.end());

        if (vec.empty()) {
            it = mStorage.erase(it);
        } else {
            ++it;
        }
    }
}

bool TagStorage::tagSave(const std::string& filename) {
    std::ofstream outFile(filename, std::ios::trunc);
    if (!outFile) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return false;
    }

    outFile << std::fixed << std::setprecision(15);

    for (const auto& entry : mStorageRt) {
        int id = entry.first;
        const auto& Rt = entry.second;
        const Eigen::Matrix3d& R = Rt.first;
        const Eigen::Vector3d& t = Rt.second;

        // 检查 Rt 是否有效，无效则跳过
        if (!isRtValid(R, t)) {
            std::cout << "跳过无效 ID: " << id << std::endl;
            continue;
        }

        // 写入数据
        outFile << id << " ";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                outFile << R(i, j) << " ";
            }
        }
        outFile << t(0) << " " << t(1) << " " << t(2) << "\n";

        std::cout << "保存 Tag " << id << ":\nR:\n" << R << "\nt: " << t.transpose() << "\n\n";
    }

    outFile.close();
    return true;
}

bool TagStorage::tagLoad(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    mStorageRt.clear();

    int id;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    while (inFile >> id) {
        // 读取 R 和 t
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (!(inFile >> R(i, j))) {
                    std::cerr << "读取旋转矩阵失败 (ID: " << id << ")" << std::endl;
                    inFile.close();
                    return false;
                }
            }
        }

        if (!(inFile >> t(0) >> t(1) >> t(2))) {
            std::cerr << "读取平移向量失败 (ID: " << id << ")" << std::endl;
            inFile.close();
            return false;
        }

        // 检查 Rt 是否有效，无效则跳过
        if (!isRtValid(R, t)) {
            std::cout << "跳过无效数据 (ID: " << id << ")" << std::endl;
            continue;
        }

        // 存入 mStorageRt
        mStorageRt[id] = std::make_pair(R, t);

        std::cout << "加载 Tag " << id << ":\nR:\n" << R << "\nt: " << t.transpose() << "\n\n";
    }

    inFile.close();
    mLoaded = true;
    return true;
}

// 获取某个关键帧的所有Tag观测
std::map<int, std::vector<TagStorage::TagObs>> TagStorage::GetObservationsForKF(int kfId) {
    std::lock_guard<std::mutex> lock(mMutex);
    std::map<int, std::vector<TagObs>> result;
    for(auto& [tagId, obsVec] : mStorage) {
        for(auto& obs : obsVec) {
            if(obs.pKF && obs.pKF->mnId == kfId) {
                result[tagId].push_back(obs);
            }
        }
    }
    return result;
}

// 获取Tag的观测次数
int TagStorage::GetObservationCount(int tagId) {
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mStorage.find(tagId);
    return (it != mStorage.end()) ? it->second.size() : 0;
}

bool TagStorage::isRtValid(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, double eps) {
    // 检查 R 是否是正交矩阵（R * R^T ≈ I）
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (!(R * R.transpose()).isApprox(I, eps)) {
        return false;
    }

    // 检查 t 是否为零向量（可选）
    if (t.norm() < eps) {
        return false;
    }

    return true;
}