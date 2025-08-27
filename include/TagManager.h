#ifndef TAGSTORAGE_H
#define TAGSTORAGE_H

#include <map>
#include <vector>
#include <mutex>
#include <Eigen/Core>
#include <string>

#include <apriltag.h>
// forward declare ORB-SLAM3 KeyFrame
namespace ORB_SLAM3 {
    class KeyFrame;
}

class TagStorage {
public:
    // 获取单例实例
    static TagStorage& Instance();

    // AprilTag 全局检测器与 Family
    static apriltag_family_t* mpTagFamily;
    static apriltag_detector_t* mpDetector;
    // 初始化 AprilTag 检测器（只需调用一次）
    void InitDetectorAndLoad(const std::string& filename = "tag_poses.txt");
    // 访问检测器指针，供外部调用 detect
    apriltag_detector_t* GetDetector();
    // 销毁检测器与 TagFamily
    void DestroyDetector();

    apriltag_detector_t* GetThreadDetector(int thread_id); // 获取线程专用检测器
    void InitDetectorPool(int pool_size = 4); // 初始化检测器池
    void DestroyDetectorPool(); // 销毁检测器池

    // 一次观测结构
    struct TagObs {
        ORB_SLAM3::KeyFrame* pKF;
        Eigen::Matrix3d R_cam_tag;
        Eigen::Vector3d t_cam_tag;
        int camID = 0;
    };

    // 写入：Tag观测记录
    void tagWrite(int id,
                  ORB_SLAM3::KeyFrame* pKF,
                  const Eigen::Matrix3d R_cam_tag,
                  const Eigen::Vector3d t_cam_tag,
                  int camID = 0);

    // 读取：
    // 未载入：观测记录的Rt是相机坐标系，需要利用pKF对应的位姿转化到世界坐标系，计算平均的世界坐标系下 R/t并更新到mStorageRt
    // 已载入：从mStorageRt取出
    bool tagRead(int id,
                 Eigen::Matrix3d& R_w_tag_avg,
                 Eigen::Vector3d& t_w_tag_avg);
    bool tagRead(int id,
                 Eigen::Matrix3d& R_w_tag_avg,
                 Eigen::Vector3d& t_w_tag_avg,
                 double& t_err_avg);

    // 自检：KeyFrame在SLAM过程中可能被删除，isbad()或指针变空，需要清理失效的tag观测记录
    void tagCleanup();

    // 保存：SLAM结束后保存mStorageRt为本地文件，并遍历cout，空id跳过
    bool tagSave(const std::string& filename = "tag_poses.txt");
    // 载入：读取文件恢复mStorageRt
    bool tagLoad(const std::string& filename = "tag_poses.txt");


    // 获取某个关键帧的所有Tag观测
    std::map<int, std::vector<TagObs>> GetObservationsForKF(int kfId);

    // 获取Tag的观测次数
    int GetObservationCount(int tagId);

    bool isRtValid(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, double eps = 1e-6);

private:
    TagStorage();
    ~TagStorage();
    TagStorage(const TagStorage&) = delete;
    TagStorage& operator=(const TagStorage&) = delete;

    std::vector<apriltag_detector_t*> mDetectorPool; // 检测器池

    // 观测记录存储
    std::map<int, std::vector<TagObs>> mStorage;
    // 观测结果存储
    std::map<int, std::pair<Eigen::Matrix3d, Eigen::Vector3d>> mStorageRt;

    std::mutex mMutex;
    // 标志：是否从文件载入过 mStorageRt
    bool mLoaded;
};

#endif // TAGSTORAGE_H