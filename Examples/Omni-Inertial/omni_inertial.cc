#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <ctime>
#include <sstream>

#include <opencv2/core/core.hpp>

#include <thread>
#include <chrono>

#include<System.h>
#include "ImuTypes.h"
#include "Optimizer.h"

#include <apriltag.h>
#include <apriltag_pose.h>
#include <tag36h11.h>
#include "TagManager.h"
#include <Eigen/Core>

using namespace std;

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathSideLeft, const string &strPathSideRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<string> &vstrImageSideLeft, vector<string> &vstrImageSideRight, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

// ===================================================================================
// 新增：带重试逻辑的健壮图像读取函数
// ===================================================================================
/**
 * @brief Robustly reads an image with retry logic to handle transient I/O errors.
 * @param filename Path to the image file.
 * @param flags Flags passed to cv::imread.
 * @param max_attempts Maximum number of read attempts.
 * @param delay_ms Initial delay in milliseconds between attempts.
 * @return The loaded cv::Mat. Returns an empty Mat if all attempts fail.
 */
cv::Mat imread_robust(const std::string& filename, int flags = cv::IMREAD_UNCHANGED, int max_attempts = 5, int delay_ms = 50) {
    cv::Mat image;
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        image = cv::imread(filename, flags);
        if (!image.empty()) {
            return image; // 成功读取
        }
        // 读取失败，记录警告，等待后重试
        std::cerr << "Warning: Failed to load image " << filename
                  << " on attempt " << attempt << ". Retrying..." << std::endl;
        // 等待一个逐渐增加的时间
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms * attempt));
    }
    // 所有尝试均失败
    std::cerr << "Error: Completely failed to load image " << filename << " after " << max_attempts << " attempts." << std::endl;
    return image; // 返回空图像
}
// ===================================================================================

int main(int argc, char **argv)
{
    if(argc < 5)
    {
        cerr << endl << "Usage: ./multi_inertial_euroc path_to_vocabulary path_to_settings path_to_sequence_folder_1 path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... path_to_image_folder_N path_to_times_file_N) " << endl;
        return 1;
    }

    const int num_seq = (argc-3)/2;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName= (((argc-3) % 2) == 1);
    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc-1]);
        cout << "file name: " << file_name << endl;
    }

    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageLeft;
    vector< vector<string> > vstrImageRight;
    vector< vector<string> > vstrImageSideLeft;
    vector< vector<string> > vstrImageSideRight;
    vector< vector<double> > vTimestampsCam;
    vector< vector<cv::Point3f> > vAcc, vGyro;
    vector< vector<double> > vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageLeft.resize(num_seq);
    vstrImageRight.resize(num_seq);
    vstrImageSideLeft.resize(num_seq);
    vstrImageSideRight.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";

        string pathSeq(argv[(2*seq) + 3]);
        string pathTimeStamps(argv[(2*seq) + 4]);
        // cam 1043
        string pathCam0 = pathSeq + "/left";  // Left Camera
        string pathCam1 = pathSeq + "/right";  // Right Camera
        string pathCam2 = pathSeq + "/sideleft";  // SideLeft Camera
        string pathCam3 = pathSeq + "/sideright";  // Sideright Camera
        string pathImu = pathSeq + "/imu/imu_data.csv";
        LoadImages(pathCam0, pathCam1, pathCam2, pathCam3, pathTimeStamps, vstrImageLeft[seq], vstrImageRight[seq], vstrImageSideLeft[seq], vstrImageSideRight[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(pathImu, vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageLeft[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first

        while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered
    }

    // Read rectification parameters
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_MULTI, true);

    // ===================================================================================
    // 新增：读取Tag保存/加载的开关
    // ===================================================================================
    bool bEnableTagStorage = false;
    cv::FileNode node = fsSettings["Tag.Enable"];
    if (!node.empty() && node.isInt())
    {
        bEnableTagStorage = (static_cast<int>(node) != 0);
    }
    if (bEnableTagStorage)
    {
        cout << "[System] INFO: Tag save/load is ENABLED in settings." << endl;
    }
    else
    {
        cout << "[System] INFO: Tag save/load is DISABLED in settings." << endl;
    }
    // ===================================================================================


    cv::Mat imLeft, imRight, imSideLeft, imSideRight;
    for (seq = 0; seq<num_seq; seq++)
    {
        // Seq loop
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        double t_rect = 0.f;
        double t_resize = 0.f;
        double t_track = 0.f;
        int num_rect = 0;
        int proccIm = 0;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            // ===================================================================================
            // 修改：使用新的健壮函数读取图像
            // ===================================================================================
            imLeft = imread_robust(vstrImageLeft[seq][ni], cv::IMREAD_UNCHANGED);
            imRight = imread_robust(vstrImageRight[seq][ni], cv::IMREAD_UNCHANGED);
            imSideLeft = imread_robust(vstrImageSideLeft[seq][ni], cv::IMREAD_UNCHANGED);
            imSideRight = imread_robust(vstrImageSideRight[seq][ni], cv::IMREAD_UNCHANGED);

            // ===================================================================================
            // 修改：调整错误处理逻辑，跳过损坏的帧而不是退出
            // ===================================================================================
            if(imLeft.empty() || imRight.empty() || imSideLeft.empty() || imSideRight.empty())
            {
                // imread_robust 内部已经打印了详细的错误信息
                cerr << "Error: Skipping frame " << ni << " for timestamp " << fixed << vTimestampsCam[seq][ni]
                     << " due to image loading failure after multiple retries." << endl;
                continue; // 跳过当前循环，处理下一帧
            }
            // ===================================================================================

            double tframe = vTimestampsCam[seq][ni];

            // Load imu measurements from previous frame
            vImuMeas.clear();

            if(ni>0)
                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni]) // while(vTimestampsImu[first_imu]<=vTimestampsCam[ni])
                {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x,vAcc[seq][first_imu[seq]].y,vAcc[seq][first_imu[seq]].z,
                                                             vGyro[seq][first_imu[seq]].x,vGyro[seq][first_imu[seq]].y,vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            // Pass the images to the SLAM system
            Sophus::SE3f Twb = SLAM.TrackMulti(imLeft, imRight, imSideLeft, imSideRight, tframe, vImuMeas);

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
    #endif

#ifdef REGISTER_TIMES
            t_track = t_rect + t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            vTimesTrack[ni]=ttrack;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];

            if(ttrack<T)
                usleep((T-ttrack)*1e6); // 1e6
        }

        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }


    }
    // Stop all threads
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    SLAM.Shutdown();

    if (bEnableTagStorage) // 修改：使用开关控制Tag的保存和统计逻辑
    {
        Eigen::Matrix3d aaaaa;
        Eigen::Vector3d bbbbb;
        double err_tagi;
        double err_all = 0;
        double count_all = 0;
        double err_avg;
        TagStorage::Instance().tagCleanup();
        cout << "-------" << endl;
        for (int i = 0; i <= 20; i++) {
        TagStorage::Instance().tagRead(i, aaaaa, bbbbb, err_tagi);
        cout << "tag" << i << "观测次数" << TagStorage::Instance().GetObservationCount(i) << endl;
        err_all = err_tagi * TagStorage::Instance().GetObservationCount(i) + err_all;
        count_all += TagStorage::Instance().GetObservationCount(i);
        }
        if (count_all > 0) {
            err_avg = err_all / count_all;
            cout << "平均观测误差为:" << err_avg << " m !!!" << endl;
        } else {
            cout << "没有Tag被观测到，无法计算平均误差。" << endl;
        }
        TagStorage::Instance().tagSave();
    }


    // Save camera trajectory
    if (bFileName)
    {
        const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    if(!vTimesTrack.empty())
    {
        std::sort(vTimesTrack.begin(), vTimesTrack.end());
        float totalTime = 0;
        for(float time : vTimesTrack)
        {
            totalTime += time;
        }
        float meanTime = totalTime / vTimesTrack.size();
        float medianTime = vTimesTrack[vTimesTrack.size() / 2];
        
        cout << "-------" << endl;
        cout << "Tracking time statistics:" << endl;
        cout << "Number of frames: " << vTimesTrack.size() << endl;
        cout << "Mean tracking time: " << meanTime * 1000 << " ms" << endl;
        cout << "Median tracking time: " << medianTime * 1000 << " ms" << endl;
    }

    return 0;
}

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathSideLeft, const string &strPathSideRight,
                     const string &strPathTimes, vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<string> &vstrImageSideLeft, vector<string> &vstrImageSideRight, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    vstrImageSideLeft.reserve(5000);
    vstrImageSideRight.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            vstrImageSideLeft.push_back(strPathSideLeft + "/" + ss.str() + ".png");
            vstrImageSideRight.push_back(strPathSideRight + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t);
        }
    }
}


void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    while(!fImu.eof())
    {
        string s;
        getline(fImu,s);
        if (s[0] == '#')
            continue;

        if(!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);

            vTimeStamps.push_back(data[0]);
            vAcc.push_back(cv::Point3f(data[4],data[5],data[6]));
            vGyro.push_back(cv::Point3f(data[1],data[2],data[3]));
        }
    }
}
