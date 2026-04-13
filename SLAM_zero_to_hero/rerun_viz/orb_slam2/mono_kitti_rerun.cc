/**
 * Patched mono_kitti.cc - streams real-time SLAM state to stdout as JSON lines
 * for an external Rerun bridge to consume. Emits:
 *
 *   {"t":"pose",  "ts":..., "tx,ty,tz","qx,qy,qz,qw"}         every frame
 *   {"t":"kpts",  "ts":..., "uv":[...]}                       every frame
 *                     (separate arrays "uv_good" for keypoints matched to a
 *                      map point, and "uv_raw" for unmatched)
 *   {"t":"tmpts", "ts":..., "xyz":[...]}                      tracked map points
 *   {"t":"lmpts", "ts":..., "xyz":[...]}                      local (reference) map
 *   {"t":"ampts", "ts":..., "xyz":[...]}                      all map points (every N frames)
 *
 * Based on ORB_SLAM2/Examples/Monocular/mono_kitti.cc (upstream).
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <unordered_set>

#include <opencv2/core/core.hpp>

#include <System.h>
#include <Map.h>
#include <MapPoint.h>
#include <KeyFrame.h>

using namespace std;

void LoadImages(const string &strPathToSequence,
                vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

// ---- Pose serialization ----
static void PrintPose(double tframe, const cv::Mat& Tcw)
{
    if (Tcw.empty()) return;

    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    cv::Mat twc = -Rwc * tcw;

    auto R = [&](int i, int j){ return Rwc.at<float>(i,j); };
    float tx = twc.at<float>(0);
    float ty = twc.at<float>(1);
    float tz = twc.at<float>(2);

    float qw, qx, qy, qz;
    float trace = R(0,0) + R(1,1) + R(2,2);
    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        qw = 0.25f * s;
        qx = (R(2,1) - R(1,2)) / s;
        qy = (R(0,2) - R(2,0)) / s;
        qz = (R(1,0) - R(0,1)) / s;
    } else if ((R(0,0) > R(1,1)) && (R(0,0) > R(2,2))) {
        float s = sqrtf(1.0f + R(0,0) - R(1,1) - R(2,2)) * 2.0f;
        qw = (R(2,1) - R(1,2)) / s;
        qx = 0.25f * s;
        qy = (R(0,1) + R(1,0)) / s;
        qz = (R(0,2) + R(2,0)) / s;
    } else if (R(1,1) > R(2,2)) {
        float s = sqrtf(1.0f + R(1,1) - R(0,0) - R(2,2)) * 2.0f;
        qw = (R(0,2) - R(2,0)) / s;
        qx = (R(0,1) + R(1,0)) / s;
        qy = 0.25f * s;
        qz = (R(1,2) + R(2,1)) / s;
    } else {
        float s = sqrtf(1.0f + R(2,2) - R(0,0) - R(1,1)) * 2.0f;
        qw = (R(1,0) - R(0,1)) / s;
        qx = (R(0,2) + R(2,0)) / s;
        qy = (R(1,2) + R(2,1)) / s;
        qz = 0.25f * s;
    }

    printf("{\"t\":\"pose\",\"ts\":%.9f,\"tx\":%.6f,\"ty\":%.6f,\"tz\":%.6f,"
           "\"qx\":%.9f,\"qy\":%.9f,\"qz\":%.9f,\"qw\":%.9f}\n",
           tframe, tx, ty, tz, qx, qy, qz, qw);
}

// Print keypoints split into "matched" (have associated map point) and "raw"
static void PrintKeypoints(double tframe,
                           const vector<cv::KeyPoint>& kps,
                           const vector<ORB_SLAM2::MapPoint*>& mps)
{
    // matched keypoints (has non-null non-bad MapPoint)
    printf("{\"t\":\"kpts\",\"ts\":%.9f,\"uv_matched\":[", tframe);
    bool first = true;
    size_t n = std::min(kps.size(), mps.size());
    for (size_t i = 0; i < n; ++i) {
        if (mps[i] && !mps[i]->isBad()) {
            if (!first) printf(",");
            printf("[%.1f,%.1f]", kps[i].pt.x, kps[i].pt.y);
            first = false;
        }
    }
    printf("],\"uv_raw\":[");
    first = true;
    for (size_t i = 0; i < kps.size(); ++i) {
        bool matched = (i < mps.size() && mps[i] && !mps[i]->isBad());
        if (!matched) {
            if (!first) printf(",");
            printf("[%.1f,%.1f]", kps[i].pt.x, kps[i].pt.y);
            first = false;
        }
    }
    printf("]}\n");
}

static void PrintMapPointsArray(const char* tag, double tframe,
                                const vector<ORB_SLAM2::MapPoint*>& mps)
{
    printf("{\"t\":\"%s\",\"ts\":%.9f,\"xyz\":[", tag, tframe);
    bool first = true;
    for (auto* mp : mps) {
        if (!mp || mp->isBad()) continue;
        cv::Mat wp = mp->GetWorldPos();
        if (wp.empty()) continue;
        if (!first) printf(",");
        printf("[%.4f,%.4f,%.4f]",
               wp.at<float>(0), wp.at<float>(1), wp.at<float>(2));
        first = false;
    }
    printf("]}\n");
}

int main(int argc, char **argv)
{
    if(argc != 4) {
        cerr << "Usage: ./mono_kitti_rerun path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();

    // Viewer disabled (false) - headless for live streaming
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, false);

    vector<float> vTimesTrack(nImages);

    cout << "-------" << endl << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];
        if(im.empty()) {
            cerr << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

        auto t1 = std::chrono::steady_clock::now();
        cv::Mat Tcw = SLAM.TrackMonocular(im, tframe);
        auto t2 = std::chrono::steady_clock::now();

        // --- Stream SLAM state ---
        PrintPose(tframe, Tcw);

        vector<cv::KeyPoint> kps = SLAM.GetTrackedKeyPointsUn();
        vector<ORB_SLAM2::MapPoint*> tracked_mps = SLAM.GetTrackedMapPoints();
        PrintKeypoints(tframe, kps, tracked_mps);
        PrintMapPointsArray("tmpts", tframe, tracked_mps);

        // Map access (requires Map* getter patch to System.h)
        ORB_SLAM2::Map* pMap = SLAM.GetMap();
        if (pMap) {
            auto local_mps = pMap->GetReferenceMapPoints();
            PrintMapPointsArray("lmpts", tframe, local_mps);

            // Send full map every 20 frames (expensive)
            if (ni % 20 == 0) {
                auto all_mps = pMap->GetAllMapPoints();
                PrintMapPointsArray("ampts", tframe, all_mps);
            }
        }

        fflush(stdout);

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        double T=0;
        if(ni<nImages-1) T = vTimestamps[ni+1]-tframe;
        else if(ni>0) T = tframe-vTimestamps[ni-1];
        if(ttrack<T) usleep((T-ttrack)*1e6);
    }

    SLAM.Shutdown();

    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++) totaltime += vTimesTrack[ni];
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof()) {
        string s;
        getline(fTimes,s);
        if(!s.empty()) {
            stringstream ss; ss << s;
            double t; ss >> t;
            vTimestamps.push_back(t);
        }
    }
    string strPrefixLeft = strPathToSequence + "/image_0/";
    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);
    for(int i=0; i<nTimes; i++) {
        stringstream ss; ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}
