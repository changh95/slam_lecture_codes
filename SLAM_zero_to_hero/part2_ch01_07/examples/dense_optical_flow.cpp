/**
 * Dense Optical Flow using Farneback Method
 *
 * Live demo: streams Farneback dense optical flow between consecutive frames
 * of a TUM RGB-D sequence into a cv::imshow window (arrow overlay | HSV side
 * by side).
 *
 * Usage:
 *   dense_optical_flow [seq_dir] [num_frames] [frame_gap]
 *
 *   seq_dir     - path to a TUM RGB-D sequence containing rgb/*.png + rgb.txt
 *                 (default: /data/tum_rgbd/rgbd_dataset_freiburg1_desk)
 *   num_frames  - number of frames to stream through (default: 500)
 *   frame_gap   - index gap between the pair fed to Farneback (default: 5)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

struct FarnebackConfig {
    double pyr_scale = 0.5;
    int levels = 3;
    int winsize = 15;
    int iterations = 3;
    int poly_n = 5;
    double poly_sigma = 1.1;
    int flags = 0;
};

static std::vector<std::string> loadTumRgbList(const std::string& seq_dir, size_t limit) {
    std::vector<std::string> frames;
    fs::path rgb_txt = fs::path(seq_dir) / "rgb.txt";
    if (fs::exists(rgb_txt)) {
        std::ifstream in(rgb_txt);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string ts, rel;
            if (!(iss >> ts >> rel)) continue;
            frames.push_back((fs::path(seq_dir) / rel).string());
            if (frames.size() >= limit) break;
        }
    } else {
        fs::path rgb_dir = fs::path(seq_dir) / "rgb";
        for (const auto& entry : fs::directory_iterator(rgb_dir)) {
            if (entry.path().extension() == ".png") {
                frames.push_back(entry.path().string());
            }
        }
        std::sort(frames.begin(), frames.end());
        if (frames.size() > limit) frames.resize(limit);
    }
    return frames;
}

static cv::Mat flowToHSV(const cv::Mat& flow) {
    std::vector<cv::Mat> parts(2);
    cv::split(flow, parts);
    cv::Mat magnitude, angle;
    cv::cartToPolar(parts[0], parts[1], magnitude, angle, true);

    double maxMag = 0.0;
    cv::minMaxLoc(magnitude, nullptr, &maxMag);
    if (maxMag > 0) magnitude /= maxMag;

    std::vector<cv::Mat> planes(3);
    angle.convertTo(planes[0], CV_8U, 0.5);
    planes[1] = cv::Mat::ones(flow.size(), CV_8U) * 255;
    magnitude.convertTo(planes[2], CV_8U, 255);

    cv::Mat hsv, bgr;
    cv::merge(planes, hsv);
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}

static cv::Mat flowToArrows(const cv::Mat& flow, const cv::Mat& background,
                            int step = 16, double scale = 2.0) {
    cv::Mat vis;
    if (background.channels() == 1) cv::cvtColor(background, vis, cv::COLOR_GRAY2BGR);
    else background.copyTo(vis);

    for (int y = step / 2; y < flow.rows; y += step) {
        for (int x = step / 2; x < flow.cols; x += step) {
            cv::Point2f f = flow.at<cv::Point2f>(y, x);
            cv::Point2f from(x, y);
            cv::Point2f to(x + f.x * scale, y + f.y * scale);
            double a = std::atan2(f.y, f.x) * 180.0 / CV_PI + 180.0;
            int hue = static_cast<int>(a / 2);
            cv::arrowedLine(vis, from, to,
                            cv::Scalar(hue * 1.4, 255 - hue, 100 + hue % 155),
                            1, cv::LINE_AA, 0, 0.3);
        }
    }
    return vis;
}

int main(int argc, char** argv) {
    std::string seq_dir = (argc > 1) ? argv[1]
        : "/data/tum_rgbd/rgbd_dataset_freiburg1_desk";
    int num_frames = (argc > 2) ? std::stoi(argv[2]) : 500;
    int frame_gap = (argc > 3) ? std::stoi(argv[3]) : 5;

    std::cout << "Dense Farneback live demo - OpenCV " << CV_VERSION << std::endl;
    std::cout << "Sequence:  " << seq_dir << std::endl;
    std::cout << "Frames:    " << num_frames << " (gap=" << frame_gap << ")" << std::endl;

    auto frames = loadTumRgbList(seq_dir, static_cast<size_t>(num_frames));
    if (frames.size() < static_cast<size_t>(frame_gap + 1)) {
        std::cerr << "Sequence too short for frame_gap=" << frame_gap << std::endl;
        return 1;
    }
    std::cout << "Loaded " << frames.size() << " frames" << std::endl;

    FarnebackConfig fb;

    cv::namedWindow("Dense Farneback (arrows | HSV)", cv::WINDOW_AUTOSIZE);

    for (size_t i = frame_gap; i < frames.size(); ++i) {
        cv::Mat bgr_prev = cv::imread(frames[i - frame_gap], cv::IMREAD_COLOR);
        cv::Mat bgr_curr = cv::imread(frames[i], cv::IMREAD_COLOR);
        if (bgr_prev.empty() || bgr_curr.empty()) continue;

        cv::Mat gp, gc;
        cv::cvtColor(bgr_prev, gp, cv::COLOR_BGR2GRAY);
        cv::cvtColor(bgr_curr, gc, cv::COLOR_BGR2GRAY);

        cv::Mat flow;
        auto t0 = std::chrono::high_resolution_clock::now();
        cv::calcOpticalFlowFarneback(gp, gc, flow,
                                     fb.pyr_scale, fb.levels, fb.winsize,
                                     fb.iterations, fb.poly_n, fb.poly_sigma,
                                     fb.flags);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cv::Mat hsv = flowToHSV(flow);
        cv::Mat arrows = flowToArrows(flow, bgr_curr, 16, 2.0);

        std::ostringstream tag;
        tag << "frame " << std::setw(4) << std::setfill('0') << i
            << "  " << std::fixed << std::setprecision(1) << ms << " ms";
        cv::putText(arrows, tag.str(), cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(hsv, tag.str(), cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        cv::Mat side_by_side;
        cv::hconcat(arrows, hsv, side_by_side);
        cv::imshow("Dense Farneback (arrows | HSV)", side_by_side);
        if ((cv::waitKey(1) & 0xff) == 27) break;

        if (i % 50 == 0 || i == frames.size() - 1) {
            std::cout << "Frame " << std::setw(4) << i
                      << ": flow time " << std::fixed << std::setprecision(2)
                      << ms << " ms" << std::endl;
        }
    }

    std::cout << "Press any key in the window to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
