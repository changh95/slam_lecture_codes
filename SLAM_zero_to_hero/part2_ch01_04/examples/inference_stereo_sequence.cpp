//
// SuperPoint + SuperGlue Stereo Sequence Inference
//
// At each timestep i, matches cam0[i] against cam1[i] (left-right stereo)
// rather than consecutive frames in a single stream.
//
// Usage: superpointglue_stereo_sequence <config.yaml> <cam0_dir> <cam1_dir> [max_frames]
//

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "utils.h"
#include "super_glue.h"
#include "super_point.h"

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <config.yaml> <cam0_dir> <cam1_dir> [max_frames]" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string cam0_dir = argv[2];
    std::string cam1_dir = argv[3];
    int max_frames = (argc == 5) ? std::atoi(argv[4]) : 0;

    std::string model_dir = "weights";
    Configs configs(config_path, model_dir);
    int width = configs.superglue_config.image_width;
    int height = configs.superglue_config.image_height;

    std::vector<std::string> cam0_names, cam1_names;
    GetFileNames(cam0_dir, cam0_names);
    GetFileNames(cam1_dir, cam1_names);

    size_t pair_count = std::min(cam0_names.size(), cam1_names.size());
    if (max_frames > 0 && static_cast<size_t>(max_frames) < pair_count) {
        pair_count = max_frames;
    }
    if (pair_count < 1) {
        std::cerr << "Error: Need at least 1 frame per camera." << std::endl;
        return 1;
    }

    std::cout << "Stereo pairs to process: " << pair_count << std::endl;
    std::cout << "Network input size:      " << width << " x " << height << std::endl;

    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()) {
        std::cerr << "Error: Failed to build SuperPoint engine." << std::endl;
        return 1;
    }
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()) {
        std::cerr << "Error: Failed to build SuperGlue engine." << std::endl;
        return 1;
    }

    std::vector<double> times;
    std::vector<int> match_counts;

    for (size_t idx = 0; idx < pair_count; ++idx) {
        cv::Mat left  = cv::imread(cam0_names[idx], cv::IMREAD_GRAYSCALE);
        cv::Mat right = cv::imread(cam1_names[idx], cv::IMREAD_GRAYSCALE);

        if (left.empty() || right.empty()) {
            std::cerr << "\nWarning: empty image at index " << idx
                      << ", skipping." << std::endl;
            continue;
        }

        cv::resize(left,  left,  cv::Size(width, height));
        cv::resize(right, right, cv::Size(width, height));

        Eigen::Matrix<double, 259, Eigen::Dynamic> feat_l, feat_r;
        std::vector<cv::DMatch> matches;

        auto t0 = std::chrono::high_resolution_clock::now();
        if (!superpoint->infer(left,  feat_l) ||
            !superpoint->infer(right, feat_r)) {
            std::cerr << "\nWarning: feature extraction failed, skipping." << std::endl;
            continue;
        }
        superglue->matching_points(feat_l, feat_r, matches);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        times.push_back(ms);
        match_counts.push_back(static_cast<int>(matches.size()));

        std::cout << "\r[" << (idx + 1) << "/" << pair_count << "] "
                  << matches.size() << " matches, " << ms << " ms      "
                  << std::flush;

        std::vector<cv::KeyPoint> kp_l, kp_r;
        kp_l.reserve(feat_l.cols());
        kp_r.reserve(feat_r.cols());
        for (Eigen::Index i = 0; i < feat_l.cols(); ++i) {
            kp_l.emplace_back(static_cast<float>(feat_l(1, i)),
                              static_cast<float>(feat_l(2, i)),
                              8, -1, static_cast<float>(feat_l(0, i)));
        }
        for (Eigen::Index i = 0; i < feat_r.cols(); ++i) {
            kp_r.emplace_back(static_cast<float>(feat_r(1, i)),
                              static_cast<float>(feat_r(2, i)),
                              8, -1, static_cast<float>(feat_r(0, i)));
        }

        cv::Mat match_image;
        VisualizeMatching(left, kp_l, right, kp_r, matches, match_image, ms);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            std::cout << "\nQuit requested." << std::endl;
            break;
        }
    }
    std::cout << std::endl;

    if (!times.empty()) {
        double sum_time = 0, sum_matches = 0;
        for (size_t i = 0; i < times.size(); ++i) {
            sum_time += times[i];
            sum_matches += match_counts[i];
        }
        double n = static_cast<double>(times.size());
        std::cout << "\n=== Stereo Statistics ===" << std::endl;
        std::cout << "Pairs processed:     " << times.size() << std::endl;
        std::cout << "Avg time per pair:   " << (sum_time / n) << " ms" << std::endl;
        std::cout << "Avg matches per pair:" << (sum_matches / n) << std::endl;
        std::cout << "Estimated FPS:       " << (1000.0 / (sum_time / n)) << std::endl;
    }

    cv::destroyAllWindows();
    return 0;
}
