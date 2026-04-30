//
// SuperPoint + SuperGlue Inference on Image Sequence
//
// This example demonstrates processing a sequence of images,
// similar to how learned features would be used in a visual odometry
// or SLAM system front-end.
//
// For each consecutive pair of images:
// 1. Extract SuperPoint features from both images
// 2. Match with SuperGlue
// 3. Visualize the matches
//

#include <memory>
#include <chrono>
#include <iostream>

#include "utils.h"
#include "super_glue.h"
#include "super_point.h"

int main(int argc, char** argv) {
    // ===========================================================================
    // Parse Command Line Arguments
    // ===========================================================================

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> <image_folder>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  " << argv[0] << " config/config.yaml /path/to/kitti/sequences/00/image_0/" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string image_folder = argv[2];

    // ===========================================================================
    // Load Configuration and Image List
    // ===========================================================================

    std::string model_dir = "weights";
    Configs configs(config_path, model_dir);

    int width = configs.superglue_config.image_width;
    int height = configs.superglue_config.image_height;

    std::vector<std::string> image_names;
    GetFileNames(image_folder, image_names);

    if (image_names.size() < 2) {
        std::cerr << "Error: Need at least 2 images in folder." << std::endl;
        return 1;
    }

    std::cout << "Processing " << image_names.size() - 1 << " image pairs" << std::endl;
    std::cout << "Image size: " << width << " x " << height << std::endl;

    // ===========================================================================
    // Build TensorRT Engines
    // ===========================================================================

    std::cout << "\n=== Building Inference Engines ===" << std::endl;

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

    std::cout << "SuperPoint and SuperGlue engines built successfully.\n" << std::endl;

    // ===========================================================================
    // Process Image Sequence
    // ===========================================================================

    // Statistics
    std::vector<double> times;
    std::vector<int> match_counts;

    for (size_t idx = 1; idx < image_names.size(); ++idx) {
        std::cout << "\r[" << idx << "/" << image_names.size() - 1 << "] "
                  << image_names[idx] << std::flush;

        // Load images
        cv::Mat image0 = cv::imread(image_names[idx - 1], cv::IMREAD_GRAYSCALE);
        cv::Mat image1 = cv::imread(image_names[idx], cv::IMREAD_GRAYSCALE);

        if (image0.empty() || image1.empty()) {
            std::cerr << "\nWarning: Failed to load images, skipping." << std::endl;
            continue;
        }

        // Resize to configured dimensions
        cv::resize(image0, image0, cv::Size(width, height));
        cv::resize(image1, image1, cv::Size(width, height));

        // Extract features and match
        Eigen::Matrix<double, 259, Eigen::Dynamic> features0, features1;
        std::vector<cv::DMatch> matches;

        auto start = std::chrono::high_resolution_clock::now();

        // SuperPoint feature extraction
        if (!superpoint->infer(image0, features0) ||
            !superpoint->infer(image1, features1)) {
            std::cerr << "\nWarning: Feature extraction failed, skipping." << std::endl;
            continue;
        }

        // SuperGlue matching
        superglue->matching_points(features0, features1, matches);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        times.push_back(duration.count());
        match_counts.push_back(matches.size());

        // Convert to OpenCV keypoints for visualization
        std::vector<cv::KeyPoint> keypoints0, keypoints1;

        for (Eigen::Index i = 0; i < features0.cols(); ++i) {
            keypoints0.emplace_back(
                static_cast<float>(features0(1, i)),
                static_cast<float>(features0(2, i)),
                8, -1, static_cast<float>(features0(0, i)));
        }

        for (Eigen::Index i = 0; i < features1.cols(); ++i) {
            keypoints1.emplace_back(
                static_cast<float>(features1(1, i)),
                static_cast<float>(features1(2, i)),
                8, -1, static_cast<float>(features1(0, i)));
        }

        // Visualize
        cv::Mat match_image;
        VisualizeMatching(image0, keypoints0, image1, keypoints1,
                         matches, match_image, duration.count());

        // Check for quit key
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // 'q' or ESC
            std::cout << "\nQuit requested." << std::endl;
            break;
        }
    }

    std::cout << std::endl;

    // ===========================================================================
    // Print Statistics
    // ===========================================================================

    if (!times.empty()) {
        double avg_time = 0, avg_matches = 0;
        for (size_t i = 0; i < times.size(); ++i) {
            avg_time += times[i];
            avg_matches += match_counts[i];
        }
        avg_time /= times.size();
        avg_matches /= times.size();

        std::cout << "\n=== Statistics ===" << std::endl;
        std::cout << "Processed " << times.size() << " image pairs" << std::endl;
        std::cout << "Average time per pair: " << avg_time << " ms" << std::endl;
        std::cout << "Average matches per pair: " << avg_matches << std::endl;
        std::cout << "Estimated FPS: " << 1000.0 / avg_time << std::endl;
    }

    cv::destroyAllWindows();
    return 0;
}
