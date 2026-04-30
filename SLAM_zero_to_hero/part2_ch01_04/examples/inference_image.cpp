//
// SuperPoint + SuperGlue Inference on Image Pair
//
// This example demonstrates:
// 1. Loading configuration and building TensorRT engines
// 2. Running SuperPoint to extract features from two images
// 3. Running SuperGlue to match features between images
// 4. Visualizing the matches
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

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> <image1> <image2>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  " << argv[0] << " config/config.yaml data/img1.png data/img2.png" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string image0_path = argv[2];
    std::string image1_path = argv[3];

    // ===========================================================================
    // Load Configuration
    // ===========================================================================

    std::string model_dir = "weights";
    Configs configs(config_path, model_dir);

    int width = configs.superglue_config.image_width;
    int height = configs.superglue_config.image_height;

    std::cout << "Image size: " << width << " x " << height << std::endl;

    // ===========================================================================
    // Build TensorRT Engines
    // ===========================================================================

    std::cout << "\n=== Building SuperPoint Engine ===" << std::endl;
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()) {
        std::cerr << "Error: Failed to build SuperPoint engine." << std::endl;
        std::cerr << "Please check your ONNX model path: "
                  << configs.superpoint_config.onnx_file << std::endl;
        return 1;
    }

    std::cout << "\n=== Building SuperGlue Engine ===" << std::endl;
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()) {
        std::cerr << "Error: Failed to build SuperGlue engine." << std::endl;
        std::cerr << "Please check your ONNX model path: "
                  << configs.superglue_config.onnx_file << std::endl;
        return 1;
    }

    std::cout << "\n=== Engines Built Successfully ===" << std::endl;

    // ===========================================================================
    // Load and Preprocess Images
    // ===========================================================================

    cv::Mat image0 = cv::imread(image0_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);

    if (image0.empty()) {
        std::cerr << "Error: Failed to load image: " << image0_path << std::endl;
        return 1;
    }
    if (image1.empty()) {
        std::cerr << "Error: Failed to load image: " << image1_path << std::endl;
        return 1;
    }

    // Resize to configured dimensions
    cv::resize(image0, image0, cv::Size(width, height));
    cv::resize(image1, image1, cv::Size(width, height));

    std::cout << "\nImage 0: " << image0_path << " (" << image0.cols << "x" << image0.rows << ")" << std::endl;
    std::cout << "Image 1: " << image1_path << " (" << image1.cols << "x" << image1.rows << ")" << std::endl;

    // ===========================================================================
    // Run Inference
    // ===========================================================================

    std::cout << "\n=== Running Inference ===" << std::endl;

    Eigen::Matrix<double, 259, Eigen::Dynamic> features0, features1;
    std::vector<cv::DMatch> matches;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Extract features with SuperPoint
    std::cout << "Extracting features from image 0..." << std::endl;
    if (!superpoint->infer(image0, features0)) {
        std::cerr << "Error: SuperPoint inference failed on image 0." << std::endl;
        return 1;
    }
    std::cout << "  Found " << features0.cols() << " keypoints" << std::endl;

    std::cout << "Extracting features from image 1..." << std::endl;
    if (!superpoint->infer(image1, features1)) {
        std::cerr << "Error: SuperPoint inference failed on image 1." << std::endl;
        return 1;
    }
    std::cout << "  Found " << features1.cols() << " keypoints" << std::endl;

    // Match features with SuperGlue
    std::cout << "Matching features with SuperGlue..." << std::endl;
    int num_matches = superglue->matching_points(features0, features1, matches, true);
    std::cout << "  Found " << num_matches << " matches" << std::endl;

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\nTotal inference time: " << duration.count() << " ms" << std::endl;
    std::cout << "  SuperPoint x2 + SuperGlue + RANSAC" << std::endl;

    // ===========================================================================
    // Visualize Results
    // ===========================================================================

    // Convert features to OpenCV keypoints
    std::vector<cv::KeyPoint> keypoints0, keypoints1;

    for (Eigen::Index i = 0; i < features0.cols(); ++i) {
        float score = static_cast<float>(features0(0, i));
        float x = static_cast<float>(features0(1, i));
        float y = static_cast<float>(features0(2, i));
        keypoints0.emplace_back(x, y, 8, -1, score);
    }

    for (Eigen::Index i = 0; i < features1.cols(); ++i) {
        float score = static_cast<float>(features1(0, i));
        float x = static_cast<float>(features1(1, i));
        float y = static_cast<float>(features1(2, i));
        keypoints1.emplace_back(x, y, 8, -1, score);
    }

    // Draw and display matches
    cv::Mat match_image;
    VisualizeMatching(image0, keypoints0, image1, keypoints1, matches, match_image, duration.count());

    // Save result
    cv::imwrite("matches_output.jpg", match_image);
    std::cout << "\nSaved result to matches_output.jpg" << std::endl;

    // Wait for key press
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);

    return 0;
}
