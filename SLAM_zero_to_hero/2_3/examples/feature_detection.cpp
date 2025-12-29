/**
 * @file feature_detection.cpp
 * @brief Demonstrates FAST, ORB, and SIFT feature detection using OpenCV
 *
 * This example shows how to:
 * 1. Detect keypoints using different algorithms
 * 2. Compare detection speed and keypoint counts
 * 3. Visualize detected features
 *
 * Relevant for SLAM: Feature detection is the first step in visual odometry
 * and loop closure detection.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>

/**
 * @brief Generate a synthetic test image with features
 *
 * Creates an image with corners, edges, and texture - typical visual features
 * found in indoor/outdoor SLAM environments.
 */
cv::Mat generateTestImage(int width, int height, int seed = 42) {
    cv::Mat image(height, width, CV_8UC1);
    cv::RNG rng(seed);

    // Create gradient background (simulates illumination variation)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image.at<uchar>(y, x) = static_cast<uchar>((x + y) % 256);
        }
    }

    // Add rectangles (simulate windows, doors, buildings)
    for (int i = 0; i < 20; i++) {
        cv::Point pt1(rng.uniform(0, width - 50), rng.uniform(0, height - 50));
        cv::Point pt2(pt1.x + rng.uniform(20, 80), pt1.y + rng.uniform(20, 80));
        cv::rectangle(image, pt1, pt2, cv::Scalar(rng.uniform(50, 200)), -1);
        // Add border for corner features
        cv::rectangle(image, pt1, pt2, cv::Scalar(rng.uniform(0, 50)), 2);
    }

    // Add circles (simulate round objects, wheels, etc.)
    for (int i = 0; i < 15; i++) {
        cv::Point center(rng.uniform(30, width - 30), rng.uniform(30, height - 30));
        int radius = rng.uniform(10, 40);
        cv::circle(image, center, radius, cv::Scalar(rng.uniform(100, 255)), -1);
        cv::circle(image, center, radius, cv::Scalar(rng.uniform(0, 100)), 2);
    }

    // Add some noise (simulates sensor noise)
    cv::Mat noise(height, width, CV_8UC1);
    rng.fill(noise, cv::RNG::NORMAL, 0, 15);
    image += noise;

    // Apply slight blur (simulates camera blur)
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.5);

    return image;
}

/**
 * @brief Measure execution time of a function
 */
template <typename Func>
double measureTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief Detect features using FAST algorithm
 *
 * FAST is the fastest detector, commonly used in real-time SLAM (ORB-SLAM, PTAM).
 * It detects corner-like features by examining pixels on a circle.
 */
void detectFAST(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, int threshold = 20) {
    // Create FAST detector
    // Parameters:
    //   threshold: Difference threshold for corner detection (default: 10)
    //   nonmaxSuppression: Apply non-maximum suppression (default: true)
    //   type: FAST variant (TYPE_5_8, TYPE_7_12, TYPE_9_16)
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(
        threshold,
        true,  // nonmaxSuppression
        cv::FastFeatureDetector::TYPE_9_16  // Most common variant
    );

    detector->detect(image, keypoints);
}

/**
 * @brief Detect features and compute descriptors using ORB
 *
 * ORB = Oriented FAST + Rotated BRIEF
 * - Uses FAST for keypoint detection
 * - Computes 256-bit binary descriptor (rotation-invariant)
 * - Patent-free, widely used in SLAM
 */
void detectORB(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
               int nfeatures = 500) {
    // Create ORB detector
    // Parameters:
    //   nfeatures: Maximum number of features to retain
    //   scaleFactor: Pyramid decimation ratio (1.2 is typical)
    //   nlevels: Number of pyramid levels
    //   edgeThreshold: Border size where features are not detected
    //   firstLevel: Pyramid level to start from (usually 0)
    //   WTA_K: Points used for BRIEF descriptor (2, 3, or 4)
    //   scoreType: HARRIS_SCORE or FAST_SCORE for keypoint ranking
    //   patchSize: Size of the patch used for BRIEF descriptor
    cv::Ptr<cv::ORB> detector = cv::ORB::create(
        nfeatures,
        1.2f,   // scaleFactor
        8,      // nlevels
        31,     // edgeThreshold
        0,      // firstLevel
        2,      // WTA_K
        cv::ORB::HARRIS_SCORE,
        31,     // patchSize
        20      // fastThreshold
    );

    detector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
}

/**
 * @brief Detect features and compute descriptors using SIFT
 *
 * SIFT produces highly distinctive 128-dimensional float descriptors.
 * More robust than ORB but slower and memory-intensive.
 */
void detectSIFT(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                int nfeatures = 0) {
    // Create SIFT detector
    // Parameters:
    //   nfeatures: Number of best features to retain (0 = all)
    //   nOctaveLayers: Number of layers in each octave (3 is default)
    //   contrastThreshold: Contrast threshold for filtering weak features
    //   edgeThreshold: Threshold for filtering edge-like features
    //   sigma: Initial Gaussian sigma
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(
        nfeatures,
        3,      // nOctaveLayers
        0.04,   // contrastThreshold
        10,     // edgeThreshold
        1.6     // sigma
    );

    detector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
}

/**
 * @brief Draw keypoints on image with additional info
 */
cv::Mat visualizeKeypoints(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints,
                           const std::string& title) {
    cv::Mat output;
    cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);

    // Draw keypoints with rich information (size, orientation)
    cv::drawKeypoints(image, keypoints, output, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Add title and count
    std::string text = title + " (" + std::to_string(keypoints.size()) + " keypoints)";
    cv::putText(output, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 255), 2);

    return output;
}

/**
 * @brief Print detection statistics
 */
void printStats(const std::string& name, size_t count, double time_ms,
                const cv::Mat& descriptors = cv::Mat()) {
    std::cout << std::left << std::setw(10) << name << " | "
              << std::right << std::setw(6) << count << " keypoints | "
              << std::fixed << std::setprecision(2) << std::setw(8) << time_ms << " ms";

    if (!descriptors.empty()) {
        std::cout << " | Descriptor: " << descriptors.cols << " dim, "
                  << (descriptors.type() == CV_32F ? "float" : "binary");
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Local Feature Detection Demo" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << std::endl;

    // Generate or load test image
    cv::Mat image;
    if (argc > 1) {
        image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Error: Could not load image: " << argv[1] << std::endl;
            return 1;
        }
        std::cout << "Loaded image: " << argv[1] << std::endl;
    } else {
        std::cout << "Generating synthetic test image (640x480)..." << std::endl;
        image = generateTestImage(640, 480, 42);
    }
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << std::endl;

    // Containers for results
    std::vector<cv::KeyPoint> keypoints_fast, keypoints_orb, keypoints_sift;
    cv::Mat descriptors_orb, descriptors_sift;
    double time_fast, time_orb, time_sift;

    // ===== FAST Detection =====
    time_fast = measureTime([&]() {
        detectFAST(image, keypoints_fast, 20);
    });

    // ===== ORB Detection =====
    time_orb = measureTime([&]() {
        detectORB(image, keypoints_orb, descriptors_orb, 500);
    });

    // ===== SIFT Detection =====
    time_sift = measureTime([&]() {
        detectSIFT(image, keypoints_sift, descriptors_sift, 500);
    });

    // Print comparison
    std::cout << "Detection Results:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    printStats("FAST", keypoints_fast.size(), time_fast);
    printStats("ORB", keypoints_orb.size(), time_orb, descriptors_orb);
    printStats("SIFT", keypoints_sift.size(), time_sift, descriptors_sift);
    std::cout << std::endl;

    // SLAM Relevance
    std::cout << "SLAM Recommendations:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "- Real-time VO (30+ FPS): Use ORB or FAST+BRIEF" << std::endl;
    std::cout << "- Loop Closure: Consider SIFT for better distinctiveness" << std::endl;
    std::cout << "- Mobile/Embedded: ORB with reduced features (200-300)" << std::endl;
    std::cout << std::endl;

    // Visualize results
    cv::Mat vis_fast = visualizeKeypoints(image, keypoints_fast, "FAST");
    cv::Mat vis_orb = visualizeKeypoints(image, keypoints_orb, "ORB");
    cv::Mat vis_sift = visualizeKeypoints(image, keypoints_sift, "SIFT");

    // Create combined visualization
    cv::Mat combined;
    cv::hconcat(std::vector<cv::Mat>{vis_fast, vis_orb, vis_sift}, combined);

    // Resize for display if too large
    if (combined.cols > 1920) {
        double scale = 1920.0 / combined.cols;
        cv::resize(combined, combined, cv::Size(), scale, scale);
    }

    // Show and save results
    cv::imshow("Feature Detection Comparison", combined);
    cv::imwrite("feature_detection_result.png", combined);
    std::cout << "Result saved to: feature_detection_result.png" << std::endl;
    std::cout << "Press any key to exit..." << std::endl;

    cv::waitKey(0);

    return 0;
}
