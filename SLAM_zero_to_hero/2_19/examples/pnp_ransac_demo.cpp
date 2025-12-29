/**
 * @file pnp_ransac_demo.cpp
 * @brief Robust PnP estimation with RANSAC for handling outliers
 *
 * This demo demonstrates the use of cv::solvePnPRansac for robust camera pose
 * estimation in the presence of outlier correspondences. This is essential
 * for real-world SLAM applications where feature matching often produces
 * incorrect correspondences.
 *
 * Key concepts:
 * - RANSAC (Random Sample Consensus) iteratively selects minimal subsets
 * - Each subset is used to compute a hypothesis pose
 * - Inliers are counted based on reprojection error threshold
 * - The hypothesis with most inliers is selected and refined
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nOptions:\n"
              << "  --points <n>          Number of 3D points (default: 50)\n"
              << "  --outliers <ratio>    Ratio of outliers 0-1 (default: 0.3)\n"
              << "  --noise <sigma>       Inlier noise in pixels (default: 1.0)\n"
              << "  --threshold <px>      RANSAC reprojection threshold (default: 8.0)\n"
              << "  --confidence <prob>   RANSAC confidence level (default: 0.99)\n"
              << "  --iterations <n>      Max RANSAC iterations (default: 100)\n"
              << "  --compare             Compare RANSAC vs standard PnP\n"
              << "  --verbose             Show detailed output\n"
              << "\nExamples:\n"
              << "  " << programName << " --points 100 --outliers 0.4\n"
              << "  " << programName << " --threshold 5.0 --confidence 0.999\n";
}

// Generate synthetic 3D points distributed in a cube
std::vector<cv::Point3f> generateObjectPoints(int numPoints) {
    std::vector<cv::Point3f> points;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (int i = 0; i < numPoints; i++) {
        points.emplace_back(dist(rng), dist(rng), dist(rng) + 2.0f);
    }

    return points;
}

// Project points and add noise/outliers
std::vector<cv::Point2f> projectPointsWithOutliers(
    const std::vector<cv::Point3f>& objectPoints,
    const cv::Mat& K, const cv::Mat& rvec, const cv::Mat& tvec,
    double noiseStd, double outlierRatio,
    std::vector<bool>& isOutlier) {

    // Project points
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(objectPoints, rvec, tvec, K, cv::Mat(), imagePoints);

    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> noise(0.0f, static_cast<float>(noiseStd));
    std::uniform_real_distribution<float> outlierDist(0.0f, 640.0f);
    std::bernoulli_distribution isOutlierDist(outlierRatio);

    isOutlier.resize(imagePoints.size());

    for (size_t i = 0; i < imagePoints.size(); i++) {
        isOutlier[i] = isOutlierDist(rng);

        if (isOutlier[i]) {
            // Random outlier position
            imagePoints[i].x = outlierDist(rng);
            imagePoints[i].y = outlierDist(rng) * 480.0f / 640.0f;
        } else {
            // Add Gaussian noise to inliers
            imagePoints[i].x += noise(rng);
            imagePoints[i].y += noise(rng);
        }
    }

    return imagePoints;
}

// Convert rotation vector to rotation matrix
cv::Mat rotationVectorToMatrix(const cv::Mat& rvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return R;
}

// Compute rotation error between two rotation matrices (in degrees)
double rotationError(const cv::Mat& R1, const cv::Mat& R2) {
    cv::Mat R_diff = R1 * R2.t();
    double trace = R_diff.at<double>(0, 0) + R_diff.at<double>(1, 1) + R_diff.at<double>(2, 2);
    double cos_angle = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
    return std::acos(cos_angle) * 180.0 / CV_PI;
}

// Compute translation error (Euclidean distance)
double translationError(const cv::Mat& t1, const cv::Mat& t2) {
    return cv::norm(t1 - t2);
}

// Compute reprojection error for all points
std::vector<double> computeReprojectionErrors(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints,
    const cv::Mat& K, const cv::Mat& rvec, const cv::Mat& tvec) {

    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(objectPoints, rvec, tvec, K, cv::Mat(), projectedPoints);

    std::vector<double> errors(imagePoints.size());
    for (size_t i = 0; i < imagePoints.size(); i++) {
        double dx = imagePoints[i].x - projectedPoints[i].x;
        double dy = imagePoints[i].y - projectedPoints[i].y;
        errors[i] = std::sqrt(dx * dx + dy * dy);
    }

    return errors;
}

struct PnPResult {
    bool success;
    cv::Mat rvec, tvec;
    cv::Mat inliers;
    double rotationError;
    double translationError;
    double meanReprojError;
    double timeMs;
    int numInliers;
};

// Test standard PnP (no RANSAC)
PnPResult testStandardPnP(const std::vector<cv::Point3f>& objectPoints,
                           const std::vector<cv::Point2f>& imagePoints,
                           const cv::Mat& K, const cv::Mat& distCoeffs,
                           const cv::Mat& R_gt, const cv::Mat& t_gt,
                           int method = cv::SOLVEPNP_EPNP) {
    PnPResult result;
    result.success = false;

    auto start = std::chrono::high_resolution_clock::now();

    result.success = cv::solvePnP(objectPoints, imagePoints, K, distCoeffs,
                                   result.rvec, result.tvec, false, method);

    auto end = std::chrono::high_resolution_clock::now();
    result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();

    if (result.success) {
        cv::Mat R_est = rotationVectorToMatrix(result.rvec);
        result.rotationError = rotationError(R_gt, R_est);
        result.translationError = translationError(t_gt, result.tvec);

        auto errors = computeReprojectionErrors(objectPoints, imagePoints, K,
                                                 result.rvec, result.tvec);
        result.meanReprojError = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
        result.numInliers = static_cast<int>(objectPoints.size());  // All points used
    }

    return result;
}

// Test RANSAC PnP
PnPResult testRansacPnP(const std::vector<cv::Point3f>& objectPoints,
                         const std::vector<cv::Point2f>& imagePoints,
                         const cv::Mat& K, const cv::Mat& distCoeffs,
                         const cv::Mat& R_gt, const cv::Mat& t_gt,
                         int maxIterations, double threshold, double confidence,
                         int method = cv::SOLVEPNP_EPNP) {
    PnPResult result;
    result.success = false;

    auto start = std::chrono::high_resolution_clock::now();

    result.success = cv::solvePnPRansac(
        objectPoints, imagePoints, K, distCoeffs,
        result.rvec, result.tvec,
        false,           // useExtrinsicGuess
        maxIterations,   // iterationsCount
        threshold,       // reprojectionError threshold
        confidence,      // confidence
        result.inliers,  // outputArray inliers
        method           // flags
    );

    auto end = std::chrono::high_resolution_clock::now();
    result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();

    if (result.success) {
        cv::Mat R_est = rotationVectorToMatrix(result.rvec);
        result.rotationError = rotationError(R_gt, R_est);
        result.translationError = translationError(t_gt, result.tvec);

        // Compute mean reprojection error for inliers only
        auto errors = computeReprojectionErrors(objectPoints, imagePoints, K,
                                                 result.rvec, result.tvec);
        double sumError = 0.0;
        for (int i = 0; i < result.inliers.rows; i++) {
            int idx = result.inliers.at<int>(i, 0);
            sumError += errors[idx];
        }
        result.meanReprojError = result.inliers.rows > 0 ? sumError / result.inliers.rows : 0;
        result.numInliers = result.inliers.rows;
    }

    return result;
}

// Print comparison results
void printComparison(const PnPResult& standard, const PnPResult& ransac,
                     int totalPoints, int trueInliers,
                     const std::vector<bool>& isOutlier,
                     const cv::Mat& ransacInliers) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "                    Standard PnP vs RANSAC PnP Comparison\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << std::left << std::setw(30) << "Metric"
              << std::setw(20) << "Standard PnP"
              << std::setw(20) << "RANSAC PnP" << "\n";
    std::cout << std::string(70, '-') << "\n";

    std::cout << std::setw(30) << "Success"
              << std::setw(20) << (standard.success ? "Yes" : "No")
              << std::setw(20) << (ransac.success ? "Yes" : "No") << "\n";

    if (standard.success || ransac.success) {
        std::cout << std::fixed << std::setprecision(4);

        std::cout << std::setw(30) << "Rotation Error (deg)"
                  << std::setw(20) << (standard.success ? std::to_string(standard.rotationError).substr(0, 8) : "-")
                  << std::setw(20) << (ransac.success ? std::to_string(ransac.rotationError).substr(0, 8) : "-") << "\n";

        std::cout << std::setw(30) << "Translation Error (m)"
                  << std::setw(20) << (standard.success ? std::to_string(standard.translationError).substr(0, 8) : "-")
                  << std::setw(20) << (ransac.success ? std::to_string(ransac.translationError).substr(0, 8) : "-") << "\n";

        std::cout << std::setw(30) << "Mean Reproj Error (px)"
                  << std::setw(20) << (standard.success ? std::to_string(standard.meanReprojError).substr(0, 8) : "-")
                  << std::setw(20) << (ransac.success ? std::to_string(ransac.meanReprojError).substr(0, 8) : "-") << "\n";

        std::cout << std::setw(30) << "Points Used"
                  << std::setw(20) << (standard.success ? std::to_string(totalPoints) : "-")
                  << std::setw(20) << (ransac.success ? std::to_string(ransac.numInliers) : "-") << "\n";

        std::cout << std::setw(30) << "Time (ms)"
                  << std::setw(20) << (standard.success ? std::to_string(standard.timeMs).substr(0, 8) : "-")
                  << std::setw(20) << (ransac.success ? std::to_string(ransac.timeMs).substr(0, 8) : "-") << "\n";
    }

    std::cout << std::string(70, '=') << "\n";

    // Analyze RANSAC inlier detection accuracy
    if (ransac.success && ransacInliers.rows > 0) {
        std::cout << "\n=== RANSAC Inlier Detection Analysis ===\n";
        std::cout << "Total points: " << totalPoints << "\n";
        std::cout << "True inliers: " << trueInliers << "\n";
        std::cout << "Detected inliers: " << ransacInliers.rows << "\n";

        // Check how many detected inliers are actually true inliers
        int correctInliers = 0;
        int falsePositives = 0;

        std::vector<bool> detectedAsInlier(totalPoints, false);
        for (int i = 0; i < ransacInliers.rows; i++) {
            int idx = ransacInliers.at<int>(i, 0);
            detectedAsInlier[idx] = true;

            if (!isOutlier[idx]) {
                correctInliers++;
            } else {
                falsePositives++;
            }
        }

        int missedInliers = 0;
        for (size_t i = 0; i < isOutlier.size(); i++) {
            if (!isOutlier[i] && !detectedAsInlier[i]) {
                missedInliers++;
            }
        }

        std::cout << "Correct inliers detected: " << correctInliers << "\n";
        std::cout << "False positives (outliers marked as inliers): " << falsePositives << "\n";
        std::cout << "Missed inliers: " << missedInliers << "\n";

        double precision = ransacInliers.rows > 0 ?
            static_cast<double>(correctInliers) / ransacInliers.rows : 0;
        double recall = trueInliers > 0 ?
            static_cast<double>(correctInliers) / trueInliers : 0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Precision: " << (precision * 100) << "%\n";
        std::cout << "Recall: " << (recall * 100) << "%\n";
    }
}

int main(int argc, char** argv) {
    // Default parameters
    int numPoints = 50;
    double outlierRatio = 0.3;
    double noiseStd = 1.0;
    double threshold = 8.0;
    double confidence = 0.99;
    int maxIterations = 100;
    bool compare = false;
    bool verbose = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--points" && i + 1 < argc) {
            numPoints = std::stoi(argv[++i]);
        } else if (arg == "--outliers" && i + 1 < argc) {
            outlierRatio = std::stod(argv[++i]);
        } else if (arg == "--noise" && i + 1 < argc) {
            noiseStd = std::stod(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::stod(argv[++i]);
        } else if (arg == "--confidence" && i + 1 < argc) {
            confidence = std::stod(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            maxIterations = std::stoi(argv[++i]);
        } else if (arg == "--compare") {
            compare = true;
        } else if (arg == "--verbose") {
            verbose = true;
        }
    }

    std::cout << "=== Robust PnP with RANSAC Demo ===\n\n";

    // Camera intrinsics
    const int imageWidth = 640;
    const int imageHeight = 480;
    const double fx = 500.0, fy = 500.0;
    const double cx = imageWidth / 2.0, cy = imageHeight / 2.0;

    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    // Ground truth pose
    cv::Mat rvec_gt = (cv::Mat_<double>(3, 1) << 0.1, 0.2, 0.05);
    cv::Mat tvec_gt = (cv::Mat_<double>(3, 1) << 0.1, -0.05, 2.0);
    cv::Mat R_gt = rotationVectorToMatrix(rvec_gt);

    // Print configuration
    std::cout << "Configuration:\n";
    std::cout << "  Number of points: " << numPoints << "\n";
    std::cout << "  Outlier ratio: " << (outlierRatio * 100) << "%\n";
    std::cout << "  Inlier noise: " << noiseStd << " pixels\n";
    std::cout << "  RANSAC threshold: " << threshold << " pixels\n";
    std::cout << "  RANSAC confidence: " << confidence << "\n";
    std::cout << "  Max iterations: " << maxIterations << "\n\n";

    // Generate test data
    std::vector<cv::Point3f> objectPoints = generateObjectPoints(numPoints);
    std::vector<bool> isOutlier;
    std::vector<cv::Point2f> imagePoints = projectPointsWithOutliers(
        objectPoints, K, rvec_gt, tvec_gt, noiseStd, outlierRatio, isOutlier);

    int trueInliers = std::count(isOutlier.begin(), isOutlier.end(), false);
    int trueOutliers = std::count(isOutlier.begin(), isOutlier.end(), true);

    std::cout << "Generated data:\n";
    std::cout << "  True inliers: " << trueInliers << "\n";
    std::cout << "  True outliers: " << trueOutliers << "\n\n";

    // Run RANSAC PnP
    std::cout << "Running RANSAC PnP...\n";
    PnPResult ransacResult = testRansacPnP(
        objectPoints, imagePoints, K, distCoeffs, R_gt, tvec_gt,
        maxIterations, threshold, confidence, cv::SOLVEPNP_EPNP);

    if (ransacResult.success) {
        std::cout << "\n=== RANSAC PnP Results ===\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Success: Yes\n";
        std::cout << "Inliers found: " << ransacResult.numInliers << "/" << numPoints << "\n";
        std::cout << "Rotation error: " << ransacResult.rotationError << " degrees\n";
        std::cout << "Translation error: " << ransacResult.translationError << " m\n";
        std::cout << "Mean reprojection error: " << ransacResult.meanReprojError << " pixels\n";
        std::cout << "Computation time: " << ransacResult.timeMs << " ms\n";

        if (verbose) {
            std::cout << "\nEstimated pose:\n";
            std::cout << "  rvec: [" << ransacResult.rvec.at<double>(0) << ", "
                      << ransacResult.rvec.at<double>(1) << ", "
                      << ransacResult.rvec.at<double>(2) << "]\n";
            std::cout << "  tvec: [" << ransacResult.tvec.at<double>(0) << ", "
                      << ransacResult.tvec.at<double>(1) << ", "
                      << ransacResult.tvec.at<double>(2) << "]\n";
        }
    } else {
        std::cout << "\nRANSAC PnP failed!\n";
    }

    // Compare with standard PnP if requested
    if (compare) {
        std::cout << "\nRunning standard PnP (no RANSAC)...\n";
        PnPResult standardResult = testStandardPnP(
            objectPoints, imagePoints, K, distCoeffs, R_gt, tvec_gt, cv::SOLVEPNP_EPNP);

        printComparison(standardResult, ransacResult, numPoints, trueInliers,
                        isOutlier, ransacResult.inliers);
    }

    // Print code example
    std::cout << "\n=== Code Example ===\n\n";
    std::cout << R"(
// Robust PnP with RANSAC in OpenCV
cv::Mat rvec, tvec, inliers;
bool success = cv::solvePnPRansac(
    objectPoints,       // 3D points in world frame
    imagePoints,        // 2D points in image
    cameraMatrix,       // 3x3 intrinsic matrix
    distCoeffs,         // Distortion coefficients
    rvec, tvec,         // Output rotation and translation
    false,              // useExtrinsicGuess
    100,                // iterationsCount
    8.0,                // reprojectionError threshold (pixels)
    0.99,               // confidence level
    inliers,            // Output inlier indices
    cv::SOLVEPNP_EPNP   // Method flag
);

// Process inliers
std::cout << "Found " << inliers.rows << " inliers" << std::endl;
for (int i = 0; i < inliers.rows; i++) {
    int idx = inliers.at<int>(i, 0);
    // idx is the index of an inlier point
}
)";

    std::cout << "\n=== Tips for RANSAC PnP ===\n\n";
    std::cout << "1. Threshold selection:\n";
    std::cout << "   - Depends on image noise and calibration accuracy\n";
    std::cout << "   - Typical values: 1-10 pixels for well-calibrated cameras\n";
    std::cout << "   - Lower threshold = stricter, fewer inliers, more accurate\n\n";

    std::cout << "2. Confidence level:\n";
    std::cout << "   - Probability of finding correct solution\n";
    std::cout << "   - Higher confidence = more iterations needed\n";
    std::cout << "   - 0.99 is a good default\n\n";

    std::cout << "3. Number of iterations:\n";
    std::cout << "   - Can be estimated: k = log(1-p) / log(1-(1-e)^n)\n";
    std::cout << "   - p=confidence, e=outlier ratio, n=minimal set size (4 for PnP)\n";
    std::cout << "   - With 30% outliers and 99% confidence: ~72 iterations\n\n";

    std::cout << "4. Method selection:\n";
    std::cout << "   - EPNP: Good general choice, fast and accurate\n";
    std::cout << "   - P3P/AP3P: Minimal solver (3 points), good for RANSAC\n";
    std::cout << "   - SQPNP: Best accuracy, slightly slower\n\n";

    std::cout << "Done.\n";
    return 0;
}
