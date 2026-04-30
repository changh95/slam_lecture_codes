/**
 * @file pnp_demo.cpp
 * @brief Demonstration and comparison of different PnP algorithms in OpenCV
 *
 * This demo shows how to estimate camera pose from 2D-3D point correspondences
 * using various PnP (Perspective-n-Point) algorithms available in OpenCV.
 * The algorithms are compared in terms of accuracy and computation time.
 *
 * PnP Methods Compared:
 * - SOLVEPNP_ITERATIVE: Levenberg-Marquardt refinement (default)
 * - SOLVEPNP_P3P: 3-point minimal solver (Kneip)
 * - SOLVEPNP_AP3P: Algebraic P3P (faster)
 * - SOLVEPNP_EPNP: Efficient PnP (O(n) complexity)
 * - SOLVEPNP_DLS: Direct Least Squares
 * - SOLVEPNP_SQPNP: Sequential Quadratic Programming
 * - SOLVEPNP_IPPE: Infinitesimal Plane-based Pose Estimation (for coplanar points)
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Structure to hold PnP method information
struct PnPMethod {
    int flag;
    std::string name;
    int minPoints;
    bool supportsPlanar;
};

// Available PnP methods in OpenCV
const std::vector<PnPMethod> PNP_METHODS = {
    {cv::SOLVEPNP_ITERATIVE, "ITERATIVE (LM)", 4, true},
    {cv::SOLVEPNP_P3P, "P3P", 4, false},
    {cv::SOLVEPNP_AP3P, "AP3P", 4, false},
    {cv::SOLVEPNP_EPNP, "EPnP", 4, true},
    {cv::SOLVEPNP_DLS, "DLS", 4, true},
    {cv::SOLVEPNP_SQPNP, "SQPnP", 4, true},
    {cv::SOLVEPNP_IPPE, "IPPE", 4, true},
    {cv::SOLVEPNP_IPPE_SQUARE, "IPPE_SQUARE", 4, true}
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nOptions:\n"
              << "  --points <n>       Number of 3D points (default: 10)\n"
              << "  --noise <sigma>    Image point noise in pixels (default: 1.0)\n"
              << "  --planar           Use coplanar 3D points\n"
              << "  --iterations <n>   Number of iterations for timing (default: 100)\n"
              << "  --verbose          Show detailed output\n"
              << "\nExamples:\n"
              << "  " << programName << " --points 20 --noise 0.5\n"
              << "  " << programName << " --planar --points 8\n";
}

// Generate synthetic 3D points (cube or planar pattern)
std::vector<cv::Point3f> generateObjectPoints(int numPoints, bool planar) {
    std::vector<cv::Point3f> points;
    std::mt19937 rng(42);  // Fixed seed for reproducibility

    if (planar) {
        // Generate points on a plane (checkerboard-like pattern)
        int side = static_cast<int>(std::ceil(std::sqrt(numPoints)));
        float spacing = 0.1f;  // 10cm spacing

        for (int i = 0; i < side && points.size() < static_cast<size_t>(numPoints); i++) {
            for (int j = 0; j < side && points.size() < static_cast<size_t>(numPoints); j++) {
                float x = (i - side / 2.0f) * spacing;
                float y = (j - side / 2.0f) * spacing;
                points.emplace_back(x, y, 0.0f);  // Z = 0 for planar
            }
        }
    } else {
        // Generate 3D points distributed in a cube
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        for (int i = 0; i < numPoints; i++) {
            points.emplace_back(dist(rng), dist(rng), dist(rng) + 2.0f);  // Centered at Z=2m
        }
    }

    return points;
}

// Project 3D points to 2D using camera intrinsics and pose
std::vector<cv::Point2f> projectPoints(const std::vector<cv::Point3f>& objectPoints,
                                        const cv::Mat& K, const cv::Mat& rvec,
                                        const cv::Mat& tvec, double noiseStd = 0.0) {
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(objectPoints, rvec, tvec, K, cv::Mat(), imagePoints);

    // Add Gaussian noise if specified
    if (noiseStd > 0) {
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> noise(0.0f, static_cast<float>(noiseStd));

        for (auto& pt : imagePoints) {
            pt.x += noise(rng);
            pt.y += noise(rng);
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
    double cos_angle = (trace - 1.0) / 2.0;
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));  // Clamp for numerical stability
    return std::acos(cos_angle) * 180.0 / CV_PI;
}

// Compute translation error (Euclidean distance)
double translationError(const cv::Mat& t1, const cv::Mat& t2) {
    cv::Mat diff = t1 - t2;
    return cv::norm(diff);
}

// Compute reprojection error
double reprojectionError(const std::vector<cv::Point3f>& objectPoints,
                         const std::vector<cv::Point2f>& imagePoints,
                         const cv::Mat& K, const cv::Mat& rvec, const cv::Mat& tvec) {
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(objectPoints, rvec, tvec, K, cv::Mat(), projectedPoints);

    double totalError = 0.0;
    for (size_t i = 0; i < imagePoints.size(); i++) {
        double dx = imagePoints[i].x - projectedPoints[i].x;
        double dy = imagePoints[i].y - projectedPoints[i].y;
        totalError += std::sqrt(dx * dx + dy * dy);
    }

    return totalError / imagePoints.size();
}

// Test a single PnP method
struct PnPResult {
    bool success;
    double rotationError;
    double translationError;
    double reprojError;
    double timeMs;
};

PnPResult testPnPMethod(const PnPMethod& method,
                        const std::vector<cv::Point3f>& objectPoints,
                        const std::vector<cv::Point2f>& imagePoints,
                        const cv::Mat& K, const cv::Mat& distCoeffs,
                        const cv::Mat& R_gt, const cv::Mat& t_gt,
                        int iterations = 1) {
    PnPResult result = {false, 0, 0, 0, 0};

    // Check minimum points requirement
    if (objectPoints.size() < static_cast<size_t>(method.minPoints)) {
        return result;
    }

    cv::Mat rvec, tvec;

    // Time the execution
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        rvec.release();
        tvec.release();

        try {
            result.success = cv::solvePnP(objectPoints, imagePoints, K, distCoeffs,
                                          rvec, tvec, false, method.flag);
        } catch (const cv::Exception& e) {
            result.success = false;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.timeMs = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    if (result.success && !rvec.empty() && !tvec.empty()) {
        cv::Mat R_est = rotationVectorToMatrix(rvec);
        result.rotationError = rotationError(R_gt, R_est);
        result.translationError = translationError(t_gt, tvec);
        result.reprojError = reprojectionError(objectPoints, imagePoints, K, rvec, tvec);
    }

    return result;
}

void printResultsTable(const std::vector<std::pair<std::string, PnPResult>>& results) {
    std::cout << "\n" << std::string(85, '=') << "\n";
    std::cout << std::left << std::setw(15) << "Method"
              << std::right << std::setw(10) << "Success"
              << std::setw(15) << "Rot Err (deg)"
              << std::setw(15) << "Trans Err (m)"
              << std::setw(15) << "Reproj (px)"
              << std::setw(12) << "Time (ms)"
              << "\n";
    std::cout << std::string(85, '-') << "\n";

    for (const auto& [name, res] : results) {
        std::cout << std::left << std::setw(15) << name
                  << std::right << std::setw(10) << (res.success ? "Yes" : "No");

        if (res.success) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(15) << res.rotationError
                      << std::setw(15) << res.translationError
                      << std::setw(15) << res.reprojError
                      << std::setw(12) << res.timeMs;
        } else {
            std::cout << std::setw(15) << "-"
                      << std::setw(15) << "-"
                      << std::setw(15) << "-"
                      << std::setw(12) << "-";
        }
        std::cout << "\n";
    }
    std::cout << std::string(85, '=') << "\n";
}

int main(int argc, char** argv) {
    // Default parameters
    int numPoints = 10;
    double noiseStd = 1.0;  // pixels
    bool planar = false;
    int iterations = 100;
    bool verbose = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--points" && i + 1 < argc) {
            numPoints = std::stoi(argv[++i]);
        } else if (arg == "--noise" && i + 1 < argc) {
            noiseStd = std::stod(argv[++i]);
        } else if (arg == "--planar") {
            planar = true;
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        }
    }

    std::cout << "=== PnP Algorithm Comparison Demo ===\n\n";

    // Camera intrinsics (typical VGA camera)
    const int imageWidth = 640;
    const int imageHeight = 480;
    const double fx = 500.0, fy = 500.0;
    const double cx = imageWidth / 2.0, cy = imageHeight / 2.0;

    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    std::cout << "Camera Intrinsics:\n";
    std::cout << "  Image size: " << imageWidth << "x" << imageHeight << "\n";
    std::cout << "  fx, fy: " << fx << ", " << fy << "\n";
    std::cout << "  cx, cy: " << cx << ", " << cy << "\n\n";

    // Ground truth pose
    // Camera looking at the scene from (0, 0, 0) towards positive Z
    // with some rotation and translation
    cv::Mat rvec_gt = (cv::Mat_<double>(3, 1) << 0.1, 0.2, 0.05);  // Small rotation
    cv::Mat tvec_gt = (cv::Mat_<double>(3, 1) << 0.1, -0.05, 2.0);  // 2m away
    cv::Mat R_gt = rotationVectorToMatrix(rvec_gt);

    std::cout << "Ground Truth Pose:\n";
    std::cout << "  Rotation (rvec): [" << rvec_gt.at<double>(0) << ", "
              << rvec_gt.at<double>(1) << ", " << rvec_gt.at<double>(2) << "]\n";
    std::cout << "  Translation: [" << tvec_gt.at<double>(0) << ", "
              << tvec_gt.at<double>(1) << ", " << tvec_gt.at<double>(2) << "]\n\n";

    // Generate test data
    std::cout << "Test Configuration:\n";
    std::cout << "  Number of points: " << numPoints << "\n";
    std::cout << "  Point distribution: " << (planar ? "Planar" : "3D") << "\n";
    std::cout << "  Image noise (sigma): " << noiseStd << " pixels\n";
    std::cout << "  Timing iterations: " << iterations << "\n\n";

    std::vector<cv::Point3f> objectPoints = generateObjectPoints(numPoints, planar);
    std::vector<cv::Point2f> imagePoints = projectPoints(objectPoints, K, rvec_gt, tvec_gt, noiseStd);

    if (verbose) {
        std::cout << "3D Points:\n";
        for (size_t i = 0; i < objectPoints.size(); i++) {
            std::cout << "  [" << objectPoints[i].x << ", " << objectPoints[i].y
                      << ", " << objectPoints[i].z << "] -> (" << imagePoints[i].x
                      << ", " << imagePoints[i].y << ")\n";
        }
        std::cout << "\n";
    }

    // Test all PnP methods
    std::vector<std::pair<std::string, PnPResult>> results;

    std::cout << "Running PnP algorithms...\n";

    for (const auto& method : PNP_METHODS) {
        // Skip non-planar methods for planar points
        if (!planar && method.flag == cv::SOLVEPNP_IPPE) {
            continue;
        }
        if (!planar && method.flag == cv::SOLVEPNP_IPPE_SQUARE) {
            continue;
        }

        PnPResult result = testPnPMethod(method, objectPoints, imagePoints,
                                          K, distCoeffs, R_gt, tvec_gt, iterations);
        results.emplace_back(method.name, result);

        if (verbose && result.success) {
            std::cout << "  " << method.name << ": OK\n";
        }
    }

    // Print results table
    printResultsTable(results);

    // Summary and recommendations
    std::cout << "\n=== Recommendations ===\n\n";
    std::cout << "1. For general use: EPnP or SQPnP offer good balance of speed and accuracy\n";
    std::cout << "2. For RANSAC inner loop: P3P or AP3P (minimal solver, fast)\n";
    std::cout << "3. For highest accuracy: ITERATIVE (uses initial guess + LM refinement)\n";
    std::cout << "4. For coplanar points: IPPE or IPPE_SQUARE\n";
    std::cout << "5. For many points: DLS or EPnP scale well with O(n) complexity\n\n";

    // Example code snippet
    std::cout << "=== Code Example ===\n\n";
    std::cout << R"(
// Basic PnP usage in OpenCV
cv::Mat rvec, tvec;
bool success = cv::solvePnP(
    objectPoints,       // 3D points in world frame
    imagePoints,        // 2D points in image
    cameraMatrix,       // 3x3 intrinsic matrix
    distCoeffs,         // Distortion coefficients
    rvec, tvec,         // Output rotation and translation
    false,              // useExtrinsicGuess
    cv::SOLVEPNP_EPNP   // Method flag
);

// Convert rotation vector to matrix
cv::Mat R;
cv::Rodrigues(rvec, R);

// The pose [R|t] transforms points from world to camera frame:
//   P_camera = R * P_world + t
)";

    std::cout << "\n\nDone.\n";
    return 0;
}
