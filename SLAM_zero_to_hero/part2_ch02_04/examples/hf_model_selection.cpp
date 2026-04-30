/**
 * @file hf_model_selection.cpp
 * @brief H/F Model Selection Demo (ORB-SLAM Style)
 *
 * This example demonstrates how to select between Homography and Fundamental
 * matrix for SLAM initialization, following the ORB-SLAM approach.
 *
 * The algorithm:
 * 1. Compute H and F in parallel with RANSAC
 * 2. Compute Symmetric Transfer Error for H
 * 3. Compute Sampson Error for F
 * 4. Select model based on ratio: R_H = S_H / (S_H + S_F)
 *    - If R_H > 0.45: Planar scene -> Use H
 *    - If R_H <= 0.45: 3D scene -> Use F (then E)
 */

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

/**
 * @brief Generate synthetic 3D points on a plane
 */
std::vector<cv::Point3d> generatePlanarPoints(int numPoints, double planeZ = 5.0) {
    std::vector<cv::Point3d> points;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> xy_dist(-2.0, 2.0);

    for (int i = 0; i < numPoints; i++) {
        // All points on z = planeZ plane
        points.emplace_back(xy_dist(rng), xy_dist(rng), planeZ);
    }
    return points;
}

/**
 * @brief Generate synthetic 3D points in 3D space (not coplanar)
 */
std::vector<cv::Point3d> generate3DPoints(int numPoints) {
    std::vector<cv::Point3d> points;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> xy_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> z_dist(3.0, 10.0);

    for (int i = 0; i < numPoints; i++) {
        // Points distributed in 3D
        points.emplace_back(xy_dist(rng), xy_dist(rng), z_dist(rng));
    }
    return points;
}

/**
 * @brief Project 3D points to 2D image coordinates
 */
std::vector<cv::Point2f> projectPoints(
    const std::vector<cv::Point3d>& points3D,
    const cv::Mat& K,
    const cv::Mat& R,
    const cv::Mat& t) {

    std::vector<cv::Point2f> points2D;

    for (const auto& p3d : points3D) {
        cv::Mat P = (cv::Mat_<double>(3, 1) << p3d.x, p3d.y, p3d.z);
        cv::Mat P_cam = R * P + t;

        double x = P_cam.at<double>(0) / P_cam.at<double>(2);
        double y = P_cam.at<double>(1) / P_cam.at<double>(2);

        double u = K.at<double>(0, 0) * x + K.at<double>(0, 2);
        double v = K.at<double>(1, 1) * y + K.at<double>(1, 2);

        points2D.emplace_back(static_cast<float>(u), static_cast<float>(v));
    }

    return points2D;
}

/**
 * @brief Add Gaussian noise to points
 */
void addNoise(std::vector<cv::Point2f>& points, double sigma) {
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, static_cast<float>(sigma));

    for (auto& p : points) {
        p.x += noise(rng);
        p.y += noise(rng);
    }
}

/**
 * @brief Compute Symmetric Transfer Error for Homography
 *
 * This measures how well H maps points bidirectionally:
 * error = d(x', Hx)^2 + d(x, H^-1 x')^2
 */
double computeSymmetricTransferError(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& H,
    const cv::Mat& mask) {

    if (H.empty()) return std::numeric_limits<double>::max();

    cv::Mat H_inv = H.inv();
    double total_error = 0;
    int count = 0;

    for (size_t i = 0; i < pts1.size(); ++i) {
        if (!mask.empty() && mask.at<uchar>(i) == 0) continue;

        // Forward: pts1 -> pts2
        cv::Mat p1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2_proj = H * p1;
        p2_proj /= p2_proj.at<double>(2);

        double dx_fwd = p2_proj.at<double>(0) - pts2[i].x;
        double dy_fwd = p2_proj.at<double>(1) - pts2[i].y;

        // Backward: pts2 -> pts1
        cv::Mat p2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat p1_proj = H_inv * p2;
        p1_proj /= p1_proj.at<double>(2);

        double dx_bwd = p1_proj.at<double>(0) - pts1[i].x;
        double dy_bwd = p1_proj.at<double>(1) - pts1[i].y;

        // Symmetric error
        double error = dx_fwd * dx_fwd + dy_fwd * dy_fwd +
                       dx_bwd * dx_bwd + dy_bwd * dy_bwd;
        total_error += error;
        count++;
    }

    return (count > 0) ? total_error / count : std::numeric_limits<double>::max();
}

/**
 * @brief Compute Sampson Error for Fundamental Matrix
 *
 * Sampson error is the first-order approximation to geometric error:
 * error = (x'^T F x)^2 / (||Fx||^2 + ||F^T x'||^2)
 */
double computeSampsonError(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& F,
    const cv::Mat& mask) {

    if (F.empty()) return std::numeric_limits<double>::max();

    double total_error = 0;
    int count = 0;

    for (size_t i = 0; i < pts1.size(); ++i) {
        if (!mask.empty() && mask.at<uchar>(i) == 0) continue;

        cv::Mat x1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat x2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);

        // Compute x2^T * F * x1
        cv::Mat Fx1 = F * x1;
        cv::Mat Ftx2 = F.t() * x2;
        double x2tFx1 = x2.dot(Fx1);

        // Sampson error denominator
        double denom = Fx1.at<double>(0) * Fx1.at<double>(0) +
                       Fx1.at<double>(1) * Fx1.at<double>(1) +
                       Ftx2.at<double>(0) * Ftx2.at<double>(0) +
                       Ftx2.at<double>(1) * Ftx2.at<double>(1);

        if (denom > 1e-10) {
            double error = (x2tFx1 * x2tFx1) / denom;
            total_error += error;
            count++;
        }
    }

    return (count > 0) ? total_error / count : std::numeric_limits<double>::max();
}

/**
 * @brief Select between H and F models based on score ratio
 */
struct ModelSelectionResult {
    bool use_homography;
    double ratio;
    double score_H;
    double score_F;
    int inliers_H;
    int inliers_F;
    cv::Mat H;
    cv::Mat F;
};

ModelSelectionResult selectModel(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    double ransac_threshold = 3.0) {

    ModelSelectionResult result;

    // Compute Homography with RANSAC
    cv::Mat mask_H;
    result.H = cv::findHomography(pts1, pts2, cv::RANSAC, ransac_threshold, mask_H, 2000, 0.995);
    result.inliers_H = cv::countNonZero(mask_H);

    // Compute Fundamental Matrix with RANSAC
    cv::Mat mask_F;
    result.F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, ransac_threshold, 0.995, mask_F);
    result.inliers_F = cv::countNonZero(mask_F);

    // Compute errors
    result.score_H = computeSymmetricTransferError(pts1, pts2, result.H, mask_H);
    result.score_F = computeSampsonError(pts1, pts2, result.F, mask_F);

    // Compute ratio (ORB-SLAM formula)
    // Note: Lower score is better, so we use score_H in numerator
    // If H is good (low score), ratio is high -> select H
    double sum = result.score_H + result.score_F;
    if (sum > 1e-10) {
        // ORB-SLAM uses: R_H = S_H / (S_H + S_F) where S is the score
        // But they define score as number of inliers with low error
        // Here we adapt: if H has lower error, ratio > 0.5
        result.ratio = result.score_F / sum;  // High if F has higher error (H is better)
    } else {
        result.ratio = 0.5;
    }

    // Select model (threshold 0.45 from ORB-SLAM paper)
    result.use_homography = (result.ratio > 0.45);

    return result;
}

/**
 * @brief Run model selection test
 */
void runTest(const std::string& scenario_name,
             const std::vector<cv::Point3d>& points3D,
             const cv::Mat& K,
             const cv::Mat& R1, const cv::Mat& t1,
             const cv::Mat& R2, const cv::Mat& t2,
             double noise_sigma) {

    std::cout << "\n=== " << scenario_name << " ===" << std::endl;

    // Project points to both cameras
    auto pts1 = projectPoints(points3D, K, R1, t1);
    auto pts2 = projectPoints(points3D, K, R2, t2);

    // Add noise
    addNoise(pts1, noise_sigma);
    addNoise(pts2, noise_sigma);

    // Run model selection
    auto result = selectModel(pts1, pts2);

    // Print results
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Points: " << pts1.size() << std::endl;
    std::cout << "  H inliers: " << result.inliers_H << "/" << pts1.size() << std::endl;
    std::cout << "  F inliers: " << result.inliers_F << "/" << pts1.size() << std::endl;
    std::cout << "  Score H (Symmetric Transfer): " << result.score_H << std::endl;
    std::cout << "  Score F (Sampson): " << result.score_F << std::endl;
    std::cout << "  Ratio R_H: " << result.ratio << std::endl;

    if (result.use_homography) {
        std::cout << "  --> Selected: HOMOGRAPHY (planar scene)" << std::endl;
    } else {
        std::cout << "  --> Selected: FUNDAMENTAL MATRIX (3D scene)" << std::endl;
    }
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "H/F Model Selection Demo (ORB-SLAM Style)" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Camera intrinsics
    double fx = 500, fy = 500;
    double cx = 320, cy = 240;
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1);

    // Camera 1: at origin
    cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64F);

    // Camera 2: rotated and translated
    double angle = 15.0 * CV_PI / 180.0;
    cv::Mat R2 = (cv::Mat_<double>(3, 3) <<
        std::cos(angle), 0, std::sin(angle),
        0, 1, 0,
        -std::sin(angle), 0, std::cos(angle));
    cv::Mat t2 = (cv::Mat_<double>(3, 1) << 0.5, 0.0, 0.0);

    const int numPoints = 50;
    const double noise_sigma = 1.0;  // 1 pixel noise

    std::cout << "\nCamera Setup:" << std::endl;
    std::cout << "  Focal length: " << fx << std::endl;
    std::cout << "  Rotation: " << angle * 180.0 / CV_PI << " degrees around Y" << std::endl;
    std::cout << "  Translation: [" << t2.at<double>(0) << ", "
              << t2.at<double>(1) << ", " << t2.at<double>(2) << "]" << std::endl;
    std::cout << "  Noise: " << noise_sigma << " pixels" << std::endl;

    // Test 1: Planar Scene (all points on a plane)
    auto planarPoints = generatePlanarPoints(numPoints, 5.0);
    runTest("Scenario 1: PLANAR SCENE (wall/floor)", planarPoints, K, R1, t1, R2, t2, noise_sigma);

    // Test 2: 3D Scene (points distributed in 3D)
    auto points3D = generate3DPoints(numPoints);
    runTest("Scenario 2: 3D SCENE (general structure)", points3D, K, R1, t1, R2, t2, noise_sigma);

    // Test 3: Mixed scene (mostly planar with some 3D)
    std::vector<cv::Point3d> mixedPoints;
    auto planar = generatePlanarPoints(numPoints * 3 / 4, 5.0);
    auto nonplanar = generate3DPoints(numPoints / 4);
    mixedPoints.insert(mixedPoints.end(), planar.begin(), planar.end());
    mixedPoints.insert(mixedPoints.end(), nonplanar.begin(), nonplanar.end());
    runTest("Scenario 3: MIXED SCENE (dominant plane)", mixedPoints, K, R1, t1, R2, t2, noise_sigma);

    std::cout << "\n=========================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - Planar scenes should have R_H > 0.45 -> Homography" << std::endl;
    std::cout << "  - 3D scenes should have R_H < 0.45 -> Fundamental Matrix" << std::endl;
    std::cout << "  - Mixed scenes depend on plane dominance" << std::endl;
    std::cout << "=========================================" << std::endl;

    return 0;
}
