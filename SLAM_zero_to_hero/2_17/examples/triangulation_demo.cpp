/**
 * Triangulation Demo for SLAM
 *
 * This example demonstrates various triangulation methods:
 * 1. OpenCV cv::triangulatePoints (DLT-based)
 * 2. Custom DLT implementation
 * 3. Mid-point method
 * 4. Stereo depth estimation
 *
 * These techniques are fundamental for 3D reconstruction in Visual SLAM.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

// ============================================================================
// Custom DLT Triangulation (Eigen-based)
// ============================================================================
/**
 * Triangulate a 3D point using the Direct Linear Transform (DLT) method.
 *
 * Given two projection matrices P1, P2 and corresponding 2D observations x1, x2,
 * we solve the linear system: A * X = 0 using SVD.
 *
 * The constraint comes from: x = P * X => x cross (P * X) = 0
 *
 * @param P1 First camera projection matrix (3x4)
 * @param P2 Second camera projection matrix (3x4)
 * @param x1 Observed 2D point in first camera (normalized or pixel coords)
 * @param x2 Observed 2D point in second camera (normalized or pixel coords)
 * @return 3D point in world coordinates
 */
Eigen::Vector3d triangulate_dlt(
    const Eigen::Matrix<double, 3, 4>& P1,
    const Eigen::Matrix<double, 3, 4>& P2,
    const Eigen::Vector2d& x1,
    const Eigen::Vector2d& x2)
{
    // Build the 4x4 matrix A from the constraint: x cross (P * X) = 0
    // Row 0: x1(0) * P1.row(2) - P1.row(0)
    // Row 1: x1(1) * P1.row(2) - P1.row(1)
    // Row 2: x2(0) * P2.row(2) - P2.row(0)
    // Row 3: x2(1) * P2.row(2) - P2.row(1)

    Eigen::Matrix4d A;
    A.row(0) = x1(0) * P1.row(2) - P1.row(0);
    A.row(1) = x1(1) * P1.row(2) - P1.row(1);
    A.row(2) = x2(0) * P2.row(2) - P2.row(0);
    A.row(3) = x2(1) * P2.row(2) - P2.row(1);

    // Solve using SVD: X is the right singular vector for smallest singular value
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X_homogeneous = svd.matrixV().col(3);

    // Dehomogenize: [X, Y, Z, W] -> [X/W, Y/W, Z/W]
    return X_homogeneous.head<3>() / X_homogeneous(3);
}

// ============================================================================
// Multi-view DLT Triangulation
// ============================================================================
/**
 * Triangulate from multiple views using DLT.
 *
 * @param projections Vector of projection matrices
 * @param observations Vector of 2D observations (one per view)
 * @return 3D point in world coordinates
 */
Eigen::Vector3d triangulate_multiview_dlt(
    const std::vector<Eigen::Matrix<double, 3, 4>>& projections,
    const std::vector<Eigen::Vector2d>& observations)
{
    const int n_views = projections.size();
    Eigen::MatrixXd A(2 * n_views, 4);

    for (int i = 0; i < n_views; ++i) {
        const auto& P = projections[i];
        const auto& x = observations[i];
        A.row(2 * i + 0) = x(0) * P.row(2) - P.row(0);
        A.row(2 * i + 1) = x(1) * P.row(2) - P.row(1);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X_homogeneous = svd.matrixV().col(3);

    return X_homogeneous.head<3>() / X_homogeneous(3);
}

// ============================================================================
// Mid-Point Triangulation
// ============================================================================
/**
 * Triangulate using the mid-point method.
 *
 * Finds the closest point of approach between two rays and returns the midpoint.
 * This is geometrically intuitive but not optimal in image space.
 *
 * @param O1 Origin of first ray (camera center 1)
 * @param d1 Direction of first ray (normalized)
 * @param O2 Origin of second ray (camera center 2)
 * @param d2 Direction of second ray (normalized)
 * @param is_valid Output: false if rays are nearly parallel
 * @return Midpoint of the closest approach
 */
Eigen::Vector3d triangulate_midpoint(
    const Eigen::Vector3d& O1, const Eigen::Vector3d& d1,
    const Eigen::Vector3d& O2, const Eigen::Vector3d& d2,
    bool& is_valid)
{
    // Ray 1: r1(t) = O1 + t * d1
    // Ray 2: r2(s) = O2 + s * d2
    // Find t, s that minimize ||r1(t) - r2(s)||^2

    Eigen::Vector3d w0 = O1 - O2;

    double a = d1.dot(d1);  // Always positive
    double b = d1.dot(d2);
    double c = d2.dot(d2);  // Always positive
    double d = d1.dot(w0);
    double e = d2.dot(w0);

    double denom = a * c - b * b;

    // Check for nearly parallel rays
    const double EPSILON = 1e-8;
    if (std::abs(denom) < EPSILON) {
        is_valid = false;
        return Eigen::Vector3d::Zero();
    }

    is_valid = true;

    // Solve for t and s
    double t = (b * e - c * d) / denom;
    double s = (a * e - b * d) / denom;

    // Closest points on each ray
    Eigen::Vector3d p1 = O1 + t * d1;
    Eigen::Vector3d p2 = O2 + s * d2;

    // Return midpoint
    return (p1 + p2) / 2.0;
}

/**
 * Compute ray direction from camera matrix and 2D point.
 *
 * @param K Camera intrinsic matrix (3x3)
 * @param R Camera rotation matrix (3x3)
 * @param x 2D point in pixel coordinates
 * @return Normalized ray direction in world coordinates
 */
Eigen::Vector3d compute_ray_direction(
    const Eigen::Matrix3d& K,
    const Eigen::Matrix3d& R,
    const Eigen::Vector2d& x)
{
    // Convert pixel to normalized camera coordinates
    Eigen::Vector3d x_normalized;
    x_normalized << x(0), x(1), 1.0;
    Eigen::Vector3d ray_camera = K.inverse() * x_normalized;

    // Transform to world coordinates
    Eigen::Vector3d ray_world = R.transpose() * ray_camera;

    return ray_world.normalized();
}

// ============================================================================
// Stereo Depth Estimation
// ============================================================================
/**
 * Compute depth from stereo disparity.
 *
 * For a rectified stereo pair:
 *   depth = (focal_length * baseline) / disparity
 *
 * @param focal_length Focal length in pixels
 * @param baseline Distance between camera centers
 * @param disparity Horizontal pixel difference (x_left - x_right)
 * @return Depth value
 */
double stereo_depth(double focal_length, double baseline, double disparity)
{
    if (std::abs(disparity) < 1e-8) {
        return std::numeric_limits<double>::infinity();
    }
    return (focal_length * baseline) / disparity;
}

// ============================================================================
// OpenCV Wrapper for Comparison
// ============================================================================
/**
 * Triangulate points using OpenCV's cv::triangulatePoints.
 *
 * @param P1 First projection matrix (3x4)
 * @param P2 Second projection matrix (3x4)
 * @param points1 2D points in first image
 * @param points2 2D points in second image
 * @return Vector of 3D points
 */
std::vector<Eigen::Vector3d> triangulate_opencv(
    const cv::Mat& P1, const cv::Mat& P2,
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2)
{
    cv::Mat pts_4d;
    cv::triangulatePoints(P1, P2, points1, points2, pts_4d);

    std::vector<Eigen::Vector3d> points_3d;
    points_3d.reserve(pts_4d.cols);

    for (int i = 0; i < pts_4d.cols; ++i) {
        cv::Mat x = pts_4d.col(i);
        double w = x.at<float>(3, 0);
        points_3d.emplace_back(
            x.at<float>(0, 0) / w,
            x.at<float>(1, 0) / w,
            x.at<float>(2, 0) / w
        );
    }

    return points_3d;
}

// ============================================================================
// Utility: Compute Reprojection Error
// ============================================================================
/**
 * Compute reprojection error for a triangulated point.
 */
double compute_reprojection_error(
    const Eigen::Matrix<double, 3, 4>& P,
    const Eigen::Vector3d& X,
    const Eigen::Vector2d& x_observed)
{
    Eigen::Vector4d X_hom;
    X_hom << X, 1.0;

    Eigen::Vector3d x_proj = P * X_hom;
    Eigen::Vector2d x_reproj(x_proj(0) / x_proj(2), x_proj(1) / x_proj(2));

    return (x_reproj - x_observed).norm();
}

// ============================================================================
// Main Demo
// ============================================================================
int main()
{
    std::cout << "========================================\n";
    std::cout << "   Triangulation Demo for SLAM\n";
    std::cout << "========================================\n\n";

    // -------------------------------------------------------------------------
    // Setup: Define cameras and 3D points
    // -------------------------------------------------------------------------
    std::cout << "[Setup] Creating synthetic stereo camera configuration...\n\n";

    // Camera intrinsics (KITTI-like)
    double fx = 718.856;
    double fy = 718.856;
    double cx = 607.193;
    double cy = 185.216;

    Eigen::Matrix3d K;
    K << fx, 0, cx,
         0, fy, cy,
         0, 0, 1;

    // Camera 1: Identity pose (at origin)
    Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t1 = Eigen::Vector3d::Zero();

    // Camera 2: Translated along X (stereo baseline = 0.54m, KITTI-like)
    double baseline = 0.54;
    Eigen::Matrix3d R2 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t2(baseline, 0, 0);

    // Build projection matrices: P = K * [R | t]
    Eigen::Matrix<double, 3, 4> P1, P2;
    P1.block<3, 3>(0, 0) = K * R1;
    P1.block<3, 1>(0, 3) = K * (-R1 * t1);

    P2.block<3, 3>(0, 0) = K * R2;
    P2.block<3, 1>(0, 3) = K * (-R2 * t2);

    // OpenCV projection matrices
    cv::Mat P1_cv = (cv::Mat_<float>(3, 4) <<
        P1(0,0), P1(0,1), P1(0,2), P1(0,3),
        P1(1,0), P1(1,1), P1(1,2), P1(1,3),
        P1(2,0), P1(2,1), P1(2,2), P1(2,3));

    cv::Mat P2_cv = (cv::Mat_<float>(3, 4) <<
        P2(0,0), P2(0,1), P2(0,2), P2(0,3),
        P2(1,0), P2(1,1), P2(1,2), P2(1,3),
        P2(2,0), P2(2,1), P2(2,2), P2(2,3));

    // -------------------------------------------------------------------------
    // Create synthetic 3D points
    // -------------------------------------------------------------------------
    std::vector<Eigen::Vector3d> ground_truth_3d = {
        {0.0, 0.0, 10.0},    // Straight ahead, 10m
        {2.0, 1.0, 15.0},    // Right-up, 15m
        {-1.5, -0.5, 8.0},   // Left-down, 8m
        {3.0, -1.0, 20.0},   // Far right, 20m
        {0.5, 0.5, 5.0}      // Close, 5m
    };

    std::cout << "Ground Truth 3D Points:\n";
    for (size_t i = 0; i < ground_truth_3d.size(); ++i) {
        std::cout << "  Point " << i << ": ["
                  << ground_truth_3d[i].transpose() << "]\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Project 3D points to both cameras
    // -------------------------------------------------------------------------
    std::vector<Eigen::Vector2d> obs1, obs2;  // Observations in each camera
    std::vector<cv::Point2f> pts1_cv, pts2_cv;

    std::cout << "Projected 2D Observations:\n";
    for (const auto& X : ground_truth_3d) {
        Eigen::Vector4d X_hom;
        X_hom << X, 1.0;

        // Project to camera 1
        Eigen::Vector3d x1_hom = P1 * X_hom;
        Eigen::Vector2d x1(x1_hom(0) / x1_hom(2), x1_hom(1) / x1_hom(2));
        obs1.push_back(x1);
        pts1_cv.emplace_back(x1(0), x1(1));

        // Project to camera 2
        Eigen::Vector3d x2_hom = P2 * X_hom;
        Eigen::Vector2d x2(x2_hom(0) / x2_hom(2), x2_hom(1) / x2_hom(2));
        obs2.push_back(x2);
        pts2_cv.emplace_back(x2(0), x2(1));

        std::cout << "  Camera1: [" << std::fixed << std::setprecision(2)
                  << x1.transpose() << "]  Camera2: [" << x2.transpose() << "]\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Demo 1: OpenCV cv::triangulatePoints
    // -------------------------------------------------------------------------
    std::cout << "========================================\n";
    std::cout << "Demo 1: OpenCV cv::triangulatePoints\n";
    std::cout << "========================================\n";

    auto opencv_result = triangulate_opencv(P1_cv, P2_cv, pts1_cv, pts2_cv);

    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < opencv_result.size(); ++i) {
        double error = (opencv_result[i] - ground_truth_3d[i]).norm();
        std::cout << "  Point " << i << ": ["
                  << std::setw(8) << opencv_result[i](0) << ", "
                  << std::setw(8) << opencv_result[i](1) << ", "
                  << std::setw(8) << opencv_result[i](2) << "]"
                  << "  Error: " << error << "m\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Demo 2: Custom DLT Implementation
    // -------------------------------------------------------------------------
    std::cout << "========================================\n";
    std::cout << "Demo 2: Custom DLT Implementation\n";
    std::cout << "========================================\n";

    for (size_t i = 0; i < obs1.size(); ++i) {
        Eigen::Vector3d X_dlt = triangulate_dlt(P1, P2, obs1[i], obs2[i]);
        double error = (X_dlt - ground_truth_3d[i]).norm();

        // Compute reprojection errors
        double reproj_err1 = compute_reprojection_error(P1, X_dlt, obs1[i]);
        double reproj_err2 = compute_reprojection_error(P2, X_dlt, obs2[i]);

        std::cout << "  Point " << i << ": ["
                  << std::setw(8) << X_dlt(0) << ", "
                  << std::setw(8) << X_dlt(1) << ", "
                  << std::setw(8) << X_dlt(2) << "]"
                  << "  Error: " << error << "m"
                  << "  Reproj: [" << reproj_err1 << ", " << reproj_err2 << "] px\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Demo 3: Mid-Point Method
    // -------------------------------------------------------------------------
    std::cout << "========================================\n";
    std::cout << "Demo 3: Mid-Point Method\n";
    std::cout << "========================================\n";

    // Camera centers
    Eigen::Vector3d C1 = t1;
    Eigen::Vector3d C2 = t2;

    for (size_t i = 0; i < obs1.size(); ++i) {
        // Compute ray directions
        Eigen::Vector3d d1 = compute_ray_direction(K, R1, obs1[i]);
        Eigen::Vector3d d2 = compute_ray_direction(K, R2, obs2[i]);

        bool is_valid;
        Eigen::Vector3d X_mid = triangulate_midpoint(C1, d1, C2, d2, is_valid);

        if (is_valid) {
            double error = (X_mid - ground_truth_3d[i]).norm();
            std::cout << "  Point " << i << ": ["
                      << std::setw(8) << X_mid(0) << ", "
                      << std::setw(8) << X_mid(1) << ", "
                      << std::setw(8) << X_mid(2) << "]"
                      << "  Error: " << error << "m\n";
        } else {
            std::cout << "  Point " << i << ": INVALID (parallel rays)\n";
        }
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Demo 4: Stereo Depth Estimation
    // -------------------------------------------------------------------------
    std::cout << "========================================\n";
    std::cout << "Demo 4: Stereo Depth Estimation\n";
    std::cout << "========================================\n";

    std::cout << "Using formula: depth = (f * b) / disparity\n";
    std::cout << "  Focal length: " << fx << " px\n";
    std::cout << "  Baseline: " << baseline << " m\n\n";

    for (size_t i = 0; i < obs1.size(); ++i) {
        double disparity = obs1[i](0) - obs2[i](0);
        double depth = stereo_depth(fx, baseline, disparity);
        double gt_depth = ground_truth_3d[i](2);
        double error = std::abs(depth - gt_depth);

        std::cout << "  Point " << i << ": "
                  << "Disparity=" << std::setw(8) << disparity << " px  "
                  << "Depth=" << std::setw(8) << depth << " m  "
                  << "GT=" << std::setw(8) << gt_depth << " m  "
                  << "Error=" << error << " m\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Demo 5: Multi-View Triangulation
    // -------------------------------------------------------------------------
    std::cout << "========================================\n";
    std::cout << "Demo 5: Multi-View Triangulation (3 views)\n";
    std::cout << "========================================\n";

    // Add a third camera
    Eigen::Matrix3d R3 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t3(0.27, 0.1, 0);  // Between cameras 1 and 2, slightly elevated

    Eigen::Matrix<double, 3, 4> P3;
    P3.block<3, 3>(0, 0) = K * R3;
    P3.block<3, 1>(0, 3) = K * (-R3 * t3);

    for (size_t i = 0; i < ground_truth_3d.size(); ++i) {
        Eigen::Vector4d X_hom;
        X_hom << ground_truth_3d[i], 1.0;

        // Get observation in third camera
        Eigen::Vector3d x3_hom = P3 * X_hom;
        Eigen::Vector2d x3(x3_hom(0) / x3_hom(2), x3_hom(1) / x3_hom(2));

        // Triangulate from all three views
        std::vector<Eigen::Matrix<double, 3, 4>> projections = {P1, P2, P3};
        std::vector<Eigen::Vector2d> observations = {obs1[i], obs2[i], x3};

        Eigen::Vector3d X_multi = triangulate_multiview_dlt(projections, observations);
        double error = (X_multi - ground_truth_3d[i]).norm();

        std::cout << "  Point " << i << ": ["
                  << std::setw(8) << X_multi(0) << ", "
                  << std::setw(8) << X_multi(1) << ", "
                  << std::setw(8) << X_multi(2) << "]"
                  << "  Error: " << error << "m\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Demo 6: Triangulation with Noise
    // -------------------------------------------------------------------------
    std::cout << "========================================\n";
    std::cout << "Demo 6: Robustness to Noise\n";
    std::cout << "========================================\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise(0.0, 1.0);  // 1 pixel std dev

    std::cout << "Adding Gaussian noise (sigma=1.0 pixel) to observations...\n\n";

    double total_error_opencv = 0;
    double total_error_dlt = 0;

    for (size_t i = 0; i < obs1.size(); ++i) {
        // Add noise to observations
        Eigen::Vector2d obs1_noisy = obs1[i] + Eigen::Vector2d(noise(gen), noise(gen));
        Eigen::Vector2d obs2_noisy = obs2[i] + Eigen::Vector2d(noise(gen), noise(gen));

        // OpenCV triangulation with noisy data
        std::vector<cv::Point2f> pts1_noisy = {cv::Point2f(obs1_noisy(0), obs1_noisy(1))};
        std::vector<cv::Point2f> pts2_noisy = {cv::Point2f(obs2_noisy(0), obs2_noisy(1))};
        auto opencv_noisy = triangulate_opencv(P1_cv, P2_cv, pts1_noisy, pts2_noisy);
        double error_opencv = (opencv_noisy[0] - ground_truth_3d[i]).norm();
        total_error_opencv += error_opencv;

        // Custom DLT with noisy data
        Eigen::Vector3d X_dlt_noisy = triangulate_dlt(P1, P2, obs1_noisy, obs2_noisy);
        double error_dlt = (X_dlt_noisy - ground_truth_3d[i]).norm();
        total_error_dlt += error_dlt;

        std::cout << "  Point " << i << ": "
                  << "OpenCV error=" << std::setw(8) << error_opencv << "m  "
                  << "DLT error=" << std::setw(8) << error_dlt << "m\n";
    }

    std::cout << "\n  Average Error - OpenCV: " << total_error_opencv / obs1.size()
              << "m, DLT: " << total_error_dlt / obs1.size() << "m\n\n";

    std::cout << "========================================\n";
    std::cout << "   Demo Complete!\n";
    std::cout << "========================================\n";

    return 0;
}
