/**
 * @file triangulation_opengv.cpp
 * @brief Triangulation demonstration using OpenGV
 *
 * This example demonstrates triangulation methods from OpenGV:
 * - triangulate: Linear triangulation
 * - triangulate2: Optimal L2 triangulation (Lee & Civera)
 *
 * OpenGV: https://laurentkneip.github.io/opengv/
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opengv/triangulation/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

/**
 * @brief Generate synthetic 3D points
 */
std::vector<Eigen::Vector3d> generatePoints3D(int numPoints) {
    std::vector<Eigen::Vector3d> points;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> x_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> y_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> z_dist(5.0, 15.0);

    for (int i = 0; i < numPoints; i++) {
        points.emplace_back(x_dist(rng), y_dist(rng), z_dist(rng));
    }
    return points;
}

/**
 * @brief Project 3D point to bearing vector
 */
Eigen::Vector3d projectToBearing(const Eigen::Vector3d& point3d,
                                  const Eigen::Matrix3d& R,
                                  const Eigen::Vector3d& t) {
    Eigen::Vector3d p_cam = R * point3d + t;
    return p_cam.normalized();
}

/**
 * @brief Add noise to bearing vectors
 */
void addNoise(opengv::bearingVectors_t& bearings, double sigma_rad) {
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, sigma_rad);

    for (auto& b : bearings) {
        Eigen::Vector3d perturb(noise(rng), noise(rng), noise(rng));
        perturb = perturb - b.dot(perturb) * b;
        b = (b + perturb).normalized();
    }
}

/**
 * @brief Custom DLT triangulation for comparison
 */
Eigen::Vector3d triangulateDLT(const Eigen::Matrix<double, 3, 4>& P1,
                                const Eigen::Matrix<double, 3, 4>& P2,
                                const Eigen::Vector2d& x1,
                                const Eigen::Vector2d& x2) {
    Eigen::Matrix4d A;
    A.row(0) = x1(0) * P1.row(2) - P1.row(0);
    A.row(1) = x1(1) * P1.row(2) - P1.row(1);
    A.row(2) = x2(0) * P2.row(2) - P2.row(0);
    A.row(3) = x2(1) * P2.row(2) - P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X_h = svd.matrixV().col(3);
    return X_h.head<3>() / X_h(3);
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "Triangulation Demo using OpenGV\n";
    std::cout << "===========================================\n\n";

    // Camera 1: Identity pose (at origin)
    Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t1 = Eigen::Vector3d::Zero();

    // Camera 2: Translated along X (stereo baseline)
    double baseline = 0.54;  // KITTI-like baseline
    Eigen::Matrix3d R2 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t2(baseline, 0, 0);

    // Camera intrinsics for comparison with OpenCV
    Eigen::Matrix3d K;
    K << 718.856, 0.0, 607.193,
         0.0, 718.856, 185.216,
         0.0, 0.0, 1.0;

    std::cout << "Stereo camera setup:\n";
    std::cout << "  Baseline: " << baseline << " m\n";
    std::cout << "  Focal length: " << K(0, 0) << " px\n\n";

    // Generate ground truth 3D points
    const int numPoints = 5;
    std::vector<Eigen::Vector3d> points3d_gt = {
        {0.0, 0.0, 10.0},
        {2.0, 1.0, 15.0},
        {-1.5, -0.5, 8.0},
        {3.0, -1.0, 20.0},
        {0.5, 0.5, 5.0}
    };

    std::cout << "Ground Truth 3D Points:\n";
    for (size_t i = 0; i < points3d_gt.size(); i++) {
        std::cout << "  Point " << i << ": [" << points3d_gt[i].transpose() << "]\n";
    }
    std::cout << "\n";

    // Project to bearing vectors
    opengv::bearingVectors_t bearings1;
    opengv::bearingVectors_t bearings2;

    for (const auto& p3d : points3d_gt) {
        bearings1.push_back(projectToBearing(p3d, R1, t1));
        bearings2.push_back(projectToBearing(p3d, R2, t2));
    }

    // Add small angular noise
    const double noise_sigma = 0.001;  // radians
    addNoise(bearings1, noise_sigma);
    addNoise(bearings2, noise_sigma);
    std::cout << "Added angular noise with sigma = " << noise_sigma * 180.0 / M_PI << " deg\n\n";

    // Create adapter for OpenGV
    // Note: OpenGV triangulation expects relative pose between cameras
    opengv::relative_pose::CentralRelativeAdapter adapter(
        bearings1, bearings2, t2, R2);

    // =========================================================================
    // OpenGV Linear Triangulation
    // =========================================================================
    std::cout << "--- OpenGV Linear Triangulation ---\n";

    double total_error_linear = 0.0;

    for (size_t i = 0; i < points3d_gt.size(); i++) {
        // OpenGV triangulate returns point in camera 1 frame
        opengv::point_t pt_linear = opengv::triangulation::triangulate(adapter, i);

        double error = (pt_linear - points3d_gt[i]).norm();
        total_error_linear += error;

        std::cout << "  Point " << i << ": ["
                  << std::fixed << std::setprecision(4)
                  << std::setw(8) << pt_linear.x() << ", "
                  << std::setw(8) << pt_linear.y() << ", "
                  << std::setw(8) << pt_linear.z() << "]"
                  << "  Error: " << error << " m\n";
    }
    std::cout << "  Average error: " << total_error_linear / points3d_gt.size() << " m\n\n";

    // =========================================================================
    // OpenGV Optimal L2 Triangulation (triangulate2)
    // =========================================================================
    std::cout << "--- OpenGV Optimal L2 Triangulation ---\n";
    std::cout << "(Lee & Civera method - optimal in L2 sense)\n";

    double total_error_optimal = 0.0;

    for (size_t i = 0; i < points3d_gt.size(); i++) {
        opengv::point_t pt_optimal = opengv::triangulation::triangulate2(adapter, i);

        double error = (pt_optimal - points3d_gt[i]).norm();
        total_error_optimal += error;

        std::cout << "  Point " << i << ": ["
                  << std::fixed << std::setprecision(4)
                  << std::setw(8) << pt_optimal.x() << ", "
                  << std::setw(8) << pt_optimal.y() << ", "
                  << std::setw(8) << pt_optimal.z() << "]"
                  << "  Error: " << error << " m\n";
    }
    std::cout << "  Average error: " << total_error_optimal / points3d_gt.size() << " m\n\n";

    // =========================================================================
    // Compare with OpenCV cv::triangulatePoints
    // =========================================================================
    std::cout << "--- OpenCV cv::triangulatePoints (for comparison) ---\n";

    // Build projection matrices
    Eigen::Matrix<double, 3, 4> P1, P2;
    P1.block<3, 3>(0, 0) = K * R1;
    P1.block<3, 1>(0, 3) = K * (-R1 * t1);
    P2.block<3, 3>(0, 0) = K * R2;
    P2.block<3, 1>(0, 3) = K * (-R2 * t2);

    // Convert bearings to pixel coordinates for OpenCV
    std::vector<cv::Point2f> pts1_cv, pts2_cv;
    for (size_t i = 0; i < bearings1.size(); i++) {
        Eigen::Vector3d px1 = K * bearings1[i] / bearings1[i].z();
        Eigen::Vector3d px2 = K * bearings2[i] / bearings2[i].z();
        pts1_cv.emplace_back(px1.x(), px1.y());
        pts2_cv.emplace_back(px2.x(), px2.y());
    }

    cv::Mat P1_cv = (cv::Mat_<float>(3, 4) <<
        P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3),
        P1(1, 0), P1(1, 1), P1(1, 2), P1(1, 3),
        P1(2, 0), P1(2, 1), P1(2, 2), P1(2, 3));

    cv::Mat P2_cv = (cv::Mat_<float>(3, 4) <<
        P2(0, 0), P2(0, 1), P2(0, 2), P2(0, 3),
        P2(1, 0), P2(1, 1), P2(1, 2), P2(1, 3),
        P2(2, 0), P2(2, 1), P2(2, 2), P2(2, 3));

    cv::Mat pts4d;
    cv::triangulatePoints(P1_cv, P2_cv, pts1_cv, pts2_cv, pts4d);

    double total_error_opencv = 0.0;

    for (int i = 0; i < pts4d.cols; i++) {
        cv::Mat x = pts4d.col(i);
        float w = x.at<float>(3, 0);
        Eigen::Vector3d pt_opencv(
            x.at<float>(0, 0) / w,
            x.at<float>(1, 0) / w,
            x.at<float>(2, 0) / w);

        double error = (pt_opencv - points3d_gt[i]).norm();
        total_error_opencv += error;

        std::cout << "  Point " << i << ": ["
                  << std::fixed << std::setprecision(4)
                  << std::setw(8) << pt_opencv.x() << ", "
                  << std::setw(8) << pt_opencv.y() << ", "
                  << std::setw(8) << pt_opencv.z() << "]"
                  << "  Error: " << error << " m\n";
    }
    std::cout << "  Average error: " << total_error_opencv / points3d_gt.size() << " m\n\n";

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "===========================================\n";
    std::cout << "Summary (Average Triangulation Error)\n";
    std::cout << "===========================================\n";
    std::cout << "  OpenGV Linear:   " << std::fixed << std::setprecision(6)
              << total_error_linear / points3d_gt.size() << " m\n";
    std::cout << "  OpenGV Optimal:  " << total_error_optimal / points3d_gt.size() << " m\n";
    std::cout << "  OpenCV DLT:      " << total_error_opencv / points3d_gt.size() << " m\n";
    std::cout << "\n";

    std::cout << "===========================================\n";
    std::cout << "Notes:\n";
    std::cout << "- OpenGV uses bearing vectors (normalized rays)\n";
    std::cout << "- triangulate: Fast linear method\n";
    std::cout << "- triangulate2: Optimal L2 triangulation\n";
    std::cout << "- OpenCV uses pixel coordinates + projection matrices\n";
    std::cout << "===========================================\n";

    return 0;
}
