/**
 * @file relpose_opengv.cpp
 * @brief 5-point relative pose solvers (OpenGV) on a real KITTI stereo pair
 *
 * Detects and matches ORB features between the bundled KITTI stereo pair,
 * converts the inlier correspondences to bearing vectors with the KITTI
 * intrinsics, runs OpenGV's Nister and Stewenius 5-point solvers, decomposes
 * the essential matrices, and compares the recovered relative pose to the known
 * KITTI stereo extrinsic. OpenCV recoverPose is shown alongside as a cross-check.
 *
 * Images default to the bundled data/ pair; pass two paths to override.
 * OpenGV: https://laurentkneip.github.io/opengv/
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

static std::string resolveDataPath(const std::string& name) {
    for (const std::string& base : {"../data/", "data/", "./data/"}) {
        if (std::filesystem::exists(base + name)) return base + name;
    }
    return "../data/" + name;
}

double rotationError(const Eigen::Matrix3d& R_est, const Eigen::Matrix3d& R_gt) {
    Eigen::Matrix3d dR = R_est.transpose() * R_gt;
    double trace = dR.trace();
    double angle = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
    return angle * 180.0 / M_PI;
}

double translationDirectionError(const Eigen::Vector3d& t_est,
                                 const Eigen::Vector3d& t_gt) {
    Eigen::Vector3d t1 = t_est.normalized();
    Eigen::Vector3d t2 = t_gt.normalized();
    double dot = std::clamp(std::abs(t1.dot(t2)), 0.0, 1.0);  // sign ambiguous
    return std::acos(dot) * 180.0 / M_PI;
}

/**
 * Pick the (R, t) decomposition of an essential matrix closest to R_gt.
 * Updates best_* in place.
 */
void scoreEssential(const Eigen::Matrix3d& E,
                    const Eigen::Matrix3d& R_gt,
                    double& best_rot_error,
                    Eigen::Matrix3d& best_R,
                    Eigen::Vector3d& best_t) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d W;
    W << 0, -1, 0,
         1, 0, 0,
         0, 0, 1;

    Eigen::Matrix3d R1 = U * W * V.transpose();
    Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();
    if (R1.determinant() < 0) R1 = -R1;
    if (R2.determinant() < 0) R2 = -R2;

    Eigen::Vector3d t = U.col(2);

    std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> candidates = {
        {R1, t}, {R1, -t}, {R2, t}, {R2, -t}};

    for (const auto& [R_c, t_c] : candidates) {
        double rot_err = rotationError(R_c, R_gt);
        if (rot_err < best_rot_error) {
            best_rot_error = rot_err;
            best_R = R_c;
            best_t = t_c;
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "===========================================\n";
    std::cout << "Relative Pose Demo using OpenGV (5-Point)\n";
    std::cout << "KITTI stereo pair, real correspondences\n";
    std::cout << "===========================================\n\n";

    // Real KITTI stereo pair (cam0/cam1). Override with two image paths.
    std::string left  = (argc >= 3) ? argv[1] : resolveDataPath("left.png");
    std::string right = (argc >= 3) ? argv[2] : resolveDataPath("right.png");

    cv::Mat img1 = cv::imread(left, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(right, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: failed to load images:\n  " << left << "\n  " << right
                  << "\nPass two image paths, or run from build/ so ../data resolves.\n";
        return 1;
    }
    std::cout << "Left:  " << left << "  (" << img1.cols << "x" << img1.rows << ")\n";
    std::cout << "Right: " << right << "  (" << img2.cols << "x" << img2.rows << ")\n\n";

    // KITTI odometry seq 00-02 rectified intrinsics (image size 1241x376).
    const double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    Eigen::Matrix3d K;
    K << fx, 0, cx,
         0, fy, cy,
         0, 0, 1;
    cv::Mat cvK = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Ground-truth extrinsic (cam1 relative to cam0): R = I, t along -X.
    const double baseline_m = 386.1448 / fx;   // ~0.5372 m
    Eigen::Matrix3d R_gt = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_gt(-baseline_m, 0.0, 0.0);
    std::cout << "Ground-truth extrinsic: R = I, t direction = "
              << t_gt.normalized().transpose() << " (baseline " << baseline_m << " m)\n\n";

    // ORB detect + match
    auto orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc1, desc2, knn, 2);

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : knn) {
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) {
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }
    std::cout << "Good matches: " << pts1.size() << "\n";
    if (pts1.size() < 8) {
        std::cerr << "Not enough matches for relative pose estimation!\n";
        return 1;
    }

    // RANSAC-filter the matches; reuse for the OpenCV cross-check.
    cv::Mat mask;
    cv::Mat E_cv = cv::findEssentialMat(pts1, pts2, cvK, cv::RANSAC, 0.999, 1.0, mask);

    std::vector<cv::Point2f> in1, in2;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (mask.at<uchar>(i)) {
            in1.push_back(pts1[i]);
            in2.push_back(pts2[i]);
        }
    }
    std::cout << "RANSAC inliers: " << in1.size() << "/" << pts1.size() << "\n\n";
    if (in1.size() < 5) {
        std::cerr << "Not enough inliers for the 5-point solvers!\n";
        return 1;
    }

    // Convert inlier pixels to bearing vectors with K^-1.
    Eigen::Matrix3d Kinv = K.inverse();
    auto toBearing = [&](const cv::Point2f& p) {
        Eigen::Vector3d ray = Kinv * Eigen::Vector3d(p.x, p.y, 1.0);
        return ray.normalized();
    };
    opengv::bearingVectors_t bearings1, bearings2;
    for (size_t i = 0; i < in1.size(); ++i) {
        bearings1.push_back(toBearing(in1[i]));
        bearings2.push_back(toBearing(in2[i]));
    }

    opengv::relative_pose::CentralRelativeAdapter adapter(bearings1, bearings2);

    // =========================================================================
    // OpenGV 5-Point Nister (uses the first 5 correspondences)
    // =========================================================================
    std::cout << "--- OpenGV 5-Point Nister Solver ---\n";

    opengv::essentials_t E_nister = opengv::relative_pose::fivept_nister(adapter);
    std::cout << "Number of essential matrices: " << E_nister.size() << "\n";

    double best_rot_error = std::numeric_limits<double>::max();
    Eigen::Matrix3d best_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d best_t = Eigen::Vector3d::Zero();
    for (const auto& E : E_nister) {
        scoreEssential(E, R_gt, best_rot_error, best_R, best_t);
    }
    std::cout << "Best (closest to GT):\n";
    std::cout << "  Rotation error:        " << std::fixed << std::setprecision(4)
              << best_rot_error << " deg\n";
    std::cout << "  Translation dir error: "
              << translationDirectionError(best_t, t_gt) << " deg\n";
    std::cout << "  Estimated rotation:\n" << best_R << "\n";
    std::cout << "  Estimated translation: " << best_t.transpose() << "\n";

    // =========================================================================
    // OpenGV 5-Point Stewenius
    // =========================================================================
    std::cout << "\n--- OpenGV 5-Point Stewenius Solver ---\n";

    opengv::complexEssentials_t E_stew_complex =
        opengv::relative_pose::fivept_stewenius(adapter);
    opengv::essentials_t E_stew;
    for (const auto& E_c : E_stew_complex) E_stew.push_back(E_c.real());
    std::cout << "Number of essential matrices: " << E_stew.size() << "\n";

    best_rot_error = std::numeric_limits<double>::max();
    for (const auto& E : E_stew) {
        scoreEssential(E, R_gt, best_rot_error, best_R, best_t);
    }
    std::cout << "Best rotation error:        " << best_rot_error << " deg\n";
    std::cout << "Best translation dir error: "
              << translationDirectionError(best_t, t_gt) << " deg\n";

    // =========================================================================
    // OpenCV 5-Point (cross-check on all inliers), vs ground truth
    // =========================================================================
    std::cout << "\n--- OpenCV 5-Point (cross-check) ---\n";

    cv::Mat R_cv, t_cv;
    cv::recoverPose(E_cv, in1, in2, cvK, R_cv, t_cv);

    Eigen::Matrix3d R_opencv;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_opencv(i, j) = R_cv.at<double>(i, j);
    Eigen::Vector3d t_opencv(t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2));

    std::cout << "Rotation error vs GT:        " << rotationError(R_opencv, R_gt)
              << " deg\n";
    std::cout << "Translation dir error vs GT: "
              << translationDirectionError(t_opencv, t_gt) << " deg\n";

    std::cout << "\n===========================================\n";
    std::cout << "Notes:\n";
    std::cout << "- Nister and Stewenius are both 5-point solvers\n";
    std::cout << "- Essential decomposition gives 4 (R,t) candidates each\n";
    std::cout << "- Translation is only recovered up to scale\n";
    std::cout << "- GT is the KITTI stereo baseline (R = I, t along -X)\n";
    std::cout << "===========================================\n";

    return 0;
}
