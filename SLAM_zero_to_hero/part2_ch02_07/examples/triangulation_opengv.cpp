/**
 * Triangulation via OpenGV
 *
 * Same KITTI stereo pipeline as triangulation_demo, but using bearing vectors
 * instead of pixel coordinates:
 *   1. ORB detect + match, fundamental-matrix RANSAC outlier rejection.
 *   2. Convert each inlier pixel to a unit bearing vector via K^-1.
 *   3. OpenGV CentralRelativeAdapter holds the known stereo relative pose.
 *   4. Triangulate with:
 *        opengv::triangulation::triangulate   (linear, DLT-family)
 *        opengv::triangulation::triangulate2  (mid-point / closest point between the two rays)
 *   5. Filter by cheirality + 0.2-100 m depth gate, report reprojection error.
 *   6. Write triangulation_opengv.json for the Rerun viewer.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opengv/triangulation/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static std::string resolveDataPath(const std::string& name) {
    for (const std::string& base : {"../data/", "data/", "./data/"}) {
        if (std::filesystem::exists(base + name)) return base + name;
    }
    return "../data/" + name;
}

static void detectAndMatchFeatures(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2) {

    auto orb = cv::ORB::create(10000);  // dense feature set -> richer point cloud
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    std::cout << "Keypoints: " << kp1.size() << " in left, " << kp2.size() << " in right\n";

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    pts1.clear(); pts2.clear();
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < ratio_thresh * m[1].distance) {
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }
    std::cout << "Good matches after ratio test: " << pts1.size() << "\n";
}

static double reprojection_error(
    const Eigen::Matrix<double, 3, 4>& P,
    const Eigen::Vector3d& X,
    const Eigen::Vector2d& x_obs) {

    Eigen::Vector4d X_h; X_h << X, 1.0;
    Eigen::Vector3d x_proj = P * X_h;
    if (std::abs(x_proj(2)) < 1e-12) return std::numeric_limits<double>::infinity();
    Eigen::Vector2d x_reproj(x_proj(0) / x_proj(2), x_proj(1) / x_proj(2));
    return (x_reproj - x_obs).norm();
}

static void write_points(std::ofstream& f, const std::string& key,
                         const std::vector<Eigen::Vector3d>& pts,
                         const std::vector<int>& indices) {
    f << "  \"" << key << "\": [\n";
    for (size_t i = 0; i < pts.size(); ++i) {
        f << "    {\"i\": " << indices[i]
          << ", \"xyz\": [" << pts[i](0) << ", " << pts[i](1) << ", " << pts[i](2) << "]}";
        f << (i + 1 == pts.size() ? "\n" : ",\n");
    }
    f << "  ]";
}

int main(int argc, char* argv[]) {
    std::cout << "=== Triangulation Demo using OpenGV (KITTI stereo pair) ===\n\n";

    std::string left_path  = (argc >= 3) ? argv[1] : resolveDataPath("left.png");
    std::string right_path = (argc >= 3) ? argv[2] : resolveDataPath("right.png");

    cv::Mat img1 = cv::imread(left_path,  cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: failed to load images:\n  " << left_path << "\n  " << right_path << "\n";
        return 1;
    }
    std::cout << "Left:  " << left_path  << "  (" << img1.cols << "x" << img1.rows << ")\n";
    std::cout << "Right: " << right_path << "  (" << img2.cols << "x" << img2.rows << ")\n\n";

    // KITTI seq 00-02 rectified intrinsics + stereo extrinsic.
    const double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    Eigen::Matrix3d K;
    K << fx, 0,  cx,
         0,  fy, cy,
         0,  0,  1;
    const Eigen::Matrix3d K_inv = K.inverse();

    const double baseline_m = 386.1448 / fx;  // ~0.5372 m
    Eigen::Matrix3d R12 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t12(baseline_m, 0, 0);    // cam1's center in cam0's frame

    // ORB matches + RANSAC inliers.
    std::vector<cv::Point2f> pts1_all, pts2_all;
    detectAndMatchFeatures(img1, img2, pts1_all, pts2_all);
    if (pts1_all.size() < 8) { std::cerr << "Not enough matches.\n"; return 1; }

    cv::Mat inlier_mask;
    cv::findFundamentalMat(pts1_all, pts2_all, cv::FM_RANSAC, 1.5, 0.99, inlier_mask);

    std::vector<cv::Point2f> pts1, pts2;
    for (size_t i = 0; i < pts1_all.size(); ++i) {
        if (inlier_mask.at<uchar>(i)) { pts1.push_back(pts1_all[i]); pts2.push_back(pts2_all[i]); }
    }
    std::cout << "Fundamental RANSAC inliers: " << pts1.size() << "/" << pts1_all.size() << "\n\n";

    // Pixels -> bearing vectors (unit-length 3D directions in each camera).
    opengv::bearingVectors_t bearings1, bearings2;
    bearings1.reserve(pts1.size());
    bearings2.reserve(pts2.size());
    for (size_t i = 0; i < pts1.size(); ++i) {
        Eigen::Vector3d r1 = K_inv * Eigen::Vector3d(pts1[i].x, pts1[i].y, 1.0);
        Eigen::Vector3d r2 = K_inv * Eigen::Vector3d(pts2[i].x, pts2[i].y, 1.0);
        bearings1.push_back(r1.normalized());
        bearings2.push_back(r2.normalized());
    }

    opengv::relative_pose::CentralRelativeAdapter adapter(bearings1, bearings2, t12, R12);

    // Projection matrices (for reprojection-error reporting).
    Eigen::Matrix<double, 3, 4> P1, P2;
    P1.block<3, 3>(0, 0) = K;  P1.col(3).setZero();
    P2.block<3, 3>(0, 0) = K * R12;  P2.col(3) = K * (-R12 * t12);

    const double Z_MIN = 0.2, Z_MAX = 100.0;
    auto cheirality_ok = [&](const Eigen::Vector3d& X) {
        return X(2) > Z_MIN && X(2) < Z_MAX;
    };

    std::vector<Eigen::Vector3d> X_lin, X_mid;
    std::vector<int> idx_lin, idx_mid;
    double err_lin = 0, err_mid = 0;

    for (size_t i = 0; i < pts1.size(); ++i) {
        Eigen::Vector3d Xl = opengv::triangulation::triangulate(adapter, i);   // linear DLT
        Eigen::Vector3d Xm = opengv::triangulation::triangulate2(adapter, i);  // mid-point (closest point between rays)

        if (cheirality_ok(Xl)) {
            X_lin.push_back(Xl); idx_lin.push_back(int(i));
            err_lin += 0.5 * (reprojection_error(P1, Xl, {pts1[i].x, pts1[i].y})
                            + reprojection_error(P2, Xl, {pts2[i].x, pts2[i].y}));
        }
        if (cheirality_ok(Xm)) {
            X_mid.push_back(Xm); idx_mid.push_back(int(i));
            err_mid += 0.5 * (reprojection_error(P1, Xm, {pts1[i].x, pts1[i].y})
                            + reprojection_error(P2, Xm, {pts2[i].x, pts2[i].y}));
        }
    }

    auto report = [](const std::string& name, size_t n, double total_err) {
        std::cout << "  " << std::left << std::setw(26) << name
                  << " kept " << std::setw(4) << n
                  << "  avg reproj err = "
                  << std::fixed << std::setprecision(3)
                  << (n ? total_err / n : 0.0) << " px\n";
    };
    std::cout << "=== Triangulation results (cheirality + depth gate " << Z_MIN << "-" << Z_MAX << " m) ===\n";
    report("OpenGV linear           ", X_lin.size(), err_lin);
    report("OpenGV mid-point        ", X_mid.size(), err_mid);
    std::cout << "\n";

    std::ofstream f("triangulation_opengv.json");
    f << std::fixed << std::setprecision(6);
    f << "{\n";
    f << "  \"left_image\": \"" << left_path << "\",\n";
    f << "  \"right_image\": \"" << right_path << "\",\n";
    f << "  \"width\": " << img1.cols << ", \"height\": " << img1.rows << ",\n";
    f << "  \"fx\": " << fx << ", \"fy\": " << fy << ",\n";
    f << "  \"cx\": " << cx << ", \"cy\": " << cy << ",\n";
    f << "  \"baseline\": " << baseline_m << ",\n";

    f << "  \"keypoints_left\": [";
    for (size_t i = 0; i < pts1.size(); ++i)
        f << "[" << pts1[i].x << ", " << pts1[i].y << "]" << (i + 1 == pts1.size() ? "" : ", ");
    f << "],\n";
    f << "  \"keypoints_right\": [";
    for (size_t i = 0; i < pts2.size(); ++i)
        f << "[" << pts2[i].x << ", " << pts2[i].y << "]" << (i + 1 == pts2.size() ? "" : ", ");
    f << "],\n";

    write_points(f, "opengv_linear",  X_lin, idx_lin);  f << ",\n";
    write_points(f, "opengv_midpoint", X_mid, idx_mid);  f << "\n";
    f << "}\n";

    std::cout << "Wrote triangulation_opengv.json\n";
    std::cout << "Visualize with:  python3 ../viz_triangulation.py triangulation_opengv.json\n";
    return 0;
}
