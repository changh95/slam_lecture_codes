/**
 * Triangulation Demo
 *
 * Pipeline on a real KITTI stereo pair:
 * 1. ORB feature detection and matching between left/right images
 * 2. Reject outliers with a fundamental-matrix RANSAC pass
 * 3. Triangulate the inlier correspondences using four methods:
 *    - OpenCV cv::triangulatePoints (DLT)
 *    - Custom DLT implementation (Eigen + SVD)
 *    - Mid-point of the two viewing rays
 *    - Stereo disparity depth (rectified formula z = f*b/d)
 * 4. Report average reprojection error per method, drop points that fail the
 *    cheirality check (z <= 0) or a 0.2-100 m depth gate.
 * 5. Write triangulation_demo.json with the keypoints, camera intrinsics /
 *    extrinsics, and the four 3D point clouds for the Rerun viewer.
 *
 * Images default to the bundled data/ pair (KITTI seq 00-02, cam0/cam1).
 * Override with two image paths on the command line.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

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

    auto orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    std::cout << "Keypoints: " << kp1.size() << " in left, " << kp2.size() << " in right\n";

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    pts1.clear();
    pts2.clear();
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < ratio_thresh * m[1].distance) {
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }
    std::cout << "Good matches after ratio test: " << pts1.size() << "\n";
}

static Eigen::Vector3d triangulate_dlt(
    const Eigen::Matrix<double, 3, 4>& P1,
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

static Eigen::Vector3d triangulate_midpoint(
    const Eigen::Vector3d& O1, const Eigen::Vector3d& d1,
    const Eigen::Vector3d& O2, const Eigen::Vector3d& d2,
    bool& is_valid) {

    Eigen::Vector3d w0 = O1 - O2;
    double a = d1.dot(d1);
    double b = d1.dot(d2);
    double c = d2.dot(d2);
    double d = d1.dot(w0);
    double e = d2.dot(w0);
    double denom = a * c - b * b;

    if (std::abs(denom) < 1e-8) { is_valid = false; return Eigen::Vector3d::Zero(); }
    is_valid = true;

    double t = (b * e - c * d) / denom;
    double s = (a * e - b * d) / denom;
    return 0.5 * ((O1 + t * d1) + (O2 + s * d2));
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
    std::cout << "=== Triangulation Demo (KITTI stereo pair) ===\n\n";

    std::string left_path  = (argc >= 3) ? argv[1] : resolveDataPath("left.png");
    std::string right_path = (argc >= 3) ? argv[2] : resolveDataPath("right.png");

    cv::Mat img1 = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: failed to load images:\n  " << left_path << "\n  " << right_path
                  << "\nRun from build/ so ../data resolves, or pass two image paths.\n";
        return 1;
    }
    std::cout << "Left:  " << left_path  << "  (" << img1.cols << "x" << img1.rows << ")\n";
    std::cout << "Right: " << right_path << "  (" << img2.cols << "x" << img2.rows << ")\n\n";

    // KITTI seq 00-02 rectified intrinsics
    const double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    Eigen::Matrix3d K;
    K << fx, 0,  cx,
         0,  fy, cy,
         0,  0,  1;

    // Cam1 is to the right of cam0 along +X in cam0's frame.
    const double baseline_m = 386.1448 / fx;  // ~0.5372 m, from KITTI P_rect_01.
    Eigen::Vector3d C1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d C2(baseline_m, 0, 0);
    Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R2 = Eigen::Matrix3d::Identity();

    // Projection matrices: P = K [R | -R*C].
    Eigen::Matrix<double, 3, 4> P1, P2;
    P1.block<3, 3>(0, 0) = K * R1;  P1.col(3) = K * (-R1 * C1);
    P2.block<3, 3>(0, 0) = K * R2;  P2.col(3) = K * (-R2 * C2);

    cv::Mat P1_cv = (cv::Mat_<float>(3, 4) <<
        P1(0,0),P1(0,1),P1(0,2),P1(0,3),
        P1(1,0),P1(1,1),P1(1,2),P1(1,3),
        P1(2,0),P1(2,1),P1(2,2),P1(2,3));
    cv::Mat P2_cv = (cv::Mat_<float>(3, 4) <<
        P2(0,0),P2(0,1),P2(0,2),P2(0,3),
        P2(1,0),P2(1,1),P2(1,2),P2(1,3),
        P2(2,0),P2(2,1),P2(2,2),P2(2,3));

    // ORB matches + RANSAC fundamental-matrix outlier rejection.
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

    const double Z_MIN = 0.2, Z_MAX = 100.0;
    const Eigen::Matrix3d K_inv = K.inverse();

    auto cheirality_ok = [&](const Eigen::Vector3d& X) {
        return X(2) > Z_MIN && X(2) < Z_MAX;
    };

    // -------- Method 1: OpenCV cv::triangulatePoints --------
    cv::Mat pts_4d;
    cv::triangulatePoints(P1_cv, P2_cv, pts1, pts2, pts_4d);

    std::vector<Eigen::Vector3d> X_opencv, X_dlt, X_mid, X_stereo;
    std::vector<int> idx_opencv, idx_dlt, idx_mid, idx_stereo;
    double err_opencv = 0, err_dlt = 0, err_mid = 0, err_stereo = 0;

    for (size_t i = 0; i < pts1.size(); ++i) {
        // OpenCV DLT
        float w = pts_4d.at<float>(3, i);
        Eigen::Vector3d Xc(pts_4d.at<float>(0, i) / w,
                           pts_4d.at<float>(1, i) / w,
                           pts_4d.at<float>(2, i) / w);
        if (cheirality_ok(Xc)) {
            X_opencv.push_back(Xc); idx_opencv.push_back(int(i));
            err_opencv += 0.5 * (reprojection_error(P1, Xc, {pts1[i].x, pts1[i].y})
                               + reprojection_error(P2, Xc, {pts2[i].x, pts2[i].y}));
        }

        // Custom DLT
        Eigen::Vector3d Xd = triangulate_dlt(P1, P2,
            {pts1[i].x, pts1[i].y}, {pts2[i].x, pts2[i].y});
        if (cheirality_ok(Xd)) {
            X_dlt.push_back(Xd); idx_dlt.push_back(int(i));
            err_dlt += 0.5 * (reprojection_error(P1, Xd, {pts1[i].x, pts1[i].y})
                            + reprojection_error(P2, Xd, {pts2[i].x, pts2[i].y}));
        }

        // Mid-point: rays in world = R^T K^-1 [u v 1]^T, from camera centers C1/C2.
        Eigen::Vector3d d1 = (R1.transpose() *
            (K_inv * Eigen::Vector3d(pts1[i].x, pts1[i].y, 1.0))).normalized();
        Eigen::Vector3d d2 = (R2.transpose() *
            (K_inv * Eigen::Vector3d(pts2[i].x, pts2[i].y, 1.0))).normalized();
        bool ok = false;
        Eigen::Vector3d Xm = triangulate_midpoint(C1, d1, C2, d2, ok);
        if (ok && cheirality_ok(Xm)) {
            X_mid.push_back(Xm); idx_mid.push_back(int(i));
            err_mid += 0.5 * (reprojection_error(P1, Xm, {pts1[i].x, pts1[i].y})
                            + reprojection_error(P2, Xm, {pts2[i].x, pts2[i].y}));
        }

        // Stereo disparity (rectified): z = f*b/d, then back-project the left observation.
        double disp = pts1[i].x - pts2[i].x;
        if (disp > 1e-3) {
            double Z = (fx * baseline_m) / disp;
            double X = (pts1[i].x - cx) * Z / fx;
            double Y = (pts1[i].y - cy) * Z / fy;
            Eigen::Vector3d Xs(X, Y, Z);
            if (cheirality_ok(Xs)) {
                X_stereo.push_back(Xs); idx_stereo.push_back(int(i));
                err_stereo += 0.5 * (reprojection_error(P1, Xs, {pts1[i].x, pts1[i].y})
                                   + reprojection_error(P2, Xs, {pts2[i].x, pts2[i].y}));
            }
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
    report("OpenCV triangulatePoints", X_opencv.size(), err_opencv);
    report("Custom DLT (Eigen SVD)  ", X_dlt.size(),    err_dlt);
    report("Mid-point of rays       ", X_mid.size(),    err_mid);
    report("Stereo disparity depth  ", X_stereo.size(), err_stereo);
    std::cout << "\n";

    // -------- Write JSON for the Rerun viewer --------
    std::ofstream f("triangulation_demo.json");
    f << std::fixed << std::setprecision(6);
    f << "{\n";
    f << "  \"left_image\": \"" << left_path << "\",\n";
    f << "  \"right_image\": \"" << right_path << "\",\n";
    f << "  \"width\": " << img1.cols << ", \"height\": " << img1.rows << ",\n";
    f << "  \"fx\": " << fx << ", \"fy\": " << fy << ",\n";
    f << "  \"cx\": " << cx << ", \"cy\": " << cy << ",\n";
    f << "  \"baseline\": " << baseline_m << ",\n";

    f << "  \"keypoints_left\": [";
    for (size_t i = 0; i < pts1.size(); ++i) {
        f << "[" << pts1[i].x << ", " << pts1[i].y << "]"
          << (i + 1 == pts1.size() ? "" : ", ");
    }
    f << "],\n";
    f << "  \"keypoints_right\": [";
    for (size_t i = 0; i < pts2.size(); ++i) {
        f << "[" << pts2[i].x << ", " << pts2[i].y << "]"
          << (i + 1 == pts2.size() ? "" : ", ");
    }
    f << "],\n";

    write_points(f, "opencv",    X_opencv,  idx_opencv);  f << ",\n";
    write_points(f, "dlt",       X_dlt,     idx_dlt);     f << ",\n";
    write_points(f, "midpoint",  X_mid,     idx_mid);     f << ",\n";
    write_points(f, "stereo",    X_stereo,  idx_stereo);  f << "\n";
    f << "}\n";

    std::cout << "Wrote triangulation_demo.json\n";
    std::cout << "Visualize with:  python3 ../viz_triangulation.py triangulation_demo.json\n";
    return 0;
}
