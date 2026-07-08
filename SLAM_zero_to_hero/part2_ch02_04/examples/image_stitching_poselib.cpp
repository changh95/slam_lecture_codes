/**
 * @file image_stitching_poselib.cpp
 * @brief Image Stitching using PoseLib 4-point Homography Solver
 *
 * Same panorama experiment as image_stitching.cpp (KITTI seq 00 turning
 * pair by default), but the homography is estimated with PoseLib's
 * minimal 4-point solver inside a hand-rolled RANSAC loop.
 *
 * The solver runs in normalized coordinates; the pixel homography is
 * recovered as H_pixel = K * H_norm * K^-1. Any invertible K works for
 * this round trip, so an approximate K built from the image size is
 * used — for a homography, K only acts as data normalization here.
 *
 * PoseLib: https://github.com/PoseLib/PoseLib
 */

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <PoseLib/poselib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

/**
 * @brief Detect and match features
 */
void detectAndMatch(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2) {

    cv::Ptr<cv::ORB> orb = cv::ORB::create(3000);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    std::cout << "  Keypoints: " << kp1.size() << " and " << kp2.size() << std::endl;

    if (desc1.empty() || desc2.empty()) return;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    for (const auto& m : knn_matches) {
        if (m.size() >= 2 && m[0].distance < ratio_thresh * m[1].distance) {
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }

    std::cout << "  Good matches: " << pts1.size() << std::endl;
}

/**
 * @brief Convert pixel coordinates to normalized coordinates
 */
std::vector<Eigen::Vector3d> pixelToNormalized(
    const std::vector<cv::Point2f>& pts,
    double fx, double fy, double cx, double cy) {

    std::vector<Eigen::Vector3d> normalized;
    for (const auto& p : pts) {
        double x = (p.x - cx) / fx;
        double y = (p.y - cy) / fy;
        normalized.emplace_back(x, y, 1.0);
    }
    return normalized;
}

/**
 * @brief Random sample without replacement
 */
std::vector<int> randomSample(int n, int k, std::mt19937& gen) {
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    indices.resize(k);
    return indices;
}

/**
 * @brief Compute reprojection error for homography
 */
double computeReprojError(
    const cv::Point2f& p1,
    const cv::Point2f& p2,
    const Eigen::Matrix3d& H) {

    Eigen::Vector3d x1(p1.x, p1.y, 1.0);
    Eigen::Vector3d x2_pred = H * x1;
    x2_pred /= x2_pred.z();

    double dx = x2_pred.x() - p2.x;
    double dy = x2_pred.y() - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * @brief RANSAC Homography using PoseLib 4-point solver
 */
Eigen::Matrix3d ransacHomographyPoseLib(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    double fx, double fy, double cx, double cy,
    double threshold,
    int max_iterations,
    std::vector<bool>& inlier_mask) {

    // Convert to normalized coordinates for PoseLib
    auto norm1 = pixelToNormalized(pts1, fx, fy, cx, cy);
    auto norm2 = pixelToNormalized(pts2, fx, fy, cx, cy);

    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    Eigen::Matrix3d K_inv = K.inverse();

    std::mt19937 gen(42);
    int best_inliers = 0;
    Eigen::Matrix3d best_H = Eigen::Matrix3d::Identity();

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Sample 4 points
        auto indices = randomSample(static_cast<int>(pts1.size()), 4, gen);

        std::vector<Eigen::Vector3d> x1_sample, x2_sample;
        for (int idx : indices) {
            x1_sample.push_back(norm1[idx]);
            x2_sample.push_back(norm2[idx]);
        }

        // PoseLib 4-point homography (in normalized coordinates)
        Eigen::Matrix3d H_norm;
        int num_solutions = poselib::homography_4pt(x1_sample, x2_sample, &H_norm);

        if (num_solutions == 0) continue;

        // Convert to pixel coordinates: H_pixel = K * H_norm * K^-1
        Eigen::Matrix3d H_pixel = K * H_norm * K_inv;

        // Count inliers
        int inliers = 0;
        for (size_t i = 0; i < pts1.size(); ++i) {
            double error = computeReprojError(pts1[i], pts2[i], H_pixel);
            if (error < threshold) {
                inliers++;
            }
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_H = H_pixel;
        }
    }

    // Compute final inlier mask
    inlier_mask.resize(pts1.size());
    for (size_t i = 0; i < pts1.size(); ++i) {
        double error = computeReprojError(pts1[i], pts2[i], best_H);
        inlier_mask[i] = (error < threshold);
    }

    std::cout << "  PoseLib RANSAC: " << best_inliers << "/" << pts1.size() << " inliers" << std::endl;

    return best_H;
}

/**
 * @brief Convert Eigen matrix to OpenCV matrix
 */
cv::Mat eigenToCvMat(const Eigen::Matrix3d& H_eigen) {
    cv::Mat H_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            H_cv.at<double>(i, j) = H_eigen(i, j);
        }
    }
    return H_cv;
}

/**
 * @brief Compute canvas size
 */
cv::Size computeCanvasSize(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const cv::Mat& H,
    cv::Point2f& offset) {

    std::vector<cv::Point2f> corners2 = {
        {0, 0}, {(float)img2.cols, 0},
        {(float)img2.cols, (float)img2.rows}, {0, (float)img2.rows}
    };

    std::vector<cv::Point2f> corners2_transformed;
    cv::perspectiveTransform(corners2, corners2_transformed, H);

    float min_x = 0, min_y = 0;
    float max_x = (float)img1.cols, max_y = (float)img1.rows;

    for (const auto& pt : corners2_transformed) {
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }

    offset = cv::Point2f(-min_x, -min_y);
    return cv::Size((int)std::ceil(max_x - min_x), (int)std::ceil(max_y - min_y));
}

/**
 * @brief Warp and blend images
 */
cv::Mat warpAndBlend(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const cv::Mat& H,
    const cv::Size& canvas_size,
    const cv::Point2f& offset) {

    cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, offset.x, 0, 1, offset.y, 0, 0, 1);
    cv::Mat H_combined = T * H;

    cv::Mat canvas = cv::Mat::zeros(canvas_size, img1.type());
    cv::Mat img2_warped;
    cv::warpPerspective(img2, img2_warped, H_combined, canvas_size);

    cv::Mat img1_canvas = cv::Mat::zeros(canvas_size, img1.type());
    img1.copyTo(img1_canvas(cv::Rect((int)offset.x, (int)offset.y, img1.cols, img1.rows)));

    // Simple compositing: img1 on top in the overlap (see image_stitching.cpp)
    for (int y = 0; y < canvas_size.height; ++y) {
        for (int x = 0; x < canvas_size.width; ++x) {
            cv::Vec3b p1 = img1_canvas.at<cv::Vec3b>(y, x);
            cv::Vec3b p2 = img2_warped.at<cv::Vec3b>(y, x);

            bool has_p1 = (p1[0] || p1[1] || p1[2]);

            if (has_p1) {
                canvas.at<cv::Vec3b>(y, x) = p1;
            } else {
                canvas.at<cv::Vec3b>(y, x) = p2;
            }
        }
    }

    return canvas;
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================" << std::endl;
    std::cout << "Image Stitching using PoseLib" << std::endl;
    std::cout << "==========================================" << std::endl;

    // Default: KITTI seq 00 turning pair (rotation-dominant -> good panorama)
    std::string path1 = "data/kitti00_turn_003677.png";
    std::string path2 = "data/kitti00_turn_003682.png";
    if (argc >= 3) {
        path1 = argv[1];
        path2 = argv[2];
    }

    cv::Mat img1 = cv::imread(path1);
    cv::Mat img2 = cv::imread(path2);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: could not load " << path1 << " / " << path2 << std::endl;
        std::cout << "Usage: " << argv[0] << " [image1 image2]" << std::endl;
        std::cout << "(run from the chapter root so the default data/ paths resolve)" << std::endl;
        return -1;
    }
    std::cout << "Loaded images: " << path1 << ", " << path2 << std::endl;

    // Approximate camera parameters from the image size; for a
    // homography this only normalizes the data (see file header).
    double fx = img1.cols;
    double fy = img1.cols;
    double cx = img1.cols / 2.0;
    double cy = img1.rows / 2.0;

    std::cout << "\nImage sizes: " << img1.cols << "x" << img1.rows << std::endl;

    // Step 1: Detect and match
    std::cout << "\n--- Step 1: Feature Matching ---" << std::endl;
    std::vector<cv::Point2f> pts1, pts2;
    detectAndMatch(img1, img2, pts1, pts2);

    if (pts1.size() < 4) {
        std::cerr << "Not enough matches!" << std::endl;
        return -1;
    }

    // Step 2: Compute homography using PoseLib
    std::cout << "\n--- Step 2: PoseLib Homography ---" << std::endl;
    std::vector<bool> inlier_mask;
    Eigen::Matrix3d H_eigen = ransacHomographyPoseLib(
        pts2, pts1,  // Note: pts2 -> pts1
        fx, fy, cx, cy,
        3.0,   // threshold in pixels
        2000,  // max iterations
        inlier_mask
    );

    cv::Mat H = eigenToCvMat(H_eigen);
    std::cout << "  Homography:\n" << H << std::endl;

    // Step 3: Canvas size
    std::cout << "\n--- Step 3: Canvas Size ---" << std::endl;
    cv::Point2f offset;
    cv::Size canvas_size = computeCanvasSize(img1, img2, H, offset);
    std::cout << "  Size: " << canvas_size.width << "x" << canvas_size.height
              << " (input " << img1.cols << "x" << img1.rows << ")" << std::endl;

    // Step 4: Warp and blend
    std::cout << "\n--- Step 4: Warping ---" << std::endl;
    cv::Mat panorama = warpAndBlend(img1, img2, H, canvas_size, offset);

    cv::imwrite("panorama_poselib.jpg", panorama);
    std::cout << "\nSaved: panorama_poselib.jpg" << std::endl;

    // Display (only when a display is available)
    if (std::getenv("DISPLAY") != nullptr) {
        cv::imshow("Image 1", img1);
        cv::imshow("Image 2", img2);
        cv::imshow("Panorama (PoseLib)", panorama);
        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0);
    }

    return 0;
}
