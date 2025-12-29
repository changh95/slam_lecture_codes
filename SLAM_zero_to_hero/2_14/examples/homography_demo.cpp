/**
 * Homography Estimation and Decomposition Demo
 *
 * This example demonstrates:
 * 1. Estimating homography from point correspondences
 * 2. Homography decomposition into rotation, translation, and normal
 * 3. Using homography for image warping and rectification
 * 4. RANSAC-based robust homography estimation
 *
 * A homography H relates points on a plane between two views:
 *   x' = H * x
 *
 * For calibrated cameras:
 *   H = K * (R - t*n'/d) * K^(-1)
 *
 * where n' is the plane normal and d is the distance to the plane.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

/**
 * Generate synthetic planar points and their transformed projections
 */
void generatePlanarCorrespondences(
    const cv::Mat& H,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2,
    int num_points = 50,
    double noise_sigma = 0.0) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> x_dist(100, 540);
    std::uniform_real_distribution<> y_dist(100, 380);
    std::normal_distribution<> noise(0, noise_sigma);

    pts1.clear();
    pts2.clear();

    for (int i = 0; i < num_points; ++i) {
        cv::Point2f p1(x_dist(gen), y_dist(gen));

        // Transform point using homography
        cv::Mat p1_h = (cv::Mat_<double>(3, 1) << p1.x, p1.y, 1.0);
        cv::Mat p2_h = H * p1_h;

        cv::Point2f p2(p2_h.at<double>(0) / p2_h.at<double>(2),
                       p2_h.at<double>(1) / p2_h.at<double>(2));

        // Add noise
        if (noise_sigma > 0) {
            p1.x += noise(gen);
            p1.y += noise(gen);
            p2.x += noise(gen);
            p2.y += noise(gen);
        }

        pts1.push_back(p1);
        pts2.push_back(p2);
    }
}

/**
 * Verify homography: compute reprojection error
 */
double computeHomographyError(
    const cv::Mat& H,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2) {

    double total_error = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat p1_h = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2_h = H * p1_h;

        cv::Point2f p2_proj(p2_h.at<double>(0) / p2_h.at<double>(2),
                            p2_h.at<double>(1) / p2_h.at<double>(2));

        double dx = p2_proj.x - pts2[i].x;
        double dy = p2_proj.y - pts2[i].y;
        total_error += std::sqrt(dx * dx + dy * dy);
    }

    return total_error / pts1.size();
}

int main(int argc, char* argv[]) {
    std::cout << "=== Homography Estimation and Decomposition ===\n" << std::endl;

    // Camera intrinsic matrix
    double fx = 500, fy = 500;
    double cx = 320, cy = 240;
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1);

    std::cout << "Camera Intrinsics K:\n" << K << std::endl << std::endl;

    // Create a ground truth homography
    // H = K * (R - t*n'/d) * K^(-1)

    // Rotation (15 degrees around Z axis)
    double angle = 15.0 * CV_PI / 180.0;
    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        std::cos(angle), -std::sin(angle), 0,
        std::sin(angle), std::cos(angle), 0,
        0, 0, 1);

    // Translation (camera moves along X)
    cv::Mat t = (cv::Mat_<double>(3, 1) << 0.3, 0.0, 0.0);

    // Plane normal (ground plane, pointing up)
    cv::Mat n = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);

    // Distance to plane
    double d = 5.0;

    // Compute homography
    cv::Mat tn = t * n.t();
    cv::Mat H_normalized = R - tn / d;
    cv::Mat K_inv = K.inv();
    cv::Mat H_gt = K * H_normalized * K_inv;

    // Normalize so H[2,2] = 1
    H_gt = H_gt / H_gt.at<double>(2, 2);

    std::cout << "Ground Truth:" << std::endl;
    std::cout << "  Rotation angle: " << angle * 180.0 / CV_PI << " degrees" << std::endl;
    std::cout << "  Translation: " << t.t() << std::endl;
    std::cout << "  Plane normal: " << n.t() << std::endl;
    std::cout << "  Plane distance: " << d << std::endl;
    std::cout << "\nGround Truth Homography:\n" << H_gt << std::endl;
    std::cout << std::endl;

    // Generate correspondences
    std::vector<cv::Point2f> pts1, pts2;
    generatePlanarCorrespondences(H_gt, pts1, pts2, 100, 1.0);  // 1 pixel noise

    std::cout << "Generated " << pts1.size() << " point correspondences"
              << std::endl << std::endl;

    // =========================================================
    // Method 1: DLT (Direct Linear Transform)
    // =========================================================
    std::cout << "=== DLT Homography Estimation ===" << std::endl;

    cv::Mat H_dlt = cv::findHomography(pts1, pts2, 0);  // Method 0 = DLT
    H_dlt = H_dlt / H_dlt.at<double>(2, 2);

    double error_dlt = computeHomographyError(H_dlt, pts1, pts2);
    std::cout << "Estimated H (DLT):\n" << H_dlt << std::endl;
    std::cout << "Mean reprojection error: " << error_dlt << " pixels" << std::endl;
    std::cout << std::endl;

    // =========================================================
    // Method 2: RANSAC (Robust to outliers)
    // =========================================================
    std::cout << "=== RANSAC Homography Estimation ===" << std::endl;

    // Add outliers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> outlier_dist(0, 640);

    std::vector<cv::Point2f> pts1_outliers = pts1;
    std::vector<cv::Point2f> pts2_outliers = pts2;

    int num_outliers = pts1.size() * 0.2;  // 20% outliers
    for (int i = 0; i < num_outliers; ++i) {
        pts1_outliers.push_back(cv::Point2f(outlier_dist(gen), outlier_dist(gen)));
        pts2_outliers.push_back(cv::Point2f(outlier_dist(gen), outlier_dist(gen)));
    }

    cv::Mat inlier_mask;
    cv::Mat H_ransac = cv::findHomography(
        pts1_outliers, pts2_outliers,
        cv::RANSAC,     // Method
        3.0,            // RANSAC threshold
        inlier_mask,
        2000,           // Max iterations
        0.995           // Confidence
    );
    H_ransac = H_ransac / H_ransac.at<double>(2, 2);

    int num_inliers = cv::countNonZero(inlier_mask);
    double error_ransac = computeHomographyError(H_ransac, pts1, pts2);

    std::cout << "Inliers: " << num_inliers << "/" << pts1_outliers.size() << std::endl;
    std::cout << "Estimated H (RANSAC):\n" << H_ransac << std::endl;
    std::cout << "Mean reprojection error: " << error_ransac << " pixels" << std::endl;
    std::cout << std::endl;

    // =========================================================
    // Homography Decomposition
    // =========================================================
    std::cout << "=== Homography Decomposition ===" << std::endl;

    std::vector<cv::Mat> Rs, ts, normals;
    int num_solutions = cv::decomposeHomographyMat(
        H_ransac, K, Rs, ts, normals);

    std::cout << "Number of solutions: " << num_solutions << std::endl;

    for (int i = 0; i < num_solutions; ++i) {
        std::cout << "\nSolution " << (i + 1) << ":" << std::endl;

        // Rotation to angle-axis
        cv::Mat rvec;
        cv::Rodrigues(Rs[i], rvec);
        double rot_angle = cv::norm(rvec) * 180.0 / CV_PI;

        std::cout << "  Rotation angle: " << rot_angle << " degrees" << std::endl;
        std::cout << "  Translation: " << ts[i].t() << std::endl;
        std::cout << "  Normal: " << normals[i].t() << std::endl;

        // Check if solution is physically plausible
        // Normal should point towards camera (z > 0 in camera frame)
        bool valid = normals[i].at<double>(2) > 0;
        std::cout << "  Physically valid: " << (valid ? "Yes" : "No") << std::endl;
    }
    std::cout << std::endl;

    // =========================================================
    // Image Warping Demo
    // =========================================================
    std::cout << "=== Image Warping Demo ===" << std::endl;

    // Create a synthetic image with a checkerboard pattern
    cv::Mat src_img(480, 640, CV_8UC3, cv::Scalar(200, 200, 200));

    // Draw checkerboard
    int square_size = 40;
    for (int y = 0; y < src_img.rows; y += square_size) {
        for (int x = 0; x < src_img.cols; x += square_size) {
            if (((x / square_size) + (y / square_size)) % 2 == 0) {
                cv::rectangle(src_img,
                    cv::Point(x, y),
                    cv::Point(x + square_size, y + square_size),
                    cv::Scalar(50, 50, 50), -1);
            }
        }
    }

    // Add some circles for reference
    cv::circle(src_img, cv::Point(320, 240), 50, cv::Scalar(0, 0, 255), 3);
    cv::circle(src_img, cv::Point(200, 150), 30, cv::Scalar(0, 255, 0), 3);
    cv::circle(src_img, cv::Point(450, 350), 40, cv::Scalar(255, 0, 0), 3);

    // Warp image using homography
    cv::Mat dst_img;
    cv::warpPerspective(src_img, dst_img, H_gt, src_img.size());

    // Save images
    cv::imwrite("homography_original.png", src_img);
    cv::imwrite("homography_warped.png", dst_img);

    std::cout << "Saved images:" << std::endl;
    std::cout << "  homography_original.png" << std::endl;
    std::cout << "  homography_warped.png" << std::endl;

    // Combine for visualization
    cv::Mat combined;
    cv::hconcat(src_img, dst_img, combined);
    cv::imwrite("homography_comparison.png", combined);
    std::cout << "  homography_comparison.png" << std::endl;

    // =========================================================
    // Inverse Warping (Rectification)
    // =========================================================
    std::cout << "\n=== Inverse Warping ===" << std::endl;

    cv::Mat H_inv = H_gt.inv();
    cv::Mat rectified;
    cv::warpPerspective(dst_img, rectified, H_inv, dst_img.size());

    cv::imwrite("homography_rectified.png", rectified);
    std::cout << "Saved: homography_rectified.png (should match original)" << std::endl;

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}
