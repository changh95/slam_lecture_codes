/**
 * Pose Recovery from Essential Matrix
 *
 * This example demonstrates:
 * 1. Feature detection and matching between images
 * 2. Essential matrix estimation with RANSAC
 * 3. Decomposition of E into four possible (R, t) pairs
 * 4. Cheirality check to select the correct solution
 *
 * The Essential matrix can be decomposed into:
 *   E = [t]_x * R
 *
 * where [t]_x is the skew-symmetric matrix of t.
 * There are 4 possible solutions, but only one has all
 * 3D points in front of both cameras (cheirality).
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>

/**
 * Detect and match ORB features between two images
 */
void detectAndMatchFeatures(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2,
    std::vector<cv::DMatch>& good_matches) {

    // Create ORB detector
    auto orb = cv::ORB::create(2000);

    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    std::cout << "Keypoints: " << kp1.size() << " in image 1, "
              << kp2.size() << " in image 2" << std::endl;

    // Match descriptors using BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    // Lowe's ratio test
    const float ratio_thresh = 0.75f;
    for (const auto& m : knn_matches) {
        if (m[0].distance < ratio_thresh * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    std::cout << "Good matches: " << good_matches.size() << std::endl;

    // Extract matched points
    pts1.clear();
    pts2.clear();
    for (const auto& m : good_matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }
}

/**
 * Triangulate a single point from two views
 */
cv::Mat triangulatePoint(
    const cv::Mat& P1,
    const cv::Mat& P2,
    const cv::Point2f& pt1,
    const cv::Point2f& pt2) {

    cv::Mat A(4, 4, CV_64F);

    // DLT method
    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);

    cv::Mat U, S, Vt;
    cv::SVD::compute(A, S, U, Vt);

    cv::Mat X = Vt.row(3).t();
    X = X / X.at<double>(3);  // Normalize

    return X.rowRange(0, 3).clone();
}

/**
 * Check cheirality (points in front of camera)
 */
int countPointsInFront(
    const cv::Mat& R,
    const cv::Mat& t,
    const cv::Mat& K,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2) {

    // Projection matrices
    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);  // [I | 0]

    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    cv::Mat P2 = K * Rt;

    int count = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat X = triangulatePoint(P1, P2, pts1[i], pts2[i]);

        // Check depth in camera 1
        double z1 = X.at<double>(2);

        // Transform to camera 2 and check depth
        cv::Mat X2 = R * X + t;
        double z2 = X2.at<double>(2);

        if (z1 > 0 && z2 > 0) {
            count++;
        }
    }

    return count;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Pose Recovery from Essential Matrix ===\n" << std::endl;

    cv::Mat img1, img2;

    if (argc >= 3) {
        // Load images from arguments
        img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty()) {
            std::cerr << "Failed to load images!" << std::endl;
            return 1;
        }
    } else {
        // Create synthetic test images
        std::cout << "No images provided, using synthetic test images." << std::endl;
        std::cout << "Usage: " << argv[0] << " <image1> <image2>" << std::endl;
        std::cout << std::endl;

        // Create simple synthetic images with features
        img1 = cv::Mat::zeros(480, 640, CV_8UC1);
        img2 = cv::Mat::zeros(480, 640, CV_8UC1);

        // Draw random patterns
        cv::RNG rng(42);
        for (int i = 0; i < 100; ++i) {
            int x = rng.uniform(50, 590);
            int y = rng.uniform(50, 430);
            int r = rng.uniform(5, 20);
            cv::circle(img1, cv::Point(x, y), r, cv::Scalar(255), -1);
            // Slightly shifted in img2 (simulating camera motion)
            cv::circle(img2, cv::Point(x + 20, y + 5), r, cv::Scalar(255), -1);
        }

        // Add some structure
        cv::rectangle(img1, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(200), 2);
        cv::rectangle(img2, cv::Point(120, 105), cv::Point(220, 205), cv::Scalar(200), 2);
    }

    std::cout << "Image size: " << img1.cols << "x" << img1.rows << std::endl;
    std::cout << std::endl;

    // Camera intrinsics (assumed or from calibration)
    double fx = 500, fy = 500;
    double cx = img1.cols / 2.0, cy = img1.rows / 2.0;
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1);

    std::cout << "Camera Intrinsics K:\n" << K << std::endl << std::endl;

    // Detect and match features
    std::vector<cv::Point2f> pts1, pts2;
    std::vector<cv::DMatch> matches;
    detectAndMatchFeatures(img1, img2, pts1, pts2, matches);

    if (pts1.size() < 8) {
        std::cerr << "Not enough matches for Essential matrix estimation!" << std::endl;
        return 1;
    }

    // Estimate Essential matrix
    std::cout << "\n=== Essential Matrix Estimation ===" << std::endl;

    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(
        pts1, pts2, K,
        cv::RANSAC,
        0.999,      // Confidence
        1.0,        // RANSAC threshold in pixels
        inlier_mask
    );

    int num_inliers = cv::countNonZero(inlier_mask);
    std::cout << "RANSAC inliers: " << num_inliers << "/" << pts1.size() << std::endl;
    std::cout << "Essential Matrix E:\n" << E << std::endl;

    // Verify Essential matrix properties
    cv::Mat U, S, Vt;
    cv::SVD::compute(E, S, U, Vt);
    std::cout << "\nSVD of E:" << std::endl;
    std::cout << "  Singular values: " << S.t() << std::endl;

    // E should have rank 2 with two equal singular values
    double sv_ratio = S.at<double>(0) / S.at<double>(1);
    std::cout << "  s1/s2 ratio: " << sv_ratio << " (should be ~1)" << std::endl;
    std::cout << "  s3: " << S.at<double>(2) << " (should be ~0)" << std::endl;

    // Decompose Essential matrix using OpenCV
    std::cout << "\n=== Pose Decomposition ===" << std::endl;

    cv::Mat R1, R2, t_decomp;
    cv::decomposeEssentialMat(E, R1, R2, t_decomp);

    std::cout << "Possible Rotations:" << std::endl;
    std::cout << "R1:\n" << R1 << std::endl;
    std::cout << "R2:\n" << R2 << std::endl;
    std::cout << "t (direction):\n" << t_decomp.t() << std::endl;

    // Four possible solutions: (R1, t), (R1, -t), (R2, t), (R2, -t)
    std::cout << "\n=== Cheirality Check ===" << std::endl;

    // Filter points using inlier mask
    std::vector<cv::Point2f> pts1_inliers, pts2_inliers;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (inlier_mask.at<uchar>(i)) {
            pts1_inliers.push_back(pts1[i]);
            pts2_inliers.push_back(pts2[i]);
        }
    }

    // Test all four solutions
    std::vector<std::pair<cv::Mat, cv::Mat>> solutions = {
        {R1, t_decomp},
        {R1, -t_decomp},
        {R2, t_decomp},
        {R2, -t_decomp}
    };

    int best_count = 0;
    int best_idx = 0;

    for (size_t i = 0; i < solutions.size(); ++i) {
        int count = countPointsInFront(
            solutions[i].first, solutions[i].second, K,
            pts1_inliers, pts2_inliers);

        std::cout << "Solution " << (i + 1) << ": "
                  << count << "/" << pts1_inliers.size()
                  << " points in front" << std::endl;

        if (count > best_count) {
            best_count = count;
            best_idx = i;
        }
    }

    std::cout << "\nBest solution: " << (best_idx + 1)
              << " (" << best_count << " valid points)" << std::endl;

    // Use recoverPose for verified solution
    std::cout << "\n=== OpenCV recoverPose ===" << std::endl;

    cv::Mat R_final, t_final;
    int valid_points = cv::recoverPose(E, pts1, pts2, K, R_final, t_final, inlier_mask);

    std::cout << "Valid points (cheirality): " << valid_points << std::endl;
    std::cout << "\nFinal Rotation:\n" << R_final << std::endl;
    std::cout << "\nFinal Translation (unit):\n" << t_final.t() << std::endl;

    // Extract rotation angles
    cv::Mat rvec;
    cv::Rodrigues(R_final, rvec);
    double angle = cv::norm(rvec) * 180.0 / CV_PI;

    std::cout << "\n=== Interpretation ===" << std::endl;
    std::cout << "Rotation angle: " << angle << " degrees" << std::endl;
    std::cout << "Translation direction: [" << t_final.at<double>(0)
              << ", " << t_final.at<double>(1)
              << ", " << t_final.at<double>(2) << "]" << std::endl;
    std::cout << "(Note: translation scale is unknown from Essential matrix alone)"
              << std::endl;

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}
