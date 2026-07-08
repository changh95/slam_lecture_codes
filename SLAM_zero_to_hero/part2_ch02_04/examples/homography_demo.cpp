/**
 * @file homography_demo.cpp
 * @brief Homography Estimation and Decomposition on Real Images
 *
 * Part 1 — Estimation (Oxford VGG "wall" dataset, planar brick wall):
 *   1. Match ORB features between two views of a planar scene
 *   2. Estimate H with plain DLT (all matches) and with RANSAC
 *   3. Compare both against the dataset's ground-truth homography
 *   4. Warp image 1 into image 3 and blend to verify alignment
 *
 * Part 2 — Decomposition (KITTI odometry seq 00, turning vehicle):
 *   1. Estimate H between two frames of a ~21 degree turn
 *   2. Decompose H into {R, t/d, n} with the real KITTI intrinsics
 *   3. Compare the recovered rotation against the KITTI ground-truth poses
 *
 * A homography H relates points on a plane (or any points under pure
 * rotation) between two views:
 *   x' = H * x,   H = K * (R - t*n^T/d) * K^(-1)
 * where n is the plane normal and d the distance to the plane.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Detect ORB features and match with ratio test
 */
static void detectAndMatch(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2,
    float ratio_thresh) {

    cv::Ptr<cv::ORB> orb = cv::ORB::create(5000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    pts1.clear();
    pts2.clear();
    for (const auto& m : knn_matches) {
        if (m.size() >= 2 && m[0].distance < ratio_thresh * m[1].distance) {
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }
}

/**
 * @brief Load a 3x3 matrix from a whitespace-separated text file
 */
static cv::Mat load3x3(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Error: could not open " << path << std::endl;
        std::exit(1);
    }
    cv::Mat M(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) file >> M.at<double>(i / 3, i % 3);
    return M;
}

/**
 * @brief Load KITTI pose rows (3x4 row-major per line) as 4x4 matrices
 */
static std::vector<cv::Mat> loadKittiPoses(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Error: could not open " << path << std::endl;
        std::exit(1);
    }
    std::vector<cv::Mat> poses;
    double v[12];
    while (file >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5]
                >> v[6] >> v[7] >> v[8] >> v[9] >> v[10] >> v[11]) {
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        for (int i = 0; i < 12; ++i) T.at<double>(i / 4, i % 4) = v[i];
        poses.push_back(T);
    }
    return poses;
}

/**
 * @brief Mean transfer error between two homographies over an image grid
 *
 * Compares where H_est and H_gt send the same grid of points; this
 * measures the estimate against the ground truth over the whole image,
 * not just at the matched features.
 */
static double gridErrorVsGroundTruth(
    const cv::Mat& H_est, const cv::Mat& H_gt, const cv::Size& img_size) {

    std::vector<cv::Point2f> grid;
    for (int y = 0; y < 12; ++y)
        for (int x = 0; x < 20; ++x)
            grid.emplace_back(x * (img_size.width - 1) / 19.0f,
                              y * (img_size.height - 1) / 11.0f);

    std::vector<cv::Point2f> proj_est, proj_gt;
    cv::perspectiveTransform(grid, proj_est, H_est);
    cv::perspectiveTransform(grid, proj_gt, H_gt);

    double total = 0;
    for (size_t i = 0; i < grid.size(); ++i)
        total += cv::norm(proj_est[i] - proj_gt[i]);
    return total / grid.size();
}

/**
 * @brief Rotation angle (degrees) of a rotation matrix
 */
static double rotationAngleDeg(const cv::Mat& R) {
    double tr = std::min(3.0, std::max(-1.0, cv::trace(R)[0]));
    return std::acos((tr - 1.0) / 2.0) * 180.0 / CV_PI;
}

int main(int argc, char* argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    std::cout << "=== Homography Estimation and Decomposition (real images) ===\n"
              << std::endl;

    // =========================================================
    // Part 1: Estimation on a planar scene (Oxford VGG "wall")
    // =========================================================
    std::cout << "--- Part 1: Estimation vs ground truth (wall, planar) ---"
              << std::endl;

    cv::Mat wall1 = cv::imread(data_dir + "/wall_img1.png");
    cv::Mat wall3 = cv::imread(data_dir + "/wall_img3.png");
    if (wall1.empty() || wall3.empty()) {
        std::cerr << "Error: could not load wall images from " << data_dir
                  << " (run from the chapter root, or pass the data dir as arg)"
                  << std::endl;
        return 1;
    }
    cv::Mat H_gt = load3x3(data_dir + "/wall_H1to3p.txt");
    H_gt /= H_gt.at<double>(2, 2);

    // Loose ratio test on purpose: keeps ~10% natural mismatches so the
    // DLT-vs-RANSAC comparison below is meaningful.
    std::vector<cv::Point2f> pts1, pts3;
    detectAndMatch(wall1, wall3, pts1, pts3, 0.85f);
    std::cout << "Matches (ratio 0.85): " << pts1.size() << std::endl;

    if (pts1.size() < 10) {
        std::cerr << "Error: not enough matches!" << std::endl;
        return 1;
    }

    // DLT on all matches (no robustness)
    cv::Mat H_dlt = cv::findHomography(pts1, pts3, 0);

    // RANSAC
    cv::Mat inlier_mask;
    cv::Mat H_ransac =
        cv::findHomography(pts1, pts3, cv::RANSAC, 3.0, inlier_mask, 2000, 0.995);
    int num_inliers = cv::countNonZero(inlier_mask);

    std::cout << "RANSAC inliers: " << num_inliers << "/" << pts1.size()
              << " (" << 100.0 * num_inliers / pts1.size() << "%)" << std::endl;
    std::cout << "\nGround-truth H (VGG H1to3p):\n" << H_gt << std::endl;
    std::cout << "\nEstimated H (RANSAC):\n"
              << H_ransac / H_ransac.at<double>(2, 2) << std::endl;

    double err_dlt = gridErrorVsGroundTruth(H_dlt, H_gt, wall1.size());
    double err_ransac = gridErrorVsGroundTruth(H_ransac, H_gt, wall1.size());
    std::cout << "\nMean grid transfer error vs ground truth:" << std::endl;
    std::cout << "  DLT (all matches, ~10% outliers): " << err_dlt << " px"
              << std::endl;
    std::cout << "  RANSAC:                           " << err_ransac << " px"
              << std::endl;

    // Warp wall1 into wall3's frame and blend to verify alignment
    cv::Mat warped, blend;
    cv::warpPerspective(wall1, warped, H_ransac, wall3.size());
    cv::addWeighted(wall3, 0.5, warped, 0.5, 0, blend);
    cv::imwrite("wall_warped.png", warped);
    cv::imwrite("wall_blend.png", blend);
    std::cout << "\nSaved: wall_warped.png, wall_blend.png "
                 "(blend should show a sharp, aligned wall)" << std::endl;

    if (std::getenv("DISPLAY") != nullptr) {
        cv::imshow("wall img3", wall3);
        cv::imshow("wall img1 warped into img3 (RANSAC H)", warped);
        cv::imshow("50/50 blend (sharp bricks = sub-pixel H)", blend);
        std::cout << "Press any key to continue..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    // =========================================================
    // Part 2: Decomposition on KITTI (turning vehicle)
    // =========================================================
    std::cout << "\n--- Part 2: Decomposition vs KITTI ground truth (turn) ---"
              << std::endl;

    cv::Mat kitti1 = cv::imread(data_dir + "/kitti00_turn_003677.png");
    cv::Mat kitti2 = cv::imread(data_dir + "/kitti00_turn_003682.png");
    if (kitti1.empty() || kitti2.empty()) {
        std::cerr << "Error: could not load KITTI turn images" << std::endl;
        return 1;
    }

    // KITTI odometry seq 00, camera 0 intrinsics (P0 in calib.txt)
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        718.856, 0.0, 607.1928,
        0.0, 718.856, 185.2157,
        0.0, 0.0, 1.0);

    std::vector<cv::Point2f> kpts1, kpts2;
    detectAndMatch(kitti1, kitti2, kpts1, kpts2, 0.75f);
    std::cout << "Matches: " << kpts1.size() << std::endl;

    cv::Mat kitti_mask;
    cv::Mat H_kitti =
        cv::findHomography(kpts1, kpts2, cv::RANSAC, 3.0, kitti_mask, 2000, 0.995);
    std::cout << "RANSAC inliers: " << cv::countNonZero(kitti_mask) << "/"
              << kpts1.size() << std::endl;

    // Ground truth relative motion from the KITTI poses (frames 3677, 3682)
    auto poses = loadKittiPoses(data_dir + "/kitti00_turn_poses.txt");
    if (poses.size() != 2) {
        std::cerr << "Error: expected 2 poses in kitti00_turn_poses.txt"
                  << std::endl;
        return 1;
    }
    cv::Mat T_rel = poses[0].inv() * poses[1];
    cv::Mat R_gt = T_rel(cv::Rect(0, 0, 3, 3));
    cv::Mat t_gt = T_rel(cv::Rect(3, 0, 1, 3));
    std::cout << "\nGround truth (KITTI poses, frame 3677 -> 3682):" << std::endl;
    std::cout << "  Rotation: " << rotationAngleDeg(R_gt) << " degrees"
              << std::endl;
    std::cout << "  Translation: " << cv::norm(t_gt) << " m" << std::endl;

    // Decompose H = K (R - t n^T / d) K^-1  ->  4 mathematical solutions
    std::vector<cv::Mat> Rs, ts, normals;
    int num_solutions = cv::decomposeHomographyMat(H_kitti, K, Rs, ts, normals);
    std::cout << "\nDecomposition solutions: " << num_solutions
              << " (twofold ambiguity; cheirality + a reference plane resolve it)"
              << std::endl;

    for (int i = 0; i < num_solutions; ++i) {
        std::cout << "  Solution " << (i + 1) << ": rotation = "
                  << rotationAngleDeg(Rs[i]) << " deg"
                  << ", t/d = " << ts[i].t()
                  << ", n = " << normals[i].t() << std::endl;
    }
    std::cout << "\nNote: the scene is not planar, but the turn is "
                 "rotation-dominant and the buildings are distant, so the "
                 "recovered rotation is close to the ground truth. The "
                 "translation is only known up to the plane distance d."
              << std::endl;

    std::cout << "\n=== Demo Complete ===" << std::endl;
    return 0;
}
