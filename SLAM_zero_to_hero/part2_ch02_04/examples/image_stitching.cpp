/**
 * @file image_stitching.cpp
 * @brief Image Stitching / Panorama Demo using OpenCV
 *
 * Stitches two real KITTI frames into a panorama using homography
 * estimation. The default pair (seq 00, frames 3677 -> 3682) is taken
 * while the car turns ~21 degrees: the motion is rotation-dominant and
 * the scene is distant, so a single homography aligns the views well.
 *
 * Try the forward-motion pair to see when this breaks down:
 *   ./image_stitching data/kitti00_fwd_000024.png data/kitti00_fwd_000025.png
 * Forward translation through a 3D scene violates the homography model:
 * the second view is just a "zoom" of the first (no new field of view)
 * and nearby structure ghosts due to parallax.
 *
 * Pipeline:
 * 1. Detect ORB features in both images
 * 2. Match features using BFMatcher + ratio test
 * 3. Compute homography with RANSAC
 * 4. Calculate output canvas size
 * 5. Warp second image and blend
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

/**
 * @brief Detect and match features between two images
 */
void detectAndMatch(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2,
    std::vector<cv::DMatch>& good_matches) {

    // Create ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(3000);

    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    std::cout << "  Keypoints: " << kp1.size() << " in img1, " << kp2.size() << " in img2" << std::endl;

    if (desc1.empty() || desc2.empty()) {
        std::cerr << "  Warning: No descriptors found!" << std::endl;
        return;
    }

    // Match using BFMatcher with kNN
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    // Apply Lowe's ratio test
    const float ratio_thresh = 0.75f;
    for (const auto& m : knn_matches) {
        if (m.size() >= 2 && m[0].distance < ratio_thresh * m[1].distance) {
            good_matches.push_back(m[0]);
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }

    std::cout << "  Good matches: " << good_matches.size() << std::endl;
}

/**
 * @brief Compute canvas size for panorama
 */
cv::Size computeCanvasSize(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const cv::Mat& H,
    cv::Point2f& offset) {

    // Corners of img2
    std::vector<cv::Point2f> corners2 = {
        {0, 0},
        {static_cast<float>(img2.cols), 0},
        {static_cast<float>(img2.cols), static_cast<float>(img2.rows)},
        {0, static_cast<float>(img2.rows)}
    };

    // Transform corners
    std::vector<cv::Point2f> corners2_transformed;
    cv::perspectiveTransform(corners2, corners2_transformed, H);

    // Find bounding box including img1
    float min_x = 0, min_y = 0;
    float max_x = static_cast<float>(img1.cols);
    float max_y = static_cast<float>(img1.rows);

    for (const auto& pt : corners2_transformed) {
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }

    // Offset to handle negative coordinates
    offset = cv::Point2f(-min_x, -min_y);

    return cv::Size(
        static_cast<int>(std::ceil(max_x - min_x)),
        static_cast<int>(std::ceil(max_y - min_y))
    );
}

/**
 * @brief Warp and blend images into panorama
 */
cv::Mat warpAndBlend(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const cv::Mat& H,
    const cv::Size& canvas_size,
    const cv::Point2f& offset) {

    // Create translation matrix for offset
    cv::Mat T = (cv::Mat_<double>(3, 3) <<
        1, 0, offset.x,
        0, 1, offset.y,
        0, 0, 1);

    // Combined transformation: translate + homography
    cv::Mat H_combined = T * H;

    // Create canvas
    cv::Mat canvas = cv::Mat::zeros(canvas_size, img1.type());

    // Warp img2 to canvas
    cv::Mat img2_warped;
    cv::warpPerspective(img2, img2_warped, H_combined, canvas_size);

    // Copy img1 to canvas (with offset)
    cv::Mat img1_canvas = cv::Mat::zeros(canvas_size, img1.type());
    img1.copyTo(img1_canvas(cv::Rect(
        static_cast<int>(offset.x),
        static_cast<int>(offset.y),
        img1.cols, img1.rows)));

    // Simple compositing: img1 on top in the overlap. Averaging instead
    // would reveal residual parallax as ghosting (try it: any structure
    // off the dominant plane cannot be aligned by a single homography).
    for (int y = 0; y < canvas_size.height; ++y) {
        for (int x = 0; x < canvas_size.width; ++x) {
            cv::Vec3b p1 = img1_canvas.at<cv::Vec3b>(y, x);
            cv::Vec3b p2 = img2_warped.at<cv::Vec3b>(y, x);

            bool has_p1 = (p1[0] != 0 || p1[1] != 0 || p1[2] != 0);

            if (has_p1) {
                canvas.at<cv::Vec3b>(y, x) = p1;
            } else {
                canvas.at<cv::Vec3b>(y, x) = p2;
            }
        }
    }

    return canvas;
}

/**
 * @brief Draw matches visualization
 */
cv::Mat drawMatchesVisualization(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& inlier_mask) {

    // Create side-by-side image
    cv::Mat vis;
    cv::hconcat(img1, img2, vis);

    // Draw matches
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Point2f p1 = pts1[i];
        cv::Point2f p2 = pts2[i];
        p2.x += img1.cols;  // Offset for second image

        cv::Scalar color;
        if (!inlier_mask.empty() && inlier_mask.at<uchar>(i)) {
            color = cv::Scalar(0, 255, 0);  // Green for inliers
        } else {
            color = cv::Scalar(0, 0, 255);  // Red for outliers
        }

        cv::circle(vis, p1, 4, color, -1);
        cv::circle(vis, p2, 4, color, -1);
        cv::line(vis, p1, p2, color, 1);
    }

    return vis;
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================" << std::endl;
    std::cout << "Image Stitching / Panorama Demo" << std::endl;
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

    std::cout << "\nImage sizes: " << img1.cols << "x" << img1.rows
              << ", " << img2.cols << "x" << img2.rows << std::endl;

    // Step 1: Detect and match features
    std::cout << "\n--- Step 1: Feature Detection and Matching ---" << std::endl;
    std::vector<cv::Point2f> pts1, pts2;
    std::vector<cv::DMatch> good_matches;
    detectAndMatch(img1, img2, pts1, pts2, good_matches);

    if (pts1.size() < 4) {
        std::cerr << "Error: Not enough matches found!" << std::endl;
        return -1;
    }

    // Step 2: Compute homography
    std::cout << "\n--- Step 2: Homography Estimation ---" << std::endl;
    cv::Mat inlier_mask;
    cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC, 3.0, inlier_mask, 2000, 0.995);

    if (H.empty()) {
        std::cerr << "Error: Could not compute homography!" << std::endl;
        return -1;
    }

    int inliers = cv::countNonZero(inlier_mask);
    std::cout << "  Inliers: " << inliers << "/" << pts1.size() << std::endl;
    std::cout << "  Homography:\n" << H << std::endl;

    // Step 3: Compute canvas size
    std::cout << "\n--- Step 3: Canvas Size Calculation ---" << std::endl;
    cv::Point2f offset;
    cv::Size canvas_size = computeCanvasSize(img1, img2, H, offset);
    std::cout << "  Canvas size: " << canvas_size.width << "x" << canvas_size.height
              << " (input " << img1.cols << "x" << img1.rows << ")" << std::endl;
    std::cout << "  Offset: (" << offset.x << ", " << offset.y << ")" << std::endl;

    // Step 4: Warp and blend
    std::cout << "\n--- Step 4: Warping and Blending ---" << std::endl;
    cv::Mat panorama = warpAndBlend(img1, img2, H, canvas_size, offset);
    std::cout << "  Panorama created!" << std::endl;

    // Visualizations
    cv::Mat matches_vis = drawMatchesVisualization(img1, img2, pts1, pts2, inlier_mask);

    // Save results
    cv::imwrite("panorama_result.jpg", panorama);
    cv::imwrite("matches_visualization.jpg", matches_vis);
    std::cout << "\nSaved: panorama_result.jpg, matches_visualization.jpg" << std::endl;

    // Display results (only when a display is available)
    if (std::getenv("DISPLAY") != nullptr) {
        cv::imshow("Image 1", img1);
        cv::imshow("Image 2", img2);
        cv::imshow("Matches (Green=Inliers, Red=Outliers)", matches_vis);
        cv::imshow("Panorama", panorama);
        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0);
    }

    return 0;
}
