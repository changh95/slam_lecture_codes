/**
 * @file image_stitching.cpp
 * @brief Image Stitching / Panorama Demo using OpenCV
 *
 * This example demonstrates how to stitch two overlapping images
 * into a panorama using homography estimation.
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

#include <iostream>
#include <vector>
#include <cmath>

/**
 * @brief Create synthetic overlapping images for testing
 */
void createSyntheticImages(cv::Mat& img1, cv::Mat& img2) {
    // Create a large "scene" image
    cv::Mat scene(400, 800, CV_8UC3, cv::Scalar(100, 100, 100));

    // Draw some features on the scene
    std::vector<cv::Point> polygons = {
        {100, 100}, {200, 80}, {250, 150}, {180, 200}, {100, 180}
    };
    cv::fillPoly(scene, std::vector<std::vector<cv::Point>>{polygons}, cv::Scalar(0, 0, 255));

    cv::circle(scene, cv::Point(400, 150), 50, cv::Scalar(0, 255, 0), -1);
    cv::circle(scene, cv::Point(450, 200), 30, cv::Scalar(255, 255, 0), -1);

    cv::rectangle(scene, cv::Point(550, 100), cv::Point(700, 250), cv::Scalar(255, 0, 0), -1);

    // Add some texture (grid pattern)
    for (int y = 0; y < scene.rows; y += 20) {
        cv::line(scene, cv::Point(0, y), cv::Point(scene.cols, y), cv::Scalar(80, 80, 80), 1);
    }
    for (int x = 0; x < scene.cols; x += 20) {
        cv::line(scene, cv::Point(x, 0), cv::Point(x, scene.rows), cv::Scalar(80, 80, 80), 1);
    }

    // Add random dots for more features
    cv::RNG rng(42);
    for (int i = 0; i < 200; ++i) {
        int x = rng.uniform(0, scene.cols);
        int y = rng.uniform(0, scene.rows);
        cv::circle(scene, cv::Point(x, y), 3, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1);
    }

    // Extract two overlapping regions
    img1 = scene(cv::Rect(0, 0, 500, 400)).clone();
    img2 = scene(cv::Rect(300, 0, 500, 400)).clone();

    // Add labels
    cv::putText(img1, "Image 1", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::putText(img2, "Image 2", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
}

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
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);

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

    // Simple blending: average in overlap, otherwise copy
    for (int y = 0; y < canvas_size.height; ++y) {
        for (int x = 0; x < canvas_size.width; ++x) {
            cv::Vec3b p1 = img1_canvas.at<cv::Vec3b>(y, x);
            cv::Vec3b p2 = img2_warped.at<cv::Vec3b>(y, x);

            bool has_p1 = (p1[0] != 0 || p1[1] != 0 || p1[2] != 0);
            bool has_p2 = (p2[0] != 0 || p2[1] != 0 || p2[2] != 0);

            if (has_p1 && has_p2) {
                // Average blend
                canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    (p1[0] + p2[0]) / 2,
                    (p1[1] + p2[1]) / 2,
                    (p1[2] + p2[2]) / 2
                );
            } else if (has_p1) {
                canvas.at<cv::Vec3b>(y, x) = p1;
            } else if (has_p2) {
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

    cv::Mat img1, img2;

    // Load images or create synthetic ones
    if (argc >= 3) {
        img1 = cv::imread(argv[1]);
        img2 = cv::imread(argv[2]);

        if (img1.empty() || img2.empty()) {
            std::cerr << "Error: Could not load images!" << std::endl;
            std::cout << "Usage: " << argv[0] << " [image1.jpg image2.jpg]" << std::endl;
            std::cout << "Running with synthetic images..." << std::endl;
            createSyntheticImages(img1, img2);
        } else {
            std::cout << "Loaded images: " << argv[1] << ", " << argv[2] << std::endl;
        }
    } else {
        std::cout << "No images provided. Using synthetic images." << std::endl;
        createSyntheticImages(img1, img2);
    }

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

    int inliers = cv::countNonZero(inlier_mask);
    std::cout << "  Inliers: " << inliers << "/" << pts1.size() << std::endl;
    std::cout << "  Homography:\n" << H << std::endl;

    if (H.empty()) {
        std::cerr << "Error: Could not compute homography!" << std::endl;
        return -1;
    }

    // Step 3: Compute canvas size
    std::cout << "\n--- Step 3: Canvas Size Calculation ---" << std::endl;
    cv::Point2f offset;
    cv::Size canvas_size = computeCanvasSize(img1, img2, H, offset);
    std::cout << "  Canvas size: " << canvas_size.width << "x" << canvas_size.height << std::endl;
    std::cout << "  Offset: (" << offset.x << ", " << offset.y << ")" << std::endl;

    // Step 4: Warp and blend
    std::cout << "\n--- Step 4: Warping and Blending ---" << std::endl;
    cv::Mat panorama = warpAndBlend(img1, img2, H, canvas_size, offset);
    std::cout << "  Panorama created!" << std::endl;

    // Visualizations
    cv::Mat matches_vis = drawMatchesVisualization(img1, img2, pts1, pts2, inlier_mask);

    // Display results
    cv::imshow("Image 1", img1);
    cv::imshow("Image 2", img2);
    cv::imshow("Matches (Green=Inliers, Red=Outliers)", matches_vis);
    cv::imshow("Panorama", panorama);

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);

    // Save results
    cv::imwrite("panorama_result.jpg", panorama);
    cv::imwrite("matches_visualization.jpg", matches_vis);
    std::cout << "Saved: panorama_result.jpg, matches_visualization.jpg" << std::endl;

    return 0;
}
