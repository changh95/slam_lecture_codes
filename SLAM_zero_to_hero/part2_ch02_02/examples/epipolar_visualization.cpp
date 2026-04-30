/**
 * Epipolar Line Visualization
 *
 * This example demonstrates:
 * 1. Computing epipolar lines from the Fundamental matrix
 * 2. Visualizing epipolar lines in both images
 * 3. Verifying that corresponding points lie on epipolar lines
 *
 * For a point x in image 1, the corresponding point x' in image 2
 * lies on the epipolar line l' = F * x.
 * Similarly, l = F^T * x' is the epipolar line in image 1.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>
#include <random>

/**
 * Draw an epipolar line on an image
 */
void drawEpipolarLine(cv::Mat& img, const cv::Mat& line, cv::Scalar color) {
    // Line equation: ax + by + c = 0
    double a = line.at<double>(0);
    double b = line.at<double>(1);
    double c = line.at<double>(2);

    // Find two points on the line within image bounds
    cv::Point pt1, pt2;

    if (std::abs(b) > 1e-6) {
        // y = -(ax + c) / b
        pt1 = cv::Point(0, -c / b);
        pt2 = cv::Point(img.cols - 1, -(a * (img.cols - 1) + c) / b);
    } else {
        // x = -c / a (vertical line)
        pt1 = cv::Point(-c / a, 0);
        pt2 = cv::Point(-c / a, img.rows - 1);
    }

    cv::line(img, pt1, pt2, color, 1, cv::LINE_AA);
}

/**
 * Compute distance from a point to a line
 */
double pointLineDistance(const cv::Point2f& pt, const cv::Mat& line) {
    double a = line.at<double>(0);
    double b = line.at<double>(1);
    double c = line.at<double>(2);

    return std::abs(a * pt.x + b * pt.y + c) / std::sqrt(a * a + b * b);
}

/**
 * Generate random colors for visualization
 */
cv::Scalar randomColor(cv::RNG& rng) {
    return cv::Scalar(rng.uniform(50, 255),
                      rng.uniform(50, 255),
                      rng.uniform(50, 255));
}

int main(int argc, char* argv[]) {
    std::cout << "=== Epipolar Line Visualization ===\n" << std::endl;

    cv::Mat img1, img2;

    if (argc >= 3) {
        img1 = cv::imread(argv[1]);
        img2 = cv::imread(argv[2]);

        if (img1.empty() || img2.empty()) {
            std::cerr << "Failed to load images!" << std::endl;
            return 1;
        }
    } else {
        std::cout << "No images provided, creating synthetic test images." << std::endl;
        std::cout << "Usage: " << argv[0] << " <image1> <image2>" << std::endl;
        std::cout << std::endl;

        // Create synthetic images
        img1 = cv::Mat(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));
        img2 = cv::Mat(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));

        cv::RNG rng(12345);
        for (int i = 0; i < 80; ++i) {
            int x = rng.uniform(50, 590);
            int y = rng.uniform(50, 430);
            int r = rng.uniform(8, 25);
            cv::Scalar color(rng.uniform(100, 255),
                           rng.uniform(100, 255),
                           rng.uniform(100, 255));
            cv::circle(img1, cv::Point(x, y), r, color, -1);
            // Shifted in img2
            cv::circle(img2, cv::Point(x + 25, y + 3), r, color, -1);
        }
    }

    // Convert to grayscale for feature detection
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    // Detect and match features
    auto orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(gray1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(gray2, cv::noArray(), kp2, desc2);

    std::cout << "Keypoints: " << kp1.size() << " / " << kp2.size() << std::endl;

    // Match features
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    // Ratio test
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> pts1, pts2;

    for (const auto& m : knn_matches) {
        if (m[0].distance < 0.75 * m[1].distance) {
            good_matches.push_back(m[0]);
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }

    std::cout << "Good matches: " << good_matches.size() << std::endl;

    if (pts1.size() < 8) {
        std::cerr << "Not enough matches!" << std::endl;
        return 1;
    }

    // Estimate Fundamental matrix
    cv::Mat inlier_mask;
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                                        3.0, 0.99, inlier_mask);

    int num_inliers = cv::countNonZero(inlier_mask);
    std::cout << "RANSAC inliers: " << num_inliers << std::endl;
    std::cout << "Fundamental Matrix:\n" << F << std::endl;
    std::cout << std::endl;

    // Filter to inliers
    std::vector<cv::Point2f> pts1_inliers, pts2_inliers;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (inlier_mask.at<uchar>(i)) {
            pts1_inliers.push_back(pts1[i]);
            pts2_inliers.push_back(pts2[i]);
        }
    }

    // Create visualization images
    cv::Mat vis1 = img1.clone();
    cv::Mat vis2 = img2.clone();

    // Draw epipolar lines
    std::cout << "=== Epipolar Line Errors ===" << std::endl;
    std::cout << "Point distances to corresponding epipolar lines:" << std::endl;

    cv::RNG rng(42);
    int num_to_draw = std::min(15, (int)pts1_inliers.size());

    double total_error1 = 0, total_error2 = 0;

    for (int i = 0; i < num_to_draw; ++i) {
        cv::Scalar color = randomColor(rng);

        // Point in image 1 -> epipolar line in image 2
        cv::Mat pt1_h = (cv::Mat_<double>(3, 1) <<
            pts1_inliers[i].x, pts1_inliers[i].y, 1.0);
        cv::Mat line2 = F * pt1_h;

        // Point in image 2 -> epipolar line in image 1
        cv::Mat pt2_h = (cv::Mat_<double>(3, 1) <<
            pts2_inliers[i].x, pts2_inliers[i].y, 1.0);
        cv::Mat line1 = F.t() * pt2_h;

        // Draw points
        cv::circle(vis1, pts1_inliers[i], 5, color, -1);
        cv::circle(vis2, pts2_inliers[i], 5, color, -1);

        // Draw epipolar lines
        drawEpipolarLine(vis1, line1, color);
        drawEpipolarLine(vis2, line2, color);

        // Compute errors
        double err1 = pointLineDistance(pts1_inliers[i], line1);
        double err2 = pointLineDistance(pts2_inliers[i], line2);
        total_error1 += err1;
        total_error2 += err2;

        std::cout << "  Point " << i << ": err1=" << err1
                  << " px, err2=" << err2 << " px" << std::endl;
    }

    std::cout << "\nMean errors:" << std::endl;
    std::cout << "  Image 1: " << total_error1 / num_to_draw << " px" << std::endl;
    std::cout << "  Image 2: " << total_error2 / num_to_draw << " px" << std::endl;

    // Find epipoles
    std::cout << "\n=== Epipoles ===" << std::endl;

    // Epipole in image 1: e1 = null(F^T)
    cv::Mat U, S, Vt;
    cv::SVD::compute(F.t(), S, U, Vt);
    cv::Mat e1 = Vt.row(2).t();
    e1 = e1 / e1.at<double>(2);  // Normalize

    // Epipole in image 2: e2 = null(F)
    cv::SVD::compute(F, S, U, Vt);
    cv::Mat e2 = Vt.row(2).t();
    e2 = e2 / e2.at<double>(2);  // Normalize

    std::cout << "Epipole in image 1: (" << e1.at<double>(0) << ", "
              << e1.at<double>(1) << ")" << std::endl;
    std::cout << "Epipole in image 2: (" << e2.at<double>(0) << ", "
              << e2.at<double>(1) << ")" << std::endl;

    // Draw epipoles if within image bounds
    cv::Point2f ep1(e1.at<double>(0), e1.at<double>(1));
    cv::Point2f ep2(e2.at<double>(0), e2.at<double>(1));

    if (ep1.x > 0 && ep1.x < vis1.cols && ep1.y > 0 && ep1.y < vis1.rows) {
        cv::circle(vis1, ep1, 10, cv::Scalar(0, 0, 255), 3);
        cv::putText(vis1, "e1", ep1 + cv::Point2f(10, 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    if (ep2.x > 0 && ep2.x < vis2.cols && ep2.y > 0 && ep2.y < vis2.rows) {
        cv::circle(vis2, ep2, 10, cv::Scalar(0, 0, 255), 3);
        cv::putText(vis2, "e2", ep2 + cv::Point2f(10, 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    // Add labels
    cv::putText(vis1, "Image 1 with epipolar lines from Image 2",
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 2);
    cv::putText(vis2, "Image 2 with epipolar lines from Image 1",
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 2);

    // Combine images side by side
    cv::Mat combined;
    cv::hconcat(vis1, vis2, combined);

    // Save result
    cv::imwrite("epipolar_visualization.png", combined);
    std::cout << "\nVisualization saved to: epipolar_visualization.png" << std::endl;

    // Display if possible
    try {
        cv::imshow("Epipolar Geometry", combined);
        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0);
    } catch (...) {
        std::cout << "(Display not available)" << std::endl;
    }

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}
