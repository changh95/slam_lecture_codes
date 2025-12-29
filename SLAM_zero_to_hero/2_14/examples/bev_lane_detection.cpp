/**
 * Lane Detection from Bird's Eye View
 *
 * This example demonstrates:
 * 1. Converting camera image to BEV
 * 2. Detecting lane lines using sliding window
 * 3. Polynomial fitting for lane curves
 * 4. Projecting detected lanes back to camera view
 *
 * Lane detection in BEV is easier because:
 * - Parallel lines remain parallel (no perspective)
 * - Lane width is constant
 * - Polynomial fitting works better
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

/**
 * Apply color and gradient thresholds to find lane pixels
 */
cv::Mat applyLaneThresholds(const cv::Mat& img) {
    cv::Mat gray, hls;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img, hls, cv::COLOR_BGR2HLS);

    // Extract S channel (saturation)
    std::vector<cv::Mat> hls_channels;
    cv::split(hls, hls_channels);
    cv::Mat s_channel = hls_channels[2];

    // Gradient threshold (Sobel X)
    cv::Mat sobelx;
    cv::Sobel(gray, sobelx, CV_64F, 1, 0, 3);
    cv::Mat abs_sobelx;
    cv::convertScaleAbs(sobelx, abs_sobelx);

    // Normalize to 0-255
    cv::normalize(abs_sobelx, abs_sobelx, 0, 255, cv::NORM_MINMAX);

    // Binary thresholds
    cv::Mat grad_binary, color_binary;
    cv::threshold(abs_sobelx, grad_binary, 30, 255, cv::THRESH_BINARY);
    cv::threshold(s_channel, color_binary, 100, 255, cv::THRESH_BINARY);

    // Combine
    cv::Mat combined;
    cv::bitwise_or(grad_binary, color_binary, combined);

    return combined;
}

/**
 * Find lane pixels using sliding window
 */
void slidingWindowLaneDetection(
    const cv::Mat& binary_bev,
    std::vector<cv::Point2f>& left_lane_pts,
    std::vector<cv::Point2f>& right_lane_pts,
    cv::Mat& visualization) {

    left_lane_pts.clear();
    right_lane_pts.clear();

    int height = binary_bev.rows;
    int width = binary_bev.cols;

    // Create visualization
    cv::cvtColor(binary_bev, visualization, cv::COLOR_GRAY2BGR);

    // Histogram of bottom half
    cv::Mat bottom_half = binary_bev(cv::Rect(0, height / 2, width, height / 2));
    cv::Mat histogram;
    cv::reduce(bottom_half, histogram, 0, cv::REDUCE_SUM, CV_32F);

    // Find peaks for left and right lanes
    int midpoint = width / 2;
    float max_left = 0, max_right = 0;
    int left_base = 0, right_base = width - 1;

    for (int x = 0; x < midpoint; ++x) {
        if (histogram.at<float>(x) > max_left) {
            max_left = histogram.at<float>(x);
            left_base = x;
        }
    }
    for (int x = midpoint; x < width; ++x) {
        if (histogram.at<float>(x) > max_right) {
            max_right = histogram.at<float>(x);
            right_base = x;
        }
    }

    // Sliding window parameters
    int num_windows = 9;
    int window_height = height / num_windows;
    int window_width = 80;
    int min_pixels = 50;  // Minimum pixels to recenter window

    int left_x = left_base;
    int right_x = right_base;

    // Slide windows
    for (int win = 0; win < num_windows; ++win) {
        int y_low = height - (win + 1) * window_height;
        int y_high = height - win * window_height;

        // Left window
        int left_x_low = std::max(0, left_x - window_width / 2);
        int left_x_high = std::min(width, left_x + window_width / 2);

        // Right window
        int right_x_low = std::max(0, right_x - window_width / 2);
        int right_x_high = std::min(width, right_x + window_width / 2);

        // Draw windows
        cv::rectangle(visualization,
            cv::Point(left_x_low, y_low),
            cv::Point(left_x_high, y_high),
            cv::Scalar(0, 255, 0), 2);

        cv::rectangle(visualization,
            cv::Point(right_x_low, y_low),
            cv::Point(right_x_high, y_high),
            cv::Scalar(0, 255, 0), 2);

        // Find pixels in windows
        std::vector<cv::Point> left_pixels, right_pixels;

        for (int y = y_low; y < y_high; ++y) {
            for (int x = left_x_low; x < left_x_high; ++x) {
                if (binary_bev.at<uchar>(y, x) > 0) {
                    left_pixels.push_back(cv::Point(x, y));
                    left_lane_pts.push_back(cv::Point2f(x, y));
                }
            }
            for (int x = right_x_low; x < right_x_high; ++x) {
                if (binary_bev.at<uchar>(y, x) > 0) {
                    right_pixels.push_back(cv::Point(x, y));
                    right_lane_pts.push_back(cv::Point2f(x, y));
                }
            }
        }

        // Recenter windows
        if (left_pixels.size() > min_pixels) {
            double sum_x = 0;
            for (const auto& p : left_pixels) sum_x += p.x;
            left_x = sum_x / left_pixels.size();
        }
        if (right_pixels.size() > min_pixels) {
            double sum_x = 0;
            for (const auto& p : right_pixels) sum_x += p.x;
            right_x = sum_x / right_pixels.size();
        }
    }
}

/**
 * Fit polynomial to lane points
 * Returns coefficients [a, b, c] for y = a*x^2 + b*x + c
 */
cv::Vec3d fitPolynomial(const std::vector<cv::Point2f>& pts) {
    if (pts.size() < 3) {
        return cv::Vec3d(0, 0, 0);
    }

    // Build matrices for least squares: A * coeffs = b
    cv::Mat A(pts.size(), 3, CV_64F);
    cv::Mat b(pts.size(), 1, CV_64F);

    for (size_t i = 0; i < pts.size(); ++i) {
        double y = pts[i].y;
        A.at<double>(i, 0) = y * y;
        A.at<double>(i, 1) = y;
        A.at<double>(i, 2) = 1;
        b.at<double>(i, 0) = pts[i].x;
    }

    // Solve using SVD
    cv::Mat coeffs;
    cv::solve(A, b, coeffs, cv::DECOMP_SVD);

    return cv::Vec3d(coeffs.at<double>(0), coeffs.at<double>(1), coeffs.at<double>(2));
}

/**
 * Draw lane on image
 */
void drawLane(cv::Mat& img, const cv::Vec3d& left_fit, const cv::Vec3d& right_fit) {
    std::vector<cv::Point> left_pts, right_pts;

    for (int y = 0; y < img.rows; y += 5) {
        double left_x = left_fit[0] * y * y + left_fit[1] * y + left_fit[2];
        double right_x = right_fit[0] * y * y + right_fit[1] * y + right_fit[2];

        if (left_x >= 0 && left_x < img.cols) {
            left_pts.push_back(cv::Point(left_x, y));
        }
        if (right_x >= 0 && right_x < img.cols) {
            right_pts.push_back(cv::Point(right_x, y));
        }
    }

    // Draw lane area
    if (!left_pts.empty() && !right_pts.empty()) {
        std::vector<cv::Point> lane_pts;
        lane_pts.insert(lane_pts.end(), left_pts.begin(), left_pts.end());
        std::reverse(right_pts.begin(), right_pts.end());
        lane_pts.insert(lane_pts.end(), right_pts.begin(), right_pts.end());

        cv::Mat overlay = img.clone();
        cv::fillPoly(overlay, lane_pts, cv::Scalar(0, 255, 0));
        cv::addWeighted(overlay, 0.3, img, 0.7, 0, img);
    }

    // Draw lane lines
    for (size_t i = 1; i < left_pts.size(); ++i) {
        cv::line(img, left_pts[i-1], left_pts[i], cv::Scalar(255, 0, 0), 3);
    }
    for (size_t i = 1; i < right_pts.size(); ++i) {
        cv::line(img, right_pts[i-1], right_pts[i], cv::Scalar(0, 0, 255), 3);
    }
}

/**
 * Create synthetic road image
 */
cv::Mat createSyntheticRoad(int width, int height) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(80, 80, 80));

    int cx = width / 2;
    int horizon = height * 0.4;

    // Lane lines with some curve
    for (int y = horizon; y < height; ++y) {
        double t = (double)(y - horizon) / (height - horizon);
        double curve = 20 * std::sin(t * 2);

        int left_x = cx - 20 - (int)(200 * t + curve);
        int right_x = cx + 20 + (int)(200 * t + curve);

        cv::circle(img, cv::Point(left_x, y), 2, cv::Scalar(255, 255, 255), -1);
        cv::circle(img, cv::Point(right_x, y), 2, cv::Scalar(255, 255, 255), -1);
    }

    // Sky
    cv::rectangle(img, cv::Point(0, 0), cv::Point(width, horizon),
                  cv::Scalar(200, 150, 100), -1);

    return img;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Lane Detection from BEV ===\n" << std::endl;

    cv::Mat input_img;

    if (argc >= 2) {
        input_img = cv::imread(argv[1]);
        if (input_img.empty()) {
            std::cerr << "Failed to load: " << argv[1] << std::endl;
            return 1;
        }
    } else {
        std::cout << "No image provided, using synthetic road." << std::endl;
        input_img = createSyntheticRoad(640, 480);
    }

    // BEV transformation
    std::vector<cv::Point2f> src_pts = {
        cv::Point2f(200, 480),
        cv::Point2f(440, 480),
        cv::Point2f(550, 300),
        cv::Point2f(90, 300)
    };

    cv::Size bev_size(400, 600);
    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(100, 600),
        cv::Point2f(300, 600),
        cv::Point2f(300, 0),
        cv::Point2f(100, 0)
    };

    cv::Mat H_bev = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat H_inv = cv::getPerspectiveTransform(dst_pts, src_pts);

    // Transform to BEV
    cv::Mat bev_img;
    cv::warpPerspective(input_img, bev_img, H_bev, bev_size);

    // Apply thresholds
    cv::Mat binary = applyLaneThresholds(bev_img);

    // Lane detection
    std::vector<cv::Point2f> left_pts, right_pts;
    cv::Mat window_vis;
    slidingWindowLaneDetection(binary, left_pts, right_pts, window_vis);

    std::cout << "Left lane points: " << left_pts.size() << std::endl;
    std::cout << "Right lane points: " << right_pts.size() << std::endl;

    // Fit polynomials
    cv::Vec3d left_fit = fitPolynomial(left_pts);
    cv::Vec3d right_fit = fitPolynomial(right_pts);

    std::cout << "Left fit: " << left_fit << std::endl;
    std::cout << "Right fit: " << right_fit << std::endl;

    // Draw detected lane on BEV
    cv::Mat bev_with_lane = bev_img.clone();
    drawLane(bev_with_lane, left_fit, right_fit);

    // Project back to original view
    cv::Mat lane_overlay = cv::Mat::zeros(bev_size, CV_8UC3);
    drawLane(lane_overlay, left_fit, right_fit);

    cv::Mat lane_camera;
    cv::warpPerspective(lane_overlay, lane_camera, H_inv, input_img.size());

    cv::Mat result = input_img.clone();
    cv::addWeighted(result, 1.0, lane_camera, 0.5, 0, result);

    // Save results
    cv::imwrite("lane_input.png", input_img);
    cv::imwrite("lane_bev.png", bev_img);
    cv::imwrite("lane_binary.png", binary);
    cv::imwrite("lane_sliding_window.png", window_vis);
    cv::imwrite("lane_bev_detected.png", bev_with_lane);
    cv::imwrite("lane_result.png", result);

    std::cout << "\nSaved images:" << std::endl;
    std::cout << "  lane_input.png" << std::endl;
    std::cout << "  lane_bev.png" << std::endl;
    std::cout << "  lane_binary.png" << std::endl;
    std::cout << "  lane_sliding_window.png" << std::endl;
    std::cout << "  lane_bev_detected.png" << std::endl;
    std::cout << "  lane_result.png" << std::endl;

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}
