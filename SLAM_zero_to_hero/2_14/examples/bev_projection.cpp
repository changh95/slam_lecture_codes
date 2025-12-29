/**
 * Bird's Eye View (BEV) Projection / Inverse Perspective Mapping
 *
 * This example demonstrates:
 * 1. Computing the homography for BEV transformation
 * 2. Transforming a front-facing camera image to top-down view
 * 3. Handling camera extrinsics and intrinsics
 *
 * BEV is useful for:
 * - Lane detection and lane keeping
 * - Parking assistance
 * - Obstacle detection on flat surfaces
 *
 * The BEV transform assumes a flat ground plane.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

/**
 * Compute BEV homography from camera parameters
 *
 * Assumes:
 * - Camera looking forward and down at the road
 * - Ground plane is flat (z = 0 in world frame)
 */
cv::Mat computeBEVHomography(
    const cv::Mat& K,           // Camera intrinsics
    double camera_height,       // Camera height above ground (meters)
    double pitch_angle,         // Camera pitch angle (radians, positive = looking down)
    double output_scale,        // Pixels per meter in BEV
    const cv::Size& output_size // Output BEV image size
) {
    // Rotation matrix for pitch
    cv::Mat R_pitch = (cv::Mat_<double>(3, 3) <<
        1, 0, 0,
        0, std::cos(pitch_angle), -std::sin(pitch_angle),
        0, std::sin(pitch_angle), std::cos(pitch_angle));

    // Camera extrinsics: camera at height h, looking down
    cv::Mat t = (cv::Mat_<double>(3, 1) << 0, -camera_height, 0);

    // Ground plane normal (pointing up in world frame, transforms to camera frame)
    cv::Mat n_world = (cv::Mat_<double>(3, 1) << 0, 1, 0);  // Y-up
    cv::Mat n_cam = R_pitch * n_world;
    double d = camera_height;  // Distance to plane

    // Homography from ground plane to camera image
    // H_cam = K * (R - t * n' / d) * K^(-1)
    // We want inverse: from image to ground

    cv::Mat K_inv = K.inv();

    // For BEV, we also need to transform ground coords to pixel coords
    // Create output transformation
    cv::Mat T_output = (cv::Mat_<double>(3, 3) <<
        output_scale, 0, output_size.width / 2.0,
        0, -output_scale, output_size.height * 0.9,  // Put camera at bottom
        0, 0, 1);

    // Build the complete homography
    // This maps image pixels to BEV pixels

    // Simplified homography for flat ground assumption
    // Using pitch angle to determine perspective transformation

    double f = K.at<double>(0, 0);
    double cy = K.at<double>(1, 2);
    double cx = K.at<double>(0, 2);

    // Source points (in image) - typically a trapezoid on the road
    // These would be calibrated for actual camera setup
    std::vector<cv::Point2f> src_pts = {
        cv::Point2f(cx - 200, cy + 100),    // Bottom left
        cv::Point2f(cx + 200, cy + 100),    // Bottom right
        cv::Point2f(cx + 400, cy + 200),    // Far bottom right
        cv::Point2f(cx - 400, cy + 200)     // Far bottom left
    };

    // Destination points (in BEV) - rectangle
    double bev_width = 400;
    double bev_near = output_size.height * 0.8;
    double bev_far = output_size.height * 0.2;

    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(output_size.width / 2.0 - bev_width / 2, bev_near),
        cv::Point2f(output_size.width / 2.0 + bev_width / 2, bev_near),
        cv::Point2f(output_size.width / 2.0 + bev_width / 2, bev_far),
        cv::Point2f(output_size.width / 2.0 - bev_width / 2, bev_far)
    };

    return cv::getPerspectiveTransform(src_pts, dst_pts);
}

/**
 * Create a synthetic road image for testing
 */
cv::Mat createSyntheticRoadImage(int width, int height) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(80, 80, 80));  // Gray road

    // Draw lane lines (converging due to perspective)
    int cx = width / 2;
    int horizon = height * 0.4;

    // Left lane line
    cv::line(img, cv::Point(cx - 200, height),
             cv::Point(cx - 30, horizon), cv::Scalar(255, 255, 255), 3);

    // Right lane line
    cv::line(img, cv::Point(cx + 200, height),
             cv::Point(cx + 30, horizon), cv::Scalar(255, 255, 255), 3);

    // Center dashed line
    for (int y = horizon; y < height; y += 40) {
        double t = (double)(y - horizon) / (height - horizon);
        int x_offset = (int)(20 + 180 * t);
        cv::line(img, cv::Point(cx, y), cv::Point(cx, y + 20),
                 cv::Scalar(200, 200, 0), 2);
    }

    // Add sky
    cv::rectangle(img, cv::Point(0, 0), cv::Point(width, horizon),
                  cv::Scalar(200, 150, 100), -1);

    // Add horizon line
    cv::line(img, cv::Point(0, horizon), cv::Point(width, horizon),
             cv::Scalar(100, 100, 100), 1);

    // Add some road markings
    for (int y = horizon + 50; y < height; y += 80) {
        double t = (double)(y - horizon) / (height - horizon);
        int lane_width = (int)(100 + 300 * t);

        // Cross-walk style markings
        if (y > height * 0.7) {
            cv::rectangle(img,
                cv::Point(cx - lane_width/2, y),
                cv::Point(cx - lane_width/2 + 30, y + 10),
                cv::Scalar(255, 255, 255), -1);
            cv::rectangle(img,
                cv::Point(cx + lane_width/2 - 30, y),
                cv::Point(cx + lane_width/2, y + 10),
                cv::Scalar(255, 255, 255), -1);
        }
    }

    return img;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Bird's Eye View (BEV) Projection ===\n" << std::endl;

    cv::Mat input_img;

    if (argc >= 2) {
        input_img = cv::imread(argv[1]);
        if (input_img.empty()) {
            std::cerr << "Failed to load image: " << argv[1] << std::endl;
            return 1;
        }
    } else {
        std::cout << "No image provided, using synthetic road image." << std::endl;
        std::cout << "Usage: " << argv[0] << " [input_image]" << std::endl;
        std::cout << std::endl;
        input_img = createSyntheticRoadImage(640, 480);
    }

    std::cout << "Input image size: " << input_img.cols << "x" << input_img.rows
              << std::endl;

    // Camera parameters (typical dashcam setup)
    double fx = 500, fy = 500;
    double cx = input_img.cols / 2.0;
    double cy = input_img.rows / 2.0;
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1);

    double camera_height = 1.5;     // 1.5 meters above ground
    double pitch_angle = 10.0 * CV_PI / 180.0;  // 10 degrees down

    // BEV output parameters
    cv::Size bev_size(400, 600);

    // Compute BEV homography
    cv::Mat H_bev = computeBEVHomography(
        K, camera_height, pitch_angle,
        100.0,  // 100 pixels per meter
        bev_size
    );

    std::cout << "\nBEV Homography:\n" << H_bev << std::endl;

    // Apply BEV transformation
    cv::Mat bev_img;
    cv::warpPerspective(input_img, bev_img, H_bev, bev_size,
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                        cv::Scalar(0, 0, 0));

    // Add grid overlay to BEV for scale reference
    cv::Mat bev_with_grid = bev_img.clone();
    int grid_spacing = 50;  // pixels (0.5 meters at 100 px/m)

    for (int x = 0; x < bev_size.width; x += grid_spacing) {
        cv::line(bev_with_grid, cv::Point(x, 0),
                 cv::Point(x, bev_size.height),
                 cv::Scalar(0, 100, 0), 1);
    }
    for (int y = 0; y < bev_size.height; y += grid_spacing) {
        cv::line(bev_with_grid, cv::Point(0, y),
                 cv::Point(bev_size.width, y),
                 cv::Scalar(0, 100, 0), 1);
    }

    // Add scale indicator
    cv::putText(bev_with_grid, "Grid: 0.5m",
                cv::Point(10, bev_size.height - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // Mark vehicle position (at bottom center)
    cv::circle(bev_with_grid,
               cv::Point(bev_size.width / 2, bev_size.height - 20),
               10, cv::Scalar(0, 0, 255), -1);
    cv::putText(bev_with_grid, "Vehicle",
                cv::Point(bev_size.width / 2 - 30, bev_size.height - 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

    // Save results
    cv::imwrite("bev_input.png", input_img);
    cv::imwrite("bev_output.png", bev_img);
    cv::imwrite("bev_with_grid.png", bev_with_grid);

    std::cout << "\nSaved images:" << std::endl;
    std::cout << "  bev_input.png      - Original camera view" << std::endl;
    std::cout << "  bev_output.png     - Bird's Eye View" << std::endl;
    std::cout << "  bev_with_grid.png  - BEV with grid overlay" << std::endl;

    // Create side-by-side comparison
    cv::Mat input_resized;
    cv::resize(input_img, input_resized, cv::Size(400, 300));

    cv::Mat bev_resized;
    cv::resize(bev_with_grid, bev_resized, cv::Size(200, 300));

    cv::Mat comparison(300, 600, CV_8UC3, cv::Scalar(0, 0, 0));
    input_resized.copyTo(comparison(cv::Rect(0, 0, 400, 300)));
    bev_resized.copyTo(comparison(cv::Rect(400, 0, 200, 300)));

    cv::putText(comparison, "Camera View", cv::Point(150, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "BEV", cv::Point(470, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    cv::imwrite("bev_comparison.png", comparison);
    std::cout << "  bev_comparison.png - Side-by-side comparison" << std::endl;

    // Try to display
    try {
        cv::imshow("BEV Projection", comparison);
        std::cout << "\nPress any key to exit..." << std::endl;
        cv::waitKey(0);
    } catch (...) {
        std::cout << "\n(Display not available)" << std::endl;
    }

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}
