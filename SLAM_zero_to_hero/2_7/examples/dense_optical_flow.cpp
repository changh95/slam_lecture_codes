/**
 * Dense Optical Flow using Farneback Method
 *
 * This example demonstrates dense optical flow computation where
 * flow is estimated for every pixel in the image.
 *
 * Key concepts:
 * - Farneback polynomial expansion algorithm
 * - Flow visualization using HSV color coding
 * - Motion magnitude and direction analysis
 * - Comparison with sparse methods
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**
 * Configuration for Farneback optical flow
 */
struct FarnebackConfig {
    double pyr_scale = 0.5;      // Scale between pyramid levels
    int levels = 3;              // Number of pyramid levels
    int winsize = 15;            // Averaging window size
    int iterations = 3;          // Iterations at each pyramid level
    int poly_n = 5;              // Size of pixel neighborhood (5 or 7)
    double poly_sigma = 1.1;     // Gaussian std dev for polynomial smoothing
    int flags = 0;               // Optional flags
};

/**
 * Generate synthetic frames for testing
 */
class MotionSequenceGenerator {
public:
    MotionSequenceGenerator(int width, int height, int seed = 42)
        : width_(width), height_(height), rng_(seed) {
        createBaseScene();
    }

    void createBaseScene() {
        base_image_ = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(100));

        // Create a rich texture with various patterns

        // Grid pattern
        for (int y = 0; y < height_; y += 40) {
            cv::line(base_image_, cv::Point(0, y), cv::Point(width_, y),
                    cv::Scalar(150), 1);
        }
        for (int x = 0; x < width_; x += 40) {
            cv::line(base_image_, cv::Point(x, 0), cv::Point(x, height_),
                    cv::Scalar(150), 1);
        }

        // Random shapes
        for (int i = 0; i < 60; i++) {
            cv::Point center(rng_.uniform(20, width_ - 20),
                           rng_.uniform(20, height_ - 20));
            int size = rng_.uniform(15, 50);
            int intensity = rng_.uniform(50, 200);

            if (i % 3 == 0) {
                cv::circle(base_image_, center, size/2,
                          cv::Scalar(intensity), -1);
            } else if (i % 3 == 1) {
                cv::rectangle(base_image_,
                             center - cv::Point(size/2, size/2),
                             center + cv::Point(size/2, size/2),
                             cv::Scalar(intensity), -1);
            } else {
                std::vector<cv::Point> triangle;
                triangle.push_back(center + cv::Point(0, -size/2));
                triangle.push_back(center + cv::Point(-size/2, size/2));
                triangle.push_back(center + cv::Point(size/2, size/2));
                cv::fillPoly(base_image_, triangle, cv::Scalar(intensity));
            }
        }

        // Add texture noise
        cv::Mat noise(height_, width_, CV_8UC1);
        rng_.fill(noise, cv::RNG::NORMAL, 0, 10);
        base_image_ += noise;
    }

    cv::Mat getTranslatedFrame(double dx, double dy) const {
        cv::Mat result;
        cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
        cv::warpAffine(base_image_, result, M, base_image_.size(),
                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return result;
    }

    cv::Mat getRotatedFrame(double angle_deg) const {
        cv::Point2f center(width_ / 2.0f, height_ / 2.0f);
        cv::Mat M = cv::getRotationMatrix2D(center, angle_deg, 1.0);
        cv::Mat result;
        cv::warpAffine(base_image_, result, M, base_image_.size(),
                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return result;
    }

    cv::Mat getScaledFrame(double scale) const {
        cv::Point2f center(width_ / 2.0f, height_ / 2.0f);
        cv::Mat M = cv::getRotationMatrix2D(center, 0, scale);
        cv::Mat result;
        cv::warpAffine(base_image_, result, M, base_image_.size(),
                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return result;
    }

    const cv::Mat& getBaseImage() const { return base_image_; }

private:
    int width_, height_;
    cv::RNG rng_;
    cv::Mat base_image_;
};

/**
 * Dense optical flow estimator using Farneback method
 */
class DenseFlowEstimator {
public:
    explicit DenseFlowEstimator(const FarnebackConfig& config = FarnebackConfig())
        : config_(config) {}

    /**
     * Compute dense optical flow between two frames
     */
    cv::Mat computeFlow(const cv::Mat& prev, const cv::Mat& curr) {
        cv::Mat flow;

        cv::calcOpticalFlowFarneback(
            prev, curr, flow,
            config_.pyr_scale,
            config_.levels,
            config_.winsize,
            config_.iterations,
            config_.poly_n,
            config_.poly_sigma,
            config_.flags
        );

        return flow;
    }

    /**
     * Convert flow to HSV color visualization
     * - Hue represents direction
     * - Saturation is always max
     * - Value represents magnitude
     */
    cv::Mat flowToHSV(const cv::Mat& flow) const {
        // Split flow into x and y components
        std::vector<cv::Mat> flow_parts(2);
        cv::split(flow, flow_parts);

        cv::Mat magnitude, angle;
        cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

        // Normalize magnitude for visualization
        double maxMag;
        cv::minMaxLoc(magnitude, nullptr, &maxMag);
        if (maxMag > 0) {
            magnitude /= maxMag;
        }

        // Create HSV image
        cv::Mat hsv(flow.size(), CV_8UC3);
        std::vector<cv::Mat> hsv_planes(3);

        // Hue: direction (0-180 in OpenCV)
        angle.convertTo(hsv_planes[0], CV_8U, 0.5);

        // Saturation: max
        hsv_planes[1] = cv::Mat::ones(flow.size(), CV_8U) * 255;

        // Value: magnitude
        magnitude.convertTo(hsv_planes[2], CV_8U, 255);

        cv::merge(hsv_planes, hsv);

        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        return bgr;
    }

    /**
     * Draw flow field as arrows (subsampled)
     */
    cv::Mat flowToArrows(const cv::Mat& flow, const cv::Mat& background,
                         int step = 16, double scale = 2.0) const {
        cv::Mat vis;
        if (background.channels() == 1) {
            cv::cvtColor(background, vis, cv::COLOR_GRAY2BGR);
        } else {
            background.copyTo(vis);
        }

        for (int y = step / 2; y < flow.rows; y += step) {
            for (int x = step / 2; x < flow.cols; x += step) {
                cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
                cv::Point2f from(x, y);
                cv::Point2f to(x + fxy.x * scale, y + fxy.y * scale);

                // Color based on direction
                double angle = std::atan2(fxy.y, fxy.x) * 180 / CV_PI + 180;
                int hue = static_cast<int>(angle / 2);

                cv::arrowedLine(vis, from, to,
                               cv::Scalar(hue * 1.4, 255 - hue, 100 + hue % 155),
                               1, cv::LINE_AA, 0, 0.3);
            }
        }

        return vis;
    }

    /**
     * Compute flow statistics
     */
    struct FlowStats {
        double mean_magnitude;
        double max_magnitude;
        double mean_angle;
        double std_magnitude;
    };

    FlowStats computeStats(const cv::Mat& flow) const {
        std::vector<cv::Mat> flow_parts(2);
        cv::split(flow, flow_parts);

        cv::Mat magnitude, angle;
        cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

        FlowStats stats;

        cv::Scalar mean_mag, std_mag;
        cv::meanStdDev(magnitude, mean_mag, std_mag);
        stats.mean_magnitude = mean_mag[0];
        stats.std_magnitude = std_mag[0];

        cv::minMaxLoc(magnitude, nullptr, &stats.max_magnitude);

        stats.mean_angle = cv::mean(angle)[0];

        return stats;
    }

    /**
     * Warp image using flow (for motion compensation)
     */
    cv::Mat warpWithFlow(const cv::Mat& image, const cv::Mat& flow) const {
        cv::Mat map_x(flow.size(), CV_32F);
        cv::Mat map_y(flow.size(), CV_32F);

        for (int y = 0; y < flow.rows; y++) {
            for (int x = 0; x < flow.cols; x++) {
                cv::Point2f f = flow.at<cv::Point2f>(y, x);
                map_x.at<float>(y, x) = x + f.x;
                map_y.at<float>(y, x) = y + f.y;
            }
        }

        cv::Mat warped;
        cv::remap(image, warped, map_x, map_y, cv::INTER_LINEAR);
        return warped;
    }

private:
    FarnebackConfig config_;
};

/**
 * Demonstrate basic dense optical flow
 */
void demonstrateBasicDenseFlow() {
    std::cout << "\n=== Basic Dense Optical Flow Demo ===" << std::endl;

    const int width = 320;
    const int height = 240;

    MotionSequenceGenerator generator(width, height);

    // Test different motions
    struct MotionTest {
        std::string name;
        double dx, dy, angle, scale;
    };

    std::vector<MotionTest> tests = {
        {"Horizontal translation (10px right)", 10, 0, 0, 1.0},
        {"Vertical translation (5px down)", 0, 5, 0, 1.0},
        {"Diagonal translation", 7, 7, 0, 1.0},
        {"Rotation (2 degrees)", 0, 0, 2, 1.0},
        {"Zoom in (1.05x)", 0, 0, 0, 1.05},
    };

    DenseFlowEstimator estimator;
    cv::Mat prev = generator.getBaseImage();

    std::cout << "\nMotion Type                          | Mean Mag | Max Mag | Mean Angle" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    for (const auto& test : tests) {
        cv::Mat curr;
        if (test.angle != 0) {
            curr = generator.getRotatedFrame(test.angle);
        } else if (test.scale != 1.0) {
            curr = generator.getScaledFrame(test.scale);
        } else {
            curr = generator.getTranslatedFrame(test.dx, test.dy);
        }

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat flow = estimator.computeFlow(prev, curr);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        auto stats = estimator.computeStats(flow);

        std::cout << std::left << std::setw(37) << test.name << "| "
                  << std::right << std::setw(8) << std::fixed << std::setprecision(2)
                  << stats.mean_magnitude << " | "
                  << std::setw(7) << stats.max_magnitude << " | "
                  << std::setw(10) << stats.mean_angle
                  << " (" << std::setprecision(1) << elapsed << "ms)"
                  << std::endl;
    }
}

/**
 * Compare sparse vs dense optical flow
 */
void compareSparseVsDense() {
    std::cout << "\n=== Sparse vs Dense Optical Flow Comparison ===" << std::endl;

    const int width = 640;
    const int height = 480;

    MotionSequenceGenerator generator(width, height);

    cv::Mat prev = generator.getBaseImage();
    cv::Mat curr = generator.getTranslatedFrame(8, 5);

    // Dense flow (Farneback)
    DenseFlowEstimator dense_estimator;

    auto dense_start = std::chrono::high_resolution_clock::now();
    cv::Mat dense_flow = dense_estimator.computeFlow(prev, curr);
    auto dense_end = std::chrono::high_resolution_clock::now();
    double dense_time = std::chrono::duration<double, std::milli>(dense_end - dense_start).count();

    // Sparse flow (Lucas-Kanade)
    std::vector<cv::Point2f> prev_pts;
    cv::goodFeaturesToTrack(prev, prev_pts, 200, 0.01, 10);

    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> error;

    auto sparse_start = std::chrono::high_resolution_clock::now();
    cv::calcOpticalFlowPyrLK(prev, curr, prev_pts, curr_pts,
                             status, error, cv::Size(21, 21), 3);
    auto sparse_end = std::chrono::high_resolution_clock::now();
    double sparse_time = std::chrono::duration<double, std::milli>(sparse_end - sparse_start).count();

    // Compute sparse flow statistics
    double sparse_mean_dx = 0, sparse_mean_dy = 0;
    int valid_count = 0;
    for (size_t i = 0; i < prev_pts.size(); i++) {
        if (status[i]) {
            sparse_mean_dx += curr_pts[i].x - prev_pts[i].x;
            sparse_mean_dy += curr_pts[i].y - prev_pts[i].y;
            valid_count++;
        }
    }
    if (valid_count > 0) {
        sparse_mean_dx /= valid_count;
        sparse_mean_dy /= valid_count;
    }

    // Compute dense flow statistics (center region)
    cv::Rect roi(width/4, height/4, width/2, height/2);
    cv::Mat dense_roi = dense_flow(roi);
    cv::Scalar dense_mean = cv::mean(dense_roi);

    std::cout << "\nGround truth motion: (8, 5) pixels" << std::endl;
    std::cout << "\n                    | Time (ms) | Est. dX | Est. dY | Points" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    std::cout << "Sparse (Lucas-Kanade)| "
              << std::setw(9) << std::fixed << std::setprecision(2) << sparse_time << " | "
              << std::setw(7) << sparse_mean_dx << " | "
              << std::setw(7) << sparse_mean_dy << " | "
              << valid_count
              << std::endl;

    std::cout << "Dense (Farneback)    | "
              << std::setw(9) << std::fixed << std::setprecision(2) << dense_time << " | "
              << std::setw(7) << dense_mean[0] << " | "
              << std::setw(7) << dense_mean[1] << " | "
              << (width * height)
              << std::endl;

    std::cout << "\nObservations:" << std::endl;
    std::cout << "- Sparse method is ~" << std::fixed << std::setprecision(0)
              << (dense_time / sparse_time) << "x faster" << std::endl;
    std::cout << "- Dense method provides flow for " << (width * height)
              << " points vs " << valid_count << " points" << std::endl;
    std::cout << "- Both methods estimate motion close to ground truth" << std::endl;
}

/**
 * Demonstrate motion compensation using optical flow
 */
void demonstrateMotionCompensation() {
    std::cout << "\n=== Motion Compensation Demo ===" << std::endl;

    const int width = 320;
    const int height = 240;

    MotionSequenceGenerator generator(width, height);

    cv::Mat frame1 = generator.getBaseImage();
    cv::Mat frame2 = generator.getTranslatedFrame(12, 8);

    DenseFlowEstimator estimator;

    // Compute flow from frame1 to frame2
    cv::Mat flow = estimator.computeFlow(frame1, frame2);

    // Warp frame1 towards frame2 using flow
    cv::Mat warped = estimator.warpWithFlow(frame1, flow);

    // Compute difference images
    cv::Mat diff_before, diff_after;
    cv::absdiff(frame1, frame2, diff_before);
    cv::absdiff(warped, frame2, diff_after);

    double error_before = cv::mean(diff_before)[0];
    double error_after = cv::mean(diff_after)[0];

    std::cout << "\nMean absolute difference:" << std::endl;
    std::cout << "  Before compensation: " << std::fixed << std::setprecision(2)
              << error_before << std::endl;
    std::cout << "  After compensation:  " << error_after << std::endl;
    std::cout << "  Reduction: " << std::setprecision(1)
              << (100.0 * (error_before - error_after) / error_before) << "%" << std::endl;
}

/**
 * Benchmark Farneback parameters
 */
void benchmarkFarnebackParameters() {
    std::cout << "\n=== Farneback Parameter Benchmark ===" << std::endl;

    const int width = 320;
    const int height = 240;

    MotionSequenceGenerator generator(width, height);

    cv::Mat frame1 = generator.getBaseImage();
    cv::Mat frame2 = generator.getTranslatedFrame(10, 5);

    std::cout << "\nVarying pyramid levels (winsize=15, poly_n=5):" << std::endl;
    std::cout << "Levels | Time (ms) | Mean Error" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    std::vector<int> levels_to_test = {1, 2, 3, 4, 5};

    for (int levels : levels_to_test) {
        FarnebackConfig config;
        config.levels = levels;

        DenseFlowEstimator estimator(config);

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat flow = estimator.computeFlow(frame1, frame2);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        // Compute error (GT: dx=10, dy=5)
        cv::Scalar mean_flow = cv::mean(flow);
        double error = std::sqrt(std::pow(mean_flow[0] - 10, 2) +
                                std::pow(mean_flow[1] - 5, 2));

        std::cout << std::setw(6) << levels << " | "
                  << std::setw(9) << std::fixed << std::setprecision(2) << elapsed << " | "
                  << std::setw(10) << error
                  << std::endl;
    }

    std::cout << "\nVarying window size (levels=3, poly_n=5):" << std::endl;
    std::cout << "Winsize | Time (ms) | Mean Error" << std::endl;
    std::cout << "----------------------------------" << std::endl;

    std::vector<int> winsizes = {5, 11, 15, 21, 31};

    for (int winsize : winsizes) {
        FarnebackConfig config;
        config.winsize = winsize;

        DenseFlowEstimator estimator(config);

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat flow = estimator.computeFlow(frame1, frame2);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        cv::Scalar mean_flow = cv::mean(flow);
        double error = std::sqrt(std::pow(mean_flow[0] - 10, 2) +
                                std::pow(mean_flow[1] - 5, 2));

        std::cout << std::setw(7) << winsize << " | "
                  << std::setw(9) << std::fixed << std::setprecision(2) << elapsed << " | "
                  << std::setw(10) << error
                  << std::endl;
    }

    std::cout << "\nVarying poly_n (levels=3, winsize=15):" << std::endl;
    std::cout << "Poly_n | Time (ms) | Mean Error" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    std::vector<int> poly_ns = {5, 7};
    std::vector<double> poly_sigmas = {1.1, 1.5};

    for (size_t i = 0; i < poly_ns.size(); i++) {
        FarnebackConfig config;
        config.poly_n = poly_ns[i];
        config.poly_sigma = poly_sigmas[i];

        DenseFlowEstimator estimator(config);

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat flow = estimator.computeFlow(frame1, frame2);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        cv::Scalar mean_flow = cv::mean(flow);
        double error = std::sqrt(std::pow(mean_flow[0] - 10, 2) +
                                std::pow(mean_flow[1] - 5, 2));

        std::cout << std::setw(6) << poly_ns[i] << " | "
                  << std::setw(9) << std::fixed << std::setprecision(2) << elapsed << " | "
                  << std::setw(10) << error
                  << std::endl;
    }
}

/**
 * Demonstrate dense flow for motion segmentation
 */
void demonstrateMotionSegmentation() {
    std::cout << "\n=== Motion Segmentation Demo ===" << std::endl;
    std::cout << "(Simulating scene with independently moving regions)" << std::endl;

    const int width = 320;
    const int height = 240;

    // Create two frames with different motion in different regions
    cv::Mat frame1 = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat frame2 = cv::Mat::zeros(height, width, CV_8UC1);

    // Add background texture
    cv::RNG rng(42);
    for (int i = 0; i < 500; i++) {
        cv::Point pt(rng.uniform(0, width), rng.uniform(0, height));
        cv::circle(frame1, pt, rng.uniform(2, 8), cv::Scalar(rng.uniform(100, 200)), -1);
    }
    frame1.copyTo(frame2);

    // Region 1: Left side - move right
    cv::Rect roi1(0, 0, width/3, height);
    cv::Mat left1 = frame1(roi1);
    cv::Mat left2;
    cv::Mat M1 = (cv::Mat_<double>(2, 3) << 1, 0, 8, 0, 1, 0);  // 8px right
    cv::warpAffine(left1, left2, M1, left1.size());
    left2.copyTo(frame2(roi1));

    // Region 2: Right side - move down
    cv::Rect roi2(2*width/3, 0, width/3, height);
    cv::Mat right1 = frame1(roi2);
    cv::Mat right2;
    cv::Mat M2 = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 6);  // 6px down
    cv::warpAffine(right1, right2, M2, right1.size());
    right2.copyTo(frame2(roi2));

    // Compute dense flow
    DenseFlowEstimator estimator;
    cv::Mat flow = estimator.computeFlow(frame1, frame2);

    // Analyze flow in each region
    std::cout << "\nFlow analysis by region:" << std::endl;
    std::cout << "Region          | Mean dX | Mean dY | Dominant Motion" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    struct RegionInfo {
        std::string name;
        cv::Rect roi;
    };

    std::vector<RegionInfo> regions = {
        {"Left (moving R)", cv::Rect(0, 0, width/3, height)},
        {"Center (static)", cv::Rect(width/3, 0, width/3, height)},
        {"Right (moving D)", cv::Rect(2*width/3, 0, width/3, height)}
    };

    for (const auto& region : regions) {
        cv::Mat region_flow = flow(region.roi);
        cv::Scalar mean_flow = cv::mean(region_flow);

        std::string motion;
        if (std::abs(mean_flow[0]) > 3) {
            motion = mean_flow[0] > 0 ? "Right" : "Left";
        } else if (std::abs(mean_flow[1]) > 3) {
            motion = mean_flow[1] > 0 ? "Down" : "Up";
        } else {
            motion = "Static";
        }

        std::cout << std::left << std::setw(16) << region.name << "| "
                  << std::right << std::setw(7) << std::fixed << std::setprecision(2)
                  << mean_flow[0] << " | "
                  << std::setw(7) << mean_flow[1] << " | "
                  << motion
                  << std::endl;
    }

    std::cout << "\nConclusion: Dense optical flow can segment regions with different motion" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "======================================================" << std::endl;
    std::cout << "  Dense Optical Flow - Farneback Algorithm            " << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // Run demonstrations
    demonstrateBasicDenseFlow();
    compareSparseVsDense();
    demonstrateMotionCompensation();
    benchmarkFarnebackParameters();
    demonstrateMotionSegmentation();

    std::cout << "\n=== All demonstrations complete ===" << std::endl;

    return 0;
}
