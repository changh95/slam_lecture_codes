/**
 * RANSAC Homography Estimation
 *
 * This example demonstrates homography estimation using different
 * RANSAC variants in OpenCV for visual SLAM applications.
 *
 * Homography relates corresponding points between two images when:
 * - The scene is planar
 * - The camera undergoes pure rotation
 * - The scene is far away (depth variation << distance)
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

// Timing helper
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsedMs() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Generate synthetic point correspondences with outliers
void generateSyntheticData(std::vector<cv::Point2f>& pts1,
                            std::vector<cv::Point2f>& pts2,
                            cv::Mat& gtHomography,
                            int numPoints = 100,
                            double outlierRatio = 0.3,
                            double noise = 1.0) {
    // Ground truth homography (rotation + translation + perspective)
    gtHomography = (cv::Mat_<double>(3, 3) << 0.9, -0.1, 50,
                                               0.1, 0.95, 30,
                                               0.0001, 0.0002, 1.0);

    pts1.clear();
    pts2.clear();

    std::srand(42);

    int numInliers = static_cast<int>(numPoints * (1.0 - outlierRatio));
    int numOutliers = numPoints - numInliers;

    // Generate inliers
    for (int i = 0; i < numInliers; ++i) {
        float x = static_cast<float>(rand() % 600 + 50);
        float y = static_cast<float>(rand() % 400 + 50);
        pts1.emplace_back(x, y);

        // Transform point using homography
        cv::Mat pt1 = (cv::Mat_<double>(3, 1) << x, y, 1.0);
        cv::Mat pt2 = gtHomography * pt1;
        float x2 = static_cast<float>(pt2.at<double>(0) / pt2.at<double>(2));
        float y2 = static_cast<float>(pt2.at<double>(1) / pt2.at<double>(2));

        // Add Gaussian noise
        x2 += static_cast<float>((rand() % 100 - 50) / 50.0 * noise);
        y2 += static_cast<float>((rand() % 100 - 50) / 50.0 * noise);
        pts2.emplace_back(x2, y2);
    }

    // Generate outliers (random correspondences)
    for (int i = 0; i < numOutliers; ++i) {
        pts1.emplace_back(rand() % 600 + 50, rand() % 400 + 50);
        pts2.emplace_back(rand() % 600 + 50, rand() % 400 + 50);
    }

    // Shuffle points
    std::vector<size_t> indices(pts1.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    std::vector<cv::Point2f> shuffled1, shuffled2;
    for (size_t idx : indices) {
        shuffled1.push_back(pts1[idx]);
        shuffled2.push_back(pts2[idx]);
    }
    pts1 = shuffled1;
    pts2 = shuffled2;
}

// Compute homography error
double computeHomographyError(const cv::Mat& H1, const cv::Mat& H2) {
    // Normalize both matrices
    cv::Mat H1_norm = H1 / H1.at<double>(2, 2);
    cv::Mat H2_norm = H2 / H2.at<double>(2, 2);
    return cv::norm(H1_norm - H2_norm, cv::NORM_L2);
}

// Count inliers based on reprojection error
int countInliers(const std::vector<cv::Point2f>& pts1,
                 const std::vector<cv::Point2f>& pts2,
                 const cv::Mat& H,
                 double threshold) {
    int count = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat pt1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat pt2_proj = H * pt1;
        float x2 = static_cast<float>(pt2_proj.at<double>(0) / pt2_proj.at<double>(2));
        float y2 = static_cast<float>(pt2_proj.at<double>(1) / pt2_proj.at<double>(2));

        double error = std::sqrt(std::pow(x2 - pts2[i].x, 2) + std::pow(y2 - pts2[i].y, 2));
        if (error < threshold) {
            ++count;
        }
    }
    return count;
}

int main() {
    std::cout << "=== RANSAC Homography Estimation ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // Generate synthetic data
    std::vector<cv::Point2f> pts1, pts2;
    cv::Mat gtH;
    const int numPoints = 200;
    const double outlierRatio = 0.4;  // 40% outliers
    const double threshold = 3.0;

    generateSyntheticData(pts1, pts2, gtH, numPoints, outlierRatio);

    std::cout << "\nData: " << numPoints << " points, "
              << static_cast<int>(outlierRatio * 100) << "% outliers\n";
    std::cout << "Ground truth homography:\n" << gtH << "\n\n";

    Timer timer;

    // ========== Method 1: Standard RANSAC ==========
    std::cout << "--- Method 1: Standard RANSAC ---" << std::endl;
    timer.start();
    cv::Mat mask1;
    cv::Mat H1 = cv::findHomography(pts1, pts2, cv::RANSAC, threshold, mask1, 2000, 0.99);
    double time1 = timer.elapsedMs();

    if (!H1.empty()) {
        double error1 = computeHomographyError(gtH, H1);
        int inliers1 = cv::countNonZero(mask1);
        std::cout << "  Error: " << error1 << ", Inliers: " << inliers1
                  << ", Time: " << time1 << " ms" << std::endl;
    }

    // ========== Method 2: LMEDS (Least Median of Squares) ==========
    std::cout << "\n--- Method 2: LMEDS ---" << std::endl;
    timer.start();
    cv::Mat mask2;
    cv::Mat H2 = cv::findHomography(pts1, pts2, cv::LMEDS, threshold, mask2);
    double time2 = timer.elapsedMs();

    if (!H2.empty()) {
        double error2 = computeHomographyError(gtH, H2);
        int inliers2 = cv::countNonZero(mask2);
        std::cout << "  Error: " << error2 << ", Inliers: " << inliers2
                  << ", Time: " << time2 << " ms" << std::endl;
    }

    // ========== Method 3: USAC_DEFAULT ==========
    std::cout << "\n--- Method 3: USAC_DEFAULT ---" << std::endl;
    timer.start();
    cv::Mat mask3;
    cv::Mat H3 = cv::findHomography(pts1, pts2, cv::USAC_DEFAULT, threshold, mask3);
    double time3 = timer.elapsedMs();

    if (!H3.empty()) {
        double error3 = computeHomographyError(gtH, H3);
        int inliers3 = cv::countNonZero(mask3);
        std::cout << "  Error: " << error3 << ", Inliers: " << inliers3
                  << ", Time: " << time3 << " ms" << std::endl;
    }

    // ========== Method 4: USAC_MAGSAC ==========
    std::cout << "\n--- Method 4: USAC_MAGSAC ---" << std::endl;
    timer.start();
    cv::Mat mask4;
    cv::Mat H4 = cv::findHomography(pts1, pts2, cv::USAC_MAGSAC, threshold, mask4);
    double time4 = timer.elapsedMs();

    if (!H4.empty()) {
        double error4 = computeHomographyError(gtH, H4);
        int inliers4 = cv::countNonZero(mask4);
        std::cout << "  Error: " << error4 << ", Inliers: " << inliers4
                  << ", Time: " << time4 << " ms" << std::endl;
    }

    // ========== Method 5: USAC_PROSAC ==========
    std::cout << "\n--- Method 5: USAC_PROSAC ---" << std::endl;
    timer.start();
    cv::Mat mask5;
    cv::Mat H5 = cv::findHomography(pts1, pts2, cv::USAC_PROSAC, threshold, mask5);
    double time5 = timer.elapsedMs();

    if (!H5.empty()) {
        double error5 = computeHomographyError(gtH, H5);
        int inliers5 = cv::countNonZero(mask5);
        std::cout << "  Error: " << error5 << ", Inliers: " << inliers5
                  << ", Time: " << time5 << " ms" << std::endl;
    }

    // ========== Method 6: USAC_ACCURATE ==========
    std::cout << "\n--- Method 6: USAC_ACCURATE ---" << std::endl;
    timer.start();
    cv::Mat mask6;
    cv::Mat H6 = cv::findHomography(pts1, pts2, cv::USAC_ACCURATE, threshold, mask6);
    double time6 = timer.elapsedMs();

    if (!H6.empty()) {
        double error6 = computeHomographyError(gtH, H6);
        int inliers6 = cv::countNonZero(mask6);
        std::cout << "  Error: " << error6 << ", Inliers: " << inliers6
                  << ", Time: " << time6 << " ms" << std::endl;
    }

    // ========== Method 7: Custom UsacParams ==========
    std::cout << "\n--- Method 7: Custom UsacParams ---" << std::endl;
    cv::UsacParams usacParams;
    usacParams.sampler = cv::SAMPLING_PROSAC;
    usacParams.score = cv::SCORE_METHOD_MAGSAC;
    usacParams.loMethod = cv::LOCAL_OPTIM_INNER_AND_ITER_LO;
    usacParams.loIterations = 15;
    usacParams.final_polisher = cv::MAGSAC;
    usacParams.final_polisher_iterations = 10;
    usacParams.threshold = threshold;
    usacParams.confidence = 0.999;
    usacParams.maxIterations = 5000;
    usacParams.isParallel = true;

    timer.start();
    cv::Mat mask7;
    cv::Mat H7 = cv::findHomography(pts1, pts2, mask7, usacParams);
    double time7 = timer.elapsedMs();

    if (!H7.empty()) {
        double error7 = computeHomographyError(gtH, H7);
        int inliers7 = cv::countNonZero(mask7);
        std::cout << "  Error: " << error7 << ", Inliers: " << inliers7
                  << ", Time: " << time7 << " ms" << std::endl;
    }

    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Method            | Error   | Inliers | Time (ms)" << std::endl;
    std::cout << "------------------|---------|---------|----------" << std::endl;
    if (!H1.empty()) std::cout << "RANSAC            | " << computeHomographyError(gtH, H1)
                               << " | " << cv::countNonZero(mask1) << "     | " << time1 << std::endl;
    if (!H2.empty()) std::cout << "LMEDS             | " << computeHomographyError(gtH, H2)
                               << " | " << cv::countNonZero(mask2) << "     | " << time2 << std::endl;
    if (!H3.empty()) std::cout << "USAC_DEFAULT      | " << computeHomographyError(gtH, H3)
                               << " | " << cv::countNonZero(mask3) << "     | " << time3 << std::endl;
    if (!H4.empty()) std::cout << "USAC_MAGSAC       | " << computeHomographyError(gtH, H4)
                               << " | " << cv::countNonZero(mask4) << "     | " << time4 << std::endl;
    if (!H5.empty()) std::cout << "USAC_PROSAC       | " << computeHomographyError(gtH, H5)
                               << " | " << cv::countNonZero(mask5) << "     | " << time5 << std::endl;
    if (!H6.empty()) std::cout << "USAC_ACCURATE     | " << computeHomographyError(gtH, H6)
                               << " | " << cv::countNonZero(mask6) << "     | " << time6 << std::endl;
    if (!H7.empty()) std::cout << "Custom UsacParams | " << computeHomographyError(gtH, H7)
                               << " | " << cv::countNonZero(mask7) << "     | " << time7 << std::endl;

    return 0;
}
