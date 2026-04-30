/**
 * RANSAC Fundamental Matrix Estimation with USAC
 *
 * This example demonstrates fundamental matrix estimation using OpenCV's
 * USAC framework. The fundamental matrix encodes the epipolar geometry
 * between two views and is crucial for:
 *
 * - Epipolar constraint verification
 * - Essential matrix computation (E = K'^T * F * K)
 * - Motion estimation in Visual SLAM
 * - Two-view geometry initialization
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Timer utility
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

// Generate synthetic stereo correspondences
void generateStereoData(std::vector<cv::Point2f>& pts1,
                         std::vector<cv::Point2f>& pts2,
                         cv::Mat& gtFundamental,
                         cv::Mat& K,
                         int numPoints = 150,
                         double outlierRatio = 0.35) {
    // Camera intrinsic matrix (typical SLAM camera)
    K = (cv::Mat_<double>(3, 3) << 500.0, 0.0, 320.0,
                                    0.0, 500.0, 240.0,
                                    0.0, 0.0, 1.0);

    // Relative pose: small rotation + translation (typical stereo baseline)
    cv::Mat R = (cv::Mat_<double>(3, 3) << 0.9998, -0.01, 0.015,
                                            0.01, 0.9999, -0.005,
                                            -0.015, 0.0052, 0.9999);

    cv::Mat t = (cv::Mat_<double>(3, 1) << 0.1, 0.01, 0.02);  // Baseline

    // Essential matrix: E = [t]_x * R
    cv::Mat tx = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2), t.at<double>(1),
                                            t.at<double>(2), 0, -t.at<double>(0),
                                            -t.at<double>(1), t.at<double>(0), 0);
    cv::Mat E = tx * R;

    // Fundamental matrix: F = K^(-T) * E * K^(-1)
    gtFundamental = K.inv().t() * E * K.inv();

    // Normalize F so that ||F|| = 1
    gtFundamental = gtFundamental / cv::norm(gtFundamental);

    pts1.clear();
    pts2.clear();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> distX(50.0f, 590.0f);
    std::uniform_real_distribution<float> distY(50.0f, 430.0f);
    std::uniform_real_distribution<float> distDepth(2.0f, 20.0f);
    std::normal_distribution<float> noise(0.0f, 0.5f);

    int numInliers = static_cast<int>(numPoints * (1.0 - outlierRatio));
    int numOutliers = numPoints - numInliers;

    // Generate inliers: project 3D points to both views
    for (int i = 0; i < numInliers; ++i) {
        // Random 3D point
        float u1 = distX(gen);
        float v1 = distY(gen);
        float depth = distDepth(gen);

        // Back-project to 3D (camera 1 frame)
        double x = (u1 - K.at<double>(0, 2)) * depth / K.at<double>(0, 0);
        double y = (v1 - K.at<double>(1, 2)) * depth / K.at<double>(1, 1);
        double z = depth;

        cv::Mat P = (cv::Mat_<double>(3, 1) << x, y, z);

        // Transform to camera 2 frame
        cv::Mat P2 = R * P + t;

        // Project to image 2
        float u2 = static_cast<float>(K.at<double>(0, 0) * P2.at<double>(0) / P2.at<double>(2)
                                      + K.at<double>(0, 2));
        float v2 = static_cast<float>(K.at<double>(1, 1) * P2.at<double>(1) / P2.at<double>(2)
                                      + K.at<double>(1, 2));

        // Add noise
        pts1.emplace_back(u1 + noise(gen), v1 + noise(gen));
        pts2.emplace_back(u2 + noise(gen), v2 + noise(gen));
    }

    // Generate outliers: random correspondences
    for (int i = 0; i < numOutliers; ++i) {
        pts1.emplace_back(distX(gen), distY(gen));
        pts2.emplace_back(distX(gen), distY(gen));
    }

    // Shuffle
    std::vector<size_t> indices(pts1.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    std::vector<cv::Point2f> shuffled1, shuffled2;
    for (size_t idx : indices) {
        shuffled1.push_back(pts1[idx]);
        shuffled2.push_back(pts2[idx]);
    }
    pts1 = shuffled1;
    pts2 = shuffled2;
}

// Compute Sampson distance (geometric error for fundamental matrix)
double computeSampsonError(const cv::Mat& F,
                           const std::vector<cv::Point2f>& pts1,
                           const std::vector<cv::Point2f>& pts2) {
    double totalError = 0.0;
    int count = 0;

    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat p1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);

        // Epipolar constraint: p2^T * F * p1 = 0
        cv::Mat Fp1 = F * p1;
        cv::Mat Ftp2 = F.t() * p2;

        double num = p2.dot(Fp1);
        double denom = Fp1.at<double>(0) * Fp1.at<double>(0)
                     + Fp1.at<double>(1) * Fp1.at<double>(1)
                     + Ftp2.at<double>(0) * Ftp2.at<double>(0)
                     + Ftp2.at<double>(1) * Ftp2.at<double>(1);

        if (denom > 1e-10) {
            double sampsonDist = (num * num) / denom;
            totalError += sampsonDist;
            ++count;
        }
    }

    return count > 0 ? totalError / count : 0.0;
}

// Normalize fundamental matrix for comparison
cv::Mat normalizeFundamental(const cv::Mat& F) {
    cv::Mat Fn;
    F.convertTo(Fn, CV_64F);
    double scale = cv::norm(Fn);
    if (scale > 1e-10) {
        Fn = Fn / scale;
    }
    // Ensure consistent sign (positive F[2,2] if non-zero)
    if (Fn.at<double>(2, 2) < 0) {
        Fn = -Fn;
    }
    return Fn;
}

int main() {
    std::cout << "=== RANSAC Fundamental Matrix Estimation with USAC ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    // Generate synthetic stereo data
    std::vector<cv::Point2f> pts1, pts2;
    cv::Mat gtF, K;
    const int numPoints = 200;
    const double outlierRatio = 0.35;
    const double threshold = 3.0;

    generateStereoData(pts1, pts2, gtF, K, numPoints, outlierRatio);

    std::cout << "\nData: " << numPoints << " correspondences, "
              << static_cast<int>(outlierRatio * 100) << "% outliers\n";
    std::cout << "Camera matrix K:\n" << K << "\n";
    std::cout << "\nGround truth F (normalized):\n" << normalizeFundamental(gtF) << "\n\n";

    Timer timer;

    // ========== Method 1: 7-Point Algorithm ==========
    std::cout << "--- Method 1: FM_7POINT ---" << std::endl;
    timer.start();
    cv::Mat mask1;
    cv::Mat F1 = cv::findFundamentalMat(pts1, pts2, cv::FM_7POINT, threshold, 0.99, mask1);
    double time1 = timer.elapsedMs();

    if (!F1.empty() && F1.rows == 3) {
        double sampson1 = computeSampsonError(F1, pts1, pts2);
        std::cout << "  Sampson error: " << sampson1 << ", Time: " << time1 << " ms\n";
    } else if (!F1.empty()) {
        std::cout << "  Multiple solutions found (" << F1.rows / 3 << "), Time: " << time1 << " ms\n";
    }

    // ========== Method 2: 8-Point Algorithm ==========
    std::cout << "\n--- Method 2: FM_8POINT ---" << std::endl;
    timer.start();
    cv::Mat mask2;
    cv::Mat F2 = cv::findFundamentalMat(pts1, pts2, cv::FM_8POINT, threshold, 0.99, mask2);
    double time2 = timer.elapsedMs();

    if (!F2.empty()) {
        double sampson2 = computeSampsonError(F2, pts1, pts2);
        std::cout << "  Sampson error: " << sampson2 << ", Time: " << time2 << " ms\n";
    }

    // ========== Method 3: RANSAC ==========
    std::cout << "\n--- Method 3: FM_RANSAC ---" << std::endl;
    timer.start();
    cv::Mat mask3;
    cv::Mat F3 = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, threshold, 0.99, mask3);
    double time3 = timer.elapsedMs();

    if (!F3.empty()) {
        double sampson3 = computeSampsonError(F3, pts1, pts2);
        int inliers3 = cv::countNonZero(mask3);
        std::cout << "  Sampson error: " << sampson3 << ", Inliers: " << inliers3
                  << ", Time: " << time3 << " ms\n";
    }

    // ========== Method 4: LMEDS ==========
    std::cout << "\n--- Method 4: FM_LMEDS ---" << std::endl;
    timer.start();
    cv::Mat mask4;
    cv::Mat F4 = cv::findFundamentalMat(pts1, pts2, cv::FM_LMEDS, threshold, 0.99, mask4);
    double time4 = timer.elapsedMs();

    if (!F4.empty()) {
        double sampson4 = computeSampsonError(F4, pts1, pts2);
        int inliers4 = cv::countNonZero(mask4);
        std::cout << "  Sampson error: " << sampson4 << ", Inliers: " << inliers4
                  << ", Time: " << time4 << " ms\n";
    }

    // ========== Method 5: USAC_DEFAULT ==========
    std::cout << "\n--- Method 5: USAC_DEFAULT ---" << std::endl;
    timer.start();
    cv::Mat mask5;
    cv::Mat F5 = cv::findFundamentalMat(pts1, pts2, cv::USAC_DEFAULT, threshold, 0.99, mask5);
    double time5 = timer.elapsedMs();

    if (!F5.empty()) {
        double sampson5 = computeSampsonError(F5, pts1, pts2);
        int inliers5 = cv::countNonZero(mask5);
        std::cout << "  Sampson error: " << sampson5 << ", Inliers: " << inliers5
                  << ", Time: " << time5 << " ms\n";
    }

    // ========== Method 6: USAC_FM_8PTS ==========
    std::cout << "\n--- Method 6: USAC_FM_8PTS ---" << std::endl;
    timer.start();
    cv::Mat mask6;
    cv::Mat F6 = cv::findFundamentalMat(pts1, pts2, cv::USAC_FM_8PTS, threshold, 0.99, mask6);
    double time6 = timer.elapsedMs();

    if (!F6.empty()) {
        double sampson6 = computeSampsonError(F6, pts1, pts2);
        int inliers6 = cv::countNonZero(mask6);
        std::cout << "  Sampson error: " << sampson6 << ", Inliers: " << inliers6
                  << ", Time: " << time6 << " ms\n";
    }

    // ========== Method 7: USAC_MAGSAC ==========
    std::cout << "\n--- Method 7: USAC_MAGSAC ---" << std::endl;
    timer.start();
    cv::Mat mask7;
    cv::Mat F7 = cv::findFundamentalMat(pts1, pts2, cv::USAC_MAGSAC, threshold, 0.99, mask7);
    double time7 = timer.elapsedMs();

    if (!F7.empty()) {
        double sampson7 = computeSampsonError(F7, pts1, pts2);
        int inliers7 = cv::countNonZero(mask7);
        std::cout << "  Sampson error: " << sampson7 << ", Inliers: " << inliers7
                  << ", Time: " << time7 << " ms\n";
    }

    // ========== Method 8: Custom UsacParams with PROSAC + MAGSAC ==========
    std::cout << "\n--- Method 8: Custom UsacParams (PROSAC + MAGSAC) ---" << std::endl;

    cv::UsacParams usacParams;
    usacParams.sampler = cv::SAMPLING_PROSAC;
    usacParams.score = cv::SCORE_METHOD_MAGSAC;
    usacParams.loMethod = cv::LOCAL_OPTIM_INNER_AND_ITER_LO;
    usacParams.loIterations = 15;
    usacParams.neighborsSearch = cv::NEIGH_FLANN_KNN;
    usacParams.final_polisher = cv::MAGSAC;
    usacParams.final_polisher_iterations = 10;
    usacParams.threshold = threshold;
    usacParams.confidence = 0.999;
    usacParams.maxIterations = 10000;
    usacParams.isParallel = true;

    timer.start();
    cv::Mat mask8;
    cv::Mat F8 = cv::findFundamentalMat(pts1, pts2, mask8, usacParams);
    double time8 = timer.elapsedMs();

    if (!F8.empty()) {
        double sampson8 = computeSampsonError(F8, pts1, pts2);
        int inliers8 = cv::countNonZero(mask8);
        std::cout << "  Sampson error: " << sampson8 << ", Inliers: " << inliers8
                  << ", Time: " << time8 << " ms\n";
    }

    // ========== Method 9: USAC_ACCURATE ==========
    std::cout << "\n--- Method 9: USAC_ACCURATE ---" << std::endl;
    timer.start();
    cv::Mat mask9;
    cv::Mat F9 = cv::findFundamentalMat(pts1, pts2, cv::USAC_ACCURATE, threshold, 0.99, mask9);
    double time9 = timer.elapsedMs();

    if (!F9.empty()) {
        double sampson9 = computeSampsonError(F9, pts1, pts2);
        int inliers9 = cv::countNonZero(mask9);
        std::cout << "  Sampson error: " << sampson9 << ", Inliers: " << inliers9
                  << ", Time: " << time9 << " ms\n";
    }

    // ========== Summary ==========
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Method           | Sampson Err | Inliers | Time (ms)" << std::endl;
    std::cout << "-----------------|-------------|---------|----------" << std::endl;

    auto printResult = [&](const std::string& name, const cv::Mat& F, const cv::Mat& mask, double time) {
        if (!F.empty() && F.rows == 3) {
            double sampson = computeSampsonError(F, pts1, pts2);
            int inliers = mask.empty() ? 0 : cv::countNonZero(mask);
            std::cout << std::left << std::setw(17) << name << "| "
                      << std::setw(11) << sampson << " | "
                      << std::setw(7) << inliers << " | "
                      << time << std::endl;
        }
    };

    printResult("FM_RANSAC", F3, mask3, time3);
    printResult("FM_LMEDS", F4, mask4, time4);
    printResult("USAC_DEFAULT", F5, mask5, time5);
    printResult("USAC_FM_8PTS", F6, mask6, time6);
    printResult("USAC_MAGSAC", F7, mask7, time7);
    printResult("Custom USAC", F8, mask8, time8);
    printResult("USAC_ACCURATE", F9, mask9, time9);

    // ========== Visualize Epipolar Lines ==========
    std::cout << "\n=== Epipolar Geometry Verification ===" << std::endl;
    if (!F7.empty()) {
        // Check epipolar constraint: x2^T * F * x1 = 0
        double totalEpipolarError = 0.0;
        int inlierCount = 0;

        for (size_t i = 0; i < pts1.size(); ++i) {
            if (mask7.at<uchar>(static_cast<int>(i))) {
                cv::Mat p1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
                cv::Mat p2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
                double error = std::abs(p2.dot(F7 * p1));
                totalEpipolarError += error;
                ++inlierCount;
            }
        }

        std::cout << "Average epipolar constraint error (inliers only): "
                  << totalEpipolarError / inlierCount << std::endl;
        std::cout << "Expected: ~0 for perfect correspondences" << std::endl;
    }

    return 0;
}
