/**
 * Fundamental Matrix Estimation using MAGSAC++
 *
 * This example demonstrates MAGSAC++ (Marginalizing Sample Consensus),
 * a state-of-the-art robust estimation algorithm that provides:
 *
 * 1. Threshold-free estimation: Marginalizes over noise scale sigma
 * 2. Sigma-scoring: Uses probabilistic scoring instead of binary inlier/outlier
 * 3. Progressive NAPSAC sampling: Spatially-aware sampling for faster convergence
 *
 * MAGSAC++ is the successor to MAGSAC and offers improved accuracy and speed.
 * It has been integrated into OpenCV USAC (USAC_MAGSAC flag).
 *
 * References:
 * - Barath et al., "MAGSAC: marginalizing sample consensus", CVPR 2019
 * - Barath et al., "MAGSAC++, a fast, reliable and accurate robust estimator", CVPR 2020
 *
 * Repository: https://github.com/danini/magsac
 */

#include "magsac.h"
#include "estimators.h"
#include "model.h"
#include "samplers/progressive_napsac_sampler.h"
#include "samplers/uniform_sampler.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// ============================================================================
// Generate Synthetic Stereo Data
// ============================================================================

void generateStereoData(cv::Mat& points,
                        cv::Mat& gtFundamental,
                        int numPoints = 150,
                        double outlierRatio = 0.35) {
    // Camera intrinsics
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        500.0, 0.0, 320.0,
        0.0, 500.0, 240.0,
        0.0, 0.0, 1.0);

    // Relative pose: small rotation + translation
    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        0.9998, -0.01, 0.015,
        0.01, 0.9999, -0.005,
        -0.015, 0.0052, 0.9999);

    cv::Mat t = (cv::Mat_<double>(3, 1) << 0.1, 0.01, 0.02);

    // Essential matrix: E = [t]_x * R
    cv::Mat tx = (cv::Mat_<double>(3, 3) <<
        0, -t.at<double>(2), t.at<double>(1),
        t.at<double>(2), 0, -t.at<double>(0),
        -t.at<double>(1), t.at<double>(0), 0);
    cv::Mat E = tx * R;

    // Fundamental matrix: F = K^(-T) * E * K^(-1)
    gtFundamental = K.inv().t() * E * K.inv();
    gtFundamental = gtFundamental / cv::norm(gtFundamental);

    int numInliers = static_cast<int>(numPoints * (1.0 - outlierRatio));
    int numOutliers = numPoints - numInliers;

    // Points matrix: [x1, y1, x2, y2] per row
    points = cv::Mat(numPoints, 4, CV_64F);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> distX(50.0, 590.0);
    std::uniform_real_distribution<double> distY(50.0, 430.0);
    std::uniform_real_distribution<double> distDepth(2.0, 20.0);
    std::normal_distribution<double> noise(0.0, 0.5);

    int idx = 0;

    // Generate inliers
    for (int i = 0; i < numInliers; ++i) {
        double u1 = distX(gen);
        double v1 = distY(gen);
        double depth = distDepth(gen);

        // Back-project to 3D
        double x = (u1 - K.at<double>(0, 2)) * depth / K.at<double>(0, 0);
        double y = (v1 - K.at<double>(1, 2)) * depth / K.at<double>(1, 1);
        double z = depth;

        cv::Mat P = (cv::Mat_<double>(3, 1) << x, y, z);
        cv::Mat P2 = R * P + t;

        // Project to image 2
        double u2 = K.at<double>(0, 0) * P2.at<double>(0) / P2.at<double>(2) + K.at<double>(0, 2);
        double v2 = K.at<double>(1, 1) * P2.at<double>(1) / P2.at<double>(2) + K.at<double>(1, 2);

        points.at<double>(idx, 0) = u1 + noise(gen);
        points.at<double>(idx, 1) = v1 + noise(gen);
        points.at<double>(idx, 2) = u2 + noise(gen);
        points.at<double>(idx, 3) = v2 + noise(gen);
        ++idx;
    }

    // Generate outliers
    for (int i = 0; i < numOutliers; ++i) {
        points.at<double>(idx, 0) = distX(gen);
        points.at<double>(idx, 1) = distY(gen);
        points.at<double>(idx, 2) = distX(gen);
        points.at<double>(idx, 3) = distY(gen);
        ++idx;
    }

    // Shuffle
    std::vector<int> indices(numPoints);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    cv::Mat shuffled(numPoints, 4, CV_64F);
    for (int i = 0; i < numPoints; ++i) {
        points.row(indices[i]).copyTo(shuffled.row(i));
    }
    shuffled.copyTo(points);
}

// ============================================================================
// Compute Sampson Error
// ============================================================================

double computeSampsonError(const cv::Mat& F,
                           const cv::Mat& points) {
    double totalError = 0.0;
    const int n = points.rows;

    for (int i = 0; i < n; ++i) {
        cv::Mat p1 = (cv::Mat_<double>(3, 1) <<
            points.at<double>(i, 0), points.at<double>(i, 1), 1.0);
        cv::Mat p2 = (cv::Mat_<double>(3, 1) <<
            points.at<double>(i, 2), points.at<double>(i, 3), 1.0);

        cv::Mat Fp1 = F * p1;
        cv::Mat Ftp2 = F.t() * p2;

        double num = p2.dot(Fp1);
        double denom = Fp1.at<double>(0) * Fp1.at<double>(0)
                     + Fp1.at<double>(1) * Fp1.at<double>(1)
                     + Ftp2.at<double>(0) * Ftp2.at<double>(0)
                     + Ftp2.at<double>(1) * Ftp2.at<double>(1);

        if (denom > 1e-10) {
            totalError += (num * num) / denom;
        }
    }

    return totalError / n;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Fundamental Matrix Estimation using MAGSAC++ ===" << std::endl;
    std::cout << "State-of-the-art threshold-free robust estimation\n" << std::endl;

    // Generate synthetic data
    cv::Mat points, gtF;
    const int numPoints = 200;
    const double outlierRatio = 0.35;
    const double maximumThreshold = 10.0;  // Maximum allowed sigma for scoring
    const double confidence = 0.99;

    generateStereoData(points, gtF, numPoints, outlierRatio);

    std::cout << "Data: " << numPoints << " correspondences, "
              << static_cast<int>(outlierRatio * 100) << "% outliers\n";
    std::cout << "Maximum threshold: " << maximumThreshold << " px\n\n";

    // Image dimensions (for sampler)
    const double imgWidth = 640.0;
    const double imgHeight = 480.0;

    // ========== MAGSAC++ with Progressive NAPSAC Sampler ==========
    std::cout << "--- MAGSAC++ with Progressive NAPSAC Sampler ---" << std::endl;

    // Create fundamental matrix estimator
    magsac::utils::DefaultFundamentalMatrixEstimator estimator(maximumThreshold);

    // Create Progressive NAPSAC sampler (spatially-aware sampling)
    // Grid layers: 16, 8, 4, 2 - from coarse to fine
    gcransac::sampler::ProgressiveNapsacSampler<4> sampler(
        &points,
        { 16, 8, 4, 2 },  // Grid cell sizes
        estimator.sampleSize(),  // Sample size (7 for 7-point algorithm)
        { imgWidth, imgHeight, imgWidth, imgHeight },  // Image dimensions
        0.5  // Blend parameter: 0 = pure spatial, 1 = pure uniform
    );

    // Create MAGSAC++ instance
    MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator> magsac(
        MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator>::MAGSAC_PLUS_PLUS
    );

    // Configure MAGSAC++
    magsac.setMaximumThreshold(maximumThreshold);
    magsac.setIterationLimit(10000);
    magsac.setMinimumIterationNumber(50);

    // Output variables
    gcransac::FundamentalMatrix model;
    ModelScore score;
    int iterations = 0;

    // Run MAGSAC++
    auto start = std::chrono::high_resolution_clock::now();
    bool success = magsac.run(points, confidence, estimator, sampler,
                              model, iterations, score);
    auto end = std::chrono::high_resolution_clock::now();
    double magsacTime = std::chrono::duration<double, std::milli>(end - start).count();

    if (success) {
        // Get the estimated fundamental matrix
        cv::Mat estimatedF = model.descriptor;

        // Compute error
        double sampsonErr = computeSampsonError(estimatedF, points);

        std::cout << "  Success: Yes" << std::endl;
        std::cout << "  Inliers: " << score.inlier_number << "/" << numPoints << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << magsacTime << " ms" << std::endl;
        std::cout << "  Avg Sampson error: " << std::setprecision(6) << sampsonErr << "\n\n";
    } else {
        std::cout << "  Failed to find a valid model.\n\n";
    }

    // ========== MAGSAC++ with Uniform Sampler ==========
    std::cout << "--- MAGSAC++ with Uniform Sampler ---" << std::endl;

    // Create uniform sampler
    gcransac::sampler::UniformSampler uniformSampler(&points);

    // Reset model
    gcransac::FundamentalMatrix model2;
    ModelScore score2;
    int iterations2 = 0;

    // Run MAGSAC++ with uniform sampling
    start = std::chrono::high_resolution_clock::now();
    bool success2 = magsac.run(points, confidence, estimator, uniformSampler,
                               model2, iterations2, score2);
    end = std::chrono::high_resolution_clock::now();
    double magsacTime2 = std::chrono::duration<double, std::milli>(end - start).count();

    if (success2) {
        cv::Mat estimatedF2 = model2.descriptor;
        double sampsonErr2 = computeSampsonError(estimatedF2, points);

        std::cout << "  Success: Yes" << std::endl;
        std::cout << "  Inliers: " << score2.inlier_number << "/" << numPoints << std::endl;
        std::cout << "  Iterations: " << iterations2 << std::endl;
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << magsacTime2 << " ms" << std::endl;
        std::cout << "  Avg Sampson error: " << std::setprecision(6) << sampsonErr2 << "\n\n";
    } else {
        std::cout << "  Failed to find a valid model.\n\n";
    }

    // ========== Original MAGSAC (for comparison) ==========
    std::cout << "--- Original MAGSAC (for comparison) ---" << std::endl;

    // Create original MAGSAC instance
    MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator> magsacOriginal(
        MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator>::MAGSAC_ORIGINAL
    );
    magsacOriginal.setMaximumThreshold(maximumThreshold);
    magsacOriginal.setIterationLimit(10000);
    magsacOriginal.setMinimumIterationNumber(50);

    gcransac::FundamentalMatrix model3;
    ModelScore score3;
    int iterations3 = 0;

    start = std::chrono::high_resolution_clock::now();
    bool success3 = magsacOriginal.run(points, confidence, estimator, sampler,
                                       model3, iterations3, score3);
    end = std::chrono::high_resolution_clock::now();
    double magsacTime3 = std::chrono::duration<double, std::milli>(end - start).count();

    if (success3) {
        cv::Mat estimatedF3 = model3.descriptor;
        double sampsonErr3 = computeSampsonError(estimatedF3, points);

        std::cout << "  Success: Yes" << std::endl;
        std::cout << "  Inliers: " << score3.inlier_number << "/" << numPoints << std::endl;
        std::cout << "  Iterations: " << iterations3 << std::endl;
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << magsacTime3 << " ms\n\n";
    } else {
        std::cout << "  Failed to find a valid model.\n\n";
    }

    // ========== OpenCV USAC_MAGSAC (for comparison) ==========
    std::cout << "--- OpenCV USAC_MAGSAC (for comparison) ---" << std::endl;

    // Convert to OpenCV point format
    std::vector<cv::Point2f> pts1(numPoints), pts2(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        pts1[i] = cv::Point2f(static_cast<float>(points.at<double>(i, 0)),
                              static_cast<float>(points.at<double>(i, 1)));
        pts2[i] = cv::Point2f(static_cast<float>(points.at<double>(i, 2)),
                              static_cast<float>(points.at<double>(i, 3)));
    }

    cv::Mat mask;
    start = std::chrono::high_resolution_clock::now();
    cv::Mat cvF = cv::findFundamentalMat(pts1, pts2, cv::USAC_MAGSAC, 3.0, confidence, mask);
    end = std::chrono::high_resolution_clock::now();
    double opencvTime = std::chrono::duration<double, std::milli>(end - start).count();

    if (!cvF.empty()) {
        int cvInliers = cv::countNonZero(mask);
        double cvSampsonErr = computeSampsonError(cvF, points);

        std::cout << "  Inliers: " << cvInliers << "/" << numPoints << std::endl;
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << opencvTime << " ms" << std::endl;
        std::cout << "  Avg Sampson error: " << std::setprecision(6) << cvSampsonErr << "\n\n";
    }

    // ========== Summary ==========
    std::cout << "=== Summary ===" << std::endl;
    std::cout << std::left << std::setw(30) << "Method"
              << std::setw(12) << "Inliers"
              << std::setw(12) << "Iters"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(66, '-') << std::endl;

    if (success) {
        std::cout << std::left << std::setw(30) << "MAGSAC++ (ProgressiveNAPSAC)"
                  << std::setw(12) << score.inlier_number
                  << std::setw(12) << iterations
                  << std::setw(12) << std::fixed << std::setprecision(2) << magsacTime << std::endl;
    }
    if (success2) {
        std::cout << std::left << std::setw(30) << "MAGSAC++ (Uniform)"
                  << std::setw(12) << score2.inlier_number
                  << std::setw(12) << iterations2
                  << std::setw(12) << std::fixed << std::setprecision(2) << magsacTime2 << std::endl;
    }
    if (success3) {
        std::cout << std::left << std::setw(30) << "MAGSAC Original"
                  << std::setw(12) << score3.inlier_number
                  << std::setw(12) << iterations3
                  << std::setw(12) << std::fixed << std::setprecision(2) << magsacTime3 << std::endl;
    }
    if (!cvF.empty()) {
        std::cout << std::left << std::setw(30) << "OpenCV USAC_MAGSAC"
                  << std::setw(12) << cv::countNonZero(mask)
                  << std::setw(12) << "-"
                  << std::setw(12) << std::fixed << std::setprecision(2) << opencvTime << std::endl;
    }

    std::cout << "\n=== Key Takeaways ===" << std::endl;
    std::cout << "1. MAGSAC++ is threshold-free: marginalizes over noise scale sigma" << std::endl;
    std::cout << "2. Progressive NAPSAC sampler provides spatially-aware sampling" << std::endl;
    std::cout << "3. MAGSAC++ (2020) is faster than original MAGSAC (2019)" << std::endl;
    std::cout << "4. OpenCV's USAC_MAGSAC integrates MAGSAC scoring" << std::endl;

    return 0;
}
