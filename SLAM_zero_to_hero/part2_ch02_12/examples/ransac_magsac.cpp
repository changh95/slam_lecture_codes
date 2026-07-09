/**
 * Homography / Fundamental Matrix Estimation using MAGSAC++
 *
 * This example demonstrates MAGSAC++ (Marginalizing Sample Consensus),
 * a state-of-the-art robust estimation algorithm that provides:
 *
 * 1. Threshold-free estimation: Marginalizes over noise scale sigma
 * 2. Sigma-scoring: Uses probabilistic scoring instead of binary inlier/outlier
 * 3. Progressive NAPSAC sampling: Spatially-aware sampling for faster convergence
 *
 * Both models run on the real KITTI ORB correspondences shared by every demo
 * in this chapter, benchmarked against OpenCV RANSAC on the same data. Since
 * MAGSAC++ has no fixed inlier threshold, its inlier count is derived
 * post-hoc with the same 3 px rule used everywhere else, and every method's
 * model is scored by the shared metrics (mean inlier reprojection error for
 * H, mean squared Sampson distance for F).
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

#include <Eigen/Core>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ransac_data.h"

// Derive a 3 px inlier mask for a homography by forward reprojection error
// (the same rule cv::findHomography(RANSAC) uses).
static cv::Mat reprojMask(const cv::Mat& H,
                          const std::vector<cv::Point2f>& pts1,
                          const std::vector<cv::Point2f>& pts2,
                          double threshold) {
    const double thrSq = threshold * threshold;
    cv::Mat mask = cv::Mat::zeros(static_cast<int>(pts1.size()), 1, CV_8U);
    if (H.empty()) return mask;
    const double* h = H.ptr<double>();
    for (size_t i = 0; i < pts1.size(); ++i) {
        double w = h[6] * pts1[i].x + h[7] * pts1[i].y + h[8];
        if (std::abs(w) < 1e-12) continue;
        double dx = (h[0] * pts1[i].x + h[1] * pts1[i].y + h[2]) / w - pts2[i].x;
        double dy = (h[3] * pts1[i].x + h[4] * pts1[i].y + h[5]) / w - pts2[i].y;
        if (dx * dx + dy * dy < thrSq) mask.at<uchar>(static_cast<int>(i)) = 1;
    }
    return mask;
}

// Derive a 3 px inlier mask for a fundamental matrix by Sampson distance.
static cv::Mat sampsonMask(const cv::Mat& F,
                           const std::vector<cv::Point2f>& pts1,
                           const std::vector<cv::Point2f>& pts2,
                           double threshold) {
    const double thrSq = threshold * threshold;
    cv::Mat mask = cv::Mat::zeros(static_cast<int>(pts1.size()), 1, CV_8U);
    if (F.empty() || F.rows != 3) return mask;
    const double* f = F.ptr<double>();
    for (size_t i = 0; i < pts1.size(); ++i) {
        double l2x = f[0] * pts1[i].x + f[1] * pts1[i].y + f[2];
        double l2y = f[3] * pts1[i].x + f[4] * pts1[i].y + f[5];
        double l2z = f[6] * pts1[i].x + f[7] * pts1[i].y + f[8];
        double l1x = f[0] * pts2[i].x + f[3] * pts2[i].y + f[6];
        double l1y = f[1] * pts2[i].x + f[4] * pts2[i].y + f[7];
        double num = pts2[i].x * l2x + pts2[i].y * l2y + l2z;
        double denom = l2x * l2x + l2y * l2y + l1x * l1x + l1y * l1y;
        if (denom < 1e-10) continue;
        if ((num * num) / denom < thrSq) mask.at<uchar>(static_cast<int>(i)) = 1;
    }
    return mask;
}

struct RowResult {
    std::string name;
    double err;
    int inliers;
    double time_ms;
};

// gcransac stores model parameters as Eigen::MatrixXd; the shared metrics
// take cv::Mat.
static cv::Mat descriptorToCv(const Eigen::MatrixXd& m) {
    cv::Mat out(static_cast<int>(m.rows()), static_cast<int>(m.cols()), CV_64F);
    for (int r = 0; r < out.rows; ++r)
        for (int c = 0; c < out.cols; ++c) out.at<double>(r, c) = m(r, c);
    return out;
}

int main(int argc, char* argv[]) {
    std::cout << "=== H / F Estimation using MAGSAC++ ===" << std::endl;
    std::cout << "State-of-the-art threshold-free robust estimation\n" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // Real correspondences shared with every other demo in this chapter
    cv::Mat img1, img2;
    std::vector<cv::Point2f> pts1, pts2;
    if (!loadRealPair(argc, argv, img1, img2, pts1, pts2, 8)) return 1;
    const int n = static_cast<int>(pts1.size());

    // MAGSAC wants correspondences as an Nx4 double matrix [x1 y1 x2 y2]
    cv::Mat points(n, 4, CV_64F);
    for (int i = 0; i < n; ++i) {
        points.at<double>(i, 0) = pts1[i].x;
        points.at<double>(i, 1) = pts1[i].y;
        points.at<double>(i, 2) = pts2[i].x;
        points.at<double>(i, 3) = pts2[i].y;
    }

    const double threshold = 3.0;          // shared inlier rule for reporting
    const double maximumThreshold = 10.0;  // sigma-marginalization upper bound
    const double confidence = 0.99;
    const double imgWidth = static_cast<double>(img1.cols);
    const double imgHeight = static_cast<double>(img1.rows);
    Timer timer;
    std::vector<RowResult> rows;

    // ================= Homography =================
    std::cout << "\n--- MAGSAC++ homography vs cv::findHomography(RANSAC) ---" << std::endl;
    cv::Mat magsacHMask;
    {
        magsac::utils::DefaultHomographyEstimator estimator;
        gcransac::sampler::ProgressiveNapsacSampler<4> sampler(
            &points, {16, 8, 4, 2}, estimator.sampleSize(),
            {imgWidth, imgHeight, imgWidth, imgHeight}, 0.5);

        MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator> magsac(
            MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator>::MAGSAC_PLUS_PLUS);
        magsac.setMaximumThreshold(maximumThreshold);
        magsac.setReferenceThreshold(threshold);  // interrupt sigma-consensus early
        magsac.setIterationLimit(2000);
        magsac.setMinimumIterationNumber(50);

        gcransac::Homography model;
        ModelScore score;
        int iterations = 0;
        timer.start();
        bool ok = magsac.run(points, confidence, estimator, sampler,
                             model, iterations, score);
        double ms = timer.elapsedMs();
        if (ok) {
            cv::Mat H = descriptorToCv(model.descriptor);
            magsacHMask = reprojMask(H, pts1, pts2, threshold);
            double err = meanInlierReproj(pts1, pts2, H, magsacHMask);
            int inliers = cv::countNonZero(magsacHMask);
            std::cout << "  MAGSAC++ : reproj " << err << " px, inliers (3px rule) "
                      << inliers << "/" << n << ", iterations " << iterations
                      << ", time " << ms << " ms\n";
            rows.push_back({"H MAGSAC++", err, inliers, ms});
        } else {
            std::cout << "  MAGSAC++ : failed to find a valid model\n";
        }
    }
    {
        cv::Mat mask;
        timer.start();
        cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, threshold, mask, 2000, confidence);
        double ms = timer.elapsedMs();
        double err = meanInlierReproj(pts1, pts2, H, mask);
        std::cout << "  OpenCV   : reproj " << err << " px, inliers "
                  << cv::countNonZero(mask) << "/" << n << ", time " << ms << " ms\n";
        rows.push_back({"H OpenCV RANSAC", err, cv::countNonZero(mask), ms});
    }

    // ================= Fundamental matrix =================
    std::cout << "\n--- MAGSAC++ fundamental vs cv::findFundamentalMat(FM_RANSAC) ---" << std::endl;
    std::cout << std::setprecision(6);
    {
        magsac::utils::DefaultFundamentalMatrixEstimator estimator(maximumThreshold);
        gcransac::sampler::ProgressiveNapsacSampler<4> sampler(
            &points, {16, 8, 4, 2}, estimator.sampleSize(),
            {imgWidth, imgHeight, imgWidth, imgHeight}, 0.5);

        MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator> magsac(
            MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator>::MAGSAC_PLUS_PLUS);
        magsac.setMaximumThreshold(maximumThreshold);
        magsac.setReferenceThreshold(threshold);  // interrupt sigma-consensus early
        // The KITTI scene is dominated by a plane, so the F confidence
        // criterion converges erratically; cap iterations to keep the demo
        // bounded (MAGSAC++'s own degensac handles the degeneracy).
        magsac.setIterationLimit(2000);
        magsac.setMinimumIterationNumber(50);

        gcransac::FundamentalMatrix model;
        ModelScore score;
        int iterations = 0;
        timer.start();
        bool ok = magsac.run(points, confidence, estimator, sampler,
                             model, iterations, score);
        double ms = timer.elapsedMs();
        if (ok) {
            cv::Mat F = descriptorToCv(model.descriptor);
            cv::Mat mask = sampsonMask(F, pts1, pts2, threshold);
            double err = meanSampson(F, pts1, pts2, mask);
            int inliers = cv::countNonZero(mask);
            std::cout << "  MAGSAC++ : Sampson " << err << ", inliers (3px rule) "
                      << inliers << "/" << n << ", iterations " << iterations
                      << ", time " << ms << " ms\n";
            rows.push_back({"F MAGSAC++", err, inliers, ms});
        } else {
            std::cout << "  MAGSAC++ : failed to find a valid model\n";
        }
    }
    {
        cv::Mat mask;
        timer.start();
        cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, threshold, confidence, mask);
        double ms = timer.elapsedMs();
        double err = meanSampson(F, pts1, pts2, mask);
        std::cout << "  OpenCV   : Sampson " << err << ", inliers "
                  << cv::countNonZero(mask) << "/" << n << ", time " << ms << " ms\n";
        rows.push_back({"F OpenCV FM_RANSAC", err, cv::countNonZero(mask), ms});
    }

    // ================= Summary =================
    std::cout << "\n=== Summary (same data, shared metrics) ===" << std::endl;
    std::cout << std::setprecision(4);
    std::cout << std::left << std::setw(24) << "Method"
              << std::right << std::setw(12) << "Error"
              << std::setw(12) << "Inliers"
              << std::setw(12) << "Time(ms)" << "\n";
    std::cout << std::string(60, '-') << "\n";
    for (const auto& r : rows) {
        std::cout << std::left << std::setw(24) << r.name
                  << std::right << std::setw(12) << r.err
                  << std::setw(12) << r.inliers
                  << std::setw(12) << r.time_ms << "\n";
    }

    std::cout << "\n=== Key Takeaways ===" << std::endl;
    std::cout << "1. MAGSAC++ is threshold-free: marginalizes over noise scale sigma" << std::endl;
    std::cout << "2. Progressive NAPSAC sampler provides spatially-aware sampling" << std::endl;
    std::cout << "3. Its inlier count here is derived post-hoc with the shared 3px rule" << std::endl;
    std::cout << "4. OpenCV's USAC_MAGSAC integrates the same scoring idea" << std::endl;

    // Visualization: MAGSAC++ homography inliers on the real pair.
    if (!magsacHMask.empty()) {
        cv::Mat vis = drawMatchesVis(img1, img2, pts1, pts2, magsacHMask);
        cv::imwrite("magsac_h_matches.jpg", vis);
        std::cout << "\nSaved: magsac_h_matches.jpg" << std::endl;
        if (std::getenv("DISPLAY") != nullptr) {
            cv::imshow("MAGSAC++ H matches (green=inlier, red=outlier)", vis);
            std::cout << "Press any key to exit..." << std::endl;
            cv::waitKey(0);
        }
    }
    return 0;
}
