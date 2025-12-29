/**
 * Custom RANSAC Implementation
 *
 * This example implements RANSAC from scratch to understand the algorithm
 * internals. We implement:
 *
 * 1. Line fitting RANSAC (simple warm-up)
 * 2. Homography RANSAC (4-point algorithm)
 *
 * Understanding RANSAC is crucial for Visual SLAM as it's used in:
 * - Feature matching outlier rejection
 * - Motion estimation (Essential/Fundamental matrix)
 * - Loop closure verification
 */

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// ============================================================================
// Part 1: Simple RANSAC for Line Fitting
// ============================================================================

/**
 * Line model: ax + by + c = 0 (normalized so that a^2 + b^2 = 1)
 */
struct Line2D {
    double a, b, c;

    Line2D() : a(0), b(1), c(0) {}
    Line2D(double a_, double b_, double c_) : a(a_), b(b_), c(c_) {
        normalize();
    }

    void normalize() {
        double norm = std::sqrt(a * a + b * b);
        if (norm > 1e-10) {
            a /= norm;
            b /= norm;
            c /= norm;
        }
    }

    // Distance from point to line
    double distance(const cv::Point2f& pt) const {
        return std::abs(a * pt.x + b * pt.y + c);
    }

    // Fit line through two points
    static Line2D fromTwoPoints(const cv::Point2f& p1, const cv::Point2f& p2) {
        // Direction vector
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;

        // Normal vector (perpendicular)
        double a = -dy;
        double b = dx;
        double c = -(a * p1.x + b * p1.y);

        return Line2D(a, b, c);
    }
};

/**
 * RANSAC for line fitting
 *
 * @param points Input 2D points
 * @param threshold Maximum distance for inlier classification
 * @param confidence Desired probability of success
 * @param maxIterations Maximum iterations
 * @return Best-fit line and inlier mask
 */
std::pair<Line2D, std::vector<bool>> ransacLineFitting(
    const std::vector<cv::Point2f>& points,
    double threshold = 3.0,
    double confidence = 0.99,
    int maxIterations = 1000) {

    const int n = static_cast<int>(points.size());
    if (n < 2) {
        return {Line2D(), std::vector<bool>(n, false)};
    }

    // Random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, n - 1);

    Line2D bestLine;
    int bestInlierCount = 0;
    std::vector<bool> bestMask(n, false);

    int iteration = 0;
    int adaptiveMaxIters = maxIterations;

    while (iteration < adaptiveMaxIters && iteration < maxIterations) {
        // 1. Random sample: select 2 points (minimum for line)
        int idx1 = dist(rng);
        int idx2 = dist(rng);
        while (idx2 == idx1) {
            idx2 = dist(rng);
        }

        // 2. Fit model (line through 2 points)
        Line2D line = Line2D::fromTwoPoints(points[idx1], points[idx2]);

        // 3. Count inliers
        int inlierCount = 0;
        std::vector<bool> currentMask(n, false);

        for (int i = 0; i < n; ++i) {
            double d = line.distance(points[i]);
            if (d < threshold) {
                currentMask[i] = true;
                ++inlierCount;
            }
        }

        // 4. Update best model
        if (inlierCount > bestInlierCount) {
            bestInlierCount = inlierCount;
            bestLine = line;
            bestMask = currentMask;

            // Adaptive iteration count based on inlier ratio
            double inlierRatio = static_cast<double>(inlierCount) / n;
            if (inlierRatio > 0.1) {
                // N = log(1 - p) / log(1 - w^s)
                // p = confidence, w = inlier ratio, s = sample size (2 for line)
                double w = inlierRatio;
                double p = confidence;
                int s = 2;
                double newMaxIters = std::log(1.0 - p) / std::log(1.0 - std::pow(w, s));
                adaptiveMaxIters = std::min(static_cast<int>(newMaxIters) + 1, maxIterations);
            }
        }

        ++iteration;
    }

    std::cout << "[Line RANSAC] Iterations: " << iteration
              << ", Inliers: " << bestInlierCount << "/" << n << std::endl;

    // Optional: Refine line using all inliers (least squares)
    if (bestInlierCount >= 2) {
        std::vector<cv::Point2f> inlierPoints;
        for (int i = 0; i < n; ++i) {
            if (bestMask[i]) {
                inlierPoints.push_back(points[i]);
            }
        }

        // Fit line using OpenCV (for refinement)
        cv::Vec4f lineParams;
        cv::fitLine(inlierPoints, lineParams, cv::DIST_L2, 0, 0.01, 0.01);

        // Convert to ax + by + c = 0 form
        double vx = lineParams[0], vy = lineParams[1];
        double x0 = lineParams[2], y0 = lineParams[3];
        bestLine = Line2D(-vy, vx, vy * x0 - vx * y0);
    }

    return {bestLine, bestMask};
}

// ============================================================================
// Part 2: Custom RANSAC for Homography
// ============================================================================

/**
 * Compute homography from 4 point correspondences using DLT
 *
 * For each correspondence (x, y) -> (x', y'):
 * [ -x  -y  -1   0   0   0   x*x'   y*x'   x' ] [ h1 ]   [ 0 ]
 * [  0   0   0  -x  -y  -1   x*y'   y*y'   y' ] [ h2 ] = [ 0 ]
 *                                                [...]
 *                                                [ h9 ]
 */
cv::Mat computeHomographyDLT(const std::vector<cv::Point2f>& src,
                              const std::vector<cv::Point2f>& dst) {
    if (src.size() != 4 || dst.size() != 4) {
        return cv::Mat();
    }

    // Build the design matrix A (8x9)
    cv::Mat A = cv::Mat::zeros(8, 9, CV_64F);

    for (int i = 0; i < 4; ++i) {
        double x = src[i].x, y = src[i].y;
        double xp = dst[i].x, yp = dst[i].y;

        A.at<double>(2 * i, 0) = -x;
        A.at<double>(2 * i, 1) = -y;
        A.at<double>(2 * i, 2) = -1;
        A.at<double>(2 * i, 6) = x * xp;
        A.at<double>(2 * i, 7) = y * xp;
        A.at<double>(2 * i, 8) = xp;

        A.at<double>(2 * i + 1, 3) = -x;
        A.at<double>(2 * i + 1, 4) = -y;
        A.at<double>(2 * i + 1, 5) = -1;
        A.at<double>(2 * i + 1, 6) = x * yp;
        A.at<double>(2 * i + 1, 7) = y * yp;
        A.at<double>(2 * i + 1, 8) = yp;
    }

    // Solve using SVD: h is the null space of A
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt);

    // Last row of Vt (or last column of V) is the solution
    cv::Mat h = vt.row(8).t();

    // Reshape to 3x3
    cv::Mat H = h.reshape(1, 3);

    // Normalize so H[2,2] = 1 (if non-zero)
    if (std::abs(H.at<double>(2, 2)) > 1e-10) {
        H = H / H.at<double>(2, 2);
    }

    return H;
}

/**
 * Compute symmetric transfer error for homography
 */
double computeTransferError(const cv::Mat& H,
                            const cv::Point2f& src,
                            const cv::Point2f& dst) {
    // Forward projection error
    cv::Mat p1 = (cv::Mat_<double>(3, 1) << src.x, src.y, 1.0);
    cv::Mat p2_proj = H * p1;
    double x2 = p2_proj.at<double>(0) / p2_proj.at<double>(2);
    double y2 = p2_proj.at<double>(1) / p2_proj.at<double>(2);
    double fwdError = std::pow(x2 - dst.x, 2) + std::pow(y2 - dst.y, 2);

    // Backward projection error
    cv::Mat Hinv = H.inv();
    cv::Mat p2 = (cv::Mat_<double>(3, 1) << dst.x, dst.y, 1.0);
    cv::Mat p1_proj = Hinv * p2;
    double x1 = p1_proj.at<double>(0) / p1_proj.at<double>(2);
    double y1 = p1_proj.at<double>(1) / p1_proj.at<double>(2);
    double bwdError = std::pow(x1 - src.x, 2) + std::pow(y1 - src.y, 2);

    // Symmetric transfer error
    return fwdError + bwdError;
}

/**
 * RANSAC for homography estimation
 *
 * @param srcPoints Source image points
 * @param dstPoints Destination image points
 * @param threshold Maximum reprojection error for inlier
 * @param confidence Desired probability of success
 * @param maxIterations Maximum iterations
 * @return Homography matrix and inlier mask
 */
std::pair<cv::Mat, std::vector<bool>> ransacHomography(
    const std::vector<cv::Point2f>& srcPoints,
    const std::vector<cv::Point2f>& dstPoints,
    double threshold = 3.0,
    double confidence = 0.99,
    int maxIterations = 2000) {

    const int n = static_cast<int>(srcPoints.size());
    if (n < 4) {
        return {cv::Mat(), std::vector<bool>(n, false)};
    }

    const int sampleSize = 4;  // Minimum points for homography
    const double thresholdSq = threshold * threshold;

    // Random number generator
    std::mt19937 rng(std::random_device{}());

    cv::Mat bestH;
    int bestInlierCount = 0;
    std::vector<bool> bestMask(n, false);

    int iteration = 0;
    int adaptiveMaxIters = maxIterations;

    while (iteration < adaptiveMaxIters && iteration < maxIterations) {
        // 1. Random sample: select 4 points
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<cv::Point2f> sampleSrc(sampleSize), sampleDst(sampleSize);
        for (int i = 0; i < sampleSize; ++i) {
            sampleSrc[i] = srcPoints[indices[i]];
            sampleDst[i] = dstPoints[indices[i]];
        }

        // Check for degenerate configuration (3 collinear points)
        cv::Point2f v1 = sampleSrc[1] - sampleSrc[0];
        cv::Point2f v2 = sampleSrc[2] - sampleSrc[0];
        cv::Point2f v3 = sampleSrc[3] - sampleSrc[0];
        double cross1 = v1.x * v2.y - v1.y * v2.x;
        double cross2 = v1.x * v3.y - v1.y * v3.x;
        if (std::abs(cross1) < 1e-6 || std::abs(cross2) < 1e-6) {
            ++iteration;
            continue;  // Skip degenerate sample
        }

        // 2. Fit model (homography from 4 points)
        cv::Mat H = computeHomographyDLT(sampleSrc, sampleDst);
        if (H.empty()) {
            ++iteration;
            continue;
        }

        // 3. Count inliers
        int inlierCount = 0;
        std::vector<bool> currentMask(n, false);

        for (int i = 0; i < n; ++i) {
            double error = computeTransferError(H, srcPoints[i], dstPoints[i]);
            if (error < thresholdSq) {
                currentMask[i] = true;
                ++inlierCount;
            }
        }

        // 4. Update best model
        if (inlierCount > bestInlierCount) {
            bestInlierCount = inlierCount;
            bestH = H.clone();
            bestMask = currentMask;

            // Adaptive iteration count
            double inlierRatio = static_cast<double>(inlierCount) / n;
            if (inlierRatio > 0.1) {
                double w = inlierRatio;
                double p = confidence;
                double newMaxIters = std::log(1.0 - p) / std::log(1.0 - std::pow(w, sampleSize));
                adaptiveMaxIters = std::min(static_cast<int>(newMaxIters) + 1, maxIterations);
            }
        }

        ++iteration;
    }

    std::cout << "[Homography RANSAC] Iterations: " << iteration
              << ", Inliers: " << bestInlierCount << "/" << n << std::endl;

    // 5. Refine homography using all inliers
    if (bestInlierCount >= 4) {
        std::vector<cv::Point2f> inlierSrc, inlierDst;
        for (int i = 0; i < n; ++i) {
            if (bestMask[i]) {
                inlierSrc.push_back(srcPoints[i]);
                inlierDst.push_back(dstPoints[i]);
            }
        }

        // Use OpenCV for refinement (DLT with normalization + SVD)
        cv::Mat refinedH = cv::findHomography(inlierSrc, inlierDst, 0);  // No RANSAC, just DLT
        if (!refinedH.empty()) {
            bestH = refinedH;
        }
    }

    return {bestH, bestMask};
}

// ============================================================================
// Test Functions
// ============================================================================

void testLineFitting() {
    std::cout << "\n========== Line Fitting RANSAC Test ==========" << std::endl;

    // Generate synthetic data: line y = 0.5x + 100 with outliers
    std::vector<cv::Point2f> points;
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0, 2.0);

    // Inliers (70%)
    for (int i = 0; i < 70; ++i) {
        float x = static_cast<float>(i * 5);
        float y = 0.5f * x + 100.0f + noise(rng);
        points.emplace_back(x, y);
    }

    // Outliers (30%)
    std::uniform_real_distribution<float> outlierX(0, 350);
    std::uniform_real_distribution<float> outlierY(0, 400);
    for (int i = 0; i < 30; ++i) {
        points.emplace_back(outlierX(rng), outlierY(rng));
    }

    // Shuffle
    std::shuffle(points.begin(), points.end(), rng);

    // Run RANSAC
    auto [line, mask] = ransacLineFitting(points, 5.0, 0.99, 500);

    std::cout << "Estimated line: " << line.a << "x + " << line.b << "y + " << line.c << " = 0"
              << std::endl;

    // Ground truth: 0.5x - y + 100 = 0, normalized
    Line2D gtLine(0.5, -1.0, 100.0);
    std::cout << "Ground truth:   " << gtLine.a << "x + " << gtLine.b << "y + " << gtLine.c << " = 0"
              << std::endl;

    // Count inliers
    int inlierCount = std::count(mask.begin(), mask.end(), true);
    std::cout << "Inliers found: " << inlierCount << " (expected ~70)" << std::endl;
}

void testHomography() {
    std::cout << "\n========== Homography RANSAC Test ==========" << std::endl;

    // Ground truth homography
    cv::Mat gtH = (cv::Mat_<double>(3, 3) << 0.95, -0.08, 30,
                                              0.05, 0.92, 20,
                                              0.0001, 0.00015, 1.0);

    // Generate synthetic correspondences
    std::vector<cv::Point2f> srcPts, dstPts;
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> ptDist(50, 550);
    std::normal_distribution<float> noise(0, 1.0);

    // Inliers (60%)
    for (int i = 0; i < 60; ++i) {
        float x = ptDist(rng);
        float y = ptDist(rng);
        srcPts.emplace_back(x, y);

        cv::Mat p = (cv::Mat_<double>(3, 1) << x, y, 1.0);
        cv::Mat pp = gtH * p;
        float xp = static_cast<float>(pp.at<double>(0) / pp.at<double>(2)) + noise(rng);
        float yp = static_cast<float>(pp.at<double>(1) / pp.at<double>(2)) + noise(rng);
        dstPts.emplace_back(xp, yp);
    }

    // Outliers (40%)
    for (int i = 0; i < 40; ++i) {
        srcPts.emplace_back(ptDist(rng), ptDist(rng));
        dstPts.emplace_back(ptDist(rng), ptDist(rng));
    }

    // Shuffle
    std::vector<size_t> indices(srcPts.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<cv::Point2f> shuffledSrc, shuffledDst;
    for (size_t idx : indices) {
        shuffledSrc.push_back(srcPts[idx]);
        shuffledDst.push_back(dstPts[idx]);
    }

    // Run custom RANSAC
    auto start = std::chrono::high_resolution_clock::now();
    auto [H, mask] = ransacHomography(shuffledSrc, shuffledDst, 3.0, 0.99, 2000);
    auto end = std::chrono::high_resolution_clock::now();
    double customTime = std::chrono::duration<double, std::milli>(end - start).count();

    // Compare with OpenCV RANSAC
    start = std::chrono::high_resolution_clock::now();
    cv::Mat cvMask;
    cv::Mat cvH = cv::findHomography(shuffledSrc, shuffledDst, cv::RANSAC, 3.0, cvMask);
    end = std::chrono::high_resolution_clock::now();
    double cvTime = std::chrono::duration<double, std::milli>(end - start).count();

    // Results
    if (!H.empty()) {
        cv::Mat normH = H / H.at<double>(2, 2);
        cv::Mat normGtH = gtH / gtH.at<double>(2, 2);
        double customError = cv::norm(normH - normGtH);

        int customInliers = std::count(mask.begin(), mask.end(), true);
        int cvInliers = cv::countNonZero(cvMask);

        std::cout << "\nGround truth H:\n" << normGtH << std::endl;
        std::cout << "\nCustom RANSAC H:\n" << normH << std::endl;
        std::cout << "Error: " << customError << ", Inliers: " << customInliers
                  << ", Time: " << customTime << " ms" << std::endl;

        if (!cvH.empty()) {
            cv::Mat normCvH = cvH / cvH.at<double>(2, 2);
            double cvError = cv::norm(normCvH - normGtH);
            std::cout << "\nOpenCV RANSAC H:\n" << normCvH << std::endl;
            std::cout << "Error: " << cvError << ", Inliers: " << cvInliers
                      << ", Time: " << cvTime << " ms" << std::endl;
        }
    }
}

int main() {
    std::cout << "=== Custom RANSAC Implementation ===" << std::endl;
    std::cout << "Understanding RANSAC internals for Visual SLAM" << std::endl;

    testLineFitting();
    testHomography();

    std::cout << "\n=== Key Takeaways ===" << std::endl;
    std::cout << "1. RANSAC requires minimum sample size (2 for line, 4 for homography)" << std::endl;
    std::cout << "2. Adaptive iteration count saves computation" << std::endl;
    std::cout << "3. Model refinement using all inliers improves accuracy" << std::endl;
    std::cout << "4. Degenerate configuration checking prevents bad solutions" << std::endl;

    return 0;
}
