/**
 * Custom RANSAC Implementation
 *
 * This example implements RANSAC from scratch to understand the algorithm
 * internals, and benchmarks each estimator against OpenCV's RANSAC on the
 * SAME data with the SAME inlier rule, so the comparison is fair:
 *
 * 1. Line fitting RANSAC (fixed-seed synthetic points; OpenCV has no line
 *    RANSAC, so cv::fitLine serves as a robust non-RANSAC reference)
 * 2. Homography RANSAC (4-point DLT) on real EuRoC ORB correspondences,
 *    vs cv::findHomography(RANSAC) -- both use forward reprojection error
 * 3. Fundamental matrix RANSAC (normalized 8-point) on the same real
 *    correspondences, vs cv::findFundamentalMat(FM_RANSAC) -- both use
 *    Sampson distance
 *
 * Understanding RANSAC is crucial for Visual SLAM as it's used in:
 * - Feature matching outlier rejection
 * - Motion estimation (Essential/Fundamental matrix)
 * - Loop closure verification
 */

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "ransac_data.h"

// Fixed seed: the estimate must be reproducible run to run so results.csv
// stays stable.
constexpr unsigned kRngSeed = 42;

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

    std::mt19937 rng(kRngSeed);
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

    // Refine line using all inliers (least squares)
    if (bestInlierCount >= 2) {
        std::vector<cv::Point2f> inlierPoints;
        for (int i = 0; i < n; ++i) {
            if (bestMask[i]) {
                inlierPoints.push_back(points[i]);
            }
        }

        cv::Vec4f lineParams;
        cv::fitLine(inlierPoints, lineParams, cv::DIST_L2, 0, 0.01, 0.01);

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

    // Solve using SVD: h is the null space of A.
    // FULL_UV is required: A is 8x9, and without it vt is reduced to 8x9,
    // which has no row 8 (the null-space vector).
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);

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

// Squared forward reprojection error |H*a - b|^2 with H as a row-major
// double pointer. Plain-double math: scoring runs once per point per
// iteration, so it must not allocate. Forward-only error is deliberately the
// same inlier rule cv::findHomography(RANSAC) uses, so inlier counts are
// directly comparable.
static inline double projectError(const double* H,
                                  const cv::Point2f& a,
                                  const cv::Point2f& b) {
    double w = H[6] * a.x + H[7] * a.y + H[8];
    if (std::abs(w) < 1e-12) return std::numeric_limits<double>::max();
    double x = (H[0] * a.x + H[1] * a.y + H[2]) / w;
    double y = (H[3] * a.x + H[4] * a.y + H[5]) / w;
    double dx = x - b.x;
    double dy = y - b.y;
    return dx * dx + dy * dy;
}

/**
 * RANSAC for homography estimation
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

    std::mt19937 rng(kRngSeed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    cv::Mat bestH;
    int bestInlierCount = 0;
    std::vector<bool> bestMask(n, false);
    std::vector<cv::Point2f> sampleSrc(sampleSize), sampleDst(sampleSize);
    std::vector<bool> currentMask(n, false);

    int iteration = 0;
    int adaptiveMaxIters = maxIterations;

    while (iteration < adaptiveMaxIters && iteration < maxIterations) {
        // 1. Random sample: 4 distinct indices by rejection sampling
        //    (O(sampleSize), not an O(n) shuffle of all indices)
        int idx[sampleSize];
        for (int i = 0; i < sampleSize; ++i) {
            bool duplicate;
            do {
                idx[i] = dist(rng);
                duplicate = false;
                for (int j = 0; j < i; ++j) {
                    if (idx[j] == idx[i]) duplicate = true;
                }
            } while (duplicate);
        }
        for (int i = 0; i < sampleSize; ++i) {
            sampleSrc[i] = srcPoints[idx[i]];
            sampleDst[i] = dstPoints[idx[i]];
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
        std::fill(currentMask.begin(), currentMask.end(), false);

        for (int i = 0; i < n; ++i) {
            double error = projectError(H.ptr<double>(), srcPoints[i], dstPoints[i]);
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
// Part 3: Custom RANSAC for Fundamental Matrix
// ============================================================================

/**
 * Normalized 8-point algorithm (works for N >= 8 points).
 *
 * 1. Hartley normalization: translate points to centroid, scale so the mean
 *    distance from origin is sqrt(2).
 * 2. Solve the Nx9 homogeneous system A f = 0 by SVD.
 * 3. Enforce rank-2 by zeroing the smallest singular value of F.
 * 4. Denormalize: F = T2^T * F_norm * T1.
 */
cv::Mat computeFundamental8pt(const std::vector<cv::Point2f>& p1,
                              const std::vector<cv::Point2f>& p2) {
    const int n = static_cast<int>(p1.size());
    if (n < 8 || p2.size() != p1.size()) return cv::Mat();

    // Hartley normalization
    auto normalize = [](const std::vector<cv::Point2f>& pts, cv::Mat& T) {
        cv::Point2d mean(0, 0);
        for (const auto& p : pts) mean += cv::Point2d(p.x, p.y);
        mean *= 1.0 / pts.size();
        double meanDist = 0.0;
        for (const auto& p : pts) {
            meanDist += std::hypot(p.x - mean.x, p.y - mean.y);
        }
        meanDist /= pts.size();
        double scale = (meanDist > 1e-12) ? std::sqrt(2.0) / meanDist : 1.0;
        T = (cv::Mat_<double>(3, 3) << scale, 0, -scale * mean.x,
                                       0, scale, -scale * mean.y,
                                       0, 0, 1);
    };
    cv::Mat T1, T2;
    normalize(p1, T1);
    normalize(p2, T2);

    // Build the Nx9 design matrix from normalized coordinates
    cv::Mat A(n, 9, CV_64F);
    for (int i = 0; i < n; ++i) {
        double x1 = T1.at<double>(0, 0) * p1[i].x + T1.at<double>(0, 2);
        double y1 = T1.at<double>(1, 1) * p1[i].y + T1.at<double>(1, 2);
        double x2 = T2.at<double>(0, 0) * p2[i].x + T2.at<double>(0, 2);
        double y2 = T2.at<double>(1, 1) * p2[i].y + T2.at<double>(1, 2);
        double* row = A.ptr<double>(i);
        row[0] = x2 * x1; row[1] = x2 * y1; row[2] = x2;
        row[3] = y2 * x1; row[4] = y2 * y1; row[5] = y2;
        row[6] = x1;      row[7] = y1;      row[8] = 1.0;
    }

    // f = null space of A. FULL_UV is only needed for the minimal 8x9 case
    // (reduced vt would have 8 rows); for tall matrices the reduced SVD
    // already yields the full 9x9 V, and FULL_UV would wastefully compute an
    // NxN U (this is the refit path over hundreds of inliers).
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, (n < 9) ? static_cast<int>(cv::SVD::FULL_UV) : 0);
    cv::Mat F = vt.row(8).reshape(1, 3);

    // Enforce rank-2
    cv::Mat wf, uf, vtf;
    cv::SVD::compute(F, wf, uf, vtf, cv::SVD::FULL_UV);
    cv::Mat wDiag = cv::Mat::zeros(3, 3, CV_64F);
    wDiag.at<double>(0, 0) = wf.at<double>(0);
    wDiag.at<double>(1, 1) = wf.at<double>(1);  // smallest singular value -> 0
    F = uf * wDiag * vtf;

    // Denormalize and scale
    F = T2.t() * F * T1;
    double norm = cv::norm(F);
    if (norm > 1e-12) F = F / norm;
    return F;
}

// Squared Sampson distance with F as a row-major double pointer.
// Allocation-free for the same reason as projectError. Sampson is also what
// cv::findFundamentalMat's RANSAC thresholds on, keeping the inlier rule
// comparable.
static inline double sampsonSq(const double* F,
                               const cv::Point2f& a,
                               const cv::Point2f& b) {
    // l2 = F * a (epipolar line in image 2), l1 = F^T * b
    double l2x = F[0] * a.x + F[1] * a.y + F[2];
    double l2y = F[3] * a.x + F[4] * a.y + F[5];
    double l2z = F[6] * a.x + F[7] * a.y + F[8];
    double l1x = F[0] * b.x + F[3] * b.y + F[6];
    double l1y = F[1] * b.x + F[4] * b.y + F[7];
    double num = b.x * l2x + b.y * l2y + l2z;
    double denom = l2x * l2x + l2y * l2y + l1x * l1x + l1y * l1y;
    if (denom < 1e-10) return std::numeric_limits<double>::max();
    return (num * num) / denom;
}

/**
 * RANSAC for fundamental matrix estimation (normalized 8-point solver).
 */
std::pair<cv::Mat, std::vector<bool>> ransacFundamental(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    double threshold = 3.0,
    double confidence = 0.99,
    int maxIterations = 2000) {

    const int n = static_cast<int>(pts1.size());
    if (n < 8) {
        return {cv::Mat(), std::vector<bool>(n, false)};
    }

    const int sampleSize = 8;
    const double thresholdSq = threshold * threshold;

    std::mt19937 rng(kRngSeed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    cv::Mat bestF;
    int bestInlierCount = 0;
    std::vector<bool> bestMask(n, false);
    std::vector<cv::Point2f> sample1(sampleSize), sample2(sampleSize);
    std::vector<bool> currentMask(n, false);

    int iteration = 0;
    int adaptiveMaxIters = maxIterations;

    while (iteration < adaptiveMaxIters && iteration < maxIterations) {
        // 1. Random sample: 8 distinct indices by rejection sampling
        int idx[sampleSize];
        for (int i = 0; i < sampleSize; ++i) {
            bool duplicate;
            do {
                idx[i] = dist(rng);
                duplicate = false;
                for (int j = 0; j < i; ++j) {
                    if (idx[j] == idx[i]) duplicate = true;
                }
            } while (duplicate);
        }
        for (int i = 0; i < sampleSize; ++i) {
            sample1[i] = pts1[idx[i]];
            sample2[i] = pts2[idx[i]];
        }

        // 2. Fit model (8-point algorithm)
        cv::Mat F = computeFundamental8pt(sample1, sample2);
        if (F.empty()) {
            ++iteration;
            continue;
        }

        // 3. Count inliers by Sampson distance
        int inlierCount = 0;
        std::fill(currentMask.begin(), currentMask.end(), false);

        const double* Fp = F.ptr<double>();
        for (int i = 0; i < n; ++i) {
            if (sampsonSq(Fp, pts1[i], pts2[i]) < thresholdSq) {
                currentMask[i] = true;
                ++inlierCount;
            }
        }

        // 4. Update best model
        if (inlierCount > bestInlierCount) {
            bestInlierCount = inlierCount;
            bestF = F.clone();
            bestMask = currentMask;

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

    std::cout << "[Fundamental RANSAC] Iterations: " << iteration
              << ", Inliers: " << bestInlierCount << "/" << n << std::endl;

    // 5. Refine F using all inliers (8-point on the full inlier set)
    if (bestInlierCount >= 8) {
        std::vector<cv::Point2f> inlier1, inlier2;
        for (int i = 0; i < n; ++i) {
            if (bestMask[i]) {
                inlier1.push_back(pts1[i]);
                inlier2.push_back(pts2[i]);
            }
        }
        cv::Mat refined = computeFundamental8pt(inlier1, inlier2);
        if (!refined.empty()) {
            bestF = refined;
        }
    }

    return {bestF, bestMask};
}

// ============================================================================
// Tests
// ============================================================================

cv::Mat testLineFitting() {
    std::cout << "\n========== Line Fitting RANSAC Test ==========" << std::endl;

    // Shared fixed-seed synthetic data: y = 0.5x + 100, 70 inliers + 30 outliers
    std::vector<cv::Point2f> points = generateLinePoints();

    auto [line, mask] = ransacLineFitting(points, 5.0, 0.99, 500);

    std::cout << "Estimated line: " << line.a << "x + " << line.b << "y + " << line.c << " = 0"
              << std::endl;

    // Ground truth: 0.5x - y + 100 = 0, normalized
    Line2D gtLine(0.5, -1.0, 100.0);
    std::cout << "Ground truth:   " << gtLine.a << "x + " << gtLine.b << "y + " << gtLine.c << " = 0"
              << std::endl;

    int inlierCount = std::count(mask.begin(), mask.end(), true);
    std::cout << "Inliers found: " << inlierCount << " (expected ~70)" << std::endl;

    // Visualization: points colored by inlier mask, estimated line in blue.
    cv::Mat canvas(420, 400, CV_8UC3, cv::Scalar(255, 255, 255));
    if (std::abs(line.b) > 1e-6) {
        cv::Point2d p0(0.0, -line.c / line.b);
        cv::Point2d p1(canvas.cols, -(line.c + line.a * canvas.cols) / line.b);
        cv::line(canvas, p0, p1, cv::Scalar(255, 0, 0), 2);
    }
    for (size_t i = 0; i < points.size(); ++i) {
        cv::Scalar color = mask[i] ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 255);
        cv::circle(canvas, points[i], 3, color, -1);
    }
    cv::putText(canvas, "green=inlier  red=outlier  blue=fit", cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    return canvas;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Custom RANSAC Implementation ===" << std::endl;
    std::cout << "Understanding RANSAC internals for Visual SLAM" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    cv::Mat lineVis = testLineFitting();

    // Real correspondences shared with every other demo in this chapter
    std::cout << "\n========== Real data (EuRoC ORB correspondences) ==========" << std::endl;
    cv::Mat img1, img2;
    std::vector<cv::Point2f> pts1, pts2;
    if (!loadRealPair(argc, argv, img1, img2, pts1, pts2, 8)) return 1;

    const double threshold = 3.0;
    Timer timer;

    // ---- Homography: custom RANSAC vs cv::findHomography(RANSAC) ----
    std::cout << "\n========== Homography RANSAC (custom vs OpenCV) ==========" << std::endl;

    timer.start();
    auto [customH, maskH] = ransacHomography(pts1, pts2, threshold, 0.99, 2000);
    double customHTime = timer.elapsedMs();
    cv::Mat customHMask = toMask(maskH);

    timer.start();
    cv::Mat cvHMask;
    cv::Mat cvH = cv::findHomography(pts1, pts2, cv::RANSAC, threshold, cvHMask, 2000, 0.99);
    double cvHTime = timer.elapsedMs();

    double customHErr = meanInlierReproj(pts1, pts2, customH, customHMask);
    double cvHErr = meanInlierReproj(pts1, pts2, cvH, cvHMask);
    std::cout << "Custom RANSAC : reproj " << customHErr << " px, inliers "
              << cv::countNonZero(customHMask) << "/" << pts1.size()
              << ", time " << customHTime << " ms\n";
    std::cout << "OpenCV RANSAC : reproj " << cvHErr << " px, inliers "
              << cv::countNonZero(cvHMask) << "/" << pts1.size()
              << ", time " << cvHTime << " ms\n";

    // ---- Fundamental matrix: custom RANSAC vs cv::findFundamentalMat ----
    std::cout << "\n========== Fundamental RANSAC (custom vs OpenCV) ==========" << std::endl;

    timer.start();
    auto [customF, maskF] = ransacFundamental(pts1, pts2, threshold, 0.99, 2000);
    double customFTime = timer.elapsedMs();
    cv::Mat customFMask = toMask(maskF);

    timer.start();
    cv::Mat cvFMask;
    cv::Mat cvF = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, threshold, 0.99, cvFMask);
    double cvFTime = timer.elapsedMs();

    double customFSampson = meanSampson(customF, pts1, pts2, customFMask);
    double cvFSampson = meanSampson(cvF, pts1, pts2, cvFMask);
    std::cout << "Custom RANSAC : Sampson " << customFSampson << ", inliers "
              << cv::countNonZero(customFMask) << "/" << pts1.size()
              << ", time " << customFTime << " ms\n";
    std::cout << "OpenCV RANSAC : Sampson " << cvFSampson << ", inliers "
              << cv::countNonZero(cvFMask) << "/" << pts1.size()
              << ", time " << cvFTime << " ms\n";

    std::cout << "\n=== Summary (same data, same inlier rules) ===" << std::endl;
    std::cout << std::left << std::setw(26) << "Method"
              << std::right << std::setw(12) << "Error"
              << std::setw(12) << "Inliers"
              << std::setw(12) << "Time(ms)" << "\n";
    std::cout << std::string(62, '-') << "\n";
    std::cout << std::left << std::setw(26) << "H  Custom RANSAC"
              << std::right << std::setw(12) << customHErr
              << std::setw(12) << cv::countNonZero(customHMask)
              << std::setw(12) << customHTime << "\n";
    std::cout << std::left << std::setw(26) << "H  OpenCV RANSAC"
              << std::right << std::setw(12) << cvHErr
              << std::setw(12) << cv::countNonZero(cvHMask)
              << std::setw(12) << cvHTime << "\n";
    std::cout << std::left << std::setw(26) << "F  Custom RANSAC (8pt)"
              << std::right << std::setw(12) << customFSampson
              << std::setw(12) << cv::countNonZero(customFMask)
              << std::setw(12) << customFTime << "\n";
    std::cout << std::left << std::setw(26) << "F  OpenCV FM_RANSAC"
              << std::right << std::setw(12) << cvFSampson
              << std::setw(12) << cv::countNonZero(cvFMask)
              << std::setw(12) << cvFTime << "\n";

    std::cout << "\n=== Key Takeaways ===" << std::endl;
    std::cout << "1. RANSAC requires minimum sample size (2 line, 4 H, 8 F)" << std::endl;
    std::cout << "2. Adaptive iteration count saves computation" << std::endl;
    std::cout << "3. Model refinement using all inliers improves accuracy" << std::endl;
    std::cout << "4. Matching the inlier rule is what makes comparisons fair" << std::endl;

    // Visualizations
    cv::Mat hVis = drawMatchesVis(img1, img2, pts1, pts2, customHMask);
    cv::Mat fVis = drawMatchesVis(img1, img2, pts1, pts2, customFMask);
    cv::imwrite("custom_ransac_line.jpg", lineVis);
    cv::imwrite("custom_ransac_homography.jpg", hVis);
    cv::imwrite("custom_ransac_fundamental.jpg", fVis);
    std::cout << "\nSaved: custom_ransac_line.jpg, custom_ransac_homography.jpg, "
                 "custom_ransac_fundamental.jpg" << std::endl;
    if (std::getenv("DISPLAY") != nullptr) {
        cv::imshow("Custom RANSAC: line fitting", lineVis);
        cv::imshow("Custom RANSAC H matches (green=inlier, red=outlier)", hVis);
        cv::imshow("Custom RANSAC F matches (green=inlier, red=outlier)", fVis);
        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0);
    }

    return 0;
}
