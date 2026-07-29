/**
 * Custom RANSAC Implementation
 *
 * RANSAC written from scratch to expose the algorithm's internals, then
 * benchmarked against OpenCV on the same data.
 *
 * Division of labour: OpenCV is the interface only -- cv::Point2f containers
 * in, cv::Mat out, image I/O, and the baseline estimators we benchmark against
 * in main(). Every numerical step is Eigen: Hartley normalization, the minimal
 * solvers, the null-space solve, rank-2 enforcement, the total-least-squares
 * line refit, and every residual. Nothing here delegates estimation to
 * cv::findHomography, cv::findFundamentalMat or cv::fitLine.
 *
 * 1. Line fitting RANSAC (fixed-seed synthetic points) with a total
 *    least-squares refit. OpenCV has no line RANSAC to compare against.
 * 2. Homography RANSAC (Hartley-normalized N-point DLT) on real EuRoC ORB
 *    correspondences, vs cv::findHomography(RANSAC). Both score with forward
 *    reprojection error, so their inlier counts ARE directly comparable
 *    (verified against OpenCV's HomographyEstimatorCallback::computeError,
 *    which is dx*dx + dy*dy on the forward-mapped point).
 * 3. Fundamental matrix RANSAC (normalized 8-point) on the same
 *    correspondences, vs cv::findFundamentalMat(FM_RANSAC). Here the two
 *    inlier rules DIFFER, and the demo prints both -- see below.
 *
 * The F inlier rule is not shared, and pretending otherwise inflates our
 * numbers. cv::findFundamentalMat's classic path scores with the max-form
 * symmetric epipolar distance
 *     err = max(d^2/A, d^2/B) = d^2 / min(A,B)
 * (OpenCV 4.x, modules/calib3d/src/fundam.cpp, FMEstimatorCallback::
 * computeError), while this file and every USAC variant score with Sampson
 *     err = d^2 / (A + B)
 * where d = x2^T F x1, A = |(F^T x2)_xy|^2, B = |(F x1)_xy|^2. Since
 * min(A,B) <= (A+B)/2, OpenCV's residual is always at least twice Sampson's,
 * so at one nominal 3 px threshold OpenCV's rule is strictly the tighter of
 * the two. OpenCV does implement Sampson, but only for USAC
 * (modules/calib3d/src/usac/estimator.cpp, SampsonErrorImpl). main() therefore
 * scores every F model under BOTH rules: the count gap between the two columns
 * is the cost of the rule, not of the estimator.
 *
 * Three implementation details do the heavy lifting, and they are the point of
 * the demo -- RANSAC's performance lives in them, not in the pseudocode:
 *   - Hartley normalization inside the minimal solver, not only the refit.
 *     Raw pixel coordinates (~750) make the design matrix badly conditioned.
 *   - Local optimization (LO): every time a new best model appears, refit it
 *     on its own inlier set and re-score. This is what LO-RANSAC adds over
 *     plain RANSAC and it is the single biggest win here.
 *   - The reported mask always belongs to the reported model. Refitting and
 *     then reporting the pre-refit mask understates the estimate.
 *
 * Understanding RANSAC is crucial for Visual SLAM as it's used in:
 * - Feature matching outlier rejection
 * - Motion estimation (Essential/Fundamental matrix)
 * - Loop closure verification
 */

// OpenCV is the interface: containers, the baseline estimators we benchmark
// against in main(), and image I/O. None of the custom estimation uses it.
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "ransac_data.h"

// Fixed seed: the estimate must be reproducible run to run so results.csv
// stays stable.
constexpr unsigned kRngSeed = 42;

// Floor for the adaptive stopping rule. The textbook N = log(1-p)/log(1-w^s)
// is a statement about the chance of drawing one all-inlier sample, and once LO
// pushes w up it can ask for barely a dozen samples. A dozen 4-point samples is
// a thin search whatever the probability says, so we keep sampling.
constexpr int kMinIterations = 50;

// LO rounds spent on each new best model, and rounds of the same refit/re-score
// step used as the final polish.
constexpr int kLoSteps = 2;
constexpr int kPolishSteps = 5;

// ============================================================================
// Shared scaffolding
// ============================================================================

// Score of a model against the full correspondence set.
struct FitScore {
    int inliers = 0;
    double errSum = std::numeric_limits<double>::max();

    // Prefer more inliers; break ties on total inlier residual, so a refit that
    // keeps the same support but fits it better still wins. Strict improvement
    // in one of the two guarantees the LO loops terminate.
    bool betterThan(const FitScore& o) const {
        return inliers > o.inliers || (inliers == o.inliers && errSum < o.errSum);
    }
};

// Eigen -> OpenCV, at the boundary only: callers expect a cv::Mat model.
static cv::Mat toCvMat(const Eigen::Matrix3d& m) {
    cv::Mat out(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) out.at<double>(r, c) = m(r, c);
    }
    return out;
}

// OpenCV -> Eigen, so a baseline model can be re-scored by our own rules.
static bool toEigen(const cv::Mat& m, Eigen::Matrix3d& out) {
    if (m.empty() || m.rows != 3 || m.cols != 3) return false;
    cv::Mat d;
    m.convertTo(d, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) out(r, c) = d.at<double>(r, c);
    }
    return true;
}

// Rejection sampling for k distinct indices: O(k), not an O(n) shuffle.
static void drawDistinct(int* idx, int k,
                         std::uniform_int_distribution<int>& dist,
                         std::mt19937& rng) {
    for (int i = 0; i < k; ++i) {
        bool duplicate;
        do {
            idx[i] = dist(rng);
            duplicate = false;
            for (int j = 0; j < i; ++j) {
                if (idx[j] == idx[i]) duplicate = true;
            }
        } while (duplicate);
    }
}

// N = log(1-p) / log(1-w^s). Returned unclamped so the demo can print what the
// formula asked for next to the floored budget we actually ran.
static int adaptiveIterations(int inliers, int n, int sampleSize,
                              double confidence, int maxIterations) {
    const double w = static_cast<double>(inliers) / n;
    if (w <= 0.1) return maxIterations;  // too few inliers to trust the estimate
    const double denom = std::log1p(-std::pow(w, sampleSize));
    // w^s underflowed (denom ~ 0): the formula is uninformative, keep sampling.
    if (!(denom < -1e-12)) return maxIterations;
    const double n_iter = std::log1p(-confidence) / denom;
    if (!std::isfinite(n_iter)) return maxIterations;
    // Compare as double before narrowing: n_iter can far exceed INT_MAX.
    if (n_iter >= maxIterations) return maxIterations;
    return static_cast<int>(n_iter) + 1;
}

// Scratch buffers reused across solver calls so the RANSAC loop does not churn
// the heap once per iteration.
struct SolverScratch {
    std::vector<Eigen::Vector2d> a, b;
    Eigen::MatrixXd A;
};

// Hartley normalization of the selected subset: translate to the centroid and
// scale so the mean distance to the origin is sqrt(2). Returns T with
// x_norm = T * x, and writes the normalized points into `out`.
static Eigen::Matrix3d hartleyNormalize(const std::vector<cv::Point2f>& pts,
                                        const int* idx, int m,
                                        std::vector<Eigen::Vector2d>& out) {
    out.resize(m);
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (int i = 0; i < m; ++i) {
        out[i] = Eigen::Vector2d(pts[idx[i]].x, pts[idx[i]].y);
        mean += out[i];
    }
    mean /= static_cast<double>(m);

    double meanDist = 0.0;
    for (int i = 0; i < m; ++i) meanDist += (out[i] - mean).norm();
    meanDist /= static_cast<double>(m);
    const double scale = (meanDist > 1e-12) ? std::sqrt(2.0) / meanDist : 1.0;

    for (int i = 0; i < m; ++i) out[i] = (out[i] - mean) * scale;

    Eigen::Matrix3d t = Eigen::Matrix3d::Identity();
    t(0, 0) = scale;
    t(1, 1) = scale;
    t(0, 2) = -scale * mean.x();
    t(1, 2) = -scale * mean.y();
    return t;
}

// The null space of A is the eigenvector of A^T A with the smallest eigenvalue.
// One code path for any row count -- the minimal sample gives a wide 8x9 A, a
// refit gives a tall one -- which sidesteps the shape-dependent thin/full-V
// rules an SVD of A would impose. The price is a squared condition number; the
// Hartley normalization above is what pays it. SelfAdjointEigenSolver returns
// eigenvalues in increasing order, so column 0 is the vector we want.
static bool solveNullSpace9(const Eigen::MatrixXd& A,
                            Eigen::Matrix<double, 9, 1>& v) {
    const Eigen::Matrix<double, 9, 9> ata = A.transpose() * A;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es(ata);
    if (es.info() != Eigen::Success) return false;
    v = es.eigenvectors().col(0);
    return v.allFinite();
}

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

    // Distance from point to line. With a^2 + b^2 = 1 this is already the
    // perpendicular distance, so no linear algebra is needed.
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

static FitScore scoreLine(const Line2D& line, const std::vector<cv::Point2f>& pts,
                          double threshold, std::vector<bool>& mask) {
    FitScore s{0, 0.0};
    const int n = static_cast<int>(pts.size());
    for (int i = 0; i < n; ++i) {
        const double d = line.distance(pts[i]);
        const bool inlier = d < threshold;
        mask[i] = inlier;
        if (inlier) {
            ++s.inliers;
            s.errSum += d;
        }
    }
    return s;
}

// Total least squares line fit: minimize the sum of squared perpendicular
// distances. The normal is the direction of least variance, i.e. the
// eigenvector of the scatter matrix's smallest eigenvalue.
static bool fitLineTLS(const std::vector<cv::Point2f>& pts,
                       const int* idx, int m, Line2D& out) {
    if (m < 2) return false;

    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (int i = 0; i < m; ++i) mean += Eigen::Vector2d(pts[idx[i]].x, pts[idx[i]].y);
    mean /= static_cast<double>(m);

    Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
    for (int i = 0; i < m; ++i) {
        const Eigen::Vector2d d =
            Eigen::Vector2d(pts[idx[i]].x, pts[idx[i]].y) - mean;
        scatter += d * d.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(scatter);
    if (es.info() != Eigen::Success) return false;
    const Eigen::Vector2d normal = es.eigenvectors().col(0);
    if (!normal.allFinite() || normal.norm() < 1e-12) return false;

    out = Line2D(normal.x(), normal.y(), -normal.dot(mean));
    return true;
}

// Local optimization for the line model: refit on the current inlier set,
// re-score, keep only while it improves. Model and mask are adopted together,
// so the mask always belongs to the model.
static void loRefineLine(const std::vector<cv::Point2f>& pts, double threshold,
                         int rounds, Line2D& bestLine, FitScore& bestScore,
                         std::vector<bool>& bestMask,
                         std::vector<int>& idxScratch,
                         std::vector<bool>& maskScratch) {
    const int n = static_cast<int>(pts.size());
    for (int r = 0; r < rounds; ++r) {
        idxScratch.clear();
        for (int i = 0; i < n; ++i) {
            if (bestMask[i]) idxScratch.push_back(i);
        }
        if (static_cast<int>(idxScratch.size()) < 2) return;

        Line2D candidate;
        if (!fitLineTLS(pts, idxScratch.data(),
                        static_cast<int>(idxScratch.size()), candidate)) {
            return;
        }

        const FitScore s = scoreLine(candidate, pts, threshold, maskScratch);
        if (!s.betterThan(bestScore)) return;
        bestLine = candidate;
        bestScore = s;
        bestMask = maskScratch;
    }
}

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

    const int sampleSize = 2;
    const int iterFloor = std::min(kMinIterations, maxIterations);

    std::mt19937 rng(kRngSeed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    Line2D bestLine;
    FitScore bestScore;
    std::vector<bool> bestMask(n, false), currentMask(n, false);
    std::vector<int> idxScratch;

    int iteration = 0;
    int adaptiveN = maxIterations;   // what the formula asks for
    int iterBudget = maxIterations;  // what we actually run

    while (iteration < iterBudget) {
        // 1. Random sample: 2 distinct points (minimum for a line)
        int idx[2];
        drawDistinct(idx, sampleSize, dist, rng);
        ++iteration;

        // 2. Fit model (line through 2 points)
        const Line2D line = Line2D::fromTwoPoints(points[idx[0]], points[idx[1]]);

        // 3. Score it against every point
        const FitScore s = scoreLine(line, points, threshold, currentMask);
        if (!s.betterThan(bestScore)) continue;

        // 4. Adopt the new best, then let LO refit it on its own support
        bestLine = line;
        bestScore = s;
        bestMask = currentMask;
        loRefineLine(points, threshold, kLoSteps, bestLine, bestScore, bestMask,
                     idxScratch, currentMask);

        // 5. Re-plan the iteration budget from the improved inlier ratio
        adaptiveN = adaptiveIterations(bestScore.inliers, n, sampleSize,
                                       confidence, maxIterations);
        iterBudget = std::max(adaptiveN, iterFloor);
    }

    // Final polish: keep refitting while the fit still improves.
    loRefineLine(points, threshold, kPolishSteps, bestLine, bestScore, bestMask,
                 idxScratch, currentMask);

    std::cout << "[Line RANSAC] Iterations: " << iteration
              << " (adaptive N=" << adaptiveN << ", floor " << iterFloor << ")"
              << ", Inliers: " << bestScore.inliers << "/" << n << std::endl;

    if (bestScore.inliers < sampleSize) {
        return {Line2D(), std::vector<bool>(n, false)};
    }
    return {bestLine, bestMask};
}

// ============================================================================
// Part 2: Custom RANSAC for Homography
// ============================================================================

// Squared forward reprojection error |H*a - b|^2. Fixed-size Eigen products are
// unrolled and heap-free, which matters: this runs once per point per iteration.
// This is exactly what OpenCV's HomographyEstimatorCallback::computeError does,
// so H inlier counts are directly comparable between the two implementations.
static inline double forwardReprojSq(const Eigen::Matrix3d& h,
                                     const cv::Point2f& a,
                                     const cv::Point2f& b) {
    const Eigen::Vector3d p = h * Eigen::Vector3d(a.x, a.y, 1.0);
    if (std::abs(p.z()) < 1e-12) return std::numeric_limits<double>::max();
    return (p.head<2>() / p.z() - Eigen::Vector2d(b.x, b.y)).squaredNorm();
}

static FitScore scoreHomography(const Eigen::Matrix3d& h,
                                const std::vector<cv::Point2f>& src,
                                const std::vector<cv::Point2f>& dst,
                                double thresholdSq, std::vector<bool>& mask) {
    FitScore s{0, 0.0};
    const int n = static_cast<int>(src.size());
    for (int i = 0; i < n; ++i) {
        const double e = forwardReprojSq(h, src[i], dst[i]);
        const bool inlier = e < thresholdSq;
        mask[i] = inlier;
        if (inlier) {
            ++s.inliers;
            s.errSum += e;
        }
    }
    return s;
}

/**
 * DLT homography from m >= 4 correspondences, selected by index.
 *
 * For each correspondence (x, y) -> (x', y'):
 * [ -x  -y  -1   0   0   0   x*x'   y*x'   x' ] [ h1 ]   [ 0 ]
 * [  0   0   0  -x  -y  -1   x*y'   y*y'   y' ] [ h2 ] = [ 0 ]
 *                                                [...]
 *                                                [ h9 ]
 *
 * One routine for both the minimal sample and the LO/polish refits. A
 * 4-points-only solver is what forces an implementation to borrow someone
 * else's estimator for the refit step.
 */
static bool dltHomography(const std::vector<cv::Point2f>& src,
                          const std::vector<cv::Point2f>& dst,
                          const int* idx, int m,
                          Eigen::Matrix3d& h, SolverScratch& scratch) {
    if (m < 4) return false;

    const Eigen::Matrix3d t1 = hartleyNormalize(src, idx, m, scratch.a);
    const Eigen::Matrix3d t2 = hartleyNormalize(dst, idx, m, scratch.b);

    scratch.A.setZero(2 * m, 9);
    for (int i = 0; i < m; ++i) {
        const double x = scratch.a[i].x(), y = scratch.a[i].y();
        const double xp = scratch.b[i].x(), yp = scratch.b[i].y();
        scratch.A.row(2 * i) << -x, -y, -1.0, 0.0, 0.0, 0.0, x * xp, y * xp, xp;
        scratch.A.row(2 * i + 1) << 0.0, 0.0, 0.0, -x, -y, -1.0, x * yp, y * yp, yp;
    }

    Eigen::Matrix<double, 9, 1> hv;
    if (!solveNullSpace9(scratch.A, hv)) return false;

    Eigen::Matrix3d hn;
    hn << hv(0), hv(1), hv(2), hv(3), hv(4), hv(5), hv(6), hv(7), hv(8);

    // Undo normalization: x2 = T2^-1 * Hn * T1 * x1
    h = t2.inverse() * hn * t1;
    if (std::abs(h(2, 2)) < 1e-12) return false;
    h /= h(2, 2);
    return h.allFinite();
}

// A homography needs four points in general position. Reject the sample if any
// triple is collinear -- all four triples, in both images: a degeneracy in
// either one ruins the fit.
static bool hasCollinearTriple(const Eigen::Vector2d* p) {
    static constexpr int kTriples[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
    for (const auto& t : kTriples) {
        const Eigen::Vector2d u = p[t[1]] - p[t[0]];
        const Eigen::Vector2d v = p[t[2]] - p[t[0]];
        if (std::abs(u.x() * v.y() - u.y() * v.x()) < 1e-6) return true;
    }
    return false;
}

// Local optimization for the homography: refit on the current inlier set,
// re-score, keep only while it improves. Used twice -- as LO-RANSAC's inner
// step on each new best model, and as the final polish -- differing only in
// round count. Model and mask are adopted together, so the mask handed back to
// the caller is the one the returned model actually produces.
static void loRefineHomography(const std::vector<cv::Point2f>& src,
                               const std::vector<cv::Point2f>& dst,
                               double thresholdSq, int rounds,
                               Eigen::Matrix3d& bestH, FitScore& bestScore,
                               std::vector<bool>& bestMask,
                               std::vector<int>& idxScratch,
                               std::vector<bool>& maskScratch,
                               SolverScratch& scratch) {
    const int n = static_cast<int>(src.size());
    for (int r = 0; r < rounds; ++r) {
        idxScratch.clear();
        for (int i = 0; i < n; ++i) {
            if (bestMask[i]) idxScratch.push_back(i);
        }
        if (static_cast<int>(idxScratch.size()) < 4) return;

        Eigen::Matrix3d candidate;
        if (!dltHomography(src, dst, idxScratch.data(),
                           static_cast<int>(idxScratch.size()), candidate, scratch)) {
            return;
        }

        const FitScore s = scoreHomography(candidate, src, dst, thresholdSq, maskScratch);
        if (!s.betterThan(bestScore)) return;
        bestH = candidate;
        bestScore = s;
        bestMask = maskScratch;
    }
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
    const int iterFloor = std::min(kMinIterations, maxIterations);

    std::mt19937 rng(kRngSeed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    Eigen::Matrix3d bestH = Eigen::Matrix3d::Identity();
    FitScore bestScore;
    std::vector<bool> bestMask(n, false), currentMask(n, false);
    std::vector<int> idxScratch;
    SolverScratch scratch;
    Eigen::Vector2d sampleSrc[4], sampleDst[4];

    int iteration = 0;
    int adaptiveN = maxIterations;
    int iterBudget = maxIterations;

    while (iteration < iterBudget) {
        // 1. Random sample: 4 distinct indices
        int idx[4];
        drawDistinct(idx, sampleSize, dist, rng);
        ++iteration;

        for (int i = 0; i < sampleSize; ++i) {
            sampleSrc[i] = Eigen::Vector2d(srcPoints[idx[i]].x, srcPoints[idx[i]].y);
            sampleDst[i] = Eigen::Vector2d(dstPoints[idx[i]].x, dstPoints[idx[i]].y);
        }
        if (hasCollinearTriple(sampleSrc) || hasCollinearTriple(sampleDst)) continue;

        // 2. Fit model (homography from 4 points, Hartley-normalized DLT)
        Eigen::Matrix3d h;
        if (!dltHomography(srcPoints, dstPoints, idx, sampleSize, h, scratch)) continue;

        // 3. Score it against every correspondence
        const FitScore s = scoreHomography(h, srcPoints, dstPoints, thresholdSq, currentMask);
        if (!s.betterThan(bestScore)) continue;

        // 4. Adopt the new best, then let LO refit it on its own support
        bestH = h;
        bestScore = s;
        bestMask = currentMask;
        loRefineHomography(srcPoints, dstPoints, thresholdSq, kLoSteps, bestH,
                           bestScore, bestMask, idxScratch, currentMask, scratch);

        // 5. Re-plan the iteration budget from the improved inlier ratio
        adaptiveN = adaptiveIterations(bestScore.inliers, n, sampleSize,
                                       confidence, maxIterations);
        iterBudget = std::max(adaptiveN, iterFloor);
    }

    // Final polish over the whole inlier set.
    loRefineHomography(srcPoints, dstPoints, thresholdSq, kPolishSteps, bestH,
                       bestScore, bestMask, idxScratch, currentMask, scratch);

    std::cout << "[Homography RANSAC] Iterations: " << iteration
              << " (adaptive N=" << adaptiveN << ", floor " << iterFloor << ")"
              << ", Inliers: " << bestScore.inliers << "/" << n << std::endl;

    // No usable model: hand back an empty mask too, so no caller can report
    // inliers for a model that was never returned.
    if (bestScore.inliers < sampleSize) {
        return {cv::Mat(), std::vector<bool>(n, false)};
    }
    return {toCvMat(bestH), bestMask};
}

// ============================================================================
// Part 3: Custom RANSAC for Fundamental Matrix
// ============================================================================

// The two residuals share d = x2^T F x1 and the epipolar-line gradients
//   A = |(F^T x2)_xy|^2   (line in image 1)
//   B = |(F x1)_xy|^2     (line in image 2)
// and differ only in how they combine them.
static inline void epipolarTerms(const Eigen::Matrix3d& f,
                                 const cv::Point2f& a, const cv::Point2f& b,
                                 double& d, double& termA, double& termB) {
    const Eigen::Vector3d x1(a.x, a.y, 1.0);
    const Eigen::Vector3d x2(b.x, b.y, 1.0);
    const Eigen::Vector3d l2 = f * x1;              // epipolar line in image 2
    const Eigen::Vector3d l1 = f.transpose() * x2;  // epipolar line in image 1
    d = x2.dot(l2);
    termA = l1.head<2>().squaredNorm();
    termB = l2.head<2>().squaredNorm();
}

// Squared Sampson distance, d^2/(A+B) -- the first-order approximation of the
// geometric (reprojection) error for the bilinear constraint x2^T F x1 = 0.
// This is what we estimate and threshold on, and what OpenCV's USAC variants
// use (usac/estimator.cpp, SampsonErrorImpl).
static inline double sampsonSq(const Eigen::Matrix3d& f,
                               const cv::Point2f& a, const cv::Point2f& b) {
    double d, termA, termB;
    epipolarTerms(f, a, b, d, termA, termB);
    const double denom = termA + termB;
    if (denom < 1e-10) return std::numeric_limits<double>::max();
    return (d * d) / denom;
}

// Max-form symmetric epipolar distance, max(d^2/A, d^2/B) = d^2/min(A,B).
// This is the rule cv::findFundamentalMat's classic path actually uses
// (fundam.cpp, FMEstimatorCallback::computeError). Reported alongside Sampson
// so the demo shows what the choice of rule costs: min(A,B) <= (A+B)/2, so this
// residual is always at least twice Sampson's and the rule is strictly tighter
// at the same nominal threshold.
static inline double maxEpipolarSq(const Eigen::Matrix3d& f,
                                   const cv::Point2f& a, const cv::Point2f& b) {
    double d, termA, termB;
    epipolarTerms(f, a, b, d, termA, termB);
    const double smaller = std::min(termA, termB);
    if (smaller < 1e-10) return std::numeric_limits<double>::max();
    return (d * d) / smaller;
}

static FitScore scoreFundamental(const Eigen::Matrix3d& f,
                                 const std::vector<cv::Point2f>& pts1,
                                 const std::vector<cv::Point2f>& pts2,
                                 double thresholdSq, std::vector<bool>& mask) {
    FitScore s{0, 0.0};
    const int n = static_cast<int>(pts1.size());
    for (int i = 0; i < n; ++i) {
        const double e = sampsonSq(f, pts1[i], pts2[i]);
        const bool inlier = e < thresholdSq;
        mask[i] = inlier;
        if (inlier) {
            ++s.inliers;
            s.errSum += e;
        }
    }
    return s;
}

/**
 * Normalized 8-point algorithm (works for m >= 8 points, selected by index).
 *
 * 1. Hartley normalization: translate points to centroid, scale so the mean
 *    distance from origin is sqrt(2).
 * 2. Solve the mx9 homogeneous system A f = 0 for its null space.
 * 3. Enforce rank-2 by zeroing the smallest singular value of F.
 * 4. Denormalize: F = T2^T * F_norm * T1.
 */
static bool eightPointFundamental(const std::vector<cv::Point2f>& pts1,
                                  const std::vector<cv::Point2f>& pts2,
                                  const int* idx, int m,
                                  Eigen::Matrix3d& f, SolverScratch& scratch) {
    if (m < 8) return false;

    const Eigen::Matrix3d t1 = hartleyNormalize(pts1, idx, m, scratch.a);
    const Eigen::Matrix3d t2 = hartleyNormalize(pts2, idx, m, scratch.b);

    scratch.A.resize(m, 9);
    for (int i = 0; i < m; ++i) {
        const double x1 = scratch.a[i].x(), y1 = scratch.a[i].y();
        const double x2 = scratch.b[i].x(), y2 = scratch.b[i].y();
        scratch.A.row(i) << x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0;
    }

    Eigen::Matrix<double, 9, 1> fv;
    if (!solveNullSpace9(scratch.A, fv)) return false;

    Eigen::Matrix3d fn;
    fn << fv(0), fv(1), fv(2), fv(3), fv(4), fv(5), fv(6), fv(7), fv(8);

    // Enforce rank 2 by zeroing the smallest singular value. This one is a
    // genuine SVD: 3x3 and square, so none of the shape caveats noted in
    // solveNullSpace9 apply.
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(fn, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d sv = svd.singularValues();
    sv(2) = 0.0;
    fn = svd.matrixU() * sv.asDiagonal() * svd.matrixV().transpose();

    // Undo normalization: x2^T F x1 = 0 with F = T2^T * Fn * T1
    f = t2.transpose() * fn * t1;
    const double norm = f.norm();
    if (norm < 1e-12) return false;
    f /= norm;
    return f.allFinite();
}

// Local optimization for F -- same refit/re-score contract as
// loRefineHomography.
static void loRefineFundamental(const std::vector<cv::Point2f>& pts1,
                                const std::vector<cv::Point2f>& pts2,
                                double thresholdSq, int rounds,
                                Eigen::Matrix3d& bestF, FitScore& bestScore,
                                std::vector<bool>& bestMask,
                                std::vector<int>& idxScratch,
                                std::vector<bool>& maskScratch,
                                SolverScratch& scratch) {
    const int n = static_cast<int>(pts1.size());
    for (int r = 0; r < rounds; ++r) {
        idxScratch.clear();
        for (int i = 0; i < n; ++i) {
            if (bestMask[i]) idxScratch.push_back(i);
        }
        if (static_cast<int>(idxScratch.size()) < 8) return;

        Eigen::Matrix3d candidate;
        if (!eightPointFundamental(pts1, pts2, idxScratch.data(),
                                   static_cast<int>(idxScratch.size()), candidate,
                                   scratch)) {
            return;
        }

        const FitScore s = scoreFundamental(candidate, pts1, pts2, thresholdSq, maskScratch);
        if (!s.betterThan(bestScore)) return;
        bestF = candidate;
        bestScore = s;
        bestMask = maskScratch;
    }
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
    const int iterFloor = std::min(kMinIterations, maxIterations);

    std::mt19937 rng(kRngSeed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    Eigen::Matrix3d bestF = Eigen::Matrix3d::Identity();
    FitScore bestScore;
    std::vector<bool> bestMask(n, false), currentMask(n, false);
    std::vector<int> idxScratch;
    SolverScratch scratch;

    int iteration = 0;
    int adaptiveN = maxIterations;
    int iterBudget = maxIterations;

    while (iteration < iterBudget) {
        // 1. Random sample: 8 distinct indices
        int idx[8];
        drawDistinct(idx, sampleSize, dist, rng);
        ++iteration;

        // 2. Fit model (normalized 8-point algorithm)
        Eigen::Matrix3d f;
        if (!eightPointFundamental(pts1, pts2, idx, sampleSize, f, scratch)) continue;

        // 3. Score by Sampson distance
        const FitScore s = scoreFundamental(f, pts1, pts2, thresholdSq, currentMask);
        if (!s.betterThan(bestScore)) continue;

        // 4. Adopt the new best, then let LO refit it on its own support
        bestF = f;
        bestScore = s;
        bestMask = currentMask;
        loRefineFundamental(pts1, pts2, thresholdSq, kLoSteps, bestF, bestScore,
                            bestMask, idxScratch, currentMask, scratch);

        // 5. Re-plan the iteration budget from the improved inlier ratio
        adaptiveN = adaptiveIterations(bestScore.inliers, n, sampleSize,
                                       confidence, maxIterations);
        iterBudget = std::max(adaptiveN, iterFloor);
    }

    // Final polish over the whole inlier set.
    loRefineFundamental(pts1, pts2, thresholdSq, kPolishSteps, bestF, bestScore,
                        bestMask, idxScratch, currentMask, scratch);

    std::cout << "[Fundamental RANSAC] Iterations: " << iteration
              << " (adaptive N=" << adaptiveN << ", floor " << iterFloor << ")"
              << ", Inliers: " << bestScore.inliers << "/" << n << std::endl;

    if (bestScore.inliers < sampleSize) {
        return {cv::Mat(), std::vector<bool>(n, false)};
    }
    return {toCvMat(bestF), bestMask};
}

// ============================================================================
// Inlier-rule comparison for F
// ============================================================================

struct RuleStats {
    int inliers = 0;
    double meanErr = -1.0;
};

// Count inliers and mean residual of a fixed F under one residual rule.
// `inclusive` mirrors OpenCV's ptsetreg.cpp findInliers, which tests
// `err <= thresh*thresh`; our own estimation loop uses a strict `<`. The flag
// keeps the replication exact rather than off by a boundary point.
template <typename Residual>
static RuleStats statsUnderRule(const Eigen::Matrix3d& f,
                                const std::vector<cv::Point2f>& pts1,
                                const std::vector<cv::Point2f>& pts2,
                                double thresholdSq, bool inclusive,
                                Residual residual) {
    RuleStats st;
    double sum = 0.0;
    const int n = static_cast<int>(pts1.size());
    for (int i = 0; i < n; ++i) {
        const double e = residual(f, pts1[i], pts2[i]);
        const bool inlier = inclusive ? (e <= thresholdSq) : (e < thresholdSq);
        if (inlier) {
            ++st.inliers;
            sum += e;
        }
    }
    if (st.inliers > 0) st.meanErr = sum / st.inliers;
    return st;
}

// Print, for each F model, the inlier count and mean residual under both rules.
// The point of the table: the two columns differ for the SAME model at the SAME
// threshold, so any count comparison across a rule boundary is meaningless.
static void printFRuleComparison(const cv::Mat& customF, const cv::Mat& opencvF,
                                 const std::vector<cv::Point2f>& pts1,
                                 const std::vector<cv::Point2f>& pts2,
                                 double threshold, int opencvReportedInliers) {
    const double thresholdSq = threshold * threshold;
    Eigen::Matrix3d cf, of;
    const bool haveCustom = toEigen(customF, cf);
    const bool haveOpenCV = toEigen(opencvF, of);

    std::cout << "\n--- F inlier rule: same models, same " << threshold
              << " px threshold, two rules ---\n";
    std::cout << std::left << std::setw(22) << "Model"
              << std::right << std::setw(10) << "Sampson"
              << std::setw(11) << "mean"
              << std::setw(12) << "max-form"
              << std::setw(11) << "mean" << "\n";
    std::cout << std::left << std::setw(22) << ""
              << std::right << std::setw(10) << "d2/(A+B)"
              << std::setw(11) << ""
              << std::setw(12) << "d2/min(A,B)"
              << std::setw(11) << "" << "\n";
    std::cout << std::string(66, '-') << "\n";

    auto row = [&](const char* name, bool have, const Eigen::Matrix3d& f) {
        std::cout << std::left << std::setw(22) << name;
        if (!have) {
            std::cout << std::right << std::setw(10) << "-" << std::setw(11) << "-"
                      << std::setw(12) << "-" << std::setw(11) << "-" << "\n";
            return;
        }
        const RuleStats s = statsUnderRule(f, pts1, pts2, thresholdSq, true, sampsonSq);
        const RuleStats m = statsUnderRule(f, pts1, pts2, thresholdSq, true, maxEpipolarSq);
        std::cout << std::right << std::setw(10) << s.inliers
                  << std::setw(11) << s.meanErr
                  << std::setw(12) << m.inliers
                  << std::setw(11) << m.meanErr << "\n";
    };
    row("custom F (8pt)", haveCustom, cf);
    row("OpenCV FM_RANSAC", haveOpenCV, of);

    if (haveOpenCV) {
        const RuleStats m = statsUnderRule(of, pts1, pts2, thresholdSq, true, maxEpipolarSq);
        std::cout << "\nCheck: re-scoring OpenCV's own F under the max-form rule gives "
                  << m.inliers << "; cv::findFundamentalMat reported "
                  << opencvReportedInliers << ".\n";
    }
    std::cout << "min(A,B) <= (A+B)/2, so max-form >= 2x Sampson always: OpenCV's\n"
                 "classic F rule is the stricter one. Comparing an inlier count\n"
                 "scored by Sampson against one scored by max-form measures the rule,\n"
                 "not the estimator. (H has no such problem -- both sides use forward\n"
                 "reprojection error, so the H counts are directly comparable.)\n";
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
    // Both sides score with forward reprojection error, so these counts are
    // directly comparable.
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
    const int cvFInliers = cvFMask.empty() ? 0 : cv::countNonZero(cvFMask);
    std::cout << "Custom RANSAC : Sampson " << customFSampson << ", inliers "
              << cv::countNonZero(customFMask) << "/" << pts1.size()
              << ", time " << customFTime << " ms  (own rule: Sampson)\n";
    std::cout << "OpenCV RANSAC : Sampson " << cvFSampson << ", inliers "
              << cvFInliers << "/" << pts1.size()
              << ", time " << cvFTime << " ms  (own rule: max-form)\n";

    // The two estimators above selected their inliers with different rules, so
    // the counts on those lines are not like-for-like. Re-score both models
    // under both rules to separate the rule's effect from the estimator's.
    printFRuleComparison(customF, cvF, pts1, pts2, threshold, cvFInliers);

    std::cout << "\n=== Summary (same data; H rules match, F rules differ) ===" << std::endl;
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
              << std::setw(12) << cvFInliers
              << std::setw(12) << cvFTime << "\n";
    std::cout << "Note: the two F rows selected inliers under different rules "
                 "(see the table above).\n";

    std::cout << "\n=== Key Takeaways ===" << std::endl;
    std::cout << "1. RANSAC requires minimum sample size (2 line, 4 H, 8 F)" << std::endl;
    std::cout << "2. Adaptive iteration count saves computation -- but needs a floor"
              << std::endl;
    std::cout << "3. Local optimization (refit on inliers, re-score) is the biggest win"
              << std::endl;
    std::cout << "4. Normalize inside the minimal solver, not just the refit" << std::endl;
    std::cout << "5. Report the mask the returned model actually produces" << std::endl;
    std::cout << "6. Check what the baseline's inlier rule really is before "
                 "comparing counts" << std::endl;

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
