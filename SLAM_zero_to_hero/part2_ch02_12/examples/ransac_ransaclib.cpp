/**
 * Line / Homography / Fundamental Matrix Estimation using RansacLib
 *
 * This example demonstrates the template-based RANSAC design pattern using
 * RansacLib. RansacLib provides a clean separation between:
 *
 * 1. Solver: Encapsulates the minimal solver and residual computation
 * 2. Sampler: Generates random samples (uniform, PROSAC, etc.)
 * 3. Estimator: RANSAC variants (MSAC, LO-MSAC, etc.)
 *
 * To prove the point, THREE solvers plug into the same LO-MSAC estimator:
 *  - Line2DSolver           (2-point minimal, PCA refinement)
 *  - HomographySolver       (4-point DLT, squared reprojection residual)
 *  - FundamentalMatrixSolver (normalized 8-point, Sampson residual)
 *
 * H and F run on the real EuRoC ORB correspondences shared by every demo in
 * this chapter, and each is benchmarked against OpenCV RANSAC on the same
 * data with the same threshold and confidence. Line fitting uses the shared
 * fixed-seed synthetic points (OpenCV has no line RANSAC).
 *
 * Reference: https://github.com/tsattler/RansacLib
 */

#include <RansacLib/ransac.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ransac_data.h"

// ============================================================================
// Line Solver (implements RansacLib Solver interface)
// ============================================================================

// Model: (a, b, c) with a^2 + b^2 = 1 for the line ax + by + c = 0.
using Line2DModel = Eigen::Vector3d;
using Line2DModelVector = std::vector<Line2DModel, Eigen::aligned_allocator<Line2DModel>>;

class Line2DSolver {
public:
    explicit Line2DSolver(const Eigen::Matrix<double, Eigen::Dynamic, 2>& data)
        : data_(data) {}

    inline int min_sample_size() const { return 2; }
    inline int non_minimal_sample_size() const { return 6; }
    inline int num_data() const { return static_cast<int>(data_.rows()); }

    int MinimalSolver(const std::vector<int>& sample, Line2DModelVector* models) const {
        if (sample.size() < 2) return 0;
        Eigen::Vector2d p1 = data_.row(sample[0]);
        Eigen::Vector2d p2 = data_.row(sample[1]);
        Eigen::Vector2d dir = p2 - p1;
        double norm = dir.norm();
        if (norm < 1e-12) return 0;
        Line2DModel line(-dir.y() / norm, dir.x() / norm, 0.0);
        line(2) = -(line(0) * p1.x() + line(1) * p1.y());
        models->push_back(line);
        return 1;
    }

    // PCA fit: line through the centroid along the dominant direction.
    int NonMinimalSolver(const std::vector<int>& sample, Line2DModel* model) const {
        if (sample.size() < 2) return 0;
        Eigen::Vector2d mean = Eigen::Vector2d::Zero();
        for (int idx : sample) mean += data_.row(idx).transpose();
        mean /= static_cast<double>(sample.size());

        Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
        for (int idx : sample) {
            Eigen::Vector2d d = data_.row(idx).transpose() - mean;
            cov += d * d.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov);
        Eigen::Vector2d normal = eig.eigenvectors().col(0);  // smallest eigenvalue
        (*model)(0) = normal.x();
        (*model)(1) = normal.y();
        (*model)(2) = -normal.dot(mean);
        return 1;
    }

    // Squared point-line distance.
    double EvaluateModelOnPoint(const Line2DModel& line, int i) const {
        double d = line(0) * data_(i, 0) + line(1) * data_(i, 1) + line(2);
        return d * d;
    }

    void LeastSquares(const std::vector<int>& sample, Line2DModel* model) const {
        NonMinimalSolver(sample, model);
    }

private:
    Eigen::Matrix<double, Eigen::Dynamic, 2> data_;
};

// ============================================================================
// Homography Solver (implements RansacLib Solver interface)
// ============================================================================

using Homography = Eigen::Matrix3d;
using HomographyVector = std::vector<Homography, Eigen::aligned_allocator<Homography>>;

class HomographySolver {
public:
    explicit HomographySolver(const Eigen::Matrix<double, Eigen::Dynamic, 4>& data)
        : data_(data) {}

    inline int min_sample_size() const { return 4; }
    inline int non_minimal_sample_size() const { return 12; }
    inline int num_data() const { return static_cast<int>(data_.rows()); }

    int MinimalSolver(const std::vector<int>& sample, HomographyVector* models) const {
        Homography H;
        if (!dlt(sample, &H)) return 0;
        models->push_back(H);
        return 1;
    }

    int NonMinimalSolver(const std::vector<int>& sample, Homography* model) const {
        return dlt(sample, model) ? 1 : 0;
    }

    // Squared forward reprojection error |H*x1 - x2|^2 -- the same inlier
    // rule cv::findHomography(RANSAC) uses.
    double EvaluateModelOnPoint(const Homography& H, int i) const {
        Eigen::Vector3d p(data_(i, 0), data_(i, 1), 1.0);
        Eigen::Vector3d q = H * p;
        if (std::abs(q.z()) < 1e-12) return std::numeric_limits<double>::max();
        double dx = q.x() / q.z() - data_(i, 2);
        double dy = q.y() / q.z() - data_(i, 3);
        return dx * dx + dy * dy;
    }

    void LeastSquares(const std::vector<int>& sample, Homography* model) const {
        NonMinimalSolver(sample, model);
    }

private:
    // Normalized DLT for n >= 4 correspondences. Solves the 2n x 9 system
    // via 9x9 normal equations (fast inside local optimization).
    bool dlt(const std::vector<int>& sample, Homography* H) const {
        const int n = static_cast<int>(sample.size());
        if (n < 4) return false;

        Eigen::Vector2d mean1 = Eigen::Vector2d::Zero(), mean2 = Eigen::Vector2d::Zero();
        for (int idx : sample) {
            mean1 += data_.block<1, 2>(idx, 0).transpose();
            mean2 += data_.block<1, 2>(idx, 2).transpose();
        }
        mean1 /= n; mean2 /= n;
        double scale1 = 0.0, scale2 = 0.0;
        for (int idx : sample) {
            scale1 += (data_.block<1, 2>(idx, 0).transpose() - mean1).norm();
            scale2 += (data_.block<1, 2>(idx, 2).transpose() - mean2).norm();
        }
        if (scale1 < 1e-12 || scale2 < 1e-12) return false;
        scale1 = n * std::sqrt(2.0) / scale1;
        scale2 = n * std::sqrt(2.0) / scale2;

        Eigen::Matrix<double, 9, 9> AtA = Eigen::Matrix<double, 9, 9>::Zero();
        Eigen::Matrix<double, 9, 1> row1, row2;
        for (int idx : sample) {
            double x = scale1 * (data_(idx, 0) - mean1.x());
            double y = scale1 * (data_(idx, 1) - mean1.y());
            double xp = scale2 * (data_(idx, 2) - mean2.x());
            double yp = scale2 * (data_(idx, 3) - mean2.y());
            row1 << -x, -y, -1, 0, 0, 0, x * xp, y * xp, xp;
            row2 << 0, 0, 0, -x, -y, -1, x * yp, y * yp, yp;
            AtA += row1 * row1.transpose() + row2 * row2.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eig(AtA);
        Eigen::Matrix<double, 9, 1> h = eig.eigenvectors().col(0);

        Homography Hn;
        Hn << h(0), h(1), h(2),
              h(3), h(4), h(5),
              h(6), h(7), h(8);

        Eigen::Matrix3d T1 = Eigen::Matrix3d::Identity(), T2 = Eigen::Matrix3d::Identity();
        T1(0, 0) = scale1; T1(1, 1) = scale1;
        T1(0, 2) = -scale1 * mean1.x(); T1(1, 2) = -scale1 * mean1.y();
        T2(0, 0) = scale2; T2(1, 1) = scale2;
        T2(0, 2) = -scale2 * mean2.x(); T2(1, 2) = -scale2 * mean2.y();

        *H = T2.inverse() * Hn * T1;
        if (std::abs((*H)(2, 2)) > 1e-12) *H /= (*H)(2, 2);
        return true;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 4> data_;
};

// ============================================================================
// Fundamental Matrix Solver (implements RansacLib Solver interface)
// ============================================================================

using FundamentalMatrix = Eigen::Matrix3d;
using FundamentalMatrixVector = std::vector<FundamentalMatrix,
    Eigen::aligned_allocator<FundamentalMatrix>>;

class FundamentalMatrixSolver {
public:
    explicit FundamentalMatrixSolver(const Eigen::Matrix<double, Eigen::Dynamic, 4>& data)
        : data_(data) {}

    inline int min_sample_size() const { return 8; }
    inline int non_minimal_sample_size() const { return 12; }
    inline int num_data() const { return static_cast<int>(data_.rows()); }

    int MinimalSolver(const std::vector<int>& sample,
                      FundamentalMatrixVector* models) const {
        FundamentalMatrix F;
        if (!eightPoint(sample, &F)) return 0;
        models->push_back(F);
        return 1;
    }

    int NonMinimalSolver(const std::vector<int>& sample, FundamentalMatrix* model) const {
        return eightPoint(sample, model) ? 1 : 0;
    }

    // Squared Sampson distance.
    double EvaluateModelOnPoint(const FundamentalMatrix& F, int i) const {
        Eigen::Vector3d p1(data_(i, 0), data_(i, 1), 1.0);
        Eigen::Vector3d p2(data_(i, 2), data_(i, 3), 1.0);
        Eigen::Vector3d Fp1 = F * p1;
        Eigen::Vector3d Ftp2 = F.transpose() * p2;
        double num = p2.dot(Fp1);
        double denom = Fp1(0) * Fp1(0) + Fp1(1) * Fp1(1)
                     + Ftp2(0) * Ftp2(0) + Ftp2(1) * Ftp2(1);
        if (denom < 1e-10) return std::numeric_limits<double>::max();
        return (num * num) / denom;
    }

    void LeastSquares(const std::vector<int>& sample, FundamentalMatrix* model) const {
        NonMinimalSolver(sample, model);
    }

private:
    // Normalized 8-point algorithm for n >= 8 correspondences. Solves the
    // n x 9 system via 9x9 normal equations, then enforces rank-2.
    bool eightPoint(const std::vector<int>& sample, FundamentalMatrix* F) const {
        const int n = static_cast<int>(sample.size());
        if (n < 8) return false;

        Eigen::Vector2d mean1 = Eigen::Vector2d::Zero(), mean2 = Eigen::Vector2d::Zero();
        for (int idx : sample) {
            mean1 += data_.block<1, 2>(idx, 0).transpose();
            mean2 += data_.block<1, 2>(idx, 2).transpose();
        }
        mean1 /= n; mean2 /= n;
        double scale1 = 0.0, scale2 = 0.0;
        for (int idx : sample) {
            scale1 += (data_.block<1, 2>(idx, 0).transpose() - mean1).norm();
            scale2 += (data_.block<1, 2>(idx, 2).transpose() - mean2).norm();
        }
        if (scale1 < 1e-12 || scale2 < 1e-12) return false;
        scale1 = n * std::sqrt(2.0) / scale1;
        scale2 = n * std::sqrt(2.0) / scale2;

        Eigen::Matrix<double, 9, 9> AtA = Eigen::Matrix<double, 9, 9>::Zero();
        Eigen::Matrix<double, 9, 1> row;
        for (int idx : sample) {
            double x1 = scale1 * (data_(idx, 0) - mean1.x());
            double y1 = scale1 * (data_(idx, 1) - mean1.y());
            double x2 = scale2 * (data_(idx, 2) - mean2.x());
            double y2 = scale2 * (data_(idx, 3) - mean2.y());
            row << x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0;
            AtA += row * row.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eig(AtA);
        Eigen::Matrix<double, 9, 1> f = eig.eigenvectors().col(0);

        Eigen::Matrix3d Fn;
        Fn << f(0), f(1), f(2),
              f(3), f(4), f(5),
              f(6), f(7), f(8);

        // Enforce rank-2 constraint
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(Fn, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d s = svd.singularValues();
        s(2) = 0.0;
        Fn = svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();

        Eigen::Matrix3d T1 = Eigen::Matrix3d::Identity(), T2 = Eigen::Matrix3d::Identity();
        T1(0, 0) = scale1; T1(1, 1) = scale1;
        T1(0, 2) = -scale1 * mean1.x(); T1(1, 2) = -scale1 * mean1.y();
        T2(0, 0) = scale2; T2(1, 1) = scale2;
        T2(0, 2) = -scale2 * mean2.x(); T2(1, 2) = -scale2 * mean2.y();

        *F = T2.transpose() * Fn * T1;
        double norm = F->norm();
        if (norm > 1e-12) *F /= norm;
        return true;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 4> data_;
};

// ============================================================================
// Helpers
// ============================================================================

// Realistic LO-MSAC options matched to the OpenCV baselines: same squared
// threshold, same 0.99 confidence, adaptive stopping, light local optim.
static ransac_lib::LORansacOptions makeOptions(double squaredThreshold) {
    ransac_lib::LORansacOptions options;
    options.min_num_iterations_ = 10;
    options.max_num_iterations_ = 10000;
    options.squared_inlier_threshold_ = squaredThreshold;
    options.success_probability_ = 0.99;
    options.min_sample_multiplicator_ = 7;
    options.num_lsq_iterations_ = 2;
    options.num_lo_steps_ = 2;
    options.random_seed_ = 42;
    return options;
}

static cv::Mat indicesToMask(const std::vector<int>& indices, int n) {
    cv::Mat mask = cv::Mat::zeros(n, 1, CV_8U);
    for (int idx : indices) mask.at<uchar>(idx) = 1;
    return mask;
}

static cv::Mat toCvMat3x3(const Eigen::Matrix3d& m) {
    cv::Mat out(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out.at<double>(r, c) = m(r, c);
    return out;
}

struct RowResult {
    std::string name;
    double err;
    int inliers;
    double time_ms;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "=== Line / H / F Estimation using RansacLib ===" << std::endl;
    std::cout << "Demonstrating the template-based RANSAC design pattern\n" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    const double threshold = 3.0;
    const double squaredThreshold = threshold * threshold;
    Timer timer;
    std::vector<RowResult> rows;

    // ---------- 1. Line fitting (shared fixed-seed synthetic points) ----------
    std::cout << "--- Line2DSolver: LO-MSAC (synthetic, 70/100 inliers) ---" << std::endl;
    std::vector<cv::Point2f> linePts = generateLinePoints();
    Eigen::Matrix<double, Eigen::Dynamic, 2> lineData(linePts.size(), 2);
    for (size_t i = 0; i < linePts.size(); ++i) {
        lineData(i, 0) = linePts[i].x;
        lineData(i, 1) = linePts[i].y;
    }
    {
        const double lineThreshold = 5.0;  // matches ransac_custom's line test
        Line2DSolver solver(lineData);
        ransac_lib::LocallyOptimizedMSAC<Line2DModel, Line2DModelVector, Line2DSolver> lomsac;
        ransac_lib::RansacStatistics stats;
        Line2DModel line;
        timer.start();
        int inliers = lomsac.EstimateModel(makeOptions(lineThreshold * lineThreshold),
                                           solver, &line, &stats);
        double ms = timer.elapsedMs();
        std::cout << "  Line: " << line(0) << "x + " << line(1) << "y + " << line(2)
                  << " = 0  (GT 0.4472x - 0.8944y + 89.44 = 0)\n";
        std::cout << "  Inliers: " << inliers << "/" << linePts.size()
                  << ", Iterations: " << stats.num_iterations
                  << ", Time: " << ms << " ms\n";
        std::cout << "  (OpenCV has no line RANSAC; cv::fitLine is a robust M-estimator,"
                     " not hypothesize-and-verify)\n\n";
        rows.push_back({"Line RansacLib (LO-MSAC)", -1.0, inliers, ms});
    }

    // ---------- Real correspondences shared with every other demo ----------
    cv::Mat img1, img2;
    std::vector<cv::Point2f> pts1, pts2;
    if (!loadRealPair(argc, argv, img1, img2, pts1, pts2, 8)) return 1;
    const int n = static_cast<int>(pts1.size());

    Eigen::Matrix<double, Eigen::Dynamic, 4> corr(n, 4);
    for (int i = 0; i < n; ++i) {
        corr(i, 0) = pts1[i].x; corr(i, 1) = pts1[i].y;
        corr(i, 2) = pts2[i].x; corr(i, 3) = pts2[i].y;
    }

    // ---------- 2. Homography: RansacLib vs OpenCV RANSAC ----------
    std::cout << "\n--- HomographySolver: LO-MSAC vs cv::findHomography(RANSAC) ---" << std::endl;
    cv::Mat ransaclibHMask;
    {
        HomographySolver solver(corr);
        ransac_lib::LocallyOptimizedMSAC<Homography, HomographyVector, HomographySolver> lomsac;
        ransac_lib::RansacStatistics stats;
        Homography H;
        timer.start();
        int inliers = lomsac.EstimateModel(makeOptions(squaredThreshold), solver, &H, &stats);
        double ms = timer.elapsedMs();
        ransaclibHMask = indicesToMask(stats.inlier_indices, n);
        double err = meanInlierReproj(pts1, pts2, toCvMat3x3(H), ransaclibHMask);
        std::cout << "  RansacLib : reproj " << err << " px, inliers " << inliers << "/" << n
                  << ", iterations " << stats.num_iterations << ", time " << ms << " ms\n";
        rows.push_back({"H RansacLib (LO-MSAC)", err, inliers, ms});
    }
    {
        cv::Mat mask;
        timer.start();
        cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, threshold, mask, 2000, 0.99);
        double ms = timer.elapsedMs();
        double err = meanInlierReproj(pts1, pts2, H, mask);
        std::cout << "  OpenCV    : reproj " << err << " px, inliers "
                  << cv::countNonZero(mask) << "/" << n << ", time " << ms << " ms\n";
        rows.push_back({"H OpenCV RANSAC", err, cv::countNonZero(mask), ms});
    }

    // ---------- 3. Fundamental matrix: RansacLib vs OpenCV RANSAC ----------
    std::cout << "\n--- FundamentalMatrixSolver: LO-MSAC vs cv::findFundamentalMat(FM_RANSAC) ---"
              << std::endl;
    std::cout << std::setprecision(6);
    {
        FundamentalMatrixSolver solver(corr);
        ransac_lib::LocallyOptimizedMSAC<FundamentalMatrix, FundamentalMatrixVector,
                                         FundamentalMatrixSolver> lomsac;
        ransac_lib::RansacStatistics stats;
        FundamentalMatrix F;
        timer.start();
        int inliers = lomsac.EstimateModel(makeOptions(squaredThreshold), solver, &F, &stats);
        double ms = timer.elapsedMs();
        cv::Mat mask = indicesToMask(stats.inlier_indices, n);
        double err = meanSampson(toCvMat3x3(F), pts1, pts2, mask);
        std::cout << "  RansacLib : Sampson " << err << ", inliers " << inliers << "/" << n
                  << ", iterations " << stats.num_iterations << ", time " << ms << " ms\n";
        rows.push_back({"F RansacLib (LO-MSAC)", err, inliers, ms});
    }
    {
        cv::Mat mask;
        timer.start();
        cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, threshold, 0.99, mask);
        double ms = timer.elapsedMs();
        double err = meanSampson(F, pts1, pts2, mask);
        std::cout << "  OpenCV    : Sampson " << err << ", inliers "
                  << cv::countNonZero(mask) << "/" << n << ", time " << ms << " ms\n";
        rows.push_back({"F OpenCV FM_RANSAC", err, cv::countNonZero(mask), ms});
    }

    // ---------- Summary ----------
    std::cout << "\n=== Summary (same data, same threshold/confidence) ===" << std::endl;
    std::cout << std::setprecision(4);
    std::cout << std::left << std::setw(28) << "Method"
              << std::right << std::setw(12) << "Error"
              << std::setw(12) << "Inliers"
              << std::setw(12) << "Time(ms)" << "\n";
    std::cout << std::string(64, '-') << "\n";
    for (const auto& r : rows) {
        std::cout << std::left << std::setw(28) << r.name << std::right;
        if (r.err < 0) std::cout << std::setw(12) << "-";
        else std::cout << std::setw(12) << r.err;
        std::cout << std::setw(12) << r.inliers
                  << std::setw(12) << r.time_ms << "\n";
    }

    std::cout << "\n=== Key Takeaways ===" << std::endl;
    std::cout << "1. One LO-MSAC estimator, three problem-specific Solver classes" << std::endl;
    std::cout << "2. The Solver encapsulates minimal solve, refinement, and residual" << std::endl;
    std::cout << "3. LO-MSAC's local optimization refines each new best model" << std::endl;
    std::cout << "4. Header-only library, easy to integrate" << std::endl;

    // Visualization: RansacLib homography inliers on the real pair.
    if (!ransaclibHMask.empty()) {
        cv::Mat vis = drawMatchesVis(img1, img2, pts1, pts2, ransaclibHMask);
        cv::imwrite("ransaclib_h_matches.jpg", vis);
        std::cout << "\nSaved: ransaclib_h_matches.jpg" << std::endl;
        if (std::getenv("DISPLAY") != nullptr) {
            cv::imshow("RansacLib H matches (green=inlier, red=outlier)", vis);
            std::cout << "Press any key to exit..." << std::endl;
            cv::waitKey(0);
        }
    }
    return 0;
}
