/**
 * Fundamental Matrix Estimation using RansacLib
 *
 * This example demonstrates the template-based RANSAC design pattern using
 * RansacLib. RansacLib provides a clean separation between:
 *
 * 1. Solver: Encapsulates the minimal solver and residual computation
 * 2. Sampler: Generates random samples (uniform, PROSAC, etc.)
 * 3. Estimator: RANSAC variants (MSAC, LO-MSAC, etc.)
 *
 * This design allows easy customization of each component independently.
 *
 * Reference: https://github.com/tsattler/RansacLib
 */

#include <RansacLib/ransac.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// ============================================================================
// Fundamental Matrix Model
// ============================================================================

// The model type: a 3x3 fundamental matrix
using FundamentalMatrix = Eigen::Matrix3d;

// Container for multiple models (RansacLib requirement)
using FundamentalMatrixVector = std::vector<FundamentalMatrix,
    Eigen::aligned_allocator<FundamentalMatrix>>;

// ============================================================================
// Fundamental Matrix Solver (implements RansacLib Solver interface)
// ============================================================================

/**
 * FundamentalMatrixSolver implements the Solver interface required by RansacLib.
 *
 * Required methods:
 * - min_sample_size(): Minimum points for minimal solver (8 for 8-point algorithm)
 * - non_minimal_sample_size(): Points for non-minimal refinement
 * - num_data(): Total number of correspondences
 * - MinimalSolver(): Compute model from minimal sample
 * - NonMinimalSolver(): Refine model using more points (optional)
 * - EvaluateModelOnPoint(): Compute squared residual for a point
 * - LeastSquares(): Refine model using least squares (optional)
 */
class FundamentalMatrixSolver {
public:
    FundamentalMatrixSolver(const Eigen::Matrix<double, Eigen::Dynamic, 4>& data,
                            double squared_threshold)
        : data_(data), squared_threshold_(squared_threshold) {}

    // Minimum sample size (8 for normalized 8-point algorithm)
    inline int min_sample_size() const { return 8; }

    // Non-minimal sample size for refinement
    inline int non_minimal_sample_size() const { return 12; }

    // Total number of point correspondences
    inline int num_data() const { return static_cast<int>(data_.rows()); }

    // Minimal solver: 8-point algorithm
    int MinimalSolver(const std::vector<int>& sample,
                      FundamentalMatrixVector* models) const {
        if (sample.size() < 8) return 0;

        // Extract points for this sample
        Eigen::Matrix<double, 8, 4> pts;
        for (int i = 0; i < 8; ++i) {
            pts.row(i) = data_.row(sample[i]);
        }

        // Normalize points (Hartley normalization)
        Eigen::Vector2d mean1 = pts.leftCols<2>().colwise().mean();
        Eigen::Vector2d mean2 = pts.rightCols<2>().colwise().mean();

        double scale1 = 0.0, scale2 = 0.0;
        for (int i = 0; i < 8; ++i) {
            scale1 += (pts.block<1, 2>(i, 0).transpose() - mean1).norm();
            scale2 += (pts.block<1, 2>(i, 2).transpose() - mean2).norm();
        }
        scale1 = 8.0 * std::sqrt(2.0) / scale1;
        scale2 = 8.0 * std::sqrt(2.0) / scale2;

        // Normalization matrices
        Eigen::Matrix3d T1 = Eigen::Matrix3d::Identity();
        T1(0, 0) = scale1; T1(1, 1) = scale1;
        T1(0, 2) = -scale1 * mean1(0);
        T1(1, 2) = -scale1 * mean1(1);

        Eigen::Matrix3d T2 = Eigen::Matrix3d::Identity();
        T2(0, 0) = scale2; T2(1, 1) = scale2;
        T2(0, 2) = -scale2 * mean2(0);
        T2(1, 2) = -scale2 * mean2(1);

        // Build the design matrix A
        Eigen::Matrix<double, 8, 9> A;
        for (int i = 0; i < 8; ++i) {
            double x1 = scale1 * (pts(i, 0) - mean1(0));
            double y1 = scale1 * (pts(i, 1) - mean1(1));
            double x2 = scale2 * (pts(i, 2) - mean2(0));
            double y2 = scale2 * (pts(i, 3) - mean2(1));

            A(i, 0) = x2 * x1;
            A(i, 1) = x2 * y1;
            A(i, 2) = x2;
            A(i, 3) = y2 * x1;
            A(i, 4) = y2 * y1;
            A(i, 5) = y2;
            A(i, 6) = x1;
            A(i, 7) = y1;
            A(i, 8) = 1.0;
        }

        // Solve using SVD
        Eigen::JacobiSVD<Eigen::Matrix<double, 8, 9>> svd(A, Eigen::ComputeFullV);
        Eigen::Matrix<double, 9, 1> f = svd.matrixV().col(8);

        // Reshape to 3x3
        Eigen::Matrix3d F_normalized;
        F_normalized << f(0), f(1), f(2),
                        f(3), f(4), f(5),
                        f(6), f(7), f(8);

        // Enforce rank-2 constraint
        Eigen::JacobiSVD<Eigen::Matrix3d> svd_F(F_normalized, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d singularValues = svd_F.singularValues();
        singularValues(2) = 0.0;  // Set smallest singular value to 0
        F_normalized = svd_F.matrixU() * singularValues.asDiagonal() * svd_F.matrixV().transpose();

        // Denormalize: F = T2^T * F_normalized * T1
        FundamentalMatrix F = T2.transpose() * F_normalized * T1;

        // Normalize so ||F|| = 1
        F /= F.norm();

        models->push_back(F);
        return 1;
    }

    // Non-minimal solver: least squares with more points
    int NonMinimalSolver(const std::vector<int>& sample,
                         FundamentalMatrix* model) const {
        if (sample.size() < 8) return 0;

        const int n = static_cast<int>(sample.size());

        // Extract points
        Eigen::MatrixXd pts(n, 4);
        for (int i = 0; i < n; ++i) {
            pts.row(i) = data_.row(sample[i]);
        }

        // Normalize
        Eigen::Vector2d mean1 = pts.leftCols<2>().colwise().mean();
        Eigen::Vector2d mean2 = pts.rightCols<2>().colwise().mean();

        double scale1 = 0.0, scale2 = 0.0;
        for (int i = 0; i < n; ++i) {
            scale1 += (pts.block<1, 2>(i, 0).transpose() - mean1).norm();
            scale2 += (pts.block<1, 2>(i, 2).transpose() - mean2).norm();
        }
        scale1 = n * std::sqrt(2.0) / scale1;
        scale2 = n * std::sqrt(2.0) / scale2;

        Eigen::Matrix3d T1 = Eigen::Matrix3d::Identity();
        T1(0, 0) = scale1; T1(1, 1) = scale1;
        T1(0, 2) = -scale1 * mean1(0);
        T1(1, 2) = -scale1 * mean1(1);

        Eigen::Matrix3d T2 = Eigen::Matrix3d::Identity();
        T2(0, 0) = scale2; T2(1, 1) = scale2;
        T2(0, 2) = -scale2 * mean2(0);
        T2(1, 2) = -scale2 * mean2(1);

        // Build design matrix
        Eigen::MatrixXd A(n, 9);
        for (int i = 0; i < n; ++i) {
            double x1 = scale1 * (pts(i, 0) - mean1(0));
            double y1 = scale1 * (pts(i, 1) - mean1(1));
            double x2 = scale2 * (pts(i, 2) - mean2(0));
            double y2 = scale2 * (pts(i, 3) - mean2(1));

            A(i, 0) = x2 * x1;
            A(i, 1) = x2 * y1;
            A(i, 2) = x2;
            A(i, 3) = y2 * x1;
            A(i, 4) = y2 * y1;
            A(i, 5) = y2;
            A(i, 6) = x1;
            A(i, 7) = y1;
            A(i, 8) = 1.0;
        }

        // Solve
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::VectorXd f = svd.matrixV().col(8);

        Eigen::Matrix3d F_normalized;
        F_normalized << f(0), f(1), f(2),
                        f(3), f(4), f(5),
                        f(6), f(7), f(8);

        // Enforce rank-2
        Eigen::JacobiSVD<Eigen::Matrix3d> svd_F(F_normalized, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d singularValues = svd_F.singularValues();
        singularValues(2) = 0.0;
        F_normalized = svd_F.matrixU() * singularValues.asDiagonal() * svd_F.matrixV().transpose();

        // Denormalize
        *model = T2.transpose() * F_normalized * T1;
        model->array() /= model->norm();

        return 1;
    }

    // Evaluate model on a single point (returns squared Sampson error)
    double EvaluateModelOnPoint(const FundamentalMatrix& F, int i) const {
        Eigen::Vector3d p1(data_(i, 0), data_(i, 1), 1.0);
        Eigen::Vector3d p2(data_(i, 2), data_(i, 3), 1.0);

        // Sampson distance
        Eigen::Vector3d Fp1 = F * p1;
        Eigen::Vector3d Ftp2 = F.transpose() * p2;

        double numerator = p2.dot(Fp1);
        double denominator = Fp1(0) * Fp1(0) + Fp1(1) * Fp1(1) +
                             Ftp2(0) * Ftp2(0) + Ftp2(1) * Ftp2(1);

        if (denominator < 1e-10) return squared_threshold_ * 2.0;

        return (numerator * numerator) / denominator;
    }

    // Least squares refinement
    void LeastSquares(const std::vector<int>& sample, FundamentalMatrix* model) const {
        NonMinimalSolver(sample, model);
    }

private:
    Eigen::Matrix<double, Eigen::Dynamic, 4> data_;  // [x1, y1, x2, y2] per row
    double squared_threshold_;
};

// ============================================================================
// Generate Synthetic Stereo Data
// ============================================================================

void generateStereoData(Eigen::Matrix<double, Eigen::Dynamic, 4>& data,
                        FundamentalMatrix& gtF,
                        int numPoints = 150,
                        double outlierRatio = 0.35) {
    // Camera intrinsics
    Eigen::Matrix3d K;
    K << 500.0, 0.0, 320.0,
         0.0, 500.0, 240.0,
         0.0, 0.0, 1.0;

    // Relative pose
    Eigen::Matrix3d R;
    R << 0.9998, -0.01, 0.015,
         0.01, 0.9999, -0.005,
         -0.015, 0.0052, 0.9999;

    Eigen::Vector3d t(0.1, 0.01, 0.02);

    // Essential matrix: E = [t]_x * R
    Eigen::Matrix3d tx;
    tx << 0, -t(2), t(1),
          t(2), 0, -t(0),
          -t(1), t(0), 0;
    Eigen::Matrix3d E = tx * R;

    // Fundamental matrix: F = K^(-T) * E * K^(-1)
    gtF = K.inverse().transpose() * E * K.inverse();
    gtF /= gtF.norm();

    int numInliers = static_cast<int>(numPoints * (1.0 - outlierRatio));
    int numOutliers = numPoints - numInliers;

    data.resize(numPoints, 4);

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
        double x = (u1 - K(0, 2)) * depth / K(0, 0);
        double y = (v1 - K(1, 2)) * depth / K(1, 1);
        double z = depth;

        Eigen::Vector3d P(x, y, z);
        Eigen::Vector3d P2 = R * P + t;

        // Project to image 2
        double u2 = K(0, 0) * P2(0) / P2(2) + K(0, 2);
        double v2 = K(1, 1) * P2(1) / P2(2) + K(1, 2);

        data(idx, 0) = u1 + noise(gen);
        data(idx, 1) = v1 + noise(gen);
        data(idx, 2) = u2 + noise(gen);
        data(idx, 3) = v2 + noise(gen);
        ++idx;
    }

    // Generate outliers
    for (int i = 0; i < numOutliers; ++i) {
        data(idx, 0) = distX(gen);
        data(idx, 1) = distY(gen);
        data(idx, 2) = distX(gen);
        data(idx, 3) = distY(gen);
        ++idx;
    }

    // Shuffle
    std::vector<int> indices(numPoints);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    Eigen::Matrix<double, Eigen::Dynamic, 4> shuffled(numPoints, 4);
    for (int i = 0; i < numPoints; ++i) {
        shuffled.row(i) = data.row(indices[i]);
    }
    data = shuffled;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Fundamental Matrix Estimation using RansacLib ===" << std::endl;
    std::cout << "Demonstrating template-based RANSAC design pattern\n" << std::endl;

    // Generate synthetic data
    Eigen::Matrix<double, Eigen::Dynamic, 4> data;
    FundamentalMatrix gtF;
    const int numPoints = 200;
    const double outlierRatio = 0.35;
    const double threshold = 3.0;
    const double squaredThreshold = threshold * threshold;

    generateStereoData(data, gtF, numPoints, outlierRatio);

    std::cout << "Data: " << numPoints << " correspondences, "
              << static_cast<int>(outlierRatio * 100) << "% outliers\n";
    std::cout << "Inlier threshold: " << threshold << " px\n\n";

    // ========== RansacLib: LocallyOptimizedMSAC ==========
    std::cout << "--- RansacLib: LocallyOptimizedMSAC ---" << std::endl;

    // Configure RANSAC options
    ransac_lib::LORansacOptions options;
    options.min_num_iterations_ = 100;
    options.max_num_iterations_ = 10000;
    options.squared_inlier_threshold_ = squaredThreshold;
    options.min_sample_multiplicator_ = 7;
    options.num_lsq_iterations_ = 4;
    options.num_lo_steps_ = 10;
    options.random_seed_ = 42;

    // Create solver
    FundamentalMatrixSolver solver(data, squaredThreshold);

    // Create LO-MSAC estimator and run
    ransac_lib::LocallyOptimizedMSAC<FundamentalMatrix, FundamentalMatrixVector,
                                      FundamentalMatrixSolver> lomsac;

    ransac_lib::RansacStatistics stats;
    FundamentalMatrix bestModel;

    auto start = std::chrono::high_resolution_clock::now();
    int numInliers = lomsac.EstimateModel(options, solver, &bestModel, &stats);
    auto end = std::chrono::high_resolution_clock::now();
    double ransaclibTime = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "  Inliers: " << numInliers << "/" << numPoints << std::endl;
    std::cout << "  Iterations: " << stats.num_iterations << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << ransaclibTime << " ms\n";

    // Compute Sampson error with ground truth inliers
    double avgError = 0.0;
    for (int i = 0; i < data.rows(); ++i) {
        avgError += solver.EvaluateModelOnPoint(bestModel, i);
    }
    avgError /= data.rows();
    std::cout << "  Avg Sampson error: " << std::setprecision(6) << avgError << "\n\n";

    // ========== OpenCV: USAC_MAGSAC for comparison ==========
    std::cout << "--- OpenCV: USAC_MAGSAC (for comparison) ---" << std::endl;

    // Convert to OpenCV format
    std::vector<cv::Point2f> pts1(numPoints), pts2(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        pts1[i] = cv::Point2f(static_cast<float>(data(i, 0)),
                              static_cast<float>(data(i, 1)));
        pts2[i] = cv::Point2f(static_cast<float>(data(i, 2)),
                              static_cast<float>(data(i, 3)));
    }

    cv::Mat mask;
    start = std::chrono::high_resolution_clock::now();
    cv::Mat cvF = cv::findFundamentalMat(pts1, pts2, cv::USAC_MAGSAC, threshold, 0.99, mask);
    end = std::chrono::high_resolution_clock::now();
    double opencvTime = std::chrono::duration<double, std::milli>(end - start).count();

    int cvInliers = cv::countNonZero(mask);
    std::cout << "  Inliers: " << cvInliers << "/" << numPoints << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << opencvTime << " ms\n\n";

    // ========== Summary ==========
    std::cout << "=== Summary ===" << std::endl;
    std::cout << std::left << std::setw(25) << "Method"
              << std::setw(12) << "Inliers"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(49, '-') << std::endl;
    std::cout << std::left << std::setw(25) << "RansacLib (LO-MSAC)"
              << std::setw(12) << numInliers
              << std::setw(12) << std::fixed << std::setprecision(2) << ransaclibTime << std::endl;
    std::cout << std::left << std::setw(25) << "OpenCV (USAC_MAGSAC)"
              << std::setw(12) << cvInliers
              << std::setw(12) << std::fixed << std::setprecision(2) << opencvTime << std::endl;

    std::cout << "\n=== Key Takeaways ===" << std::endl;
    std::cout << "1. RansacLib uses a template-based design for flexibility" << std::endl;
    std::cout << "2. Custom Solver class encapsulates problem-specific logic" << std::endl;
    std::cout << "3. LO-MSAC adds local optimization for better accuracy" << std::endl;
    std::cout << "4. Header-only library, easy to integrate" << std::endl;

    return 0;
}
