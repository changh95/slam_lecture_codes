/**
 * g2o Tutorial: Bundle Adjustment with the BAL dataset
 *
 * Implements the BAL camera model with custom g2o types so the projection
 * matches the data exactly:
 *   - VertexCamera      : 6-DoF pose (quaternion rotation with a proper SO(3)
 *                         update in oplusImpl, plus translation)
 *   - VertexPoint       : 3-DoF landmark, marginalized for the Schur complement
 *   - EdgeReprojection  : residual  p' = f (1 + k1 r^2 + k2 r^4) (-P/P.z)
 *
 * The per-camera intrinsics (f, k1, k2) are read from the dataset and held
 * FIXED - they are constant members of the edge, not part of any vertex
 * estimate. Only the 6-DoF poses and the 3D points are optimized. Jacobians
 * are computed numerically by g2o.
 *
 * This is the shared bundle-adjustment exercise of part3 chapter 1: same
 * dataset, no robust kernel, camera 0 fixed, 30 iterations, and the same two
 * reported metrics as the GTSAM / Ceres / SymForce chapters.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/hyper_graph_action.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include "rerun_viz.hpp"

namespace {

constexpr int kMaxIterations = 30;

/// Per-camera intrinsics, read from the dataset and never optimized.
struct Intrinsics {
    double f = 0.0, k1 = 0.0, k2 = 0.0;
};

/// The optimized part of a BAL camera: world-to-camera rotation + translation.
struct CameraPose {
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    /// Camera centre in world coordinates: C = -R^T t.
    Eigen::Vector3d center() const { return -(q.conjugate() * t); }
};

/// BAL projection: P_cam = R X + t, p = (-P.x/P.z, -P.y/P.z),
/// p' = f (1 + k1 r^2 + k2 r^4) p.
Eigen::Vector2d project(const CameraPose& cam, const Intrinsics& in,
                        const Eigen::Vector3d& X) {
    const Eigen::Vector3d pc = cam.q * X + cam.t;
    const Eigen::Vector2d p(-pc[0] / pc[2], -pc[1] / pc[2]);
    const double r2 = p.squaredNorm();
    return in.f * (1.0 + in.k1 * r2 + in.k2 * r2 * r2) * p;
}

// 6-DoF camera vertex: only the pose. (f, k1, k2) are not here on purpose.
class VertexCamera : public g2o::BaseVertex<6, CameraPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate = CameraPose(); }
    void oplusImpl(const double* u) override {
        const Eigen::Vector3d w(u[0], u[1], u[2]);
        const double th = w.norm();
        const Eigen::Quaterniond dq =
            (th < 1e-12)
                ? Eigen::Quaterniond(1.0, 0.5 * w[0], 0.5 * w[1], 0.5 * w[2])
                : Eigen::Quaterniond(Eigen::AngleAxisd(th, w / th));
        _estimate.q = (dq * _estimate.q).normalized();
        _estimate.t += Eigen::Vector3d(u[3], u[4], u[5]);
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

class VertexPoint : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate.setZero(); }
    void oplusImpl(const double* u) override {
        _estimate += Eigen::Vector3d(u[0], u[1], u[2]);
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

/// Reprojection residual. Carries its camera's fixed intrinsics as constants.
class EdgeReprojection
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCamera, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit EdgeReprojection(const Intrinsics& in) : intrinsics_(in) {}

    void computeError() override {
        const auto* cam = static_cast<const VertexCamera*>(_vertices[0]);
        const auto* pt = static_cast<const VertexPoint*>(_vertices[1]);
        _error = project(cam->estimate(), intrinsics_, pt->estimate()) - _measurement;
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

private:
    const Intrinsics intrinsics_;  // fixed, never updated by the optimizer
};

/**
 * Post-iteration hook: streams the landmark cloud, the camera centres and the
 * two reported metrics to the viewer after every optimizer iteration.
 */
struct StructureRecorder : public g2o::HyperGraphAction {
    StructureRecorder(const std::vector<VertexCamera*>& cams,
                      const std::vector<VertexPoint*>& pts,
                      const std::vector<Intrinsics>& intrinsics,
                      const std::vector<int>& cam_idx, const std::vector<int>& pt_idx,
                      const std::vector<Eigen::Vector2d>& obs, part3viz::Viz* viz)
        : cams_(cams),
          pts_(pts),
          intrinsics_(intrinsics),
          cam_idx_(cam_idx),
          pt_idx_(pt_idx),
          obs_(obs),
          viz_(viz) {}

    /**
     * The one shared metric pair:
     *   sq_error = sum over observations of |projected - measured|^2
     *   rmse_px  = sqrt(sq_error / num_observations)     [per observation]
     * Computed straight from the model, so it is the raw least-squares
     * objective and means the same thing in every chapter.
     */
    std::pair<double, double> metrics() const {
        double sq = 0.0;
        for (std::size_t i = 0; i < obs_.size(); ++i) {
            const Eigen::Vector2d p =
                project(cams_[cam_idx_[i]]->estimate(), intrinsics_[cam_idx_[i]],
                        pts_[pt_idx_[i]]->estimate());
            sq += (p - obs_[i]).squaredNorm();
        }
        return {sq, std::sqrt(sq / static_cast<double>(obs_.size()))};
    }

    void record(int iter) {
        std::vector<part3viz::Vec3> points(pts_.size()), centers(cams_.size());
        for (std::size_t i = 0; i < pts_.size(); ++i) {
            const Eigen::Vector3d& p = pts_[i]->estimate();
            points[i] = {p.x(), p.y(), p.z()};
        }
        for (std::size_t i = 0; i < cams_.size(); ++i) {
            const Eigen::Vector3d c = cams_[i]->estimate().center();
            centers[i] = {c.x(), c.y(), c.z()};
        }
        const auto [sq, rmse] = metrics();
        viz_->baIteration(iter, points, centers, sq, rmse);
        history.push_back({sq, rmse});
    }

    g2o::HyperGraphAction* operator()(const g2o::HyperGraph*,
                                      Parameters* params) override {
        // Frame 0 is the initial state, so g2o's iteration i is frame i + 1.
        // g2o also fires the post-iteration actions once with iteration = -1
        // (the "about to start" notification) - that is not an iteration.
        int iter = static_cast<int>(history.size());
        if (const auto* p = dynamic_cast<const ParametersIteration*>(params)) {
            if (p->iteration < 0) return this;
            iter = p->iteration + 1;
        }
        record(iter);
        std::printf("  iter %2d  sq_error %.6g  rmse %.4f px\n", iter,
                    history.back().sq_error, history.back().rmse_px);
        std::fflush(stdout);
        return this;
    }

    struct Frame {
        double sq_error;
        double rmse_px;
    };
    std::vector<Frame> history;

private:
    const std::vector<VertexCamera*>& cams_;
    const std::vector<VertexPoint*>& pts_;
    const std::vector<Intrinsics>& intrinsics_;
    const std::vector<int>& cam_idx_;
    const std::vector<int>& pt_idx_;
    const std::vector<Eigen::Vector2d>& obs_;
    part3viz::Viz* viz_;
};

}  // namespace

int main(int argc, char** argv) {
    std::cout << "=== g2o Tutorial: Bundle Adjustment (BAL) ===\n" << std::endl;

    const std::string bal_file = (argc > 1) ? argv[1] : "problem-21-11315-pre.txt";
    std::ifstream fin(bal_file);
    if (!fin) {
        std::cerr << "Error: cannot open " << bal_file << "\n"
                  << "Download a BAL dataset from "
                     "https://grail.cs.washington.edu/projects/bal/\n";
        return 1;
    }

    int num_cameras = 0, num_points = 0, num_obs = 0;
    fin >> num_cameras >> num_points >> num_obs;
    std::cout << "Cameras: " << num_cameras << "  Points: " << num_points
              << "  Observations: " << num_obs << std::endl;

    std::vector<int> cam_idx(num_obs), pt_idx(num_obs);
    std::vector<Eigen::Vector2d> obs(num_obs);
    for (int i = 0; i < num_obs; ++i) {
        fin >> cam_idx[i] >> pt_idx[i] >> obs[i].x() >> obs[i].y();
    }

    std::vector<CameraPose> cams(num_cameras);
    std::vector<Intrinsics> intrinsics(num_cameras);
    std::vector<Eigen::Vector3d> pts(num_points);
    for (int i = 0; i < num_cameras; ++i) {
        Eigen::Vector3d aa, t;
        double f = 0, k1 = 0, k2 = 0;
        fin >> aa.x() >> aa.y() >> aa.z() >> t.x() >> t.y() >> t.z() >> f >> k1 >> k2;
        const double th = aa.norm();
        cams[i].q = (th < 1e-12) ? Eigen::Quaterniond::Identity()
                                 : Eigen::Quaterniond(Eigen::AngleAxisd(th, aa / th));
        cams[i].t = t;
        intrinsics[i] = {f, k1, k2};
    }
    for (int i = 0; i < num_points; ++i) {
        fin >> pts[i].x() >> pts[i].y() >> pts[i].z();
    }
    fin.close();

    std::cout << "Intrinsics (f, k1, k2) are read from the dataset and held FIXED; "
                 "only the 6-DoF poses and the 3D points are optimized."
              << std::endl;

    // Solver: 6-DoF cameras, 3-DoF points, sparse Schur (Eigen), Levenberg.
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    std::vector<VertexCamera*> vcams(num_cameras);
    for (int i = 0; i < num_cameras; ++i) {
        vcams[i] = new VertexCamera();
        vcams[i]->setId(i);
        vcams[i]->setEstimate(cams[i]);
        optimizer.addVertex(vcams[i]);
    }
    // Gauge: hard-fix camera 0's pose. That removes 6 of the 7 gauge degrees of
    // freedom; overall SCALE stays free, because the BAL projection is
    // scale-invariant (scaling every t and X leaves p' unchanged). LM damping
    // handles that remaining direction. Fixing a point as well would
    // over-constrain the problem.
    vcams[0]->setFixed(true);

    std::vector<VertexPoint*> vpts(num_points);
    for (int i = 0; i < num_points; ++i) {
        vpts[i] = new VertexPoint();
        vpts[i]->setId(num_cameras + i);
        vpts[i]->setEstimate(pts[i]);
        vpts[i]->setMarginalized(true);  // Schur complement
        optimizer.addVertex(vpts[i]);
    }
    // No robust kernel. g2o offers g2o::RobustKernelHuber (and Cauchy, Tukey,
    // ...) via edge->setRobustKernel(), but it is deliberately off here so that
    // the reported error IS the objective being minimized and the four
    // chapters' numbers are directly comparable.
    for (int i = 0; i < num_obs; ++i) {
        auto* e = new EdgeReprojection(intrinsics[cam_idx[i]]);
        e->setVertex(0, vcams[cam_idx[i]]);
        e->setVertex(1, vpts[pt_idx[i]]);
        e->setMeasurement(obs[i]);
        e->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(e);
    }

    part3viz::Viz viz(part3viz::kBundleAdjustmentRecording, "g2o");
    std::vector<part3viz::Vec3> initial_points(num_points);
    for (int i = 0; i < num_points; ++i) {
        initial_points[i] = {pts[i].x(), pts[i].y(), pts[i].z()};
    }
    viz.baSetup(initial_points);

    StructureRecorder recorder(vcams, vpts, intrinsics, cam_idx, pt_idx, obs, &viz);

    optimizer.initializeOptimization();
    recorder.record(0);  // frame 0: the initial state
    optimizer.addPostIterationAction(&recorder);

    std::printf("\nInitial sq_error: %.6g   RMSE: %.6f px\n",
                recorder.history.front().sq_error, recorder.history.front().rmse_px);
    std::cout << std::endl;

    optimizer.optimize(kMaxIterations);

    const double sq0 = recorder.history.front().sq_error;
    const double sq1 = recorder.history.back().sq_error;
    const double rmse0 = recorder.history.front().rmse_px;
    const double rmse1 = recorder.history.back().rmse_px;
    const int iterations = static_cast<int>(recorder.history.size()) - 1;

    std::cout << "\n--- Results (raw least squares, no robust kernel) ---" << std::endl;
    std::printf("sq_error : %.6g -> %.6g   (%.2f%% lower)\n", sq0, sq1,
                (1.0 - sq1 / sq0) * 100.0);
    std::printf("rmse_px  : %.6f -> %.6f   (%.2f%% lower)\n", rmse0, rmse1,
                (1.0 - rmse1 / rmse0) * 100.0);
    std::printf("Iterations: %d (logged frames 0..%d on the 'iteration' timeline)\n",
                iterations, iterations);

    return 0;
}
