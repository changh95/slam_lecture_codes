/**
 * g2o Tutorial: Bundle Adjustment with the BAL dataset
 *
 * Implements the BAL camera model with custom g2o types so the projection
 * matches the data exactly:
 *   - VertexCamera : 9-DoF (rotation as quaternion + translation + f, k1, k2)
 *                    with a proper SO(3) update in oplusImpl()
 *   - VertexPoint  : 3-DoF landmark (marginalized for the Schur complement)
 *   - EdgeReprojection : reprojection residual  p' = f (1+k1 r^2+k2 r^4)(-P/P.z)
 * Jacobians are computed numerically by g2o. Dumps `bundle_adjustment.txt`
 * for viz/show_bundle_adjustment.py (rerun 3D).
 */

#include <cmath>
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
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

using namespace std;

struct CameraParam {
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double f = 0, k1 = 0, k2 = 0;

    Eigen::Vector2d project(const Eigen::Vector3d& X) const {
        Eigen::Vector3d pc = q * X + t;
        Eigen::Vector2d p(-pc[0] / pc[2], -pc[1] / pc[2]);
        double r2 = p.squaredNorm();
        double dist = 1.0 + k1 * r2 + k2 * r2 * r2;
        return f * dist * p;
    }
    Eigen::Vector3d center() const { return -(q.conjugate() * t); }
};

// 9-DoF camera vertex with proper SO(3) (quaternion) update.
class VertexCamera : public g2o::BaseVertex<9, CameraParam> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate = CameraParam(); }
    void oplusImpl(const double* u) override {
        Eigen::Vector3d w(u[0], u[1], u[2]);
        double th = w.norm();
        Eigen::Quaterniond dq =
            (th < 1e-12) ? Eigen::Quaterniond(1.0, 0.5 * w[0], 0.5 * w[1], 0.5 * w[2])
                         : Eigen::Quaterniond(Eigen::AngleAxisd(th, w / th));
        _estimate.q = (dq * _estimate.q).normalized();
        _estimate.t += Eigen::Vector3d(u[3], u[4], u[5]);
        _estimate.f += u[6];
        _estimate.k1 += u[7];
        _estimate.k2 += u[8];
    }
    bool read(istream&) override { return false; }
    bool write(ostream&) const override { return false; }
};

class VertexPoint : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate.setZero(); }
    void oplusImpl(const double* u) override {
        _estimate += Eigen::Vector3d(u[0], u[1], u[2]);
    }
    bool read(istream&) override { return false; }
    bool write(ostream&) const override { return false; }
};

class EdgeReprojection
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCamera, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const auto* cam = static_cast<const VertexCamera*>(_vertices[0]);
        const auto* pt = static_cast<const VertexPoint*>(_vertices[1]);
        _error = cam->estimate().project(pt->estimate()) - _measurement;
    }
    bool read(istream&) override { return false; }
    bool write(ostream&) const override { return false; }
};

// Records the landmark cloud + camera centres after every optimizer iteration.
struct StructureRecorder : public g2o::HyperGraphAction {
    StructureRecorder(g2o::SparseOptimizer* opt,
                      const std::vector<VertexCamera*>& cams,
                      const std::vector<VertexPoint*>& pts)
        : opt_(opt), cams_(cams), pts_(pts) {}

    void capture() {
        std::vector<Eigen::Vector3d> p(pts_.size()), c(cams_.size());
        for (size_t i = 0; i < pts_.size(); ++i) p[i] = pts_[i]->estimate();
        for (size_t i = 0; i < cams_.size(); ++i) c[i] = cams_[i]->estimate().center();
        point_frames.push_back(std::move(p));
        camera_frames.push_back(std::move(c));
        opt_->computeActiveErrors();
        errors.push_back(opt_->chi2());  // total squared reprojection error
    }

    g2o::HyperGraphAction* operator()(const g2o::HyperGraph*, Parameters*) override {
        capture();
        return this;
    }

    g2o::SparseOptimizer* opt_;
    const std::vector<VertexCamera*>& cams_;
    const std::vector<VertexPoint*>& pts_;
    std::vector<std::vector<Eigen::Vector3d>> point_frames, camera_frames;
    std::vector<double> errors;
};

int main(int argc, char** argv) {
    cout << "=== g2o Tutorial: Bundle Adjustment (BAL) ===\n" << endl;

    string bal_file = (argc > 1) ? argv[1] : "problem-21-11315-pre.txt";
    ifstream fin(bal_file);
    if (!fin) {
        cerr << "Error: cannot open " << bal_file << "\n"
             << "Download a BAL dataset from "
                "https://grail.cs.washington.edu/projects/bal/\n";
        return 1;
    }

    int num_cameras, num_points, num_obs;
    fin >> num_cameras >> num_points >> num_obs;
    cout << "Cameras: " << num_cameras << "  Points: " << num_points
         << "  Observations: " << num_obs << endl;

    vector<int> cam_idx(num_obs), pt_idx(num_obs);
    vector<Eigen::Vector2d> obs(num_obs);
    for (int i = 0; i < num_obs; ++i)
        fin >> cam_idx[i] >> pt_idx[i] >> obs[i].x() >> obs[i].y();

    vector<CameraParam> cams(num_cameras);
    vector<Eigen::Vector3d> pts(num_points);
    for (int i = 0; i < num_cameras; ++i) {
        Eigen::Vector3d aa, t;
        double f, k1, k2;
        fin >> aa.x() >> aa.y() >> aa.z() >> t.x() >> t.y() >> t.z() >> f >> k1 >> k2;
        double th = aa.norm();
        cams[i].q = (th < 1e-12) ? Eigen::Quaterniond::Identity()
                                 : Eigen::Quaterniond(Eigen::AngleAxisd(th, aa / th));
        cams[i].t = t;
        cams[i].f = f;
        cams[i].k1 = k1;
        cams[i].k2 = k2;
    }
    for (int i = 0; i < num_points; ++i)
        fin >> pts[i].x() >> pts[i].y() >> pts[i].z();
    fin.close();

    // Solver: 9-DoF cameras, 3-DoF points, sparse Schur (Eigen), Levenberg.
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>>;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    vector<VertexCamera*> vcams(num_cameras);
    for (int i = 0; i < num_cameras; ++i) {
        vcams[i] = new VertexCamera();
        vcams[i]->setId(i);
        vcams[i]->setEstimate(cams[i]);
        optimizer.addVertex(vcams[i]);
    }
    vector<VertexPoint*> vpts(num_points);
    for (int i = 0; i < num_points; ++i) {
        vpts[i] = new VertexPoint();
        vpts[i]->setId(num_cameras + i);
        vpts[i]->setEstimate(pts[i]);
        vpts[i]->setMarginalized(true);  // Schur complement
        optimizer.addVertex(vpts[i]);
    }
    for (int i = 0; i < num_obs; ++i) {
        auto* e = new EdgeReprojection();
        e->setVertex(0, vcams[cam_idx[i]]);
        e->setVertex(1, vpts[pt_idx[i]]);
        e->setMeasurement(obs[i]);
        e->setInformation(Eigen::Matrix2d::Identity());
        e->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    double chi0 = optimizer.chi2();
    cout << "Initial chi2: " << chi0 << endl;

    // Record structure after every iteration; capture iteration 0 (initial) now.
    StructureRecorder recorder(&optimizer, vcams, vpts);
    recorder.capture();
    optimizer.addPostIterationAction(&recorder);

    optimizer.optimize(30);
    optimizer.computeActiveErrors();
    double chi1 = optimizer.chi2();
    cout << "Final chi2:   " << chi1 << "  (" << (1.0 - chi1 / chi0) * 100 << "% lower)" << endl;
    cout << "RMSE: " << sqrt(chi1 / num_obs) << " px" << endl;

    const auto& pf = recorder.point_frames;
    const auto& cf = recorder.camera_frames;
    const int K = static_cast<int>(pf.size());

    ofstream out("bundle_adjustment.txt");
    out << "points " << num_points << "\n";
    out << "cameras " << num_cameras << "\n";
    out << "steps " << K << "\n";
    for (int k = 0; k < K; ++k) {
        out << "step " << k << " " << recorder.errors[k] << "\n";
        for (int i = 0; i < num_points; ++i)
            out << pf[k][i].x() << " " << pf[k][i].y() << " " << pf[k][i].z() << "\n";
        for (int i = 0; i < num_cameras; ++i)
            out << cf[k][i].x() << " " << cf[k][i].y() << " " << cf[k][i].z() << "\n";
    }
    out.close();
    cout << "\nWrote bundle_adjustment.txt (" << K << " iterations) -> visualize with "
            "viz/show_bundle_adjustment.py" << endl;

    return 0;
}
