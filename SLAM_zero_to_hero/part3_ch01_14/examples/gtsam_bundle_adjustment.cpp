/**
 * GTSAM: bundle adjustment on a BAL problem
 *
 * Loads BAL Trafalgar Square problem-21-11315-pre.txt with GTSAM's own reader
 * (SfmData::FromBalFile) and optimizes it with Levenberg-Marquardt, one step at
 * a time so every iteration streams to a live rerun viewer.
 *
 * Shared exercise setup (identical in the g2o / Ceres / SymForce chapters):
 *   - The per-camera intrinsics (f, k1, k2) are read from the dataset and held
 *     FIXED. Only the 6-DoF camera poses and the 3D point positions are
 *     optimized. That is why the factor here is
 *     GenericProjectionFactor<Pose3, Point3, Cal3Bundler> - it takes the
 *     calibration as a constant constructor argument - and not
 *     GeneralSFMFactor<SfmCamera, Point3>, which optimizes the calibration
 *     jointly with the pose.
 *   - No robust kernel. GTSAM offers noiseModel::Robust::Create(
 *     noiseModel::mEstimator::Huber::Create(k), model) and it is deliberately
 *     unused here so the reported error IS the objective being minimized and is
 *     directly comparable to the other chapters.
 *   - Gauge: camera 0 is pinned with a tight prior (GTSAM has no per-variable
 *     "fix" flag). No point is pinned - that would over-constrain.
 *   - 30 LM iterations, every iteration logged including the initial state.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtsam/geometry/Cal3Bundler.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/sfm/SfmData.h>
#include <gtsam/slam/ProjectionFactor.h>

#include "rerun_viz.hpp"

using namespace std;
using namespace gtsam;

// GTSAM's VerbosityLM output rewrites cout's precision while it prints, so this
// puts it back before each of our own lines.
static ostream& msg() { return cout << setprecision(6); }

int main(int argc, char** argv) {
    msg() << "=== GTSAM: bundle adjustment (BAL) ===\n" << endl;

    part3viz::Viz viz(part3viz::kBundleAdjustmentRecording, "gtsam");

    const string bal_file = (argc > 1) ? argv[1] : "problem-21-11315-pre.txt";
    SfmData db;
    try {
        db = SfmData::FromBalFile(bal_file);
    } catch (const exception& e) {
        cerr << "Error: cannot read BAL file " << bal_file << " (" << e.what() << ")\n"
             << "Download from https://grail.cs.washington.edu/projects/bal/\n";
        return 1;
    }

    const size_t nC = db.numberCameras(), nT = db.numberTracks();
    size_t nObs = 0;
    for (size_t j = 0; j < nT; ++j) nObs += db.track(j).measurements.size();
    msg() << "File: " << bal_file << endl;
    msg() << "Cameras: " << nC << "  Points: " << nT << "  Observations: " << nObs
         << endl;
    msg() << "Intrinsics (f, k1, k2) are FIXED at the dataset values; only the "
            "6-DoF poses and the 3D points are optimized."
         << endl;

    // Fixed calibrations, one per camera, straight from the file.
    vector<shared_ptr<Cal3Bundler>> calib;
    calib.reserve(nC);
    for (size_t i = 0; i < nC; ++i) {
        // Qualified: both std:: and gtsam:: provide make_shared.
        calib.push_back(std::make_shared<Cal3Bundler>(db.camera(i).calibration()));
    }

    NonlinearFactorGraph graph;
    auto noise = noiseModel::Isotropic::Sigma(2, 1.0);  // pixels
    for (size_t j = 0; j < nT; ++j) {
        for (const auto& m : db.track(j).measurements) {  // (camera index, pixel)
            graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3Bundler>>(
                m.second, noise, Symbol('x', m.first), Symbol('l', j),
                calib[m.first]);
        }
    }
    // Gauge: pin camera 0. This removes 6 of the 7 gauge degrees of freedom;
    // overall scale stays free because the BAL projection is scale-invariant, so
    // LM's damping term handles that remaining direction.
    graph.addPrior(Symbol('x', 0), db.camera(0).pose(),
                   noiseModel::Isotropic::Sigma(6, 1e-6));

    Values initial;
    for (size_t i = 0; i < nC; ++i) initial.insert(Symbol('x', i), db.camera(i).pose());
    for (size_t j = 0; j < nT; ++j) initial.insert(Symbol('l', j), db.track(j).p);

    // Reported metrics, with the same formulas in every chapter:
    //   sq_error = sum over observations of |projected - measured|^2  (raw)
    //   rmse_px  = sqrt(sq_error / num_observations)   <- per observation
    // The projection is written out rather than taken from PinholeCamera::project
    // so that a point behind a camera produces the same (mirrored) reprojection
    // the other chapters compute instead of a CheiralityException.
    const auto metrics = [&](const Values& v) {
        double sq = 0.0;
        for (size_t j = 0; j < nT; ++j) {
            const Point3 P = v.at<Point3>(Symbol('l', j));
            for (const auto& m : db.track(j).measurements) {
                const Pose3& pose = v.at<Pose3>(Symbol('x', m.first));
                const Point3 pc = pose.transformTo(P);
                const Point2 uv =
                    calib[m.first]->uncalibrate(Point2(pc.x() / pc.z(), pc.y() / pc.z()));
                const Point2 d = uv - m.second;
                sq += d.squaredNorm();
            }
        }
        return sq;
    };

    const auto landmarks = [&](const Values& v) {
        vector<part3viz::Vec3> pts;
        pts.reserve(nT);
        for (size_t j = 0; j < nT; ++j) {
            const Point3 p = v.at<Point3>(Symbol('l', j));
            pts.push_back({p.x(), p.y(), p.z()});
        }
        return pts;
    };
    // GTSAM stores camera-to-world poses, so the camera centre is the pose
    // translation directly - no -R^T t as in the g2o / Ceres / SymForce
    // chapters, which carry world-to-camera parameters.
    const auto centers = [&](const Values& v) {
        vector<part3viz::Vec3> cams;
        cams.reserve(nC);
        for (size_t i = 0; i < nC; ++i) {
            const Point3 c = v.at<Pose3>(Symbol('x', i)).translation();
            cams.push_back({c.x(), c.y(), c.z()});
        }
        return cams;
    };

    const double sq0 = metrics(initial);
    const double rmse0 = sqrt(sq0 / static_cast<double>(nObs));
    msg() << "\nInitial: sq_error = " << sq0 << "  rmse = " << rmse0 << " px" << endl;

    viz.baSetup(landmarks(initial));
    viz.baIteration(0, landmarks(initial), centers(initial), sq0, rmse0);

    LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");
    LevenbergMarquardtOptimizer optimizer(graph, initial, params);

    const int kMaxIterations = 30;
    double sq = sq0, rmse = rmse0;
    int iterations = 0;
    for (int it = 1; it <= kMaxIterations; ++it) {
        optimizer.iterate();
        const Values& v = optimizer.values();
        const double next = metrics(v);
        const double next_rmse = sqrt(next / static_cast<double>(nObs));
        ++iterations;
        viz.baIteration(it, landmarks(v), centers(v), next, next_rmse);
        msg() << "Iteration " << it << ": sq_error = " << next << "  rmse = "
             << next_rmse << " px" << endl;
        const bool converged = (sq - next) <= 1e-6 * max(1.0, sq);
        sq = next;
        rmse = next_rmse;
        if (converged) break;
    }

    // The gauge anchor is a prior, not a hard constraint, so show how far
    // camera 0 actually drifted - it should be numerically zero.
    const Pose3 cam0_before = db.camera(0).pose();
    const Pose3 cam0_after = optimizer.values().at<Pose3>(Symbol('x', 0));
    msg() << "\nCamera 0 drift from its prior: "
          << (cam0_after.translation() - cam0_before.translation()).norm()
          << " (translation), "
          << Rot3::Logmap(cam0_before.rotation().between(cam0_after.rotation())).norm()
          << " rad (rotation)" << endl;

    msg() << "\nsq_error: " << sq0 << " -> " << sq << endl;
    msg() << "rmse    : " << rmse0 << " px -> " << rmse << " px" << endl;
    msg() << "Reduction: " << (1.0 - sq / sq0) * 100.0 << "% of the squared error  ("
         << iterations << " LM iterations)" << endl;

    return 0;
}
