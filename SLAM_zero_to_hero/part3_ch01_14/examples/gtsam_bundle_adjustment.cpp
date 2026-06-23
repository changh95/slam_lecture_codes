/**
 * GTSAM Tutorial: Bundle Adjustment with the BAL dataset
 *
 * Loads a BAL problem with GTSAM's own reader (SfmData::FromBalFile, which
 * handles the BAL camera/sign convention), builds a factor graph of
 * GeneralSFMFactor reprojection factors over PinholeCamera<Cal3Bundler> camera
 * variables and Point3 landmarks, anchors gauge with priors, and optimizes with
 * Levenberg-Marquardt. Dumps `bundle_adjustment.txt` for
 * viz/show_bundle_adjustment.py (rerun 3D).
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include <gtsam/geometry/Cal3Bundler.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/sfm/SfmData.h>
#include <gtsam/slam/GeneralSFMFactor.h>

using namespace std;
using namespace gtsam;

int main(int argc, char** argv) {
    cout << "=== GTSAM Tutorial: Bundle Adjustment (BAL) ===\n" << endl;

    string bal_file = (argc > 1) ? argv[1] : "problem-21-11315-pre.txt";
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
    cout << "Cameras: " << nC << "  Points: " << nT << "  Observations: " << nObs << endl;

    NonlinearFactorGraph graph;
    auto noise = noiseModel::Isotropic::Sigma(2, 1.0);
    for (size_t j = 0; j < nT; ++j) {
        for (const auto& m : db.track(j).measurements) {  // (camera index, pixel)
            graph.emplace_shared<GeneralSFMFactor<SfmCamera, Point3>>(
                m.second, noise, Symbol('c', m.first), Symbol('p', j));
        }
    }
    // Anchor gauge: prior on first camera and first point.
    graph.addPrior(Symbol('c', 0), db.camera(0), noiseModel::Isotropic::Sigma(9, 0.1));
    graph.addPrior(Symbol('p', 0), db.track(0).p, noiseModel::Isotropic::Sigma(3, 0.1));

    Values initial;
    for (size_t i = 0; i < nC; ++i) initial.insert(Symbol('c', i), db.camera(i));
    for (size_t j = 0; j < nT; ++j) initial.insert(Symbol('p', j), db.track(j).p);

    double e0 = graph.error(initial);
    cout << "Initial error: " << e0 << "  (RMSE "
         << sqrt(2.0 * e0 / nObs) << " px)" << endl;

    LevenbergMarquardtParams params;
    params.setMaxIterations(50);
    LevenbergMarquardtOptimizer optimizer(graph, initial, params);

    // Capture the landmarks + camera centres at every iteration (frame 0 = initial).
    struct Frame { vector<Point3> pts, cams; double err; };
    vector<Frame> frames;
    auto grab = [&](const Values& v, double err) {
        Frame fr;
        fr.err = err;
        fr.pts.reserve(nT);
        fr.cams.reserve(nC);
        for (size_t j = 0; j < nT; ++j) fr.pts.push_back(v.at<Point3>(Symbol('p', j)));
        for (size_t i = 0; i < nC; ++i)
            fr.cams.push_back(v.at<SfmCamera>(Symbol('c', i)).translation());
        frames.push_back(std::move(fr));
    };

    grab(optimizer.values(), e0);
    double prev = e0;
    for (int k = 0; k < 25; ++k) {
        optimizer.iterate();
        double e = optimizer.error();
        grab(optimizer.values(), e);
        if (k > 2 && prev - e < 1e-3 * prev) { prev = e; break; }
        prev = e;
    }

    Values result = optimizer.values();
    double e1 = graph.error(result);
    cout << "Final error:   " << e1 << "  (RMSE "
         << sqrt(2.0 * e1 / nObs) << " px)" << endl;
    cout << "Improvement: " << (1.0 - e1 / e0) * 100 << "%  ("
         << frames.size() << " iterations)" << endl;

    ofstream out("bundle_adjustment.txt");
    out << "points " << nT << "\n";
    out << "cameras " << nC << "\n";
    out << "steps " << frames.size() << "\n";
    for (size_t k = 0; k < frames.size(); ++k) {
        out << "step " << k << " " << 2.0 * frames[k].err << "\n";
        for (const Point3& p : frames[k].pts)
            out << p.x() << " " << p.y() << " " << p.z() << "\n";
        for (const Point3& c : frames[k].cams)
            out << c.x() << " " << c.y() << " " << c.z() << "\n";
    }
    out.close();
    cout << "\nWrote bundle_adjustment.txt (" << frames.size()
         << " iterations) -> visualize with viz/show_bundle_adjustment.py" << endl;

    return 0;
}
