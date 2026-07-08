/**
 * @file pnp_opengv.cpp
 * @brief AprilTag camera localization with OpenGV's absolute-pose solvers.
 *
 * Runs the shared pipeline (see apriltag_pnp.hpp) with OpenGV as the PnP
 * backend:
 *
 *   1. Corners are undistorted and converted to unit bearing vectors
 *      (OpenGV is intrinsics-free: it works on rays, not pixels).
 *   2. Kneip's P3P inside OpenGV's RANSAC rejects outliers. The RANSAC
 *      threshold is angular: 1 - cos(atan(px_thresh / f)).
 *   3. opengv::absolute_pose::optimize_nonlinear refines the pose on the
 *      inlier set.
 *
 * Note on conventions: OpenGV returns the camera pose in the world frame
 * (R_wc and the camera position), i.e. bearing ~ R_wc^T * (X - t_w). The
 * pipeline expects world->camera, so the result is inverted here.
 */

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "apriltag_pnp.hpp"

namespace {

atpnp::PnPSolverResult solveOpenGV(const std::vector<cv::Point2f>& pts2d,
                                   const std::vector<cv::Point3f>& pts3d,
                                   const atpnp::CameraIntrinsics& cam) {
    atpnp::PnPSolverResult res;

    // Undistorted normalized coordinates -> unit bearing vectors.
    const std::vector<cv::Point2f> norm = atpnp::undistortToNormalized(pts2d, cam);
    opengv::bearingVectors_t bearings;
    opengv::points_t points;
    bearings.reserve(norm.size());
    points.reserve(pts3d.size());
    for (size_t i = 0; i < norm.size(); i++) {
        bearings.push_back(Eigen::Vector3d(norm[i].x, norm[i].y, 1.0).normalized());
        points.push_back(Eigen::Vector3d(pts3d[i].x, pts3d[i].y, pts3d[i].z));
    }

    opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearings, points);

    // Kneip P3P inside RANSAC. The threshold is on the angle between the
    // measured and reprojected bearing: ~3 px at the focal length.
    using SacProblem = opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
    opengv::sac::Ransac<SacProblem> ransac;
    ransac.sac_model_ = std::make_shared<SacProblem>(adapter, SacProblem::KNEIP);
    ransac.threshold_ = 1.0 - std::cos(std::atan(3.0 / cam.fx));
    ransac.max_iterations_ = 200;

    if (!ransac.computeModel() || ransac.inliers_.size() < 4) return res;

    // Nonlinear refinement on the inliers, initialized at the RANSAC model.
    opengv::transformation_t T = ransac.model_coefficients_;
    adapter.setR(T.block<3, 3>(0, 0));
    adapter.sett(T.col(3));
    T = opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

    // Convert camera-to-world (OpenGV) -> world-to-camera (pipeline).
    const Eigen::Matrix3d R_wc = T.block<3, 3>(0, 0);
    const Eigen::Vector3d t_w = T.col(3);
    const Eigen::Matrix3d R_cw = R_wc.transpose();
    const Eigen::Vector3d t_cw = -R_cw * t_w;

    res.ok = true;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) res.R_cw(r, c) = R_cw(r, c);
    res.t_cw = cv::Vec3d(t_cw.x(), t_cw.y(), t_cw.z());
    res.num_inliers = static_cast<int>(ransac.inliers_.size());
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    atpnp::PipelineOptions opts;
    if (!atpnp::parseArgs(argc, argv, "opengv", &opts)) return 0;
    return atpnp::runPipeline("opengv", opts, solveOpenGV);
}
