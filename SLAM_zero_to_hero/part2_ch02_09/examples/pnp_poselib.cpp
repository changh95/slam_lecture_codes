/**
 * @file pnp_poselib.cpp
 * @brief AprilTag camera localization with PoseLib's robust absolute-pose solver.
 *
 * Runs the shared pipeline (see apriltag_pnp.hpp) with PoseLib as the PnP
 * backend:
 *
 *   poselib::estimate_absolute_pose = LO-RANSAC over P3P minimal samples
 *   followed by non-linear (bundle) refinement of the winning model — the
 *   whole robust pipeline in a single call.
 *
 * PoseLib's camera models do not include OpenCV's 14-coefficient rational
 * distortion, so the detected corners are undistorted first
 * (cv::undistortPoints with P=K) and fed to an ideal PINHOLE camera model.
 */

#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <PoseLib/poselib.h>

#include "apriltag_pnp.hpp"

namespace {

atpnp::PnPSolverResult solvePoseLib(const std::vector<cv::Point2f>& pts2d,
                                    const std::vector<cv::Point3f>& pts3d,
                                    const atpnp::CameraIntrinsics& cam) {
    atpnp::PnPSolverResult res;

    // Undistort to ideal pixel coordinates (same K, distortion removed).
    const std::vector<cv::Point2f> und = atpnp::undistortToPixels(pts2d, cam);

    std::vector<poselib::Point2D> x;
    std::vector<poselib::Point3D> X;
    x.reserve(und.size());
    X.reserve(pts3d.size());
    for (size_t i = 0; i < und.size(); i++) {
        x.emplace_back(und[i].x, und[i].y);
        X.emplace_back(pts3d[i].x, pts3d[i].y, pts3d[i].z);
    }

    // Ideal pinhole camera (distortion already removed above).
    poselib::Image image;
    image.camera = poselib::Camera("PINHOLE", {cam.fx, cam.fy, cam.cx, cam.cy},
                                   cam.width, cam.height);

    poselib::AbsolutePoseOptions opt;
    opt.max_error = 3.0;  // RANSAC reprojection threshold (px)
    opt.ransac.min_iterations = 100;
    opt.ransac.max_iterations = 1000;

    std::vector<char> inliers;
    const poselib::RansacStats stats =
        poselib::estimate_absolute_pose(x, X, opt, &image, &inliers);

    if (stats.num_inliers < 4) return res;

    // PoseLib CameraPose is world->camera (x_cam = R * X + t), matching the
    // pipeline convention.
    const Eigen::Matrix3d R = image.pose.R();
    const Eigen::Vector3d t = image.pose.t;

    res.ok = true;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) res.R_cw(r, c) = R(r, c);
    res.t_cw = cv::Vec3d(t.x(), t.y(), t.z());
    res.num_inliers = static_cast<int>(stats.num_inliers);
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    atpnp::PipelineOptions opts;
    if (!atpnp::parseArgs(argc, argv, "poselib", &opts)) return 0;
    return atpnp::runPipeline("poselib", opts, solvePoseLib);
}
