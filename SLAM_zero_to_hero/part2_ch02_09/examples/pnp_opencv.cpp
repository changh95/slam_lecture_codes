/**
 * @file pnp_opencv.cpp
 * @brief AprilTag camera localization with OpenCV's PnP solvers.
 *
 * Runs the shared pipeline (see apriltag_pnp.hpp) with OpenCV as the PnP
 * backend:
 *
 *   1. cv::solvePnPRansac  — EPnP hypotheses inside RANSAC reject outlier
 *      corner correspondences (reprojection threshold in pixels).
 *   2. cv::solvePnPRefineLM — Levenberg-Marquardt refinement on the inliers.
 *
 * OpenCV works directly in distorted pixel coordinates: the 14-coefficient
 * rational distortion model of the recording is passed straight to the
 * solver, no undistortion step is needed.
 */

#include <iostream>
#include <vector>

#include <opencv2/calib3d.hpp>

#include "apriltag_pnp.hpp"

namespace {

atpnp::PnPSolverResult solveOpenCV(const std::vector<cv::Point2f>& pts2d,
                                   const std::vector<cv::Point3f>& pts3d,
                                   const atpnp::CameraIntrinsics& cam) {
    atpnp::PnPSolverResult res;

    cv::Mat rvec, tvec, inliers;
    const bool ok = cv::solvePnPRansac(pts3d, pts2d, cam.K(), cam.distCoeffs(),
                                       rvec, tvec,
                                       false,               // useExtrinsicGuess
                                       200,                 // iterationsCount
                                       3.0,                 // reprojection threshold (px)
                                       0.99,                // confidence
                                       inliers,
                                       cv::SOLVEPNP_EPNP);  // hypothesis solver
    if (!ok || inliers.rows < 4) return res;

    // LM refinement on the RANSAC inliers.
    std::vector<cv::Point3f> in3d;
    std::vector<cv::Point2f> in2d;
    in3d.reserve(inliers.rows);
    in2d.reserve(inliers.rows);
    for (int i = 0; i < inliers.rows; i++) {
        const int idx = inliers.at<int>(i, 0);
        in3d.push_back(pts3d[idx]);
        in2d.push_back(pts2d[idx]);
    }
    cv::solvePnPRefineLM(in3d, in2d, cam.K(), cam.distCoeffs(), rvec, tvec);

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    res.ok = true;
    res.R_cw = cv::Matx33d(R);
    res.t_cw = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    res.num_inliers = inliers.rows;
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    atpnp::PipelineOptions opts;
    if (!atpnp::parseArgs(argc, argv, "opencv", &opts)) return 0;
    return atpnp::runPipeline("opencv", opts, solveOpenCV);
}
