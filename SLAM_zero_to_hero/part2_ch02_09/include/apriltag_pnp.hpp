/**
 * @file apriltag_pnp.hpp
 * @brief Shared infrastructure for the AprilTag + PnP camera localization demos.
 *
 * All three demos (pnp_opencv / pnp_poselib / pnp_opengv) run the exact same
 * pipeline on data/plantage_shed.mp4 and differ only in the PnP solver used
 * to estimate the camera pose:
 *
 *   1. Detect tagStandard52h13 AprilTags (42 mm side length) in each frame.
 *   2. Bootstrap the world frame from tag 10 (always visible in the video):
 *      world origin = tag 10 center, world z = tag 10 normal.
 *   3. Estimate the camera pose from all 2D-3D correspondences of tags that
 *      are already in the map (the method-specific PnP under comparison).
 *   4. Extend the map: unmapped tags observed with an unambiguous
 *      IPPE_SQUARE pose are transformed into the world frame.
 *   5. Write a JSON file consumed by viz_pnp.py (rerun visualization).
 *
 * When built with the rerun C++ SDK, every processed frame is additionally
 * streamed live to a rerun viewer (see rerun_stream.hpp).
 *
 * Camera intrinsics come from the recording's calibration
 * (monumental_assessment cam.json). Units are millimetres throughout.
 */

#pragma once

#include <array>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct apriltag_detector;
struct apriltag_family;

namespace atpnp {

// ---------------------------------------------------------------------------
// Constants of the dataset
// ---------------------------------------------------------------------------

/// AprilTag side length in millimetres (tagStandard52h13 family).
constexpr double kTagSizeMm = 42.0;

/// Tag that defines the world coordinate frame (visible in every frame).
constexpr int kWorldTagId = 10;

/// Calibrated intrinsics of the camera that recorded plantage_shed.mp4.
struct CameraIntrinsics {
    int width = 1920;
    int height = 1080;
    double fx = 1153.24951171875;
    double fy = 1153.24951171875;
    double cx = 952.6605834960938;
    double cy = 526.54736328125;

    /// 3x3 camera matrix K.
    cv::Mat K() const;

    /// 14-coefficient rational + thin-prism distortion model
    /// (k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tx ty), as used by OpenCV.
    cv::Mat distCoeffs() const;
};

// ---------------------------------------------------------------------------
// AprilTag detection
// ---------------------------------------------------------------------------

/// One detected tag: id + 4 pixel corners in detector order.
struct TagDetection {
    int id = -1;
    std::array<cv::Point2f, 4> corners;
};

/// RAII wrapper around the AprilTag C library (tagStandard52h13).
class TagDetector {
public:
    TagDetector();
    ~TagDetector();
    TagDetector(const TagDetector&) = delete;
    TagDetector& operator=(const TagDetector&) = delete;

    /// Detect tags in an 8-bit grayscale image.
    std::vector<TagDetection> detect(const cv::Mat& gray) const;

private:
    apriltag_detector* td_ = nullptr;
    apriltag_family* tf_ = nullptr;
};

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Tag-local 3D corners on the z=0 plane, in detector corner order (mm).
/// The order matches cv::SOLVEPNP_IPPE_SQUARE:
///   (-s/2,+s/2) (+s/2,+s/2) (+s/2,-s/2) (-s/2,-s/2)
std::array<cv::Point3f, 4> tagLocalCorners();

/// Remove lens distortion, returning normalized image coordinates (K^-1 applied).
std::vector<cv::Point2f> undistortToNormalized(const std::vector<cv::Point2f>& px,
                                               const CameraIntrinsics& cam);

/// Remove lens distortion, returning pixel coordinates in the ideal
/// (distortion-free) camera with the same K.
std::vector<cv::Point2f> undistortToPixels(const std::vector<cv::Point2f>& px,
                                           const CameraIntrinsics& cam);

/// Estimate a single tag's pose in the camera frame (x_cam = R * x_tag + t)
/// with cv::SOLVEPNP_IPPE_SQUARE. Returns false when the solution is
/// ambiguous (the two IPPE solutions have similar reprojection error) or
/// inaccurate — callers use this to decide whether a tag observation is good
/// enough to be added to the map.
bool estimateSingleTagPose(const TagDetection& det, const CameraIntrinsics& cam,
                           cv::Matx33d* R_cam_tag, cv::Vec3d* t_cam_tag);

// ---------------------------------------------------------------------------
// Tag map (world = tag 10)
// ---------------------------------------------------------------------------

/// Map from tag id to its 4 corners in the world frame (mm).
class TagMap {
public:
    bool empty() const { return tags_.empty(); }
    bool has(int id) const { return tags_.count(id) > 0; }

    /// Place the world tag (id 10) at the origin.
    void bootstrap();

    /// Register a new tag given the camera pose in the world frame
    /// (x_world = R_wc * x_cam + t_wc) and the tag pose in the camera frame.
    void addTag(int id, const cv::Matx33d& R_wc, const cv::Vec3d& t_wc,
                const cv::Matx33d& R_cam_tag, const cv::Vec3d& t_cam_tag);

    /// Gather 2D-3D correspondences for all detections of mapped tags.
    void correspondences(const std::vector<TagDetection>& dets,
                         std::vector<cv::Point2f>* pts2d,
                         std::vector<cv::Point3f>* pts3d) const;

    const std::map<int, std::array<cv::Point3f, 4>>& tags() const { return tags_; }

private:
    std::map<int, std::array<cv::Point3f, 4>> tags_;
};

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Result of one method-specific PnP solve (world -> camera convention:
/// x_cam = R_cw * x_world + t_cw).
struct PnPSolverResult {
    bool ok = false;
    cv::Matx33d R_cw;
    cv::Vec3d t_cw;
    int num_inliers = 0;
};

/// Method-specific PnP solver: pixel corners (distorted, as detected) +
/// world corners + intrinsics -> camera pose.
using PnPSolver = std::function<PnPSolverResult(const std::vector<cv::Point2f>& pts2d,
                                                const std::vector<cv::Point3f>& pts3d,
                                                const CameraIntrinsics& cam)>;

/// Per-frame pipeline output.
struct FramePose {
    int frame = 0;
    bool valid = false;
    cv::Matx33d R_wc;              ///< camera-to-world rotation
    cv::Vec3d t_wc;                ///< camera position in the world (mm)
    double reproj_err_px = 0.0;    ///< mean reprojection error over used points
    double solve_ms = 0.0;         ///< PnP solve time only
    int num_tags = 0;              ///< tags detected in the frame
    int num_points = 0;            ///< 2D-3D correspondences fed to PnP
    int num_inliers = 0;           ///< inliers reported by the solver
    std::vector<TagDetection> detections;
};

/// Options shared by all demos (parsed from the command line).
struct PipelineOptions {
    std::string video = "../data/plantage_shed.mp4";
    std::string output;      ///< default: pnp_<method>.json
    int max_frames = 0;      ///< 0 = whole video
    bool verbose = false;

    /// Live rerun streaming (only when built with the rerun C++ SDK).
    /// On by default: each processed frame is sent to a rerun viewer
    /// (typically running on the host) while the pipeline runs.
    bool stream = true;
    std::string stream_url = "rerun+http://127.0.0.1:9876/proxy";
    /// All demos share one recording id so their trajectories land in the
    /// same viewer recording and can be compared in one 3D scene.
    std::string stream_recording = "plantage_shed";
    /// Stream the video frames + detection overlays (the heaviest part of
    /// the stream). They are identical for every method and logged to a
    /// shared entity, so when running several demos into one recording only
    /// the first one needs them.
    bool stream_images = true;
};

/// Parse the common command line. Returns false (after printing usage) on
/// --help or a malformed argument.
bool parseArgs(int argc, char** argv, const std::string& method, PipelineOptions* opts);

/// Run the full detect -> PnP -> map-extension pipeline and write the JSON.
/// Returns the process exit code.
int runPipeline(const std::string& method, const PipelineOptions& opts,
                const PnPSolver& solver);

}  // namespace atpnp
