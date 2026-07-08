/**
 * @file rerun_stream.hpp
 * @brief Live rerun streaming of the AprilTag + PnP pipeline.
 *
 * When the project is built with the rerun C++ SDK (CMake finds rerun_sdk,
 * ATPNP_HAVE_RERUN is defined), RerunStreamer sends each processed frame to a
 * running rerun viewer over gRPC while the pipeline executes:
 *
 *   - the camera pose (frustum) and the growing trajectory (3D),
 *   - the mapped AprilTag squares in the world frame (3D),
 *   - the undistorted video frame with tag outlines overlaid (2D),
 *   - reprojection error and PnP solve time (time-series plots).
 *
 * All demos default to the same recording id, so running pnp_opencv /
 * pnp_poselib / pnp_opengv one after another overlays the three methods in a
 * single viewer recording: trajectories, tag maps and plot series are logged
 * per method, while the camera, video frames and detections (identical for
 * every method) live on one shared entity — one image window in the viewer.
 * blueprint.py sends the matching viewer layout (one 3D view, one camera
 * view, one consolidated plot per metric).
 *
 * Without the SDK the class compiles to a no-op, so the demos build and run
 * unchanged (viz_pnp.py remains the offline visualization path).
 */

#pragma once

#include <memory>
#include <string>

#include "apriltag_pnp.hpp"

namespace atpnp {

class RerunStreamer {
public:
    /// Connects to the viewer at opts.stream_url (no-op when opts.stream is
    /// false or the demos were built without the rerun SDK).
    RerunStreamer(const std::string& method, const PipelineOptions& opts,
                  const CameraIntrinsics& cam);
    ~RerunStreamer();
    RerunStreamer(const RerunStreamer&) = delete;
    RerunStreamer& operator=(const RerunStreamer&) = delete;

    /// True when streaming is compiled in, enabled, and connected.
    bool active() const;

    /// Log one processed frame: camera pose + trajectory, current tag map,
    /// video frame with detections, and the per-frame metrics.
    void logFrame(const FramePose& fp, const cv::Mat& frame_bgr, const TagMap& map);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace atpnp
