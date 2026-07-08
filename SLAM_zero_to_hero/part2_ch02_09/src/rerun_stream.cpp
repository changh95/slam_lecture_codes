/**
 * @file rerun_stream.cpp
 * @brief Live rerun streaming implementation (see rerun_stream.hpp).
 */

#include "rerun_stream.hpp"

#include <iostream>

#ifndef ATPNP_HAVE_RERUN

// Built without the rerun C++ SDK: every call is a no-op.
namespace atpnp {

struct RerunStreamer::Impl {};

RerunStreamer::RerunStreamer(const std::string&, const PipelineOptions& opts,
                             const CameraIntrinsics&) {
    if (opts.stream) {
        std::cout << "Note: built without the rerun C++ SDK - live streaming "
                     "disabled (visualize afterwards with viz_pnp.py)\n";
    }
}
RerunStreamer::~RerunStreamer() = default;
bool RerunStreamer::active() const { return false; }
void RerunStreamer::logFrame(const FramePose&, const cv::Mat&, const TagMap&) {}

}  // namespace atpnp

#else  // ATPNP_HAVE_RERUN

#include <cerrno>
#include <vector>

#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <rerun.hpp>

namespace atpnp {

namespace {

constexpr double kMmToM = 1.0e-3;
/// Streamed images are downscaled to keep the gRPC bandwidth low
/// (matches viz_pnp.py's --img-scale default).
constexpr double kImgScale = 0.5;

/// Same color per method as viz_pnp.py.
rerun::Color methodColor(const std::string& method) {
    if (method == "opencv") return rerun::Color(0, 200, 0);
    if (method == "poselib") return rerun::Color(0, 120, 255);
    if (method == "opengv") return rerun::Color(220, 60, 60);
    return rerun::Color(200, 200, 0);
}

/// The camera, video frames and tag detections are identical for every
/// method, so they live on one shared entity (one image window in the
/// viewer) with a method-neutral color.
constexpr const char* kCameraEntity = "world/camera";
constexpr const char* kImageEntity = "world/camera/image";
const rerun::Color kDetectionColor(255, 200, 0);

/// True when a TCP server accepts connections at the URL's host:port.
/// connect_grpc() itself never fails on an absent viewer - it just retries
/// and stalls in the background - so the demos probe reachability first and
/// fall back to running without streaming.
bool viewerReachable(const std::string& url) {
    // Expect rerun+http://HOST:PORT/... - fall back to trying when unparsable.
    const auto scheme = url.find("://");
    if (scheme == std::string::npos) return true;
    std::string hostport = url.substr(scheme + 3);
    const auto slash = hostport.find('/');
    if (slash != std::string::npos) hostport = hostport.substr(0, slash);
    const auto colon = hostport.rfind(':');
    if (colon == std::string::npos) return true;
    const std::string host = hostport.substr(0, colon);
    const std::string port = hostport.substr(colon + 1);

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) return false;

    bool ok = false;
    for (addrinfo* ai = res; ai && !ok; ai = ai->ai_next) {
        const int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            ok = true;
        } else if (errno == EINPROGRESS) {
            fd_set wfds;
            FD_ZERO(&wfds);
            FD_SET(fd, &wfds);
            timeval tv{2, 0};  // 2 s connect timeout
            if (select(fd + 1, nullptr, &wfds, nullptr, &tv) == 1) {
                int err = 0;
                socklen_t len = sizeof(err);
                getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
                ok = (err == 0);
            }
        }
        close(fd);
    }
    freeaddrinfo(res);
    return ok;
}

/// cv row-major 3x3 -> rerun column-major Mat3x3.
rerun::datatypes::Mat3x3 toMat3x3(const cv::Matx33d& R) {
    return rerun::datatypes::Mat3x3(std::array<float, 9>{
        static_cast<float>(R(0, 0)), static_cast<float>(R(1, 0)), static_cast<float>(R(2, 0)),
        static_cast<float>(R(0, 1)), static_cast<float>(R(1, 1)), static_cast<float>(R(2, 1)),
        static_cast<float>(R(0, 2)), static_cast<float>(R(1, 2)), static_cast<float>(R(2, 2))});
}

}  // namespace

struct RerunStreamer::Impl {
    rerun::RecordingStream rec;
    std::string method;
    CameraIntrinsics cam;
    rerun::Color color;
    bool connected = false;
    bool stream_images = true;

    cv::Mat undist_map1, undist_map2;    ///< full-res -> scaled undistorted image
    std::vector<rerun::Vec3D> traj;      ///< camera positions so far (m)
    size_t tags_logged = 0;              ///< tag-map size at the last map log

    Impl(const std::string& method_, const PipelineOptions& opts,
         const CameraIntrinsics& cam_)
        : rec("apriltag_pnp", opts.stream_recording), method(method_), cam(cam_),
          color(methodColor(method_)) {
        if (!viewerReachable(opts.stream_url)) {
            std::cerr << "Warning: no rerun viewer reachable at " << opts.stream_url
                      << " - running without live streaming.\n"
                         "         Start one on the host first (rerun &), or pass "
                         "--no-stream to silence this warning.\n";
            return;
        }
        const rerun::Error err = rec.connect_grpc(opts.stream_url);
        if (!err.is_ok()) {
            std::cerr << "Warning: rerun streaming disabled - cannot connect to "
                      << opts.stream_url << " (" << err.description << ")\n";
            return;
        }
        connected = true;
        stream_images = opts.stream_images;

        // Undistortion to a downscaled ideal pinhole image: streamed frames
        // and corner overlays must match the (distortion-free) frustum model.
        const cv::Size out_size(static_cast<int>(cam.width * kImgScale),
                                static_cast<int>(cam.height * kImgScale));
        if (stream_images) {
            const cv::Mat K = cam.K();
            cv::Mat K_scaled = K * kImgScale;
            K_scaled.at<double>(2, 2) = 1.0;
            cv::initUndistortRectifyMap(K, cam.distCoeffs(), cv::Mat(), K_scaled,
                                        out_size, CV_16SC2, undist_map1, undist_map2);
        }

        // Static scene: world axes (0.2 m), the pinhole camera model, and the
        // line style of this method's plot series.
        rec.log_static(
            "world",
            rerun::Arrows3D::from_vectors({{0.2f, 0.0f, 0.0f},
                                           {0.0f, 0.2f, 0.0f},
                                           {0.0f, 0.0f, 0.2f}})
                .with_colors({rerun::Color(255, 0, 0), rerun::Color(0, 255, 0),
                              rerun::Color(0, 0, 255)})
                .with_labels({"x", "y", "z"}));

        rec.log_static(
            kCameraEntity,
            rerun::Pinhole::from_focal_length_and_resolution(
                {static_cast<float>(cam.fx * kImgScale),
                 static_cast<float>(cam.fy * kImgScale)},
                {static_cast<float>(out_size.width), static_cast<float>(out_size.height)})
                .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
                .with_image_plane_distance(0.15f));

        for (const char* plot : {"plots/reproj_err_px/", "plots/solve_ms/"}) {
            rec.log_static(plot + method,
                           rerun::SeriesLines()
                               .with_colors({color})
                               .with_names({method})
                               .with_widths({1.5f}));
        }
    }
};

RerunStreamer::RerunStreamer(const std::string& method, const PipelineOptions& opts,
                             const CameraIntrinsics& cam) {
    if (!opts.stream) return;
    impl_ = std::make_unique<Impl>(method, opts, cam);
    if (impl_->connected) {
        std::cout << "Streaming live to rerun viewer at " << opts.stream_url
                  << " (recording \"" << opts.stream_recording
                  << "\"; disable with --no-stream)\n";
    } else {
        impl_.reset();
    }
}

RerunStreamer::~RerunStreamer() = default;

bool RerunStreamer::active() const { return impl_ != nullptr; }

void RerunStreamer::logFrame(const FramePose& fp, const cv::Mat& frame_bgr,
                             const TagMap& map) {
    if (!impl_) return;
    auto& s = *impl_;
    s.rec.set_time_sequence("frame", fp.frame);

    // Camera pose + trajectory + metrics (valid PnP solves only).
    if (fp.valid) {
        const rerun::Vec3D pos{static_cast<float>(fp.t_wc[0] * kMmToM),
                               static_cast<float>(fp.t_wc[1] * kMmToM),
                               static_cast<float>(fp.t_wc[2] * kMmToM)};
        s.rec.log(kCameraEntity,
                  rerun::Transform3D::from_translation(pos).with_mat3x3(toMat3x3(fp.R_wc)));

        s.traj.push_back(pos);
        if (s.traj.size() >= 2) {
            s.rec.log("world/" + s.method + "/trajectory",
                      rerun::LineStrips3D(
                          rerun::components::LineStrip3D(s.traj))
                          .with_colors({s.color}));
        }

        s.rec.log("plots/reproj_err_px/" + s.method, rerun::Scalars(fp.reproj_err_px));
        s.rec.log("plots/solve_ms/" + s.method, rerun::Scalars(fp.solve_ms));
    }

    // Tag map: re-log whenever a tag was added (shows the map growing).
    if (map.tags().size() != s.tags_logged) {
        s.tags_logged = map.tags().size();
        std::vector<rerun::components::LineStrip3D> strips;
        std::vector<rerun::Vec3D> centers;
        std::vector<std::string> labels;
        for (const auto& [id, corners] : map.tags()) {
            std::vector<rerun::Vec3D> pts;
            cv::Point3f c(0, 0, 0);
            for (int j = 0; j < 5; j++) {
                const auto& p = corners[j % 4];
                pts.push_back({static_cast<float>(p.x * kMmToM),
                               static_cast<float>(p.y * kMmToM),
                               static_cast<float>(p.z * kMmToM)});
                if (j < 4) c += corners[j];
            }
            // std::move so the component takes ownership: rerun Collections
            // only borrow lvalues, and pts dies before rec.log() serializes.
            strips.emplace_back(std::move(pts));
            c *= 0.25f;
            centers.push_back({static_cast<float>(c.x * kMmToM),
                               static_cast<float>(c.y * kMmToM),
                               static_cast<float>(c.z * kMmToM)});
            labels.push_back("tag " + std::to_string(id));
        }
        s.rec.log("world/tags/" + s.method,
                  rerun::LineStrips3D(strips).with_colors({s.color}));
        s.rec.log("world/tags/" + s.method + "/ids",
                  rerun::Points3D(centers)
                      .with_radii({0.004f})
                      .with_colors({s.color})
                      .with_labels(labels));
    }

    if (!s.stream_images) return;

    // Undistorted, downscaled video frame (JPEG over the wire).
    cv::Mat shown;
    cv::remap(frame_bgr, shown, s.undist_map1, s.undist_map2, cv::INTER_LINEAR);
    std::vector<uchar> jpeg;
    cv::imencode(".jpg", shown, jpeg, {cv::IMWRITE_JPEG_QUALITY, 75});
    s.rec.log(kImageEntity, rerun::EncodedImage::from_bytes(jpeg));

    // Tag outlines + ids in the undistorted image.
    const std::string det_entity = std::string(kImageEntity) + "/detections";
    if (!fp.detections.empty()) {
        std::vector<rerun::components::LineStrip2D> strips;
        std::vector<std::string> labels;
        for (const auto& det : fp.detections) {
            const std::vector<cv::Point2f> px(det.corners.begin(), det.corners.end());
            const std::vector<cv::Point2f> und = undistortToPixels(px, s.cam);
            std::vector<rerun::Vec2D> pts;
            for (int j = 0; j < 5; j++) {
                const auto& p = und[j % 4];
                pts.push_back({static_cast<float>(p.x * kImgScale),
                               static_cast<float>(p.y * kImgScale)});
            }
            // std::move: take ownership, pts dies before rec.log() (see above).
            strips.emplace_back(std::move(pts));
            labels.push_back("id " + std::to_string(det.id));
        }
        s.rec.log(det_entity,
                  rerun::LineStrips2D(strips)
                      .with_colors({kDetectionColor})
                      .with_labels(labels));
    } else {
        s.rec.log(det_entity, rerun::Clear::FLAT);
    }
}

}  // namespace atpnp

#endif  // ATPNP_HAVE_RERUN
