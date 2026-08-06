#pragma once

/**
 * Live rerun streaming for the part3 chapter-1 optimizer-backend exercises.
 *
 * The five chapters (g2o / GTSAM / Ceres / SymForce / Kimera-RPGO) solve the
 * same problems with different libraries, so they log to the same recording
 * with the library name in the entity path. Run two chapters against one
 * viewer and their solutions overlay for comparison; run one and you just see
 * that one.
 *
 * This file is vendored byte-identical into every chapter that needs it: each
 * chapter's Docker build context is its own directory, so it cannot reach a
 * shared copy one level up. Keep the copies in sync - compare their md5sums
 * across the part3_ch01_1x/examples directories after editing.
 *
 * Everything degrades to a no-op when the rerun C++ SDK is absent (no
 * HAVE_RERUN) or when no viewer answers, so the demos always run and print
 * their numbers regardless.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef HAVE_RERUN
#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <optional>

#include <rerun.hpp>
#endif

namespace part3viz {

/// A planar pose: x [m], y [m], theta [rad].
using Pose2 = std::array<double, 3>;
/// A 3D position, also used for 3D pose translations.
using Vec3 = std::array<double, 3>;
/// Model parameters (a, b, c) of the shared curve-fitting exercise.
using Abc = std::array<double, 3>;
/// An RGB colour. Deliberately not rerun::Color: this type appears in the
/// public API below, which must also compile when the SDK is absent.
using Color3 = std::array<std::uint8_t, 3>;

/// How a pose-graph edge should be drawn.
enum class EdgeKind {
    Odometry = 0,      ///< sequential constraint (grey)
    Loop = 1,          ///< loop closure kept by the solver (blue)
    LoopRejected = 2,  ///< loop closure discarded as an outlier (red)
};

struct Edge {
    int i = 0;
    int j = 0;
    EdgeKind kind = EdgeKind::Odometry;
};

/// The model every chapter's curve-fitting exercise fits: y = exp(a x^2 + b x + c).
inline double curveModel(const Abc& abc, double x) {
    return std::exp(abc[0] * x * x + abc[1] * x + abc[2]);
}

/// Shared palette. These are the public names a chapter should use when it
/// picks a colour itself (only the 3D pose-graph call takes one). They are
/// plain RGB rather than rerun::Color so they exist with or without the SDK.
inline constexpr Color3 kGroundTruth{60, 190, 110};   // green
inline constexpr Color3 kInitial{150, 150, 150};      // grey
inline constexpr Color3 kOptimized{225, 70, 70};      // red
inline constexpr Color3 kObservation{120, 120, 120};  // dark grey
inline constexpr Color3 kLoop{40, 130, 240};          // blue
inline constexpr Color3 kRejected{225, 70, 70};       // red
inline constexpr Color3 kCamera{0, 120, 255};         // bright blue
inline constexpr Color3 kLandmark{80, 200, 120};      // light green

/// Recording names, shared across chapters so solutions overlay in one viewer.
inline const char* kCurveFittingRecording = "part3_curve_fitting";
inline const char* kPoseGraphRecording = "part3_pose_graph";
inline const char* kBundleAdjustmentRecording = "part3_bundle_adjustment";

#ifdef HAVE_RERUN

/// True when a TCP server accepts connections at the URL's host:port.
/// connect_grpc() itself never fails on an absent viewer - it just retries in
/// the background - so probe reachability first and fall back to running
/// without streaming rather than buffering data nobody will read.
inline bool viewerReachable(const std::string& url) {
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

/// Viewer address: RERUN_URL env var, or the default local viewer.
inline std::string viewerUrl() {
    const char* env_url = std::getenv("RERUN_URL");
    return env_url ? env_url : "rerun+http://127.0.0.1:9876/proxy";
}

namespace detail {

inline rerun::Color rr(const Color3& c) { return rerun::Color(c[0], c[1], c[2]); }

// Derived from the public palette above so there is one source of truth.
inline const rerun::Color kGtColor = rr(kGroundTruth);
inline const rerun::Color kInitColor = rr(kInitial);
inline const rerun::Color kOptColor = rr(kOptimized);
inline const rerun::Color kDataColor = rr(kObservation);
inline const rerun::Color kLoopColor = rr(kLoop);
inline const rerun::Color kRejectColor = rr(kRejected);
inline const rerun::Color kCamColor = rr(kCamera);
inline const rerun::Color kPointColor = rr(kLandmark);

inline std::vector<rerun::Vec2D> xy(const std::vector<Pose2>& poses) {
    std::vector<rerun::Vec2D> out;
    out.reserve(poses.size());
    for (const auto& p : poses) {
        out.push_back({static_cast<float>(p[0]), static_cast<float>(p[1])});
    }
    return out;
}

inline std::vector<rerun::Vec3D> xyz(const std::vector<Vec3>& pts) {
    std::vector<rerun::Vec3D> out;
    out.reserve(pts.size());
    for (const auto& p : pts) {
        out.push_back({static_cast<float>(p[0]), static_cast<float>(p[1]),
                       static_cast<float>(p[2])});
    }
    return out;
}

/// Sample the fitted curve densely enough to look smooth.
inline rerun::components::LineStrip2D curveStrip(const Abc& abc, double x_min,
                                                 double x_max, int samples = 200) {
    std::vector<rerun::Vec2D> pts;
    pts.reserve(static_cast<std::size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const double x = x_min + (x_max - x_min) * i / (samples - 1);
        pts.push_back({static_cast<float>(x), static_cast<float>(curveModel(abc, x))});
    }
    return rerun::components::LineStrip2D(pts);
}

}  // namespace detail

/**
 * One streaming connection, scoped to a single exercise.
 *
 * `lib` is the library name ("g2o", "gtsam", ...) and becomes the entity-path
 * segment that keeps two chapters' results apart in a shared recording.
 */
class Viz {
public:
    Viz(const std::string& recording, std::string lib) : lib_(std::move(lib)) {
        const std::string url = viewerUrl();
        if (!viewerReachable(url)) {
            std::cerr << "Note: no rerun viewer reachable at " << url
                      << " - running without live streaming.\n"
                         "      Start one on the host first (rerun &); with Docker "
                         "add --network=host.\n";
            return;
        }
        // A fixed recording id (not a fresh random one) is what lets a second
        // chapter's process append to the same recording and overlay.
        rec_.emplace(recording, recording);
        if (!rec_->connect_grpc(url).is_ok()) {
            rec_.reset();
            return;
        }
        connected_ = true;
        std::cout << "Streaming to rerun viewer at " << url << " as '" << lib_ << "'"
                  << std::endl;
    }

    bool active() const { return connected_; }

    // ---------------------------------------------------------------- curve fit

    /// Static context for the curve-fitting exercise: the samples, the true
    /// curve, and the initial guess the solver starts from.
    void curveSetup(const std::vector<double>& xs, const std::vector<double>& ys,
                    const Abc& gt, const Abc& init) {
        if (!connected_ || xs.empty()) return;
        std::vector<rerun::Vec2D> pts;
        pts.reserve(xs.size());
        double x_min = xs.front(), x_max = xs.front();
        for (std::size_t i = 0; i < xs.size(); ++i) {
            pts.push_back({static_cast<float>(xs[i]), static_cast<float>(ys[i])});
            x_min = std::min(x_min, xs[i]);
            x_max = std::max(x_max, xs[i]);
        }
        x_min_ = x_min;
        x_max_ = x_max;

        rec_->log_static("curve/observations",
                         rerun::Points2D(pts)
                             .with_colors({detail::kDataColor})
                             .with_radii({rerun::Radius::ui_points(2.0f)}));
        rec_->log_static("curve/ground_truth",
                         rerun::LineStrips2D({detail::curveStrip(gt, x_min, x_max)})
                             .with_colors({detail::kGtColor})
                             .with_radii({rerun::Radius::ui_points(2.0f)}));
        rec_->log_static("curve/" + lib_ + "/initial",
                         rerun::LineStrips2D({detail::curveStrip(init, x_min, x_max)})
                             .with_colors({detail::kInitColor})
                             .with_radii({rerun::Radius::ui_points(1.5f)}));

        rec_->log_static("cost/" + lib_, rerun::SeriesLines()
                                            .with_names({lib_})
                                            .with_colors({detail::kOptColor})
                                            .with_widths({1.5f}));
    }

    /// One solver iteration: the current curve plus the cost and parameters.
    void curveIteration(int64_t iter, const Abc& abc, double cost) {
        if (!connected_) return;
        rec_->set_time_sequence("iteration", iter);
        rec_->log("curve/" + lib_ + "/fitted",
                  rerun::LineStrips2D({detail::curveStrip(abc, x_min_, x_max_)})
                      .with_colors({detail::kOptColor})
                      .with_radii({rerun::Radius::ui_points(2.5f)}));
        rec_->log("cost/" + lib_, rerun::Scalars(cost));
        rec_->log("params/" + lib_ + "/a", rerun::Scalars(abc[0]));
        rec_->log("params/" + lib_ + "/b", rerun::Scalars(abc[1]));
        rec_->log("params/" + lib_ + "/c", rerun::Scalars(abc[2]));
    }

    // -------------------------------------------------------- 2D pose graph

    /// Static context for the 2D pose-graph exercise: ground truth, the noisy
    /// initial estimate, and the edges.
    ///
    /// Heading arrows are logged alongside the positions because the shared
    /// square-loop problem ends where it started: pose 4 sits exactly on pose 0
    /// and differs only in orientation, so the loop-closure constraint is
    /// invisible in a position-only plot.
    void poseGraphSetup(const std::vector<Pose2>& gt, const std::vector<Pose2>& init,
                        const std::vector<Edge>& edges) {
        if (!connected_ || gt.empty()) return;
        logPoses2D("graph/ground_truth", gt, detail::kGtColor, true);
        logPoses2D("graph/" + lib_ + "/initial", init, detail::kInitColor, true);
        logLoops2D("graph/ground_truth/loop_closures", gt, edges);
        rec_->log_static("cost/" + lib_, rerun::SeriesLines()
                                            .with_names({lib_})
                                            .with_colors({detail::kOptColor})
                                            .with_widths({1.5f}));
    }

    /// One solver iteration of the 2D pose graph.
    void poseGraphIteration(int64_t iter, const std::vector<Pose2>& poses, double cost,
                            const std::vector<Edge>& edges = {}) {
        if (!connected_ || poses.empty()) return;
        rec_->set_time_sequence("iteration", iter);
        logPoses2D("graph/" + lib_ + "/optimized", poses, detail::kOptColor, false);
        if (!edges.empty()) {
            logLoops2D("graph/" + lib_ + "/optimized/loop_closures", poses, edges,
                       false);
        }
        rec_->log("cost/" + lib_, rerun::Scalars(cost));
    }

    // -------------------------------------------------------- 3D pose graph

    /// A 3D pose graph, with edges coloured by kind. Used by the robust-PGO
    /// chapter, where showing which loop closures were rejected is the point.
    void poseGraph3D(const std::string& name, const std::vector<Vec3>& positions,
                     const std::vector<Edge>& edges, const Color3& rgb,
                     bool is_static = false) {
        if (!connected_ || positions.empty()) return;
        const std::string base = "graph3d/" + lib_ + "/" + name;
        const auto pts = detail::xyz(positions);
        const rerun::Color color(rgb[0], rgb[1], rgb[2]);

        auto points = rerun::Points3D(pts).with_colors({color}).with_radii(
            {rerun::Radius::ui_points(4.0f)});
        if (is_static) {
            rec_->log_static(base + "/poses", points);
        } else {
            rec_->log(base + "/poses", points);
        }

        // The trajectory itself, plus one strip group per edge kind so the
        // viewer can toggle rejected loop closures on their own.
        std::vector<rerun::components::LineStrip3D> odom, loops, rejected;
        for (const auto& e : edges) {
            if (e.i < 0 || e.j < 0 || e.i >= static_cast<int>(pts.size()) ||
                e.j >= static_cast<int>(pts.size())) {
                continue;
            }
            std::vector<rerun::Vec3D> seg{pts[e.i], pts[e.j]};
            switch (e.kind) {
                case EdgeKind::Odometry:
                    odom.emplace_back(seg);
                    break;
                case EdgeKind::Loop:
                    loops.emplace_back(seg);
                    break;
                case EdgeKind::LoopRejected:
                    rejected.emplace_back(seg);
                    break;
            }
        }
        logStrips3D(base + "/odometry", odom, color, 2.0f, is_static);
        logStrips3D(base + "/loop_closures", loops, detail::kLoopColor, 3.0f, is_static);
        logStrips3D(base + "/loop_closures_rejected", rejected, detail::kRejectColor,
                    3.0f, is_static);
    }

    // --------------------------------------------------- bundle adjustment

    /// Static reference cloud for bundle adjustment: where the landmarks started.
    void baSetup(const std::vector<Vec3>& initial_points) {
        if (!connected_) return;
        rec_->log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
        if (initial_points.empty()) return;
        rec_->log_static("world/initial_points",
                         rerun::Points3D(detail::xyz(initial_points))
                             .with_colors({detail::kDataColor})
                             .with_radii({rerun::Radius::ui_points(1.0f)}));
        rec_->log_static("reprojection_error/" + lib_,
                         rerun::SeriesLines()
                             .with_names({lib_})
                             .with_colors({detail::kOptColor})
                             .with_widths({1.5f}));
    }

    /// One bundle-adjustment iteration.
    ///
    /// `sq_error` must be the raw sum of squared reprojection error and
    /// `rmse_px` its per-observation RMS, so the number means the same thing in
    /// every chapter whether or not that chapter uses a robust kernel. Pass
    /// `robust_cost >= 0` to additionally graph the robustified objective the
    /// solver is actually minimizing.
    void baIteration(int64_t iter, const std::vector<Vec3>& points,
                     const std::vector<Vec3>& cameras, double sq_error, double rmse_px,
                     double robust_cost = -1.0) {
        if (!connected_) return;
        rec_->set_time_sequence("iteration", iter);
        if (!points.empty()) {
            rec_->log("world/" + lib_ + "/landmarks",
                      rerun::Points3D(detail::xyz(points))
                          .with_colors({detail::kPointColor})
                          .with_radii({rerun::Radius::ui_points(1.5f)}));
        }
        if (!cameras.empty()) {
            rec_->log("world/" + lib_ + "/cameras",
                      rerun::Points3D(detail::xyz(cameras))
                          .with_colors({detail::kCamColor})
                          .with_radii({rerun::Radius::ui_points(6.0f)}));
        }
        rec_->log("reprojection_error/" + lib_, rerun::Scalars(sq_error));
        rec_->log("rmse_px/" + lib_, rerun::Scalars(rmse_px));
        if (robust_cost >= 0.0) {
            rec_->log("robust_cost/" + lib_, rerun::Scalars(robust_cost));
        }
    }

private:
    void logPoses2D(const std::string& base, const std::vector<Pose2>& poses,
                    const rerun::Color& color, bool is_static) {
        const auto pts = detail::xy(poses);
        auto points = rerun::Points2D(pts).with_colors({color}).with_radii(
            {rerun::Radius::ui_points(4.0f)});
        auto path = rerun::LineStrips2D({rerun::components::LineStrip2D(pts)})
                        .with_colors({color})
                        .with_radii({rerun::Radius::ui_points(2.0f)});

        // Heading arrows, scaled to a fraction of the trajectory extent so they
        // stay readable whatever the problem size.
        float span = 0.0f;
        for (const auto& a : pts) {
            for (const auto& b : pts) {
                span = std::max(span, std::abs(a.x() - b.x()));
                span = std::max(span, std::abs(a.y() - b.y()));
            }
        }
        const float len = span > 0.0f ? 0.18f * span : 0.2f;
        std::vector<rerun::Vec2D> origins, vectors;
        origins.reserve(poses.size());
        vectors.reserve(poses.size());
        for (const auto& p : poses) {
            origins.push_back({static_cast<float>(p[0]), static_cast<float>(p[1])});
            vectors.push_back({len * static_cast<float>(std::cos(p[2])),
                               len * static_cast<float>(std::sin(p[2]))});
        }
        auto arrows = rerun::Arrows2D::from_vectors(vectors)
                          .with_origins(origins)
                          .with_colors({color});

        if (is_static) {
            rec_->log_static(base + "/poses", points);
            rec_->log_static(base + "/path", path);
            rec_->log_static(base + "/heading", arrows);
        } else {
            rec_->log(base + "/poses", points);
            rec_->log(base + "/path", path);
            rec_->log(base + "/heading", arrows);
        }
    }

    void logLoops2D(const std::string& path, const std::vector<Pose2>& poses,
                    const std::vector<Edge>& edges, bool is_static = true) {
        std::vector<rerun::components::LineStrip2D> strips;
        std::vector<rerun::Vec2D> markers;
        for (const auto& e : edges) {
            if (e.kind == EdgeKind::Odometry) continue;
            if (e.i < 0 || e.j < 0 || e.i >= static_cast<int>(poses.size()) ||
                e.j >= static_cast<int>(poses.size())) {
                continue;
            }
            const rerun::Vec2D a{static_cast<float>(poses[e.i][0]),
                                 static_cast<float>(poses[e.i][1])};
            const rerun::Vec2D b{static_cast<float>(poses[e.j][0]),
                                 static_cast<float>(poses[e.j][1])};
            strips.push_back(rerun::components::LineStrip2D(std::vector<rerun::Vec2D>{a, b}));
            // The square loop closes onto its own start, so the edge can be
            // zero-length; a marker keeps it visible either way.
            markers.push_back(a);
            markers.push_back(b);
        }
        if (markers.empty()) return;
        auto lines = rerun::LineStrips2D(strips)
                         .with_colors({detail::kLoopColor})
                         .with_radii({rerun::Radius::ui_points(2.0f)});
        auto dots = rerun::Points2D(markers)
                        .with_colors({detail::kLoopColor})
                        .with_radii({rerun::Radius::ui_points(7.0f)});
        if (is_static) {
            rec_->log_static(path, lines);
            rec_->log_static(path + "/endpoints", dots);
        } else {
            rec_->log(path, lines);
            rec_->log(path + "/endpoints", dots);
        }
    }

    void logStrips3D(const std::string& path,
                     const std::vector<rerun::components::LineStrip3D>& strips,
                     const rerun::Color& color, float width, bool is_static) {
        if (strips.empty()) return;
        auto lines = rerun::LineStrips3D(strips).with_colors({color}).with_radii(
            {rerun::Radius::ui_points(width)});
        if (is_static) {
            rec_->log_static(path, lines);
        } else {
            rec_->log(path, lines);
        }
    }

    std::string lib_;
    std::optional<rerun::RecordingStream> rec_;
    bool connected_ = false;
    double x_min_ = 0.0;
    double x_max_ = 1.0;
};

#else  // !HAVE_RERUN

/// No-op stand-in so the demos build and run without the rerun C++ SDK.
class Viz {
public:
    Viz(const std::string&, std::string) {
        std::cerr << "Note: built without the rerun C++ SDK - no visualization.\n";
    }
    bool active() const { return false; }
    void curveSetup(const std::vector<double>&, const std::vector<double>&, const Abc&,
                    const Abc&) {}
    void curveIteration(int64_t, const Abc&, double) {}
    void poseGraphSetup(const std::vector<Pose2>&, const std::vector<Pose2>&,
                        const std::vector<Edge>&) {}
    void poseGraphIteration(int64_t, const std::vector<Pose2>&, double,
                            const std::vector<Edge>& = {}) {}
    void poseGraph3D(const std::string&, const std::vector<Vec3>&,
                     const std::vector<Edge>&, const Color3&, bool = false) {}
    void baSetup(const std::vector<Vec3>&) {}
    void baIteration(int64_t, const std::vector<Vec3>&, const std::vector<Vec3>&, double,
                     double, double = -1.0) {}
};

#endif  // HAVE_RERUN

}  // namespace part3viz
