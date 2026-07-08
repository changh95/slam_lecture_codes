/**
 * @file apriltag_pnp.cpp
 * @brief Implementation of the shared AprilTag + PnP pipeline (see header).
 */

#include "apriltag_pnp.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

extern "C" {
#include <apriltag/apriltag.h>
#include <apriltag/tagStandard52h13.h>
}

namespace atpnp {

// ---------------------------------------------------------------------------
// CameraIntrinsics
// ---------------------------------------------------------------------------

cv::Mat CameraIntrinsics::K() const {
    return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

cv::Mat CameraIntrinsics::distCoeffs() const {
    // From the recording's calibration (cam.json of the original dataset).
    return (cv::Mat_<double>(14, 1) <<
        2.703267812728882, 7.070939540863037,
        0.00046289729652926326, 0.0009217377519235015,
        -1.177968978881836, 2.820115804672241,
        8.292820930480957, -0.521291196346283,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// TagDetector
// ---------------------------------------------------------------------------

TagDetector::TagDetector() {
    tf_ = tagStandard52h13_create();
    td_ = apriltag_detector_create();
    // maxhamming=1: tagStandard52h13 has ~48k codes, the default hamming-2
    // quick-decode table would need ~1 GB and a long startup.
    apriltag_detector_add_family_bits(td_, tf_, 1);
    td_->quad_decimate = 2.0;  // detect on a half-resolution image (1920x1080 input)
    td_->nthreads = 4;
}

TagDetector::~TagDetector() {
    if (td_) apriltag_detector_destroy(td_);
    if (tf_) tagStandard52h13_destroy(tf_);
}

std::vector<TagDetection> TagDetector::detect(const cv::Mat& gray) const {
    CV_Assert(gray.type() == CV_8UC1 && gray.isContinuous());

    image_u8_t im = {gray.cols, gray.rows, gray.cols, gray.data};

    zarray_t* dets = apriltag_detector_detect(td_, &im);

    std::vector<TagDetection> out;
    out.reserve(zarray_size(dets));
    for (int i = 0; i < zarray_size(dets); i++) {
        apriltag_detection_t* det;
        zarray_get(dets, i, &det);

        TagDetection d;
        d.id = det->id;
        for (int j = 0; j < 4; j++) {
            d.corners[j] = cv::Point2f(static_cast<float>(det->p[j][0]),
                                       static_cast<float>(det->p[j][1]));
        }
        out.push_back(d);
    }
    apriltag_detections_destroy(dets);
    return out;
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

std::array<cv::Point3f, 4> tagLocalCorners() {
    const float s = static_cast<float>(kTagSizeMm);
    // Order required by cv::SOLVEPNP_IPPE_SQUARE, matching the AprilTag
    // detector's corner order.
    return {cv::Point3f(-s / 2, s / 2, 0), cv::Point3f(s / 2, s / 2, 0),
            cv::Point3f(s / 2, -s / 2, 0), cv::Point3f(-s / 2, -s / 2, 0)};
}

std::vector<cv::Point2f> undistortToNormalized(const std::vector<cv::Point2f>& px,
                                               const CameraIntrinsics& cam) {
    std::vector<cv::Point2f> out;
    if (px.empty()) return out;
    cv::undistortPoints(px, out, cam.K(), cam.distCoeffs());
    return out;
}

std::vector<cv::Point2f> undistortToPixels(const std::vector<cv::Point2f>& px,
                                           const CameraIntrinsics& cam) {
    std::vector<cv::Point2f> out;
    if (px.empty()) return out;
    cv::undistortPoints(px, out, cam.K(), cam.distCoeffs(), cv::noArray(), cam.K());
    return out;
}

bool estimateSingleTagPose(const TagDetection& det, const CameraIntrinsics& cam,
                           cv::Matx33d* R_cam_tag, cv::Vec3d* t_cam_tag) {
    const auto local = tagLocalCorners();
    std::vector<cv::Point3f> obj(local.begin(), local.end());
    std::vector<cv::Point2f> img(det.corners.begin(), det.corners.end());

    // IPPE_SQUARE returns both planar-pose solutions with their reprojection
    // errors, so the flip ambiguity can be detected explicitly.
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat errs;
    int n = cv::solvePnPGeneric(obj, img, cam.K(), cam.distCoeffs(), rvecs, tvecs,
                                false, cv::SOLVEPNP_IPPE_SQUARE, cv::noArray(),
                                cv::noArray(), errs);
    if (n < 1 || rvecs.empty()) return false;

    const double e0 = errs.at<float>(0);
    const double e1 = (n > 1) ? errs.at<float>(1) : std::numeric_limits<double>::max();

    // Reject inaccurate or ambiguous observations: the wrong IPPE solution
    // would poison the map with a mirrored tag pose.
    if (e0 > 2.0) return false;
    if (e1 < 1.5 * e0) return false;

    cv::Mat R;
    cv::Rodrigues(rvecs[0], R);
    *R_cam_tag = cv::Matx33d(R);
    *t_cam_tag = cv::Vec3d(tvecs[0].at<double>(0), tvecs[0].at<double>(1),
                           tvecs[0].at<double>(2));
    return true;
}

// ---------------------------------------------------------------------------
// TagMap
// ---------------------------------------------------------------------------

void TagMap::bootstrap() {
    const auto local = tagLocalCorners();
    std::array<cv::Point3f, 4> corners;
    std::copy(local.begin(), local.end(), corners.begin());
    tags_[kWorldTagId] = corners;
}

void TagMap::addTag(int id, const cv::Matx33d& R_wc, const cv::Vec3d& t_wc,
                    const cv::Matx33d& R_cam_tag, const cv::Vec3d& t_cam_tag) {
    const auto local = tagLocalCorners();
    std::array<cv::Point3f, 4> corners;
    for (int j = 0; j < 4; j++) {
        const cv::Vec3d p_tag(local[j].x, local[j].y, local[j].z);
        const cv::Vec3d p_cam = R_cam_tag * p_tag + t_cam_tag;
        const cv::Vec3d p_world = R_wc * p_cam + t_wc;
        corners[j] = cv::Point3f(static_cast<float>(p_world[0]),
                                 static_cast<float>(p_world[1]),
                                 static_cast<float>(p_world[2]));
    }
    tags_[id] = corners;
}

void TagMap::correspondences(const std::vector<TagDetection>& dets,
                             std::vector<cv::Point2f>* pts2d,
                             std::vector<cv::Point3f>* pts3d) const {
    pts2d->clear();
    pts3d->clear();
    for (const auto& det : dets) {
        auto it = tags_.find(det.id);
        if (it == tags_.end()) continue;
        for (int j = 0; j < 4; j++) {
            pts2d->push_back(det.corners[j]);
            pts3d->push_back(it->second[j]);
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

namespace {

double meanReprojectionError(const std::vector<cv::Point2f>& pts2d,
                             const std::vector<cv::Point3f>& pts3d,
                             const CameraIntrinsics& cam,
                             const cv::Matx33d& R_cw, const cv::Vec3d& t_cw) {
    if (pts2d.empty()) return 0.0;
    cv::Mat rvec;
    cv::Rodrigues(cv::Mat(R_cw), rvec);
    std::vector<cv::Point2f> proj;
    cv::projectPoints(pts3d, rvec, cv::Mat(t_cw), cam.K(), cam.distCoeffs(), proj);

    double sum = 0.0;
    for (size_t i = 0; i < pts2d.size(); i++) sum += cv::norm(pts2d[i] - proj[i]);
    return sum / static_cast<double>(pts2d.size());
}

void writeMat44(std::ostream& os, const cv::Matx33d& R, const cv::Vec3d& t) {
    os << "[";
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) os << R(r, c) << ", ";
        os << t[r] << ", ";
    }
    os << "0.0, 0.0, 0.0, 1.0]";
}

bool writeResultsJson(const std::string& path, const std::string& method,
                      const std::string& video, const CameraIntrinsics& cam,
                      const TagMap& map, const std::vector<FramePose>& frames) {
    std::ofstream os(path);
    if (!os.is_open()) return false;
    os << std::setprecision(10);

    os << "{\n";
    os << "  \"method\": \"" << method << "\",\n";
    os << "  \"video\": \"" << video << "\",\n";
    os << "  \"tag_size_mm\": " << kTagSizeMm << ",\n";
    os << "  \"world_tag_id\": " << kWorldTagId << ",\n";
    os << "  \"camera\": {\"width\": " << cam.width << ", \"height\": " << cam.height
       << ", \"fx\": " << cam.fx << ", \"fy\": " << cam.fy
       << ", \"cx\": " << cam.cx << ", \"cy\": " << cam.cy
       << ", \"dist_coeffs\": [";
    const cv::Mat dist = cam.distCoeffs();
    for (int i = 0; i < dist.rows; i++) {
        os << dist.at<double>(i);
        if (i + 1 < dist.rows) os << ", ";
    }
    os << "]},\n";

    // Tag map: world-frame corner positions (mm).
    os << "  \"tag_map\": [\n";
    bool first = true;
    for (const auto& [id, corners] : map.tags()) {
        if (!first) os << ",\n";
        first = false;
        os << "    {\"id\": " << id << ", \"corners\": [";
        for (int j = 0; j < 4; j++) {
            os << "[" << corners[j].x << ", " << corners[j].y << ", " << corners[j].z << "]";
            if (j < 3) os << ", ";
        }
        os << "]}";
    }
    os << "\n  ],\n";

    // Per-frame camera poses (T_wc: camera-to-world, mm) and detections.
    os << "  \"frames\": [\n";
    first = true;
    for (const auto& f : frames) {
        if (!first) os << ",\n";
        first = false;
        os << "    {\"frame\": " << f.frame << ", \"valid\": " << (f.valid ? "true" : "false")
           << ", \"num_tags\": " << f.num_tags << ", \"num_points\": " << f.num_points
           << ", \"num_inliers\": " << f.num_inliers
           << ", \"reproj_err_px\": " << f.reproj_err_px
           << ", \"solve_ms\": " << f.solve_ms;
        if (f.valid) {
            os << ", \"T_wc\": ";
            writeMat44(os, f.R_wc, f.t_wc);
        }
        os << ", \"detections\": [";
        for (size_t i = 0; i < f.detections.size(); i++) {
            const auto& d = f.detections[i];
            os << "{\"id\": " << d.id << ", \"px\": [";
            for (int j = 0; j < 4; j++) {
                os << "[" << d.corners[j].x << ", " << d.corners[j].y << "]";
                if (j < 3) os << ", ";
            }
            os << "]}";
            if (i + 1 < f.detections.size()) os << ", ";
        }
        os << "]}";
    }
    os << "\n  ]\n}\n";
    return true;
}

}  // namespace

bool parseArgs(int argc, char** argv, const std::string& method, PipelineOptions* opts) {
    opts->output = "pnp_" + method + ".json";

    auto usage = [&]() {
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "\nAprilTag detection + " << method
                  << " PnP camera localization on plantage_shed.mp4\n"
                  << "\nOptions:\n"
                  << "  --video <path>       Input video (default: " << opts->video << ")\n"
                  << "  --output <path>      Output JSON (default: " << opts->output << ")\n"
                  << "  --max-frames <n>     Stop after n frames (default: whole video)\n"
                  << "  --verbose            Per-frame console output\n";
    };

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage();
            return false;
        } else if (arg == "--video" && i + 1 < argc) {
            opts->video = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            opts->output = argv[++i];
        } else if (arg == "--max-frames" && i + 1 < argc) {
            opts->max_frames = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            opts->verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage();
            return false;
        }
    }
    return true;
}

int runPipeline(const std::string& method, const PipelineOptions& opts,
                const PnPSolver& solver) {
    const CameraIntrinsics cam;

    cv::VideoCapture cap(opts.video);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video " << opts.video << "\n";
        return 1;
    }

    std::cout << "=== AprilTag + PnP camera localization [" << method << "] ===\n"
              << "Video: " << opts.video << " (" << cam.width << "x" << cam.height << ")\n"
              << "Tag family: tagStandard52h13, size " << kTagSizeMm << " mm, world tag "
              << kWorldTagId << "\n\n";

    TagDetector detector;
    TagMap map;
    std::vector<FramePose> frames;

    cv::Mat frame, gray;
    int frameIdx = 0;
    int validFrames = 0;
    double sumReproj = 0.0, sumSolveMs = 0.0;

    while (cap.read(frame)) {
        if (opts.max_frames > 0 && frameIdx >= opts.max_frames) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        FramePose fp;
        fp.frame = frameIdx;
        fp.detections = detector.detect(gray);
        fp.num_tags = static_cast<int>(fp.detections.size());

        // Bootstrap the world frame from tag 10 the first time it is seen.
        if (map.empty()) {
            for (const auto& d : fp.detections) {
                if (d.id == kWorldTagId) {
                    map.bootstrap();
                    break;
                }
            }
        }

        if (!map.empty()) {
            std::vector<cv::Point2f> pts2d;
            std::vector<cv::Point3f> pts3d;
            map.correspondences(fp.detections, &pts2d, &pts3d);
            fp.num_points = static_cast<int>(pts2d.size());

            if (fp.num_points >= 4) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                const PnPSolverResult res = solver(pts2d, pts3d, cam);
                const auto t1 = std::chrono::high_resolution_clock::now();
                fp.solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                if (res.ok) {
                    fp.valid = true;
                    fp.num_inliers = res.num_inliers;
                    // Invert world->camera to get the camera pose in the world.
                    fp.R_wc = res.R_cw.t();
                    fp.t_wc = -(res.R_cw.t() * res.t_cw);
                    fp.reproj_err_px =
                        meanReprojectionError(pts2d, pts3d, cam, res.R_cw, res.t_cw);

                    // Extend the map from a trustworthy pose only.
                    if (fp.reproj_err_px < 2.0) {
                        for (const auto& d : fp.detections) {
                            if (map.has(d.id)) continue;
                            cv::Matx33d R_ct;
                            cv::Vec3d t_ct;
                            if (estimateSingleTagPose(d, cam, &R_ct, &t_ct)) {
                                map.addTag(d.id, fp.R_wc, fp.t_wc, R_ct, t_ct);
                                if (opts.verbose) {
                                    std::cout << "  [frame " << frameIdx << "] added tag "
                                              << d.id << " to map\n";
                                }
                            }
                        }
                    }
                }
            }
        }

        if (fp.valid) {
            validFrames++;
            sumReproj += fp.reproj_err_px;
            sumSolveMs += fp.solve_ms;
            if (opts.verbose) {
                std::cout << "frame " << frameIdx << ": tags=" << fp.num_tags
                          << " inliers=" << fp.num_inliers << "/" << fp.num_points
                          << " reproj=" << std::fixed << std::setprecision(3)
                          << fp.reproj_err_px << "px cam=[" << std::setprecision(1)
                          << fp.t_wc[0] << ", " << fp.t_wc[1] << ", " << fp.t_wc[2]
                          << "] mm\n";
            }
        }

        frames.push_back(std::move(fp));
        frameIdx++;
        if (frameIdx % 200 == 0) {
            std::cout << "Processed " << frameIdx << " frames (" << validFrames
                      << " with pose)...\n";
        }
    }

    std::cout << "\n=== Summary [" << method << "] ===\n"
              << "Frames processed:    " << frameIdx << "\n"
              << "Frames with pose:    " << validFrames << "\n"
              << "Tags mapped:         " << map.tags().size() << "\n";
    if (validFrames > 0) {
        std::cout << std::fixed << std::setprecision(3)
                  << "Mean reproj error:   " << sumReproj / validFrames << " px\n"
                  << "Mean PnP solve time: " << sumSolveMs / validFrames << " ms\n";
    }

    // Sanity check: known ground-truth distances between tag centers
    // (from the dataset description: 2-3 = 1090 mm, 3-39 = 1940 mm).
    auto center = [&](int id) -> cv::Point3f {
        const auto& c = map.tags().at(id);
        return (c[0] + c[1] + c[2] + c[3]) / 4.0f;
    };
    if (map.has(2) && map.has(3)) {
        std::cout << "Tag 2-3 center dist:  " << cv::norm(center(2) - center(3))
                  << " mm (ground truth 1090 mm)\n";
    }
    if (map.has(3) && map.has(39)) {
        std::cout << "Tag 3-39 center dist: " << cv::norm(center(3) - center(39))
                  << " mm (ground truth 1940 mm)\n";
    }

    if (!writeResultsJson(opts.output, method, opts.video, cam, map, frames)) {
        std::cerr << "Error: cannot write " << opts.output << "\n";
        return 1;
    }
    std::cout << "\nResults written to " << opts.output << "\n"
              << "Visualize with: python3 ../viz_pnp.py " << opts.output << "\n";
    return 0;
}

}  // namespace atpnp
