/**
 * @file frontend_rerun.cpp
 * @brief The KLT front end on a whole EuRoC sequence, CPU and GPU side by side on one Rerun timeline
 *
 * This example demonstrates:
 * 1. The two front ends of tracker_benchmark (include/front_ends.hpp) on the same frames:
 *    - CPU: OpenCV's pyramidal KLT + vilib's CPU FAST (Rosten), on 1 thread by default
 *    - GPU: vilib's FeatureTrackerGPU + its CUDA FAST
 * 2. Every frame timed end to end, from the frame in host memory to its tracks on the host.
 *    On the GPU side that is the upload, the pyramid, the tracking (and detection when
 *    needed) and the cudaDeviceSynchronize() that waits for it: what a VIO pipeline on a
 *    desktop GPU would wait for, not only the GPU's work as in tracker_benchmark's table
 * 3. Both runs on one Rerun timeline, "frontend_time", that starts at 0 for both. Frame k
 *    is logged at the time its front end finished it: the sum of the times of frames 0..k.
 *    Playing that timeline replays the two runs as if they had started together, so the
 *    GPU view runs ahead and finishes first. The "frame" timeline shows the same frame k
 *    on both sides instead, to compare the tracks themselves
 *
 * The two front ends run one after the other, not at the same time: run together, they
 * would share the CPU (the GPU side needs a CPU thread to launch its kernels and wait for
 * them) and slow each other down. Nothing is logged while a front end runs either; the
 * results are kept in memory and logged afterwards, with the measured times, so logging
 * does not disturb the timing.
 *
 * With a viewer reachable (default rerun+http://127.0.0.1:9876/proxy, or $RERUN_URL) the
 * recording streams to it, and blueprint.py sends the layout. Without one, or with --save,
 * it is written to an .rrd file and the layout to an .rbl file next to it.
 */

#include "benchmark_common.hpp"
#include "front_ends.hpp"

#include <rerun.hpp>

#include <opencv2/imgcodecs.hpp>

#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef FRONTEND_BLUEPRINT
#define FRONTEND_BLUEPRINT "../blueprint.py"
#endif

constexpr char kAppId[] = "vilib_frontend";  // must match blueprint.py
constexpr size_t kTrailLength = 10;          // frames of history drawn behind each feature
constexpr int kJpegQuality = 80;             // the frames are logged as JPEG to keep the recording small
constexpr double kPlaybackSeconds = 20.0;    // the slower run plays back in at least this long

struct Options {
    cuda_bench::BenchmarkArgs args;
    int threads = 1;        // OpenCV threads for the CPU front end; 0 = all
    std::string save_path;  // write an .rrd file instead of streaming
};

Options parseOptions(int argc, char** argv) {
    Options options;
    options.args.frames = 0;  // 0 = to the end of the sequence
    options.args.start = 0;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            options.threads = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "--save" && i + 1 < argc) {
            options.save_path = argv[++i];
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " [cam0/data folder] [frames, 0 = all] [first frame] [--threads N] [--save PATH.rrd]"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() > 0) options.args.sequence_dir = positional[0];
    if (positional.size() > 1) options.args.frames = std::max(0, std::atoi(positional[1].c_str()));
    if (positional.size() > 2) options.args.start = std::max(0, std::atoi(positional[2].c_str()));
    if (options.args.frames == 0) options.args.frames = 1000000;  // loadSequence stops at the last frame
    return options;
}

// ---------------------------------------------------------------------------
// Running the front ends
// ---------------------------------------------------------------------------

struct FrameResult {
    double ms = 0.0;  // front-end time of this frame
    FrameStats stats;
    std::vector<Feature> features;
};

struct Run {
    std::string entity;  // "cpu" or "gpu"
    std::string label;   // console and summary
    std::string legend;  // plot legend, short so that it does not cover the plots
    rerun::Color color;
    std::vector<FrameResult> frames;

    double totalSeconds() const {
        double ms = 0.0;
        for (const auto& f : frames) ms += f.ms;
        return ms / 1000.0;
    }
};

double msSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

Run runCpu(const std::vector<cv::Mat>& frames, const std::vector<cv::Mat>& warmup, int threads) {
    const int all_threads = cv::getNumThreads();
    const int used = threads == 0 ? all_threads : threads;
    cv::setNumThreads(used);

    const std::string threads_text = std::to_string(used) + (used == 1 ? " thread" : " threads");
    Run run{"cpu", "CPU: OpenCV KLT + Rosten FAST, " + threads_text, "CPU, " + threads_text,
            rerun::Color(66, 133, 244), {}};
    OpenCvFrontEnd front_end(frames.front().cols, frames.front().rows);
    for (const auto& frame : warmup) front_end.track(frame);
    front_end.reset();

    run.frames.reserve(frames.size());
    for (const cv::Mat& image : frames) {
        FrameResult result;
        const auto start = std::chrono::steady_clock::now();
        result.stats = front_end.track(image);
        result.ms = msSince(start);
        result.features = front_end.features();
        run.frames.push_back(std::move(result));
    }
    cv::setNumThreads(all_threads);
    return run;
}

Run runGpu(const std::vector<cv::Mat>& frames, const std::vector<cv::Mat>& warmup) {
    Run run{"gpu", "GPU: vilib FeatureTrackerGPU + FASTGPU (with upload)", "GPU, with upload",
            rerun::Color(118, 185, 0), {}};
    VilibFrontEnd front_end(frames.front().cols, frames.front().rows);
    // Wait after every call: track() rewrites a mapped host buffer the previous call's kernels may still read
    for (const auto& frame : warmup) {
        front_end.track(front_end.makeFrame(frame));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    front_end.reset();

    run.frames.reserve(frames.size());
    for (const cv::Mat& image : frames) {
        FrameResult result;
        const auto start = std::chrono::steady_clock::now();
        const auto frame = front_end.makeFrame(image);  // upload + pyramid
        result.stats = front_end.track(frame);
        CUDA_CHECK(cudaDeviceSynchronize());  // track() ends with kernels that precompute the new templates
        result.ms = msSince(start);
        result.features = VilibFrontEnd::features(*frame);
        run.frames.push_back(std::move(result));
    }
    return run;
}

// ---------------------------------------------------------------------------
// Rerun
// ---------------------------------------------------------------------------

std::string viewerUrl() {
    const char* env_url = std::getenv("RERUN_URL");
    return env_url ? env_url : "rerun+http://127.0.0.1:9876/proxy";
}

/// True when a TCP server accepts connections at the URL's host:port. connect_grpc() never
/// fails on an absent viewer (it retries in the background), so reachability is probed first.
bool viewerReachable(const std::string& url) {
    const auto scheme = url.find("://");
    if (scheme == std::string::npos) return true;
    std::string hostport = url.substr(scheme + 3);
    hostport = hostport.substr(0, hostport.find('/'));
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

/// Sends the layout (two image views, three plots, the summary) with blueprint.py, or writes
/// it to an .rbl file. The rerun C++ SDK cannot author blueprints, the Python SDK can.
void sendBlueprint(const std::string& target_flag, const std::string& target, double playback_speed) {
    if (!std::filesystem::exists(FRONTEND_BLUEPRINT)) {
        std::cerr << "Note: " << FRONTEND_BLUEPRINT << " not found, the viewer picks its own layout." << std::endl;
        return;
    }
    std::ostringstream command;
    command << "python3 '" << FRONTEND_BLUEPRINT << "' " << target_flag << " '" << target << "' --playback-speed "
            << playback_speed;
    if (std::system(command.str().c_str()) != 0)
        std::cerr << "Note: blueprint.py failed, the viewer picks its own layout." << std::endl;
}

/// The fastest of 1, 0.5, 0.2, ..., 0.01x at which the slower run plays back in at least kPlaybackSeconds
double playbackSpeed(double slowest_seconds) {
    for (const double speed : {1.0, 0.5, 0.2, 0.1, 0.05, 0.02})
        if (slowest_seconds / speed >= kPlaybackSeconds) return speed;
    return 0.01;
}

/// A distinct, stable color per track id (golden-angle hue steps)
rerun::Color trackColor(int id) {
    const float h = std::fmod(static_cast<float>(id) * 137.508f, 360.f) / 60.f;
    const float s = 0.75f, v = 0.95f;
    const float c = v * s, x = c * (1.f - std::fabs(std::fmod(h, 2.f) - 1.f)), m = v - c;
    float r = 0, g = 0, b = 0;
    if (h < 1) r = c, g = x;
    else if (h < 2) r = x, g = c;
    else if (h < 3) g = c, b = x;
    else if (h < 4) g = x, b = c;
    else if (h < 5) r = x, b = c;
    else r = c, b = x;
    return rerun::Color(static_cast<uint8_t>((r + m) * 255), static_cast<uint8_t>((g + m) * 255),
                        static_cast<uint8_t>((b + m) * 255));
}

/// Logs one run frame by frame: the image, the features (new ones in yellow), their trails,
/// and the per-frame plots, each at the time the front end finished that frame
class RunLogger {
public:
    RunLogger(const rerun::RecordingStream& rec, const Run& run) : rec_(rec), run_(run) {
        const std::string e = run.entity;
        for (const char* plot : {"frame_time_ms", "frames_done", "tracks"})
            rec_.log_static(std::string(plot) + "/" + e, rerun::SeriesLines()
                                                             .with_names(rerun::components::Name(run.legend))
                                                             .with_colors(run.color)
                                                             .with_widths(2.f));
        // The common start: no frame done yet at frontend_time 0 (on that timeline only)
        rec_.reset_time();
        rec_.set_time_duration_secs("frontend_time", 0.0);
        rec_.log("frames_done/" + e, rerun::Scalars(0.0));
    }

    void log(size_t k, const rerun::EncodedImage& image) {
        const FrameResult& f = run_.frames[k];
        elapsed_ms_ += f.ms;
        rec_.set_time_sequence("frame", static_cast<int64_t>(k));
        rec_.set_time_duration_secs("frontend_time", elapsed_ms_ / 1000.0);

        const std::string e = run_.entity;
        rec_.log(e + "/image", image);

        std::vector<rerun::Position2D> tracked, fresh;
        std::vector<rerun::Color> tracked_colors;
        std::vector<rerun::components::LineStrip2D> trails;
        std::vector<rerun::Color> trail_colors;
        std::unordered_map<int, std::deque<rerun::datatypes::Vec2D>> next_trails;
        for (const Feature& p : f.features) {
            auto& trail = next_trails[p.id];
            const auto old = trails_.find(p.id);
            if (old != trails_.end()) trail = std::move(old->second);
            trail.push_back({p.x, p.y});
            if (trail.size() > kTrailLength) trail.pop_front();

            if (seen_.insert(p.id).second) {
                fresh.push_back({p.x, p.y});
            } else {
                tracked.push_back({p.x, p.y});
                tracked_colors.push_back(trackColor(p.id));
            }
            if (trail.size() > 1) {
                trails.emplace_back(std::vector<rerun::datatypes::Vec2D>(trail.begin(), trail.end()));
                trail_colors.push_back(trackColor(p.id));
            }
        }
        trails_ = std::move(next_trails);

        // rerun::Collection borrows lvalues: move the vectors in so the batches own them
        rec_.log(e + "/image/tracks", rerun::Points2D(std::move(tracked))
                                          .with_colors(std::move(tracked_colors))
                                          .with_radii(rerun::Radius::ui_points(2.5f)));
        rec_.log(e + "/image/new", rerun::Points2D(std::move(fresh))
                                       .with_colors(rerun::Color(255, 220, 0))
                                       .with_radii(rerun::Radius::ui_points(4.5f)));
        rec_.log(e + "/image/trails", rerun::LineStrips2D(std::move(trails))
                                          .with_colors(std::move(trail_colors))
                                          .with_radii(rerun::Radius::ui_points(1.f)));

        rec_.log("frame_time_ms/" + e, rerun::Scalars(f.ms));
        rec_.log("frames_done/" + e, rerun::Scalars(static_cast<double>(k + 1)));
        rec_.log("tracks/" + e, rerun::Scalars(static_cast<double>(f.features.size())));
    }

private:
    const rerun::RecordingStream& rec_;
    const Run& run_;
    double elapsed_ms_ = 0.0;
    std::unordered_set<int> seen_;
    std::unordered_map<int, std::deque<rerun::datatypes::Vec2D>> trails_;
};

void printRun(const Run& run, double reference_seconds) {
    const double seconds = run.totalSeconds();
    double tracked = 0.0;
    size_t detections = 0;
    for (const auto& f : run.frames) {
        tracked += f.stats.tracked;
        detections += f.stats.detected > 0;
    }
    std::cout << "  " << std::left << std::setw(56) << run.label << std::right << std::fixed << std::setprecision(3)
              << std::setw(8) << seconds << " s" << std::setw(9) << 1000.0 * seconds / run.frames.size() << " ms"
              << std::setprecision(1) << std::setw(9) << tracked / run.frames.size() << std::setw(8) << detections
              << std::setprecision(2) << std::setw(8) << reference_seconds / seconds << "x" << std::endl;
}

std::string summaryMarkdown(const Run& cpu, const Run& gpu, size_t frames) {
    std::ostringstream md;
    md << std::fixed << "**" << frames << " frames, both runs start at frontend_time = 0**\n\n";
    for (const Run* run : {&cpu, &gpu})
        md << "- " << run->label << ": " << std::setprecision(3) << run->totalSeconds() << " s, "
           << 1000.0 * run->totalSeconds() / frames << " ms per frame\n";
    md << "\n**GPU " << std::setprecision(2) << cpu.totalSeconds() / gpu.totalSeconds() << "x faster**\n\n"
       << "Each frame appears when its front end finished it. Yellow dots: features detected in that frame.\n\n"
       << "Switch the timeline to `frame` to compare the same frame on both sides.";
    return md.str();
}

int main(int argc, char** argv) {
    const Options options = parseOptions(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "KLT front end on EuRoC: CPU vs CUDA, one Rerun timeline" << std::endl;
    std::cout << "========================================" << std::endl;

    // The stream is set up before CUDA starts, so a missing viewer is reported at once
    const std::string url = viewerUrl();
    std::string save_path = options.save_path;
    rerun::RecordingStream rec(kAppId);
    const bool streaming = save_path.empty() && viewerReachable(url);
    if (streaming) {
        rec.connect_grpc(url).exit_on_failure();
        std::cout << "Streaming to the rerun viewer at " << url << std::endl;
    } else {
        if (save_path.empty()) {
            save_path = "frontend_rerun.rrd";
            std::cout << "No rerun viewer reachable at " << url << " (start one on the host with `rerun &`, and "
                      << "run the container with --network=host); writing a file instead." << std::endl;
        }
        rec.save(save_path).exit_on_failure();
    }
    rec.send_recording_name("EuRoC front end: CPU vs GPU");

    cuda_bench::initGpu();
    const std::vector<cv::Mat> frames = cuda_bench::loadSequence(options.args);
    const std::vector<cv::Mat> warmup(frames.begin(),
                                      frames.begin() + std::min<size_t>(options.args.warmup, frames.size()));
    std::cout << kPyramidLevels << " levels, " << kPatchSize << "x" << kPatchSize << " patch, up to " << kMaxFeatures
              << " features, detect below " << kMinTracks << " tracks" << std::endl;

    std::cout << "\nRunning the CPU front end..." << std::endl;
    const Run cpu = runCpu(frames, warmup, options.threads);
    std::cout << "Running the GPU front end..." << std::endl;
    const Run gpu = runGpu(frames, warmup);

    std::cout << "\n  " << std::left << std::setw(56) << "Front end" << std::right << std::setw(10) << "total"
              << std::setw(12) << "per frame" << std::setw(9) << "tracked" << std::setw(8) << "detect"
              << std::setw(9) << "vs CPU" << std::endl;
    printRun(cpu, cpu.totalSeconds());
    printRun(gpu, cpu.totalSeconds());
    std::cout << "  (per frame = mean; tracked = mean tracks carried over; detect = frames with detection)"
              << std::endl;

    // ---- Log both runs; frontend_time starts at 0 for both ----
    std::cout << "\nLogging " << frames.size() << " frames per run..." << std::endl;
    rec.log_static("summary", rerun::TextDocument(summaryMarkdown(cpu, gpu, frames.size()))
                                  .with_media_type(rerun::MediaType::markdown()));
    RunLogger cpu_logger(rec, cpu), gpu_logger(rec, gpu);

    // The layout selects the frontend_time timeline, which now exists in the recording
    const double speed = playbackSpeed(std::max(cpu.totalSeconds(), gpu.totalSeconds()));
    const std::string blueprint_path = std::filesystem::path(save_path).replace_extension(".rbl").string();
    rec.flush_blocking().handle();
    if (streaming)
        sendBlueprint("--connect-url", url, speed);
    else
        sendBlueprint("--save", blueprint_path, speed);

    std::vector<uint8_t> jpeg;
    for (size_t k = 0; k < frames.size(); ++k) {
        cv::imencode(".jpg", frames[k], jpeg, {cv::IMWRITE_JPEG_QUALITY, kJpegQuality});
        const auto image = rerun::EncodedImage::from_bytes(std::move(jpeg), rerun::MediaType::jpeg());
        jpeg = {};
        cpu_logger.log(k, image);
        gpu_logger.log(k, image);
    }
    rec.flush_blocking().handle();

    if (streaming)
        std::cout << "Done. In the viewer, play the frontend_time timeline from the start (playback speed "
                  << speed << "x)." << std::endl;
    else
        std::cout << "Saved " << save_path << ". Open it with: rerun " << save_path << " " << blueprint_path
                  << std::endl;
    return 0;
}
