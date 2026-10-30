/**
 * @file tracker_benchmark.cpp
 * @brief KLT feature tracking over EuRoC MH_01: OpenCV (CPU, 1 thread and all threads) vs vilib (CUDA)
 *
 * This example demonstrates:
 * 1. A VIO-style front end, run once per frame: track every feature into the new
 *    frame with pyramidal Lucas-Kanade (KLT), drop the ones that fail, and when
 *    fewer than kMinTracks survive, detect FAST corners in the grid cells that
 *    have no feature and add the strongest until there are kMaxFeatures
 * 2. The same front end twice:
 *    - CPU: cv::buildOpticalFlowPyramid + cv::calcOpticalFlowPyrLK, with vilib's
 *      CPU FAST (Rosten) for the detection, on 1 thread and on all of OpenCV's threads
 *    - CUDA: vilib's FeatureTrackerGPU with its CUDA FAST, with the frame already on
 *      the GPU, timed in two stages: the pyramid and the tracking (KLT + detection
 *      when needed). The upload, which a desktop GPU needs and a Jetson can avoid,
 *      is timed on its own
 * 3. A check of every step vilib's tracker took (a feature from frame k to frame k+1):
 *    redone with vilib's rules in plain C++ (klt_reference.hpp), it must give the
 *    same result
 *
 * Both sides use the same settings: 5 pyramid levels, a 16x16 patch on every level,
 * at most 30 iterations per level, stop when the update is below sqrt(0.0005) px.
 * Both use vilib's default template: the patch around the feature in the frame where
 * it was detected (as SVO does), not in the previous frame. KLT starts from the
 * feature's position in the previous frame and aligns that first patch, so errors
 * do not add up from frame to frame.
 *
 * How vilib's CUDA KLT works (feature_tracker_cuda_tools.cu):
 * - One warp (32 threads) tracks one feature; a 16x16 patch is 8 pixels per thread
 * - Inverse compositional Lucas-Kanade: the template patch and the inverse of its
 *   Hessian are computed once, when the feature is detected (a separate kernel), so
 *   an iteration only samples the current image, takes the template's gradients from
 *   the stored patch, and accumulates J^T r
 * - The 32 partial sums are added with __shfl_xor_sync (a butterfly), which leaves
 *   the total in every lane, so every thread computes the same update; no shared
 *   memory and no __syncthreads()
 * - The per-feature results sit in mapped (zero-copy) host memory, which the CPU
 *   reads directly after one cudaStreamSynchronize()
 */

#include "benchmark_common.hpp"
#include "front_ends.hpp"
#include "klt_reference.hpp"
#include "profile_summary.hpp"

#include <vilib/preprocess/pyramid.h>
#include <vilib/storage/subframe.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct TrackCounts {
    std::vector<FrameStats> per_frame;

    void print(const std::string& label) const {
        double tracked = 0.0, detected = 0.0;
        size_t detections = 0;
        for (const auto& s : per_frame) {
            tracked += s.tracked;
            detected += s.detected;
            detections += s.detected > 0;
        }
        std::cout << "  " << std::left << std::setw(12) << label << std::right << std::fixed << std::setprecision(1)
                  << std::setw(10) << tracked / per_frame.size() << std::setw(12) << detected / per_frame.size()
                  << std::setw(24) << detections << std::endl;
    }
};

int main(int argc, char** argv) {
    EASY_PROFILER_ENABLE;
    EASY_MAIN_THREAD;

    const cuda_bench::BenchmarkArgs args = cuda_bench::parseArgs(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "KLT feature tracking: CPU vs CUDA (easy_profiler)" << std::endl;
    std::cout << "========================================" << std::endl;

    cuda_bench::initGpu();

    const std::vector<cv::Mat> frames = cuda_bench::loadSequence(args);
    const int width = frames.front().cols, height = frames.front().rows;
    const int all_threads = cv::getNumThreads();
    std::cout << kPyramidLevels << " levels, " << kPatchSize << "x" << kPatchSize << " patch, up to "
              << kMaxFeatures << " features, detect below " << kMinTracks << " tracks" << std::endl;

    // A front end carries state from frame to frame, so the warm-up runs on the
    // first frames and the front end is reset before the timed pass from frame 0
    const std::vector<cv::Mat> warmup_frames(frames.begin(),
                                             frames.begin() + std::min<size_t>(args.warmup, frames.size()));

    // ---- CPU: OpenCV KLT, on 1 thread and on all threads ----
    OpenCvFrontEnd opencv(width, height);
    TrackCounts opencv_counts;
    const std::vector<int> thread_counts = all_threads > 1 ? std::vector<int>{1, all_threads} : std::vector<int>{1};
    for (const int threads : thread_counts) {
        cv::setNumThreads(threads);
        for (const auto& frame : warmup_frames) opencv.track(frame);
        opencv.reset();
        opencv_counts.per_frame.clear();
        cuda_bench::timeFrames("Tracker/OpenCV_" + std::to_string(threads) + "T", frames, 0,
                               [&](const cv::Mat& frame) { opencv_counts.per_frame.push_back(opencv.track(frame)); });
    }
    cv::setNumThreads(all_threads);

    // ---- CUDA: vilib ----
    VilibFrontEnd vilib_gpu(width, height);
    TrackCounts vilib_counts;
    // Wait after every call: track() rewrites a mapped host buffer the previous call's kernels may still read
    for (const auto& frame : warmup_frames) {
        vilib_gpu.track(vilib_gpu.makeFrame(frame));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    vilib_gpu.reset();

    // Each frame is uploaded and its pyramid built before its block, untimed (vilib's Frame does
    // both in its constructor). The timed block builds the pyramid again, which gives the same
    // pyramid, and then tracks: the GPU's work with the frame already on the GPU.
    std::shared_ptr<vilib::Frame> gpu_frame;
    cuda_bench::timeStages("Tracker/vilib_GPU", frames, 0,
                           {{"pyramid",
                             [&](const cv::Mat&) {
                                 vilib::pyramid_create_gpu(gpu_frame->pyramid_, 0);
                                 CUDA_CHECK(cudaDeviceSynchronize());
                             }},
                            {"track", [&](const cv::Mat&) {
                                 // track() ends with kernels that precompute the new templates; wait for them too
                                 vilib_counts.per_frame.push_back(vilib_gpu.track(gpu_frame));
                                 CUDA_CHECK(cudaDeviceSynchronize());
                                 gpu_frame.reset();
                             }}},
                           [&](const cv::Mat& image) {
                               gpu_frame = vilib_gpu.makeFrame(image);
                               CUDA_CHECK(cudaDeviceSynchronize());
                           });

    // The upload alone, the way vilib's Frame does it (a Jetson can avoid it)
    vilib::Subframe upload_target(width, height, 1, vilib::Subframe::MemoryType::PITCHED_DEVICE_MEMORY);
    cuda_bench::timeFrames("Tracker/upload", frames, args.warmup, [&](const cv::Mat& image) {
        upload_target.copy_from(image, true, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    const std::string profile_path = "tracker_benchmark.prof";
    profiler::dumpBlocksToFile(profile_path.c_str());

    std::cout << "\nFeatures per frame (mean)" << std::endl;
    std::cout << "  " << std::left << std::setw(12) << "" << std::right << std::setw(10) << "tracked" << std::setw(12)
              << "new" << std::setw(24) << "frames with detection" << std::endl;
    opencv_counts.print("OpenCV");
    vilib_counts.print("vilib_GPU");

    // ---- Verification (not profiled) ----
    // A step: a feature at vilib's position in frame k, aligned in frame k+1 against the patch
    // of the frame it was detected in. The reference gets the same template, start and frame.
    vilib_gpu.reset();
    struct FirstSeen {
        int frame;
        cv::Point2f px;
    };
    std::map<int, FirstSeen> first_seen;                // track id -> where vilib detected it
    std::map<int, std::vector<cv::Mat>> box_templates;  // template frame -> its pyramid
    std::map<int, cv::Point2f> previous;                // track id -> vilib's position in frame k
    double reference_max_diff = 0.0;
    size_t reference_steps = 0, reference_disagree = 0, vilib_dropped = 0;
    for (int k = 0; k < static_cast<int>(frames.size()); ++k) {
        std::vector<cv::Mat> box_pyramid;
        vilib::pyramid_create_cpu(frames[k], box_pyramid, kPyramidLevels, false);  // = vilib's GPU pyramid
        const auto frame = vilib_gpu.makeFrame(frames[k]);
        vilib_gpu.track(frame);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::map<int, cv::Point2f> current;
        for (size_t i = 0; i < frame->num_features_; ++i)
            current[frame->track_id_vec_[i]] = cv::Point2f(static_cast<float>(frame->px_vec_(0, i)),
                                                           static_cast<float>(frame->px_vec_(1, i)));

        // Tracks vilib dropped in this frame: the reference must drop them too
        for (const auto& [id, p] : previous) {
            if (current.count(id)) continue;
            const FirstSeen& seen = first_seen.at(id);
            cv::Point2f end = p;
            ++vilib_dropped;
            if (cuda_bench::kltReferenceStep(box_templates.at(seen.frame), seen.px, box_pyramid, end, kPatchSize,
                                             kMaxIterations, static_cast<float>(kMinUpdateSquared)))
                ++reference_disagree;
        }

        // Tracks vilib kept: the reference must keep them, at the same place
        bool detected_here = false;
        for (const auto& [id, p] : current) {
            const auto seen = first_seen.find(id);
            if (seen == first_seen.end()) {
                first_seen[id] = {k, p};
                detected_here = true;
                continue;
            }
            cv::Point2f end = previous.at(id);
            ++reference_steps;
            if (!cuda_bench::kltReferenceStep(box_templates.at(seen->second.frame), seen->second.px, box_pyramid, end,
                                              kPatchSize, kMaxIterations, static_cast<float>(kMinUpdateSquared)))
                ++reference_disagree;
            else
                reference_max_diff = std::max(reference_max_diff, cv::norm(end - p));
        }
        if (detected_here) box_templates[k] = box_pyramid;
        previous = std::move(current);
    }

    std::cout << "\nVerification: every step vilib took (a feature from frame k to k+1), redone on the CPU" << std::endl;
    std::cout << "  vilib's rules in plain C++ (klt_reference.hpp): " << reference_steps << " kept + " << vilib_dropped
              << " dropped, " << reference_disagree << " different keep/drop decisions, max |diff| "
              << std::scientific << std::setprecision(1) << reference_max_diff << " px" << std::fixed << std::endl;

    cuda_bench::printSummary(profile_path, cuda_bench::readProfile(profile_path));
    std::cout << "\nProfile saved to: " << profile_path << std::endl;
    return 0;
}
