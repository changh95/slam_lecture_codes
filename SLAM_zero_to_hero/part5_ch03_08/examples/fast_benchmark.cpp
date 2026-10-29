/**
 * @file fast_benchmark.cpp
 * @brief FAST-9 corners with one corner per 32x32 cell: vilib CPU (Rosten) and OpenCV vs vilib CUDA
 *
 * This example demonstrates:
 * 1. A VIO-style detector: FAST-9 corners, 3x3 non-maximum suppression, then the
 *    strongest corner of every 32x32 grid cell (at most 24 x 15 = 360 corners)
 * 2. The same detector three ways: Rosten's FAST C code and cv::FAST on the CPU
 *    (single thread, both wrapped by vilib with the same grid), and vilib's CUDA kernels
 * 3. The CUDA detection with the frame already on the GPU: corner response kernel,
 *    grid NMS kernel, and the copy of the per-cell results back to the host (the
 *    kernels alone cannot be called: vilib keeps them private). The upload of the
 *    frame, which a desktop GPU needs and a Jetson can avoid, is timed on its own.
 *
 * How vilib's CUDA FAST works (fast_gpu_cuda_tools.cu, detector_base_gpu_cuda_tools.cu):
 * - Corner response: one thread per pixel. A pre-check on two opposite pairs of circle
 *   pixels rejects most pixels early: if both pixels of a pair are within the threshold
 *   of the centre, no 9-pixel arc fits. For the rest, the 16 circle pixels give a 16-bit
 *   "darker" and a 16-bit "brighter" mask, and a precomputed table of 65536 bits (one per
 *   mask) says whether the mask holds 9 contiguous ones. No loop over arc start points,
 *   so the threads of a warp do not diverge. The score goes into a float response image
 * - Grid NMS: one thread block per 32x32 cell. Each thread suppresses its pixels
 *   against their 3x3 neighbours with sign arithmetic instead of branches, then warp
 *   shuffles (__shfl_down_sync) and shared memory reduce the cell to its best corner
 * - Only the per-cell results (position, score, level) travel back to the CPU, not
 *   the response image
 */

#include "benchmark_common.hpp"
#include "profile_summary.hpp"

#include <vilib/feature_detection/fast/fast_gpu.h>
#include <vilib/feature_detection/fast/opencv/fast_cpu.h>
#include <vilib/feature_detection/fast/rosten/fast_cpu.h>
#include <vilib/storage/pyramid_pool.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

constexpr float kThreshold = 10.f;  // FAST intensity threshold (vilib's and SVO's default)
constexpr int kArcLength = 9;       // FAST-9: 9 contiguous circle pixels, as cv::FAST's default TYPE_9_16
constexpr int kCellSize = 32;       // grid NMS: at most one corner per 32x32 cell
constexpr int kBorder = 8;          // no corners closer than 8 px to the image edge

/**
 * @brief Per-cell agreement of two grid detectors, summed over all frames
 */
struct GridAgreement {
    size_t same = 0, moved = 0, moved_tied = 0, only_ref = 0, only_other = 0;

    template <typename Ref, typename Other>
    void add(Ref& ref, Other& other) {
        const auto& ref_points = ref.getPoints();
        const auto& other_points = other.getPoints();
        for (size_t i = 0; i < ref.getGrid().size(); ++i) {
            const bool in_ref = ref.getGrid().isOccupied(i), in_other = other.getGrid().isOccupied(i);
            if (in_ref && in_other) {
                const bool same_px = static_cast<int>(ref_points[i].x_) == static_cast<int>(other_points[i].x_) &&
                                     static_cast<int>(ref_points[i].y_) == static_cast<int>(other_points[i].y_);
                ++(same_px ? same : moved);
                // Two corners of the cell with the same score: either one is a valid pick
                if (!same_px && std::abs(ref_points[i].score_ - other_points[i].score_) < 1e-3) ++moved_tied;
            } else if (in_ref) {
                ++only_ref;
            } else if (in_other) {
                ++only_other;
            }
        }
    }

    void print(const std::string& label) const {
        const size_t both = same + moved;
        std::cout << "  " << label << ": " << same << " of " << both << " shared cells pick the same pixel ("
                  << std::fixed << std::setprecision(2) << (both ? 100.0 * same / both : 0.0) << "%), "
                  << moved_tied << " of the other " << moved << " pick an equally scored corner; " << only_ref
                  << " cells only in vilib_CPU, " << only_other << " only here" << std::endl;
    }
};

int main(int argc, char** argv) {
    EASY_PROFILER_ENABLE;
    EASY_MAIN_THREAD;

    const cuda_bench::BenchmarkArgs args = cuda_bench::parseArgs(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "FAST-9 + grid NMS: CPU vs CUDA (easy_profiler)" << std::endl;
    std::cout << "========================================" << std::endl;

    cuda_bench::initGpu();
    const std::vector<cv::Mat> frames = cuda_bench::loadSequence(args);
    const int width = frames.front().cols, height = frames.front().rows;
    cv::setNumThreads(1);  // the CPU rows are single-threaded
    std::cout << "FAST-" << kArcLength << ", threshold " << kThreshold << ", " << kCellSize << "x" << kCellSize
              << " cells, border " << kBorder << " px" << std::endl;

    // Detection on level 0 only: min_level 0, max_level 1 (exclusive)
    vilib::rosten::FASTCPU<true> rosten_cpu(width, height, kCellSize, kCellSize, 0, 1, kBorder, kBorder, kThreshold,
                                            kArcLength, vilib::SUM_OF_ABS_DIFF_ON_ARC);
    vilib::opencv::FASTCPU<true> opencv_cpu(width, height, kCellSize, kCellSize, 0, 1, kBorder, kBorder, kThreshold);
    vilib::FASTGPU fast_gpu(width, height, kCellSize, kCellSize, 0, 1, kBorder, kBorder, kThreshold, kArcLength,
                            vilib::SUM_OF_ABS_DIFF_ON_ARC);

    // ---- CPU ----
    auto rosten = [&](const cv::Mat& frame) {
        rosten_cpu.reset();
        rosten_cpu.detect({frame});
    };
    auto opencv = [&](const cv::Mat& frame) {
        opencv_cpu.reset();
        opencv_cpu.detect({frame});
    };
    cuda_bench::timeFrames("FAST/vilib_CPU", frames, args.warmup, rosten);
    cuda_bench::timeFrames("FAST/OpenCV_CPU", frames, args.warmup, opencv);

    // ---- CUDA: a one-level pyramid is just the frame in a pitched device buffer ----
    vilib::PyramidPool::init(1, width, height, 1, 1, vilib::Subframe::MemoryType::PITCHED_DEVICE_MEMORY);
    std::vector<std::shared_ptr<vilib::Subframe>> d_image;
    vilib::PyramidPool::get(1, width, height, 1, 1, vilib::Subframe::MemoryType::PITCHED_DEVICE_MEMORY, d_image);

    // Every GPU call ends with a synchronization, so its block covers the GPU work and not just the launch
    auto upload = [&](const cv::Mat& frame) {
        d_image[0]->copy_from(frame, true, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto detect = [&](const cv::Mat&) {
        // detect() already waits for the per-cell results (it copies them to the host)
        fast_gpu.reset();
        fast_gpu.detect(d_image);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    // Detection with the frame already on the GPU (uploaded before each block, untimed), then the upload alone
    cuda_bench::timeFrames("FAST/vilib_GPU", frames, args.warmup, detect, upload);
    cuda_bench::timeFrames("FAST/upload", frames, args.warmup, upload);

    const std::string profile_path = "fast_benchmark.prof";
    profiler::dumpBlocksToFile(profile_path.c_str());

    // ---- Verification over every frame (not profiled) ----
    GridAgreement gpu_vs_cpu, opencv_vs_cpu;
    size_t corners_cpu = 0, corners_gpu = 0;
    for (const cv::Mat& frame : frames) {
        rosten(frame);
        opencv(frame);
        d_image[0]->copy_from(frame, false, 0);
        fast_gpu.reset();
        fast_gpu.detect(d_image);
        corners_cpu += rosten_cpu.count();
        corners_gpu += fast_gpu.count();
        gpu_vs_cpu.add(rosten_cpu, fast_gpu);
        opencv_vs_cpu.add(rosten_cpu, opencv_cpu);
    }
    std::cout << "\nCorners per frame: vilib_CPU " << std::fixed << std::setprecision(1)
              << static_cast<double>(corners_cpu) / frames.size() << ", vilib_GPU "
              << static_cast<double>(corners_gpu) / frames.size() << " (of "
              << rosten_cpu.getGrid().size() << " cells)" << std::endl;
    std::cout << "Verification (reference: vilib_CPU, all " << frames.size() << " frames):" << std::endl;
    gpu_vs_cpu.print("vilib_GPU ");
    opencv_vs_cpu.print("OpenCV_CPU");
    std::cout << "  (cv::FAST scores corners differently, so it may keep another corner of the same cell)"
              << std::endl;

    vilib::PyramidPool::release(d_image);
    vilib::PyramidPool::deinit();

    cuda_bench::printSummary(profile_path, cuda_bench::readProfile(profile_path));
    std::cout << "\nProfile saved to: " << profile_path << std::endl;
    return 0;
}
