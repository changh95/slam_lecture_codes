/**
 * @file pyramid_benchmark.cpp
 * @brief Image pyramid (5 levels of 2x2 averaging): vilib CPU (SSE2) and OpenCV vs vilib CUDA
 *
 * This example demonstrates:
 * 1. vilib's CPU pyramid (SSE2 2x2 averaging) and the same pyramid built with
 *    cv::resize(INTER_AREA), both single-threaded
 * 2. vilib's CUDA pyramid: the kernels that halve level 0 four times, with the
 *    frame already on the GPU (it is uploaded before each call, untimed)
 * 3. The upload on its own, which a desktop GPU needs and a Jetson, whose CPU and
 *    GPU share one DRAM, can avoid. Three ways:
 *    - upload: vilib's own, one cudaMemcpy2DAsync from ordinary (pageable) host memory
 *      straight into the pitched device buffer
 *    - upload_pinned: the same copy from pinned host memory
 *    - upload_pinned_1D: one contiguous cudaMemcpy from pinned memory into a
 *      plain device buffer, then a device-to-device cudaMemcpy2D into the pitched
 *      buffer. The same bytes, but EuRoC's rows are 752 bytes, and on the machine
 *      in the README a 2D copy from the host with rows that are not a multiple of
 *      128 bytes was several times slower than a contiguous one
 *
 * Every level halves the previous one by averaging 2x2 blocks:
 * 752x480 -> 376x240 -> 188x120 -> 94x60 -> 47x30. This is the pyramid that
 * vilib's FAST detector and KLT tracker run on (tracker_benchmark). vilib's SSE2 path
 * needs a width that is a multiple of 16, so only 752 -> 376 uses it; the other
 * levels run its scalar loop.
 *
 * How the CUDA pyramid works (vilib's pyramid_gpu.cu):
 * - The frame is copied into a pitched device buffer (cudaMallocPitch): every row
 *   starts on an aligned address, so the loads of a warp along one row coalesce
 * - One thread writes 4 consecutive output pixels (2 or 1 when the output width is
 *   not a multiple of 4): it reads 4 uchar2 pairs from the top and the bottom input
 *   row and averages each 2x2 block as (a + b + c + d) >> 2, which rounds down
 * - Each level is one kernel launch; all launches go into one stream, so they
 *   run back to back without the CPU waiting in between
 */

#include "benchmark_common.hpp"
#include "profile_summary.hpp"

#include <vilib/preprocess/pyramid.h>
#include <vilib/storage/pyramid_pool.h>
#include <vilib/storage/subframe.h>

#include <opencv2/imgproc.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

constexpr int kLevels = 5;  // 752x480 down to 47x30

/**
 * @brief Largest |difference| and number of differing pixels on levels 1..4, summed over all frames
 */
struct PyramidDiff {
    double max_diff[kLevels] = {};
    size_t differing[kLevels] = {};
    size_t total[kLevels] = {};

    void add(const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b) {
        for (int l = 1; l < kLevels; ++l) {
            cv::Mat diff;
            cv::absdiff(a[l], b[l], diff);
            double level_max = 0.0;
            cv::minMaxLoc(diff, nullptr, &level_max);
            max_diff[l] = std::max(max_diff[l], level_max);
            differing[l] += cv::countNonZero(diff);
            total[l] += diff.total();
        }
    }

    void print(const std::string& label) const {
        std::cout << "  " << label << ":";
        for (int l = 1; l < kLevels; ++l)
            std::cout << "  L" << l << " max " << static_cast<int>(max_diff[l]) << " (" << std::fixed
                      << std::setprecision(1) << 100.0 * differing[l] / total[l] << "% px)";
        std::cout << std::endl;
    }
};

int main(int argc, char** argv) {
    EASY_PROFILER_ENABLE;
    EASY_MAIN_THREAD;

    const cuda_bench::BenchmarkArgs args = cuda_bench::parseArgs(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "Image pyramid: CPU vs CUDA (easy_profiler)" << std::endl;
    std::cout << "========================================" << std::endl;

    cuda_bench::initGpu();
    const std::vector<cv::Mat> frames = cuda_bench::loadSequence(args);
    const cuda_bench::PinnedFrames pinned(frames);
    const int width = frames.front().cols, height = frames.front().rows;
    cv::setNumThreads(1);  // the CPU rows are single-threaded

    // ---- CPU: vilib (SSE2) and OpenCV ----
    std::vector<cv::Mat> pyr_vilib;
    auto vilib_cpu = [&](const cv::Mat& frame) { vilib::pyramid_create_cpu(frame, pyr_vilib, kLevels, false); };

    std::vector<cv::Mat> pyr_cv(kLevels);
    auto opencv_cpu = [&](const cv::Mat& frame) {
        pyr_cv[0] = frame;
        for (int l = 1; l < kLevels; ++l)
            cv::resize(pyr_cv[l - 1], pyr_cv[l], cv::Size(pyr_cv[l - 1].cols / 2, pyr_cv[l - 1].rows / 2), 0, 0,
                       cv::INTER_AREA);
    };

    cuda_bench::timeFrames("Pyramid/vilib_CPU", frames, args.warmup, vilib_cpu);
    cuda_bench::timeFrames("Pyramid/OpenCV_CPU", frames, args.warmup, opencv_cpu);

    // ---- CUDA: upload level 0 into a preallocated pitched buffer, then build levels 1..4 on the GPU ----
    vilib::PyramidPool::init(1, width, height, 1, kLevels, vilib::Subframe::MemoryType::PITCHED_DEVICE_MEMORY);
    std::vector<std::shared_ptr<vilib::Subframe>> d_pyramid;
    vilib::PyramidPool::get(1, width, height, 1, kLevels, vilib::Subframe::MemoryType::PITCHED_DEVICE_MEMORY,
                            d_pyramid);

    // Every GPU call ends with a synchronization, so its block covers the GPU work and not just the launch
    auto gpu_pyramid = [&](const cv::Mat&) {
        vilib::pyramid_create_gpu(d_pyramid, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto vilib_upload = [&](const cv::Mat& frame) {
        d_pyramid[0]->copy_from(frame, true, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    // One contiguous host-to-device copy, then the 2D copy into the pitched buffer on the device
    unsigned char* d_staging = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_staging), frames.front().total()));
    auto flat_upload = [&](const cv::Mat& frame) {
        CUDA_CHECK(cudaMemcpyAsync(d_staging, frame.data, frame.total(), cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpy2DAsync(d_pyramid[0]->data_, d_pyramid[0]->pitch_, d_staging, width, width, height,
                                     cudaMemcpyDeviceToDevice, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // The kernels alone: each frame is uploaded before its block, untimed
    cuda_bench::timeFrames("Pyramid/vilib_GPU", frames, args.warmup, gpu_pyramid, vilib_upload);
    // The upload alone, three ways (a desktop GPU needs it, a Jetson can avoid it)
    cuda_bench::timeFrames("Pyramid/upload", frames, args.warmup, vilib_upload);
    cuda_bench::timeFrames("Pyramid/upload_pinned", pinned.frames(), args.warmup, vilib_upload);
    cuda_bench::timeFrames("Pyramid/upload_pinned_1D", pinned.frames(), args.warmup, flat_upload);

    const std::string profile_path = "pyramid_benchmark.prof";
    profiler::dumpBlocksToFile(profile_path.c_str());

    // ---- Verification over every frame (not profiled) ----
    std::vector<cv::Mat> pyr_gpu(kLevels);
    for (int l = 0; l < kLevels; ++l)
        pyr_gpu[l].create(static_cast<int>(d_pyramid[l]->height_), static_cast<int>(d_pyramid[l]->width_), CV_8UC1);
    PyramidDiff opencv_vs_vilib, gpu_vs_vilib;
    for (const cv::Mat& frame : frames) {
        vilib_cpu(frame);
        opencv_cpu(frame);
        d_pyramid[0]->copy_from(frame, false, 0);
        vilib::pyramid_create_gpu(d_pyramid, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int l = 1; l < kLevels; ++l) d_pyramid[l]->copy_to(pyr_gpu[l], false, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        opencv_vs_vilib.add(pyr_cv, pyr_vilib);
        gpu_vs_vilib.add(pyr_gpu, pyr_vilib);
    }
    std::cout << "\nVerification (reference: vilib_CPU, all " << frames.size() << " frames):" << std::endl;
    gpu_vs_vilib.print("vilib_GPU ");
    opencv_vs_vilib.print("OpenCV_CPU");

    CUDA_CHECK(cudaFree(d_staging));
    vilib::PyramidPool::release(d_pyramid);
    vilib::PyramidPool::deinit();

    cuda_bench::printSummary(profile_path, cuda_bench::readProfile(profile_path));
    std::cout << "\nProfile saved to: " << profile_path << std::endl;
    return 0;
}
