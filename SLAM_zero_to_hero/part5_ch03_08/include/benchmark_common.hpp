/**
 * @file benchmark_common.hpp
 * @brief Helpers shared by the three CUDA benchmarks (arguments, EuRoC frames, GPU checks, timing loop)
 */

#pragma once

#include <cuda_runtime_api.h>
#include <easy/profiler.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                             \
    do {                                                                                             \
        const cudaError_t status = (call);                                                           \
        if (status != cudaSuccess) {                                                                 \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":"  \
                      << __LINE__ << std::endl;                                                      \
            std::exit(EXIT_FAILURE);                                                                 \
        }                                                                                            \
    } while (0)

namespace cuda_bench {

struct BenchmarkArgs {
    std::string sequence_dir = "/data/MH_01_easy/mav0/cam0/data";  // EuRoC MH_01 left camera, 752x480
    int frames = 500;   // frames loaded into memory; every variant runs once per frame
    int start = 1000;   // first frame: steady flight from here; around frames 400-800 the drone stands still
    int warmup = 20;    // unprofiled calls per variant (CUDA context, caches, allocations)
};

inline BenchmarkArgs parseArgs(int argc, char** argv) {
    BenchmarkArgs args;
    if (argc > 1) args.sequence_dir = argv[1];
    if (argc > 2) args.frames = std::max(2, std::atoi(argv[2]));
    if (argc > 3) args.start = std::max(0, std::atoi(argv[3]));
    return args;
}

/**
 * @brief Load frames [start, start + frames) of an EuRoC camera folder as 8-bit grayscale
 *
 * EuRoC names every image after its timestamp in nanoseconds, so sorting the
 * file names sorts the frames in time. Everything is loaded before timing
 * starts, so no benchmark reads from disk. Any folder of equally sized PNG
 * frames works the same way.
 */
inline std::vector<cv::Mat> loadSequence(const BenchmarkArgs& args) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(args.sequence_dir, error))
        if (entry.path().extension() == ".png") files.push_back(entry.path().string());
    std::sort(files.begin(), files.end());

    if (files.size() < static_cast<size_t>(args.start) + 2) {
        std::cerr << "Error: found " << files.size() << " PNG frames in " << args.sequence_dir
                  << ", need more than " << args.start + 1 << std::endl;
        std::cerr << "Mount EuRoC MH_01_easy (ASL format) with -v <path>/MH_01_easy:/data/MH_01_easy:ro" << std::endl;
        std::cerr << "Usage: <benchmark> [cam0/data folder] [frames] [first frame]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const size_t first = args.start;
    const size_t last = std::min(files.size(), first + args.frames);
    std::vector<cv::Mat> frames;
    frames.reserve(last - first);
    for (size_t i = first; i < last; ++i) {
        cv::Mat image = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Error: could not load " << files[i] << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (!frames.empty() && image.size() != frames.front().size()) {
            std::cerr << "Error: " << files[i] << " is " << image.cols << "x" << image.rows << ", the first frame "
                      << frames.front().cols << "x" << frames.front().rows << std::endl;
            std::exit(EXIT_FAILURE);
        }
        frames.push_back(image);
    }
    std::cout << "Frames: " << frames.size() << " (" << frames.front().cols << "x" << frames.front().rows
              << ", #" << first << "-#" << last - 1 << ") from " << args.sequence_dir << std::endl;
    return frames;
}

/**
 * @brief Exit with a hint when no CUDA device is visible, otherwise print it and create the context
 *
 * The first CUDA call creates the context, which takes tens to hundreds of
 * milliseconds. cudaFree(0) does it here, before anything is timed.
 */
inline void initGpu() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        std::cerr << "Error: no CUDA device (" << cudaGetErrorString(status) << ")." << std::endl;
        std::cerr << "Run the container with --gpus all (see the README)." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int driver = 0, runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ", " << prop.multiProcessorCount
              << " SMs), CUDA runtime " << runtime / 1000 << "." << (runtime % 1000) / 10 << ", driver "
              << driver / 1000 << "." << (driver % 1000) / 10 << std::endl;
    CUDA_CHECK(cudaFree(nullptr));
}

/**
 * @brief Copies of the frames in page-locked (pinned) host memory
 *
 * cudaMemcpy from ordinary (pageable) memory goes through a staging buffer
 * inside the driver. From pinned memory the GPU's copy engine reads the frame
 * directly over PCIe.
 */
class PinnedFrames {
public:
    explicit PinnedFrames(const std::vector<cv::Mat>& frames) {
        const size_t bytes = frames.front().total();
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&buffer_), bytes * frames.size()));
        for (size_t i = 0; i < frames.size(); ++i) {
            mats_.emplace_back(frames[i].rows, frames[i].cols, CV_8UC1, buffer_ + i * bytes);
            frames[i].copyTo(mats_.back());
        }
    }
    ~PinnedFrames() { cudaFreeHost(buffer_); }
    PinnedFrames(const PinnedFrames&) = delete;
    PinnedFrames& operator=(const PinnedFrames&) = delete;

    const std::vector<cv::Mat>& frames() const { return mats_; }

private:
    unsigned char* buffer_ = nullptr;
    std::vector<cv::Mat> mats_;
};

using FrameFn = std::function<void(const cv::Mat&)>;

/**
 * @brief One timed part of a variant, e.g. the pyramid kernels; recorded as block "<variant>:<name>"
 */
struct Stage {
    std::string name;
    FrameFn run;
};

/**
 * @brief Time a variant once per frame, after a few unprofiled warm-up calls
 *
 * Every frame gets a block named after the variant, and inside it one block per
 * stage. GPU stages must end with a synchronization, otherwise their block
 * closes when the work is queued, not when it is done. prepare, if given, runs
 * before each frame's block and is not timed (e.g. uploading the frame to the GPU).
 */
inline void timeStages(const std::string& name, const std::vector<cv::Mat>& frames, int warmup,
                       const std::vector<Stage>& stages, const FrameFn& prepare = nullptr) {
    for (int i = 0; i < warmup; ++i) {
        const cv::Mat& frame = frames[i % frames.size()];
        if (prepare) prepare(frame);
        for (const Stage& stage : stages) stage.run(frame);
    }

    std::vector<std::string> stage_names;
    for (const Stage& stage : stages) stage_names.push_back(name + ":" + stage.name);
    for (const cv::Mat& frame : frames) {
        if (prepare) prepare(frame);
        EASY_BLOCK(name.c_str(), profiler::colors::Orange500);
        for (size_t s = 0; s < stages.size(); ++s) {
            EASY_BLOCK(stage_names[s].c_str(), profiler::colors::Green500);
            stages[s].run(frame);
        }
    }
}

/**
 * @brief Time a variant that has no separate stages
 */
inline void timeFrames(const std::string& name, const std::vector<cv::Mat>& frames, int warmup, const FrameFn& fn,
                       const FrameFn& prepare = nullptr) {
    for (int i = 0; i < warmup; ++i) {
        const cv::Mat& frame = frames[i % frames.size()];
        if (prepare) prepare(frame);
        fn(frame);
    }
    for (const cv::Mat& frame : frames) {
        if (prepare) prepare(frame);
        EASY_BLOCK(name.c_str(), profiler::colors::Orange500);
        fn(frame);
    }
}

}  // namespace cuda_bench
