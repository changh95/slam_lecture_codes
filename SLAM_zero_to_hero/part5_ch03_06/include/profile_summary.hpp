/**
 * @file profile_summary.hpp
 * @brief Read an easy_profiler capture back and print per-block timing statistics
 *
 * The benchmarks time every call with EASY_BLOCK and dump the capture to a
 * .prof file. The table printed here is computed from that same file, so the
 * numbers are easy_profiler's own measurements, the same ones profiler_gui shows.
 */

#pragma once

#include <easy/profiler.h>
#include <easy/reader.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace simd_bench {

struct BlockTimes {
    std::string name;         // e.g. "RGB2Y/OpenCV_1T"
    std::vector<double> us;   // one duration per call, in microseconds
};

/**
 * @brief Collect the duration of every block in a .prof file, grouped by block name
 *
 * Blocks are returned in the order they first appear on the timeline.
 */
inline std::vector<BlockTimes> readProfile(const std::string& path) {
    profiler::SerializedData serialized_blocks, serialized_descriptors;
    profiler::descriptors_list_t descriptors;
    profiler::blocks_t blocks;
    profiler::thread_blocks_tree_t threads;
    profiler::bookmarks_t bookmarks;
    profiler::BeginEndTime begin_end_time;
    uint32_t descriptors_count = 0;
    uint32_t version = 0;
    profiler::processid_t pid = 0;
    std::stringstream log;

    const auto num_blocks = fillTreesFromFile(path.c_str(), begin_end_time, serialized_blocks,
                                              serialized_descriptors, descriptors, blocks, threads, bookmarks,
                                              descriptors_count, version, pid, false, log);
    if (num_blocks == 0) {
        std::cerr << "Error: could not read " << path << ": " << log.str() << std::endl;
        return {};
    }

    // (begin time, block index) of every block on every thread, nested blocks included
    std::vector<std::pair<profiler::timestamp_t, profiler::block_index_t>> order;
    std::vector<profiler::block_index_t> stack;
    for (const auto& thread : threads) {
        stack.assign(thread.second.children.begin(), thread.second.children.end());
        while (!stack.empty()) {
            const auto index = stack.back();
            stack.pop_back();
            order.emplace_back(blocks[index].node->begin(), index);
            stack.insert(stack.end(), blocks[index].children.begin(), blocks[index].children.end());
        }
    }
    std::sort(order.begin(), order.end());

    std::vector<BlockTimes> result;
    std::map<std::string, size_t> slot;
    for (const auto& entry : order) {
        const auto* node = blocks[entry.second].node;

        // EASY_BLOCK("literal") stores its name in the descriptor; runtime names live in the block
        std::string name = node->name();
        if (name.empty()) name = descriptors[node->id()]->name();

        auto it = slot.find(name);
        if (it == slot.end()) {
            it = slot.emplace(name, result.size()).first;
            result.push_back({name, {}});
        }
        // The reader has already converted CPU ticks to nanoseconds
        result[it->second].us.push_back(static_cast<double>(node->duration()) * 1e-3);
    }
    return result;
}

inline double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    if (values.size() % 2 == 1) return values[mid];
    const double upper = values[mid];
    return 0.5 * (upper + *std::max_element(values.begin(), values.begin() + mid));
}

/**
 * @brief Print median/mean/min per block and the speed-up over the plain C/C++ baseline
 *
 * Block names follow "<kernel>/<variant>_<threads>", e.g. "FAST/KFAST_MT".
 * Every block is compared with "<kernel>/plain_O3_1T", the plain code built
 * with -O3. Speed-up = baseline median / block median.
 */
inline void printSummary(const std::string& path, const std::vector<BlockTimes>& blocks) {
    auto kernel_of = [](const std::string& name) { return name.substr(0, name.find('/')); };

    std::map<std::string, double> medians;
    for (const auto& block : blocks) medians[block.name] = median(block.us);

    std::cout << "\n=== easy_profiler summary: " << path << " ===" << std::endl;
    std::cout << std::left << std::setw(30) << "Block" << std::right << std::setw(7) << "calls" << std::setw(14)
              << "median [us]" << std::setw(12) << "mean [us]" << std::setw(12) << "min [us]" << std::setw(16)
              << "vs plain_O3" << std::endl;
    std::cout << std::string(91, '-') << std::endl;

    std::string last_kernel;
    for (const auto& block : blocks) {
        if (block.name.find('/') == std::string::npos) continue;  // not a benchmark block
        if (!last_kernel.empty() && kernel_of(block.name) != last_kernel) std::cout << std::endl;
        last_kernel = kernel_of(block.name);

        const double mean = std::accumulate(block.us.begin(), block.us.end(), 0.0) / block.us.size();
        const double min = *std::min_element(block.us.begin(), block.us.end());
        const auto reference = medians.find(last_kernel + "/plain_O3_1T");

        std::ostringstream speedup;
        if (reference != medians.end()) {
            speedup << std::fixed << std::setprecision(2) << reference->second / medians[block.name] << "x";
        }

        std::cout << std::left << std::setw(30) << block.name << std::right << std::setw(7) << block.us.size()
                  << std::fixed << std::setprecision(1) << std::setw(14) << medians[block.name] << std::setw(12)
                  << mean << std::setw(12) << min << std::setw(16) << speedup.str() << std::endl;
    }
}

}  // namespace simd_bench
