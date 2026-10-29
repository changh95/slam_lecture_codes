/**
 * @file profile_summary.hpp
 * @brief Read an easy_profiler capture back and print the comparison tables
 *
 * The benchmarks time every call with EASY_BLOCK and dump the capture to a
 * .prof file. The tables printed here are computed from that same file, so the
 * numbers are easy_profiler's own measurements, the same ones profiler_gui shows.
 */

#pragma once

#include <easy/profiler.h>
#include <easy/reader.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
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
 * @brief Print the two comparison tables: single-threaded and multi-threaded
 *
 * Block names follow "<kernel>/<variant>_<threads>", e.g. "RGB2Y/komrad36_MT8":
 * variant is plain_O3, plain_native, OpenCV, or the SIMD variant (komrad36 in an
 * x86-64 capture, NEON in an AArch64 one); threads is 1T or MT<n>.
 * Every cell is the median over all calls, in microseconds, with the speed-up
 * over the kernel's single-threaded -O3 block in parentheses.
 */
inline void printSummary(const std::string& path, const std::vector<BlockTimes>& blocks) {
    std::map<std::string, double> medians;
    std::vector<std::string> kernels;               // in the order they were run
    std::map<std::string, std::vector<int>> mt_counts;  // kernel -> thread counts of its _MT blocks
    size_t calls = 0;
    std::string simd = "komrad36";  // the SIMD column: komrad36's kernels, or the NEON ports
    for (const auto& block : blocks) {
        if (block.name.find("/NEON_") != std::string::npos) simd = "NEON";
        const size_t slash = block.name.find('/');
        const size_t underscore = block.name.rfind('_');
        if (slash == std::string::npos || underscore == std::string::npos || underscore < slash) continue;
        const std::string kernel = block.name.substr(0, slash);
        const std::string threads = block.name.substr(underscore + 1);

        medians[block.name] = median(block.us);
        calls = std::max(calls, block.us.size());
        if (std::find(kernels.begin(), kernels.end(), kernel) == kernels.end()) kernels.push_back(kernel);
        if (threads.rfind("MT", 0) == 0) {
            char* end = nullptr;
            const long n = std::strtol(threads.c_str() + 2, &end, 10);
            if (n <= 0 || *end != '\0') continue;  // not "MT<n>", e.g. a capture from an older version
            auto& counts = mt_counts[kernel];
            if (std::find(counts.begin(), counts.end(), n) == counts.end()) counts.push_back(static_cast<int>(n));
        }
    }

    auto cell = [&](const std::string& kernel, const std::string& name) {
        const auto it = medians.find(name);
        if (it == medians.end()) return std::string("-");
        std::ostringstream text;
        text << std::fixed << std::setprecision(1) << it->second;
        const auto baseline = medians.find(kernel + "/plain_O3_1T");
        if (baseline != medians.end()) text << " (" << std::setprecision(2) << baseline->second / it->second << "x)";
        return text.str();
    };

    std::cout << "\n=== easy_profiler summary: " << path << " ===" << std::endl;
    std::cout << "median of " << calls << " calls in us; (x) = speed-up over single-threaded -O3" << std::endl;

    std::cout << "\n[1] Single thread" << std::endl;
    std::cout << std::left << std::setw(8) << "Kernel" << std::right << std::setw(20) << "-O3" << std::setw(20)
              << "-O3 -march=native" << std::setw(20) << "OpenCV" << std::setw(20) << simd << std::endl;
    std::cout << std::string(88, '-') << std::endl;
    for (const auto& kernel : kernels) {
        std::cout << std::left << std::setw(8) << kernel << std::right;
        for (const std::string variant : {"plain_O3", "plain_native", "OpenCV", simd.c_str()})
            std::cout << std::setw(20) << cell(kernel, kernel + "/" + variant + "_1T");
        std::cout << std::endl;
    }

    std::cout << "\n[2] Multi thread" << std::endl;
    std::cout << std::left << std::setw(8) << "Kernel" << std::right << std::setw(8) << "Threads" << std::setw(20)
              << "OpenCV" << std::setw(20) << simd << std::endl;
    std::cout << std::string(56, '-') << std::endl;
    for (const auto& kernel : kernels) {
        auto counts = mt_counts[kernel];
        std::sort(counts.begin(), counts.end());
        for (const int n : counts) {
            const std::string suffix = "_MT" + std::to_string(n);
            std::cout << std::left << std::setw(8) << kernel << std::right << std::setw(8) << n << std::setw(20)
                      << cell(kernel, kernel + "/OpenCV" + suffix) << std::setw(20)
                      << cell(kernel, kernel + "/" + simd + suffix) << std::endl;
        }
    }
}

}  // namespace simd_bench
