/**
 * @file profile_summary.hpp
 * @brief Read an easy_profiler capture back and print the comparison table
 *
 * The benchmarks time every call with EASY_BLOCK and dump the capture to a
 * .prof file. The table printed here is computed from that same file, so the
 * numbers are easy_profiler's own measurements.
 */

#pragma once

#include <easy/profiler.h>
#include <easy/reader.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace cuda_bench {

struct BlockTimes {
    std::string name;         // e.g. "FAST/vilib_GPU" or the stage "Tracker/vilib_GPU:track"
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

inline double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double v : values) sum += v;
    return values.empty() ? 0.0 : sum / values.size();
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
 * @brief Print one table per task: every variant's median and mean time, and its speed-up over the first one
 *
 * Block names follow "<task>/<variant>", e.g. "FAST/vilib_GPU". A block named
 * "<task>/<variant>:<stage>" is a part of that variant; it is listed under its
 * variant, without a speed-up. The first variant of every task is a CPU one, so
 * the speed-up reads "x times faster than the CPU". The speed-up compares medians;
 * the mean also counts the rare expensive calls (e.g. the frames in which the
 * tracker detects new features).
 *
 * Variants named "upload..." time the copy of the frame to the GPU. The GPU
 * variants do not include it, so these rows are listed apart, without a speed-up.
 */
inline void printSummary(const std::string& path, const std::vector<BlockTimes>& blocks) {
    std::vector<std::string> tasks;                                     // in the order they were run
    std::map<std::string, std::vector<const BlockTimes*>> rows;         // task -> its blocks, in run order
    for (const auto& block : blocks) {
        const size_t slash = block.name.find('/');
        if (slash == std::string::npos) continue;
        const std::string task = block.name.substr(0, slash);
        if (std::find(tasks.begin(), tasks.end(), task) == tasks.end()) tasks.push_back(task);
        rows[task].push_back(&block);
    }

    auto print_row = [](const std::string& shown, const BlockTimes& block, const std::string& speedup) {
        std::cout << std::left << std::setw(30) << shown << std::right << std::setw(12) << std::fixed
                  << std::setprecision(1) << median(block.us) << std::setw(12) << mean(block.us) << std::setw(12)
                  << speedup << std::setw(8) << block.us.size() << std::endl;
    };

    std::cout << "\n=== easy_profiler summary: " << path << " ===" << std::endl;
    for (const auto& task : tasks) {
        const auto& list = rows[task];
        std::cout << "\n" << std::left << std::setw(30) << task << std::right << std::setw(12) << "median us"
                  << std::setw(12) << "mean us" << std::setw(12) << "vs CPU" << std::setw(8) << "calls" << std::endl;
        std::cout << std::string(74, '-') << std::endl;

        double baseline = 0.0;
        std::string baseline_name;
        std::vector<const BlockTimes*> uploads;
        for (const auto* block : list) {
            const std::string label = block->name.substr(task.size() + 1);
            if (label.find(':') != std::string::npos) continue;  // a stage, printed under its variant
            if (label.rfind("upload", 0) == 0) {
                uploads.push_back(block);
                continue;
            }
            const double value = median(block->us);
            if (baseline_name.empty()) {
                baseline = value;
                baseline_name = label;
            }
            std::ostringstream speedup;
            speedup << std::fixed << std::setprecision(2) << baseline / value << "x";
            print_row("  " + label, *block, speedup.str());
            // Stages print right after their variant, whatever order they were recorded in
            for (const auto* stage : list)
                if (stage->name.rfind(block->name + ":", 0) == 0)
                    print_row("    - " + stage->name.substr(block->name.size() + 1), *stage, "");
        }
        if (!uploads.empty()) {
            std::cout << "  frame upload, not in the GPU rows:" << std::endl;
            for (const auto* block : uploads) print_row("    " + block->name.substr(task.size() + 1), *block, "");
        }
        if (!baseline_name.empty())
            std::cout << "(vs CPU = speed-up of the median over " << baseline_name
                      << "; \"- \" rows are stages of the row above)" << std::endl;
    }
}

}  // namespace cuda_bench
