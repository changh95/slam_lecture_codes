/**
 * @file prof_summary.cpp
 * @brief Print the comparison tables of one or more easy_profiler captures
 *
 * Useful for .prof files recorded on another machine (e.g. a Jetson) without
 * re-running the benchmarks:
 *   ./prof_summary pyramid_benchmark.prof fast_benchmark.prof tracker_benchmark.prof
 */

#include "profile_summary.hpp"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.prof> [file.prof ...]" << std::endl;
        return 1;
    }
    for (int i = 1; i < argc; ++i) cuda_bench::printSummary(argv[i], cuda_bench::readProfile(argv[i]));
    return 0;
}
