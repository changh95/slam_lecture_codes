/**
 * @file prof_summary.cpp
 * @brief Print the comparison tables of one or more easy_profiler captures
 *
 * Useful for .prof files recorded on another machine (e.g. a Jetson or a
 * laptop) without re-running the benchmarks:
 *   ./prof_summary rgb2y_benchmark.prof kfast_benchmark.prof
 */

#include "profile_summary.hpp"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.prof> [file.prof ...]" << std::endl;
        return 1;
    }
    for (int i = 1; i < argc; ++i) simd_bench::printSummary(argv[i], simd_bench::readProfile(argv[i]));
    return 0;
}
