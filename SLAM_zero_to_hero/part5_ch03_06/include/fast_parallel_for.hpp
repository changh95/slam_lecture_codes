/**
 * @file fast_parallel_for.hpp
 * @brief A minimal parallel-for on komrad36's FastThreadPool (the multi-threaded variants)
 *
 * FastThreadPool is only a lock-free queue of captureless function pointers
 * (AddWork / RemoveWork). It has no worker threads, no way to pass arguments,
 * and no way to wait for completion. This wrapper adds the three:
 * - N-1 persistent workers that spin on RemoveWork(); the calling thread is the N-th
 * - the current job and the next stripe index live in statics, since a task is a plain void(*)()
 * - a counter of unfinished stripes that run() waits on
 *
 * The workers never sleep, so they keep N-1 cores busy for as long as the
 * FastParallelFor object lives, even with no work (see idleCoresBusy()).
 * Only one FastParallelFor may exist at a time, because the job state is static.
 */

#pragma once

#include "FastThreadPool/fast_thread_pool.h"

#include <immintrin.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace simd_bench {

class FastParallelFor {
public:
    explicit FastParallelFor(int threads) {
        for (int i = 0; i < threads - 1; ++i) workers_.emplace_back([this] { workerLoop(); });
    }

    ~FastParallelFor() {
        stop_.store(true, std::memory_order_relaxed);
        for (auto& worker : workers_) worker.join();
    }

    FastParallelFor(const FastParallelFor&) = delete;
    FastParallelFor& operator=(const FastParallelFor&) = delete;

    /**
     * @brief Call body(s) for every stripe s in [0, num_stripes), spread over the workers
     */
    template <typename F>
    void run(int num_stripes, F& body) {
        s_job = &body;
        s_call = [](void* job, int stripe) { (*static_cast<F*>(job))(stripe); };
        s_next_stripe.store(0, std::memory_order_relaxed);
        s_remaining.store(num_stripes, std::memory_order_relaxed);

        // AddWork's release CAS publishes the job state above to whichever thread removes the task
        for (int i = 0; i < num_stripes; ++i) s_pool.AddWork(&runOneStripe);

        // The calling thread works too, then waits for stripes still running on the workers
        while (auto task = s_pool.RemoveWork()) task();
        while (s_remaining.load(std::memory_order_acquire) != 0) _mm_pause();
    }

private:
    static void runOneStripe() {
        s_call(s_job, s_next_stripe.fetch_add(1, std::memory_order_relaxed));
        s_remaining.fetch_sub(1, std::memory_order_release);
    }

    void workerLoop() {
        while (!stop_.load(std::memory_order_relaxed)) {
            if (auto task = s_pool.RemoveWork()) task();
            else _mm_pause();
        }
    }

    static inline FastThreadPool s_pool;
    static inline void* s_job = nullptr;
    static inline void (*s_call)(void*, int) = nullptr;
    static inline std::atomic<int> s_next_stripe{0};
    static inline std::atomic<int> s_remaining{0};

    std::atomic<bool> stop_{false};
    std::vector<std::thread> workers_;
};

/**
 * @brief Thread counts to benchmark: 4, 8, 16, ... up to the hardware thread count, which is always included
 */
inline std::vector<int> threadCounts() {
    const int hardware = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<int> counts;
    for (int t = 4; t < hardware; t *= 2) counts.push_back(t);
    counts.push_back(hardware);
    return counts;
}

/**
 * @brief Cores this process keeps busy while the calling thread sleeps for `duration`
 *
 * With a FastParallelFor alive this is about threads - 1: the workers spin even without work.
 */
inline double idleCoresBusy(std::chrono::milliseconds duration = std::chrono::milliseconds(100)) {
    auto cpu_seconds = [] {
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_utime.tv_sec + usage.ru_stime.tv_sec +
               1e-6 * (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec);
    };
    const double start = cpu_seconds();
    std::this_thread::sleep_for(duration);
    return (cpu_seconds() - start) / std::chrono::duration<double>(duration).count();
}

}  // namespace simd_bench
